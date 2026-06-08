import os
import re
import shutil
import asyncio
import threading
import subprocess
import math
import gc
from pathlib import Path
from typing import List, Optional
import ffmpeg
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Project, Book, Chapter


# Thread-safe global trackers for active/cancelled jobs
active_projects = set()
cancelled_projects = set()


def module_print(*args, **kwargs):
    """
    Overrides built-in print for this module to automatically mirror standard
    terminal console prints straight into the NiceGUI state log queue.
    """
    import builtins
    builtins.print(*args, **kwargs)
    
    sep = kwargs.get('sep', ' ')
    msg = sep.join(str(arg) for arg in args)
    msg_clean = msg.strip()
    if msg_clean:
        try:
            from ui.state import add_console_log
            add_console_log(msg_clean)
        except Exception:
            pass

# Override standard print in the module's global namespace
print = module_print


def chunk_audio_with_ffmpeg(audio_path: Path, output_dir: Path) -> tuple[List[Path], List[tuple[float, float]]]:
    """
    Splits a 16kHz mono WAV file into ~60-second chunks using FFmpeg silence detection.
    Extremely fast, low-overhead, and completely self-contained. Returns chunk paths and (start, end) timings.
    """
    SILENCE_THRESHOLD_DB = "-30dB"
    SILENCE_DURATION_S = "0.5"
    TARGET_CHUNK_S = 60

    chunk_paths = []
    chunk_timings = []
    try:
        command = [
            'ffmpeg', '-i', str(audio_path),
            '-af', f'silencedetect=n={SILENCE_THRESHOLD_DB}:d={SILENCE_DURATION_S}',
            '-f', 'null', '-'
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        stderr_output = result.stderr

        silence_ends = re.findall(r"silence_end: (\d+\.?\d*)", stderr_output)
        cut_timestamps = [float(t) for t in silence_ends]

        probe = ffmpeg.probe(str(audio_path))
        duration = float(probe['format']['duration'])

        last_cut = 0.0
        final_cuts = [0.0]
        for t in cut_timestamps:
            if t - last_cut > TARGET_CHUNK_S:
                final_cuts.append(t)
                last_cut = t
        final_cuts.append(duration)

        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(final_cuts) - 1):
            start = final_cuts[i]
            end = final_cuts[i+1]
            if end - start < 0.5:
                continue

            chunk_file = output_dir / f"chunk_{len(chunk_paths) + 1}.wav"
            (
                ffmpeg.input(str(audio_path), ss=start, to=end)
                .output(str(chunk_file), acodec='pcm_s16le', ac=1, ar='16000', loglevel="panic")
                .run(overwrite_output=True)
            )
            chunk_paths.append(chunk_file)
            chunk_timings.append((start, end))

        return chunk_paths, chunk_timings
    except Exception as e:
        print(f"ERROR: FFmpeg chunking failed for {audio_path.name}: {e}")
        return [], []


def get_onnx_model():
    """
    Dynamically loads the onnx-asr model locally or via huggingface.
    Optimized for high-performance FP16 execution on Nvidia GPUs with memory limits.
    """
    import onnx_asr
    import onnxruntime as ort
    
    model_dir = os.path.abspath(".models/parakeet")
    device_setting = get_setting("stt_device", "GPU/CUDA")
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 8  # Use multiple CPU cores for audio feature extraction
    sess_options.inter_op_num_threads = 8
    sess_options.enable_mem_pattern = False
    sess_options.enable_cpu_mem_arena = False
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    def patch_onnx_asr_model(model):
        """
        Monkeypatches the onnx-asr TDT decoder to prevent AssertionErrors on FP16 models
        due to off-by-one or mismatched downsampling length calculations.
        """
        if hasattr(model, 'asr'):
            asr_class = model.asr.__class__
            if hasattr(asr_class, '_decoding'):
                if not getattr(asr_class, '_patched_for_fp16', False):
                    original_decoding = asr_class._decoding
                    
                    def patched_decoding(self, encodings, encodings_len, **kwargs):
                        import numpy as np
                        limit = encodings.shape[1]
                        
                        if isinstance(encodings_len, np.ndarray):
                            encodings_len = np.minimum(encodings_len, limit)
                        elif isinstance(encodings_len, (list, tuple)):
                            encodings_len = type(encodings_len)([min(x, limit) for x in encodings_len])
                        else:
                            val = int(encodings_len.item()) if hasattr(encodings_len, "item") else int(encodings_len)
                            encodings_len = min(val, limit)
                            
                        return original_decoding(self, encodings, encodings_len, **kwargs)
                    
                    asr_class._decoding = patched_decoding
                    asr_class._patched_for_fp16 = True
                    print("[ABI-Pipeline] Patched TDT decoder sequence lengths to prevent FP16 assertions.")
        return model

    if device_setting == "GPU/CUDA":
        gpu_options = {
            "device_id": "0",
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",   # <--- MASSIVE SPEEDUP
            "cudnn_conv_use_max_workspace": "1",      # Allows RTX 3090 to use its massive VRAM for math
            "do_copy_in_default_stream": "1",
        }
        providers = [("CUDAExecutionProvider", gpu_options), "CPUExecutionProvider"]
        
        print(f"Initializing Parakeet ONNX engine in FP16 mode on: {device_setting}")
        try:
            local_fp16_exists = os.path.exists(os.path.join(model_dir, "encoder-model.fp16.onnx"))
            if local_fp16_exists:
                model = onnx_asr.load_model(
                    "nemo-parakeet-tdt-0.6b-v3", 
                    model_dir, 
                    quantization="fp16",
                    providers=providers,
                    sess_options=sess_options
                )
            else:
                model = onnx_asr.load_model(
                    "nemo-parakeet-tdt-0.6b-v3", 
                    quantization="fp16",
                    providers=providers,
                    sess_options=sess_options
                )
            return patch_onnx_asr_model(model)
        except Exception as e:
            print(f"GPU FP16 initialization failed: {e}. Falling back gracefully to CPU Execution...")
            providers = ["CPUExecutionProvider"]

    print(f"Loading Parakeet ONNX model on CPU (Providers: {providers})")
    if os.path.exists(os.path.join(model_dir, "encoder-model.onnx")):
        model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3", 
            model_dir, 
            providers=providers,
            sess_options=sess_options
        )
    else:
        model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3", 
            providers=providers,
            sess_options=sess_options
        )
    return patch_onnx_asr_model(model)


def start_project_transcription(project_id: int) -> None:
    """Begins the sequential multi-book transcription pipeline in a background thread."""
    if project_id in active_projects:
        return
        
    active_projects.add(project_id)
    if project_id in cancelled_projects:
        cancelled_projects.remove(project_id)

    thread = threading.Thread(
        target=transcribe_project_worker, 
        args=(project_id,), 
        daemon=True
    )
    thread.start()


def cancel_project_transcription(project_id: int) -> None:
    """Requests a cancellation/stop of any active transcription for this project."""
    cancelled_projects.add(project_id)


def transcribe_project_worker(project_id: int) -> None:
    """Main background thread worker coordinating sequential book transcriptions."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            active_projects.discard(project_id)
            return

        project.status = "Transcribing"
        session.add(project)
        session.commit()

        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        book_ids = [b.id for b in books]

    try:
        stt_engine = get_setting("stt_engine", "Parakeet ONNX")
        if stt_engine == "Whisper":
            from faster_whisper import WhisperModel
            model_dir = os.path.abspath(".models/whisper")
            device_setting = get_setting("stt_device", "GPU/CUDA")
            device = "cuda" if device_setting == "GPU/CUDA" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            print(f"\n[ABI-Pipeline] Loading Faster-Whisper on {device} ({compute_type})...")
            model = WhisperModel(model_dir, device=device, compute_type=compute_type, local_files_only=True)
        else:
            model = get_onnx_model()
    except Exception as e:
        print(f"CRITICAL: Failed to load STT model weights: {e}")
        with Session(engine) as session:
            project = session.get(Project, project_id)
            if project:
                project.status = "Failed"
                session.add(project)
                session.commit()
        active_projects.discard(project_id)
        return

    for book_id in book_ids:
        if project_id in cancelled_projects:
            break
        transcribe_book(book_id, model, project_id)

    try:
        if hasattr(model, 'asr'):
            if hasattr(model.asr, '_encoder'):
                model.asr._encoder.set_providers([])
            if hasattr(model.asr, '_decoder_joint'):
                model.asr._decoder_joint.set_providers([])
    except Exception as e:
        pass

    del model
    gc.collect()

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project:
            if project_id in cancelled_projects:
                project.status = "Imported"
            else:
                project.status = "Transcribed"
            session.add(project)
            session.commit()

    active_projects.discard(project_id)


def transcribe_book(book_id: int, model, project_id: int) -> None:
    """Processes all chapters of an individual audiobook sequential track-by-track."""
    import json
    with Session(engine) as session:
        book = session.get(Book, book_id)
        if not book or book.status == "Transcribed":
            return

        book_name = book.name
        book_path = book.path
        
        project = session.get(Project, project_id)
        project_name = project.name if project else "Default_Project"
        project_path = project.path if project else ""

        book.status = "Transcribing"
        session.add(book)
        session.commit()

        chapters = session.exec(
            select(Chapter).where(Chapter.book_id == book_id)
        ).all()
        total_chapters = len(chapters)

    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    book_output_dir = base_output_dir / project_name / book_name
    book_output_dir.mkdir(parents=True, exist_ok=True)

    working_dir = Path("./workspace_temp") / f"book_{book_id}"
    working_dir.mkdir(parents=True, exist_ok=True)

    try:
        state_data = {
            "project_name": project_name,
            "project_path": project_path,
            "book_name": book_name,
            "book_path": book_path,
            "audio_type": "multi_file" if any(c.type == 'file' for c in chapters) else "single_file"
        }
        with open(working_dir / "transcription_state.json", "w", encoding="utf-8") as sf:
            json.dump(state_data, sf, indent=4)
    except Exception as se:
        print(f"[Sync-Engine] Failed to write transcription recovery state: {se}")

    for chapter in chapters:
        if project_id in cancelled_projects:
            break

        if chapter.status == "Completed":
            chapter_txt = working_dir / f"chapter_{chapter.chapter_num}.txt"
            if chapter_txt.exists():
                continue

        with Session(engine) as session:
            db_chapter = session.get(Chapter, chapter.id)
            if db_chapter:
                db_chapter.status = "Transcribing"
                session.add(db_chapter)
                session.commit()

        transcript_text = transcribe_chapter(chapter, model, working_dir)

        if transcript_text:
            chapter_txt = working_dir / f"chapter_{chapter.chapter_num}.txt"
            with open(chapter_txt, "w", encoding="utf-8") as f:
                f.write(transcript_text)

            with Session(engine) as session:
                db_chapter = session.get(Chapter, chapter.id)
                if db_chapter:
                    db_chapter.status = "Completed"
                    session.add(db_chapter)
                    session.commit()
        else:
            with Session(engine) as session:
                db_chapter = session.get(Chapter, chapter.id)
                if db_chapter:
                    db_chapter.status = "Pending"
                    session.add(db_chapter)
                    session.commit()

        with Session(engine) as session:
            completed_count = len(session.exec(
                select(Chapter)
                .where(Chapter.book_id == book_id)
                .where(Chapter.status == "Completed")
            ).all())

            db_book = session.get(Book, book_id)
            if db_book:
                db_book.progress = (
                    completed_count / total_chapters if total_chapters > 0 else 1.0
                )
                session.add(db_book)
                session.commit()

        gc.collect()

    if project_id not in cancelled_projects:
        combine_chapters(working_dir, book_output_dir)

        try:
            metadata_file = book_output_dir / "metadata.json"
            meta_data = {
                "project_name": project_name,
                "project_path": project_path,
                "book_name": book_name,
                "book_path": book_path,
                "audio_type": "multi_file" if any(c.type == 'file' for c in chapters) else "single_file"
            }
            with open(metadata_file, "w", encoding="utf-8") as mf:
                json.dump(meta_data, mf, indent=4)
        except Exception as me:
            print(f"[Sync-Engine] Failed to write completed book metadata: {me}")

        with Session(engine) as session:
            db_book = session.get(Book, book_id)
            if db_book:
                db_book.status = "Transcribed"
                session.add(db_book)
                session.commit()
        if working_dir.exists():
            try:
                shutil.rmtree(working_dir)
            except Exception:
                pass
    else:
        with Session(engine) as session:
            db_book = session.get(Book, book_id)
            if db_book:
                db_book.status = "Imported"
                session.add(db_book)
                
            active_ch = session.exec(
                select(Chapter)
                .where(Chapter.book_id == book_id)
                .where(Chapter.status == "Transcribing")
            ).all()
            for ch in active_ch:
                ch.status = "Pending"
                session.add(ch)
                
            session.commit()


def transcribe_chapter(chapter: Chapter, model, working_dir: Path) -> str:
    """Preprocesses a chapter's audio track, slices it, and performs speech-to-text in parallel batches."""
    import traceback
    import time
    import json
    
    preprocessed_wav = working_dir / f"temp_chapter_{chapter.chapter_num}_preprocessed.wav"
    try:
        ffmpeg_input = ffmpeg.input(chapter.input_file)
        if chapter.type == 'segment':
            ffmpeg_input = ffmpeg.input(
                chapter.input_file, 
                ss=chapter.start_time, 
                to=chapter.end_time
            )

        print(f"\n[ABI-Pipeline] Preprocessing '{chapter.title}'...")
        (
            ffmpeg_input
            .output(str(preprocessed_wav), acodec='pcm_s16le', ac=1, ar='16000', loglevel="panic")
            .run(overwrite_output=True)
        )
    except Exception as e:
        print(f"FFmpeg preprocessing failed for chapter {chapter.chapter_num}: {e}")
        return ""

    if type(model).__name__ == "WhisperModel":
        print(f"[ABI-Pipeline] Transcribing '{chapter.title}' with Faster-Whisper (built-in VAD)...")
        start_time = time.time()
        
        segments, info = model.transcribe(
            str(preprocessed_wav), 
            vad_filter=True, 
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        segments_list = list(segments)
        timing_map = []
        current_char_offset = 0
        cleaned_texts = []
        
        for segment in segments_list:
            text_str = segment.text.strip()
            cleaned_texts.append(text_str)
            text_len = len(text_str)
            
            timing_map.append({
                "start": segment.start,
                "end": segment.end,
                "char_start": current_char_offset,
                "char_end": current_char_offset + text_len
            })
            current_char_offset += text_len + 1  # accounts for space separator in join()
            
        chapter_json = working_dir / f"chapter_{chapter.chapter_num}.json"
        with open(chapter_json, "w", encoding="utf-8") as jf:
            json.dump(timing_map, jf, indent=4)
            
        total_time = time.time() - start_time
        print(f"[ABI-Pipeline] Chapter {chapter.chapter_num} complete! Total time: {total_time:.2f}s")
        
        if preprocessed_wav.exists():
            preprocessed_wav.unlink()
            
        return " ".join(cleaned_texts).strip()

    temp_chunk_dir = working_dir / f"chapter_{chapter.chapter_num}_chunks"
    print(f"[ABI-Pipeline] Chunking with ffmpeg silence detection...")
    chunk_paths, chunk_timings = chunk_audio_with_ffmpeg(preprocessed_wav, temp_chunk_dir)

    if not chunk_paths:
        if preprocessed_wav.exists(): preprocessed_wav.unlink()
        return ""

    batch_size = int(get_setting("batch_size", 8))
    print(f"[ABI-Pipeline] Generated {len(chunk_paths)} chunks. Starting inference (Batch Size: {batch_size})...")
    
    chunks_with_metadata = [(i, str(p), os.path.getsize(p)) for i, p in enumerate(chunk_paths)]
    chunks_with_metadata.sort(key=lambda x: x[2], reverse=True)
    
    all_texts = [None] * len(chunk_paths)
    num_batches = math.ceil(len(chunks_with_metadata) / batch_size)
    start_time_total = time.time()
    
    for i in range(num_batches):
        batch_start_time = time.time()
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_meta = chunks_with_metadata[start_idx:end_idx]
        batch_chunks = [m[1] for m in batch_meta]
        original_indices = [m[0] for m in batch_meta]

        try:
            batch_results = model.recognize(batch_chunks)
            if batch_results:
                if isinstance(batch_results, str):
                    batch_results = [batch_results]
                for idx, result in zip(original_indices, batch_results):
                    if result:
                        all_texts[idx] = result.strip()
                        
        except Exception as batch_error:
            print(f"\n[STT Fallback] Batch inference failed on chapter {chapter.chapter_num}, batch {i+1}. Error: {batch_error}")
            for idx, chunk in zip(original_indices, batch_chunks):
                try:
                    result = model.recognize(chunk)
                    if result: all_texts[idx] = result.strip()
                except Exception: pass
            
        batch_time = time.time() - batch_start_time
        chunks_per_sec = len(batch_chunks) / batch_time if batch_time > 0 else 0
        print(f"  -> Batch {i+1}/{num_batches} processed {len(batch_chunks)} chunks in {batch_time:.2f}s ({chunks_per_sec:.2f} chunk/s)")
        gc.collect()

    timing_map = []
    current_char_offset = 0
    cleaned_texts = []
    for idx, text in enumerate(all_texts):
        text_str = text.strip() if text else ""
        cleaned_texts.append(text_str)
        text_len = len(text_str)
        
        start_t, end_t = chunk_timings[idx]
        timing_map.append({
            "start": start_t,
            "end": end_t,
            "char_start": current_char_offset,
            "char_end": current_char_offset + text_len
        })
        current_char_offset += text_len + 1  # accounts for space separator in join()

    chapter_json = working_dir / f"chapter_{chapter.chapter_num}.json"
    with open(chapter_json, "w", encoding="utf-8") as jf:
        json.dump(timing_map, jf, indent=4)

    total_time = time.time() - start_time_total
    avg_speed = len(chunk_paths) / total_time if total_time > 0 else 0
    print(f"[ABI-Pipeline] Chapter {chapter.chapter_num} complete! Total time: {total_time:.2f}s ({avg_speed:.2f} chunk/s avg)\n")

    if preprocessed_wav.exists(): preprocessed_wav.unlink()
    if temp_chunk_dir.exists():
        try: shutil.rmtree(temp_chunk_dir)
        except Exception: pass

    return " ".join(cleaned_texts).strip()


def combine_chapters(working_dir: Path, book_output_dir: Path) -> None:
    """Appends all temporary chapter text files into the final transcript.txt inside the structured book directory."""
    import json
    final_text_path = book_output_dir / "transcript.txt"
    chapter_files = sorted(
        list(working_dir.glob("chapter_*.txt")),
        key=lambda x: int(x.stem.split('_')[1])
    )
    if not chapter_files:
        final_text_path.touch()
        return

    with open(final_text_path, "w", encoding="utf-8") as final_file:
        for ch_file in chapter_files:
            final_file.write("==CHAPTER==\n\n")
            with open(ch_file, "r", encoding="utf-8") as f:
                final_file.write(f.read())
            final_file.write("\n\n")

    # Combine chapter timing metadata JSON files
    timing_files = sorted(
        list(working_dir.glob("chapter_*.json")),
        key=lambda x: int(x.stem.split('_')[1])
    )
    master_timing = {"chapters": {}}
    for t_file in timing_files:
        try:
            ch_num = t_file.stem.split('_')[1]
            with open(t_file, "r", encoding="utf-8") as f:
                master_timing["chapters"][ch_num] = json.load(f)
        except Exception as e:
            print(f"[Timing-Sync] Error merging chapter timing metadata: {e}")

    if master_timing["chapters"]:
        with open(book_output_dir / "transcript_timing.json", "w", encoding="utf-8") as f:
            json.dump(master_timing, f, indent=4)