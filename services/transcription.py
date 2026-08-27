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
    Optimized for high-performance CUDA/GPU execution on Nvidia GPUs.
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
        Monkeypatches the onnx-asr TDT decoder to prevent AssertionErrors
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
                    print("[ABI-Pipeline] Patched TDT decoder sequence lengths to prevent dimension assertions.")
        return model

    if device_setting == "GPU/CUDA":
        gpu_options = {
            "device_id": "0",
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "HEURISTIC",   # Restored fast HEURISTIC search
            "cudnn_conv_use_max_workspace": "1",      # Allows RTX 3090/4090 to use VRAM for math
            "do_copy_in_default_stream": "1",
        }
        providers = [("CUDAExecutionProvider", gpu_options), "CPUExecutionProvider"]
        
        print(f"Initializing Parakeet ONNX v2 engine on: {device_setting}")
        try:
            local_exists = os.path.exists(os.path.join(model_dir, "encoder-model.onnx"))
            if local_exists:
                model = onnx_asr.load_model(
                    "nemo-parakeet-tdt-0.6b-v2", 
                    model_dir, 
                    providers=providers,
                    sess_options=sess_options
                )
            else:
                model = onnx_asr.load_model(
                    "nemo-parakeet-tdt-0.6b-v2", 
                    providers=providers,
                    sess_options=sess_options
                )
            return patch_onnx_asr_model(model)
        except Exception as e:
            print(f"GPU initialization failed: {e}. Falling back gracefully to CPU Execution...")
            providers = ["CPUExecutionProvider"]

    print(f"Loading Parakeet ONNX v2 model on CPU (Providers: {providers})")
    if os.path.exists(os.path.join(model_dir, "encoder-model.onnx")):
        model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2", 
            model_dir, 
            providers=providers,
            sess_options=sess_options
        )
    else:
        model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2", 
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


def transcribe_book(book_id: int, model, project_id: int, force_retranscribe: bool = False) -> None:
    """Processes all chapters of an individual audiobook sequential track-by-track."""
    import json
    import shutil
    
    working_dir = Path("./workspace_temp") / f"book_{book_id}"
    
    if force_retranscribe:
        print(f"[ABI-Pipeline] Force flag active. Clearing transcription cache directory: {working_dir}")
        if working_dir.exists():
            try:
                shutil.rmtree(working_dir)
            except Exception as e:
                print(f"[ABI-Pipeline] Warning: Could not remove directory {working_dir}: {e}")
        
        # Reset all chapter statuses to Pending in database
        with Session(engine) as session:
            chapters_to_reset = session.exec(
                select(Chapter).where(Chapter.book_id == book_id)
            ).all()
            for ch in chapters_to_reset:
                ch.status = "Pending"
                session.add(ch)
            session.commit()
            print(f"[ABI-Pipeline] Reset {len(chapters_to_reset)} chapter statuses to 'Pending'.")

    with Session(engine) as session:
        book = session.get(Book, book_id)
        if not book:
            print(f"[ABI-Pipeline] Error: Book ID {book_id} not found.")
            return
            
        if book.status == "Transcribed" and not force_retranscribe:
            print(f"[ABI-Pipeline] Skipping book '{book.name}' as status is already 'Transcribed'.")
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

    print(f"[ABI-Pipeline] Starting transcription for '{book_name}' with {total_chapters} chapter(s)...")

    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    book_output_dir = base_output_dir / project_name / book_name
    book_output_dir.mkdir(parents=True, exist_ok=True)

    working_dir.mkdir(parents=True, exist_ok=True)

    # Determine original audiobook source directory next to audio files
    source_audio_dir = None
    if book_path:
        p = Path(book_path)
        source_audio_dir = p if p.is_dir() else p.parent

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

    cumulative_offset_ms = 0

    for chapter in chapters:
        if project_id in cancelled_projects:
            print("[ABI-Pipeline] Transcription cancelled by user.")
            break

        chapter_txt = working_dir / f"chapter_{chapter.chapter_num}.txt"
        
        # Determine exact timeline offset in milliseconds for continuous timestamps
        if chapter.type == 'segment':
            start_offset_ms = int(round((chapter.start_time or 0.0) * 1000))
        else:
            start_offset_ms = cumulative_offset_ms

        if chapter.status == "Completed" and not force_retranscribe:
            if chapter_txt.exists():
                print(f"[ABI-Pipeline] Chapter {chapter.chapter_num} ('{chapter.title}') already complete. Skipping transcription.")
                # Advance cumulative offset using file duration
                try:
                    probe = ffmpeg.probe(chapter.input_file)
                    ch_dur_ms = int(round(float(probe['format']['duration']) * 1000))
                    cumulative_offset_ms += ch_dur_ms
                except Exception:
                    pass
                continue
            else:
                print(f"[ABI-Pipeline] Chapter {chapter.chapter_num} status is 'Completed' but text file is missing. Will re-transcribe.")

        print(f"[ABI-Pipeline] Processing Chapter {chapter.chapter_num}: '{chapter.title}' (Status: {chapter.status}, Offset: {start_offset_ms}ms)")

        with Session(engine) as session:
            db_chapter = session.get(Chapter, chapter.id)
            if db_chapter:
                db_chapter.status = "Transcribing"
                session.add(db_chapter)
                session.commit()

        transcript_text, chapter_dur_ms = transcribe_chapter(
            chapter, 
            model, 
            working_dir, 
            start_offset_ms=start_offset_ms
        )

        cumulative_offset_ms = start_offset_ms + chapter_dur_ms

        if transcript_text:
            with open(chapter_txt, "w", encoding="utf-8") as f:
                f.write(transcript_text)

            with Session(engine) as session:
                db_chapter = session.get(Chapter, chapter.id)
                if db_chapter:
                    db_chapter.status = "Completed"
                    session.add(db_chapter)
                    session.commit()
        else:
            print(f"[ABI-Pipeline] Error: Transcription returned empty text for Chapter {chapter.chapter_num}")
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
        print(f"[ABI-Pipeline] Combining chapter texts and merging timing metadata for '{book_name}'...")
        combine_chapters(working_dir, book_output_dir, source_audio_dir=source_audio_dir)

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
        if working_dir.exists():
            try:
                shutil.rmtree(working_dir)
            except Exception:
                pass
        print(f"[ABI-Pipeline] Book '{book_name}' transcription finished successfully.")
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


def tokens_to_words(tokens: list[str], timestamps: list[float], start_offset_ms: int = 0) -> list[dict]:
    """
    Combines subword tokens and token-level timestamps from Parakeet ONNX into whole word objects
    with native millisecond start_ms and end_ms boundaries.
    """
    if not tokens or not timestamps or len(tokens) != len(timestamps):
        return []

    words = []
    curr_word_chars = []
    curr_start_ms = None
    curr_last_ts_ms = None

    for i, (tok, ts) in enumerate(zip(tokens, timestamps)):
        ts_ms = start_offset_ms + int(round(ts * 1000))
        
        # Detect token word boundaries (leading spaces or BPE unicode space prefix)
        is_word_start = tok.startswith(' ') or tok.startswith(' ') or (i == 0)
        
        if is_word_start and curr_word_chars:
            word_str = "".join(curr_word_chars).strip()
            if word_str:
                end_ms = max(curr_start_ms + 80, ts_ms)
                words.append({
                    "word": word_str,
                    "start_ms": curr_start_ms,
                    "end_ms": end_ms
                })
            curr_word_chars = []
            curr_start_ms = None

        clean_tok = tok.lstrip(' ').lstrip(' ')
        if curr_start_ms is None:
            curr_start_ms = ts_ms

        curr_word_chars.append(clean_tok)
        curr_last_ts_ms = ts_ms

    # Final word closeout
    if curr_word_chars and curr_start_ms is not None:
        word_str = "".join(curr_word_chars).strip()
        if word_str:
            end_ms = max(curr_start_ms + 80, curr_last_ts_ms + 120)
            words.append({
                "word": word_str,
                "start_ms": curr_start_ms,
                "end_ms": end_ms
            })

    return words


def split_words_into_sentences(words_data: list[dict]) -> list[dict]:
    """
    Groups a flat list of word-timestamp dicts into sentence-level subtitle blocks.
    Each block contains start_ms, end_ms, text, and its corresponding list of word objects.
    """
    if not words_data:
        return []

    sentences = []
    current_words = []

    for item in words_data:
        current_words.append(item)
        word_str = item["word"]
        
        # Split sentence at punctuation (. ! ?)
        if re.search(r'[.!?]["’\'”]?$', word_str):
            sent_text = " ".join(w["word"] for w in current_words)
            sentences.append({
                "start_ms": current_words[0]["start_ms"],
                "end_ms": current_words[-1]["end_ms"],
                "text": sent_text,
                "words": current_words
            })
            current_words = []

    # Handle trailing words without end punctuation
    if current_words:
        sent_text = " ".join(w["word"] for w in current_words)
        sentences.append({
            "start_ms": current_words[0]["start_ms"],
            "end_ms": current_words[-1]["end_ms"],
            "text": sent_text,
            "words": current_words
        })

    return sentences


def transcribe_chapter(chapter: Chapter, model, working_dir: Path, start_offset_ms: int = 0) -> tuple[str, int]:
    """Preprocesses a chapter's audio track, slices it, and performs speech-to-text with native word timing extraction."""
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
        return "", 0

    chapter_dur_ms = 0
    try:
        probe = ffmpeg.probe(str(preprocessed_wav))
        chapter_dur_ms = int(round(float(probe['format']['duration']) * 1000))
    except Exception:
        pass

    save_subtitles = get_setting("save_word_timestamps", "True") in ("True", True)

    if type(model).__name__ == "WhisperModel":
        print(f"[ABI-Pipeline] Transcribing '{chapter.title}' with Faster-Whisper...")
        start_time = time.time()
        
        segments, info = model.transcribe(
            str(preprocessed_wav), 
            vad_filter=True, 
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=save_subtitles
        )
        
        segments_list = list(segments)
        timing_map = []
        all_chapter_words = []
        current_char_offset = 0
        cleaned_texts = []
        
        for segment in segments_list:
            text_str = segment.text.strip()
            if not text_str:
                continue

            cleaned_texts.append(text_str)
            text_len = len(text_str)
            
            timing_map.append({
                "start": segment.start,
                "end": segment.end,
                "char_start": current_char_offset,
                "char_end": current_char_offset + text_len
            })
            current_char_offset += text_len + 1  # accounts for space separator

            if save_subtitles:
                if hasattr(segment, "words") and segment.words:
                    for w in segment.words:
                        w_text = w.word.strip()
                        if w_text:
                            all_chapter_words.append({
                                "word": w_text,
                                "start_ms": start_offset_ms + int(round(w.start * 1000)),
                                "end_ms": start_offset_ms + int(round(w.end * 1000))
                            })
                else:
                    words_in_seg = text_str.split()
                    total_chars = max(1, sum(len(w) for w in words_in_seg))
                    seg_dur_ms = max(1, int(round((segment.end - segment.start) * 1000)))
                    curr_ms = start_offset_ms + int(round(segment.start * 1000))
                    for w_text in words_in_seg:
                        w_dur = int(round((len(w_text) / total_chars) * seg_dur_ms))
                        all_chapter_words.append({
                            "word": w_text,
                            "start_ms": curr_ms,
                            "end_ms": curr_ms + w_dur
                        })
                        curr_ms += w_dur

        chapter_json = working_dir / f"chapter_{chapter.chapter_num}.json"
        with open(chapter_json, "w", encoding="utf-8") as jf:
            json.dump(timing_map, jf, indent=4)

        if save_subtitles:
            subtitle_sentences = split_words_into_sentences(all_chapter_words)
            sub_json = working_dir / f"chapter_{chapter.chapter_num}_subtitles.json"
            with open(sub_json, "w", encoding="utf-8") as sjf:
                json.dump(subtitle_sentences, sjf, indent=4)
            
        total_time = time.time() - start_time
        print(f"[ABI-Pipeline] Chapter {chapter.chapter_num} complete! Total time: {total_time:.2f}s")
        
        if preprocessed_wav.exists():
            preprocessed_wav.unlink()
            
        return " ".join(cleaned_texts).strip(), chapter_dur_ms

    temp_chunk_dir = working_dir / f"chapter_{chapter.chapter_num}_chunks"
    print(f"[ABI-Pipeline] Chunking with ffmpeg silence detection...")
    chunk_paths, chunk_timings = chunk_audio_with_ffmpeg(preprocessed_wav, temp_chunk_dir)

    if not chunk_paths:
        if preprocessed_wav.exists(): preprocessed_wav.unlink()
        return "", chapter_dur_ms

    batch_size = int(get_setting("batch_size", 8))
    print(f"[ABI-Pipeline] Generated {len(chunk_paths)} chunks. Starting inference (Batch Size: {batch_size}, Native Timestamps: {save_subtitles})...")
    
    chunks_with_metadata = [(i, str(p), os.path.getsize(p)) for i, p in enumerate(chunk_paths)]
    chunks_with_metadata.sort(key=lambda x: x[2], reverse=True)
    
    all_texts = [None] * len(chunk_paths)
    all_chunk_words = [[] for _ in range(len(chunk_paths))]
    num_batches = math.ceil(len(chunks_with_metadata) / batch_size)
    start_time_total = time.time()

    # Wrap model with timestamp provider if subtitles are requested
    inference_model = model.with_timestamps() if save_subtitles else model
    
    for i in range(num_batches):
        batch_start_time = time.time()
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_meta = chunks_with_metadata[start_idx:end_idx]
        batch_chunks = [m[1] for m in batch_meta]
        original_indices = [m[0] for m in batch_meta]

        try:
            batch_results = inference_model.recognize(batch_chunks)
            if batch_results:
                if not isinstance(batch_results, list):
                    batch_results = [batch_results]
                
                for idx, result in zip(original_indices, batch_results):
                    if result:
                        text_val = result.text.strip() if hasattr(result, "text") else str(result).strip()
                        all_texts[idx] = text_val
                        
                        if save_subtitles and hasattr(result, "tokens") and hasattr(result, "timestamps"):
                            chunk_start_t, _ = chunk_timings[idx]
                            chunk_offset_ms = start_offset_ms + int(round(chunk_start_t * 1000))
                            chunk_words = tokens_to_words(result.tokens, result.timestamps, chunk_offset_ms)
                            all_chunk_words[idx] = chunk_words

        except Exception as batch_error:
            print(f"\n[STT Fallback] Batch inference failed on chapter {chapter.chapter_num}, batch {i+1}. Error: {batch_error}")
            for idx, chunk in zip(original_indices, batch_chunks):
                try:
                    result = inference_model.recognize(chunk)
                    if result:
                        text_val = result.text.strip() if hasattr(result, "text") else str(result).strip()
                        all_texts[idx] = text_val
                        if save_subtitles and hasattr(result, "tokens") and hasattr(result, "timestamps"):
                            chunk_start_t, _ = chunk_timings[idx]
                            chunk_offset_ms = start_offset_ms + int(round(chunk_start_t * 1000))
                            all_chunk_words[idx] = tokens_to_words(result.tokens, result.timestamps, chunk_offset_ms)
                except Exception: pass
            
        batch_time = time.time() - batch_start_time
        chunks_per_sec = len(batch_chunks) / batch_time if batch_time > 0 else 0
        print(f"  -> Batch {i+1}/{num_batches} processed {len(batch_chunks)} chunks in {batch_time:.2f}s ({chunks_per_sec:.2f} chunk/s)")
        gc.collect()

    timing_map = []
    all_chapter_words = []
    current_char_offset = 0
    cleaned_texts = []

    for idx in range(len(chunk_paths)):
        text = all_texts[idx]
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
        current_char_offset += text_len + 1  # accounts for space separator

        if save_subtitles and all_chunk_words[idx]:
            all_chapter_words.extend(all_chunk_words[idx])

    chapter_json = working_dir / f"chapter_{chapter.chapter_num}.json"
    with open(chapter_json, "w", encoding="utf-8") as jf:
        json.dump(timing_map, jf, indent=4)

    if save_subtitles:
        subtitle_sentences = split_words_into_sentences(all_chapter_words)
        sub_json = working_dir / f"chapter_{chapter.chapter_num}_subtitles.json"
        with open(sub_json, "w", encoding="utf-8") as sjf:
            json.dump(subtitle_sentences, sjf, indent=4)

    total_time = time.time() - start_time_total
    avg_speed = len(chunk_paths) / total_time if total_time > 0 else 0
    print(f"[ABI-Pipeline] Chapter {chapter.chapter_num} complete! Total time: {total_time:.2f}s ({avg_speed:.2f} chunk/s avg)\n")

    if preprocessed_wav.exists(): preprocessed_wav.unlink()
    if temp_chunk_dir.exists():
        try: shutil.rmtree(temp_chunk_dir)
        except Exception: pass

    return " ".join(cleaned_texts).strip(), chapter_dur_ms


def combine_chapters(working_dir: Path, book_output_dir: Path, source_audio_dir: Optional[Path] = None) -> None:
    """Appends all temporary chapter text files into final transcript.txt and merges transcript.json."""
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
        if t_file.stem.endswith("_subtitles"):
            continue
        try:
            ch_num = t_file.stem.split('_')[1]
            with open(t_file, "r", encoding="utf-8") as f:
                master_timing["chapters"][ch_num] = json.load(f)
        except Exception as e:
            print(f"[Timing-Sync] Error merging chapter timing metadata: {e}")

    if master_timing["chapters"]:
        with open(book_output_dir / "transcript_timing.json", "w", encoding="utf-8") as f:
            json.dump(master_timing, f, indent=4)

    # Combine chapter subtitle word-timestamp JSON files into continuous transcript.json
    subtitle_files = sorted(
        list(working_dir.glob("chapter_*_subtitles.json")),
        key=lambda x: int(x.stem.split('_')[1])
    )
    if subtitle_files:
        master_subtitles = []
        for s_file in subtitle_files:
            try:
                with open(s_file, "r", encoding="utf-8") as sf:
                    chapter_subs = json.load(sf)
                    master_subtitles.extend(chapter_subs)
            except Exception as e:
                print(f"[Subtitle-Sync] Error reading {s_file.name}: {e}")

        if master_subtitles:
            # 1. Save copy to ABI-Pipeline output directory
            transcript_json_path = book_output_dir / "transcript.json"
            with open(transcript_json_path, "w", encoding="utf-8") as sof:
                json.dump(master_subtitles, sof, indent=2)
            print(f"[ABI-Pipeline] Saved transcript.json to output directory: {transcript_json_path}")

            # 2. Save directly to original audiobook folder next to audio files
            if source_audio_dir and source_audio_dir.exists():
                try:
                    audio_dir_transcript = source_audio_dir / "transcript.json"
                    with open(audio_dir_transcript, "w", encoding="utf-8") as asof:
                        json.dump(master_subtitles, asof, indent=2)
                    print(f"[ABI-Pipeline] Saved transcript.json directly to audiobook folder: {audio_dir_transcript}")
                except Exception as e:
                    print(f"[ABI-Pipeline] Warning: Could not write transcript.json to audiobook folder {source_audio_dir}: {e}")