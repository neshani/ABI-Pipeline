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


def chunk_audio_with_ffmpeg(audio_path: Path, output_dir: Path) -> List[Path]:
    """
    Splits a 16kHz mono WAV file into ~60-second chunks using FFmpeg silence detection.
    Extremely fast, low-overhead, and completely self-contained.
    """
    SILENCE_THRESHOLD_DB = "-30dB"
    SILENCE_DURATION_S = "0.5"
    TARGET_CHUNK_S = 60

    chunk_paths = []
    try:
        # Run ffmpeg silencedetect and capture stderr output
        command = [
            'ffmpeg', '-i', str(audio_path),
            '-af', f'silencedetect=n={SILENCE_THRESHOLD_DB}:d={SILENCE_DURATION_S}',
            '-f', 'null', '-'
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        stderr_output = result.stderr

        # Extract safety silence cut-points
        silence_ends = re.findall(r"silence_end: (\d+\.?\d*)", stderr_output)
        cut_timestamps = [float(t) for t in silence_ends]

        # Probe total audio duration
        probe = ffmpeg.probe(str(audio_path))
        duration = float(probe['format']['duration'])

        # Group cuts to align chunks around our TARGET_CHUNK_S limit
        last_cut = 0.0
        final_cuts = [0.0]
        for t in cut_timestamps:
            if t - last_cut > TARGET_CHUNK_S:
                final_cuts.append(t)
                last_cut = t
        final_cuts.append(duration)

        # Slice the audio file into segment chunks
        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(final_cuts) - 1):
            start = final_cuts[i]
            end = final_cuts[i+1]
            if end - start < 0.5:
                continue

            chunk_file = output_dir / f"chunk_{i+1}.wav"
            (
                ffmpeg.input(str(audio_path), ss=start, to=end)
                .output(str(chunk_file), acodec='pcm_s16le', ac=1, ar='16000', loglevel="panic")
                .run(overwrite_output=True)
            )
            chunk_paths.append(chunk_file)

        return chunk_paths
    except Exception as e:
        print(f"ERROR: FFmpeg chunking failed for {audio_path.name}: {e}")
        return []


def get_onnx_model():
    """
    Dynamically loads the onnx-asr model locally or via huggingface.
    Optimized for high-performance FP16 execution on Nvidia GPUs with memory limits.
    """
    import onnx_asr
    model_dir = os.path.abspath(".models/parakeet")
    device_setting = get_setting("stt_device", "GPU/CUDA")

    if device_setting == "GPU/CUDA":
        # Highly optimized hardware-level parameters for Nvidia GPUs (RTX 3090/4090)
        gpu_options = {
            "device_id": "0",
            "arena_extend_strategy": "kSameAsRequested",
            "do_copy_in_default_stream": "1",
        }
        providers = [("CUDAExecutionProvider", gpu_options), "CPUExecutionProvider"]
        
        print(f"Initializing Parakeet ONNX engine in FP16 mode on: {device_setting}")
        try:
            # Check if the specific local fp16 quantized files exist locally
            local_fp16_exists = os.path.exists(os.path.join(model_dir, "encoder-model.fp16.onnx"))
            
            if local_fp16_exists:
                # Load FP16 from local directory
                return onnx_asr.load_model(
                    "nemo-parakeet-tdt-0.6b-v3", 
                    model_dir, 
                    quantization="fp16",
                    providers=providers
                )
            else:
                # Let onnx-asr fetch the FP16 model automatically (cached on Hugging Face hub)
                print("Local FP16 files not found in .models folder. Fetching from Hugging Face cache...")
                return onnx_asr.load_model(
                    "nemo-parakeet-tdt-0.6b-v3", 
                    quantization="fp16",
                    providers=providers
                )
        except Exception as e:
            print(f"GPU FP16 initialization failed: {e}")
            print("Falling back gracefully to CPU Execution...")
            providers = ["CPUExecutionProvider"]

    # CPU standard path
    print(f"Loading Parakeet ONNX model on CPU (Providers: {providers})")
    if os.path.exists(os.path.join(model_dir, "encoder-model.onnx")):
        return onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3", 
            model_dir, 
            providers=providers
        )
    else:
        return onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3", 
            providers=providers
        )


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

        # Fetch all books tied to this project
        books = session.exec(
            select(Book).where(Book.project_id == project_id)
        ).all()

    # 1. Initialize our STT engine
    try:
        model = get_onnx_model()
    except Exception as e:
        print(f"CRITICAL: Failed to load ONNX model weights: {e}")
        with Session(engine) as session:
            project = session.get(Project, project_id)
            if project:
                project.status = "Failed"
                session.add(project)
                session.commit()
        active_projects.discard(project_id)
        return

    # 2. Transcribe books sequentially
    for book in books:
        if project_id in cancelled_projects:
            break
        transcribe_book(book.id, model, project_id)

    # 3. Release model execution session VRAM explicitly
    try:
        if hasattr(model, 'asr'):
            if hasattr(model.asr, '_encoder'):
                model.asr._encoder.set_providers([])
            if hasattr(model.asr, '_decoder_joint'):
                model.asr._decoder_joint.set_providers([])
    except Exception as e:
        print(f"Error resetting provider sessions: {e}")

    # Delete references and run garbage collection
    del model
    gc.collect()

    # 4. Finalize project status
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
    with Session(engine) as session:
        book = session.get(Book, book_id)
        if not book or book.status == "Transcribed":
            return

        book.status = "Transcribing"
        session.add(book)
        session.commit()

        chapters = session.exec(
            select(Chapter).where(Chapter.book_id == book_id)
        ).all()
        total_chapters = len(chapters)

    # Output directory configuration
    output_dir = Path(get_setting("output_dir", "./output")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Establish localized working folder
    working_dir = Path("./workspace_temp") / f"book_{book_id}"
    if working_dir.exists():
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    # Transcribe chapters sequentially
    for chapter in chapters:
        if project_id in cancelled_projects:
            break

        if chapter.status == "Completed":
            continue

        with Session(engine) as session:
            db_chapter = session.get(Chapter, chapter.id)
            if db_chapter:
                db_chapter.status = "Transcribing"
                session.add(db_chapter)
                session.commit()

        # Run preprocessing, chunking, and model inference
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

        # Calculate and write back progress bar state
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

        # Clean python references after each chapter
        gc.collect()

    # Wrap up book state
    if project_id not in cancelled_projects:
        combine_chapters(book.name, working_dir, output_dir)
        with Session(engine) as session:
            db_book = session.get(Book, book_id)
            if db_book:
                db_book.status = "Transcribed"
                session.add(db_book)
                session.commit()
    else:
        # Reset the book status if cancelled mid-run
        with Session(engine) as session:
            db_book = session.get(Book, book_id)
            if db_book:
                db_book.status = "Imported"
                session.add(db_book)
                session.commit()

    # Workspace cleanup
    if working_dir.exists():
        try:
            shutil.rmtree(working_dir)
        except Exception:
            pass


def transcribe_chapter(chapter: Chapter, model, working_dir: Path) -> str:
    """Preprocesses a chapter's audio track, slices it, and performs speech-to-text in parallel batches."""
    preprocessed_wav = working_dir / f"temp_chapter_{chapter.chapter_num}_preprocessed.wav"
    try:
        ffmpeg_input = ffmpeg.input(chapter.input_file)
        if chapter.type == 'segment':
            ffmpeg_input = ffmpeg.input(
                chapter.input_file, 
                ss=chapter.start_time, 
                to=chapter.end_time
            )

        (
            ffmpeg_input
            .output(str(preprocessed_wav), acodec='pcm_s16le', ac=1, ar='16000', loglevel="panic")
            .run(overwrite_output=True)
        )
    except Exception as e:
        print(f"FFmpeg preprocessing failed for chapter {chapter.chapter_num}: {e}")
        return ""

    # Slice the preprocessed track using the self-contained ffmpeg silences detector
    temp_chunk_dir = working_dir / f"chapter_{chapter.chapter_num}_chunks"
    chunk_paths = chunk_audio_with_ffmpeg(preprocessed_wav, temp_chunk_dir)

    if not chunk_paths:
        if preprocessed_wav.exists():
            preprocessed_wav.unlink()
        return ""

    # Sequence parallel chunk batch inference
    all_texts = []
    batch_size = int(get_setting("batch_size", 8))
    num_batches = math.ceil(len(chunk_paths) / batch_size)
    
    try:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_chunks = [str(p) for p in chunk_paths[start_idx:end_idx]]

            # Pass the list directly to onnx-asr's recognize method to trigger GPU batching!
            batch_results = model.recognize(batch_chunks)
            
            if batch_results:
                if isinstance(batch_results, list):
                    all_texts.extend([r.strip() for r in batch_results if r])
                elif isinstance(batch_results, str):
                    all_texts.append(batch_results.strip())
            
            # Explicit garbage collection after each batch run to clear numpy structures
            gc.collect()

    except Exception as e:
        print(f"ONNX Inference error on chapter {chapter.chapter_num}: {e}")

    # File cleanups
    if preprocessed_wav.exists():
        preprocessed_wav.unlink()
    if temp_chunk_dir.exists():
        try:
            shutil.rmtree(temp_chunk_dir)
        except Exception:
            pass

    return " ".join(all_texts).strip()


def combine_chapters(book_title: str, working_dir: Path, output_dir: Path) -> None:
    """Appends all temporary chapter text files into the final transcript."""
    final_text_path = output_dir / f"{book_title}.txt"
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