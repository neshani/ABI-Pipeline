import csv
import re
import ffmpeg
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Book, Chapter

def get_audio_duration_with_ffmpeg(audio_path: str) -> float:
    """Queries ffmpeg.probe to retrieve track duration in seconds."""
    try:
        probe = ffmpeg.probe(audio_path)
        return float(probe['format']['duration'])
    except Exception as e:
        raise RuntimeError(f"ffmpeg-python failed to probe file '{audio_path}': {str(e)}")


def get_normalized_and_map(text: str) -> Tuple[str, List[int]]:
    """
    Normalizes a text string to alphanumeric lowercase characters 
    while preserving a mapping array pointing to original indexes.
    """
    normalized_chars = []
    original_indices = []
    for idx, char in enumerate(text):
        char_lower = char.lower()
        if char_lower.isalnum():
            normalized_chars.append(char_lower)
            original_indices.append(idx)
    return "".join(normalized_chars), original_indices


def find_quote_offset(transcript_text: str, quote: str) -> int:
    """
    Attempts exact search, falling back to mapping-preserved normalized character offsets, 
    and finally prefix/suffix-based sliding mapping for maximum fuzzy matching resilience.
    """
    if not transcript_text or not quote:
        return 0

    # 1. Quick exact substring match
    exact_idx = transcript_text.find(quote)
    if exact_idx != -1:
        return exact_idx

    # 2. Build index map and normalize
    norm_text, text_map = get_normalized_and_map(transcript_text)
    norm_quote, _ = get_normalized_and_map(quote)

    if not norm_quote or not norm_text:
        return 0

    # Try substring match on normalized text
    norm_idx = norm_text.find(norm_quote)
    if norm_idx != -1:
        return text_map[norm_idx]

    # 3. Sliding fallback prefixes (for trailing paraphrase/truncation anomalies)
    for prefix_len in [40, 30, 20]:
        if len(norm_quote) >= prefix_len:
            prefix = norm_quote[:prefix_len]
            norm_idx = norm_text.find(prefix)
            if norm_idx != -1:
                return text_map[norm_idx]

    # 4. Sliding fallback suffixes (for leading paraphrase anomalies like "I clamped" vs "I was clamping")
    for suffix_len in [40, 30, 20]:
        if len(norm_quote) >= suffix_len:
            suffix = norm_quote[-suffix_len:]
            norm_idx = norm_text.find(suffix)
            if norm_idx != -1:
                # Estimate the start of the quote by offsetting backward from the matched suffix
                estimated_norm_idx = max(0, norm_idx - (len(norm_quote) - suffix_len))
                return text_map[estimated_norm_idx]

    return 0


def format_seconds(seconds: float) -> str:
    """Converts seconds float into HH:MM:SS format string."""
    s = int(round(seconds))
    hours = s // 3600
    minutes = (s % 3600) // 60
    secs = s % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sync_book_timing(book_id: int, project_name: str, book_name: str, console_log_cb=None) -> bool:
    """
    Scans the book's chapters and original audio sources stored in the database,
    calculates exact timestamps for all extracted quotes, and writes them back to prompts.csv.
    """
    def log(msg: str):
        if console_log_cb:
            console_log_cb(msg)
        else:
            print(msg)

    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    book_dir = base_output_dir / project_name / book_name
    transcript_path = book_dir / "transcript.txt"
    prompts_path = book_dir / "prompts.csv"

    if not transcript_path.exists():
        log(f"[Timing-Sync] Error: Missing transcript.txt at: {transcript_path}")
        return False
    if not prompts_path.exists():
        log(f"[Timing-Sync] Error: Missing prompts.csv at: {prompts_path}")
        return False

    with Session(engine) as session:
        book = session.get(Book, book_id)
        if not book:
            log(f"[Timing-Sync] Error: Book ID {book_id} not found in database.")
            return False

        # Load all chapters sorted by chapter number
        chapters_db = session.exec(
            select(Chapter).where(Chapter.book_id == book_id).order_by(Chapter.chapter_num)
        ).all()

    if not chapters_db:
        log(f"[Timing-Sync] Error: No chapters found in database for book: {book_name}")
        return False

    log(f"[Timing-Sync] Synchronizing timing for book '{book_name}' using {len(chapters_db)} chapters...")

    # Load chapter splits from master transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        full_transcript = f.read()

    transcript_splits = [ch.strip() for ch in full_transcript.split("==CHAPTER==") if ch.strip()]
    log(f"[Timing-Sync] Transcript parsed into {len(transcript_splits)} split text block(s).")

    # Cache file durations dynamically to avoid duplicate probes
    timing_map = {}
    cumulative_time = 0.0

    for ch in chapters_db:
        c_num = ch.chapter_num
        start_t = 0.0
        dur = 0.0

        if ch.type == 'segment':
            # Single-file structure: Chapter start and end times are absolute offsets in the original audio
            start_t = ch.start_time or 0.0
            if ch.end_time is not None:
                dur = ch.end_time - start_t
            else:
                try:
                    total_dur = get_audio_duration_with_ffmpeg(ch.input_file)
                    dur = total_dur - start_t
                except Exception as e:
                    log(f"[Timing-Sync] Probe failed for segment chapter {c_num}: {str(e)}")
                    dur = 0.0
        else:
            # Multi-file structure: Chapter is a whole track file
            start_t = cumulative_time
            try:
                dur = get_audio_duration_with_ffmpeg(ch.input_file)
            except Exception as e:
                log(f"[Timing-Sync] Probe failed for track file chapter {c_num}: {str(e)}")
                dur = 0.0
            cumulative_time += dur

        timing_map[c_num] = (start_t, dur)

    # Read the current prompts.csv
    rows = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        for row in reader:
            rows.append(row)

    if not rows:
        log("[Timing-Sync] No prompt entries in prompts.csv to process.")
        return True

    if "timestamp" not in fieldnames:
        fieldnames.append("timestamp")

    # Load transcript_timing.json sub-chapter chunk alignment if available
    import json
    timing_json_path = book_dir / "transcript_timing.json"
    timing_data = {}
    if timing_json_path.exists():
        try:
            with open(timing_json_path, "r", encoding="utf-8") as f:
                timing_data = json.load(f).get("chapters", {})
            log("[Timing-Sync] Loaded sub-chapter timing alignments from transcript_timing.json!")
        except Exception as e:
            log(f"[Timing-Sync] Warning: Failed to load transcript_timing.json: {e}")

    # Match rows to quotes
    for row in rows:
        quote = row.get("quote", "").strip()
        if not quote or quote.upper() == "NONE" or quote.upper() == "REFUSAL":
            row["timestamp"] = "00:00:00"
            continue

        try:
            chapter_num = int(float(row.get("chapter", 1)))
        except ValueError:
            chapter_num = 1

        # Retrieve mapped text for this chapter
        chapter_idx = max(0, chapter_num - 1)
        if chapter_idx < len(transcript_splits):
            target_text = transcript_splits[chapter_idx]
        else:
            target_text = transcript_splits[-1] if transcript_splits else ""

        # Retrieve timing config for this chapter
        ch_start, ch_dur = timing_map.get(chapter_num, (0.0, 0.0))

        offset = find_quote_offset(target_text, quote)
        total_len = len(target_text) if target_text else 1
        if total_len == 0:
            total_len = 1

        # Locate exact sub-chapter chunk mapping
        ch_timing = timing_data.get(str(chapter_num))
        matched_chunk = None
        if ch_timing:
            for chunk in ch_timing:
                if chunk["char_start"] <= offset <= chunk["char_end"]:
                    matched_chunk = chunk
                    break
            # Fallback to closest chunk on error
            if not matched_chunk and ch_timing:
                matched_chunk = min(
                    ch_timing,
                    key=lambda c: min(abs(offset - c["char_start"]), abs(offset - c["char_end"]))
                )

        if matched_chunk:
            chunk_char_start = matched_chunk["char_start"]
            chunk_char_end = matched_chunk["char_end"]
            chunk_char_len = max(1, chunk_char_end - chunk_char_start)
            
            chunk_ratio = (offset - chunk_char_start) / chunk_char_len
            chunk_ratio = max(0.0, min(1.0, chunk_ratio))
            
            chunk_start_time = matched_chunk["start"]
            chunk_end_time = matched_chunk["end"]
            chunk_dur = chunk_end_time - chunk_start_time
            
            estimated_seconds_in_chapter = chunk_start_time + (chunk_ratio * chunk_dur)
            estimated_seconds = ch_start + estimated_seconds_in_chapter
        else:
            # Fallback to linear chapter interpolation
            ratio = offset / total_len
            estimated_seconds = ch_start + (ratio * ch_dur)

        row["timestamp"] = format_seconds(estimated_seconds)

    # Save timing results back to prompts.csv
    with open(prompts_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()
        writer.writerows(rows)

    log(f"[Timing-Sync] Timestamps successfully mapped and written to prompts.csv!")
    return True