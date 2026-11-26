import csv
import json
import re
import ffmpeg
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Book, Chapter, Project
from services import compat

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


class QuoteTime(NamedTuple):
    """Where a quote sits, and how much that number can be trusted.

    ``how`` is ``"chunk"`` when the quote was found inside a timed chunk - the
    normal case - or ``"chapter"`` when no chunk covers it and the position was
    interpolated across the whole chapter by character ratio. A chapter-level
    number is the suspicious one: it can be minutes out.
    """

    seconds: float
    how: str
    offset: int


def _relative_chapters(chapters: Dict[str, list],
                       timing_map: Dict[int, Tuple[float, float]]
                       ) -> Tuple[Dict[str, list], int]:
    """Chunk times rebased onto their own chapter, plus how many chapters needed it.

    The stored convention is a property of the writer, not of a chapter, so it is decided once
    per file. The witness is the chapter with the largest start: an early chapter differs from
    chapter-relative by only its own (small) start, so it cannot tell the two apart, but the last
    chapter's absolute times land near the book's end instead of near its own length.

    A file that stores whole-file seconds was written by the transcription path that did so for
    single-file books (fixed in ``transcription.py``); :meth:`BookTiming.estimate` adds the
    chapter start below, so such a book would have every chapter after the first counted twice.
    Books transcribed before that fix are repaired here rather than silently mistimed - the rebase
    is exact, because it is the same constant the writer should have subtracted.
    """
    copied: Dict[str, list] = {
        key: [dict(chunk) for chunk in chunks] for key, chunks in (chapters or {}).items()}

    # The stored convention is a property of the writer, not of a chapter, so fit both hypotheses
    # across the whole file and keep the better one:
    #   relative -> the last chunk ends near the chapter's own length
    #   absolute -> the last chunk ends near start + length
    # One chapter cannot decide this (an early chapter's two answers differ by only its small
    # start, and a long final chapter can make the relative answer look absolute), which is why
    # this is a sum over every chapter rather than a threshold on one.
    err_relative = err_absolute = 0.0
    for key, rows in copied.items():
        try:
            start, duration = timing_map.get(int(float(key)), (0.0, 0.0))
        except (TypeError, ValueError):
            continue
        if not rows or duration <= 0:
            continue
        max_end = max(chunk["end"] for chunk in rows)
        err_relative += abs(max_end - duration)
        err_absolute += abs(max_end - (start + duration))

    rebased = 0
    if err_absolute * 2 < err_relative:
        for key, rows in copied.items():
            try:
                start = timing_map.get(int(float(key)), (0.0, 0.0))[0]
            except (TypeError, ValueError):
                continue
            if not rows or not start:
                continue
            for chunk in rows:
                chunk["start"] = chunk["start"] - start
                chunk["end"] = chunk["end"] - start
            rebased += 1
    return copied, rebased


class BookTiming:
    """Maps a quote to an absolute timestamp for one book.

    The bulk pass (:func:`sync_book_timing`) and the single-row path
    (:func:`retime_quote`) both go through :meth:`estimate`, so a row retimed
    in the editor cannot disagree with what a later full sync computes for the
    same quote.
    """

    def __init__(self, splits: List[str], full_text: str,
                 timing_chapters: Dict[str, list],
                 timing_map: Dict[int, Tuple[float, float]],
                 is_single_db_chapter: bool,
                 chapter_numbers: List[int]):
        self.splits = splits
        self.full_text = full_text
        self.timing_map = timing_map or {}
        self.is_single_db_chapter = is_single_db_chapter
        self.chapter_numbers = list(chapter_numbers)
        self.timing_chapters, self.rebased = _relative_chapters(
            timing_chapters, self.timing_map)

    def _key(self, chapter_num: int) -> int:
        """The chapter the database times and chunking are filed under.

        A book transcribed as one continuous file has a single chapter row, so
        every prompt row maps onto that one entry.
        """
        if self.is_single_db_chapter and self.chapter_numbers:
            return self.chapter_numbers[0]
        return chapter_num

    def _text(self, chapter_num: int) -> str:
        if self.is_single_db_chapter:
            return self.full_text
        index = max(0, chapter_num - 1)
        if index < len(self.splits):
            return self.splits[index]
        return self.splits[-1] if self.splits else ""

    def _chunks(self, chapter_num: int) -> Optional[list]:
        return self.timing_chapters.get(str(self._key(chapter_num)))

    def estimate(self, chapter_num: int, quote: str) -> QuoteTime:
        """Best available timestamp for ``quote`` in ``chapter_num``."""
        text = self._text(chapter_num)
        start, duration = self.timing_map.get(self._key(chapter_num), (0.0, 0.0))
        offset = find_quote_offset(text, quote)
        total_len = len(text) if text else 1
        if total_len == 0:
            total_len = 1

        matched_chunk = None
        chunks = self._chunks(chapter_num)
        if chunks:
            for chunk in chunks:
                if chunk["char_start"] <= offset <= chunk["char_end"]:
                    matched_chunk = chunk
                    break
            # Fallback to closest chunk on error
            if not matched_chunk:
                matched_chunk = min(
                    chunks,
                    key=lambda c: min(abs(offset - c["char_start"]),
                                      abs(offset - c["char_end"]))
                )

        if matched_chunk:
            chunk_char_start = matched_chunk["char_start"]
            chunk_char_end = matched_chunk["char_end"]
            chunk_char_len = max(1, chunk_char_end - chunk_char_start)
            ratio = (offset - chunk_char_start) / chunk_char_len
            ratio = max(0.0, min(1.0, ratio))
            chunk_duration = matched_chunk["end"] - matched_chunk["start"]
            return QuoteTime(
                start + matched_chunk["start"] + (ratio * chunk_duration),
                "chunk", offset)

        # No chunk data at all: linear interpolation across the chapter.
        return QuoteTime(start + (offset / total_len) * duration, "chapter", offset)


# One timing context at a time is all the editor needs; a full pass holds its
# own instance. Keyed on every input, so a re-alignment or a chapter retime
# rebuilds it rather than serving stale positions.
_CONTEXT_CACHE: Dict[tuple, BookTiming] = {}


def _file_signature(path: Path) -> tuple:
    try:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (0, 0)


def load_book_timing(project_name: str, book_name: str) -> Optional[BookTiming]:
    """Timing context for one book, without probing audio.

    Reads the transcript, the chunk timings and the chapters' own start/end
    times - enough to place a single quote - so it is cheap enough to call from
    a click handler.
    """
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    book_dir = base_output_dir / project_name / book_name
    transcript_path = book_dir / "transcript.txt"
    timing_path = book_dir / "transcript_timing.json"
    if not transcript_path.exists():
        return None

    with Session(engine) as session:
        project = session.exec(
            select(Project).where(Project.name == project_name)).first()
        if not project:
            return None
        book = session.exec(select(Book).where(
            Book.project_id == project.id, Book.name == book_name)).first()
        if not book:
            return None
        chapters = session.exec(
            select(Chapter).where(Chapter.book_id == book.id)).all()

    timing_map: Dict[int, Tuple[float, float]] = {}
    for chapter in chapters:
        start = float(chapter.start_time or 0.0)
        if chapter.end_time is not None:
            timing_map[chapter.chapter_num] = (start, float(chapter.end_time) - start)
        else:
            timing_map[chapter.chapter_num] = (start, 0.0)

    key = (
        book.id,
        _file_signature(transcript_path),
        _file_signature(timing_path),
        tuple(sorted(timing_map.items())),
    )
    cached = _CONTEXT_CACHE.get(key)
    if cached is not None:
        return cached

    full_text = transcript_path.read_text(encoding="utf-8")
    splits = [block.strip() for block in full_text.split("==CHAPTER==") if block.strip()]
    timing_data: Dict[str, list] = {}
    if timing_path.exists():
        try:
            timing_data = json.loads(
                timing_path.read_text(encoding="utf-8")).get("chapters", {})
        except Exception:
            timing_data = {}

    context = BookTiming(
        splits=splits,
        full_text=full_text,
        timing_chapters=timing_data,
        timing_map=timing_map,
        is_single_db_chapter=(len(chapters) == 1),
        chapter_numbers=[chapter.chapter_num for chapter in chapters],
    )
    _CONTEXT_CACHE.clear()
    _CONTEXT_CACHE[key] = context
    return context


def retime_quote(project_name: str, book_name: str, chapter_num,
                 quote: str) -> Tuple[Optional[str], str]:
    """Timestamp for one quote, in the form prompts.csv stores, plus its basis.

    Call this whenever a row's quote changes: the quote is what anchors the row
    in the audio, so a new quote with the old timestamp is a shot showing one
    passage while another is narrated.

    Returns ``(None, reason)`` and leaves the caller's timestamp alone when
    there is nothing to place - no quote (``"empty"``), a chapter that is not a
    number (``"bad_chapter"``), or no transcript/timing for the volume
    (``"unavailable"``). ``reason`` is ``"chunk"`` or ``"chapter"`` when a
    timestamp was produced.
    """
    text = (quote or "").strip()
    if not text or text.upper() in ("NONE", "REFUSAL"):
        return None, "empty"
    try:
        chapter = int(float(chapter_num))
    except (TypeError, ValueError):
        return None, "bad_chapter"
    if chapter < 1:
        # No book numbers a chapter 0; treating it as the last split (what the
        # chapter-text lookup would do) would place the row in a real chapter.
        return None, "bad_chapter"

    context = load_book_timing(project_name, book_name)
    if context is None:
        return None, "unavailable"

    result = context.estimate(chapter, text)
    return format_seconds(result.seconds), result.how


def sync_book_timing(book_id: int, project_name: str, book_name: str, console_log_cb=None, auto_approve: bool = False) -> bool:
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

    log(f"[Timing-Sync] Synchronizing timing for book '{book_name}' using {len(chapters_db)} database chapters...")
    # Load chapter splits from master transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        full_transcript = f.read()

    transcript_splits = [ch.strip() for ch in full_transcript.split("==CHAPTER==") if ch.strip()]
    log(f"[Timing-Sync] Transcript parsed into {len(transcript_splits)} split text block(s).")

    # Cache file durations dynamically to avoid duplicate probes
    timing_map = {}
    cumulative_time = 0.0

    # Write and persist probed duration metadata on Chapters and the Book to the DB
    with Session(engine) as write_session:
        # Re-fetch chapters to bind them to the active write session
        write_chapters = write_session.exec(
            select(Chapter).where(Chapter.book_id == book_id).order_by(Chapter.chapter_num)
        ).all()
        
        for ch in write_chapters:
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
                        # Persist back to DB
                        ch.start_time = start_t
                        ch.end_time = total_dur
                        write_session.add(ch)
                    except Exception as e:
                        log(f"[Timing-Sync] Probe failed for segment chapter {c_num}: {str(e)}")
                        dur = 0.0
            else:
                # Multi-file structure: Chapter is a whole track file
                start_t = cumulative_time
                try:
                    dur = get_audio_duration_with_ffmpeg(ch.input_file)
                    # Persist back to DB
                    ch.start_time = start_t
                    ch.end_time = start_t + dur
                    write_session.add(ch)
                except Exception as e:
                    log(f"[Timing-Sync] Probe failed for track file chapter {c_num}: {str(e)}")
                    dur = 0.0
                cumulative_time += dur

            timing_map[c_num] = (start_t, dur)
            
        # Cache total book duration directly on Book row
        db_book = write_session.get(Book, book_id)
        if db_book:
            if cumulative_time > 0:
                db_book.duration = cumulative_time
            elif timing_map:
                # For single-file segments, get max end_time
                max_end = max((t[0] + t[1]) for t in timing_map.values())
                if max_end > 0:
                    db_book.duration = max_end
            write_session.add(db_book)
            
        write_session.commit()

    # Read the current prompts.csv. Normalise the header first: the desired
    # order below is expressed in current names, so a pre-rename file must be
    # mapped onto them *before* the unknown-column diff is computed, or the
    # rewritten file would carry both a ``shot`` and a ``scene`` column.
    rows = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        reader.fieldnames = compat.normalize_fieldnames(reader.fieldnames)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        for row in reader:
            rows.append(row)

    if not rows:
        log("[Timing-Sync] No prompt entries in prompts.csv to process.")
        return True

    # Define exact, structured column layout required by the desktop client
    desired_order = ["chapter", "shot", "prompt", "quote", "approved", "timestamp"]
    other_fields = [f for f in fieldnames if f not in desired_order]
    final_fieldnames = desired_order + other_fields

    # Load transcript_timing.json sub-chapter chunk alignment if available
    timing_json_path = book_dir / "transcript_timing.json"
    timing_data = {}
    if timing_json_path.exists():
        try:
            with open(timing_json_path, "r", encoding="utf-8") as f:
                timing_data = json.load(f).get("chapters", {})
            log("[Timing-Sync] Loaded sub-chapter timing alignments from transcript_timing.json!")
        except Exception as e:
            log(f"[Timing-Sync] Warning: Failed to load transcript_timing.json: {e}")

    # Determine if we are matching a single-file book structure against a multi-chapter transcript
    is_single_db_chapter = (len(chapters_db) == 1)
    if is_single_db_chapter:
        log("[Timing-Sync] Continuous single-audio-file mapping enabled.")

    # Match rows to quotes. The estimator here is the same one the single-row
    # retime path uses, so the two can never disagree about where a quote sits.
    book_timing = BookTiming(
        splits=transcript_splits,
        full_text=full_transcript,
        timing_chapters=timing_data,
        timing_map=timing_map,
        is_single_db_chapter=is_single_db_chapter,
        chapter_numbers=[ch.chapter_num for ch in chapters_db],
    )
    if book_timing.rebased:
        log(f"[Timing-Sync] Warning: {book_timing.rebased} chapter(s) stored whole-file chunk "
            f"times (single-file transcription bug); rebased them before matching. Re-transcribe "
            f"or rewrite the file to clear this.")

    for row in rows:
        # Defensive check to initialize any missing keys
        for key in desired_order:
            if key not in row:
                row[key] = ""

        # Determine approval status based on context
        existing_approval = (row.get("approved") or "").strip().lower()
        if auto_approve:
            row["approved"] = "true"
        else:
            row["approved"] = existing_approval if existing_approval in ["true", "false"] else "false"

        raw_quote = row.get("quote")
        quote = raw_quote.strip() if raw_quote else ""
        if not quote or quote.upper() == "NONE" or quote.upper() == "REFUSAL":
            row["timestamp"] = "00:00:00"
            continue

        try:
            chapter_num = int(float(row.get("chapter", 1) or 1))
        except (ValueError, TypeError):
            chapter_num = 1

        if not is_single_db_chapter and chapter_num not in timing_map:
            log(f"[Timing-Sync] Warning: Chapter {chapter_num} from CSV not found in database chapters map.")

        row["timestamp"] = format_seconds(book_timing.estimate(chapter_num, quote).seconds)

    # Save timing results back to prompts.csv
    with open(prompts_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames, delimiter="|")
        writer.writeheader()
        writer.writerows(rows)

    log(f"[Timing-Sync] Timestamps successfully mapped and written to prompts.csv!")
    return True