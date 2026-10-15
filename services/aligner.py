"""Align an ASR transcript against a clean source (EPUB) text.

The Parakeet transcript mangles proper nouns and lore terms badly, but the
words are in the right order.  When a clean source text exists we can align the
two streams and *replace the transcript text with the book's own text*, while
carrying the ASR word timings across and deriving a wrong->right correction map.

Why not a plain diff
--------------------
The two streams are near-identical in order, but an edition can carry extra
material (a prologue the recording lacks, a glossary, publisher front matter)
and the recording can carry audio-only material (a publisher intro, a bonus
interview).  A global diff mis-anchors on those and dumps huge blocks into
``replace``.  Instead we chain unique N-gram anchors monotonically (LIS) and
only diff the small gaps between them - so the alignment is local, bounded and
drift-free, and the edition extras fall out as the head/tail of the chain.

What gets emitted
-----------------
* source text for the aligned middle (fixes spelling, punctuation, quotes,
  capitalisation at once),
* transcript text with the correction map applied for audio-only head/tail,
* nothing for source-only material (prologue/glossary the recording never had).

Timings are re-expressed as ``transcript_timing.json`` chunks that tile the new
chapter text, using the timestamps already present in ``transcript.json``.
"""
from __future__ import annotations

import bisect
import csv
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ANCHOR_N = 12
# Words per transcript_timing.json chunk.  The original files land near this
# granularity, and timing_sync interpolates inside a chunk by character ratio.
TIMING_CHUNK_WORDS = 12
# Above this many token-pair operations a gap is treated as pure insert/delete
# rather than diffed.  Only the edition extras (prologue / glossary / interview)
# are ever this large, and diffing them would be slow for no benefit.
_GAP_DIFF_LIMIT = 1_000_000

_APOSTROPHES = re.compile(r"[\u2018\u2019\u02bc\u201b]")
_WORD = re.compile(r"[A-Za-z0-9']+")


# --------------------------------------------------------------------------
# tokenisation helpers
# --------------------------------------------------------------------------

def tokenize(text: str) -> List[Tuple[str, int, int, str]]:
    """``[(normalised, char_start, char_end, surface), ...]`` for ``text``."""
    out: List[Tuple[str, int, int, str]] = []
    folded = _APOSTROPHES.sub("'", text)
    for m in _WORD.finditer(folded):
        surface = m.group(0)
        norm = surface.strip("'").lower()
        if norm:
            out.append((norm, m.start(), m.end(), surface))
    return out


def _load_chapter_blocks(transcript_text: str) -> List[str]:
    """Chapter blocks exactly as ``timing_sync`` sees them (split, stripped)."""
    text = transcript_text.replace("\r\n", "\n")
    return [block.strip() for block in text.split("==CHAPTER==") if block.strip()]


def _load_transcript_words(timing_json_path: Path
                           ) -> List[Tuple[str, int, int, int]]:
    """``[(normalised, start_ms, end_ms, segment_index), ...]`` from transcript.json."""
    segments = json.loads(Path(timing_json_path).read_text(encoding="utf-8"))
    words: List[Tuple[str, int, int, int]] = []
    for seg_index, seg in enumerate(segments):
        for w in seg.get("words", []):
            for token in _WORD.findall(_APOSTROPHES.sub("'", w["word"])):
                token = token.strip("'").lower()
                if token:
                    words.append((token, int(w["start_ms"]), int(w["end_ms"]),
                                  seg_index))
    return words


# --------------------------------------------------------------------------
# anchoring
# --------------------------------------------------------------------------

def _unique_ngrams(seq: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    """Index of N-grams that occur exactly once in ``seq``."""
    index: Dict[Tuple[str, ...], int] = {}
    dup: set = set()
    for i in range(len(seq) - n + 1):
        gram = tuple(seq[i:i + n])
        if gram in index:
            dup.add(gram)
        else:
            index[gram] = i
    return {g: i for g, i in index.items() if g not in dup}


def find_anchor_chain(asr: Sequence[str], epub: Sequence[str],
                      n: int = ANCHOR_N) -> List[Tuple[int, int]]:
    """Monotonic chain of unique-N-gram matches (LIS on the epub position)."""
    if len(asr) < n or len(epub) < n:
        return []
    epub_index = _unique_ngrams(epub, n)
    if not epub_index:
        return []
    raw = sorted(
        (i, epub_index[tuple(asr[i:i + n])])
        for i in range(len(asr) - n + 1)
        if tuple(asr[i:i + n]) in epub_index
    )
    tails: List[int] = []
    tail_pos: List[int] = []
    prev = [-1] * len(raw)
    for k, (_, j) in enumerate(raw):
        p = bisect.bisect_left(tails, j)
        if p == len(tails):
            tails.append(j)
            tail_pos.append(k)
        else:
            tails[p] = j
            tail_pos[p] = k
        prev[k] = tail_pos[p - 1] if p > 0 else -1
    chain: List[Tuple[int, int]] = []
    k = tail_pos[-1] if tail_pos else -1
    while k != -1:
        chain.append(raw[k])
        k = prev[k]
    chain.reverse()
    return chain


def _append_gap(ops: List[Tuple[str, int, int, int, int]],
                asr: Sequence[str], epub: Sequence[str],
                a0: int, a1: int, b0: int, b1: int) -> None:
    """Diff one unanchored gap and append its opcodes."""
    if a0 >= a1 and b0 >= b1:
        return
    if a0 >= a1:
        ops.append(("insert", a0, a1, b0, b1))
        return
    if b0 >= b1:
        ops.append(("delete", a0, a1, b0, b1))
        return
    if (a1 - a0) * (b1 - b0) > _GAP_DIFF_LIMIT:
        ops.append(("delete", a0, a1, b0, b0))
        ops.append(("insert", a1, a1, b0, b1))
        return
    matcher = SequenceMatcher(None, asr[a0:a1], epub[b0:b1], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ops.append((tag, a0 + i1, a0 + i2, b0 + j1, b0 + j2))


def compute_opcodes(asr: Sequence[str], epub: Sequence[str],
                    chain: Sequence[Tuple[int, int]], n: int = ANCHOR_N
                    ) -> List[Tuple[str, int, int, int, int]]:
    """Full opcode list covering ``asr`` and ``epub`` (anchors + diffed gaps)."""
    ops: List[Tuple[str, int, int, int, int]] = []
    prev_a = prev_b = 0
    for i, j in chain:
        _append_gap(ops, asr, epub, prev_a, i, prev_b, j)
        ops.append(("equal", i, i + n, j, j + n))
        prev_a, prev_b = i + n, j + n
    _append_gap(ops, asr, epub, prev_a, len(asr), prev_b, len(epub))
    return ops


def _absorb_head(asr: Sequence[str], epub: Sequence[str],
                 a_first: int, b_first: int) -> Tuple[int, int]:
    """Pull a short exact/near run at the region head back into the alignment."""
    if a_first <= 0 or b_first <= 0:
        return a_first, b_first
    head = asr[:a_first]
    lo = max(0, b_first - len(head) - 64)
    window = epub[lo:b_first]
    if not window:
        return a_first, b_first
    matcher = SequenceMatcher(None, head, window, autojunk=False)
    ga, gb = len(head), len(window)
    for tag, i1, i2, j1, j2 in reversed(matcher.get_opcodes()):
        if i2 != ga or j2 != gb:
            break
        if tag == "equal" or (tag == "replace" and (i2 - i1) <= 4 and (j2 - j1) <= 6):
            ga, gb = i1, j1
        else:
            break
    if ga == len(head):
        return a_first, b_first
    return ga, b_first - (len(window) - gb)


def _absorb_tail(asr: Sequence[str], epub: Sequence[str],
                 a_last: int, b_last: int) -> Tuple[int, int]:
    """Pull a short exact/near run at the region tail forward into the alignment."""
    tail = asr[a_last:]
    if not tail or b_last >= len(epub):
        return a_last, b_last
    window = epub[b_last:b_last + len(tail) + 64]
    matcher = SequenceMatcher(None, tail, window, autojunk=False)
    ga = gb = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if i1 != ga:
            break
        if tag == "equal" or (tag == "replace" and (i2 - i1) <= 4 and (j2 - j1) <= 6):
            ga, gb = i2, j2
        else:
            break
    if ga == 0:
        return a_last, b_last
    return a_last + ga, b_last + gb


# --------------------------------------------------------------------------
# correction map
# --------------------------------------------------------------------------

def build_correction_map(
    asr: Sequence[str],
    asr_surface: Sequence[str],
    epub: Sequence[str],
    epub_surface: Sequence[str],
    ops: Sequence[Tuple[str, int, int, int, int]],
    epub_vocab: set,
    region: Optional[Tuple[int, int]] = None,
    limit: int = 3,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """Derive ``(wrong, right, count)`` pairs from aligned ``replace`` spans.

    Only spans inside ``region`` (the aligned middle) are used: the head/tail
    gaps are edition extras (intro, interview vs front matter, glossary) and
    diffing them produces garbage pairs that would corrupt the text.  A pair is
    *unsafe* when the wrong side is also a legitimate word in the source book -
    substituting it globally would corrupt real words (``land -> lan``).
    """
    counts: Dict[Tuple[str, str], int] = {}
    surfaces: Dict[Tuple[str, str], Dict[str, int]] = {}
    for tag, a0, a1, b0, b1 in ops:
        if tag != "replace":
            continue
        if region is not None and not (region[0] <= a0 and a1 <= region[1]):
            continue
        if not (1 <= a1 - a0 <= limit and 1 <= b1 - b0 <= limit):
            continue
        wrong = " ".join(asr[a0:a1])
        right = " ".join(epub[b0:b1])
        if wrong == right:
            continue
        counts[(wrong, right)] = counts.get((wrong, right), 0) + 1
        surface = " ".join(s for s in epub_surface[b0:b1])
        bucket = surfaces.setdefault((wrong, right), {})
        bucket[surface] = bucket.get(surface, 0) + 1

    safe: List[Tuple[str, str, int]] = []
    review: List[Tuple[str, str, int]] = []
    for (wrong, right), count in counts.items():
        best_surface = max(surfaces[(wrong, right)].items(), key=lambda kv: kv[1])[0]
        entry = (wrong, best_surface, count)
        # Only single-token wrong sides can be vouched for against the book's
        # vocabulary. Multi-word pairs are almost always alignment noise or
        # style ("with a" -> "were the", "he had" -> "He'd") and would rewrite
        # ordinary prose, so they go to review rather than the auto-apply set.
        if " " not in wrong and wrong not in epub_vocab:
            safe.append(entry)
        else:
            review.append(entry)
    safe.sort(key=lambda e: e[2], reverse=True)
    review.sort(key=lambda e: e[2], reverse=True)
    return safe, review


def apply_correction_map(text: str, mapping: Dict[str, str]) -> str:
    """Whole-token, longest-first substitution of ``wrong -> right``."""
    if not text or not mapping:
        return text
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z0-9'])(?:" + "|".join(re.escape(k) for k in keys)
        + r")(?![A-Za-z0-9'])",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: mapping.get(m.group(0).lower(), m.group(0)), text)


# --------------------------------------------------------------------------
# main alignment
# --------------------------------------------------------------------------

@dataclass
class AlignmentReport:
    book: str
    epub_path: str
    chapters: int
    asr_words: int
    source_words: int
    emitted_words: int
    anchors: int
    audio_only_words: int
    source_only_words: int
    timing_chunks: int
    safe_corrections: List[Tuple[str, str, int]] = field(default_factory=list)
    review_corrections: List[Tuple[str, str, int]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    text: str = ""
    timing: Dict[str, list] = field(default_factory=dict)


def _epub_text(epub_path: str) -> str:
    # Imported lazily: keeps the alignment core usable without the DB/UI import
    # chain that services.import_engine pulls in.
    from services.import_engine import extract_epub_text_by_chapters
    docs = extract_epub_text_by_chapters(str(epub_path))
    return "\n\n".join(text for _title, text in docs)


def _chapter_offset_ms(block: str, asr_index_base: int, asr_times, old_entries
                       ) -> Optional[float]:
    """Estimate how far the ASR clock sits above the chapter's own clock.

    ``transcript.json`` timestamps are absolute book time; ``transcript_timing``
    entries are chapter-relative.  The difference is constant per chapter, so a
    median over the existing entries recovers it robustly.
    """
    if not old_entries:
        return None
    tokens = tokenize(block)
    if not tokens:
        return None
    samples = []
    for entry in old_entries:
        target = entry.get("char_start", 0)
        idx = None
        for k, (_n, _cs, ce, _s) in enumerate(tokens):
            if ce > target:
                idx = k
                break
        global_idx = asr_index_base + (idx if idx is not None else 0)
        if global_idx >= len(asr_times):
            continue
        abs_ms = asr_times[global_idx][0]
        samples.append(abs_ms - float(entry.get("start", 0.0)) * 1000.0)
    if not samples:
        return None
    samples.sort()
    return samples[len(samples) // 2]


def align_book(transcript_path: Path, words_path: Path, timing_path: Path,
               epub_path: str, book_label: str = "") -> AlignmentReport:
    """Align one book's transcript against its source text.

    ``words_path`` is ``transcript.json`` (word timestamps); ``timing_path`` is
    ``transcript_timing.json`` (chunk timestamps consumed by ``timing_sync``).
    """
    transcript_path = Path(transcript_path)
    timing_path = Path(timing_path)

    raw_transcript = transcript_path.read_text(encoding="utf-8")
    blocks = _load_chapter_blocks(raw_transcript)

    asr: List[str] = []
    asr_surface: List[str] = []
    asr_chapter: List[int] = []              # 1-based
    asr_offset: List[Tuple[int, int]] = []   # offsets within the chapter block
    chapter_first_index: Dict[int, int] = {}
    for chapter_num, block in enumerate(blocks, start=1):
        chapter_first_index[chapter_num] = len(asr)
        for norm, cs, ce, surface in tokenize(block):
            asr.append(norm)
            asr_surface.append(surface)
            asr_chapter.append(chapter_num)
            asr_offset.append((cs, ce))

    timed_words = _load_transcript_words(words_path)
    warnings: List[str] = []
    if len(timed_words) != len(asr):
        warnings.append(
            f"transcript.json token count ({len(timed_words)}) != transcript.txt "
            f"token count ({len(asr)}); relying on the first {min(len(timed_words), len(asr))}"
        )
    asr_times = [(w[1], w[2]) for w in timed_words[:len(asr)]]
    if len(asr_times) < len(asr):
        asr_times += [asr_times[-1] if asr_times else (0, 0)] * (len(asr) - len(asr_times))

    source_text = _epub_text(epub_path)
    epub_tokens = tokenize(source_text)
    epub = [t[0] for t in epub_tokens]
    epub_surface = [t[3] for t in epub_tokens]
    epub_offsets = [(t[1], t[2]) for t in epub_tokens]
    epub_vocab = set(epub)

    chain = find_anchor_chain(asr, epub)
    if not chain:
        raise ValueError(
            f"no anchors found between '{transcript_path.name}' and '{epub_path}' "
            "- wrong source file?"
        )
    ops = compute_opcodes(asr, epub, chain)

    a_first, b_first = chain[0]
    a_last = chain[-1][0] + ANCHOR_N
    b_last = chain[-1][1] + ANCHOR_N
    a_first, b_first = _absorb_head(asr, epub, a_first, b_first)
    a_last, b_last = _absorb_tail(asr, epub, a_last, b_last)

    safe_corr, review_corr = build_correction_map(
        asr, asr_surface, epub, epub_surface, ops, epub_vocab,
        region=(a_first, a_last))
    mapping = {wrong: right for wrong, right, _ in safe_corr}

    # --- timings for the source region -------------------------------------
    epub_times: List[Optional[Tuple[int, int]]] = [None] * len(epub)
    for tag, a0, a1, b0, b1 in ops:
        if tag == "equal":
            for k in range(min(a1 - a0, b1 - b0)):
                epub_times[b0 + k] = asr_times[a0 + k]
        elif tag == "replace" and b1 > b0 and a1 > a0:
            t0 = asr_times[a0][0]
            t1 = asr_times[a1 - 1][1]
            span = max(1, b1 - b0)
            for k in range(b1 - b0):
                epub_times[b0 + k] = (
                    t0 + (t1 - t0) * k // span,
                    t0 + (t1 - t0) * (k + 1) // span,
                )
    last: Optional[Tuple[int, int]] = asr_times[0] if asr_times else (0, 0)
    epub_chapter: List[Optional[int]] = [None] * len(epub)
    for j in range(b_first, b_last):
        if epub_times[j] is not None:
            last = epub_times[j]
        elif epub_times[j] is None and last is not None:
            epub_times[j] = last

    # chapter of each source token, from the transcript token it aligns to
    asr_to_epub: Dict[int, int] = {}
    for tag, a0, a1, b0, b1 in ops:
        if tag == "equal":
            for k in range(min(a1 - a0, b1 - b0)):
                asr_to_epub[a0 + k] = b0 + k
        elif tag == "replace" and a1 > a0 and b1 > b0:
            for k in range(a1 - a0):
                asr_to_epub[a0 + k] = b0 + min(b1 - b0 - 1, (b1 - b0) * k // (a1 - a0))
    for a_idx, b_idx in asr_to_epub.items():
        if b_first <= b_idx < b_last:
            epub_chapter[b_idx] = asr_chapter[a_idx]
    running = asr_chapter[0]
    for j in range(b_first, b_last):
        if epub_chapter[j] is not None:
            running = epub_chapter[j]
        else:
            epub_chapter[j] = running

    # --- emit per-chapter text + per-token times ---------------------------
    chapter_text: List[List[str]] = [[] for _ in blocks]
    chapter_tok_ms: List[List[Tuple[int, int]]] = [[] for _ in blocks]

    def emit_asr_only(i0: int, i1: int) -> None:
        """Emit a transcript-only span, split at chapter boundaries."""
        k = i0
        while k < i1:
            chapter = asr_chapter[k]
            end = k
            while end < i1 and asr_chapter[end] == chapter:
                end += 1
            src = blocks[chapter - 1][asr_offset[k][0]:asr_offset[end - 1][1]]
            chapter_text[chapter - 1].append(apply_correction_map(src, mapping))
            chapter_tok_ms[chapter - 1].extend(asr_times[k:end])
            k = end

    # audio-only head
    if a_first > 0:
        emit_asr_only(0, a_first)

    # aligned source middle, cut at chapter boundaries
    start = b_first
    while start < b_last:
        chap = epub_chapter[start]
        end = start
        while end < b_last and epub_chapter[end] == chap:
            end += 1
        chapter_text[chap - 1].append(
            source_text[epub_offsets[start][0]:epub_offsets[end - 1][1]])
        chapter_tok_ms[chap - 1].extend(epub_times[start:end])
        start = end

    # audio-only tail
    if a_last < len(asr):
        emit_asr_only(a_last, len(asr))

    # --- old timing entries, for the per-chapter clock offset --------------
    old_entries_by_chapter: Dict[int, list] = {}
    if timing_path.exists():
        try:
            old = json.loads(timing_path.read_text(encoding="utf-8")).get("chapters", {})
            for key, entries in old.items():
                old_entries_by_chapter[int(key)] = entries
        except Exception as exc:  # noqa: BLE001 - report, don't fail the run
            warnings.append(f"could not read {timing_path.name}: {exc}")

    # --- rebuild text + transcript_timing.json -----------------------------
    text_parts: List[str] = []
    timing_out: Dict[str, list] = {}
    total_chunks = 0
    for chapter_num, block in enumerate(blocks, start=1):
        buffer = " ".join(chapter_text[chapter_num - 1]).strip()
        tokens = tokenize(buffer)
        times = chapter_tok_ms[chapter_num - 1]
        if len(tokens) != len(times):
            if times:
                t0 = min(t[0] for t in times)
                t1 = max(t[1] for t in times)
            else:
                t0 = t1 = 0
            n = len(tokens) or 1
            times = [(t0 + (t1 - t0) * k // n, t0 + (t1 - t0) * (k + 1) // n)
                     for k in range(len(tokens))]
        offset_ms = _chapter_offset_ms(
            block, chapter_first_index.get(chapter_num, 0), asr_times,
            old_entries_by_chapter.get(chapter_num))
        if offset_ms is None:
            offset_ms = float(times[0][0]) if times else 0.0
        chunks: List[dict] = []
        for i in range(0, len(tokens), TIMING_CHUNK_WORDS):
            j = min(i + TIMING_CHUNK_WORDS, len(tokens))
            t0 = min(times[k][0] for k in range(i, j))
            t1 = max(times[k][1] for k in range(i, j))
            chunks.append({
                "start": round((t0 - offset_ms) / 1000.0, 2),
                "end": round((t1 - offset_ms) / 1000.0, 2),
                "char_start": tokens[i][1],
                # extend to the next chunk's start so ranges tile the text
                "char_end": tokens[j][1] if j < len(tokens) else tokens[j - 1][2],
            })
        timing_out[str(chapter_num)] = chunks
        total_chunks += len(chunks)
        text_parts.append("==CHAPTER==\n\n")
        text_parts.append(buffer)
        text_parts.append("\n\n")

    new_text = "".join(text_parts)
    emitted_words = len(tokenize(new_text))
    audio_only = a_first + (len(asr) - a_last)
    source_only = len(epub) - (b_last - b_first)

    return AlignmentReport(
        book=book_label,
        epub_path=str(epub_path),
        chapters=len(blocks),
        asr_words=len(asr),
        source_words=len(epub),
        emitted_words=emitted_words,
        anchors=len(chain),
        audio_only_words=audio_only,
        source_only_words=source_only,
        timing_chunks=total_chunks,
        safe_corrections=safe_corr,
        review_corrections=review_corr,
        warnings=warnings,
        text=new_text,
        timing={"chapters": timing_out},
    )


def write_alignment(report: AlignmentReport, book_dir: Path, commit: bool = False,
                    backup_name: str = "transcript.asr.txt",
                    timing_backup_name: str = "transcript_timing.asr.json"
                    ) -> Dict[str, str]:
    """Persist an alignment.  Dry-run writes ``*.aligned`` siblings; ``commit``
    overwrites ``transcript.txt`` / ``transcript_timing.json`` after backing the
    ASR originals up once.  ``corrections.csv`` is always written."""
    book_dir = Path(book_dir)
    book_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    corrections_path = book_dir / "corrections.csv"
    with open(corrections_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["wrong", "right", "count", "safe"])
        for wrong, right, count in report.safe_corrections:
            writer.writerow([wrong, right, count, "yes"])
        for wrong, right, count in report.review_corrections:
            writer.writerow([wrong, right, count, "no"])
    written["corrections"] = str(corrections_path)

    if commit:
        transcript_path = book_dir / "transcript.txt"
        timing_path = book_dir / "transcript_timing.json"
        transcript_backup = book_dir / backup_name
        timing_backup = book_dir / timing_backup_name
        # Back the ASR originals up once, so the cleanup is reversible.
        if transcript_path.exists() and not transcript_backup.exists():
            transcript_backup.write_text(transcript_path.read_text(encoding="utf-8"),
                                         encoding="utf-8")
            written["backup"] = str(transcript_backup)
        if timing_path.exists() and not timing_backup.exists():
            timing_backup.write_text(timing_path.read_text(encoding="utf-8"),
                                     encoding="utf-8")
            written["timing_backup"] = str(timing_backup)
        transcript_path.write_text(report.text, encoding="utf-8")
        timing_path.write_text(json.dumps(report.timing, indent=4), encoding="utf-8")
        written["transcript"] = str(transcript_path)
        written["timing"] = str(timing_path)
    else:
        (book_dir / "transcript.aligned.txt").write_text(report.text, encoding="utf-8")
        (book_dir / "transcript_timing.aligned.json").write_text(
            json.dumps(report.timing, indent=4), encoding="utf-8")
        written["transcript"] = str(book_dir / "transcript.aligned.txt")
        written["timing"] = str(book_dir / "transcript_timing.aligned.json")
    return written


# --------------------------------------------------------------------------
# source-text discovery
# --------------------------------------------------------------------------

def discover_source_epubs(dirs: Sequence[Path]) -> List[Path]:
    found: List[Path] = []
    for d in dirs:
        d = Path(d)
        if d.is_dir():
            found.extend(sorted(d.glob("*.epub")))
    return found


def _discriminating_tokens(book_name: str,
                           candidate_token_sets: Sequence[set]) -> set:
    """Book-name tokens that can actually tell the candidates apart.

    Tokens present in most candidates (``the``, the author/series name) carry no
    signal and are dropped, so they can't make every volume look like a match.
    Falls back to all tokens when the filter would leave nothing (e.g. a single
    candidate, where every shared token looks "common").
    """
    book_tokens = {t for t in _WORD.findall(book_name.lower()) if len(t) > 2}
    if not book_tokens or not candidate_token_sets:
        return book_tokens
    counts: Dict[str, int] = {}
    for tokens in candidate_token_sets:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    limit = len(candidate_token_sets) * 0.5
    useful = {t for t in book_tokens if counts.get(t, 0) <= limit}
    return useful or book_tokens


def rank_source_epubs(book_name: str,
                      candidates: Sequence[Path]) -> List[Tuple[Path, float]]:
    """Candidate source files for ``book_name``, best match first.

    Every entry with any title overlap is returned so the UI can surface
    conflicts (e.g. two editions of the same volume) instead of silently
    picking one.
    """
    candidates = [Path(p) for p in candidates]
    candidate_token_sets = [set(_WORD.findall(p.name.lower())) for p in candidates]
    book_tokens = _discriminating_tokens(book_name, candidate_token_sets)
    book_index = re.match(r"\s*(\d+)", book_name)

    scored: List[Tuple[Path, float]] = []
    for path, tokens in zip(candidates, candidate_token_sets):
        title = len(book_tokens & tokens) / max(1, len(book_tokens))
        if title <= 0:
            continue
        score = title
        if book_index and re.search(r"\[[^\]]*?" + re.escape(book_index.group(1)) + r"\]",
                                    path.name.lower()):
            score += 1.0
        if "retail" in path.name.lower():
            score += 0.1
        scored.append((path, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def match_source_epub(book_name: str, candidates: Sequence[Path]) -> Optional[Path]:
    """Best single match for ``book_name``, or ``None``.

    A numeric series index boosts an otherwise-valid title match but never
    stands alone, so a book with no source returns ``None`` rather than
    grabbing the neighbouring volume.
    """
    ranked = rank_source_epubs(book_name, candidates)
    return ranked[0][0] if ranked else None
