"""Series-wide spelling repair from the aligner's correction maps.

The EPUB cleanup writes, per volume, a ``corrections.csv`` of ``wrong,right``
pairs discovered by aligning the ASR transcript against the real book text.
Aggregating those across a project yields a reusable spelling map that repairs
volumes which have **no** source text (a companion novella, an unaligned book),
and it works for any series - it is derived from the app's own alignment output,
never from an external glossary.

Application is deliberately conservative:

* whole-token, longest-first, case-insensitive substitution (so ``Mat`` can't
  clobber ``material``),
* only pairs flagged ``safe`` in every book they were seen in,
* a minimum-occurrence threshold (one-off pairs are more likely alignment
  noise than real misspellings),

and ``transcript_timing.json`` character offsets are remapped so quote
timestamps stay accurate after the text lengths change.
"""
from __future__ import annotations

import csv
import json
import re
from bisect import bisect_right
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from database.connection import get_setting

CORRECTIONS_FILENAME = "corrections.csv"
TRANSCRIPT_BACKUP = "transcript.prespell.txt"
TIMING_BACKUP = "transcript_timing.prespell.json"

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def _base() -> Path:
    return Path(get_setting("output_dir", "./output")).resolve()


def book_dir(project_name: str, book_name: str) -> Path:
    return _base() / project_name / book_name


def corrections_path(project_name: str, book_name: str) -> Path:
    return book_dir(project_name, book_name) / CORRECTIONS_FILENAME


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def load_book_corrections(project_name: str, book_name: str) -> List[Dict]:
    path = corrections_path(project_name, book_name)
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            wrong = (row.get("wrong") or "").strip()
            right = (row.get("right") or "").strip()
            if not wrong or not right or wrong == right:
                continue
            try:
                count = int(row.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
            safe = (row.get("safe") or "").strip().lower() in ("yes", "true", "1")
            rows.append({"wrong": wrong, "right": right, "count": count, "safe": safe})
    return rows


def project_clean_vocab(project_name: str, book_names: Sequence[str]) -> set:
    """Every token appearing in an already-aligned (clean) volume.

    The per-book ``safe`` flag is computed against one book's vocabulary, so a
    pair can look safe in the volume it was found in yet still collide with a
    real word elsewhere in the series. This project-wide union closes that hole
    for the auto-apply path (which has no preview): if the "wrong" spelling is
    a legitimate word anywhere in the series' clean text, the pair is dropped.
    """
    vocab: set = set()
    for book in book_names:
        directory = book_dir(project_name, book)
        if not (directory / ".transcript_aligned").exists():
            continue
        path = directory / "transcript.txt"
        if not path.exists():
            continue
        for token in _TOKEN_RE.findall(path.read_text(encoding="utf-8").lower()):
            token = token.strip("'")
            if token:
                vocab.add(token)
    return vocab


def aggregate_corrections(project_name: str, book_names: Sequence[str],
                          min_count: int = 2, safe_only: bool = True,
                          clean_vocab: Optional[set] = None) -> List[Dict]:
    """Merge every book's correction map into one ranked list.

    A pair is only kept when it is ``safe`` in *every* book it appeared in, its
    occurrences are summed across the series, and (when ``clean_vocab`` is
    given) its "wrong" spelling is not a real word anywhere in the series.
    """
    merged: Dict[str, Dict] = {}
    for book in book_names:
        for row in load_book_corrections(project_name, book):
            if safe_only and not row["safe"]:
                continue
            # Only single-token repairs belong in a spelling tool. Multi-word
            # pairs are alignment noise or style rewrites and would alter
            # ordinary prose ("with a" -> "were the").
            if " " in row["wrong"]:
                continue
            if clean_vocab is not None and row["wrong"] in clean_vocab:
                continue
            entry = merged.get(row["wrong"])
            if entry is None:
                entry = merged[row["wrong"]] = {
                    "wrong": row["wrong"], "right": row["right"],
                    "count": 0, "safe": True, "books": set(),
                }
            entry["count"] += row["count"]
            entry["safe"] = entry["safe"] and row["safe"]
            entry["books"].add(book)
            if row["count"] > entry.get("_top", 0):
                entry["right"], entry["_top"] = row["right"], row["count"]
    out = [e for e in merged.values() if e["count"] >= min_count]
    out.sort(key=lambda e: e["count"], reverse=True)
    for entry in out:
        entry["books"] = sorted(entry["books"])
        entry.pop("_top", None)
    return out


def books_without_source(project_name: str, book_names: Sequence[str]) -> List[str]:
    """Volumes that were never aligned to a source text (the ones that need this)."""
    return [b for b in book_names
            if not (book_dir(project_name, b) / ".transcript_aligned").exists()
            and (book_dir(project_name, b) / "transcript.txt").exists()]


# --------------------------------------------------------------------------
# application (with offset remapping)
# --------------------------------------------------------------------------

def _build_pattern(mapping: Dict[str, str]) -> "re.Pattern":
    keys = sorted(mapping, key=len, reverse=True)
    return re.compile(
        r"(?<![A-Za-z0-9'])(?:" + "|".join(re.escape(k) for k in keys)
        + r")(?![A-Za-z0-9'])",
        re.IGNORECASE,
    )


def apply_mapping(text: str, mapping: Dict[str, str]
                  ) -> Tuple[str, Callable[[int], int], int]:
    """Whole-token substitution returning ``(new_text, old->new offset map, hits)``."""
    pattern = _build_pattern(mapping)
    edits: List[Tuple[int, int]] = []   # (old_end_index, cumulative_delta_after)
    state = {"delta": 0, "hits": 0}

    def repl(match: "re.Match") -> str:
        old = match.group(0)
        new = mapping.get(old.lower(), old)
        if new != old:
            state["delta"] += len(new) - len(old)
            edits.append((match.end(), state["delta"]))
            state["hits"] += 1
        return new

    new_text = pattern.sub(repl, text)
    ends = [e[0] for e in edits]
    deltas = [e[1] for e in edits]

    def map_pos(pos: int) -> int:
        i = bisect_right(ends, pos) - 1
        return pos + (deltas[i] if i >= 0 else 0)

    return new_text, map_pos, state["hits"]


def apply_to_book(project_name: str, book_name: str, mapping: Dict[str, str],
                  dry_run: bool = True) -> Dict:
    """Apply a spelling map to one volume, remapping its timing offsets."""
    directory = book_dir(project_name, book_name)
    transcript_path = directory / "transcript.txt"
    timing_path = directory / "transcript_timing.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"{book_name}: no transcript.txt")

    original = transcript_path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in original.split("==CHAPTER==") if b.strip()]

    timing: Dict[str, list] = {}
    if timing_path.exists():
        try:
            timing = json.loads(timing_path.read_text(encoding="utf-8")).get("chapters", {})
        except Exception:
            timing = {}

    new_blocks: List[str] = []
    new_timing: Dict[str, list] = {}
    hits = 0
    for index, block in enumerate(blocks, start=1):
        new_block, map_pos, n = apply_mapping(block, mapping)
        new_blocks.append(new_block)
        hits += n
        entries = timing.get(str(index)) or []
        if entries:
            remapped = []
            for entry in entries:
                entry = dict(entry)
                entry["char_start"] = max(0, min(map_pos(int(entry.get("char_start", 0))), len(new_block)))
                entry["char_end"] = max(0, min(map_pos(int(entry.get("char_end", 0))), len(new_block)))
                remapped.append(entry)
            new_timing[str(index)] = remapped

    new_text = "".join(f"==CHAPTER==\n\n{block}\n\n" for block in new_blocks)

    if not dry_run:
        _write_with_backup(transcript_path, new_text, TRANSCRIPT_BACKUP)
        if new_timing:
            _write_with_backup(timing_path,
                               json.dumps({"chapters": new_timing}, indent=4),
                               TIMING_BACKUP)

    return {"book": book_name, "hits": hits, "chapters": len(blocks),
            "chars_before": len(original), "chars_after": len(new_text),
            "dry_run": dry_run}


def _write_with_backup(path: Path, content: str, backup_name: str) -> None:
    if path.exists():
        backup = path.parent / backup_name
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(content, encoding="utf-8")


def revert_book(project_name: str, book_name: str) -> bool:
    """Restore a volume's pre-spelling-fix transcript + timing."""
    directory = book_dir(project_name, book_name)
    backup = directory / TRANSCRIPT_BACKUP
    if not backup.exists():
        return False
    (directory / "transcript.txt").write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    timing_backup = directory / TIMING_BACKUP
    if timing_backup.exists():
        (directory / "transcript_timing.json").write_text(
            timing_backup.read_text(encoding="utf-8"), encoding="utf-8")
    return True
