"""Per-book record of which pipeline built a book's coordinates.

``prompts.csv`` rows are keyed by ``(chapter, shot)``, and that mapping is only
stable while the chunking that produced it is unchanged. A book generated with
the linear paragraph chunker and later resumed with a scene map would append
rows in one coordinate system to rows in another, silently - every later stage
would then read the wrong text for a shot.

This manifest is the record that makes the change detectable. It is deliberately
additive: a book with ``prompts.csv`` and no manifest predates the manifest, is
treated as legacy, and may only be extended by the linear chunker. That is what
keeps every already-generated project renderable without re-running anything.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

MANIFEST_FILENAME = "pipeline.json"


def manifest_path(book_dir) -> Path:
    return Path(book_dir) / MANIFEST_FILENAME


def read(book_dir) -> Dict:
    """The manifest, or ``{}`` when absent or unreadable."""
    path = manifest_path(book_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write(book_dir, **values) -> Path:
    """Merge values into the manifest and return its path."""
    path = manifest_path(book_dir)
    data = read(book_dir)
    data.update(values)
    data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def chunking(book_dir) -> str:
    """``"scene_map"``, ``"linear"``, or ``""`` when the manifest predates this."""
    return str(read(book_dir).get("chunking") or "").strip()


def record_chunking(book_dir, source: str, chunk_words: int,
                    seg_version: int = 0, transcript_sha: str = "",
                    scene_map_sha: str = "") -> Path:
    """Record the chunking that owns this book from now on.

    Only called after :func:`check` has allowed the book, so overwriting a
    manifest is safe: either it agreed, or the book has no prompt rows yet.
    """
    return write(
        book_dir,
        chunking=source,
        chunk_words=int(chunk_words),
        seg_version=int(seg_version),
        transcript_sha=transcript_sha,
        scene_map_sha=scene_map_sha,
    )


def check(book_dir, source: str, chunk_words: int, transcript_sha: str = "",
          scene_map_sha: str = "", prompts_exist: bool = True) -> Tuple[bool, str]:
    """May rows be appended to this book's ``prompts.csv`` with this chunking?

    Returns ``(allowed, reason)``. ``reason`` is empty when allowed and a
    sentence the user can act on when not.
    """
    if not prompts_exist:
        return True, ""

    data = read(book_dir)
    recorded = str(data.get("chunking") or "").strip()
    if not recorded:
        if source == "linear":
            # A book generated before the manifest existed. The linear chunker
            # reproduces its coordinates, so appending is safe; anything else
            # would rewrite what each shot means.
            return True, ""
        return False, (
            "this book's prompts.csv predates the scene map, so its shots were "
            "numbered by the linear chunker; rebuild the book before generating "
            "prompts from a scene map.")

    if recorded != source:
        return False, (
            f"this book's prompts.csv was generated with '{recorded}' chunking, "
            f"but the book now resolves to '{source}'; rebuild it rather than "
            f"mixing two coordinate systems.")

    recorded_words = data.get("chunk_words")
    if recorded_words is not None and int(recorded_words) != int(chunk_words):
        return False, (
            f"this book's prompts.csv was generated at {recorded_words} words per "
            f"shot, but the current setting is {chunk_words}; change the setting "
            f"back or rebuild the book.")

    recorded_sha = str(data.get("transcript_sha") or "")
    if transcript_sha and recorded_sha and recorded_sha != transcript_sha:
        return False, (
            "the transcript has changed since this book's prompts were generated; "
            "rebuild the book so every shot maps back to the text it describes.")

    recorded_map = str(data.get("scene_map_sha") or "")
    if (source == "scene_map" and scene_map_sha and recorded_map
            and recorded_map != scene_map_sha):
        return False, (
            "the scene map has been rebuilt since this book's prompts were "
            "generated; rebuild the book so its shots follow the new scenes.")

    return True, ""
