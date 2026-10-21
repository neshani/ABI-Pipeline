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

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

MANIFEST_FILENAME = "pipeline.json"

# The stages whose provenance is recorded under ``stages``. Each entry answers
# "what was this built from", so staleness is computed instead of guessed.
STAGE_KEYS = ("segment", "gather", "refname", "extract", "sex", "plan", "prompt")


def digest(value) -> str:
    """Short stable digest of a string, bytes, or any JSON-serialisable value."""
    if isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray)):
        payload = bytes(value)
    else:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                             default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


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


# --- per-stage provenance ----------------------------------------------------
#
# The chunking guard above answers one question ("may rows be appended?"). This
# block answers the pipeline-wide version of it: for each stage, what was it
# built from, and is that still true? Without it the app can only offer "re-run
# the book", because it cannot tell a stale stage from a current one.

def record_stage(book_dir, stage: str, *, version=None, inputs=None,
                 model: str = "", prompt_sha: str = "") -> Path:
    """Record what one stage was built from, merging into the manifest."""
    if stage not in STAGE_KEYS:
        raise ValueError(f"unknown stage: {stage}")
    data = read(book_dir)
    stages = dict(data.get("stages") or {})
    stages[stage] = {
        "version": version,
        "inputs": dict(inputs or {}),
        "model": model or "",
        "prompt_sha": prompt_sha or "",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return write(book_dir, stages=stages)


def stage_record(book_dir, stage: str) -> Dict:
    """One stage's provenance, or ``{}`` when it has never been recorded."""
    stages = read(book_dir).get("stages") or {}
    record = stages.get(stage)
    return dict(record) if isinstance(record, dict) else {}


def stages_summary(book_dir) -> Dict[str, Dict]:
    """Every recorded stage, for a UI that lists what exists and what it used."""
    stages = read(book_dir).get("stages") or {}
    return {key: dict(value) for key, value in stages.items()
            if isinstance(value, dict)}


def stage_current(book_dir, stage: str, *, version=None, inputs=None,
                  model=None, prompt_sha=None) -> Tuple[bool, List[str]]:
    """``(is_current, reasons)`` for one stage against the values supplied now.

    Only the arguments given are compared, so a caller can check just the input
    it cares about. A stage that was never recorded is never current: "unknown"
    must not read as "fine", which is the mistake the truncation bug taught.
    """
    record = stage_record(book_dir, stage)
    if not record:
        return False, ["never run"]
    reasons: List[str] = []
    if version is not None and record.get("version") != version:
        reasons.append("stage version changed")
    recorded_inputs = record.get("inputs") or {}
    for key, value in (inputs or {}).items():
        if recorded_inputs.get(key) != value:
            reasons.append(f"{key} changed")
    if model is not None and (record.get("model") or "") != (model or ""):
        reasons.append("model changed")
    if prompt_sha is not None and (record.get("prompt_sha") or "") != (prompt_sha or ""):
        reasons.append("prompt changed")
    return (not reasons), reasons


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
