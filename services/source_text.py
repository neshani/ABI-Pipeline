"""Per-project source-text (EPUB) configuration and transcript cleanup.

A project can be pointed at one or more folders of EPUBs.  The app matches each
volume to a source file, shows the pairing, and (on request) rewrites the
volume's transcript from the clean book text while carrying the ASR word timings
across.  This is a *repair* step: it is never a required pipeline stage, and a
volume with no source is simply skipped.

Storage is a sidecar (``<project>/source_text.json``) rather than a DB column:
``init_db`` only creates missing tables and would never ALTER an existing one,
so a column would need a hand-written migration against the live database.

    {
      "epub_dirs": ["G:\\Books\\...\\ePubs"],
      "attachments": {"01 The Eye of the World": "G:\\...\\The Eye Of The World (Retail).epub"}
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from database.connection import get_setting
from services.aligner import (align_book, discover_source_epubs, rank_source_epubs,
                              write_alignment)

CONFIG_FILENAME = "source_text.json"
ALIGNED_MARKER = ".transcript_aligned"


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def output_base() -> Path:
    return Path(get_setting("output_dir", "./output")).resolve()


def project_dir(project_name: str) -> Path:
    return output_base() / project_name


def book_dir(project_name: str, book_name: str) -> Path:
    return project_dir(project_name) / book_name


def config_path(project_name: str) -> Path:
    return project_dir(project_name) / CONFIG_FILENAME


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def load_source_config(project_name: str) -> Dict:
    path = config_path(project_name)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("epub_dirs", [])
            data.setdefault("attachments", {})
            return data
        except Exception:
            pass
    return {"epub_dirs": [], "attachments": {}}


def save_source_config(project_name: str, config: Dict) -> None:
    path = config_path(project_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def add_epub_dir(project_name: str, directory: str) -> bool:
    directory = str(Path(directory))
    cfg = load_source_config(project_name)
    if directory in cfg["epub_dirs"]:
        return False
    cfg["epub_dirs"].append(directory)
    save_source_config(project_name, cfg)
    return True


def remove_epub_dir(project_name: str, directory: str) -> None:
    cfg = load_source_config(project_name)
    cfg["epub_dirs"] = [d for d in cfg["epub_dirs"] if d != directory]
    save_source_config(project_name, cfg)


def set_attachment(project_name: str, book_name: str, epub_path: Optional[str]) -> None:
    cfg = load_source_config(project_name)
    if epub_path:
        cfg["attachments"][book_name] = str(epub_path)
    else:
        cfg["attachments"].pop(book_name, None)
    save_source_config(project_name, cfg)


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def resolve_project_sources(project_name: str, book_names: Sequence[str]
                            ) -> Dict[str, Dict]:
    """``{book_name: {chosen, candidates, origin}}``.

    ``chosen`` honours a saved per-book attachment first, then the best
    filename match.  ``candidates`` is the full ranked list so the UI can offer
    a choice when a volume has more than one plausible source.
    """
    cfg = load_source_config(project_name)
    available = discover_source_epubs([Path(d) for d in cfg.get("epub_dirs", [])])
    resolved: Dict[str, Dict] = {}
    for name in book_names:
        candidates: List[Tuple[Path, float]] = rank_source_epubs(name, available)
        attachment = cfg["attachments"].get(name)
        chosen: Optional[Path] = None
        origin = "none"
        if attachment and Path(attachment).exists():
            chosen, origin = Path(attachment), "attachment"
            if all(Path(p) != Path(attachment) for p, _ in candidates):
                candidates = [(Path(attachment), float("inf"))] + candidates
        elif candidates:
            chosen, origin = candidates[0][0], "auto"
        resolved[name] = {"chosen": chosen, "candidates": candidates, "origin": origin}
    return resolved


def is_cleaned(project_name: str, book_name: str) -> bool:
    return (book_dir(project_name, book_name) / ALIGNED_MARKER).exists()


def can_revert(project_name: str, book_name: str) -> bool:
    return (book_dir(project_name, book_name) / "transcript.asr.txt").exists()


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def clean_book(project_name: str, book_name: str, epub_path: str) -> Dict:
    """Align and commit one volume.  Returns report stats; raises on failure.

    Re-runs are idempotent: when the ASR originals exist (``transcript.asr.txt``
    / ``transcript_timing.asr.json``, written on the first cleanup) the
    alignment is taken from *those*, not from the already-cleaned
    ``transcript.txt``. That keeps ``transcript.json`` (ASR word timings) in
    sync with what we align, and means a fixed correction map repairs the
    audio-only spans on a re-run instead of leaving baked-in damage.
    """
    directory = book_dir(project_name, book_name)
    transcript = directory / "transcript.txt"
    words = directory / "transcript.json"
    timing = directory / "transcript_timing.json"
    for path, label in ((transcript, "transcript.txt"),
                        (words, "transcript.json"),
                        (timing, "transcript_timing.json")):
        if not path.exists():
            raise FileNotFoundError(f"{book_name}: missing {label}")
    if not Path(epub_path).exists():
        raise FileNotFoundError(f"{book_name}: source not found: {epub_path}")
    if (directory / "prompts.csv").exists():
        print(f"[Align] WARNING: {book_name} already has prompts.csv; "
              "chunk boundaries shift, so re-run prompt generation afterwards.")

    asr_transcript = directory / "transcript.asr.txt"
    asr_timing = directory / "transcript_timing.asr.json"
    source_transcript = asr_transcript if asr_transcript.exists() else transcript
    source_timing = asr_timing if (asr_transcript.exists() and asr_timing.exists()) else timing

    report = align_book(source_transcript, words, source_timing, str(epub_path),
                        book_label=book_name)
    written = write_alignment(report, directory, commit=True)
    (directory / ALIGNED_MARKER).touch()
    return {
        "chapters": report.chapters,
        "emitted_words": report.emitted_words,
        "audio_only_words": report.audio_only_words,
        "source_only_words": report.source_only_words,
        "timing_chunks": report.timing_chunks,
        "safe_corrections": len(report.safe_corrections),
        "review_corrections": len(report.review_corrections),
        "warnings": report.warnings,
        "files": written,
    }


def revert_book(project_name: str, book_name: str) -> bool:
    """Restore the ASR transcript + timing from the cleanup backups."""
    directory = book_dir(project_name, book_name)
    transcript_backup = directory / "transcript.asr.txt"
    timing_backup = directory / "transcript_timing.asr.json"
    if not transcript_backup.exists():
        return False
    (directory / "transcript.txt").write_text(
        transcript_backup.read_text(encoding="utf-8"), encoding="utf-8")
    if timing_backup.exists():
        (directory / "transcript_timing.json").write_text(
            timing_backup.read_text(encoding="utf-8"), encoding="utf-8")
    marker = directory / ALIGNED_MARKER
    if marker.exists():
        marker.unlink()
    return True
