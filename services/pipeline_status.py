"""Per-book pipeline state for the dashboard table.

The dashboard used to render one card per phase and pick between them on
``Project.status``. That cannot express what a project actually is: a project has
N books, each with its own prompts, plan and images, so a single status word is
always a rollup and the cards were wrong most of the time (the analysis controls
disappeared the moment the project reached "Prompts Created", which is exactly
when a re-run is most useful).

This module answers the per-book question instead: for each stage, does the
artifact exist, is it current, and is it half-finished? It is deliberately
side-effect free and cheap enough to call on every poll - the expensive inputs
(the transcript digest) are cached against file modification times, the same
trick ``sync_engine.get_book_stats_cached`` uses.

The scene map and the cast stages are gone: a book is chunked directly from its
transcript now. ``prompts`` and ``images`` are the stages; the shot plan is not a
stage, it is planned inline by prompt generation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from services import pipeline_manifest, scene_map

# Measured on a ~300k-word book (``.docs/pipeline-brief.md``: prompts ~30 min
# including the shot plan). Word count is the only book-agnostic way to scale it,
# so the estimate is proportional and floored rather than pretending to be exact.
_REFERENCE_WORDS = 300_000
_MINUTES_AT_REFERENCE = {"prompts": 30.0}

STAGE_LABELS = {"prompts": "Prompts", "images": "Images"}


def _state(state: str, detail: str = "", **extra) -> Dict:
    row = {"state": state, "detail": detail}
    row.update(extra)
    return row


def _count_rows(path: Path) -> int:
    """Data rows in a pipe-delimited csv, without parsing it."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return max(0, sum(1 for line in fh if line.strip()) - 1)
    except Exception:
        return 0


def _count_images(book_dir: Path) -> int:
    for folder in (book_dir / "images", book_dir):
        try:
            if folder.is_dir():
                n = sum(1 for f in folder.iterdir()
                        if f.suffix.lower() in (".png", ".webp"))
                if n:
                    return n
        except Exception:
            pass
    return 0


def _signature(book_dir: Path, **extra) -> str:
    """Cheap fingerprint: any change to an input or an artifact invalidates."""
    parts = []
    for name in ("transcript.txt", "prompts.csv", "shot_plan.csv",
                 "pipeline.json", "plan.partial.jsonl"):
        path = book_dir / name
        try:
            if path.exists():
                stat = path.stat()
                parts.append(f"{name}:{int(stat.st_mtime)}:{stat.st_size}")
        except Exception:
            pass
    parts.append(f"img:{_count_images(book_dir)}")
    for key, value in sorted(extra.items()):
        parts.append(f"{key}:{value}")
    return "|".join(parts)


def snapshot(book_dir, *, chunk_words: int = 350, expected_prompts: int = 0,
             cached: bool = True) -> Dict:
    """Everything the dashboard needs to draw one book's row.

    ``expected_prompts`` is the book's ``Book.total_images`` when known - it comes
    from the caller so this module holds no session.
    """
    book_dir = Path(book_dir)
    signature = _signature(book_dir, chunk_words=int(chunk_words),
                           expected=int(expected_prompts))
    if cached:
        from ui import state
        entry = state._pipeline_cache.get(str(book_dir))
        if entry and entry[0] == signature:
            return entry[1]

    row = _snapshot_uncached(book_dir, chunk_words=chunk_words,
                             expected_prompts=expected_prompts)
    if cached:
        from ui import state
        state._pipeline_cache[str(book_dir)] = (signature, row)
    return row


def _snapshot_uncached(book_dir: Path, *, chunk_words: int,
                       expected_prompts: int) -> Dict:
    transcript_path = book_dir / scene_map.TRANSCRIPT_FILENAME
    prompts_path = book_dir / "prompts.csv"

    text, words = "", 0
    if transcript_path.exists():
        try:
            text = transcript_path.read_text(encoding="utf-8", errors="ignore")
            words = len(text.split())
        except Exception:
            pass
    transcript_sha = scene_map.transcript_digest(text) if text else ""

    # --- transcript -------------------------------------------------------
    transcript = (_state("ok", f"{words:,} words") if text
                  else _state("missing", "no transcript"))

    # --- prompts ----------------------------------------------------------
    rows = _count_rows(prompts_path) if prompts_path.exists() else 0
    # The shot plan is the coordinate set the prompts are supposed to cover: one
    # row per chunk, written by the same chunker at the same size, and the guard
    # below refuses a plan and a prompts.csv that disagree. So a prompts.csv
    # shorter than its own plan is missing scenes. ``Book.total_images`` cannot
    # answer this - the progress counter is recomputed from the rows that are
    # present, so deleting a broken row silently lowers the expectation with it.
    plan_rows = _count_rows(book_dir / "shot_plan.csv")
    chunking = pipeline_manifest.chunking(book_dir)
    version = scene_map.chunker_version(book_dir)
    blocked = ""
    if rows:
        allowed, reason = pipeline_manifest.check(
            book_dir, chunking or "linear", chunk_words,
            transcript_sha=transcript_sha, prompts_exist=True,
            chunker_version=version)
        if not allowed:
            blocked = reason
    if rows and blocked:
        # The guard's sentence is precise but long; the cell needs a two-word
        # version of it, the tooltip keeps the whole thing. Every one of these is
        # "the rows on disk belong to different text or a different shot grid",
        # so the cell reads as out of date, not as a failure: the images already
        # rendered from these prompts are still the right images.
        if "packer is v" in blocked:
            short = "chunker changed"
        elif "scene map has been rebuilt" in blocked:
            short = "map changed"
        elif "generated at" in blocked and "words per shot" in blocked:
            short = "size changed"
        elif "predates the scene map" in blocked:
            short = "legacy"
        elif "transcript has changed" in blocked:
            short = "text changed"
        else:
            short = "out of date"
        prompts = _state("stale", blocked, rows=rows, short=short)
    elif rows:
        # ``expected_prompts`` is a chunk count recorded by a previous run, and a
        # re-chunk can leave it one or two off the real total. The plan's row count
        # is the exact coordinate set, so it is preferred; only a larger
        # expectation than the rows present means scenes are still missing, and
        # "720 of 720" must not render as work in progress.
        expected = max(plan_rows, expected_prompts)
        total = expected if expected > rows else 0
        prompts = (_state("partial", f"{rows} of {total}", rows=rows, total=total)
                   if total else _state("ok", f"{rows} prompts", rows=rows))
    else:
        prompts = _state("missing", "not built", rows=0)

    # --- images -----------------------------------------------------------
    images = _count_images(book_dir)
    prompts_total = max(rows, plan_rows, expected_prompts)
    if not rows:
        imgs = _state("blocked", "needs prompts", done=images, expected=prompts_total)
    elif prompts_total and images >= prompts_total:
        imgs = _state("ok", f"{images} images", done=images, expected=prompts_total)
    elif images:
        imgs = _state("partial", f"{images} of {prompts_total}", done=images,
                      expected=prompts_total)
    else:
        imgs = _state("missing", "not rendered", done=0, expected=prompts_total)

    return {
        "transcript": transcript,
        "prompts": prompts,
        "images": imgs,
        "words": words,
        "chunking": chunking,
        "chunker_version": version,
        "transcript_sha": transcript_sha,
        "path": str(book_dir),
    }


def estimate_minutes(stage: str, words: int, *, remaining: float = 1.0) -> float:
    """Rough per-book cost for one stage, scaled by word count.

    An estimate exists so the run bar can say "~2h 10m" instead of nothing; it
    is not a promise, and the UI always renders it with a ``~``.
    """
    base = _MINUTES_AT_REFERENCE.get(stage, 0.0)
    if not base:
        return 0.0
    scaled = base * (max(1, int(words or 0)) / _REFERENCE_WORDS)
    return max(1.0, scaled * max(0.0, min(1.0, remaining)))


def needs_work(row: Dict, stage: str) -> bool:
    """Whether ``stage`` would do anything for this book.

    ``ok`` and ``off`` are the only states that need no work; ``blocked`` and
    ``missing`` do, which is what makes the run bar's default selection honest
    rather than "everything".
    """
    state = (row.get(stage) or {}).get("state")
    return state not in ("ok", "off")


def can_run(row: Dict, stage: str) -> bool:
    """Whether ``stage`` *may* run now, regardless of whether it needs to."""
    transcript_ok = (row.get("transcript") or {}).get("state") == "ok"
    if stage == "prompts":
        return transcript_ok
    return False


def plan(books, *, chunk_words: int = 350,
         stages=("prompts",), force: bool = False,
         expected_by_id: Optional[Dict[int, int]] = None):
    """``(plan, rows)`` for a set of books.

    ``plan`` maps each stage to the book ids that need it *now*; ``rows`` is the
    snapshot per book id, so a caller does not snapshot twice. The dashboard and
    the run engine both use this, which is what keeps "what the button says" and
    "what the run does" the same computation.
    """
    expected_by_id = expected_by_id or {}
    rows: Dict[int, Dict] = {}
    out: Dict[str, list] = {}
    for book in books:
        book_id = book[0]
        book_dir = book[2]
        rows[book_id] = snapshot(
            book_dir, chunk_words=chunk_words,
            expected_prompts=expected_by_id.get(book_id, 0))
    for stage in stages:
        targets = [book[0] for book in books
                   if can_run(rows[book[0]], stage)
                   and (force or needs_work(rows[book[0]], stage))]
        if targets:
            out[stage] = targets
    return out, rows


def clear_cache() -> None:
    from ui import state
    state._pipeline_cache.clear()
