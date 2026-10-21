"""Per-book pipeline state for the dashboard table.

The dashboard used to render one card per phase and pick between them on
``Project.status``. That cannot express what a project actually is: a project has
N books, each with its own scene map, cast, prompt file and images, so a single
status word is always a rollup and the cards were wrong most of the time (the
analysis controls disappeared the moment the project reached "Prompts Created",
which is exactly when a re-analysis is most useful).

This module answers the per-book question instead: for each stage, does the
artifact exist, is it current, and is it half-finished? It is deliberately
side-effect free and cheap enough to call on every poll - the expensive inputs
(the transcript digest) are cached against file modification times, the same
trick ``sync_engine.get_book_stats_cached`` uses, and the DB counts are passed in
by the caller as one aggregate query rather than one per row.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from services import checkpoints, pipeline_manifest, scene_map

# Measured on a ~300k-word book (``.docs/pipeline-brief.md``: scenes ~20 min,
# cast ~31 min - gather 22 + refname 3 + extract 6 - prompts ~30 min including
# the shot plan). Word count is the only book-agnostic way to scale them, so the
# estimate is proportional and floored rather than pretending to be exact.
_REFERENCE_WORDS = 300_000
_MINUTES_AT_REFERENCE = {"map": 20.0, "cast": 31.0, "prompts": 30.0}

STAGE_LABELS = {"map": "Scene map", "cast": "Cast", "prompts": "Prompts",
                "images": "Images"}


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
    for name in ("transcript.txt", "scene_map.json", "gather.json",
                 "extract.json", "prompts.csv", "shot_plan.csv",
                 "pipeline.json", "scene.partial.jsonl", "gather.partial.jsonl",
                 "extract.partial.jsonl", "refname.partial.jsonl",
                 "plan.partial.jsonl"):
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


def snapshot(book_dir, *, chunk_words: int = 350, cast_enabled: bool = True,
             events: int = 0, expected_prompts: int = 0,
             cached: bool = True) -> Dict:
    """Everything the dashboard needs to draw one book's row.

    ``events`` is the book's extracted timeline-event count and
    ``expected_prompts`` its ``Book.total_images`` when known - both come from
    the caller so this module holds no session.
    """
    book_dir = Path(book_dir)
    signature = _signature(book_dir, chunk_words=int(chunk_words),
                           cast=bool(cast_enabled), events=int(events),
                           expected=int(expected_prompts))
    if cached:
        from ui import state
        entry = state._pipeline_cache.get(str(book_dir))
        if entry and entry[0] == signature:
            return entry[1]

    row = _snapshot_uncached(book_dir, chunk_words=chunk_words,
                             cast_enabled=cast_enabled, events=events,
                             expected_prompts=expected_prompts)
    if cached:
        from ui import state
        state._pipeline_cache[str(book_dir)] = (signature, row)
    return row


def _snapshot_uncached(book_dir: Path, *, chunk_words: int,
                       cast_enabled: bool, events: int,
                       expected_prompts: int) -> Dict:
    transcript_path = book_dir / scene_map.TRANSCRIPT_FILENAME
    map_path = scene_map.scene_map_path(book_dir)
    gather_path = book_dir / "gather.json"
    extract_path = book_dir / "extract.json"
    prompts_path = book_dir / "prompts.csv"

    text, words = "", 0
    if transcript_path.exists():
        try:
            text = transcript_path.read_text(encoding="utf-8", errors="ignore")
            words = len(text.split())
        except Exception:
            pass
    transcript_sha = scene_map.transcript_digest(text) if text else ""
    map_sha = ""
    if map_path.exists():
        try:
            map_sha = pipeline_manifest.digest(map_path.read_text(encoding="utf-8"))
        except Exception:
            map_sha = ""

    # --- transcript -------------------------------------------------------
    transcript = (_state("ok", f"{words:,} words") if text
                  else _state("missing", "no transcript"))

    # --- scene map --------------------------------------------------------
    segment_sha = pipeline_manifest.digest({
        "transcript_sha": transcript_sha,
        "window_words": scene_map.configured_window_words(),
        "seg_version": scene_map.SEG_VERSION,
    })
    partial_map = checkpoints.summary(book_dir, "segment", segment_sha)["done"]
    if not text:
        scene = _state("blocked", "needs a transcript")
    elif map_path.exists():
        current, reasons = pipeline_manifest.stage_current(
            book_dir, "segment", version=scene_map.SEG_VERSION,
            inputs={"transcript_sha": transcript_sha,
                    "window_words": scene_map.configured_window_words()})
        record = pipeline_manifest.stage_record(book_dir, "segment")
        if current:
            scene = _state("ok", "")
        elif not record:
            # Built before stage records existed: the artifact may well be fine,
            # but nothing can prove it, and "unknown" must not read as "fine".
            scene = _state("unrecorded",
                           "built before stage records; inputs cannot be verified")
        elif reasons:
            scene = _state("stale", ", ".join(reasons))
        else:
            scene = _state("stale", "inputs changed")
    elif partial_map:
        scene = _state("partial", f"{partial_map} windows done", done=partial_map)
    else:
        scene = _state("missing", "not built")

    # --- cast (optional by design) ---------------------------------------
    gather_partial = checkpoints.summary(
        book_dir, "gather", map_sha)["done"] if map_sha else 0
    gather_sha = ""
    if gather_path.exists():
        try:
            gather_sha = pipeline_manifest.digest(
                (json.loads(gather_path.read_text(encoding="utf-8")) or {})
                .get("quotes", []))
        except Exception:
            gather_sha = ""
    extract_partial = checkpoints.summary(
        book_dir, "extract", gather_sha)["done"] if gather_sha else 0

    if not cast_enabled:
        cast = _state("off", "character descriptions are off", events=events)
    elif not map_path.exists():
        cast = _state("blocked", "needs a scene map", events=events)
    elif extract_path.exists():
        # Lazy import: this module is drawn on every poll and character_analysis
        # is a large import graph. The version and cap are the same inputs
        # ``analysis_stage_status`` compares, so the two never disagree.
        from services import character_analysis as ca
        current, reasons = pipeline_manifest.stage_current(
            book_dir, "extract", version=ca.ANALYSIS_VERSION,
            inputs={"gather_sha": gather_sha,
                    "quote_cap": ca.EXTRACT_QUOTE_CAP})
        if current and events:
            cast = _state("ok", f"{events} events", events=events)
        elif not pipeline_manifest.stage_record(book_dir, "extract"):
            cast = _state("unrecorded",
                          "built before stage records; inputs cannot be verified",
                          events=events)
        elif reasons:
            cast = _state("stale", ", ".join(reasons), events=events)
        else:
            cast = _state("stale", "inputs changed", events=events)
    elif extract_partial:
        cast = _state("partial", f"{extract_partial} characters done",
                     done=extract_partial, events=events)
    elif gather_partial:
        cast = _state("partial", f"{gather_partial} quote windows done",
                     done=gather_partial, events=events)
    else:
        cast = _state("missing", "not built", events=events)

    # --- prompts ----------------------------------------------------------
    rows = _count_rows(prompts_path) if prompts_path.exists() else 0
    chunking = pipeline_manifest.chunking(book_dir)
    blocked = ""
    if rows:
        allowed, reason = pipeline_manifest.check(
            book_dir, chunking or "linear", chunk_words,
            transcript_sha=transcript_sha, scene_map_sha=map_sha,
            prompts_exist=True)
        if not allowed:
            blocked = reason
    if rows and blocked:
        # The guard's sentence is precise but long; the cell needs a two-word
        # version of it, the tooltip keeps the whole thing.
        if "scene map has been rebuilt" in blocked:
            short = "map changed"
        elif "generated at" in blocked and "words per shot" in blocked:
            short = "size changed"
        elif "predates the scene map" in blocked:
            short = "legacy"
        elif "transcript has changed" in blocked:
            short = "text changed"
        else:
            short = "blocked"
        prompts = _state("blocked", blocked, rows=rows, short=short)
    elif rows:
        total = expected_prompts if expected_prompts >= rows else 0
        prompts = (_state("partial", f"{rows} of {total}", rows=rows, total=total)
                   if total else _state("ok", f"{rows} prompts", rows=rows))
    else:
        prompts = _state("missing", "not built", rows=0)

    # --- images -----------------------------------------------------------
    images = _count_images(book_dir)
    prompts_total = (expected_prompts if expected_prompts >= rows else rows)
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
        "map": scene,
        "cast": cast,
        "prompts": prompts,
        "images": imgs,
        "words": words,
        "chunking": chunking,
        "transcript_sha": transcript_sha,
        "map_sha": map_sha,
        "gather_sha": gather_sha,
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
    map_state = (row.get("map") or {}).get("state")
    cast_state = (row.get("cast") or {}).get("state")
    map_usable = map_state in ("ok", "stale", "partial", "unrecorded")
    if stage == "map":
        return (row.get("transcript") or {}).get("state") == "ok"
    if stage == "cast":
        return cast_state != "off" and map_usable
    if stage == "prompts":
        return map_usable
    return False


def plan(books, *, chunk_words: int = 350, cast_enabled: bool = True,
         events_by_id: Optional[Dict[int, int]] = None,
         stages=("map", "cast", "prompts"), force: bool = False,
         expected_by_id: Optional[Dict[int, int]] = None):
    """``(plan, rows)`` for a set of books.

    ``plan`` maps each stage to the book ids that need it *now*; ``rows`` is the
    snapshot per book id, so a caller does not snapshot twice. The dashboard and
    the run engine both use this, which is what keeps "what the button says" and
    "what the run does" the same computation.
    """
    events_by_id = events_by_id or {}
    expected_by_id = expected_by_id or {}
    rows: Dict[int, Dict] = {}
    out: Dict[str, list] = {}
    for book in books:
        book_id = book[0]
        book_dir = book[2]
        rows[book_id] = snapshot(
            book_dir, chunk_words=chunk_words, cast_enabled=cast_enabled,
            events=events_by_id.get(book_id, 0),
            expected_prompts=expected_by_id.get(book_id, 0))
    for stage in stages:
        if stage == "cast" and not cast_enabled:
            continue
        targets = [book[0] for book in books
                   if can_run(rows[book[0]], stage)
                   and (force or needs_work(rows[book[0]], stage))]
        if targets:
            out[stage] = targets
    return out, rows


def clear_cache() -> None:
    from ui import state
    state._pipeline_cache.clear()
