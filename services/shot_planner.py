"""Shot Planner: assign a shot type to every scene of a book.

Runs the shared ``shot_direction`` prompt over windows of chunks and returns
one of ``WIDE`` / ``CHARACTER`` / ``INSERT`` / ``NONE`` per scene.

The plan is stored in its own ``shot_plan.csv`` next to the transcript, not in
``prompts.csv``: ``start_project_prompt_gen`` treats any existing prompts.csv
row as an already-finished scene, so pre-writing rows there would make prompt
generation skip the whole book.

The chunk size used is recorded in the plan, and each scene carries a digest of
its chunk, because ``(chapter, shot)`` only maps back to the same text while
the transcript, the chunk source and the chunk size are unchanged.
``scene_chunk`` re-derives a chunk on demand; ``verify_plan`` reports any scene
whose text has drifted. The plan also records whether its chunks came from a
scene map or the linear chunker, so a book cannot be planned against one
coordinate system and generated against the other.

The plan also records a digest of the Shot Planner prompt that produced it, so
a plan built with one planner variant is never reused after the active variant
is switched (or edited). Without that, a stored plan silently decides a book's
shot choices for every later run.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from database.connection import get_setting
from services import compat
from services import scene_map
from services.prompt_engine import get_llm_response, smart_chunk_text
from services.prompt_library import SHOT_TYPES, planner_text

# Chunks per request. 30 keeps the window well inside a 98k context and keeps
# the reply list short enough for a small model to keep in order.
DEFAULT_WINDOW = 30
DEFAULT_CHUNK_WORDS = 350

PLAN_FILENAME = "shot_plan.csv"
PLAN_COLUMNS = ("chapter", "shot", "shot_type", "chunk_words", "chunk_hash",
                "prompt_hash", "focus", "chunking")

_LINE = re.compile(r"^\s*(\d+)\s*[:.\)\-]\s*\**([A-Za-z]+)\**\s*(?:\|\s*(.*))?")
_ALIASES = {
    "GROUP": "WIDE", "CROWD": "WIDE", "CROWDSHOT": "WIDE",
    "ESTABLISHING": "WIDE", "LANDSCAPE": "WIDE", "SCENERY": "WIDE",
    "DETAIL": "INSERT", "OBJECT": "INSERT", "ITEM": "INSERT",
    "CHARACTERS": "CHARACTER", "PEOPLE": "CHARACTER", "FIGURE": "CHARACTER",
}


# --- parsing and planning --------------------------------------------------

def parse_shot_plan(raw: str, count: int) -> Dict[int, Tuple[str, str]]:
    """``{1-based index: (shot type, focus)}``.

    The planner prompt requires ``<n>: <SHOT> | <focus>`` and the focus names exactly what
    the plate shows - the place, the people, or the object. It used to be parsed away, so
    the object of every INSERT was generated on every call and thrown out.
    """
    found: Dict[int, Tuple[str, str]] = {}
    for line in (raw or "").splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        if not 1 <= idx <= count:
            continue
        label = m.group(2).upper()
        if label not in SHOT_TYPES:
            label = _ALIASES.get(label, "")
            if not label:
                continue
        focus = (m.group(3) or "").strip().strip("*").strip()
        found[idx] = (label, focus)
    return found


def parse_shot_lines(raw: str, count: int) -> Dict[int, str]:
    """``<n>: <SHOT>`` lines into ``{1-based index: shot type}`` (focus discarded)."""
    return {i: label for i, (label, _focus) in parse_shot_plan(raw, count).items()}


# What a consecutive INSERT is demoted to. Kept as a constant because the choice is
# editorial, not mechanical: the second passage in a run of two object-worthy passages is
# usually the one with people in it, so CHARACTER is the likeliest second option.
_INSERT_DEMOTION = "CHARACTER"


def enforce_insert_spacing(labels: List[str]) -> List[str]:
    """The one INSERT rule worth enforcing in code, and only the absolute half of it.

    The prompt says an INSERT is "never in consecutive passages, and no more than once in
    any run of about eight", and the model obeys neither: the first full-book plan had **6
    consecutive INSERTs and 27 pairs within eight** out of 59. This corrects only the
    consecutive case - the one stated as *never*, where a violation is unambiguous. The
    softer "one in about eight" is left to the prompt, because enforcing it would rewrite
    27 more and turn most inserts into CHARACTER shots.
    """
    out = list(labels)
    for i in range(1, len(out)):
        if out[i] == "INSERT" and out[i - 1] == "INSERT":
            out[i] = _INSERT_DEMOTION
    return out


def plan_prompt(chunks: List[str], template: str) -> str:
    """The full shot-direction request for one window of chunks."""
    body = "\n".join(f"{i + 1}: {c}" for i, c in enumerate(chunks))
    return (f"{template.rstrip()}\n\nPASSAGES:\n{body}\n\n"
            f"Answer with {len(chunks)} lines.")


async def plan_chunks(
    chunks: List[str],
    llm_url: str,
    model: str,
    window: int = DEFAULT_WINDOW,
    log: Optional[Callable[[str], None]] = None,
    template: Optional[str] = None,
) -> List[str]:
    """Shot type per chunk, in order. Anything unparsed falls back to NONE."""
    detail = await plan_chunks_detail(chunks, llm_url, model, window, log, template)
    return [label for label, _focus in detail]


async def plan_chunks_detail(
    chunks: List[str],
    llm_url: str,
    model: str,
    window: int = DEFAULT_WINDOW,
    log: Optional[Callable[[str], None]] = None,
    template: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """``[(shot_type, focus)]`` per chunk, in order. Unparsed falls back to ``NONE``.

    ``template`` pins the Shot Planner prompt; when omitted the active variant
    is read live, which is what the Prompt Lab test bench wants.
    """
    template = template if template is not None else planner_text()
    out: List[Tuple[str, str]] = []
    for start in range(0, len(chunks), window):
        window_chunks = chunks[start:start + window]
        if log:
            log(f"[Shot-Plan] classifying scenes {start + 1}-{start + len(window_chunks)}")
        raw = await get_llm_response(
            plan_prompt(window_chunks, template), llm_url, model,
            temperature=0.0,
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
        )
        got = parse_shot_plan(raw, len(window_chunks))
        out.extend(got.get(i + 1, ("NONE", "")) for i in range(len(window_chunks)))
    return out


def chunk_digest(text: str) -> str:
    """Stable short digest of a chunk, whitespace-normalised."""
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:10]


def prompt_digest(text: str) -> str:
    """Stable short digest of a prompt template, whitespace-normalised."""
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:10]


# --- chunking --------------------------------------------------------------

def chunk_book_chapters(text: str, chunk_words: int = DEFAULT_CHUNK_WORDS
                        ) -> List[Tuple[int, List[str]]]:
    """``[(chapter_num, chunks)]`` from the LINEAR chunker.

    Kept as the legacy/fast fallback. Planning reads ``scene_map.book_chunks``
    instead, so the plan and ``prompts.csv`` always share one coordinate system.
    """
    chapters = [c for c in text.replace("\r\n", "\n").split("==CHAPTER==") if c.strip()]
    return [(i + 1, smart_chunk_text(c, chunk_words)) for i, c in enumerate(chapters)]


def book_dir(project_name: str, book_name: str, base: Optional[Path] = None) -> Path:
    root = Path(base) if base is not None else Path(get_setting("output_dir", "./output")).resolve()
    return root / project_name / book_name


def book_transcript_path(project_name: str, book_name: str,
                         base: Optional[Path] = None) -> Path:
    return book_dir(project_name, book_name, base) / "transcript.txt"


def plan_path(project_name: str, book_name: str, base: Optional[Path] = None) -> Path:
    return book_dir(project_name, book_name, base) / PLAN_FILENAME


def read_book_text(project_name: str, book_name: str,
                   base: Optional[Path] = None) -> Optional[str]:
    path = book_transcript_path(project_name, book_name, base)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


# --- whole-book planning ---------------------------------------------------

async def plan_book(
    project_name: str,
    book_name: str,
    llm_url: str,
    model: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    window: int = DEFAULT_WINDOW,
    base: Optional[Path] = None,
    log: Optional[Callable[[str], None]] = None,
    template: Optional[str] = None,
    chunks: Optional[Sequence[scene_map.Chunk]] = None,
) -> List[Tuple[int, int, str, str, str]]:
    """``[(chapter_num, shot_num, shot_type, chunk_hash, focus)]`` for a whole book.

    The chunks come from ``scene_map.book_chunks``, which honors the chunking the
    book is bound to: its scene map, or the linear chunker for a legacy book. That
    is the same list prompt generation numbers, which is what makes
    ``(chapter, shot)`` mean the same text in the plan and in ``prompts.csv``.

    ``focus`` is what the plate shows - the place, the people, or the object - and
    it is the only place an INSERT's subject is ever named.
    ``enforce_insert_spacing`` corrects consecutive INSERTs per chapter, which the
    prompt forbids and the model does anyway.
    """
    if chunks is None:
        chunks, _source = scene_map.book_chunks(
            book_dir(project_name, book_name, base), chunk_words, log)

    plan: List[Tuple[int, int, str, str, str]] = []
    for chapter_num, chapter_chunks in sorted(
            scene_map.chunks_by_chapter(chunks).items()):
        if not chapter_chunks:
            continue
        if log:
            log(f"[Shot-Plan] Chapter {chapter_num}: {len(chapter_chunks)} scenes")
        detail = await plan_chunks_detail(
            [chunk.text for chunk in chapter_chunks], llm_url, model, window, log,
            template)
        labels = enforce_insert_spacing([label for label, _focus in detail])
        for chunk, label, (_raw_label, focus) in zip(chapter_chunks, labels, detail):
            plan.append((chunk.chapter, chunk.shot, label,
                         chunk_digest(chunk.text), focus))
    return plan


# --- plan persistence ------------------------------------------------------

def save_plan(
    project_name: str,
    book_name: str,
    plan: List[Tuple[int, int, str, str, str]],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    base: Optional[Path] = None,
    prompt_hash: str = "",
    chunking: str = "",
) -> Path:
    """Write ``shot_plan.csv``. Self-contained: it records the chunk size used,
    the digest of the planner prompt the shot types came from, and whether the
    chunks came from a scene map or the linear chunker."""
    path = plan_path(project_name, book_name, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(PLAN_COLUMNS)
        for chapter, scene, shot, digest, focus in plan:
            writer.writerow([chapter, scene, shot, chunk_words, digest, prompt_hash,
                             focus, chunking])
    return path


def load_focus_map(
    project_name: str, book_name: str, base: Optional[Path] = None
) -> Dict[Tuple[int, int], str]:
    """``{(chapter, shot): focus}`` from the stored plan.

    The focus names what each plate shows - the place, the people, or the object - and it is
    the only place an INSERT's subject is ever written down. Empty when the plan predates the
    column, so callers must treat "" as "no focus".
    """
    return {(int(r["chapter"]), int(r["shot"])): (r.get("focus") or "")
            for r in load_plan(project_name, book_name, base)
            if r.get("chapter") and r.get("shot")}


def load_plan(
    project_name: str, book_name: str, base: Optional[Path] = None
) -> List[Dict[str, str]]:
    """Rows of ``shot_plan.csv`` as ``{column: value}``; empty list if absent."""
    path = plan_path(project_name, book_name, base)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        reader.fieldnames = compat.normalize_fieldnames(reader.fieldnames)
        return [{k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                for row in reader]


def plan_chunk_words(project_name: str, book_name: str,
                     base: Optional[Path] = None) -> Optional[int]:
    """The chunk size the plan was built with, read back from the first row."""
    rows = load_plan(project_name, book_name, base)
    if not rows or not rows[0].get("chunk_words"):
        return None
    try:
        return int(rows[0]["chunk_words"])
    except ValueError:
        return None


def plan_prompt_hash(project_name: str, book_name: str,
                     base: Optional[Path] = None) -> Optional[str]:
    """Digest of the planner prompt the plan was built with, from the first row.

    ``None`` for a plan that predates the column, which forces a re-plan.
    """
    rows = load_plan(project_name, book_name, base)
    if not rows:
        return None
    return rows[0].get("prompt_hash") or None


def plan_chunking(project_name: str, book_name: str,
                  base: Optional[Path] = None) -> str:
    """``"scene_map"``, ``"linear"``, or ``""`` for a plan that predates the column."""
    rows = load_plan(project_name, book_name, base)
    if not rows:
        return ""
    return (rows[0].get("chunking") or "").strip()


def scene_chunk(
    project_name: str,
    book_name: str,
    chapter_num: int,
    shot_num: int,
    chunk_words: Optional[int] = None,
    base: Optional[Path] = None,
) -> Optional[str]:
    """Re-derive one scene's chunk text.

    ``chunk_words`` defaults to the size pinned in the plan, so a stored
    ``(chapter, shot)`` maps back to the same text it was planned from. The
    chunk source is the same one prompt generation uses; a stale scene map raises
    rather than quietly returning a linear chunk.
    """
    if chunk_words is None:
        chunk_words = plan_chunk_words(project_name, book_name, base) or DEFAULT_CHUNK_WORDS
    chunks, _source = scene_map.book_chunks(
        book_dir(project_name, book_name, base), chunk_words)
    for chunk in chunks:
        if chunk.chapter == int(chapter_num) and chunk.shot == int(shot_num):
            return chunk.text
    return None


def verify_plan(
    project_name: str, book_name: str, base: Optional[Path] = None
) -> List[Tuple[int, int, str, str]]:
    """Re-derive every planned scene and return ``(chapter, shot, stored, actual)``
    for those whose chunk text no longer matches the plan."""
    rows = load_plan(project_name, book_name, base)
    if not rows:
        return []
    words = plan_chunk_words(project_name, book_name, base) or DEFAULT_CHUNK_WORDS
    chunks, _source = scene_map.book_chunks(
        book_dir(project_name, book_name, base), words)
    actual_by_key = {(chunk.chapter, chunk.shot): chunk_digest(chunk.text)
                     for chunk in chunks}
    mismatches: List[Tuple[int, int, str, str]] = []
    for row in rows:
        chapter, scene = int(row["chapter"]), int(row["shot"])
        actual = actual_by_key.get((chapter, scene), "")
        if actual != row.get("chunk_hash", ""):
            mismatches.append((chapter, scene, row.get("chunk_hash", ""), actual))
    return mismatches


async def ensure_book_plan(
    project_name: str,
    book_name: str,
    llm_url: str,
    model: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    window: int = DEFAULT_WINDOW,
    mode: str = "auto",
    base: Optional[Path] = None,
    log: Optional[Callable[[str], None]] = None,
    selection: Optional[Dict[str, str]] = None,
) -> Dict[Tuple[int, int], str]:
    """``{(chapter, shot): shot_type}`` for a book.

    Only consulted in ``auto`` mode. A stored plan is reused when it was built
    with the same chunk size, the same chunk source and the same Shot Planner
    prompt; otherwise the book is planned afresh and saved. ``selection`` is the
    run's variant snapshot, so a variant change made mid-run cannot split a book
    between two planner prompts either.
    """
    if mode != "auto":
        return {}

    chunks, source = scene_map.book_chunks(
        book_dir(project_name, book_name, base), chunk_words, log)
    template = planner_text(selection)
    digest = prompt_digest(template)
    stored = load_plan(project_name, book_name, base)
    stored_words = plan_chunk_words(project_name, book_name, base)
    stored_digest = plan_prompt_hash(project_name, book_name, base)
    stored_chunking = plan_chunking(project_name, book_name, base)
    # A plan with no chunking column predates it and was built by the linear
    # chunker, so it is only reusable while the book still resolves to linear.
    chunking_compatible = (stored_chunking == source
                           or (stored_chunking == "" and source == "linear"))
    if (stored and stored_words == chunk_words and stored_digest == digest
            and chunking_compatible):
        if log:
            log(f"[Shot-Plan] reusing stored plan for '{book_name}' ({len(stored)} scenes)")
        return {(int(r["chapter"]), int(r["shot"])): r["shot_type"] for r in stored}
    if stored and log:
        if stored_words != chunk_words:
            log(f"[Shot-Plan] stored plan for '{book_name}' used {stored_words} words "
                f"per scene, now {chunk_words} - replanning.")
        elif not chunking_compatible:
            log(f"[Shot-Plan] stored plan for '{book_name}' was built from "
                f"'{stored_chunking or 'linear'}' chunks, now '{source}' - replanning.")
        else:
            log(f"[Shot-Plan] planner prompt changed since the stored plan "
                f"({stored_digest or 'none'} -> {digest}) - replanning '{book_name}'.")

    plan = await plan_book(project_name, book_name, llm_url, model,
                           chunk_words=chunk_words, window=window, base=base, log=log,
                           template=template, chunks=chunks)
    if plan and all(shot == "NONE" for _chapter, _scene, shot, _digest, _focus in plan):
        # A dead or misconfigured LLM produces an all-NONE plan. Don't persist
        # that - fall back to the single prompt instead of poisoning the plan.
        if log:
            log("[Shot-Plan] planning produced no usable labels (LLM unreachable?) "
                "- falling back to the single prompt for this book.")
        return {}
    save_plan(project_name, book_name, plan, chunk_words, base, prompt_hash=digest,
              chunking=source)
    return {(chapter, scene): shot for chapter, scene, shot, _digest, _focus in plan}
