"""Shot Planner: assign a shot type to every scene of a book.

Runs the shared ``shot_direction`` prompt over windows of chunks and returns
one of ``WIDE`` / ``CHARACTER`` / ``INSERT`` / ``NONE`` per scene.

The plan is stored in its own ``shot_plan.csv`` next to the transcript, not in
``prompts.csv``: ``start_project_prompt_gen`` treats any existing prompts.csv
row as an already-finished scene, so pre-writing rows there would make prompt
generation skip the whole book.

The chunk size used is recorded in the plan, and each scene carries a digest of
its chunk, because ``(chapter, shot)`` only maps back to the same text while
the transcript and the chunk size are unchanged. ``scene_chunk`` re-derives a
chunk on demand; ``verify_plan`` reports any scene whose text has drifted.

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
from typing import Callable, Dict, List, Optional, Tuple

from database.connection import get_setting
from services import compat
from services.prompt_engine import get_llm_response, smart_chunk_text
from services.prompt_library import SHOT_TYPES, planner_text

# Chunks per request. 30 keeps the window well inside a 98k context and keeps
# the reply list short enough for a small model to keep in order.
DEFAULT_WINDOW = 30
DEFAULT_CHUNK_WORDS = 350

PLAN_FILENAME = "shot_plan.csv"
PLAN_COLUMNS = ("chapter", "shot", "shot_type", "chunk_words", "chunk_hash",
                "prompt_hash")

_LINE = re.compile(r"^\s*(\d+)\s*[:.\)\-]\s*\**([A-Za-z]+)")
_ALIASES = {
    "GROUP": "WIDE", "CROWD": "WIDE", "CROWDSHOT": "WIDE",
    "ESTABLISHING": "WIDE", "LANDSCAPE": "WIDE", "SCENERY": "WIDE",
    "DETAIL": "INSERT", "OBJECT": "INSERT", "ITEM": "INSERT",
    "CHARACTERS": "CHARACTER", "PEOPLE": "CHARACTER", "FIGURE": "CHARACTER",
}


# --- parsing and planning --------------------------------------------------

def parse_shot_lines(raw: str, count: int) -> Dict[int, str]:
    """Parse ``<n>: <SHOT>`` lines into ``{1-based index: shot type}``."""
    found: Dict[int, str] = {}
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
        found[idx] = label
    return found


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
    """Shot type per chunk, in order. Anything unparsed falls back to NONE.

    ``template`` pins the Shot Planner prompt; when omitted the active variant
    is read live, which is what the Prompt Lab test bench wants.
    """
    template = template if template is not None else planner_text()
    out: List[str] = []
    for start in range(0, len(chunks), window):
        window_chunks = chunks[start:start + window]
        if log:
            log(f"[Shot-Plan] classifying scenes {start + 1}-{start + len(window_chunks)}")
        raw = await get_llm_response(
            plan_prompt(window_chunks, template), llm_url, model,
            temperature=0.0,
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
        )
        got = parse_shot_lines(raw, len(window_chunks))
        out.extend(got.get(i + 1, "NONE") for i in range(len(window_chunks)))
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
    """``[(chapter_num, chunks)]`` using the same split and numbering as prompt gen."""
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
) -> List[Tuple[int, int, str, str]]:
    """``[(chapter_num, shot_num, shot_type, chunk_hash)]`` for a whole book.

    Chapter numbering matches prompt generation exactly: the transcript is
    split on ``==CHAPTER==`` and numbered from 1, so on these transcripts the
    front matter + prologue segment is chapter 1 and each book chapter is
    shifted by one.
    """
    text = read_book_text(project_name, book_name, base)
    if text is None:
        raise FileNotFoundError(book_transcript_path(project_name, book_name, base))

    plan: List[Tuple[int, int, str, str]] = []
    for chapter_num, chunks in chunk_book_chapters(text, chunk_words):
        if not chunks:
            continue
        if log:
            log(f"[Shot-Plan] Chapter {chapter_num}: {len(chunks)} scenes")
        shots = await plan_chunks(chunks, llm_url, model, window, log, template)
        for shot_num, (shot, chunk) in enumerate(zip(shots, chunks), start=1):
            plan.append((chapter_num, shot_num, shot, chunk_digest(chunk)))
    return plan


# --- plan persistence ------------------------------------------------------

def save_plan(
    project_name: str,
    book_name: str,
    plan: List[Tuple[int, int, str, str]],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    base: Optional[Path] = None,
    prompt_hash: str = "",
) -> Path:
    """Write ``shot_plan.csv``. Self-contained: it records the chunk size used
    and the digest of the planner prompt the shot types came from."""
    path = plan_path(project_name, book_name, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(PLAN_COLUMNS)
        for chapter, scene, shot, digest in plan:
            writer.writerow([chapter, scene, shot, chunk_words, digest, prompt_hash])
    return path


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
    ``(chapter, shot)`` maps back to the same text it was planned from.
    """
    if chunk_words is None:
        chunk_words = plan_chunk_words(project_name, book_name, base) or DEFAULT_CHUNK_WORDS
    text = read_book_text(project_name, book_name, base)
    if text is None:
        return None
    for num, chunks in chunk_book_chapters(text, chunk_words):
        if num == int(chapter_num):
            idx = int(shot_num) - 1
            return chunks[idx] if 0 <= idx < len(chunks) else None
    return None


def verify_plan(
    project_name: str, book_name: str, base: Optional[Path] = None
) -> List[Tuple[int, int, str, str]]:
    """Re-derive every planned scene and return ``(chapter, shot, stored, actual)``
    for those whose chunk text no longer matches the plan."""
    rows = load_plan(project_name, book_name, base)
    if not rows:
        return []
    words = plan_chunk_words(project_name, book_name, base)
    mismatches: List[Tuple[int, int, str, str]] = []
    for row in rows:
        chapter, scene = int(row["chapter"]), int(row["shot"])
        chunk = scene_chunk(project_name, book_name, chapter, scene, words, base)
        actual = chunk_digest(chunk) if chunk is not None else ""
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
    with the same chunk size *and* the same Shot Planner prompt; otherwise the
    book is planned afresh and saved. ``selection`` is the run's variant
    snapshot, so a variant change made mid-run cannot split a book between two
    planner prompts either.
    """
    if mode != "auto":
        return {}

    template = planner_text(selection)
    digest = prompt_digest(template)
    stored = load_plan(project_name, book_name, base)
    stored_words = plan_chunk_words(project_name, book_name, base)
    stored_digest = plan_prompt_hash(project_name, book_name, base)
    if stored and stored_words == chunk_words and stored_digest == digest:
        if log:
            log(f"[Shot-Plan] reusing stored plan for '{book_name}' ({len(stored)} scenes)")
        return {(int(r["chapter"]), int(r["shot"])): r["shot_type"] for r in stored}
    if stored and log:
        if stored_words != chunk_words:
            log(f"[Shot-Plan] stored plan for '{book_name}' used {stored_words} words "
                f"per scene, now {chunk_words} - replanning.")
        else:
            log(f"[Shot-Plan] planner prompt changed since the stored plan "
                f"({stored_digest or 'none'} -> {digest}) - replanning '{book_name}'.")

    plan = await plan_book(project_name, book_name, llm_url, model,
                           chunk_words=chunk_words, window=window, base=base, log=log,
                           template=template)
    if plan and all(shot == "NONE" for _chapter, _scene, shot, _digest in plan):
        # A dead or misconfigured LLM produces an all-NONE plan. Don't persist
        # that - fall back to the single prompt instead of poisoning the plan.
        if log:
            log("[Shot-Plan] planning produced no usable labels (LLM unreachable?) "
                "- falling back to the single prompt for this book.")
        return {}
    save_plan(project_name, book_name, plan, chunk_words, base, prompt_hash=digest)
    return {(chapter, scene): shot for chapter, scene, shot, _digest in plan}
