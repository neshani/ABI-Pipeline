"""Bulk descriptions: an ordered list of sources, applied to the Base profile.

The steps are a queue rather than five independent buttons, because the useful
run is "reference first, wiki for what it missed, MCP for the residue". Each step
therefore re-selects its targets from the characters that are still below the
threshold, and a disabled step is skipped without affecting the order.

Two rules the code enforces rather than documents:

* **Fill empty, never overwrite.** A source that disagrees with an existing value
  produces a *conflict* that is reported and not written. Both directions of
  error happen in practice - a wiki says "brown hair" where the book says "dark",
  and the reverse - so the decision belongs to the user.
* **Base only.** Nothing here writes per-book events. Positional change tracking
  is left for manual edits; automated extraction writes the profile.

``dry_run`` computes everything and writes nothing, which is what the headless
tests use and what the modal's preview ends up calling.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from sqlmodel import Session, select

from database.connection import engine, get_setting
from database.models import (Book, Character, CharacterAlias, CharacterTimelineEvent,
                             Project)
from services import character_coverage
from services.character_manager import is_non_human  # noqa: F401  (re-exported for UI)

STEPS: List[Dict] = [
    {"id": "reference", "title": "Reference file",
     "hint": "A companion volume or a hand-written markdown sheet. Highest accuracy.",
     "default": True},
    {"id": "wiki", "title": "Wiki",
     "hint": "One MediaWiki site, read through its API with a one request per second limit.",
     "default": True},
    {"id": "mcp", "title": "Web (MCP)",
     "hint": "The MCP server from Settings. Best for characters nothing else covers.",
     "default": True},
    {"id": "transcript", "title": "Book text",
     "hint": "Last resort: reads the book itself. Slowest and least reliable.",
     "default": False},
]

STEP_IDS = [step["id"] for step in STEPS]


def default_steps() -> List[Dict]:
    return [{"id": step["id"], "enabled": bool(step["default"])} for step in STEPS]


def ordered_steps(steps: Optional[Sequence[Dict]]) -> List[str]:
    """Enabled step ids in the user's order, unknown ids dropped."""
    if not steps:
        return [step["id"] for step in STEPS if step["default"]]
    known = {step["id"] for step in STEPS}
    return [str(step.get("id")) for step in steps
            if step.get("enabled", True) and str(step.get("id")) in known]


def validate_steps(steps: Optional[Sequence[Dict]], config: Dict) -> Dict:
    """Which enabled steps can actually run, and why the others cannot.

    Called before a run so a missing file or URL is reported as a problem instead
    of becoming a step that quietly does nothing. The reasons here are the same
    ones ``run`` would skip a step for.
    """
    enabled = ordered_steps(steps)
    runnable: List[str] = []
    skipped: List[Dict] = []
    for step_id in enabled:
        if step_id == "reference":
            path = str(config.get("reference_path") or "").strip()
            if not path:
                skipped.append({"id": step_id, "reason": "no reference file chosen"})
            elif not Path(path).exists():
                skipped.append({"id": step_id, "reason": f"file not found: {path}"})
            else:
                runnable.append(step_id)
        elif step_id == "wiki":
            if not str(config.get("wiki_url") or "").strip():
                skipped.append({"id": step_id, "reason": "no wiki URL set"})
            else:
                runnable.append(step_id)
        elif step_id == "mcp":
            if not str(config.get("mcp_url") or "").strip():
                skipped.append({"id": step_id, "reason": "no MCP server set"})
            else:
                runnable.append(step_id)
        elif step_id == "transcript":
            # Whether transcripts exist is a per-book fact, checked in ``run``;
            # there is nothing to configure here, so it is always runnable.
            runnable.append(step_id)
    return {"enabled": enabled, "runnable": runnable, "skipped": skipped}


# --- targets ----------------------------------------------------------------

def load_rows(project_id: int) -> List[Dict]:
    """Every character with its Base profile and aliases, for target selection."""
    with Session(engine) as session:
        characters = session.exec(
            select(Character).where(Character.project_id == project_id)).all()
        aliases: Dict[int, List[str]] = {}
        for alias in session.exec(select(CharacterAlias)).all():
            aliases.setdefault(alias.character_id, []).append(alias.alias)
        out = []
        for char in characters:
            base = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == char.id)
                .where(CharacterTimelineEvent.book_id == None)  # noqa: E711
            ).first()
            out.append({"id": char.id, "name": char.name,
                        "event": base,
                        "aliases": sorted(a for a in aliases.get(char.id, [])
                                          if a.strip())})
    return out


def targets(rows: Sequence[Dict], threshold: int) -> List[Dict]:
    return [row for row in rows
            if character_coverage.needs_work(row["event"], threshold)]


# --- applying ---------------------------------------------------------------

_FIELD_LABELS = {
    "sex": "sex", "species": "species", "age": "age", "build": "build",
    "hair_color": "hair colour", "hair_style": "hair style",
    "eye_color": "eye colour",
}


def apply_fields(event, fields: Dict[str, str]) -> Dict:
    """Merge ``fields`` into ``event`` in memory; return what happened.

    Not committed here: the caller owns the session, which keeps the dry run and
    the real run on exactly the same code path.
    """
    filled, conflicts, unchanged = [], [], []
    for field, value in fields.items():
        value = (value or "").strip()
        if not value or field not in _FIELD_LABELS:
            continue
        current = (getattr(event, field, None) or "").strip()
        if not current:
            setattr(event, field, value)
            filled.append(field)
        elif current.lower() == value.lower():
            unchanged.append(field)
        else:
            conflicts.append({"field": field, "was": current, "found": value})
    return {"filled": filled, "conflicts": conflicts, "unchanged": unchanged}


def ensure_base_event(session: Session, character_id: int):
    event = session.exec(
        select(CharacterTimelineEvent)
        .where(CharacterTimelineEvent.character_id == character_id)
        .where(CharacterTimelineEvent.book_id == None)  # noqa: E711
    ).first()
    if event is not None:
        return event
    event = CharacterTimelineEvent(character_id=character_id, book_id=None,
                                   chapter_num=0, shot_num=0, label="Base State")
    session.add(event)
    session.flush()
    return event


# --- sources ----------------------------------------------------------------

async def _fields_from_reference(name: str, aliases: List[str], *, blocks,
                                 llm_url: str, model: str) -> Dict:
    from services import description_extract, reference_import
    names = {name: aliases}
    matched, _unmatched = reference_import.match_blocks(blocks, names)
    block = matched.get(name)
    if not block:
        return {"ok": False, "fields": {}, "error": "no entry in the reference file"}
    return await description_extract.extract_from_text(
        name, block[1], llm_url=llm_url, model=model)


async def _fields_from_wiki(name: str, aliases: List[str], *, wiki_url: str,
                            wiki_term: str, llm_url: str, model: str) -> Dict:
    from services import description_extract, wiki_lookup
    got = wiki_lookup.lookup(name, base=wiki_url, term=wiki_term, aliases=aliases)
    if not got["ok"]:
        return {"ok": False, "fields": {}, "error": got["error"]}
    result = await description_extract.extract_from_text(
        name, got["block"], llm_url=llm_url, model=model)
    result["source_title"] = got.get("title", "")
    return result


async def _fields_from_mcp(name: str, series: str, *, mcp_url: str,
                           mcp_auth: str, tool_prefs: Dict, llm_url: str,
                           model: str) -> Dict:
    from services import web_lookup
    result = await web_lookup.lookup_character(
        name, series, mcp_url=mcp_url, mcp_auth=mcp_auth, tool_prefs=tool_prefs,
        llm_url=llm_url, model=model)
    return {"ok": result.get("ok", False), "fields": result.get("fields") or {},
            "error": result.get("error", ""),
            "sources": result.get("sources") or []}


async def _fields_from_transcript(name: str, aliases: List[str], *, books,
                                  llm_url: str, model: str) -> Dict:
    from services import book_lookup
    result = await book_lookup.lookup(
        name, books=books, aliases=aliases, llm_url=llm_url, model=model)
    return {"ok": result.get("ok", False), "fields": result.get("fields") or {},
            "error": result.get("error", "")}


# --- run --------------------------------------------------------------------

async def run(project_id: int, *, threshold: int = 0,
              steps: Optional[Sequence[Dict]] = None,
              config: Optional[Dict] = None,
              dry_run: bool = False,
              limit: int = 0,
              on_event: Optional[Callable[[Dict], None]] = None,
              should_cancel: Optional[Callable[[], bool]] = None) -> Dict:
    """Run the enabled steps over the characters below ``threshold``.

    Returns ``{steps: [...], characters: {...}, cancelled: bool, seconds: float}``
    where each step carries its own counts and every character carries the fields
    it gained and the conflicts it raised.
    """
    config = dict(config or {})
    llm_url = config.get("llm_url") or get_setting("llm_url", "http://127.0.0.1:11434")
    model = config.get("model") or get_setting("llm_model", "local-model")

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return {"ok": False, "error": "project not found", "steps": [],
                    "characters": {}, "cancelled": False, "seconds": 0.0}
        project_name = project.name

    books: List[tuple] = []
    if "transcript" in ordered_steps(steps):
        base_dir = Path(get_setting("output_dir", "./output")).resolve()
        with Session(engine) as session:
            for book in session.exec(select(Book).where(Book.project_id == project_id)
                                     .order_by(Book.book_order)).all():
                path = base_dir / project_name / book.name / "transcript.txt"
                if path.exists():
                    books.append((book.name, path.read_text(encoding="utf-8",
                                                            errors="ignore")))

    blocks = None
    if "reference" in ordered_steps(steps) and config.get("reference_path"):
        from services import reference_import
        try:
            blocks = reference_import.load_blocks(Path(config["reference_path"]))
        except Exception as ex:
            blocks = {"error": str(ex)}
    elif "reference" in ordered_steps(steps):
        blocks = {"error": "no reference file chosen"}

    report = {"ok": True, "error": "", "steps": [], "characters": {},
              "cancelled": False, "seconds": 0.0}
    started = time.time()

    def emit(payload: Dict) -> None:
        if on_event:
            on_event(payload)

    rows = load_rows(project_id)
    scope = targets(rows, threshold)
    if limit:
        scope = scope[:limit]
    scope_ids = {row["id"] for row in scope}
    emit({"type": "start", "targets": len(scope)})

    enabled = ordered_steps(steps)
    for step_id in enabled:
        if should_cancel and should_cancel():
            report["cancelled"] = True
            break
        # Re-select from the database, not from the rows loaded above: a step that
        # just filled a character can push them over the threshold, and they should
        # not be passed to the next source.
        step_rows = [row for row in targets(load_rows(project_id), threshold)
                     if row["id"] in scope_ids]
        if not step_rows:
            report["steps"].append({"id": step_id, "targets": 0, "filled": 0,
                                    "fields": 0, "failed": 0, "skipped": 0,
                                    "conflicts": 0, "skipped_reason": "nothing left to fill"})
            continue
        summary = {"id": step_id, "targets": len(step_rows), "filled": 0,
                   "fields": 0, "failed": 0, "skipped": 0, "conflicts": 0,
                   "skipped_reason": ""}
        emit({"type": "step", "id": step_id, "targets": len(step_rows)})

        if step_id == "reference" and isinstance(blocks, dict):
            summary["skipped_reason"] = blocks.get("error", "no reference file")
            summary["targets"] = 0
            report["steps"].append(summary)
            continue
        if step_id == "wiki" and not config.get("wiki_url"):
            summary["skipped_reason"] = "no wiki URL set"
            summary["targets"] = 0
            report["steps"].append(summary)
            continue
        if step_id == "mcp" and not config.get("mcp_url"):
            summary["skipped_reason"] = "no MCP server set"
            summary["targets"] = 0
            report["steps"].append(summary)
            continue
        if step_id == "transcript" and not books:
            summary["skipped_reason"] = "no transcripts found"
            summary["targets"] = 0
            report["steps"].append(summary)
            continue

        for row in step_rows:
            if should_cancel and should_cancel():
                report["cancelled"] = True
                break
            entry = report["characters"].setdefault(
                row["id"], {"name": row["name"], "steps": {}})
            try:
                if step_id == "reference":
                    result = await _fields_from_reference(
                        row["name"], row["aliases"], blocks=blocks,
                        llm_url=llm_url, model=model)
                elif step_id == "wiki":
                    result = await _fields_from_wiki(
                        row["name"], row["aliases"], wiki_url=config["wiki_url"],
                        wiki_term=config.get("wiki_term", ""),
                        llm_url=llm_url, model=model)
                elif step_id == "mcp":
                    result = await _fields_from_mcp(
                        row["name"], config.get("series") or project_name,
                        mcp_url=config["mcp_url"],
                        mcp_auth=config.get("mcp_auth", ""),
                        tool_prefs=config.get("mcp_prefs") or {},
                        llm_url=llm_url, model=model)
                else:
                    result = await _fields_from_transcript(
                        row["name"], row["aliases"], books=books,
                        llm_url=llm_url, model=model)
            except Exception as ex:
                result = {"ok": False, "fields": {}, "error": f"{type(ex).__name__}: {ex}"}

            if not result.get("ok"):
                summary["failed"] += 1
                entry["steps"][step_id] = {"status": "failed",
                                           "error": result.get("error", "")}
                emit({"type": "character", "id": row["id"], "name": row["name"],
                      "step": step_id, "status": "failed",
                      "error": result.get("error", "")})
                continue

            with Session(engine) as session:
                event = ensure_base_event(session, row["id"])
                outcome = apply_fields(event, result.get("fields") or {})
                if outcome["filled"] and not dry_run:
                    if event.source != "manual":
                        event.source = "manual"  # a curated import stays put
                    session.add(event)
                if dry_run:
                    session.rollback()
                else:
                    session.commit()

            if outcome["filled"]:
                summary["filled"] += 1
                summary["fields"] += len(outcome["filled"])
            else:
                summary["skipped"] += 1
            summary["conflicts"] += len(outcome["conflicts"])
            entry["steps"][step_id] = {
                "status": "filled" if outcome["filled"] else "no change",
                "filled": outcome["filled"],
                "conflicts": outcome["conflicts"],
                "sources": result.get("sources") or [],
                "source_title": result.get("source_title", ""),
            }
            emit({"type": "character", "id": row["id"], "name": row["name"],
                  "step": step_id,
                  "status": entry["steps"][step_id]["status"],
                  "filled": outcome["filled"], "conflicts": outcome["conflicts"]})

        report["steps"].append(summary)
        if report["cancelled"]:
            break

    report["seconds"] = round(time.time() - started, 1)
    emit({"type": "done", "report": report})
    return report


def summarise(report: Dict) -> str:
    """One line for the console log."""
    parts = []
    for step in report.get("steps", []):
        if step.get("skipped_reason"):
            parts.append(f"{step['id']}: skipped ({step['skipped_reason']})")
        else:
            parts.append(f"{step['id']}: {step['filled']}/{step['targets']} "
                         f"characters, {step['fields']} fields"
                         + (f", {step['failed']} failed" if step["failed"] else ""))
    return "; ".join(parts) or "nothing to do"
