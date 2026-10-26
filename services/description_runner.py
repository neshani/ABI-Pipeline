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
from services import trait_vocab
from services.character_manager import is_non_human  # noqa: F401  (re-exported for UI)
from services.character_manager import is_placeholder_description

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

def load_transcripts(project_id: int) -> List[tuple]:
    """``[(book name, transcript text)]`` in reading order, for the book-text step.

    Shared with the single-character lookup so both paths search the same books in
    the same order. A missing transcript is skipped rather than fatal: a project
    mid-transcription should still be able to describe the books it has.
    """
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return []
        base_dir = Path(get_setting("output_dir", "./output")).resolve()
        books = []
        for book in session.exec(select(Book).where(Book.project_id == project_id)
                                 .order_by(Book.book_order)).all():
            path = base_dir / project.name / book.name / "transcript.txt"
            if path.exists():
                books.append((book.name, path.read_text(encoding="utf-8",
                                                        errors="ignore")))
        return books


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

# The fields a lookup may write. ``face`` is one of them on purpose: a reference
# source states it as a stable feature ("a narrow face, high cheekbones"), which is
# exactly what the renderer wants, whereas the transcript extractor sees mostly
# transient prose ("her face reddened") and deliberately does not ask for it.
# ``height`` is absent because no column stores it - it is folded into ``build``
# below, the same way every other import folds it.
APPLIED_FIELDS = ("sex", "species", "age", "build", "hair_color", "hair_style",
                   "eye_color", "face")

# Marks and notes have no field of their own in the compiled prompt, so they are
# collected in ``notes`` - stored, searchable, never rendered. That is the point of
# notes: a detail worth keeping does not have to be worth drawing.
_NOTE_FIELDS = ("notes", "marks")

# Asked for by a source but impossible to store: saying so beats showing a value
# in the results and then dropping it in silence.
_UNSTORED_FIELDS = {
    "attire": "clothing is a scene prop, not a profile field",
    "carried": "a carried object is a scene prop, not a profile field",
}


def _drop_reason(field: str, before: str = "") -> str:
    """Why the vocabulary emptied a value, in words a user can act on."""
    if field == "build":
        if trait_vocab.states_average_build(before):
            return ("a middling frame adds nothing to a prompt, so it is not "
                    "stored - type the wording you want in the Character tab")
        return ("no frame or height word the renderer can use - type the wording "
                "you want in the Character tab")
    if field == "age":
        return "not an age stage this app knows"
    if field in ("hair_color", "eye_color"):
        return "not a colour"
    return "nothing left after normalising"


def _merge_notes(event, additions: List[tuple]) -> List[Dict[str, str]]:
    """Append source ``notes``/``marks`` to the event's notes; report what was added.

    Notes is a note drawer, not a trait, so this appends instead of filling only
    when empty: a second run that mentions the same scar must not delete it, and a
    hand-written note must not be replaced by a search result. The dedup is on the
    text itself, which is what makes a repeated run a no-op.
    """
    current = (getattr(event, "notes", None) or "").strip()
    pieces = [piece.strip().strip(" .,;") for piece in current.split(";") if piece]
    added: List[Dict[str, str]] = []

    def known(text: str) -> bool:
        low = text.lower()
        existing = pieces + [item["value"] for item in added]
        return any(low in other.lower() for other in existing)

    for field, value in additions:
        text = " ".join(str(value or "").split()).strip(" .,;")
        if not text or trait_vocab.is_non_answer(text) or known(text):
            continue
        added.append({"field": field, "value": text})
        pieces.append(text)

    if added:
        event.notes = "; ".join(pieces)
    return added


def apply_fields(event, fields: Dict[str, str]) -> Dict:
    """Merge ``fields`` into ``event`` in memory; return what happened.

    Not committed here: the caller owns the session, which keeps the dry run and
    the real run on exactly the same code path.

    Every input value ends up in exactly one bucket, and a value the vocabulary
    cannot express goes to ``dropped`` with its original wording rather than
    vanishing. A silent drop reads to the user as "the search found it and the app
    threw it away", which is indistinguishable from nothing having been found.
    ``notes`` and ``marks`` are appended to the notes field and reported in
    ``noted``: kept for reference, never compiled into the prompt.
    """
    raw: Dict[str, str] = {}
    for field, value in (fields or {}).items():
        text = str(value or "").strip()
        if text:
            raw[str(field)] = text

    # Normalise once, on the way in, exactly as the reference/wiki/book paths do:
    # this is what folds a separate ``height`` into ``build`` and canonicalises the
    # frame wording. A hand edit in the Character tab never passes through here.
    changes: List[tuple] = []
    values = trait_vocab.normalise_traits(raw, changes)
    rewritten_by_field = {field: after for field, _before, after in changes}

    filled, conflicts, unchanged, dropped, rewritten = [], [], [], [], []
    for field in APPLIED_FIELDS:
        before = raw.get(field)
        if not before:
            continue
        value = (values.get(field) or "").strip()
        if not value:
            dropped.append({"field": field, "found": before,
                            "reason": _drop_reason(field, before)})
            continue
        if field in rewritten_by_field and value != before:
            rewritten.append({"field": field, "was": before, "now": value})
        current = (getattr(event, field, None) or "").strip()
        if not current:
            setattr(event, field, value)
            filled.append(field)
        elif current.lower() == value.lower():
            unchanged.append(field)
        else:
            # ``found`` is the stored shape, ``source`` the source's own wording,
            # so the "kept X (source said Y)" line quotes what was actually read.
            conflicts.append({"field": field, "was": current, "found": value,
                              "source": before})

    # A height on its own has no column and is only worth mentioning when it did
    # not produce a build: an average height is dropped by design (it says nothing
    # an image model can draw), and that is the case a user would otherwise report
    # as "it found the height and saved nothing".
    height = raw.get("height")
    if height and not raw.get("build") and not (values.get("build") or "").strip():
        dropped.append({
            "field": "height", "found": height,
            "reason": ("folded into build, and this height is either average "
                       "(dropped as uninformative) or not readable as a height")})

    # Notes and marks are kept, not drawn: no trait field renders them, so they are
    # appended to notes. Notes is the one field where appending is right - it is a
    # note drawer rather than a value the prompt reads, and the dedup in
    # ``_merge_notes`` keeps a repeated lookup from writing the same scar twice.
    noted = _merge_notes(event, [(field, raw[field]) for field in _NOTE_FIELDS
                                 if raw.get(field)])
    if noted and "notes" not in filled:
        filled.append("notes")

    for field, value in raw.items():
        if field in APPLIED_FIELDS or field in _NOTE_FIELDS or field == "height":
            continue
        dropped.append({"field": field, "found": value,
                        "reason": _UNSTORED_FIELDS.get(
                            field, "there is no such field in the profile")})

    return {"filled": filled, "conflicts": conflicts, "unchanged": unchanged,
            "dropped": dropped, "rewritten": rewritten, "noted": noted}


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
        books = load_transcripts(project_id)

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
                   "dropped": 0, "skipped_reason": ""}
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
                    # A no-trait placeholder stamped at creation is not an
                    # override; leaving it would keep hiding the traits just filled.
                    if is_placeholder_description(event.visual_description):
                        event.visual_description = None
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
            summary["dropped"] += len(outcome["dropped"])
            entry["steps"][step_id] = {
                "status": "filled" if outcome["filled"] else "no change",
                "filled": outcome["filled"],
                "conflicts": outcome["conflicts"],
                "dropped": outcome["dropped"],
                "rewritten": outcome["rewritten"],
                "noted": outcome["noted"],
                "sources": result.get("sources") or [],
                "source_title": result.get("source_title", ""),
            }
            emit({"type": "character", "id": row["id"], "name": row["name"],
                  "step": step_id,
                  "status": entry["steps"][step_id]["status"],
                  "filled": outcome["filled"], "conflicts": outcome["conflicts"],
                  "dropped": outcome["dropped"], "noted": outcome["noted"]})

        report["steps"].append(summary)
        if report["cancelled"]:
            break

    report["seconds"] = round(time.time() - started, 1)
    report["report_path"] = _save_report(project_name, report, dry_run=dry_run)
    emit({"type": "done", "report": report})
    return report


def _save_report(project_name: str, report: Dict, *, dry_run: bool = False) -> str:
    """Write the run report to disk and return its path.

    The modal is the only place a report used to exist, so closing it destroyed
    the answer to "why did this character get nothing?". A run that reports only
    inside a dialog it cannot outlive is a run nobody can audit - and the failures
    are exactly the part worth keeping (a dead MCP server, an ambiguous reference
    entry). Never raises: losing a report must not fail an import.
    """
    import json
    from datetime import datetime
    try:
        base = Path(get_setting("output_dir", "./output"))
        folder = base / project_name / "description_runs"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        suffix = "_dry" if dry_run else ""
        path = folder / f"{stamp}{suffix}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return str(path)
    except Exception:
        return ""


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
