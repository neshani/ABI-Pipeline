"""What is this character wearing, at this position in the book?

The profile fields describe a body; clothing is not one of them. ``attire`` and
``carried`` live on a timeline event, which is a *change point*: the value holds
from that position until the next event says otherwise. That is why this module
writes an event and not the Base profile, and why every result is anchored to a
(chapter, shot) rather than to the character.

Two things it deliberately does not do:

* **It does not guess a name from a pronoun.** A run of prose names four people and
  then says "he wore a cloak"; the model is asked which of them is being dressed
  and how sure it is, and code crosses that with a second, mechanical test - does
  the quoted text name the character verbatim? Only a mention that passes both is
  pre-ticked. Everything else is evidence the user reads and accepts by hand.
* **It does not write on the strength of the model's answer.** ``find`` returns
  mentions; ``apply_attire`` is a separate call the dialog makes when a value is
  ticked. Attire renders straight into a prompt, so a wrong value is visible in an
  image, not just in a form.

The window is a fixed run of chunks around the position (-3, +2). Clothing is
established a scene or two before it is worn, so the previous chunks carry most of
the evidence, while a mention *after* the position is often the one that says they
changed. Each chunk comes from ``scene_map.book_chunks`` on the book's own recorded
chunking, so (chapter, shot) here means the same thing it means to the renderer.

Scope: **worn clothing only**. A thing held or carried is ``carried`` on the same
timeline row, compiles as "carrying ...", and is a separate feature - asking for
both in one pass produced "wearing a silver goblet of wine" often enough to matter.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sqlmodel import Session, select

from database.connection import engine, get_setting
from database.models import (Book, Character, CharacterAlias,
                             CharacterTimelineEvent, Project)
from services import book_lookup, scene_map

WINDOW_BACK = 3
WINDOW_AHEAD = 2
MAX_TOKENS = 900
TIMEOUT = 240.0

CLOTHING_PROMPT = """You are indexing a novel for an illustrated edition.

Below are consecutive passages from {book}, in reading order. Passage {at_n} is the
scene that matters; the passages before it lead up to it and the ones after it run
on from it.

Find every mention of what a person is WEARING: a garment, a coat, a dress, a
uniform, a hat, footwear, jewellery, armour, a cloak, a bandage or binding. A
mention counts even when it is only a phrase inside a longer sentence.

For each mention, output one JSON object on its own line:
{{"character": "...", "attire": "...", "quote": "...", "target": "...", "when": "..."}}

- "character": the name these passages give the person. If they are never named,
  write "unclear" - never invent a name from a pronoun alone.
- "attire": a short clause a painter can draw - "a long dark cloak with a hood",
  "a grey travelling dress", "a white shirt, sleeves rolled". Only what is worn.
- "quote": the sentence or phrase, copied VERBATIM from the passages.
- "target": "yes" when the mention is about {name}; "no" when it is about someone
  else; "unsure" when the passages do not settle who it is.
- "when": "now" when this is what they have on by the end of passage {at_n};
  "earlier" when the passage says it was worn before, or has since been changed.

These are NOT clothing, and must be left out entirely: a thing held, carried, eaten
or used (a cup, a pipe, a knife, a harp, a bundle); posture, gesture or action (arms
folded, hands on hips, sleeves pushed up as a movement); a body part or a quality of
one; a mood or an expression.

Most passages mention no clothing at all, and an empty answer is a correct answer.
One object per mention, nothing else: no prose, no code fence.

PASSAGES:
{passages}"""


def project_books(project_id: int) -> List[Book]:
    with Session(engine) as session:
        return list(session.exec(select(Book).where(Book.project_id == project_id)
                                 .order_by(Book.book_order)).all())


def book_directory(project_id: int, book_id: int) -> Optional[Path]:
    """``output/<project>/<book>``, or None when either row is missing."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        book = session.get(Book, book_id)
        if not project or not book:
            return None
        base = Path(get_setting("output_dir", "./output")).resolve()
        return base / project.name / book.name


def chunk_window(project_id: int, book_id: int, chapter: int, shot: int,
                 back: int = WINDOW_BACK, ahead: int = WINDOW_AHEAD) -> Dict:
    """The chunks around ``(chapter, shot)``, in reading order.

    Never crosses into another book: a shared world does not mean a shared volume,
    and the numbering restarts. A position with no exact chunk (a change row sits on
    shot 0) takes the last chunk at or before it, which is where that change starts.
    """
    directory = book_directory(project_id, book_id)
    if directory is None:
        return {"ok": False, "error": "that book does not exist",
                "chunks": [], "index": 0, "at": 0}
    try:
        chunks, source = scene_map.book_chunks(directory)
    except FileNotFoundError:
        return {"ok": False, "error": f"no transcript.txt in {directory.name}",
                "chunks": [], "index": 0, "at": 0}
    except scene_map.StaleSceneMap as ex:
        return {"ok": False, "error": str(ex), "chunks": [], "index": 0, "at": 0}
    except Exception as ex:                                  # noqa: BLE001
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}",
                "chunks": [], "index": 0, "at": 0}
    if not chunks:
        return {"ok": False, "error": "that book has no chunks",
                "chunks": [], "index": 0, "at": 0}

    index = 0
    for position, chunk in enumerate(chunks):
        if (chunk.chapter, chunk.shot) <= (chapter, shot):
            index = position
    start = max(0, index - back)
    end = min(len(chunks), index + ahead + 1)
    window = chunks[start:end]
    return {"ok": True, "error": "", "chunks": window, "index": index,
            "at": index - start, "source": source,
            "total": len(chunks), "book_dir": str(directory)}


def _objects(raw: str) -> List[Dict]:
    """Every top-level JSON object in a reply, one per line or not.

    A 12B writing one object per line will sometimes wrap the whole answer in a
    fence, and sometimes leave a trailing comma. Scanning for balanced braces gets
    the objects out without requiring the model to be tidy.
    """
    text = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M)
    found: List[Dict] = []
    depth, start = 0, None
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    data = json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict):
                    found.append(data)
                start = None
    return found


def _verdict(value) -> str:
    text = str(value or "").strip().lower()
    if text in ("yes", "y", "true", "target", "same"):
        return "yes"
    if text in ("no", "n", "false", "other"):
        return "no"
    return "unsure"


def _when(value) -> str:
    text = str(value or "").strip().lower()
    return "earlier" if text.startswith(("earlier", "before", "previous", "past")) \
        else "now"


def _names(character_id: int) -> List[str]:
    with Session(engine) as session:
        char = session.get(Character, character_id)
        if char is None:
            return []
        aliases = [row.alias for row in session.exec(
            select(CharacterAlias).where(
                CharacterAlias.character_id == character_id)).all()]
        return list(dict.fromkeys([char.name, *[a for a in aliases
                                               if (a or "").strip()]]))


def _locate(quote: str, passages: Sequence[Dict]) -> Dict:
    """Which passage a quote came from, so a mention can show its own position."""
    needle = re.sub(r"\s+", " ", quote or "").strip().lower()
    if needle:
        for passage in passages:
            haystack = re.sub(r"\s+", " ", passage["text"]).lower()
            if needle in haystack:
                return passage
        probe = needle[:60]
        for passage in passages:
            if probe and probe in re.sub(r"\s+", " ", passage["text"]).lower():
                return passage
    return passages[0] if passages else {"chapter": 0, "shot": 0, "text": ""}


async def find(project_id: int, book_id: int, character_id: int, chapter: int,
               shot: int, *, llm_url: str, model: str,
               log: Optional[Callable[[str], None]] = None) -> Dict:
    """Find every mention of clothing in the window around ``(chapter, shot)``.

    Returns ``mentions`` (target ones first, pre-ticked only where the quoted text
    names the character), the passages used, and the position actually searched -
    which may not be the one asked for when a change row sits between shots.
    """
    started = time.time()
    if log is None:
        log = lambda _message: None                       # noqa: E731
    names = _names(character_id)
    if not names:
        return {"ok": False, "error": "character not found", "mentions": [],
                "passages": [], "seconds": 0.0, "position": {}}

    with Session(engine) as session:
        book = session.get(Book, book_id)
        project = session.get(Project, project_id)
        book_name = book.name if book else ""
        project_name = project.name if project else ""
    if not book_name:
        return {"ok": False, "error": "book not found", "mentions": [],
                "passages": [], "seconds": 0.0, "position": {}}

    # Re-chunking is real CPU work - reading the transcript and re-splitting it
    # takes about 6s for a 400k-word volume - and it used to run on the event
    # loop, before this coroutine's first await. That is long enough for the
    # browser to give up on the websocket (NiceGUI pings every 4s and the client
    # allows 2s more), so the page reloaded mid-search and the dialog's render
    # landed on a deleted client. Keep it off the loop.
    window = await asyncio.to_thread(
        chunk_window, project_id, book_id, int(chapter), int(shot or 1))
    if not window["ok"]:
        return {"ok": False, "error": window["error"], "mentions": [],
                "passages": [], "seconds": 0.0, "position": {}}

    passages = [{"chapter": chunk.chapter, "shot": chunk.shot, "text": chunk.text}
                for chunk in window["chunks"]]
    at_n = window["at"] + 1
    listing = "\n\n".join(
        f"[{index + 1}] chapter {item['chapter']}, shot {item['shot']}\n{item['text']}"
        for index, item in enumerate(passages))
    log(f"[Clothing] window: {len(passages)} chunk(s), passage {at_n} is the "
        f"target ({passages[window['at']]['chapter']}, "
        f"{passages[window['at']]['shot']})")

    prompt = CLOTHING_PROMPT.format(
        book=book_name, name=names[0], at_n=at_n, passages=listing)
    from services.prompt_engine import get_llm_response
    try:
        raw, finish = await get_llm_response(
            prompt, llm_url, model, temperature=0.2, timeout=TIMEOUT,
            with_meta=True, extra_payload={"max_tokens": MAX_TOKENS})
    except Exception as ex:                                  # noqa: BLE001
        return {"ok": False, "error": f"the model call failed: {ex}",
                "mentions": [], "passages": passages,
                "seconds": round(time.time() - started, 1),
                "position": _position(book_name, passages, window)}

    objects = _objects(raw)
    log(f"[Clothing] {len(objects)} mention(s) in the reply")

    mentions: Dict[Tuple[str, str], Dict] = {}
    for item in objects:
        attire = " ".join(str(item.get("attire") or "").split()).strip(" .,;")
        # A transcript read with errors="ignore" can carry U+FFFD where an
        # apostrophe used to be ("boys" -> "boy<?>s"). The quote stays verbatim as
        # evidence; the stored value must not carry a replacement character into a
        # prompt.
        attire = attire.replace("\ufffd", "'")
        if not attire:
            continue
        who = " ".join(str(item.get("character") or "").split()).strip()
        quote = " ".join(str(item.get("quote") or "").split())
        if not quote:
            continue
        target = _verdict(item.get("target"))
        # Mechanical check, independent of the model's own verdict: does the quoted
        # text name this character? A "yes" whose quote only says "he" is still a
        # proposal, but it is not something to tick on the user's behalf.
        named = book_lookup._mentions(quote, names)  # one definition of "names them"
        place = _locate(quote, passages)
        unnamed = not who or who.lower() in ("unclear", "unknown", "unnamed",
                                             "someone", "a man", "a woman")
        entry = {
            "character": "unclear" if unnamed else who,
            "attire": attire,
            "quote": quote,
            "target": target,
            "when": _when(item.get("when")),
            "names_target": named,
            "unnamed": unnamed,
            # A named mention the model placed with someone else is that person's
            # clothing and is shown as evidence only - storing it here would dress
            # the wrong character. An *unnamed* mention is the ambiguous case the
            # whole dialog exists for, so it stays selectable and the user reads
            # the quote before deciding.
            "selectable": unnamed or target in ("yes", "unsure"),
            "about_target": target == "yes",
            "chapter": place.get("chapter") or passages[0]["chapter"],
            "shot": place.get("shot") or passages[0]["shot"],
        }
        entry["pre_tick"] = (entry["about_target"] and named
                             and entry["when"] == "now")
        key = (entry["character"].lower(), attire.lower())
        previous = mentions.get(key)
        if previous is None:
            mentions[key] = entry
        else:
            # Same person, same outfit, mentioned twice: keep one row, and let the
            # "now" mention (and a named quote) win the position and the pre-tick.
            if entry["when"] == "now" and previous["when"] != "now":
                previous.update({"when": "now", "quote": entry["quote"],
                                 "chapter": entry["chapter"],
                                 "shot": entry["shot"],
                                 "names_target": entry["names_target"]})
            previous["pre_tick"] = previous["pre_tick"] or entry["pre_tick"]
            previous["about_target"] = (previous["about_target"]
                                        or entry["about_target"])
            previous["selectable"] = (previous["selectable"]
                                      or entry["selectable"])
            previous["unnamed"] = previous["unnamed"] and entry["unnamed"]

    ordered = sorted(
        mentions.values(),
        key=lambda item: (0 if item["pre_tick"] else 1,
                          0 if item["about_target"] else
                          (1 if item["target"] == "unsure" else
                           (2 if not item["unnamed"] else 3)),
                          item["chapter"], item["shot"], item["attire"].lower()))
    error = ""
    if not ordered:
        # "Nothing found" has two very different causes - the passages describe no
        # clothing, or the reply could not be read - and they look identical to a
        # user staring at an empty list. Say which, with enough of the reply to
        # recognise it.
        snippet = " ".join((raw or "").split())[:180]
        detail = f"the reply was {len(raw or '')} characters"
        if finish == "length":
            detail += ", cut off at the token limit"
        error = ("no mention could be read from the model's reply ("
                 + detail + ")" + (f": {snippet}" if snippet else ": it was empty"))
    return {
        "ok": True,
        "error": error,
        "mentions": ordered,
        "passages": passages,
        "names": names,
        "seconds": round(time.time() - started, 1),
        "position": _position(book_name, passages, window),
    }


def _position(book_name: str, passages: Sequence[Dict], window: Dict) -> Dict:
    target = passages[window["at"]]
    return {"book": book_name, "chapter": target["chapter"], "shot": target["shot"],
            "window": [(item["chapter"], item["shot"]) for item in passages],
            "source": window.get("source", ""), "total_chunks": window.get("total", 0)}


def _ordered_events(session: Session, character: Character):
    """``(events, base, order_map, names, side_books)`` for one character."""
    books = session.exec(
        select(Book).where(Book.project_id == character.project_id)).all()
    order = {b.id: (b.book_order if b.book_order is not None else 10 ** 6)
             for b in books}
    names = {b.id: b.name for b in books}
    side = {b.id for b in books if getattr(b, "is_side_story", False)}
    events = session.exec(
        select(CharacterTimelineEvent)
        .where(CharacterTimelineEvent.character_id == character.id)).all()
    base = next((e for e in events if e.book_id is None), None)
    return events, base, order, names, side


def attire_target(character_id: int, book_id: int, chapter: int,
                  shot: int) -> Dict:
    """The row a stored outfit would land on right now, and how far back it is.

    This is the whole point of the dialog's warning: **storing does not create a
    change**. It writes the outfit into the row the position already resolves to -
    the last change at or before it, or the Base profile - and the caller says so
    before the value is stored. A new change point is a separate, explicit act.

    ``event_id`` is that row. ``at_position`` is true when it sits exactly here, in
    which case storing touches no other shot. Otherwise the caller shows the
    distance, because the outfit from that row onward is what changes.
    """
    with Session(engine) as session:
        char = session.get(Character, character_id)
        if char is None:
            return {"ok": False, "error": "character not found", "event_id": None}
        events, base, order, names, side = _ordered_events(session, char)
        target_order = order.get(book_id, 10 ** 6)
        best, best_key = None, None
        for event in events:
            if event.book_id is None:
                continue
            if event.book_id in side and event.book_id != book_id:
                continue
            order_no = order.get(event.book_id, 10 ** 6)
            event_chapter = event.chapter_num or 0
            event_shot = event.shot_num or 0
            if order_no < target_order:
                pass
            elif order_no == target_order:
                if event_chapter > chapter:
                    continue
                if event_chapter == chapter and event_shot > int(shot or 0):
                    continue
            else:
                continue
            key = (1, order_no, event_chapter, event_shot, event.id or 0)
            if best_key is None or key > best_key:
                best, best_key = event, key
        target = best or base
        common = {"ok": True, "error": "", "position": {
            "book_id": book_id, "chapter": int(chapter), "shot": int(shot or 0)}}
        if target is None:
            # No Base and no changes: the first outfit becomes the profile.
            return {**common, "event_id": None, "is_base": True, "at_position": False,
                    "has_change": False, "attire": "", "distance": {
                        "kind": "none", "books": 0, "chapters": 0, "shots": 0}}
        at_position = (target.book_id == book_id
                       and (target.chapter_num or 0) == int(chapter)
                       and (target.shot_num or 0) == int(shot or 0))
        if target.book_id is None:
            distance = {"kind": "base", "books": 0, "chapters": 0, "shots": 0}
        elif target.book_id == book_id:
            distance = {"kind": "same_book", "books": 0,
                        "chapters": int(chapter) - (target.chapter_num or 0),
                        "shots": int(shot or 0) - (target.shot_num or 0)}
        else:
            distance = {"kind": "earlier_book",
                        "books": target_order - order.get(target.book_id, target_order),
                        "chapters": 0, "shots": 0}
        return {
            **common,
            "event_id": target.id,
            "is_base": target.book_id is None,
            "at_position": at_position,
            "has_change": True,
            "book_id": target.book_id,
            "book_name": names.get(target.book_id, "the base profile"),
            "chapter": target.chapter_num,
            "shot": target.shot_num,
            "attire": (target.attire or "").strip(),
            "distance": distance,
        }


def create_attire_change(project_id: int, character_id: int, book_id: int,
                         chapter: int, shot: int) -> Dict:
    """Insert an explicit change point here, as a complete snapshot.

    This is the "+" button. Making the row explicit is the point: a change written
    because the user asked for one is data, while a row written because a search
    found a coat is clutter - and with events as complete snapshots, it is also a
    row that asserts a look for sixteen traits the search never looked at.
    """
    from services.character_manager import RENDERED_TRAIT_FIELDS

    result = {"created": False, "event_id": None, "error": ""}
    with Session(engine) as session:
        char = session.get(Character, character_id)
        if char is None:
            result["error"] = "character not found"
            return result
        existing = session.exec(
            select(CharacterTimelineEvent)
            .where(CharacterTimelineEvent.character_id == character_id)
            .where(CharacterTimelineEvent.book_id == book_id)
            .where(CharacterTimelineEvent.chapter_num == int(chapter))
            .where(CharacterTimelineEvent.shot_num == int(shot or 0))
        ).first()
        if existing is not None:
            result["event_id"] = existing.id
            return result
        previous = _snapshot_before(session, char, book_id, chapter, shot,
                                    exclude_id=None)
        event = CharacterTimelineEvent(
            character_id=character_id, book_id=book_id,
            chapter_num=int(chapter), shot_num=int(shot or 0),
            label="Attire", source="manual")
        # A change point starts as a copy of the state it inherits, outfit
        # included, so inserting one changes nothing until a value is stored.
        for field in RENDERED_TRAIT_FIELDS:
            if previous is not None:
                setattr(event, field, getattr(previous, field, None))
        session.add(event)
        session.commit()
        result.update(created=True, event_id=event.id)
    from services.character_manager import save_project_characters_to_json
    save_project_characters_to_json(project_id)
    return result


def _snapshot_before(session: Session, char: Character, book_id: int,
                     chapter: int, shot: int,
                     exclude_id: Optional[int]) -> Optional[CharacterTimelineEvent]:
    """The effective state just before a position, ignoring one row at it.

    Events are complete snapshots: whatever this returns has a value for every
    trait the character had at that point, which is exactly what a new row has to
    carry or it will blank everything it does not mention.
    """
    from services.character_manager import resolve_event_at

    query = select(CharacterTimelineEvent).where(
        CharacterTimelineEvent.character_id == char.id)
    if exclude_id is not None:
        query = query.where(CharacterTimelineEvent.id != exclude_id)
    previous, _base = resolve_event_at(
        session, char, book_id, int(chapter), int(shot or 0),
        events=list(session.exec(query).all()))
    return previous


def apply_attire(project_id: int, character_id: int, event_id: int,
                 attire: str) -> Dict:
    """Store one outfit on a specific row, and carry it forward like a hand edit.

    The row is chosen by the caller (see :func:`attire_target`), not invented here:
    a change point is data the user asked for, so it is created by an explicit act
    (:func:`create_attire_change`) and never as a side effect of storing a value.
    Writing into the row a position already resolves to is how "this outfit, from
    wherever that change starts" is expressed.

    A row this feature created before the snapshot fix is completed on the way: any
    rendered trait it left empty is filled from the state before it, because events
    are complete snapshots and an empty field asserts "nothing" from here on.

    Returns ``{"written": bool, "replaced": str, "events": int, "error": str}``.
    """
    from services.character_manager import (RENDERED_TRAIT_FIELDS,
                                            compile_character_visual_prompt,
                                            propagate_trait_change,
                                            save_project_characters_to_json)

    value = " ".join((attire or "").split()).strip(" .,;")
    result = {"written": False, "replaced": "", "events": 0, "error": ""}
    if not value:
        result["error"] = "nothing to store"
        return result
    with Session(engine) as session:
        char = session.get(Character, character_id)
        if char is None:
            result["error"] = "character not found"
            return result
        if char.locked:
            result["error"] = "this character is locked against automated changes"
            return result
        if event_id is None:
            from services.description_runner import ensure_base_event
            event = ensure_base_event(session, character_id)
        else:
            event = session.get(CharacterTimelineEvent, int(event_id))
        if event is None:
            result["error"] = "that timeline row no longer exists"
            return result
        previous = _snapshot_before(
            session, char,
            event.book_id if event.book_id is not None else None,
            event.chapter_num or 0, event.shot_num or 0,
            exclude_id=event.id)
        replaced = (event.attire or "").strip()
        if event.source == "extraction":
            # A re-analysis must not undo a value the user chose.
            event.source = "manual"
        if event.label == "Attire" and previous is not None:
            for field in RENDERED_TRAIT_FIELDS:
                if field == "attire" or (getattr(event, field, None) or "").strip():
                    continue
                earlier = getattr(previous, field, None)
                if (earlier or "").strip():
                    setattr(event, field, earlier)
        event.attire = value
        if event.visual_description:
            event.visual_description = compile_character_visual_prompt(
                event, name=char.name)
        session.add(event)
        result["events"] = propagate_trait_change(
            session, char, "attire", value, replaced or None,
            book_id=event.book_id, chapter_num=event.chapter_num or 0,
            shot_num=event.shot_num or 0)
        session.commit()
    save_project_characters_to_json(project_id)
    result["written"] = True
    result["replaced"] = replaced
    return result


def repair_partial_attire_events(*, project_id: Optional[int] = None,
                                 dry_run: bool = True) -> Dict:
    """Give rows this feature created the snapshot they should have had.

    ``apply_attire`` used to create an event containing only ``attire``. Events are
    complete snapshots, so that row asserted "nothing" for every other trait from
    its position onward, and the character looked wiped while the Base profile still
    held everything. This fills only the rendered fields that are empty in such a
    row and non-empty in the state just before it.

    It touches only rows labelled ``Attire`` - the exact signature this module
    writes - so a hand-made row that deliberately clears a field is left alone.
    ``dry_run`` reports without writing.
    """
    from services.character_manager import RENDERED_TRAIT_FIELDS

    report: Dict[str, Any] = {"rows": [], "repaired": 0, "dry_run": dry_run}
    with Session(engine) as session:
        events = list(session.exec(
            select(CharacterTimelineEvent)
            .where(CharacterTimelineEvent.label == "Attire")).all())
        for event in events:
            character = session.get(Character, event.character_id)
            if character is None:
                continue
            if project_id is not None and character.project_id != project_id:
                continue
            previous = _snapshot_before(session, character, event.book_id,
                                        event.chapter_num or 0, event.shot_num or 0,
                                        exclude_id=event.id)
            if previous is None:
                continue
            filled = {}
            for field in RENDERED_TRAIT_FIELDS:
                if field == "attire":
                    continue
                if (getattr(event, field, None) or "").strip():
                    continue
                earlier = getattr(previous, field, None)
                if (earlier or "").strip():
                    filled[field] = earlier
            if not filled:
                continue
            report["rows"].append({
                "character": character.name, "character_id": character.id,
                "book_id": event.book_id, "chapter": event.chapter_num,
                "shot": event.shot_num, "filled": filled,
            })
            if not dry_run:
                for field, earlier in filled.items():
                    setattr(event, field, earlier)
                session.add(event)
                report["repaired"] += 1
        if not dry_run and report["repaired"]:
            session.commit()
    return report
