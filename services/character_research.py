"""One character's look: researched from the books, and optionally deduced.

Two operations, because they are the same request with a different amount of
evidence tolerated:

* ``research`` reads the book's own text and reports only what it states. It is the
  direct ``book_lookup`` mode: every paragraph naming the character, no "who is
  described" pass. That extra pass is what keeps two principals in one scene apart,
  and it is also what a walk-on cannot afford - a minor character is usually
  described in the one paragraph that also names half the room.
* ``deduce`` runs that first, then invents the traits that are still empty, using
  the passages as evidence of role, age and bearing rather than as a description.
  That is the "keep them consistent" case: a character the book never describes
  still needs one face for thirty illustrations, and the same one every time.

Both write the **Base profile only**, through ``description_runner.apply_fields``, so
they follow every other import's rules: fill empty, never overwrite, report what was
kept. An invented value is recorded in ``notes`` - never rendered - because a look
nobody can trace back to the text must not be indistinguishable from a sourced one
six months later.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sqlmodel import Session, select

from database.connection import engine, get_setting
from database.models import Character, CharacterAlias, Project
from services import book_lookup, description_runner

PASSAGE_LIMIT = 5
PASSAGE_CHARS = 900

# Cues for *who this person is*, not for what they look like: a title, a job, an
# age word, or a line of dialogue. General English - a project's own honorifics and
# ranks belong in its text, not in this file.
_SAID = re.compile(
    r"\b(said|says|asked|answered|replied|told|muttered|murmured|whispered|"
    r"shouted|called|added|began|continued|snapped|breathed|offered|repeated|"
    r"exclaimed|insisted|agreed)\b", re.I)
_ADDRESS = re.compile(
    r"\b(mr|mrs|ms|miss|master|mistress|sir|madam|madame|lady|lord|king|queen|"
    r"prince|princess|captain|lieutenant|sergeant|general|admiral|doctor|"
    r"father|mother|brother|sister|elder|chief|sheriff|mayor|judge|preacher|"
    r"innkeeper|smith|cook|guard|soldier|servant|clerk|merchant|teacher)\b", re.I)
_AGE = re.compile(
    r"\b(young|younger|old|older|elderly|middle-aged|boy|girl|lads?|lasses?|"
    r"child|children|teen|teenage|youth|man|woman)\b", re.I)

_CAST_PROMPT = (
    "You are the casting director for a consistent illustrated adaptation of a "
    "novel. One character is in front of you and needs a definite, repeatable look.\n"
    "Character: {name}\n"
    "Also called: {aliases}\n"
    "Project: {project}\n"
    "How the text refers to them: {titles}\n"
    "Traits already decided (never change or contradict these):\n{known}\n\n"
    "Passages where the character appears:\n{passages}\n\n"
    "Task: decide every trait listed in the schema. This is a casting decision, not "
    "an extraction: a walk-on with two lines still needs one face, one head of hair "
    "and one build, because thirty illustrations have to agree with each other.\n"
    "Evidence to use: the words used for them above (a title, a rank, a job), how "
    "others treat them, what they do, their age cues, and the world the passages "
    "imply. Where a passage does state one of these traits, use its wording.\n"
    "Rules:\n"
    "- Do not repeat a decided trait, and do not contradict it.\n"
    "- Be concrete and specific, never a judgement: \"long narrow face, deeply lined\" "
    "rather than \"handsome\" or \"menacing\"; \"close-cropped grey\" rather than "
    "\"short hair\". A painter must be able to draw the value.\n"
    "- face means the shape of the face and its complexion, not a mood or an "
    "expression.\n"
    "- hair_color and eye_color are colour words. hair_style is a short phrase.\n"
    "- build is a frame and a height word: lean, stocky, broad-shouldered, tall, "
    "short.\n"
    "- Stay with an ordinary human unless a passage says otherwise: never invent a "
    "species, a power, or an accessory. \"species\" is the only field that may be "
    "null.\n"
    "- Permanent, paintable features only: no clothing, no props, no expression, no "
    "mood, nothing true only of this scene.\n"
    "- Short lowercase clauses, no name, no \"who is\", no final punctuation.\n"
    "- Answer every field. A blank or \"not specified\" is a failure.\n"
    'Reply with JSON only, no prose and no code fence.\n'
    "Schema:\n"
    '{{"sex": "male|female", "species": "noun or null", "age": "child|teen|young|'
    'adult|middle-aged|old", "build": "frame and height", "hair_color": "colour", '
    '"hair_style": "short phrase", "eye_color": "colour", "face": "shape and '
    'complexion"}}'
)


def _failure(message: str) -> Dict:
    return {"ok": False, "error": message, "filled": [], "conflicts": [],
            "dropped": [], "rewritten": [], "noted": [], "unchanged": [],
            "invented": [], "fields": {}, "used": [], "seconds": 0.0}


def _paragraph_score(text: str) -> int:
    """How much a paragraph says about *who someone is*.

    Weighted towards speech and address because that is where a role shows up
    ("Yes, Captain", "the innkeeper said"), and appearance words last: they help,
    but the pass this feeds is looking for a person, not a feature list.
    """
    score = 0
    if _SAID.search(text):
        score += 3
    if _ADDRESS.search(text):
        score += 2
    if _AGE.search(text):
        score += 1
    if book_lookup.CUE.search(text):
        score += 1
    return score


def passages_for(name: str, aliases: Sequence[str],
                 transcripts: Sequence[Tuple[str, str]],
                 limit: int = PASSAGE_LIMIT) -> List[Dict]:
    """Paragraphs that put the character in view, for context rather than description.

    The old profiler fed one "most visual" chunk, which for a minor character is
    usually an action scene they happen to stand in. This keeps the first mention -
    where an introduction lives - and then the highest-scoring paragraphs, and
    returns them in reading order so the model reads the character as the book
    reveals them.
    """
    patterns = [re.compile(r"\b" + re.escape(alias) + r"\b", re.I)
                for alias in dict.fromkeys([name, *(aliases or [])])
                if alias and alias.strip()]
    if not patterns:
        return []
    hits: List[Dict] = []
    names = list(dict.fromkeys([name, *(aliases or [])]))
    for book_name, transcript in transcripts:
        for paragraph in book_lookup.paragraphs_of(transcript):
            if not any(pattern.search(paragraph) for pattern in patterns):
                continue
            hits.append({"book": book_name,
                         "text": book_lookup.excerpt_for(paragraph, names,
                                                         PASSAGE_CHARS),
                         "score": _paragraph_score(paragraph)})
    if not hits:
        return []
    chosen = [0]
    for index in sorted(range(len(hits)), key=lambda i: -hits[i]["score"]):
        if len(chosen) >= limit:
            break
        if index not in chosen:
            chosen.append(index)
    return [hits[index] for index in sorted(chosen)]


def base_values(character_id: int) -> Dict[str, str]:
    """The Base profile's rendered values, for "what is still missing"."""
    values: Dict[str, str] = {}
    with Session(engine) as session:
        event = description_runner.ensure_base_event(session, character_id)
        for field in description_runner.APPLIED_FIELDS:
            values[field] = (getattr(event, field, None) or "").strip()
        values["notes"] = (event.notes or "").strip()
        session.rollback()
    return values


def apply_source_fields(project_id: int, character_id: int, fields: Dict[str, str]) -> Dict:
    """Write one set of fields to the Base event; return ``apply_fields``' outcome."""
    written = False
    with Session(engine) as session:
        event = description_runner.ensure_base_event(session, character_id)
        outcome = description_runner.apply_fields(event, fields)
        if outcome["filled"]:
            # A no-trait placeholder stamped at creation would keep hiding them.
            if description_runner.is_placeholder_description(event.visual_description):
                event.visual_description = None
            if event.source != "manual":
                event.source = "manual"  # a curated import stays put
            session.add(event)
            session.commit()
            written = True
        else:
            session.rollback()
    if written:
        from services.character_manager import save_project_characters_to_json
        save_project_characters_to_json(project_id)
    return outcome


def _titles_in(paragraphs: Sequence[Dict]) -> str:
    """The words the text uses for this person: titles, ranks, jobs.

    Pulled out of the passages and shown to the caster as evidence. "Knight
    Commander" is a rank and a bearing; "innkeeper" is a body and a trade. It is the
    cheapest grounding an invented look can get, and it is already in the text.
    """
    seen: List[str] = []
    for item in paragraphs:
        for match in _ADDRESS.finditer(item.get("text") or ""):
            word = match.group(0).lower()
            if word not in seen:
                seen.append(word)
    return ", ".join(seen[:8]) or "none given"


async def _deduce_fields(*, name: str, aliases: Sequence[str], project_name: str,
                         known: Dict[str, str], paragraphs: Sequence[Dict],
                         missing: Sequence[str], llm_url: str,
                         model: str) -> Dict[str, str]:
    """Cast a look for the missing traits. Returns normalised fields.

    Two attempts at most: a 12B asked for eight values will sometimes return two of
    them, and the second call says which ones were left out. The pass is only worth
    running if it produces a look rather than a fragment.
    """
    from services.prompt_engine import get_llm_response
    from services.web_lookup import sanitise

    known_text = "\n".join(f"- {field}: {value}"
                           for field, value in known.items() if value) or "None"
    passage_text = "\n\n".join(f"[{index + 1}] {item['book']}\n{item['text']}"
                               for index, item in enumerate(paragraphs))
    prompt = _CAST_PROMPT.format(
        name=name, aliases=", ".join(aliases) or name, project=project_name,
        titles=_titles_in(paragraphs), known=known_text, passages=passage_text)

    wanted = [field for field in missing]
    picked: Dict[str, str] = {}
    for attempt in range(2):
        ask = prompt if attempt == 0 else (
            prompt + "\n\nYou left out: " + ", ".join(
                field for field in wanted if field not in picked)
            + ". Answer again with JSON for every field, including those.")
        try:
            raw = await get_llm_response(ask, llm_url, model, temperature=0.5,
                                         timeout=180,
                                         extra_payload={"max_tokens": 400})
        except Exception:
            if picked:
                break
            raise
        data = book_lookup.json_object(raw)
        if isinstance(data.get("fields"), dict):
            data = data["fields"]
        for key, value in data.items():
            if (key in wanted and key not in picked
                    and value not in (None, "", "null", "None")):
                picked[key] = str(value)
        if len(picked) >= len(wanted) - 1:
            break
    return sanitise(picked)


def _merge(outcomes: Sequence[Dict]) -> Dict:
    merged = {"filled": [], "conflicts": [], "dropped": [], "rewritten": [],
              "noted": [], "unchanged": []}
    for outcome in outcomes:
        for key in merged:
            merged[key].extend(outcome.get(key) or [])
    return merged


async def research(project_id: int, character_id: int, *,
                   speculate: bool = False,
                   log: Optional[Callable[[str], None]] = None) -> Dict:
    """Research one character, and optionally deduce what the books leave out.

    ``speculate=False`` is the "Research (LLM)" button: facts only, nothing invented.
    ``speculate=True`` is "Deduce Vibe": the same factual pass, then a casting pass
    that fills the remaining traits from role evidence.

    Returns ``ok`` plus what happened, in the same shape the bulk runner reports:
    every value is filled, kept, rewritten or dropped, and ``invented`` names the
    fields the casting pass supplied.
    """
    llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
    model = get_setting("llm_model", "local-model")
    started = time.time()

    def say(message: str) -> None:
        if log:
            log(message)

    with Session(engine) as session:
        char = session.get(Character, character_id)
        project = session.get(Project, project_id)
        if not char or not project:
            return _failure("character not found")
        if char.locked:
            return _failure("this character is locked against automated changes")
        name = char.name
        aliases = [a.alias.strip() for a in session.exec(
            select(CharacterAlias)
            .where(CharacterAlias.character_id == character_id)).all()
            if (a.alias or "").strip()]
        # Other people's names, so a mention inside "Sir Canton Cass" is not read as
        # a mention of "Cass". The project's alias table already knows both.
        other_names = [row.name for row in session.exec(
            select(Character)
            .where(Character.project_id == project_id)
            .where(Character.id != character_id)).all()]
        other_names += [a.alias for a in session.exec(
            select(CharacterAlias)
            .join(Character, Character.id == CharacterAlias.character_id)
            .where(Character.project_id == project_id)
            .where(CharacterAlias.character_id != character_id)).all()
            if (a.alias or "").strip()]
        project_name = project.name

    transcripts = await asyncio.to_thread(description_runner.load_transcripts,
                                          project_id)
    if not transcripts:
        return _failure("no transcript.txt on disk for this project")

    say(f"[Research] {name}: {len(transcripts)} book(s)")
    found = await book_lookup.lookup(name, books=transcripts, aliases=aliases,
                                     llm_url=llm_url, model=model,
                                     attribute=False, other_names=other_names,
                                     log=log)
    fields = found.get("fields") or {}
    if fields:
        outcome = apply_source_fields(project_id, character_id, fields)
    else:
        outcome = {"filled": [], "conflicts": [], "dropped": [], "rewritten": [],
                   "noted": [], "unchanged": []}

    invented: List[str] = []
    invented_values: Dict[str, str] = {}
    invented_outcome = None
    reasons: List[str] = []
    if not fields:
        reasons.append(found.get("error") or "the text states nothing about them")
    if speculate:
        known = base_values(character_id)
        missing = [field for field in description_runner.APPLIED_FIELDS
                   if not known.get(field)]
        if not missing:
            reasons.append("every rendered trait is already set")
        else:
            paragraphs = await asyncio.to_thread(
                passages_for, name, aliases, transcripts)
            if not paragraphs:
                if not reasons:
                    reasons.append("no passage names this character, so there is "
                                   "nothing to deduce from")
            else:
                say(f"[Vibe] {name}: {len(paragraphs)} passages, "
                    f"{len(missing)} trait(s) missing")
                try:
                    guessed = await _deduce_fields(
                        name=name, aliases=[name, *aliases],
                        project_name=project_name, known=known,
                        paragraphs=paragraphs, missing=missing,
                        llm_url=llm_url, model=model)
                except Exception as ex:
                    guessed = {}
                    reasons.append(f"the casting call failed: {ex}")
                    say(f"[Vibe] {name}: casting call failed: {ex}")
                if guessed:
                    invented_outcome = apply_source_fields(project_id, character_id,
                                                           guessed)
                    invented = list(invented_outcome["filled"])
                    invented_values = {field: guessed[field] for field in invented
                                       if field in guessed}
                    if invented:
                        # Provenance, in the one field that is stored and never
                        # rendered: an invented value must not look sourced.
                        apply_source_fields(project_id, character_id, {
                            "notes": "look deduced, not stated in the text: "
                                     + ", ".join(sorted(invented))})
                elif not reasons:
                    reasons.append("the casting pass had nothing to add")

    merged = _merge([outcome] + ([invented_outcome] if invented_outcome else []))
    ok = bool(merged["filled"])
    if not ok and not reasons:
        reasons.append("nothing new: every value this source could supply is "
                       "already set")
    return {
        "ok": ok,
        "error": "" if ok else "; ".join(reasons),
        "fields": fields,
        "used": found.get("used") or [],
        "invented": sorted(invented),
        "invented_values": invented_values,
        **merged,
        "seconds": round(time.time() - started, 1),
    }
