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
  What the passages cannot settle is decided from ``_STARTERS``, a general palette
  of single features picked per character, because a model asked to invent a face
  from nothing answers with the same face for everybody in the book.

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
import zlib
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sqlmodel import Session, select

from database.connection import engine, get_setting
from database.models import (
    Character, CharacterAlias, CharacterTimelineEvent, Project)
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
    "Traits already decided:\n{known}\n"
    "Looks already cast for the rest of the cast:\n{used}\n"
    "Starting points for the traits this book leaves open:\n{starters}\n\n"
    "Passages where the character appears:\n{passages}\n\n"
    "Task: decide every trait listed in the schema. This is a casting decision, not "
    "an extraction: a walk-on with two lines still needs one face, one head of hair "
    "and one build, because thirty illustrations have to agree with each other.\n"
    "Evidence to use: the words used for them above (a title, a rank, a job), how "
    "others treat them, what they do, their age cues, and the world the passages "
    "imply. Where a passage does state one of these traits, use its wording.\n"
    "Rules:\n"
    "- A trait already decided above is context, not a binding answer. Keep it "
    "unless the evidence clearly supports a different value, and then state the "
    "different value: the user is shown both and chooses.\n"
    "- A starting point is where the look begins, not the answer. Keep the shape it "
    "describes and bring it into line with this character: the same cut gone grey "
    "and thin, the same face worn by more years, a feature that sits too heavily on "
    "a young face made lighter. Where a passage states the trait itself, the "
    "passage's wording wins over the starting point.\n"
    "- A look already cast belongs to somebody else and is not available: two people "
    "in one drawing have to be tellable apart. This holds even where the text says "
    "nothing about this character, and a value that merely rounds out the schema "
    "does not count as a choice.\n"
    "- Choose every value for this character, not for people in general. Before you "
    "answer, read each value back and ask whether it would equally describe a dozen "
    "other people in this book; if it would, it is a placeholder and not a casting "
    "decision, so replace it with the detail that follows from their age, their work "
    "and their world. Be concrete: a painter must be able to draw it, and a "
    "judgement such as \"handsome\" cannot be drawn.\n"
    "- face is ONE feature of the head - a shape, a feature, a complexion or a mark. "
    "One, not a pair, and not a mood or an expression.\n"
    "- hair_color and eye_color are colour words. hair_style says how the hair is "
    "worn or cut, and what holds it there.\n"
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
    '"hair_style": "how it is worn or cut", "eye_color": "colour", "face": "one '
    'feature of the head"}}'
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
    patterns = book_lookup.name_patterns(
        list(dict.fromkeys([name, *(aliases or [])])))
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


# Starting points for the two traits nothing in a book ever settles for a
# walk-on: the shape of a face and the way hair is worn. The caster is given one
# of each and builds the look around it.
#
# These exist because the model cannot invent variety it does not have. Asked for
# a face from nothing it returns the same face for every woman in the project
# ("oval face, pale complexion") and the same head of hair ("pulled back in a
# tight knot"), and no wording fixes that: the question has one modal answer, and
# asking again only walks to the next most likely one. Given a concrete starting
# point, the same model writes high cheekbones for one woman and a square jaw for
# the next.
#
# General English, and deliberately sex-neutral and age-neutral: which of these a
# character gets is decided by their own id, never by a rule about how men,
# women, young people or old people look. Books do not agree on that, and neither
# should a guess. Each entry is ONE feature, for the same reason the field asks
# for one: a pair of adjectives ("heavy jowls", "a receding hairline") carries an
# age with it, and a 19-year-old handed one keeps it, because the model has no
# reason to overrule the starting point it was given. Nothing here is a colour, a
# prop or a style of dress - those are separate fields or separate features.
_FACE_STARTERS = (
    "a long narrow face",
    "a broad flat face",
    "a square jaw",
    "a narrow chin",
    "a pointed chin",
    "a jutting chin",
    "high cheekbones",
    "full cheeks",
    "a round face",
    "a wide mouth",
    "a small mouth",
    "thin lips",
    "a long upper lip",
    "a sharp nose",
    "a hooked nose",
    "a snub nose",
    "a flat nose",
    "a broad nose",
    "heavy brows",
    "deep-set eyes",
    "close-set eyes",
    "a high forehead",
    "a heavy brow ridge",
    "sunken eyes",
)
_HAIR_STARTERS = (
    "cropped close to the head",
    "cut level with the jaw",
    "swept back from the face",
    "loose and tucked behind the ears",
    "parted in the middle and worn loose",
    "gathered at the crown",
    "close-cut at the sides and longer on top",
    "curling loose about the shoulders",
    "trimmed short at the neck",
    "in a knot at the nape",
    "in a single plait over one shoulder",
    "cut in a fringe across the brow",
    "parted on the side and combed flat",
    "combed straight back",
    "long and bound at the base of the neck",
    "in rows of tight braids",
    "short and brushed forward",
    "parted low and smoothed flat",
    "coiled over the ears",
    "shorn close at the sides",
    "brushed back from a high hairline",
    "in a thick braid down the back",
    "loose and tangled",
    "swept across the brow",
)
_STARTERS = {"face": _FACE_STARTERS, "hair_style": _HAIR_STARTERS}

# The fields a reader uses to tell two people apart in one drawing. Only these:
# `build` and the colour fields legitimately repeat across a cast, and telling a
# 300-character project that "brown" is taken would send it hunting for a colour
# nobody has, which is not a look, it is a gimmick.
_CAST_FIELDS = ("face", "hair_style")

# Per field, most-common first. A longer list buys nothing: the values that
# actually collide are the frequent ones, and they are exactly the ones a caster
# with no list to work from keeps handing out.
_CAST_LIMIT = 10
# Words that carry no look, so two phrasings of one appearance compare equal:
# "oval face with fair complexion" and "oval face and fair complexion" are the
# same face, and both are already taken.
_LOOK_STOPWORDS = {"a", "an", "and", "the", "with", "of", "in", "on", "to",
                   "his", "her", "their", "very", "is"}


def _look_words(value: str) -> frozenset:
    words = re.findall(r"[a-z]+", (value or "").lower())
    return frozenset(word for word in words if word not in _LOOK_STOPWORDS)


def same_look(first: str, second: str) -> bool:
    """Whether two trait values describe the same look in different words.

    Exact matching is useless here: a model that has decided on "pulled back in a
    tight knot" will write "pulled back in a practical knot" next time and call it
    a different hairstyle. Two content words in common, making up most of the
    shorter phrase, is the same look reworded.
    """
    left, right = _look_words(first), _look_words(second)
    if not left or not right:
        return False
    shared = len(left & right)
    return shared >= 2 and shared >= 0.7 * min(len(left), len(right))


def cast_context(project_id: int, character_id: int) -> Tuple[str, Dict[str, set]]:
    """The Base looks already assigned to the rest of this project's cast.

    Returns the block shown to the caster, and the same values as word sets for
    the duplicate check afterwards - the block is capped so the prompt stays
    short, but a look that is in use is in use whether or not it was printed.

    A caster that invents a face in isolation returns the model's default face for
    that demographic, on every call: a whole project of "oval face, smooth
    complexion", or of "pulled back in a tight knot" for every woman. No wording
    in the prompt fixes it, because there is nothing for the answer to be
    different *from* - the model is being asked to choose, with its own mode as
    the only guide. This is the missing half: the looks a reader has already been
    given, most common first. Shown them, the same model stops reaching for its
    default, and ``starting_points`` gives it somewhere else to land.
    """
    counts: Dict[str, Counter] = {field: Counter() for field in _CAST_FIELDS}
    with Session(engine) as session:
        events = session.exec(
            select(CharacterTimelineEvent)
            .join(Character, Character.id == CharacterTimelineEvent.character_id)
            .where(Character.project_id == project_id)
            .where(CharacterTimelineEvent.character_id != character_id)
            .where(CharacterTimelineEvent.book_id == None)  # noqa: E711
        ).all()
    for event in events:
        for field in _CAST_FIELDS:
            value = " ".join((getattr(event, field, None) or "").split()).lower()
            if value:
                counts[field][value] += 1
    taken = {field: set(counts[field]) for field in _CAST_FIELDS}
    lines = [f"- {field}: " + "; ".join(
        value for value, _ in counts[field].most_common(_CAST_LIMIT))
        for field in _CAST_FIELDS if counts[field]]
    return ("\n".join(lines) or "none yet - this is the first look in the project",
            taken)


def starting_points(character_id: int, missing: Sequence[str],
                    taken: Optional[Dict[str, set]] = None) -> str:
    """One concrete starting look per trait the book leaves open.

    The character's own id picks where in the list to start, so the same character
    gets the same starting point on every run - a guess that changed each time
    would be useless for thirty illustrations. It then walks forward to the first
    entry no one else in the project already has, so two characters handed the
    same starting point is the exception rather than the norm.

    Only traits in ``missing`` are seeded: a trait the text states, or that the
    user has already decided, is not up for suggestion.
    """
    taken = taken or {}
    lines: List[str] = []
    for field, pool in _STARTERS.items():
        if field not in missing or not pool:
            continue
        start = zlib.crc32(f"{character_id}:{field}".encode("utf-8")) % len(pool)
        chosen = pool[start]
        for offset in range(len(pool)):
            candidate = pool[(start + offset) % len(pool)]
            if not any(same_look(candidate, other) for other in taken.get(field, ())):
                chosen = candidate
                break
        lines.append(f"- {field}: {chosen}")
    return "\n".join(lines) or "none - the text settles every trait asked for"


def apply_source_fields(project_id: int, character_id: int, fields: Dict[str, str],
                        *, overwrite: Sequence[str] = ()) -> Dict:
    """Write one set of fields to the Base event; return ``apply_fields``' outcome.

    ``overwrite`` names the fields the user chose to replace. Only the dialog's
    checkbox list passes it, and only for a row that showed the existing value, so
    the "fill empty" rule is not weakened for any other caller. A replacement is
    carried forward with ``propagate_trait_change``: later events hold complete
    snapshots, so without it a book event that repeated the old value keeps
    asserting it and the user's choice never reaches a rendered prompt.
    """
    written = False
    with Session(engine) as session:
        event = description_runner.ensure_base_event(session, character_id)
        char = session.get(Character, character_id)
        outcome = description_runner.apply_fields(
            event, fields, overwrite=bool(overwrite))
        if outcome["filled"]:
            # A no-trait placeholder stamped at creation would keep hiding them.
            if description_runner.is_placeholder_description(event.visual_description):
                event.visual_description = None
            if event.source != "manual":
                event.source = "manual"  # a curated import stays put
            session.add(event)
            if outcome.get("replaced") and char is not None and not char.locked:
                from services.character_manager import propagate_trait_change
                for change in outcome["replaced"]:
                    propagate_trait_change(
                        session, char, change["field"], change["now"],
                        change["was"], book_id=None)
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
                         missing: Sequence[str], used: str, starters: str = "",
                         taken: Optional[Dict[str, set]] = None, llm_url: str = "",
                         model: str = "") -> Dict[str, str]:
    """Cast a look for the missing traits. Returns normalised fields.

    Two attempts at most, and the second is not just "you left something out". A
    12B asked for eight values will sometimes return two of them, and a value it
    has already seen in the taken list can come back anyway - the instruction is
    a steer, not a guarantee, and on a name the model already knows it will
    overrule it. Naming the collision in the retry is what makes it let go, and
    only one field is worth doing that for: measurements and colours repeat
    across a cast, faces and heads of hair do not.
    """
    from services.prompt_engine import get_llm_response
    from services.web_lookup import sanitise

    taken = taken or {}
    known_text = "\n".join(f"- {field}: {value}"
                           for field, value in known.items() if value) or "None"
    passage_text = "\n\n".join(f"[{index + 1}] {item['book']}\n{item['text']}"
                               for index, item in enumerate(paragraphs))
    prompt = _CAST_PROMPT.format(
        name=name, aliases=", ".join(aliases) or name, project=project_name,
        titles=_titles_in(paragraphs), known=known_text, used=used,
        starters=starters or "none - the text settles every trait asked for",
        passages=passage_text)

    wanted = [field for field in missing]
    picked: Dict[str, str] = {}
    reused: List[str] = []
    for attempt in range(2):
        ask = prompt
        if attempt:
            notes: List[str] = []
            absent = [field for field in wanted
                      if field not in picked and field not in reused]
            if absent:
                notes.append("You left out: " + ", ".join(absent) + ".")
            if reused:
                notes.append("Already cast for somebody else in this project, so "
                             "not available: " + ", ".join(
                                 f'{field} "{picked[field]}"' for field in reused)
                             + ".")
            notes.append("Answer again with JSON for every field.")
            ask = prompt + "\n\n" + " ".join(notes)
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
            if (key in wanted and (key not in picked or key in reused)
                    and value not in (None, "", "null", "None")):
                picked[key] = str(value)
        reused = [field for field in _CAST_FIELDS
                  if field in picked and any(
                      same_look(picked[field], other)
                      for other in taken.get(field, ()))]
        if len(picked) >= len(wanted) - 1 and not reused:
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
                   write: bool = True,
                   log: Optional[Callable[[str], None]] = None) -> Dict:
    """Research one character, and optionally deduce what the books leave out.

    ``speculate=False`` is the dialog's Book text tab: facts only, nothing invented.
    ``speculate=True`` is "Best guess": the same factual pass, then a casting pass
    that fills the remaining traits from role evidence.

    ``write=False`` is the dialog's normal mode now: it returns the same fields
    without touching the profile, because the user picks what to keep. The values
    are still normalised where they are written, not here - ``preview_fields``
    shows the stored shape, and applying it goes through the same code either way.

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
    if fields and write:
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
        # When the caller writes, only the empty traits are asked for: there is
        # nowhere to put a replacement. When the caller collects a choice, a trait
        # the source did not supply is worth a proposal even where a value exists -
        # an existing value the search disagrees with is exactly the case the user
        # wants to see side by side.
        missing = [field for field in description_runner.APPLIED_FIELDS
                   if not known.get(field)
                   or (not write and field not in fields)]
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
                say(f"[Guess] {name}: {len(paragraphs)} passages, "
                    f"{len(missing)} trait(s) to cast")
                try:
                    used, taken = await asyncio.to_thread(
                        cast_context, project_id, character_id)
                    # Only the traits with nothing in them: a starting point beside
                    # a value that is already decided invites the model to blend the
                    # two ("short and twisted into a knot"), which is worse than
                    # either. A decided trait is already a look of its own.
                    empty = [field for field in missing if not known.get(field)]
                    starters = await asyncio.to_thread(
                        starting_points, character_id, empty, taken)
                    guessed = await _deduce_fields(
                        name=name, aliases=[name, *aliases],
                        project_name=project_name, known=known,
                        paragraphs=paragraphs, missing=missing, used=used,
                        starters=starters, taken=taken, llm_url=llm_url,
                        model=model)
                except Exception as ex:
                    guessed = {}
                    reasons.append(f"the casting call failed: {ex}")
                    say(f"[Guess] {name}: casting call failed: {ex}")
                if guessed and write:
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
                elif guessed:
                    # Collected, not written: the whole set is a proposal, and a
                    # proposal that contradicts an existing value is still a
                    # proposal the user can accept. The notes line only makes
                    # sense once something is stored, so the dialog adds it to the
                    # applied set instead.
                    invented_values = dict(guessed)
                elif not reasons:
                    reasons.append("the casting pass had nothing to add")

    merged = _merge([outcome] + ([invented_outcome] if invented_outcome else []))
    ok = bool(merged["filled"]) or bool(write is False and (fields or invented_values))
    if not ok and not reasons:
        reasons.append("nothing new: every value this source could supply is "
                       "already set")
    return {
        "ok": ok,
        "error": "" if ok else "; ".join(reasons),
        "fields": fields,
        "used": found.get("used") or [],
        "invented": sorted(invented) if write else sorted(invented_values),
        "invented_values": invented_values,
        "written": bool(write),
        **merged,
        "seconds": round(time.time() - started, 1),
    }
