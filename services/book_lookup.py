"""Last-resort description from a book's own text.

The old approach - a window around every mention of the name - fails because the
sentence that describes someone usually has no name in it ("Rand's gaze fell to
the woman who had spoken... her eyes met his"), while the windows it does return
are dominated by dialogue and action. Measured on the Eye of the World: 24% of
the selected text contained no appearance word at all, and for a principal it
found one of six known details in 20,000 characters.

So the work is split by what each half is good at:

  1. code narrows - paragraphs that contain BOTH an appearance word and the
     character's name. No pronoun heuristics: they cannot be made reliable.
  2. the model attributes - "who is being described here?" over that short list.
     Its answers are a reviewable artefact, which is what a proximity rule never
     was.
  3. the model extracts - the eight rendered traits, from the paragraphs it
     attributed to this character only.

That is the ``attribute=True`` path, and it is the right one when there is enough
text to get wrong. ``attribute=False`` is the opposite trade for a walk-on: every
paragraph that names them, no attribution pass, one yes/no relevance question and
then one extraction call. It reads a description phrased without any cue word
("wide as a door"), which the shortlist above cannot see, and it never needs the
model to name a *different* character to decide a passage is useless.

`face` is not requested: it is a manual field, and prose descriptions of it are
transient expressions ("cheeks turning pink") rather than a look.

This is deliberately the last source in the queue. A reference volume or a wiki
is cheaper, more accurate, and cannot put one character's description on another.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from services import trait_vocab

MAX_PARAGRAPHS = 14
MAX_PARAGRAPH_CHARS = 900
# How much of the window sits before the mention. The sentence that introduces
# someone usually starts before their name, so a window that begins *on* the name
# loses half of it.
EXCERPT_CONTEXT = 300

# Appearance vocabulary, deliberately general English: nothing here is series
# specific, and an over-broad word only costs a paragraph slot.
CUE = re.compile(
    r"\b(hair|eyes?|eyed|tall|short|lean|wiry|beard|bearded|moustache|mustache|"
    r"scarred|scar|scars|skin|complexion|face|faced|features|shoulders?|slender|"
    r"stocky|plump|bald|handsome|pretty|beautiful|cheekbones|nose|jaw|wrists?|"
    r"fingers?|brows?|broad|thin|fat|heavy|muscular|older|younger)\b", re.I)

EXTRACT_FIELDS = ("sex", "species", "age", "height", "build", "hair_color",
                  "hair_style", "eye_color")

_WHO_PROMPT = (
    "For each numbered paragraph, name the character whose physical appearance is "
    "being described. Use the name the paragraph uses. If no one's appearance is "
    "described, answer NONE.\n"
    'Reply with JSON only, no prose: {"0": "Name", "1": "NONE", ...}'
)

_DIRECT_PROMPT = (
    "You are reading passages from a novel to describe ONE character's physical "
    "appearance for an illustration pipeline.\n"
    "Character: {name}\n"
    "Also called: {aliases}\n"
    "Rules:\n"
    "- Report only what these passages state about {name}. A passage may describe "
    "someone else - another character in the same scene - and those details are not "
    "theirs. Ignore them.\n"
    "- A description can be phrased without any obvious appearance word (\"wide as a "
    "door\", \"the colour of wet slate\"). Size, shape, colour and texture all count.\n"
    "- Never guess and never add what is typical for the setting.\n"
    "- Null for every field the passages do not state.\n"
    'Reply with JSON only, no prose and no code fence.\n'
    'Schema: {{"sex","species","age","height","build","hair_color","hair_style",'
    '"eye_color"}} - null for anything not stated.\n'
    "Shapes: sex = male|female. species = lowercase noun, null for an ordinary "
    "human. age = child|teen|young|adult|middle-aged|old. height = as the text says "
    "it, including comparisons. build = adjectives only. hair_color/eye_color = "
    "colour words. hair_style = a short phrase."
)

_EXTRACT_PROMPT = (
    "Extract only what this text states about physical appearance. Never guess and "
    "never add what is typical.\n"
    'Reply with JSON only: {"sex","species","age","height","build","hair_color",'
    '"hair_style","eye_color"} - null for anything not stated. No prose, no fence.\n'
    "Shapes: sex = male|female. species = lowercase noun, null for a human. "
    "age = child|teen|young|adult|middle-aged|old. height = as the text says it, including "
    "comparisons (\"half a head shorter than Rand\"). build = adjectives only. "
    "hair_color/eye_color = colour words. hair_style = a short phrase."
)


_GATE_PROMPT = (
    "For each numbered passage, decide whether it describes the physical "
    "appearance of {name} - their body, face, hair, eyes, colour, size or age.\n"
    "Answer NO when the passage describes someone else, or says nothing physical, "
    "even if the name appears in it.\n"
    "Numbering starts at 0. Reply with JSON only, no prose and no code fence: "
    '{{"0": "YES", "1": "NO"}}'
)


def _verdict(value) -> Optional[bool]:
    """A yes/no answer as a bool, or ``None`` when it is neither.

    A model that was not held to JSON mode may answer ``true``/``false``, or write
    "YES - his face is described". An answer that cannot be read is not a "no":
    treating it as one silently discards every paragraph.
    """
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text.startswith(("yes", "true", "y ")) or text in ("y", "y."):
        return True
    if text.startswith(("no", "false", "n ")) or text in ("n", "n."):
        return False
    return None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def json_object(raw: str) -> Dict:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def paragraphs_of(transcript: str) -> List[str]:
    text = (transcript or "").replace("\r\n", "\n")
    return [_clean(p) for p in re.split(r"\n\s*\n", text) if p.strip()]


def _name_patterns(aliases: Sequence[str]) -> List["re.Pattern"]:
    return [re.compile(r"\b" + re.escape(a) + r"\b", re.I)
            for a in aliases if a.strip()]


def _first_mention(paragraph: str, patterns: Sequence["re.Pattern"],
                   shadows: Optional["re.Pattern"] = None) -> Optional[int]:
    """Position of the earliest mention that is not inside another person's name."""
    covered: List[Tuple[int, int]] = []
    if shadows is not None:
        covered = [(span.start(), span.end()) for span in shadows.finditer(paragraph)]
    best = None
    for pattern in patterns:
        for match in pattern.finditer(paragraph):
            if any(start <= match.start() and match.end() <= end
                   for start, end in covered):
                continue
            if best is None or match.start() < best:
                best = match.start()
            break
    return best


def excerpt(paragraph: str, at: Optional[int],
            limit: int = MAX_PARAGRAPH_CHARS) -> str:
    """A window around a mention, not the head of the paragraph.

    A transcript with no blank lines puts a whole chapter in one paragraph, and
    "the first 900 characters" of a chapter is the weather, not the character. The
    window starts a little before the mention so the surrounding sentence survives.
    """
    if at is None or len(paragraph) <= limit:
        return paragraph[:limit]
    start = max(0, at - EXCERPT_CONTEXT)
    end = min(len(paragraph), start + limit)
    start = max(0, end - limit)
    if start:
        space = paragraph.find(" ", start)
        if -1 < space < at:
            start = space + 1
    if end < len(paragraph):
        space = paragraph.rfind(" ", at, end)
        if space != -1:
            end = space
    return paragraph[start:end]


def excerpt_for(paragraph: str, names: Sequence[str],
                limit: int = MAX_PARAGRAPH_CHARS) -> str:
    """``excerpt`` for callers that only have a paragraph and some names."""
    first = _first_mention(paragraph, _name_patterns(names))
    return excerpt(paragraph, first, limit)


def cue_paragraphs(books: Sequence[Tuple[str, str]], aliases: Sequence[str],
                   limit: int = MAX_PARAGRAPHS) -> List[Dict]:
    """Paragraphs naming the character that also carry appearance words.

    ``books`` is ``[(book_name, transcript_text)]`` in reading order. Reading
    order matters: the earliest mention is where an introduction usually lives.

    This is the *attributed* mode's candidate list. Requiring a cue word is what
    makes the attribution pass cheap, and it is also what loses a description
    phrased without one - see ``mention_paragraphs``.
    """
    patterns = _name_patterns(aliases)
    if not patterns:
        return []
    found: List[Dict] = []
    for book_name, transcript in books:
        for index, paragraph in enumerate(paragraphs_of(transcript)):
            if not CUE.search(paragraph):
                continue
            first = _first_mention(paragraph, patterns)
            if first is None:
                continue
            found.append({"book": book_name, "index": index,
                          "text": excerpt(paragraph, first)})
            if len(found) >= limit:
                return found
    return found


def _shadow_pattern(aliases: Sequence[str],
                    other_names: Sequence[str]) -> Optional["re.Pattern"]:
    """One regex over the *other* characters' names that contain this one's.

    A shared surname is the commonest way a direct lookup goes wrong: a paragraph
    about "Sir Canton Cass" contains "Cass", so it looks like a mention of Knight
    Commander Cass and its description lands on the wrong profile. The project's
    own alias table knows both names, so the fix needs no vocabulary: a match of
    this character's name *inside* another character's longer name is not a mention.
    """
    own = [alias for alias in aliases if alias.strip()]
    shadows = set()
    for name in other_names:
        name = (name or "").strip()
        if not name:
            continue
        lowered = name.lower()
        for alias in own:
            if (len(alias) < len(name)
                    and re.search(r"\b" + re.escape(alias.lower()) + r"\b", lowered)):
                shadows.add(name)
                break
    if not shadows:
        return None
    ordered = sorted(shadows, key=len, reverse=True)
    return re.compile("|".join(r"\b" + re.escape(s) + r"\b" for s in ordered), re.I)


def mention_paragraphs(books: Sequence[Tuple[str, str]], aliases: Sequence[str],
                       limit: int = MAX_PARAGRAPHS,
                       other_names: Sequence[str] = ()) -> List[Dict]:
    """Every paragraph that names the character, cue-bearing ones first.

    The direct mode has no attribution pass, so this list *is* the evidence handed
    to the extraction model. A paragraph that names the character but contains no
    word from ``CUE`` is kept: "he was wide as a door", "a handspan taller than his
    brother", "the colour of wet slate" are descriptions, and a vocabulary list is
    a budget device, not a definition of one. The cue-bearing paragraphs go first
    because they are the likelier half when the list has to be truncated.

    ``other_names`` is every other character's name and alias in the project. It is
    used to reject a mention that only appears inside someone else's longer name,
    which is the one mis-attribution code can see without asking a model.
    """
    patterns = _name_patterns(aliases)
    if not patterns:
        return []
    shadows = _shadow_pattern(aliases, other_names)
    with_cue: List[Dict] = []
    without_cue: List[Dict] = []
    for book_name, transcript in books:
        for index, paragraph in enumerate(paragraphs_of(transcript)):
            first = _first_mention(paragraph, patterns, shadows)
            if first is None:
                continue
            entry = {"book": book_name, "index": index,
                     "text": excerpt(paragraph, first)}
            (with_cue if CUE.search(paragraph) else without_cue).append(entry)
    return (with_cue + without_cue)[:limit]


def _mentions(answer: str, aliases: Sequence[str]) -> bool:
    """Does a 'who is described' answer name this character?

    Word boundaries rather than a length heuristic. The first version required a
    first token of four characters or more, which quietly made every three-letter
    name in a project unattributable - "Tam", "Mat", "Min", "Lan" all came back as
    "none of the candidate paragraphs describe this character" while the model had
    named them correctly. ``\\b`` is what keeps "Mat" out of "Math" without
    refusing the name itself.
    """
    text = _clean(answer)
    if not text or text.lower().startswith(("none", "no one", "nobody")):
        return False
    for alias in aliases:
        alias = (alias or "").strip()
        if alias and re.search(r"\b" + re.escape(alias) + r"\b", text, re.I):
            return True
    return False


async def lookup(
    name: str,
    *,
    books: Sequence[Tuple[str, str]],
    aliases: Optional[Sequence[str]] = None,
    llm_url: str,
    model: str,
    limit: int = MAX_PARAGRAPHS,
    attribute: bool = True,
    gate: bool = True,
    other_names: Sequence[str] = (),
    should_cancel: Optional[Callable[[], bool]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Research one character from the book text. Never raises; ``ok`` says it.

    ``attribute=True`` is the two-stage path: code shortlists paragraphs naming the
    character, the model says who each one describes, and only the ones it gives to
    this character are extracted. It costs one extra call and it is the right answer
    for a character who has plenty of text to get wrong.

    ``attribute=False`` is the direct path for a character with almost nothing: take
    every paragraph that names them, drop the ones that only name them inside
    someone else's longer name (the project's alias table knows those), then ask one
    yes/no question per passage - "does this describe <name>'s appearance?" - and
    extract from the passages that pass. The gate is not attribution: it never needs
    another character's name, which is the part that fails on a walk-on. Set
    ``gate=False`` to skip even that and extract from the passages as they are.

    Returns ``fields`` (eight rendered traits, height folded into ``build``),
    the paragraphs it used, and the attribution map, so a reviewer can see why
    a value was accepted.
    """
    from services.prompt_engine import get_llm_response
    from services.web_lookup import sanitise

    aliases = list(dict.fromkeys([name, *(aliases or [])]))
    started = time.time()

    def say(message: str) -> None:
        if log:
            log(message)

    if attribute:
        candidates = cue_paragraphs(books, aliases, limit=limit)
        empty_error = ("no paragraph in this book mentions the character alongside "
                       "an appearance word")
    else:
        candidates = mention_paragraphs(books, aliases, limit=limit,
                                        other_names=other_names)
        empty_error = "no paragraph in these books names the character"
    if not candidates:
        return {"ok": False, "error": empty_error, "fields": {}, "used": [],
                "who": {}, "seconds": 0.0}

    listing = "\n\n".join(f"[{i}] {c['text']}"
                          for i, c in enumerate(candidates))
    say(f"[Book] {name}: {len(candidates)} candidate paragraphs"
        + ("" if attribute else " (direct, no attribution pass)"))

    if should_cancel and should_cancel():
        return {"ok": False, "error": "cancelled", "fields": {}, "used": [],
                "who": {}, "seconds": round(time.time() - started, 1)}

    who: Dict = {}
    if attribute:
        try:
            who_raw = await get_llm_response(
                f"{_WHO_PROMPT}\n\n{listing}", llm_url, model,
                temperature=0.0, timeout=180,
                extra_payload={"max_tokens": 400})
        except Exception as ex:
            return {"ok": False, "error": f"attribution call failed: {ex}",
                    "fields": {}, "used": [], "who": {}, "seconds": 0.0}

        who = json_object(who_raw)
        chosen = []
        for index, answer in who.items():
            try:
                position = int(index)
            except (TypeError, ValueError):
                continue
            if 0 <= position < len(candidates) and _mentions(str(answer), aliases):
                chosen.append(position)
        chosen.sort()
        say(f"[Book] {name}: model attributed {len(chosen)} of {len(candidates)} "
            f"to them")
        if not chosen:
            return {"ok": False,
                    "error": "none of the candidate paragraphs describe this character",
                    "fields": {}, "used": [], "who": who,
                    "seconds": round(time.time() - started, 1)}
        used = [candidates[i] for i in chosen]
        body = "\n\n".join(c["text"] for c in used)
        prompt = f"{_EXTRACT_PROMPT}\n\nCharacter: {name}\n\n{body}"
    else:
        # The gate is the direct mode's only filter: a yes/no per passage, no name
        # to produce and no third party needed. It exists because a paragraph that
        # names the character is often describing the person next to them, and that
        # is how a walk-on acquires someone else's hair.
        if gate:
            try:
                gate_raw = await get_llm_response(
                    f"{_GATE_PROMPT.format(name=name)}\n\n{listing}",
                    llm_url, model, temperature=0.0, timeout=180,
                    extra_payload={"max_tokens": 300})
                verdicts = json_object(gate_raw)
            except Exception as ex:
                return {"ok": False, "error": f"relevance check failed: {ex}",
                        "fields": {}, "used": [], "who": {}, "seconds": 0.0}
            keep, read_any, refused = [], False, False
            for index, answer in verdicts.items():
                try:
                    position = int(index)
                except (TypeError, ValueError):
                    continue
                if not 0 <= position < len(candidates):
                    continue
                verdict = _verdict(answer)
                if verdict is None:
                    continue
                read_any = True
                if verdict:
                    keep.append(position)
                else:
                    refused = True
            keep.sort()
            if not read_any:
                # The answer could not be read (wrong shape, prose, booleans the
                # model declined). Fail open: extracting with the entity shield from
                # every passage beats treating an unreadable answer as "no".
                say(f"[Book] {name}: relevance check unreadable, using every passage")
            else:
                say(f"[Book] {name}: {len(keep)} of {len(candidates)} passages "
                    f"are about them")
                if not keep and refused:
                    return {"ok": False,
                            "error": "every passage naming this character describes "
                                     "someone else, or nothing physical",
                            "fields": {}, "used": [], "who": {}, "seconds": 0.0}
            if keep:
                candidates = [candidates[index] for index in keep]
            used = list(candidates)
        else:
            used = list(candidates)
        body = "\n\n".join(c["text"] for c in used)
        prompt = (_DIRECT_PROMPT.format(name=name, aliases=", ".join(aliases))
                  + f"\n\nPassages:\n{body}")
    try:
        raw = await get_llm_response(
            prompt, llm_url, model, temperature=0.0, timeout=180,
            extra_payload={"max_tokens": 500})
    except Exception as ex:
        return {"ok": False, "error": f"extraction call failed: {ex}",
                "fields": {}, "used": [], "who": who,
                "seconds": round(time.time() - started, 1)}

    data = json_object(raw)
    if isinstance(data.get("fields"), dict):
        data = data["fields"]
    fields = sanitise({k: str(v) for k, v in data.items()
                       if k in EXTRACT_FIELDS and v not in (None, "", "null")})
    fields = merge_height(fields)
    say(f"[Book] {name}: {len(fields)} fields from {len(used)} paragraphs")
    return {"ok": bool(fields), "error": "" if fields else "no traits stated",
            "fields": fields, "used": used, "who": who,
            "seconds": round(time.time() - started, 1)}


def merge_height(fields: Dict[str, str]) -> Dict[str, str]:
    """Fold ``height`` into ``build`` and put both into the canonical vocabulary.

    There is no height column by design: heights asked for separately from build
    drift apart, and a measurement in the prompt does not help an image model.
    ``8'3" tall`` becomes ``tall`` and ``170 pounds`` disappears, which is why this
    runs on the way in from the extractor rather than at render time.
    """
    return trait_vocab.normalise_traits(fields)
