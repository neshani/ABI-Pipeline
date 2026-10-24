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

MAX_PARAGRAPHS = 14
MAX_PARAGRAPH_CHARS = 900

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

_EXTRACT_PROMPT = (
    "Extract only what this text states about physical appearance. Never guess and "
    "never add what is typical.\n"
    'Reply with JSON only: {"sex","species","age","height","build","hair_color",'
    '"hair_style","eye_color"} - null for anything not stated. No prose, no fence.\n'
    "Shapes: sex = male|female. species = lowercase noun, null for a human. "
    "age = young|adult|middle-aged|old. height = as the text says it, including "
    "comparisons (\"half a head shorter than Rand\"). build = adjectives only. "
    "hair_color/eye_color = colour words. hair_style = a short phrase."
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _json_object(raw: str) -> Dict:
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


def cue_paragraphs(books: Sequence[Tuple[str, str]], aliases: Sequence[str],
                   limit: int = MAX_PARAGRAPHS) -> List[Dict]:
    """Paragraphs naming the character that also carry appearance words.

    ``books`` is ``[(book_name, transcript_text)]`` in reading order. Reading
    order matters: the earliest mention is where an introduction usually lives.
    """
    patterns = [re.compile(r"\b" + re.escape(a) + r"\b", re.I)
                for a in aliases if a.strip()]
    if not patterns:
        return []
    found: List[Dict] = []
    for book_name, transcript in books:
        for index, paragraph in enumerate(paragraphs_of(transcript)):
            if not CUE.search(paragraph):
                continue
            if not any(pattern.search(paragraph) for pattern in patterns):
                continue
            found.append({"book": book_name, "index": index,
                          "text": paragraph[:MAX_PARAGRAPH_CHARS]})
            if len(found) >= limit:
                return found
    return found


def _mentions(answer: str, aliases: Sequence[str]) -> bool:
    """Does a 'who is described' answer name this character?"""
    text = _clean(answer).lower()
    if not text or text.startswith(("none", "no one", "nobody")):
        return False
    for alias in aliases:
        token = alias.strip().lower().split(" ")[0]
        if len(token) >= 4 and token in text:
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
    should_cancel: Optional[Callable[[], bool]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Research one character from the book text. Never raises; ``ok`` says it.

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

    candidates = cue_paragraphs(books, aliases, limit=limit)
    if not candidates:
        return {"ok": False, "error": "no paragraph in this book mentions the "
                                      "character alongside an appearance word",
                "fields": {}, "used": [], "who": {}, "seconds": 0.0}

    listing = "\n\n".join(f"[{i}] {c['text']}"
                          for i, c in enumerate(candidates))
    say(f"[Book] {name}: {len(candidates)} candidate paragraphs")

    if should_cancel and should_cancel():
        return {"ok": False, "error": "cancelled", "fields": {}, "used": [],
                "who": {}, "seconds": round(time.time() - started, 1)}

    try:
        who_raw = await get_llm_response(
            f"{_WHO_PROMPT}\n\n{listing}", llm_url, model,
            temperature=0.0, timeout=180,
            extra_payload={"max_tokens": 400})
    except Exception as ex:
        return {"ok": False, "error": f"attribution call failed: {ex}",
                "fields": {}, "used": [], "who": {}, "seconds": 0.0}

    who = _json_object(who_raw)
    chosen = []
    for index, answer in who.items():
        try:
            position = int(index)
        except (TypeError, ValueError):
            continue
        if 0 <= position < len(candidates) and _mentions(str(answer), aliases):
            chosen.append(position)
    chosen.sort()
    say(f"[Book] {name}: model attributed {len(chosen)} of {len(candidates)} to them")
    if not chosen:
        return {"ok": False,
                "error": "none of the candidate paragraphs describe this character",
                "fields": {}, "used": [], "who": who,
                "seconds": round(time.time() - started, 1)}

    body = "\n\n".join(candidates[i]["text"] for i in chosen)
    try:
        raw = await get_llm_response(
            f"{_EXTRACT_PROMPT}\n\nCharacter: {name}\n\n{body}",
            llm_url, model, temperature=0.0, timeout=180,
            extra_payload={"max_tokens": 500})
    except Exception as ex:
        return {"ok": False, "error": f"extraction call failed: {ex}",
                "fields": {}, "used": [], "who": who,
                "seconds": round(time.time() - started, 1)}

    data = _json_object(raw)
    fields = sanitise({k: str(v) for k, v in data.items()
                       if k in EXTRACT_FIELDS and v not in (None, "", "null")})
    fields = merge_height(fields)
    used = [candidates[i] for i in chosen]
    say(f"[Book] {name}: {len(fields)} fields from {len(used)} paragraphs")
    return {"ok": bool(fields), "error": "" if fields else "no traits stated",
            "fields": fields, "used": used, "who": who,
            "seconds": round(time.time() - started, 1)}


def merge_height(fields: Dict[str, str]) -> Dict[str, str]:
    """Fold ``height`` into ``build``, the field that renders.

    There is no height column by design: heights asked for separately from build
    drift apart, and a number in the prompt does not help an image model. The
    verbal part of build is kept first so the clause still reads, with the
    height - often a comparison ("half a head shorter than Rand") - appended.
    """
    out = dict(fields)
    height = _clean(out.pop("height", ""))
    build = _clean(out.get("build", ""))
    if height and build:
        if height.lower() in build.lower():
            pass
        elif build.lower() in height.lower():
            build = height
        else:
            build = f"{build}, {height}"
    elif height:
        build = height
    if build:
        out["build"] = build
    else:
        out.pop("build", None)
    return out
