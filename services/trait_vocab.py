"""One canonical vocabulary for the trait fields, applied where a value enters the app.

The renderer passes trait values through untouched on purpose: what the user sees
in the Character tab is exactly what reaches the image prompt. That only works if
the values are already in a shape an image model can use, so the conversion happens
here - on the way in from an LLM, never on the way out to ComfyUI.

Three rules:

* **Extraction normalises, rendering does not.** Every path that takes traits from
  a model (the cast extractor, reference/wiki/web/book imports) passes its values
  through ``normalise_traits``. A value the user types in the Character tab is
  stored and rendered verbatim; if they want a round face, they get a round face.
* **A number is not a picture.** "170 pounds" and ``5'3" tall`` say nothing an
  image model can draw, so weights are dropped and heights become a word
  ("short" / "tall" at the thresholds below).
* **Unrecognised is not destroyed.** A build the vocabulary cannot classify is
  dropped from ``build`` (it is not a frame) and reported by the fixer, so the
  user can see the value and decide. An unrecognised age is kept verbatim.

The vocabulary is general English. A book's own coinages belong in the project's
data, not in this file.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

# Age stages, in reading order. The first two change the noun ("boy"/"girl"),
# the rest are adjectives in front of it ("young man").
AGE_STAGES = ("child", "teen", "young", "adult", "middle-aged", "old")
CHILD_AGES = ("child", "teen")

# Height in inches. A value between the two is average: neither word adds
# anything to a prompt, so it is dropped rather than guessed at. The band is
# deliberately sex-blind and wide - 5'5" is short, 6'2" is tall, and the
# commonest adult heights land in the middle and say nothing.
SHORT_BELOW_IN = 66.0   # 5'6"
TALL_FROM_IN = 72.0     # 6'0"

# --- build -------------------------------------------------------------------

_FRAME_SYNONYMS: Dict[str, str] = {}


def _frames(canonical: str, *words: str) -> None:
    for word in words:
        _FRAME_SYNONYMS[word] = canonical


_frames("thin", "thin", "skinny", "scrawny", "bony", "gaunt", "spare", "lean",
        "slight", "wiry", "rawboned", "raw-boned", "underweight", "gangly",
        "gnarled", "stork-like", "stringy")
_frames("slim", "slim", "slender", "willowy", "lithe", "svelte", "delicate")
_frames("lanky", "lanky", "rangy", "gangling")
_frames("average", "average", "medium", "normal", "moderate", "medium build",
        "well-proportioned", "well proportioned")
_frames("stocky", "stocky", "sturdy", "solid", "thickset", "thick-set",
        "big-boned", "big boned", "stocky but strong", "broad", "thick")
_frames("stout", "stout", "portly", "plump", "round", "rotund", "chubby",
        "pudgy", "fleshy", "paunchy", "round-faced")
_frames("heavyset", "heavyset", "heavy-set", "heavy", "fat", "obese",
        "corpulent", "hulking", "massive", "enormous", "big", "large",
        "larger", "very big", "big-bodied")
_frames("muscular", "muscular", "brawny", "powerful", "well-built",
        "well built", "beefy", "athletic", "sinewy", "strong", "rippling")
_frames("broad-shouldered", "broad-shouldered", "broad shouldered",
        "broad chest and shoulders", "broad chest", "broad in the shoulders",
        "heavy in the shoulders", "wide-shouldered", "wide shouldered",
        "big-shouldered", "square-shouldered")

# Height as a word, for values that never say a number.
_TALL_WORDS = ("tall", "taller", "giant", "giants", "giant-sized", "towering",
               "long-limbed")
_SHORT_WORDS = ("short", "shorter", "small", "petite", "little", "diminutive",
                "stunted", "waif-like")

# A build value that is really a body part ("narrow face") is not a frame.
_BODY_PART = re.compile(
    r"\b(face|jaw|chin|nose|eyes?|brow|cheeks?|mouth|neck|waist|belly|stomach|"
    r"arms?|legs?|hands?|fingers?|knees?|throat|hair)\b", re.I)

_WEIGHT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:pounds?|lbs?|kilos?|kilograms?|kg|stone)\b", re.I)
_CM = re.compile(r"\b(\d{2,3})\s*(?:cm|centimet(?:er|re)s?)\b", re.I)
_FEET = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight)\s*"
    r"(?:'|’|ft\b|feet\b|foot\b)\s*"
    r"(\d+)?\s*(?:\"|”|in\b|inch(?:es)?\b)?", re.I)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "eleven": 11, "twelve": 12}


def _low(text: str) -> str:
    return " ".join((text or "").lower().split())


def _find_phrase(text: str, table: Dict[str, str]) -> str:
    """The longest synonym present as whole words, mapped to its canonical form."""
    low = _low(text)
    if not low:
        return ""
    for phrase in sorted(table, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", low):
            return table[phrase]
    return ""


def _find_any(text: str, words: Iterable[str]) -> str:
    low = _low(text)
    for word in sorted(words, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", low):
            return word
    return ""


def height_inches(text: str) -> Optional[float]:
    """Feet/inches or centimetres found in a value, as inches. ``None`` if absent."""
    if not text:
        return None
    match = _FEET.search(text)
    if match:
        feet_raw = match.group(1).lower()
        feet = _NUMBER_WORDS.get(feet_raw)
        if feet is None:
            try:
                feet = int(feet_raw)
            except ValueError:
                feet = None
        if feet is not None:
            inches = float(match.group(2) or 0)
            return feet * 12.0 + min(inches, 11.0)
    match = _CM.search(text)
    if match:
        return float(match.group(1)) / 2.54
    return None


def height_word(inches: Optional[float]) -> str:
    """``short`` / ``tall`` / ``average`` for a measurement in inches."""
    if inches is None:
        return ""
    if inches < SHORT_BELOW_IN:
        return "short"
    if inches >= TALL_FROM_IN:
        return "tall"
    return "average"


def states_average_build(text: str) -> bool:
    """True when a build value names a middling frame.

    An average frame is dropped from the prompt on purpose (it says nothing an
    image model can draw), which is a different outcome from "not a frame at all"
    and deserves a different explanation to the user.
    """
    return _find_phrase(text or "", _FRAME_SYNONYMS) == "average"


def normalise_build(build: str, height: str = "") -> str:
    """A canonical frame/height phrase, or ``""`` when nothing usable is stated.

    The measurement wins over the word when both are present: "5'3" tall" is a
    short character whose extractor reached for "tall" from the sentence around it.
    """
    build = build or ""
    inches = height_inches(build)
    if inches is None:
        inches = height_inches(height or "")
    tall = height_word(inches)
    if not tall:
        if _find_any(build, _TALL_WORDS) or _find_any(height, _TALL_WORDS):
            tall = "tall"
        elif _find_any(build, _SHORT_WORDS) or _find_any(height, _SHORT_WORDS):
            tall = "short"

    frame = _find_phrase(build, _FRAME_SYNONYMS) or _find_phrase(height, _FRAME_SYNONYMS)

    # A body part with no frame word is a face/belly detail that landed in build.
    if not frame and not tall and _BODY_PART.search(build):
        return ""

    parts: List[str] = []
    if tall in ("tall", "short") and tall != frame:
        parts.append(tall)
    if frame and frame != "average":
        parts.append(frame)
    return " and ".join(parts)


# --- age ---------------------------------------------------------------------

_AGE_SYNONYMS: Dict[str, str] = {}


def _ages(canonical: str, *words: str) -> None:
    for word in words:
        _AGE_SYNONYMS[word] = canonical


_ages("child", "child", "children", "kid", "kids", "boy", "boys", "girl",
      "girls", "lad", "lass", "little", "youngster", "young child")
_ages("teen", "teen", "teens", "teenager", "teenaged", "adolescent",
      "adolescence", "juvenile", "youth", "young lad", "young girl", "young boy")
_ages("young", "young", "youthful", "younger", "young man", "young woman",
      "in his twenties", "in her twenties", "early twenties", "twenties")
_ages("adult", "adult", "grown", "grown-up", "grown up", "in his thirties",
      "in her thirties", "thirties", "in his forties", "in her forties",
      "forties", "in his fifties", "in her fifties", "fifties", "mature adult")
_ages("middle-aged", "middle-aged", "middle aged", "middle years", "middle",
      "mature", "in his sixties", "in her sixties", "sixties")
_ages("old", "old", "older", "elderly", "elder", "aged", "ancient", "grizzled",
      "gray-haired", "grey-haired", "white-haired", "in his seventies",
      "in her seventies", "seventies", "venerable")

_NON_ANSWERS = {"", "-", "?", "null", "none", "n/a", "na", "unknown",
                "unspecified", "unclear", "unstated", "unmentioned",
                "unremarkable", "not stated", "not specified", "not mentioned",
                "not described", "not known", "not given", "not available",
                "no value", "blank", "nobody", "nothing"}


def is_non_answer(value: str) -> bool:
    """True when a value is a refusal to answer rather than a value.

    A model asked for a face and given no basis writes "not specified" about as
    often as it omits the key. Stored, that renders as "a man, not specified",
    so it is filtered on the way in - for every source, not just the caster.
    """
    text = " ".join((value or "").lower().split()).strip(" .,;:")
    return text in _NON_ANSWERS

_NUMERIC_AGE = re.compile(r"\b(\d{1,3})\b")

# "nine years old" must beat the "old" synonym, or a nine-year-old becomes old.
_YEARS_OLD = re.compile(
    r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s*\+?\s*(?:years?|yrs?)[\s-]*old\b", re.I)
_AGED = re.compile(
    r"\b(?:age|aged)\s+(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve)\b", re.I)


def _years_from(match) -> Optional[int]:
    raw = match.group(1).lower()
    if raw in _NUMBER_WORDS:
        return _NUMBER_WORDS[raw]
    try:
        return int(raw)
    except ValueError:
        return None


def _stage_from_years(years: int) -> str:
    if years < 13:
        return "child"
    if years < 18:
        return "teen"
    if years < 30:
        return "young"
    if years < 50:
        return "adult"
    if years < 65:
        return "middle-aged"
    return "old"


def normalise_age(value: str) -> Optional[str]:
    """Canonical stage, ``""`` for a non-answer, ``None`` when unrecognised.

    ``None`` is the "keep what the extractor wrote" answer: an age the vocabulary
    does not know is still the user's data, and ``_base_noun`` simply ignores it.
    """
    low = _low(value)
    if low in _NON_ANSWERS:
        return ""
    # A stated age beats a stated stage: "nine years old" is a child, even though
    # the sentence also contains the word "old".
    for pattern in (_YEARS_OLD, _AGED):
        match = pattern.search(value or "")
        if match:
            years = _years_from(match)
            if years is not None:
                return _stage_from_years(years)
    stage = _find_phrase(low, _AGE_SYNONYMS)
    if stage:
        return stage
    match = _NUMERIC_AGE.search(low)
    if match:
        return _stage_from_years(int(match.group(1)))
    return None


# --- entry points ------------------------------------------------------------

def normalise_traits(fields: Dict[str, str],
                     report: Optional[list] = None) -> Dict[str, str]:
    """Normalise the fields an extractor produced. ``height`` never leaves here.

    ``report`` collects ``(field, before, after)`` for every value that changed,
    so a caller can show the user exactly what was rewritten.
    """
    out = dict(fields)
    height = str(out.pop("height", "") or "")
    before_build = str(out.get("build", "") or "")
    if before_build or height:
        build = normalise_build(before_build, height)
        if build:
            out["build"] = build
        else:
            out.pop("build", None)
        if report is not None and (out.get("build") or "") != before_build:
            report.append(("build", before_build, out.get("build") or ""))

    if out.get("age"):
        before_age = str(out["age"])
        age = normalise_age(before_age)
        if age is None:
            pass
        elif age:
            out["age"] = age
        else:
            out.pop("age", None)
        if report is not None and (out.get("age") or "") != before_age:
            report.append(("age", before_age, out.get("age") or ""))
    return out


def normalise_value(field: str, value: str) -> Optional[str]:
    """One field's value. ``None`` means "unrecognised, keep as written"."""
    if field == "build":
        return normalise_build(value)
    if field == "height":
        return ""
    if field == "age":
        return normalise_age(value)
    return value


def describe(field: str, before: str, after: str) -> str:
    """One human-readable line for a change, used by the fixer's report."""
    if not after:
        return f"{field}: {before!r} -> dropped"
    return f"{field}: {before!r} -> {after!r}"
