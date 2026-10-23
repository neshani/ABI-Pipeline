"""Character analysis: find descriptions, name the references, extract facts.

Three stages, each with its own question, because the single-pass extractor failed
at the job it was worst placed to do - deciding WHO a description belonged to
while also inventing a field and a value:

1. **gather** reads one window (whole groups, never across a chapter) and returns
   ``{"scene", "subject", "quote"}`` only. No fields, no values, nothing to invent.
   A quote whose subject the window does not name is dropped and kept in
   ``dropped`` for stage 2.
2. **reference names** asks one narrow question per window about those dropped
   references ("the white-haired man"), with the window visible. ``unnamed`` is an
   allowed answer, and an animal reference is never attached to a person - the
   measured failure was 'the sleek white mare' -> the rider.
3. **extract** reads only one character's gathered quotes, in slices of about 120,
   and sorts them into the field set. Evidence must be a fragment of one of that
   character's own quotes, which is what makes re-attribution impossible. A slice
   whose reply is cut off at ``max_tokens`` is halved and re-asked, because a
   truncated reply parses as a shorter one.
4. **sex** is a short follow-up question per character whose sex the pronoun vote
   could not settle, answered from that character's own quotes. It is separate from
   pass 3 on purpose: `sex` inside the field set competes with the body traits.

Positions are ``(chapter, group-index-within-chapter)``. The group is the
extraction unit and the prompt numbers the groups it can see, so a long chapter
split into two windows still reports into the same chapter coordinate; the old
probe conflated the window index with the chapter number and only worked because
no chapter in its test book exceeded one window.

The cast writer maps canonical names to ``Character`` records (creating them when
needed), links them to the book, and writes one ``CharacterTimelineEvent`` per
position where the description changes. Each event carries the COMPLETE state in
force there, including fields inherited from earlier volumes, because the render
path compiles from the latest event rather than merging fields across events.

No prompt in this module names a book, a character or a series fact.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlmodel import Session, select

from database.connection import engine, get_setting
from database.models import (Book, Character, CharacterAlias, CharacterBookLink,
                             CharacterTimelineEvent, Project)
from services import scene_map
from services import checkpoints
from services import pipeline_manifest
from services import llm_config
from services.character_manager import (NEW_TRAIT_FIELDS, RENDERED_TRAIT_FIELDS,
                                        normalize_character_text,
                                        save_project_characters_to_json)

ANALYSIS_VERSION = 2
GATHER_TIMEOUT = 900.0
REFNAME_TIMEOUT = 900.0
EXTRACT_TIMEOUT = 900.0
GATHER_MAX_TOKENS = 4096
REFNAME_MAX_TOKENS = 1500
EXTRACT_MAX_TOKENS = 3000
# Quotes per pass-2 call. Reply length tracks how many quotes a call is given, so
# this bounds the reply as much as the prompt; the measured 3,000-token cap was
# cutting the biggest characters off mid-book. Even slices, never a greedy fill:
# 130 quotes become 65 + 65, not 120 + 10.
EXTRACT_QUOTE_CAP = 120
# Below this a truncated slice is accepted rather than divided again.
EXTRACT_MIN_SLICE = 20
# A sex question is one short answer, so the reply is a handful of tokens and the
# call costs a fraction of a second. Quotes are sampled evenly across the book.
SEX_MAX_TOKENS = 24
SEX_QUOTE_CAP = 100
SEX_TIMEOUT = 300.0

ANALYSIS_FIELDS = tuple(f for f in NEW_TRAIT_FIELDS
                        if f in ("sex", "species", "age", "hair_color", "hair_style",
                                 "eye_color", "face", "build", "attire", "carried",
                                 "notes"))


# --- prompts ----------------------------------------------------------------

GATHER_PROMPT = """You are preparing an illustrated edition of {book}.

Read this chapter. It is divided into numbered SCENES, each labelled with where it
takes place. In each scene, find every sentence - or a phrase inside one - that says
how a CHARACTER looks: their body, face, hair, eyes, skin, size, clothing, or what
they carry. A character is any being the text describes, including animals.

Do not decide what kind of trait it is, and do not summarise. Just find the text.

For each one, give:
- "scene": the number of the scene it appears in.
- "subject": the character's NAME as the chapter gives it. Never a bare noun ("the
  man", "the stableman"), never an object, never a group. If the chapter does not
  name them, skip it.
- "quote": the sentence or phrase, copied VERBATIM from the chapter.

Rules:
- One line per description. If one sentence describes two characters, give it once
  for each, with that character's name.
- Most sentences describe nobody. Finding nothing in a scene is normal.
- Copy the text exactly. Do not tidy it or shorten it.

Output one JSON object per line and nothing else:
{{"scene": 1, "subject": "...", "quote": "..."}}"""

REFNAME_PROMPT = """You are preparing an illustrated edition of {book}.

Below is a chapter. Under it is a list of REFERENCES: phrases the text uses for a
character before giving their name, or instead of it - "the white-haired man",
"the gleeman".

For each reference, name the character in this chapter it means, using the name the
chapter gives that character. Answer "unnamed" if the chapter never gives them a
name, and "unknown" if you cannot tell.

A reference is about the being it names. If the chapter describes a horse as "the
sleek white mare", that reference is the horse - not the person who rides it.

Output one JSON object per line and nothing else:
{{"reference": "...", "name": "..."}}

CHAPTER:
{text}

REFERENCES:
{refs}"""

SEX_FIELD = (
    '- sex: "male" or "female" - ONE word and nothing else. Use the pronouns the\n'
    '  sentences use; answer "unknown" if they do not settle it.\n')
SPECIES_FIELD = (
    '- species: what KIND of being they are, one word - "human", "horse", "cat".\n'
    '  "human" is the default; say anything else only when a sentence calls them it.\n')
AGE_FIELD = (
    '- age: ONE word - "young", "adult", "middle-aged", "old". Not a number and not a\n'
    '  description. Answer "unknown" if no sentence says.\n')
HAIR_COLOR_FIELD = (
    '- hair_color: the COLOUR of the hair, and nothing else - "gray", "black",\n'
    '  "reddish", "fair". Not a length, not a style, not a colour of anything else.\n')
HAIR_STYLE_FIELD = (
    '- hair_style: how the hair is WORN or CUT, and nothing else - "in a long braid",\n'
    '  "loose and unbound", "cropped short", "curly". Not a colour.\n')
EYE_COLOR_FIELD = (
    '- eye_color: the COLOUR of the eyes, and nothing else - "gray", "brown", "blue",\n'
    '  "amber". Not a shape, not a size, not an expression, and not what the eyes were\n'
    '  doing in the scene.\n')
BUILD_FIELD = (
    '- build: height and frame ONLY - "tall", "slim", "broad-shouldered",\n'
    '  "a heavyset man". A face, an arm, a belly, a knee, a posture, a\n'
    '  manner or an action is NOT build; leave it out.\n')
FIELDS_BLOCK = "\n".join([
    SEX_FIELD, SPECIES_FIELD, AGE_FIELD, HAIR_COLOR_FIELD, HAIR_STYLE_FIELD,
    EYE_COLOR_FIELD, BUILD_FIELD,
    "- attire: what they are wearing now.",
    "- carried: an object they hold or wear - a staff, a sword, a pipe."])
SINK_BLOCK = """
LAST RESORT - "notes". If a passage states something about a person that is NOT part
of their body and fits none of the fields above, put it here rather than forcing it
into a field. It is a scratch pad - stored, never shown to anyone, never used in an
image - so a body detail put here is LOST. Do not go looking for something to put in
it: when nothing fits, report nothing, which is the expected answer.
"""

SEX_PROMPT = """You are indexing a novel for an illustrated edition.

Below are sentences that mention one character, in the order they appear. Decide
the character's sex.

Rules:
- Use how the sentences refer to this character: pronouns and gendered words.
- Other people are mentioned in the same sentences. A pronoun that belongs to one
  of them is not evidence about this character.
- A gendered title is evidence.
- Answer "unknown" only when the sentences carry no usable evidence.

Output one JSON object and nothing else:
{{"sex": "male"}} or {{"sex": "female"}} or {{"sex": "unknown"}}

CHARACTER: {name}

SENTENCES:
{lines}"""

EXTRACT_PROMPT = """You are preparing an illustrated edition of {book}.

Below are the sentences from throughout the book that describe ONE character, in the order
they appear. Report every physical trait those sentences state about them.

The sentences are already attributed. Report traits of the NAMED character only - if a
sentence also mentions somebody else, ignore the other person entirely.

Fields:
{fields}
Rules:
- "subject": the character's name exactly as written above.
- "evidence": copy VERBATIM the shortest part of the sentence that states the trait.
  It must come from one of that character's own sentences.
- "value": a short noun phrase of a few words - no verb, no sentence.
- "window": the chapter number on the line ("W7" means 7), and "scene": the scene number.
- Report nothing for a field the sentences do not state. Most fields will be empty.
- An expression is not a feature: "unblinking eyes", "anxious eyes" and "eyes that
  glowed" are things the eyes were doing, not what they look like.
- Only what is happening now. A memory or a retelling does not describe how someone
  looks today.
{sink}
Output one JSON object per line and nothing else, with the keys in this order:
{{"scene": 1, "window": 2, "subject": "...", "evidence": "...", "field": "...", "value": "..."}}

THE CHARACTER:
{characters}"""


# --- text helpers -----------------------------------------------------------

_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)
_PUNCT = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
          "\u2014": "-", "\u2013": "-", "\u00a0": " "}


def _apos(text: str) -> str:
    return normalize_character_text(text)


def normalise_for_grounding(text: str) -> str:
    for a, b in _PUNCT.items():
        text = text.replace(a, b)
    return " ".join(text.replace("\n", " ").split()).lower()


def evidence_in_text(evidence: str, text_norm: str) -> bool:
    """Is the quote present in the text, allowing an ellipsis to join two fragments?"""
    frags = []
    for raw in re.split(r"\.\.\.|\u2026", evidence or ""):
        frag = normalise_for_grounding(raw).strip(' .,;:"\'')
        if len(frag.split()) >= 2:
            frags.append(frag)
    if not frags:
        return False
    return all(frag in text_norm for frag in frags)


def _clean_name(name: str) -> str:
    return " ".join(_apos(str(name or "")).replace("`", "").split())


def _book_title(name: str) -> str:
    return re.sub(r"^\d+\s+", "", name or "")


# --- names: scope, generics, canonical forms --------------------------------

# General-English role nouns. A description filed under one of these belongs to no
# single character, and the bucket would poison the other-character test. This list
# is deliberately species- and series-neutral: a book's own invented kinds are not
# known here and must not be listed.
GENERIC_SUBJECTS = {
    "man", "men", "woman", "women", "boy", "girl", "child", "children", "baby",
    "figure", "rider", "person", "people", "soldier", "soldiers", "guard",
    "guards", "servant", "servants", "speaker", "voice", "crowd", "youth",
    "stranger", "strangers", "fellow", "lad", "lass", "elder", "goodwife",
    "innkeeper", "farmer", "villager", "villagers", "captain", "lord", "lady",
    "watchman", "watchmen", "watch", "gatekeeper", "gatekeepers",
    # Roles and collectives the model reaches for instead of a name. General
    # English only: a book's invented kinds are not listed here.
    "mayor", "maid", "maids", "townspeople", "townsman", "townsmen", "wind",
    "serving", "group", "groups", "party", "company", "folk",
}
NAME_NOISE = {"the", "of", "a", "an", "lord", "lady", "master", "mistress",
              "captain", "sir", "one", "this", "that", "two", "three", "fat",
              "tall", "short", "old", "young", "thin", "big", "little", "child"}
_NAME_STOP = {"the", "lord", "lady", "master", "mistress", "captain",
              "child", "old", "young"}

# Pronouns and quantifiers are not names. Kept deliberately short and generic.
_PRONOUN_SUBJECTS = {"they", "them", "their", "theirs", "themselves", "he", "him",
                     "his", "himself", "she", "her", "hers", "herself", "it", "its",
                     "itself", "we", "us", "our", "ours", "ourselves", "you", "your",
                     "yours", "yourselves", "i", "me", "my", "mine", "myself",
                     "everyone", "someone", "anyone", "nobody", "somebody", "anybody"}
_QUANTIFIER_PREFIX = re.compile(
    r"^(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|many|"
    r"several|some|both|all|few|each|every)\b", re.I)


def generic_token(token: str) -> bool:
    """Is this token a generic noun in any number?"""
    if token in GENERIC_SUBJECTS:
        return True
    for suffix in ("s", "es"):
        if token.endswith(suffix) and token[:-len(suffix)] in GENERIC_SUBJECTS:
            return True
    return False


def bad_subject(name: str) -> bool:
    """A character's name is capitalised. Anything else is not a character."""
    if not name or not name[0].isupper():
        return True
    low = _apos(name).lower().strip()
    if low in _PRONOUN_SUBJECTS:
        return True
    if re.match(r"^(unnamed|unknown|some|a |an |the |several|group|figure|"
                r"creature|crowd|voice|they|them|he|she|it|we|you)\b", low):
        return True
    if _QUANTIFIER_PREFIX.match(low):
        return True
    # Two names joined by 'and' (or '&') are a group, not a person. The
    # canonical map refuses to merge them, but they must not become a character.
    if re.search(r"(^|\s)(and|&)(\s|$)", low):
        return True
    if any(word in low for word in ("unnamed", "unknown", " creature", " figure",
                                    "something", "someone", "somebody", "anyone",
                                    "everyone", "nobody", "no one")):
        return True
    if re.search(r"[a-z]'s\b", low):
        return True
    toks = [t for t in re.findall(r"[a-z']+", low) if t not in NAME_NOISE]
    if not toks:
        return True
    if any(generic_token(t) for t in toks):
        return True
    return " ".join(toks) in GENERIC_SUBJECTS


def tokens(name: str) -> set:
    return {w for w in re.findall(r"[a-z']{3,}", _apos(name).lower())
            if w not in _NAME_STOP}


def given_tokens(subjects: Iterable[str]) -> set:
    """The tokens that identify a person, computed from the names we have."""
    given = set()
    for subject in subjects:
        parts = [w for w in re.findall(r"[a-z']{3,}", _apos(subject).lower())
                 if w not in _NAME_STOP]
        if len(parts) > 1:
            given.add(parts[0])
    return given


def _title_words(name: str) -> set:
    return {w for w in name.lower().split() if w in _NAME_STOP}


def same_person(a: str, b: str, given: set) -> bool:
    """Are two name forms the same character? ``given`` is ``given_tokens(all)``."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        ta_title, tb_title = _title_words(a), _title_words(b)
        if ta_title and tb_title and ta_title != tb_title:
            return False
        return True
    if ta < tb:
        small, big = ta, tb
    elif tb < ta:
        small, big = tb, ta
    else:
        return False
    # A two-name label is a GROUP, not a person. 'Rand and Mat' contains both given
    # names and must not swallow either character.
    if len(big & given) > 1:
        return False
    return bool(small & given)


def canonical_map(names: Sequence[str], counts: Dict[str, int]):
    """``({form: canonical}, groups, tokens)`` - the longest non-titled form wins."""
    given = given_tokens(names)
    toks = {n: tokens(n) for n in names}
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if same_person(a, b, given):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    groups = collections.defaultdict(list)
    for n in names:
        groups[find(n)].append(n)
    out = {}
    for members in groups.values():
        def rank(n):
            return (len(toks[n]), -bool(_title_words(n)), counts[n], len(n))
        canon = max(members, key=rank)
        for m in members:
            out[m] = canon
    return out, groups, toks


# --- validators -------------------------------------------------------------
#
# Ported from the measured v15 extractor. The active rules are the SYNTAX rules
# (a value must be a noun phrase), the three closed-field grounding rules, the
# closed-class BUILD check, and the other-character check. Vocabulary gates were
# measured to cost more real values than they saved and are deliberately absent.

PLACEHOLDERS = {"none", "n/a", "na", "null", "nil", "nothing", "unknown", "no",
                "-", "--", "?", "not applicable", "not specified", "no value",
                "blank", "blanks", "empty", "none given", "not stated", "unclear"}

ELLIPSIS = re.compile(r"\.\.\.|\u2026")
RELATIVE = re.compile(r"\b(that|which|who|whom|whose)\b", re.I)
COMPARATIVE = re.compile(
    r"\b(than|as\s+\w+\s+as|double\s+that|twice\b|half\s+(as|that)|"
    r"more\s+\w+\s+than|less\s+\w+\s+than|older|younger|taller|shorter|"
    r"tall\s+enough|long\s+enough|size\s+of|(?<![\w-])like(?![\w-]))\b", re.I)
SIMILE = re.compile(r"\bas\s+(a|an|the|[A-Z][a-z]+)\b", re.I)
FINITE = re.compile(
    r"\b(was|were|is|are|am|been|being|has|have|had|do|does|did|would|will|"
    r"shall|should|could|can|must|may|might|"
    r"felt|bore|burst|led|sprang|saw|gave|drew|threw|caught|knew|found|left|"
    r"brought|spoke|stole|tore|woke|won|wrote|drove|ate|grew|got|hid|lost|"
    r"paid|rode|rang|shot|showed|slid|spun|spread|stuck|struck|swept|swung|"
    r"flung|fled|dug|fed|clung|slung|wrung|sank|slept|spent|taught|meant|met|"
    r"wound|rewound|bound|undid|hung|lit|shut|sat|stood|lay|ran|came|went|"
    r"kept|held|took|made|wore|said|told|sent|bought|fought|sought|read|"
    r"beat|bit|blew|broke|chose|drank|froze|hit|hurt|laid|rose|sung|sunk|"
    r"swam|shook|shone|fit|sling|spat|swore|trod|knelt|built|burnt|bent|dealt|"
    r"wept|crept|stank|stung|glance|glances|stare|stares|glimpse|glimpses|"
    r"peer|peers|gaz\w*|turning|becoming|getting)\b", re.I)
VERB_CONT = {
    "by", "with", "in", "at", "on", "into", "onto", "from", "as", "like", "to",
    "through", "over", "around", "across", "beneath", "behind", "above",
    "below", "past", "toward", "towards", "upon", "against", "of", "off",
    "out", "up", "down", "for", "and", "but", "or", "nor", "so", "yet", "then",
    "when", "while", "that", "which", "who", "whom", "his", "her", "their",
    "its", "my", "your", "our", "the", "a", "an", "this", "these", "those",
    "each", "every", "him", "them", "it", "me", "us", "all", "both", "no",
    "not", "too", "very", "again", "back", "away", "aside", "forward",
    "himself", "herself", "themselves", "itself", "myself", "yourself",
    "ourselves", "oneself",
}
EYED_VERB = re.compile(r"(?<![\w-])eyed(?![\w-])", re.I)
MORPH = re.compile(r"(?<![\w-])([a-z]{2,}(?:ing|ed))(?![\w-])", re.I)
ADJ_OK = {
    "colored", "coloured", "freckled", "tanned", "pointed", "bearded", "beaded",
    "winged", "striped", "checked", "haired", "limbed", "muscled", "boned",
    "featured", "toothed", "shaped", "marked", "weathered", "curved", "hooked",
    "slanted", "cropped", "shaven", "puffed", "padded", "pleated", "ruffled",
    "frilled", "laced", "trimmed", "embroidered", "brocaded", "scarred",
    "browned", "stained", "spotted", "speckled", "ringed", "banded", "studded",
    "jewelled", "jeweled", "gilded", "fringed", "tasseled", "flared", "piercing",
    "unseeing", "knowing", "haunting", "commanding", "brooding", "wizened",
    "sunburned", "rugged", "jagged", "ragged", "sacred", "wicked", "hundred",
    "longed", "hatted", "coated", "booted", "gloved", "caped", "belted",
    "sashed", "veiled", "robed", "gowned", "skirted", "sleeved", "collared",
    "hooded", "clothed", "dressed", "battered", "thinning", "towering",
    "gleaming", "drooping", "overfed", "tattered", "patched", "faded", "braided",
    "plaited", "quilted", "slashed", "buttoned", "buckled", "lidded", "greaved",
    "gauntleted", "spectacled", "wrinkled",
}
NOUN_OK = {
    "clothing", "stocking", "stockings", "string", "wing", "ring", "earring",
    "earrings", "bowstring", "thing", "lining", "sheeting", "matting", "bedding",
    "trimming", "backing", "facing", "drawing", "hanging", "wrapping", "evening",
    "morning", "opening", "gathering", "spring", "king", "beard", "bread",
    "head", "thread", "shred", "stone", "sword", "board", "glove", "hose",
    "silver", "shoulder", "shoulders", "legging", "leggings", "padding",
    "edging", "tie", "ties", "bead", "beads", "shade", "shades", "reed", "seed",
    "weeds", "steed", "horse", "horses", "tweed", "coloring", "colouring",
    "bearing", "outerwear", "underwear",
}
COLOUR_WORDS = {"black", "blue", "brown", "green", "gray", "grey", "hazel", "amber",
                "violet", "golden", "gold", "yellow", "red", "white", "blond",
                "blonde", "silver", "auburn", "sandy", "ruddy", "copper", "bronze",
                "olive", "tan", "pale", "dark", "fair", "cream"}
FRAME = re.compile(
    r"\b(tall|taller|tallest|short|shorter|shortest|average|medium|height|stature|"
    r"slim|thin|skinny|slender|slight|lean|leaner|heavyset|heavy-set|heavy|heavier|"
    r"bulky|bulk|wiry|muscular|muscled|scrawny|bony|plump|stocky|broad|broad-shouldered|"
    r"narrow|big|bigger|large|larger|huge|giant|massive|sturdy|squat|rotund|burly|"
    r"strong|stronger|gangly|lanky|spare|petite|small|smaller|little|frail|wide|stout|"
    r"towering|willowy|thick|thickset|powerful|solid|fat|obese|portly|pudgy|round|"
    r"rangy|rawboned|big-boned|well-built|athletic|dumpy|whipcord|blocky|gaunt|"
    r"girthy|skeletal|diminutive|overfed|fleshy|stumpy|breadth|deep|barrel-chested)\b",
    re.I)
BUILD_BODY_PART = re.compile(
    r"\b(face|faced|eye\w*|nose|mouth|chin|jaw\w*|cheek\w*|lip\w*|brow\w*|forehead|"
    r"beard\w*|mustach\w*|moustach\w*|arm\w*|hand\w*|fist\w*|finger\w*|leg\w*|knee\w*|"
    r"foot|feet|tooth|teeth|belly|waist|throat|skull)\b", re.I)
SEX_WORDS = {"male", "female", "man", "woman", "boy", "girl", "masculine",
             "feminine", "m", "f"}
SEX_VALUES = {"male", "female"}
PRON_MALE = re.compile(r"\b(he|him|his|himself)\b", re.I)
PRON_FEMALE = re.compile(r"\b(she|her|hers|herself)\b", re.I)
MALE_WORDS = re.compile(r"\b(man|men|male|boy|boys|lad|lads|sir|master|father|son|"
                        r"brother|husband|stallion|gelding|buck|bull|ram|cock)\b", re.I)
FEMALE_WORDS = re.compile(r"\b(woman|women|female|girl|girls|lady|ladies|madam|mistress|"
                          r"mother|daughter|sister|wife|mare|ewe|cow|doe|hen)\b", re.I)
AGE_SYN = {
    "young": ("young", "youth", "youthful", "boy", "girl", "child", "lad", "lass",
              "younger", "little"),
    "adult": ("adult", "grown", "grown-up", "man", "woman", "men", "women"),
    "middle-aged": ("middle-aged", "middle aged", "middle years", "mature", "middle"),
    "old": ("old", "older", "elderly", "elder", "aged", "ancient", "grizzled",
            "gray-haired", "grey-haired"),
}
SPECIES_SYN = {
    "horse": ("horse", "mare", "stallion", "gelding", "filly", "colt", "pony",
              "steed", "foal"),
    "sheep": ("sheep", "ewe", "ram", "lamb", "wether"),
    "cattle": ("cattle", "cow", "bull", "calf", "ox", "heifer", "steer"),
    "pig": ("pig", "sow", "boar", "piglet", "swine", "hog"),
    "dog": ("dog", "pup", "puppy", "hound", "mastiff"),
    "cat": ("cat", "kitten", "tomcat", "tom"),
    "goat": ("goat", "kid", "billy", "nanny"),
    "bird": ("bird", "hen", "cock", "chicken", "goose", "duck", "drake", "crow",
             "raven", "sparrow", "hawk", "eagle", "owl"),
    "wolf": ("wolf", "cub", "whelp"),
}

FIELD_ALIASES = {
    "clothing": "attire", "clothes": "attire", "dress": "attire",
    "outfit": "attire", "wardrobe": "attire", "garment": "attire",
    "garments": "attire", "wear": "attire", "wearing": "attire",
    "appearance": "build", "physique": "build", "body": "build",
    "posture": "build", "stance": "build", "figure": "build",
    "arms": "build", "hands": "build", "legs": "build", "fingers": "build",
    "shoulders": "build", "chest": "build", "muscles": "build",
    "stomach": "build", "torso": "build", "height": "build", "stature": "build",
    "eye": "eye_color", "eyes": "eye_color", "eye_color": "eye_color",
    "eye_colour": "eye_color", "haircolor": "hair_color", "hair_color": "hair_color",
    "hair_colour": "hair_color", "hair_and_face": "hair", "hair_face": "hair",
    "object": "carried", "held_object": "carried", "prop": "carried",
    "accessory": "carried", "item": "carried", "weapon": "carried",
    "signature_object": "carried", "held": "carried",
    "age_range": "age",
    # Facial hair is hair: for most characters it *is* the hairstyle, and the
    # composer renders it either in the hair phrase ("long white hair and a
    # bushy beard") or as its own clause ("white hair, bushy white brows"). The
    # model reaches for hair_style for it unaided, so no extra question is asked
    # and the value is not moved out of the field it was written to.
    "facial_hair": "hair_style", "beard": "hair_style", "beards": "hair_style",
    "moustache": "hair_style", "mustache": "hair_style", "whiskers": "hair_style",
    "sideburns": "hair_style", "eyebrows": "hair_style",
    # Fields the schema does not render. Keep the observation in the sink
    # rather than forcing it into a body field or throwing it away.
    "skin": "notes", "complexion": "notes", "skin_tone": "notes",
    "face": "notes", "features": "notes", "expression": "notes", "mood": "notes",
    "marks": "notes", "mark": "notes", "scars": "notes",
    "distinguishing_marks": "notes", "affiliation": "notes", "group": "notes",
    "faction": "notes",
}


def _canonical_field(field: str, value: str) -> str:
    f = FIELD_ALIASES.get(field, field)
    if f == "hair":
        return "hair_color" if value.strip().lower() in COLOUR_WORDS else "hair_style"
    return f


def _ungrounded_closed(field: str, value: str, evidence: str) -> str:
    v = value.strip().strip(".").lower()
    ev = evidence or ""
    if field == "sex":
        if v not in SEX_VALUES:
            return f"not-a-sex-value:{v[:20]}"
        pat = PRON_MALE if v == "male" else PRON_FEMALE
        words = MALE_WORDS if v == "male" else FEMALE_WORDS
        if pat.search(ev) or words.search(ev):
            return ""
        return f"ungrounded-sex:{v}"
    if field == "species":
        if len(v.split()) > 2:
            return "species-too-long"
        if v in ("human", "person", "people", ""):
            return ""
        core = re.findall(r"[a-z]+", v)
        if not core:
            return "species-empty"
        fam = SPECIES_SYN.get(core[0], (core[0],))
        if any(re.search(rf"\b{re.escape(w)}\b", ev, re.I) for w in fam):
            return ""
        return f"ungrounded-species:{v[:20]}"
    if field == "age":
        bucket = AGE_SYN.get(v)
        if bucket is None:
            return f"not-an-age:{v[:20]}"
        if any(re.search(rf"\b{re.escape(w)}\b", ev, re.I) for w in bucket):
            return ""
        return f"ungrounded-age:{v}"
    return ""


def _ungrounded_colour(value: str, evidence: str) -> str:
    v = value.strip().strip(".").lower()
    if v in COLOUR_WORDS and not re.search(rf"\b{re.escape(v)}\b", evidence.lower()):
        return f"ungrounded:{v}"
    return ""


def _not_a_build(value: str) -> str:
    if not FRAME.search(value):
        return "not-a-build"
    if BUILD_BODY_PART.search(value):
        return "body-part-in-build"
    return ""


def attribute_shaped(field: str, value: str) -> str:
    """Is the value a noun phrase rather than a clause or a stage direction?"""
    if ELLIPSIS.search(value):
        return "ellipsis"
    if RELATIVE.search(value):
        return "relative-clause"
    if COMPARATIVE.search(value):
        return "comparative"
    if field not in ("attire", "carried") and SIMILE.search(value):
        return "simile"
    if field in ("attire", "carried") and re.match(
            r"^(wearing|wore|clad|clothed|dressed|carrying|carried|holding|held|in)\b",
            value, re.I):
        return ""
    match = FINITE.search(value)
    if match:
        return f"finite-verb:{match.group(0)}"
    if EYED_VERB.search(value):
        return "verb-form:eyed"
    for match in MORPH.finditer(value):
        token = match.group(1).lower()
        if len(token) < 5:
            continue
        if token in ADJ_OK or token in NOUN_OK:
            continue
        rest = value[match.end():].lstrip(" ,;:")
        nxt = re.match(r"[A-Za-z']+", rest)
        if (nxt and nxt.group(0).lower() not in VERB_CONT
                and not nxt.group(0).lower().endswith("ly")):
            continue
        return f"verb-form:{token}"
    return ""


def other_character_named(subject: str, evidence: str, subjects: Iterable[str]):
    """Does the evidence name a DIFFERENT character?"""
    ev = evidence.lower()
    mine = tokens(subject)
    for other in subjects:
        if other.lower() == subject.lower():
            continue
        other_tokens = tokens(other)
        if not other_tokens or other_tokens <= mine:
            continue
        distinctive = {w for w in other_tokens - mine if len(w) >= 5}
        if not distinctive:
            continue
        if any(re.search(rf"\b{re.escape(w)}\b", ev) for w in distinctive):
            return other
    return None


def validate_fact(fact: dict, subjects: Iterable[str]) -> Tuple[bool, str]:
    """``(ok, rule)`` for one extracted fact. ``notes`` is stored without shape rules."""
    field = fact["field"]
    value = fact["value"]
    evidence = fact["evidence"]
    if not value:
        return False, "empty-value"
    if not evidence:
        return False, "empty-evidence"
    if value.strip().lower() in PLACEHOLDERS:
        return False, "placeholder"
    if not re.search(r"[A-Za-z]", value):
        return False, "placeholder"
    if re.match(r"^(no|not)\b", value.strip().lower()):
        return False, "negation"
    if field == "notes":
        return True, ""
    if field in ("sex", "species", "age"):
        rule = _ungrounded_closed(field, value, evidence)
        if rule:
            return False, rule
    ungrounded = _ungrounded_colour(value, evidence)
    if ungrounded:
        return False, ungrounded
    if field == "build":
        rule = _not_a_build(value)
        if rule:
            return False, rule
    if value.strip().lower().rstrip(".") in {
            "face", "eyes", "eye", "hair", "skin", "body", "build", "height",
            "sex", "attire", "carried", "notes", "species", "age"}:
        return False, "bare-noun"
    if field == "sex":
        toks = re.findall(r"[a-z]+", value.lower())
        if len(toks) > 3 or not (set(toks) & SEX_WORDS):
            return False, "not-a-sex"
    stripped = value.strip().strip(",;.")
    if re.search(r"\b(of|with|in|on|at|and|or|to|from|by)$", stripped, re.I):
        return False, "trailing-connector"
    if field not in ("attire", "carried"):
        lead = re.match(
            r"^([a-z]{3,}ing)\s+(a|an|the|his|her|their|its|old|new|young|dark|"
            r"long|short)\b", value, re.I)
        if (lead and lead.group(1).lower() not in ADJ_OK
                and lead.group(1).lower() not in NOUN_OK):
            return False, "verb-phrase"
    rule = attribute_shaped(field, value)
    if rule:
        return False, rule
    other = other_character_named(fact["subject"], evidence, subjects)
    if other:
        return False, f"other-character:{other}"
    return True, ""


# --- sex from pronouns ------------------------------------------------------

MALE_PRON = re.compile(r"\b(he|him|his|himself)\b", re.I)
FEMALE_PRON = re.compile(r"\b(she|her|hers|herself)\b", re.I)
TITLE_WORDS = {"queen", "king", "prince", "princess", "lord", "lady", "master",
               "mistress", "captain", "sir", "dame", "father", "mother",
               "sister", "brother", "general", "commander", "sergeant",
               "lieutenant", "doctor", "guard", "guards", "guardsman"}


def name_index(sentences: Sequence[str], names: Sequence[str]):
    """``({name: [sentence indices]}, lowercased sentences, per-sentence counts)``.

    Names are matched case-sensitively; pronouns are matched against the lowercased
    copy. Built once per book because the alternative is a scan per name per sentence.
    """
    patterns = {}
    for name in names:
        parts = {x for x in re.findall(r"[A-Za-z\u2019']{3,}", name)
                 if x.lower() not in TITLE_WORDS}
        if parts:
            patterns[name] = re.compile("|".join(
                rf"\b{re.escape(x)}\b" for x in sorted(parts)))
    lows = [_apos(s).lower() for s in sentences]
    idx = {name: [] for name in patterns}
    for i, sentence in enumerate(sentences):
        sentence = _apos(sentence)
        for name, pattern in patterns.items():
            if pattern.search(sentence):
                idx[name].append(i)
    counts = [0] * len(lows)
    for hits in idx.values():
        for i in hits:
            counts[i] += 1
    return idx, lows, counts


def character_index(sentences: Sequence[str], forms_by_subject: Dict[str, Iterable[str]]):
    """``({subject: [sentence indices]}, lowercased sentences, per-sentence counts)``.

    Built from each form's FIRST non-title token. The older `name_index` counted
    name FORMS, so a sentence saying "Rand" matched both the "Rand" and the
    "Rand al'Thor" patterns - and the family token `al'Thor` matched Kari and Tam
    as well. That made "a sentence naming exactly one character" almost never
    true, which is why the sex vote abstained on Rand, Mat and most of the cast.
    One pattern per character is what the vote actually needs.
    """
    patterns = {}
    for subject, forms in forms_by_subject.items():
        parts = set()
        for form in forms:
            toks = [t for t in re.findall(r"[A-Za-z\u2019']{3,}", form)
                    if t.lower() not in TITLE_WORDS and t.lower() not in _NAME_STOP]
            if toks:
                parts.add(toks[0])
        if not parts:
            parts = {t for form in forms
                     for t in re.findall(r"[A-Za-z\u2019']{3,}", form)}
        if parts:
            patterns[subject] = re.compile(
                "|".join(rf"\b{re.escape(p)}\b" for p in sorted(parts)))
    lows = [_apos(s).lower() for s in sentences]
    idx = {subject: [] for subject in patterns}
    counts = [0] * len(lows)
    for i, sentence in enumerate(sentences):
        sentence = _apos(sentence)
        for subject, pattern in patterns.items():
            if pattern.search(sentence):
                idx[subject].append(i)
                counts[i] += 1
    return idx, lows, counts


def derive_sex(name: str, lows, idx, counts, min_evidence: int = 6,
               margin: int = 2) -> Optional[str]:
    """'male'/'female' from the pronouns of sentences that can only be about them.

    A sentence counts only when it names exactly one known character, so a pronoun
    in it cannot belong to someone else. It abstains on thin evidence because a
    wrong sex is the worst error the extractor can make.
    """
    male = female = 0
    for i in idx.get(name) or []:
        if counts[i] != 1:
            continue
        male += len(MALE_PRON.findall(lows[i]))
        female += len(FEMALE_PRON.findall(lows[i]))
    if male + female < min_evidence or male == female:
        return None
    if min(male, female) * margin > max(male, female):
        return None
    return "male" if male > female else "female"


# --- parsing -----------------------------------------------------------------

def parse_quotes(raw: str) -> List[dict]:
    out = []
    for match in _OBJ.finditer((raw or "").replace("\\'", "'").replace("`", "")):
        try:
            obj = json.loads(match.group(0))
        except Exception:
            continue
        if isinstance(obj, dict) and "quote" in obj:
            out.append(obj)
    return out


def parse_names(raw: str) -> List[Tuple[str, str]]:
    out = []
    for match in _OBJ.finditer((raw or "").replace("\\'", "'").replace("`", "")):
        try:
            obj = json.loads(match.group(0))
        except Exception:
            continue
        if isinstance(obj, dict) and "name" in obj:
            out.append((str(obj.get("reference", "")).strip(),
                        str(obj.get("name", "")).strip()))
    return out


def parse_facts(raw: str) -> List[dict]:
    raw = (raw or "").replace("\\'", "'").replace("`", "")
    out = []
    for match in re.finditer(r"\{[^{}]*\}", raw, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if isinstance(obj.get("facts"), list):
            out += [x for x in obj["facts"] if isinstance(x, dict)]
        elif "subject" in obj or "field" in obj:
            out.append(obj)
    return out


def even_sample(items: Sequence, cap: int) -> List:
    """At most ``cap`` items, spread evenly, keeping their order.

    Sampling matters for a character who is described early and late: the first
    100 quotes of a 554-quote character are all from the first half of the book.
    """
    items = list(items)
    if len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def parse_sex(raw: str) -> str:
    """``"male"`` / ``"female"`` / ``""`` from a short reply."""
    raw = raw or ""
    match = re.search(r'"sex"\s*:\s*"(male|female)"', raw, re.I)
    if match:
        return match.group(1).lower()
    match = re.search(r"\b(male|female)\b", raw, re.I)
    return match.group(1).lower() if match else ""


async def resolve_sex_llm(candidates: Sequence[str], by_subject: Dict[str, List[dict]],
                          llm_url: str, model: str, counts=None) -> Dict[str, str]:
    """Ask the model directly for the sex of characters the pronoun vote could not settle.

    The vote is a good guard and a poor decider: it abstains unless one sex leads
    the other 2:1 in sentences that name exactly one character, and a sentence
    naming one character still has pronouns for everybody else in it. Measured on
    the test book it returned one `female` in 112 subjects while getting no answer
    wrong. One short call per unresolved character costs ~0.3 s and is right on
    every character with three or more quotes.

    ``unknown`` is passed through as no answer, not as a value: a wrong sex is
    worse than a blank one.
    """
    counts = counts if counts is not None else collections.Counter()
    out: Dict[str, str] = {}
    for name in candidates:
        quotes = by_subject.get(name) or []
        if not quotes:
            continue
        lines = "\n".join(f'- "{q["quote"]}"'
                          for q in even_sample(quotes, SEX_QUOTE_CAP))
        prompt = SEX_PROMPT.format(name=name, lines=lines)
        raw, finish = await _llm(prompt, llm_url, model, SEX_MAX_TOKENS, SEX_TIMEOUT)
        if finish == "error":
            counts["sex-llm-error"] += 1
            continue
        sex = parse_sex(raw)
        if sex in ("male", "female"):
            out[name] = sex
            counts["sex-llm"] += 1
        else:
            counts["sex-llm-abstained"] += 1
    return out


# --- LLM ---------------------------------------------------------------------

async def _llm(prompt: str, llm_url: str, model: str, max_tokens: int,
               timeout: float) -> Tuple[str, str]:
    """``(text, finish_reason)``.

    ``finish_reason`` is what makes a truncation visible. The stage parsers can
    only recover complete JSON objects, so a reply cut off at ``max_tokens``
    looks exactly like a model with less to say - the tail is simply absent.
    """
    from services.prompt_engine import get_llm_response
    return await get_llm_response(
        prompt, llm_url, model, temperature=0.0,
        extra_payload={"max_tokens": max_tokens,
                       "chat_template_kwargs": {"enable_thinking": False}},
        timeout=timeout, with_meta=True)


# --- stage 1: windows and gather ---------------------------------------------

def plan_group_windows(groups, sentences: Sequence[str], budget: int,
                       chapters: Sequence[int]) -> List[List[int]]:
    """Consecutive WHOLE groups, packed to ``budget``, never across a chapter.

    Packing groups (not scenes) is what guarantees every group is presented to
    exactly one call. The probe packed scenes and then kept only groups wholly
    inside each window, so a group that straddled a window edge was silently
    extracted nowhere.
    """
    windows: List[List[int]] = []
    current: List[int] = []
    words = 0
    for gi, (first, last, _setting) in enumerate(groups, 1):
        size = sum(len(x.split()) for x in sentences[first - 1:last])
        chapter = chapters[first - 1]
        starts_new = bool(current) and chapters[groups[current[0] - 1][0] - 1] != chapter
        if current and (starts_new or words + size > budget):
            windows.append(current)
            current, words = [], 0
        current.append(gi)
        words += size
    if current:
        windows.append(current)
    return windows


def _window_text(groups, window: Sequence[int], scenes,
                 sentences: Sequence[str]) -> str:
    parts = []
    for local, gi in enumerate(window, 1):
        first, last, ids = groups[gi - 1]
        labels = []
        for scene_id in ids:
            label = scenes[scene_id - 1][2]
            if label and label not in labels:
                labels.append(label)
        parts.append(f"SCENE {local} ({'; '.join(labels) or 'unlabelled'})")
        parts.append("\n".join(
            sentences[i - 1].replace("\n", " ").strip()
            for i in range(first, last + 1)
            if sentences[i - 1].strip() != scene_map.CHAPTER_MARKER))
    return "\n".join(parts)


def _locate_group(quote: str, groups, window: Sequence[int],
                  sentences: Sequence[str]) -> Optional[int]:
    """The global group containing the first sentence the quote matches."""
    q = normalise_for_grounding(quote)
    if len(q) < 12:
        return None
    for gi in window:
        first, last, _ids = groups[gi - 1]
        for i in range(first, last + 1):
            if normalise_for_grounding(sentences[i - 1]).startswith(q[:60]):
                return gi
    for gi in window:
        first, last, _ids = groups[gi - 1]
        for i in range(first, last + 1):
            if q[:40] in normalise_for_grounding(sentences[i - 1]):
                return gi
    return None


async def gather_book(book_name: str, sentences: Sequence[str],
                      scenes, groups, chapters: Sequence[int],
                      llm_url: str, model: str, window_words: int,
                      log=None, book_dir=None, checkpoint_sha: str = "",
                      should_cancel=None) -> Dict[str, Any]:
    """Stage 1. Returns quotes, dropped references, counts and windows.

    One checkpoint per window: this stage is the longest single run in the
    pipeline (~22 min, ~55 model calls), and a window is the smallest unit whose
    answer is worth keeping - its quotes are absolute (chapter + global group),
    so they slot straight back in on resume.
    """
    say = log or (lambda *_a, **_k: None)
    windows = plan_group_windows(groups, sentences, window_words, chapters)

    quotes: List[dict] = []
    dropped: List[dict] = []
    counts: collections.Counter = collections.Counter()
    title = _book_title(book_name)
    done = (checkpoints.load(book_dir, "gather", checkpoint_sha)
            if book_dir and checkpoint_sha else {})
    reused = 0

    for wi, window in enumerate(windows, 1):
        if str(wi) in done:
            row = done[str(wi)] or {}
            quotes.extend(row.get("quotes") or [])
            dropped.extend(row.get("dropped") or [])
            counts.update(row.get("counts") or {})
            reused += 1
            continue
        checkpoints.stopped(should_cancel)
        n_quotes, n_dropped = len(quotes), len(dropped)
        before = collections.Counter(counts)

        def save_window() -> None:
            if book_dir and checkpoint_sha:
                checkpoints.add(book_dir, "gather", checkpoint_sha, str(wi), {
                    "quotes": quotes[n_quotes:],
                    "dropped": dropped[n_dropped:],
                    "counts": dict(counts - before),
                })

        text = _window_text(groups, window, scenes, sentences)
        prompt = (GATHER_PROMPT.format(book=title) + "\n\nPASSAGES:\n" + text)
        raw, finish = await _llm(prompt, llm_url, model, GATHER_MAX_TOKENS,
                                 GATHER_TIMEOUT)
        if finish == "length":
            counts["gather-truncated"] += 1
        got = parse_quotes(raw)
        if not got:
            counts["unparsed-window"] += 1
            say(f"  [gather] window {wi}/{len(windows)}: no quotes parsed "
                f"({len(raw)} chars)")
            save_window()
            continue
        win_norm = normalise_for_grounding(" ".join(
            sentences[i - 1] for gi in window
            for i in range(groups[gi - 1][0], groups[gi - 1][1] + 1)))
        kept = 0
        for obj in got:
            subject = _clean_name(obj.get("subject"))
            quote = " ".join(str(obj.get("quote", "")).split())
            try:
                local = int(obj.get("scene"))
            except (TypeError, ValueError):
                local = 0
            if not subject or not quote:
                counts["empty-field"] += 1
                dropped.append({"window": wi, "subject": subject, "quote": quote,
                                "reason": "empty-field"})
                continue
            gi = window[local - 1] if 1 <= local <= len(window) else None
            if gi is None:
                gi = _locate_group(quote, groups, window, sentences)
                if gi is not None:
                    counts["scene-corrected"] += 1
            if gi is None:
                counts["scene-unfindable"] += 1
                dropped.append({"window": wi, "subject": subject, "quote": quote,
                                "reason": "scene-unfindable"})
                continue
            first, last, _ids = groups[gi - 1]
            chapter = chapters[first - 1]
            if bad_subject(subject):
                counts["unnamed-subject"] += 1
                dropped.append({"window": wi, "chapter": chapter, "group": gi,
                                "subject": subject, "quote": quote,
                                "reason": "unnamed-subject"})
                continue
            if not evidence_in_text(quote, win_norm):
                counts["quote-not-in-window"] += 1
                dropped.append({"window": wi, "chapter": chapter, "group": gi,
                                "subject": subject, "quote": quote,
                                "reason": "quote-not-in-window"})
                continue
            quotes.append({"window": wi, "chapter": chapter, "group_global": gi,
                           "subject": subject, "quote": quote})
            kept += 1
        save_window()
        say(f"  [gather] window {wi}/{len(windows)}: {len(window)} groups, "
            f"{kept} quotes, {len(got) - kept} dropped")
    if reused:
        say(f"  [gather] resumed: {reused} window(s) reused from the checkpoint")
    say(f"  [gather] {len(quotes)} quotes, {len(dropped)} dropped {dict(counts)}")
    return {"quotes": quotes, "dropped": dropped, "counts": dict(counts),
            "windows": windows, "window_count": len(windows)}


async def resolve_references(book_name: str, gather: dict, groups,
                              sentences: Sequence[str], chapters: Sequence[int],
                              llm_url: str, model: str, log=None,
                              book_dir=None, checkpoint_sha: str = "",
                              should_cancel=None) -> Dict[str, Any]:
    """Stage 2. Name the unnamed-subject references, one narrow call per window.

    Checkpointed per window like stage 1: the recovered quotes are absolute, so
    a window that was answered does not need asking again.
    """
    say = log or (lambda *_a, **_k: None)
    by_window: Dict[int, List[dict]] = collections.defaultdict(list)
    for row in gather.get("dropped", []):
        if row.get("reason") == "unnamed-subject" and row.get("group"):
            by_window[row["window"]].append(row)
    if not by_window:
        return gather

    title = _book_title(book_name)
    counts = collections.Counter(gather.get("counts", {}))
    done = (checkpoints.load(book_dir, "refname", checkpoint_sha)
            if book_dir and checkpoint_sha else {})
    added = 0
    for wi in sorted(by_window):
        if str(wi) in done:
            row = done[str(wi)] or {}
            gather["quotes"].extend(row.get("quotes") or [])
            counts.update(row.get("counts") or {})
            added += len(row.get("quotes") or [])
            continue
        checkpoints.stopped(should_cancel)
        n_quotes = len(gather["quotes"])
        before = collections.Counter(counts)
        rows = by_window[wi]
        refs = []
        for row in rows:
            if row["subject"] not in refs:
                refs.append(row["subject"])
        window = gather["windows"][wi - 1]
        lo = groups[window[0] - 1][0]
        hi = groups[window[-1] - 1][1]
        text = " ".join(sentences[i - 1] for i in range(lo, hi + 1)
                        if sentences[i - 1].strip() != scene_map.CHAPTER_MARKER)
        prompt = REFNAME_PROMPT.format(
            book=title, text=text, refs="\n".join(f"- {r}" for r in refs))
        raw, _finish = await _llm(prompt, llm_url, model, REFNAME_MAX_TOKENS,
                                  REFNAME_TIMEOUT)
        answers = {ref.lower(): name for ref, name in parse_names(raw) if ref}
        for row in rows:
            if ANIMAL_REF.search(row["subject"]):
                counts["reference-animal"] += 1
                continue
            name = answers.get(row["subject"].lower(), "")
            if not name or name.lower() in ("unnamed", "unknown") or bad_subject(name):
                counts["reference-unresolved"] += 1
                continue
            first = groups[row["group"] - 1][0]
            gather["quotes"].append({
                "window": wi, "chapter": chapters[first - 1],
                "group_global": row["group"], "subject": _clean_name(name),
                "quote": row["quote"], "from_reference": row["subject"]})
            counts["reference-recovered"] += 1
            added += 1
        if book_dir and checkpoint_sha:
            checkpoints.add(book_dir, "refname", checkpoint_sha, str(wi), {
                "quotes": gather["quotes"][n_quotes:],
                "counts": dict(counts - before),
            })
    say(f"  [refnames] {added} recovered from references")
    gather = dict(gather)
    gather["counts"] = dict(counts)
    return gather


ANIMAL_REF = re.compile(
    r"\b(mare|stallion|gelding|horse|colt|filly|steed|ram|ewe|lamb|cow|bull|calf|ox|"
    r"goat|kid|pig|sow|boar|dog|bitch|pup|cat|kitten|wolf|bird|hen|cock|chick|goose|"
    r"duck|flock|herd|pack|mule|donkey)\b", re.I)


# --- stage 3: extract --------------------------------------------------------

def extract_slices(quotes: Sequence[dict],
                   cap: int = EXTRACT_QUOTE_CAP) -> List[List[dict]]:
    """Even, contiguous, reading-order slices of at most ``cap`` quotes.

    Even rather than greedy: 130 quotes become 65 + 65, never 120 + 10. The short
    tail is the slice most likely to come back thin, and a full first slice buys
    nothing. Order is preserved because pass 2 reports positions per quote line
    and the resolver applies facts in reading order.
    """
    if not quotes:
        return []
    cap = max(1, cap)
    chunks = -(-len(quotes) // cap)             # ceil
    size = -(-len(quotes) // chunks)            # ceil, so size <= cap
    return [list(quotes[i:i + size]) for i in range(0, len(quotes), size)]


async def extract_book(book_name: str, gather: dict, groups,
                       chapters: Sequence[int], sentences: Sequence[str],
                       llm_url: str, model: str, log=None,
                       sex_model: Optional[str] = None,
                       book_dir=None, checkpoint_sha: str = "",
                       should_cancel=None) -> Dict[str, Any]:
    """Stage 3. Sort each character's quotes into fields, one call per quote slice.

    One checkpoint per character. Characters are independent - each is asked
    only about its own quotes and the subject list is rebuilt from the quotes on
    resume - so a run stopped on character 60 of 118 keeps the 60.

    The sex follow-up is not checkpointed: it is one short call per remaining
    character (~0.3 s each), and a partial answer here would have to be merged
    into a subject record whose shape the writer depends on.
    """
    say = log or (lambda *_a, **_k: None)
    quotes = gather.get("quotes", [])
    names = sorted({q["subject"] for q in quotes})
    counts = collections.Counter()
    canon, _groups, _toks = canonical_map(
        names, collections.Counter(q["subject"] for q in quotes))

    # group -> (chapter, index of that group within the chapter)
    per_chapter: Dict[int, List[int]] = collections.defaultdict(list)
    for gi in range(1, len(groups) + 1):
        per_chapter[chapters[groups[gi - 1][0] - 1]].append(gi)
    position = {gi: (chapter, index)
                for chapter, gis in per_chapter.items()
                for index, gi in enumerate(gis, 1)}

    by_subject: Dict[str, List[dict]] = collections.defaultdict(list)
    for quote in quotes:
        by_subject[canon.get(quote["subject"], quote["subject"])].append(quote)
    for subject in by_subject:
        by_subject[subject].sort(key=lambda q: (
            q["chapter"], position[q["group_global"]][1]))

    subjects = sorted(by_subject)
    title = _book_title(book_name)
    facts: List[dict] = []
    rejects: List[dict] = []
    done = (checkpoints.load(book_dir, "extract", checkpoint_sha)
            if book_dir and checkpoint_sha else {})
    for ci, subject in enumerate(subjects, 1):
        if subject in done:
            row = done[subject] or {}
            facts.extend(row.get("facts") or [])
            rejects.extend(row.get("rejects") or [])
            counts.update(row.get("counts") or {})
            continue
        checkpoints.stopped(should_cancel)
        n_facts, n_rejects = len(facts), len(rejects)
        before = collections.Counter(counts)

        def save_subject() -> None:
            if book_dir and checkpoint_sha:
                checkpoints.add(book_dir, "extract", checkpoint_sha, subject, {
                    "facts": facts[n_facts:],
                    "rejects": rejects[n_rejects:],
                    "counts": dict(counts - before),
                })

        own = by_subject[subject]

        async def ask_raw(quote_subset):
            lines = [subject]
            for quote in quote_subset:
                chapter, group = position[quote["group_global"]]
                lines.append(f'  W{chapter} SCENE {group}: "{quote["quote"]}"')
            prompt = EXTRACT_PROMPT.format(
                book=title, fields=FIELDS_BLOCK, sink=SINK_BLOCK.strip("\n"),
                characters="\n".join(lines))
            raw, finish = await _llm(prompt, llm_url, model, EXTRACT_MAX_TOKENS,
                                     EXTRACT_TIMEOUT)
            if finish == "error":
                counts["llm-error"] += 1
            return parse_facts(raw), finish

        async def ask(quote_subset):
            return (await ask_raw(quote_subset))[0]

        async def ask_slice(quote_subset):
            """One call for a slice, divided in half if the reply was truncated.

            Halving keeps the positional contract: each half is still contiguous
            in reading order, and every line carries its own ``(chapter, scene)``.
            It is the only recovery that does not silently lose the end of a
            character's book, because a truncated reply parses as a shorter one.
            """
            got, finish = await ask_raw(quote_subset)
            if finish != "length":
                return got
            if len(quote_subset) <= EXTRACT_MIN_SLICE:
                # Not worth dividing further; count it so the loss is visible.
                counts["extract-truncated-accepted"] += 1
                return got
            mid = len(quote_subset) // 2
            counts["extract-truncation-split"] += 1
            return (await ask_slice(quote_subset[:mid])
                    + await ask_slice(quote_subset[mid:]))

        def has_evidence(objects):
            return any(str(o.get("evidence", "")).strip() for o in objects)

        slices = extract_slices(own)
        got: List[dict] = []
        for part in slices:
            got += await ask_slice(part)
        if len(slices) > 1:
            counts["extract-characters-sliced"] += 1

        # A mid-sized quote set can come back with facts whose evidence is
        # missing (the model loses the field in a long reply) or with nothing at
        # all. Halving it is a smaller ask and keeps the positional contract.
        if (not got or not has_evidence(got)) and 2 <= len(own) <= 48:
            mid = len(own) // 2
            second = (await ask(own[:mid])) + (await ask(own[mid:]))
            if second and (not got or has_evidence(second)):
                got = second
                counts["extract-retried"] += 1
                say(f"  [extract] {ci}/{len(subjects)} {subject}: recovered by "
                    f"halving {len(own)} quote(s)")
        if not got:
            counts["unparsed-window"] += 1
            say(f"  [extract] {ci}/{len(subjects)} {subject}: no facts parsed")
            save_subject()
            continue
        allowed_positions = {position[q["group_global"]]: q for q in own}
        own_norm = [normalise_for_grounding(q["quote"]) for q in own]
        kept = 0
        for obj in got:
            raw_subject = _clean_name(obj.get("subject"))
            field = str(obj.get("field", "")).strip().lower()
            value = str(obj.get("value", "")).strip()
            evidence = " ".join(str(obj.get("evidence", "")).split())
            if not raw_subject or bad_subject(raw_subject):
                counts["generic-subject"] += 1
                continue
            emitted = canon.get(raw_subject, raw_subject)
            if (_norm_name(emitted) != _norm_name(subject)
                    and _norm_name(raw_subject) != _norm_name(subject)):
                counts["subject-not-in-character"] += 1
                continue
            if not evidence:
                counts["empty-evidence"] += 1
                continue
            if not any(evidence_in_text(evidence, norm) for norm in own_norm):
                counts["evidence-not-in-own-quote"] += 1
                continue
            field = _canonical_field(field, value)
            if field not in ANALYSIS_FIELDS:
                counts[f"unknown-field:{field[:20]}"] += 1
                continue
            pos = None
            try:
                ch = int(obj.get("window"))
                sc = int(obj.get("scene"))
                if (ch, sc) in allowed_positions:
                    pos = (ch, sc)
            except (TypeError, ValueError):
                pass
            if pos is None:
                for quote, norm in zip(own, own_norm):
                    if evidence_in_text(evidence, norm):
                        pos = position[quote["group_global"]]
                        counts["position-inferred"] += 1
                        break
            if pos is None:
                counts["position-unfindable"] += 1
                continue
            fact = {"chapter": pos[0], "group": pos[1], "subject": subject,
                    "field": field, "value": value, "evidence": evidence}
            ok, rule = validate_fact(fact, subjects)
            if not ok:
                counts[rule.split(":")[0]] += 1
                rejects.append({"rule": rule, "subject": subject,
                                "field": field, "value": value[:80]})
                continue
            facts.append(fact)
            kept += 1
        save_subject()
        say(f"  [extract] {ci}/{len(subjects)} {subject}: {kept} facts kept")

    # Sex is not reliably in pass 2. Vote from the pronouns of sentences that can
    # only be about this character; a value the extractor did give is never
    # overwritten. One pattern per character, built from its given-name tokens.
    forms_by_subject: Dict[str, set] = collections.defaultdict(set)
    for quote in quotes:
        if bad_subject(quote["subject"]):
            continue
        forms_by_subject[canon.get(quote["subject"], quote["subject"])].add(
            quote["subject"])
    idx, lows, mention_counts = character_index(sentences, forms_by_subject)
    sex_guess = {}
    for subject in subjects:
        guess = derive_sex(subject, lows, idx, mention_counts)
        if guess:
            sex_guess[subject] = guess

    # The vote only decides the clear cases. For everything else - which on the
    # test book includes most of the women - ask the model directly, one short
    # call each, using only that character's own quotes. Subjects that already
    # carry a pass-2 sex fact are skipped: the fact is applied later anyway.
    settled = set(sex_guess) | {f["subject"] for f in facts if f["field"] == "sex"}
    unresolved = [s for s in subjects
                  if s not in settled and not bad_subject(s) and by_subject.get(s)]
    if unresolved:
        asked = await resolve_sex_llm(unresolved, by_subject, llm_url,
                                      sex_model or model, counts=counts)
        sex_guess.update(asked)
        say(f"  [extract] sex pass: {len(asked)}/{len(unresolved)} resolved "
            f"by the model")

    subject_meta = {}
    for subject in subjects:
        subject_meta[subject] = {
            "quotes": len(by_subject[subject]),
            "facts": sum(1 for f in facts if f["subject"] == subject),
            "sex": sex_guess.get(subject),
        }
    say(f"  [extract] {len(facts)} facts for {len(subjects)} characters, "
        f"{len(rejects)} rejected")
    return {"facts": facts, "rejects": rejects, "counts": dict(counts),
            "subjects": subject_meta, "canonical": canon}


# --- the public analysis entry point ----------------------------------------

async def analyse_book_text(book_name: str, sentences: Sequence[str],
                            para_breaks: Sequence[bool], scenes, chunks,
                            llm_url: str, model: str, window_words: int,
                            log=None, stages=None,
                            gather: Optional[dict] = None,
                            book_dir=None, scene_map_sha: str = "",
                            should_cancel=None) -> Dict[str, Any]:
    """Run gather -> reference names -> extract, or only the stages asked for.

    ``stages`` defaults to all three. Passing a ``gather`` payload with
    ``"gather"`` omitted reuses the existing quotes, which is what makes a
    re-extract cheap - and, more importantly, keeps the quotes the new facts are
    allowed to cite the same ones the old facts cited.

    Per-stage model overrides (``llm_config``) apply on top of ``model``, which
    stays the fallback so a caller can still pin one model for everything.
    """
    say = log or (lambda *_a, **_k: None)
    stages = set(stages) if stages is not None else {"gather", "refname", "extract"}
    chosen = llm_config.overrides()

    def model_for(stage: str) -> str:
        return chosen.get(stage) or model

    chapters = scene_map.chapter_of_unit(sentences)
    groups = scene_map.group_scenes(scenes, sentences, chapters=chapters)
    say(f"[Analysis] '{book_name}': {len(sentences)} sentences, {len(scenes)} scenes, "
        f"{len(groups)} groups")

    if gather is None or "gather" in stages:
        if book_dir and scene_map_sha:
            checkpoints.reset_if_stale(book_dir, "gather", scene_map_sha)
        gather = await gather_book(book_name, sentences, scenes, groups, chapters,
                                   llm_url, model_for("gather"), window_words, log,
                                   book_dir=book_dir, checkpoint_sha=scene_map_sha,
                                   should_cancel=should_cancel)
    if "refname" in stages:
        if book_dir and scene_map_sha:
            checkpoints.reset_if_stale(book_dir, "refname", scene_map_sha)
        gather = await resolve_references(book_name, gather, groups, sentences,
                                          chapters, llm_url, model_for("refname"), log,
                                          book_dir=book_dir,
                                          checkpoint_sha=scene_map_sha,
                                          should_cancel=should_cancel)

    result = {
        "book": book_name,
        "version": ANALYSIS_VERSION,
        "windows": gather.get("window_count", 0),
        "quotes": gather.get("quotes", []),
        "dropped": gather.get("dropped", []),
        "counts": dict(gather.get("counts", {})),
        "groups": groups,
    }
    if "extract" not in stages:
        return result
    gather_sha = pipeline_manifest.digest(result.get("quotes", []))
    if book_dir:
        checkpoints.reset_if_stale(book_dir, "extract", gather_sha)
    extracted = await extract_book(book_name, gather, groups, chapters, sentences,
                                   llm_url, model_for("extract"), log,
                                   sex_model=model_for("sex"),
                                   book_dir=book_dir, checkpoint_sha=gather_sha,
                                   should_cancel=should_cancel)
    result.update({
        "counts": {**gather.get("counts", {}), **extracted.get("counts", {})},
        "facts": extracted["facts"],
        "rejects": extracted["rejects"],
        "subjects": extracted["subjects"],
        "canonical": extracted["canonical"],
    })
    return result


# --- the cast writer ---------------------------------------------------------

def _sex_from_text(text: str) -> str:
    low = (text or "").lower()
    if re.search(r"\b(male|man|men|boy|boyish|masculine|sir|master|father|son|"
                 r"brother|husband)\b", low):
        return "male"
    if re.search(r"\b(female|woman|women|girl|girlish|feminine|lady|madam|mistress|"
                 r"mother|daughter|sister|wife)\b", low):
        return "female"
    return ""


def _age_from_text(text: str) -> str:
    low = (text or "").lower()
    if "middle-aged" in low or "middle aged" in low:
        return "middle-aged"
    for word in ("young", "adult", "old"):
        if re.search(rf"\b{word}\b", low):
            return word
    if re.search(r"\b(child|boy|girl|youth|lad|lass)\b", low):
        return "young"
    return ""


def _species_from_text(text: str) -> str:
    low = (text or "").lower()
    for species, syns in SPECIES_SYN.items():
        if any(re.search(rf"\b{re.escape(w)}\b", low) for w in syns):
            return species
    return ""


def _norm_name(name: str) -> str:
    return normalise_for_grounding(normalize_character_text(str(name or "")))


def _match_character(session: Session, project_id: int, book_id: int,
                     name: str, characters: List[Character],
                     by_name: Dict[str, List[Character]],
                     by_alias: Dict[str, List[Character]],
                     links_by_char: Dict[int, set],
                     order_map: Dict[int, int],
                     side_stories: set) -> Character:
    key = _norm_name(name)
    candidates = []
    for char in by_name.get(key, []) + by_alias.get(key, []):
        if char not in candidates:
            candidates.append(char)
    if not candidates:
        char = Character(project_id=project_id, name=name, origin="Extraction")
        session.add(char)
        session.flush()
        characters.append(char)
        by_name.setdefault(key, []).append(char)
        links_by_char[char.id] = set()
        return char
    if len(candidates) == 1:
        return candidates[0]
    # Prefer one already linked to this book; then the exact name; then the one
    # linked to the nearest earlier book.
    target_order = order_map.get(book_id, 0)
    linked = [c for c in candidates if book_id in links_by_char.get(c.id, set())]
    if linked:
        candidates = linked
    exact = [c for c in candidates if _norm_name(c.name) == key]
    if exact:
        candidates = exact

    def rank(char):
        orders = [order_map.get(bid, 0) for bid in links_by_char.get(char.id, set())
                  if bid in order_map and bid not in side_stories
                  and order_map.get(bid, 0) < target_order]
        return (max(orders) if orders else -1, char.name == name)
    return max(candidates, key=rank)


def _link_present_characters(session: Session, project_id: int, book_id: int,
                             text: str, characters: List[Character],
                             aliases_by_char: Dict[int, List[str]],
                             links_by_char: Dict[int, set]) -> int:
    """Link a cast member to this book when their name actually appears in it.

    A character who appears but is never described gets no gather quote, so without
    this the book link (which book-aware tag resolution uses) is missing. The scan
    is deliberately conservative: single-word names are case-sensitive, phrases
    are not, and all-lowercase aliases are skipped.
    """
    singles, phrases = set(), set()
    char_by_form: Dict[str, Character] = {}
    for char in characters:
        for form in [char.name] + aliases_by_char.get(char.id, []):
            form = str(form or "").strip()
            if len(form) < 3:
                continue
            key = form.lower()
            char_by_form.setdefault(key, char)
            if " " in form:
                phrases.add(form)
            elif form[0].isupper() and not form.islower():
                singles.add(form)
    matched: set = set()
    if singles:
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(
            sorted((re.escape(x) for x in singles), key=len, reverse=True)) + r")(?!\w)")
        for match in pattern.finditer(text):
            char = char_by_form.get(match.group(0).lower())
            if char:
                matched.add(char.id)
    if phrases:
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(
            sorted((re.escape(x) for x in phrases), key=len, reverse=True)) + r")(?!\w)",
            re.I)
        for match in pattern.finditer(text):
            char = char_by_form.get(match.group(0).lower())
            if char:
                matched.add(char.id)
    added = 0
    for char_id in matched:
        if book_id not in links_by_char.get(char_id, set()):
            session.add(CharacterBookLink(character_id=char_id, book_id=book_id))
            links_by_char.setdefault(char_id, set()).add(book_id)
            added += 1
    return added


def _prior_state(session: Session, char_id: int, book_id: int,
                 order_map: Dict[int, int], books_by_id: Dict[int, Book]) -> Dict[str, str]:
    """The full description in force at the end of the books before this one.

    Side stories are skipped: they inherit but pass nothing forward.
    """
    events = session.exec(
        select(CharacterTimelineEvent)
        .where(CharacterTimelineEvent.character_id == char_id)
    ).all()
    target_order = order_map.get(book_id, 0)
    best, best_key, base = None, None, None
    for event in events:
        if event.book_id is None:
            base = event
            continue
        if event.book_id == book_id:
            continue
        other = books_by_id.get(event.book_id)
        if other is None or getattr(other, "is_side_story", False):
            continue
        order = order_map.get(event.book_id, 0)
        if order >= target_order:
            continue
        key = (order, event.chapter_num, event.shot_num, event.id or 0)
        if best_key is None or key > best_key:
            best_key, best = key, event
    source = best or base
    if source is None:
        return {}
    state = {field: getattr(source, field, None) for field in NEW_TRAIT_FIELDS}
    state = {k: v for k, v in state.items() if v}
    # Old profiler rows have no new fields. Their demographics still carries the
    # base noun, and that is worth keeping.
    demographics = getattr(source, "demographics", None) or ""
    state.setdefault("sex", _sex_from_text(demographics))
    state.setdefault("age", _age_from_text(demographics))
    state.setdefault("species", _species_from_text(demographics))
    return {k: v for k, v in state.items() if v}


def _rendered_tuple(values) -> Tuple[str, ...]:
    """The fields that reach the prompt, lowercased, for change detection.

    ``notes`` is deliberately excluded: a note-only edit must not create a
    timeline event, because nothing in the render path reads it.
    """
    if isinstance(values, dict):
        get = values.get
    else:
        get = lambda field: getattr(values, field, None)
    return tuple((get(field) or "").strip().lower()
                 for field in RENDERED_TRAIT_FIELDS)


def _is_synthetic_book_start(event: CharacterTimelineEvent) -> bool:
    """True for the old writer's ``(1,1)`` book-start marker.

    The old writer seeded sex/species (and inherited state) and wrote a row at
    chapter 1, shot 1 for every character with a fact, whether or not the
    character appears there. A row that carries nothing beyond sex/species is
    that marker, not a description; a real prologue row carries a physical
    trait (attire, build, age, ...) and is kept.
    """
    if event.chapter_num != 1 or event.shot_num != 1:
        return False
    return not any(getattr(event, field, None) for field in
                   ("age", "hair_color", "hair_style", "eye_color", "face",
                    "build", "attire", "carried"))


def _sync_base_event(session: Session, char: Character,
                     order_map: Dict[int, int],
                     side_stories: Optional[set] = None) -> bool:
    """Copy the earliest real description into the character's Base event.

    The Base event is the canonical first appearance: the full snapshot at the
    first position where the extractor actually described the character. It is
    not a synthetic chapter-1 row, and it is never overwritten when the user has
    edited it by hand (``source == "manual"``).

    Side stories are excluded when the character has a main-line event: a side
    story inherits but must not pass its appearance forward as the series base.
    """
    base = session.exec(
        select(CharacterTimelineEvent)
        .where(CharacterTimelineEvent.character_id == char.id)
        .where(CharacterTimelineEvent.book_id == None)  # noqa: E711
    ).first()
    if base is not None and base.source == "manual":
        return False

    events = session.exec(
        select(CharacterTimelineEvent)
        .where(CharacterTimelineEvent.character_id == char.id)
        .where(CharacterTimelineEvent.book_id.is_not(None))
    ).all()
    real_events = [event for event in events if not _is_synthetic_book_start(event)]
    # A character whose only row is the old book-start marker still needs a base;
    # use the marker rather than leaving the profile empty.
    pool = real_events or events
    if not pool:
        return False
    if side_stories:
        main_line = [event for event in pool if event.book_id not in side_stories]
        if main_line:
            pool = main_line
    pool.sort(key=lambda event: (order_map.get(event.book_id, 0),
                                 event.chapter_num, event.shot_num,
                                 event.id or 0))
    first = pool[0]

    if base is None:
        base = CharacterTimelineEvent(
            character_id=char.id, book_id=None, chapter_num=0, shot_num=0,
            label="Base State", source="extraction")
        session.add(base)
    elif base.source in (None, "extraction"):
        base.source = "extraction"

    for field in NEW_TRAIT_FIELDS:
        setattr(base, field, getattr(first, field, None))
    base.book_id = None
    base.chapter_num = 0
    base.shot_num = 0
    base.label = "Base State"
    session.add(base)
    return True



def write_book_analysis(project_id: int, book_id: int, payload: Dict[str, Any],
                        chunks, log=None, chunk_words: int = 350,
                        save_json: bool = True) -> Dict[str, Any]:
    """Match the gathered cast and write the timeline events for one book."""
    say = log or (lambda *_a, **_k: None)
    group_shot = order_by_chapter_shot(chunks)
    facts = payload.get("facts", [])
    subjects = payload.get("subjects", {})
    canonical = payload.get("canonical", {})
    stats = collections.Counter()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        book = session.get(Book, book_id)
        if not project or not book:
            raise ValueError("project or book not found")
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        order_map = {b.id: (b.book_order if b.book_order is not None else 0)
                     for b in books}
        books_by_id = {b.id: b for b in books}
        side_stories = {b.id for b in books if getattr(b, "is_side_story", False)}

        characters = session.exec(
            select(Character).where(Character.project_id == project_id)).all()
        aliases = session.exec(
            select(CharacterAlias).join(Character)
            .where(Character.project_id == project_id)).all()
        links = session.exec(
            select(CharacterBookLink).join(Character)
            .where(Character.project_id == project_id)).all()
        by_name: Dict[str, List[Character]] = collections.defaultdict(list)
        by_alias: Dict[str, List[Character]] = collections.defaultdict(list)
        links_by_char: Dict[int, set] = collections.defaultdict(set)
        aliases_by_char: Dict[int, List[str]] = collections.defaultdict(list)
        for char in characters:
            by_name[_norm_name(char.name)].append(char)
        for alias in aliases:
            by_alias[_norm_name(alias.alias)].append(
                next((c for c in characters if c.id == alias.character_id), None))
            aliases_by_char[alias.character_id].append(alias.alias)
        # Drop the Nones the alias join can produce if a character vanished.
        for key in list(by_alias):
            by_alias[key] = [c for c in by_alias[key] if c is not None]
        for link in links:
            links_by_char[link.character_id].add(link.book_id)

        resolved: Dict[str, Character] = {}
        existing_ids = {c.id for c in characters}
        for subject in sorted(subjects):
            char = _match_character(session, project_id, book_id, subject,
                                    characters, by_name, by_alias, links_by_char,
                                    order_map, side_stories)
            resolved[subject] = char
            stats["characters_matched"] += 1
            # Keep every name form pass 1 used as an alias.
            for form, canon in canonical.items():
                if canon != subject:
                    continue
                key = _norm_name(form)
                if key == _norm_name(char.name):
                    continue
                if key not in {_norm_name(a) for a in aliases_by_char.get(char.id, [])}:
                    session.add(CharacterAlias(character_id=char.id, alias=form))
                    aliases_by_char.setdefault(char.id, []).append(form)
                    by_alias[key].append(char)
            if book_id not in links_by_char.get(char.id, set()):
                session.add(CharacterBookLink(character_id=char.id, book_id=book_id))
                links_by_char.setdefault(char.id, set()).add(book_id)
        session.flush()
        stats["characters_created"] = sum(
            1 for char in resolved.values() if char.id not in existing_ids)
        # Artifacts live under the app's output tree, not the imported source
        # folder that Book.path points at.
        base_dir = Path(get_setting("output_dir", "./output", session)).resolve()
        book_dir = base_dir / project.name / book.name
        transcript_path = book_dir / scene_map.TRANSCRIPT_FILENAME
        if transcript_path.exists():
            text = transcript_path.read_text(encoding="utf-8", errors="ignore")
            stats["links_added"] = _link_present_characters(
                session, project_id, book_id, text, characters, aliases_by_char,
                links_by_char)
        # Timeline per character, at group resolution.
        by_position: Dict[str, Dict[Tuple[int, int], List[dict]]] = collections.defaultdict(
            lambda: collections.defaultdict(list))
        for fact in facts:
            by_position[fact["subject"]][(fact["chapter"], fact["group"])].append(fact)

        for subject, char in resolved.items():
            if char.locked:
                stats["locked_skipped"] += 1
                continue
            own_positions = by_position.get(subject, {})
            if not own_positions:
                continue
            prior = _prior_state(session, char.id, book_id, order_map, books_by_id)
            state: Dict[str, str] = dict(prior)
            meta = subjects.get(subject) or {}
            base_sex = meta.get("sex") or _sex_from_text(prior.get("sex", ""))
            if base_sex:
                state["sex"] = base_sex
            state.setdefault("species", "human")
            if not state.get("sex"):
                state.pop("sex", None)

            # Existing rows: extraction rows are replaced; manual ones are kept.
            existing = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == char.id)
                .where(CharacterTimelineEvent.book_id == book_id)
            ).all()
            manual_events = [ev for ev in existing if ev.source != "extraction"]
            manual_by_position = {(ev.chapter_num, ev.shot_num): ev
                                  for ev in manual_events}
            for ev in existing:
                if ev.source == "extraction":
                    session.delete(ev)
            # Flush the deletes so the base sync below does not see the rows that
            # are being replaced.
            session.flush()

            # The first position is always written: it is the book's "first
            # described" anchor and the source of the Base event. After that only
            # a change in a RENDERED field earns a row; notes-only edits and
            # repeated snapshots are skipped.
            timeline: List[Tuple[int, int, int, Dict[str, str]]] = []
            last_rendered = None
            for position in sorted(own_positions):
                chapter, group = position
                shot = group_shot.get(position, 1)
                manual = manual_by_position.get((chapter, shot))
                if manual is not None:
                    # A manual override at this position wins over extraction and
                    # carries forward into later snapshots.
                    for field in NEW_TRAIT_FIELDS:
                        value = getattr(manual, field, None)
                        if value:
                            state[field] = value
                        else:
                            state.pop(field, None)
                    snapshot = dict(state)
                    timeline.append((chapter, shot, group, snapshot))
                    last_rendered = _rendered_tuple(snapshot)
                    stats["manual_position_skipped"] += 1
                    continue

                # Preserve the model's order within one position; only the
                # position is sorted. Last value wins, as in reading order.
                for fact in own_positions[position]:
                    field, value = fact["field"], fact["value"]
                    current = state.get(field)
                    if current and value.strip().lower() == current.strip().lower():
                        continue
                    if current and value.lower() and _bare_value(value) \
                            and _head_noun(value) == _head_noun(current):
                        stats["guarded:bare-value"] += 1
                        continue
                    state[field] = value

                snapshot = dict(state)
                rendered = _rendered_tuple(snapshot)
                if timeline and rendered == last_rendered:
                    stats["unchanged_positions_skipped"] += 1
                    continue
                timeline.append((chapter, shot, group, snapshot))
                last_rendered = rendered

            for chapter, shot, group, snapshot in timeline:
                if (chapter, shot) in manual_by_position:
                    # The manual row already exists; this timeline entry only
                    # carried its state into later snapshots.
                    continue
                if not any(snapshot.get(f) for f in NEW_TRAIT_FIELDS):
                    continue
                event = CharacterTimelineEvent(
                    character_id=char.id, book_id=book_id, chapter_num=chapter,
                    shot_num=shot, label=f"Ch {chapter} · Shot {shot}",
                    source="extraction",
                    **{field: snapshot.get(field) for field in NEW_TRAIT_FIELDS})
                session.add(event)
                stats["events_written"] += 1

            session.flush()
            if _sync_base_event(session, char, order_map, side_stories):
                stats["base_updated"] += 1

        # A subject with no facts is skipped above, so a resolved sex would be
        # dropped with it. Sex is identity, not a described change, so it belongs
        # on the Base event, which has no position and therefore nothing to get
        # wrong. Fill an empty value only - an existing one is never overruled.
        for subject, char in resolved.items():
            if char.locked or by_position.get(subject):
                continue
            sex = (subjects.get(subject) or {}).get("sex")
            if not sex:
                continue
            base = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == char.id)
                .where(CharacterTimelineEvent.book_id == None)  # noqa: E711
            ).first()
            if base is not None and base.sex:
                continue
            if base is None:
                base = CharacterTimelineEvent(
                    character_id=char.id, book_id=None, chapter_num=0, shot_num=0,
                    label="Base State", source="extraction")
            base.sex = sex
            if not base.species:
                base.species = "human"
            session.add(base)
            stats["base_sex_filled"] += 1

        # The events' shot numbers are the first chunk of their group at THIS
        # chunk size. Prompt generation must not run at a different size without
        # re-analysing, or every description lands a few shots off.
        pipeline_manifest.write(
            book_dir, analysis_chunk_words=int(chunk_words),
            analysis_version=ANALYSIS_VERSION)
        session.commit()
    if save_json:
        save_project_characters_to_json(project_id)
    say(f"[Analysis] wrote {stats.get('events_written', 0)} events for "
        f"{len(resolved)} characters")
    return dict(stats)


def normalize_character_timeline(project_id: int, log=None,
                                 save_json: bool = True) -> Dict[str, int]:
    """Normalize existing extraction rows to the current event rules.

    One-off repair for data written before the Base/change model:

    * drop synthetic ``(1,1)`` book-start rows when a real description exists;
    * drop extraction rows that merely repeat the previous snapshot;
    * drop extraction rows at a position the user has manually overridden;
    * copy the earliest real description into the Base event.

    Manual rows are never deleted. The writer applies these rules to new
    analyses; this only rewrites the rows already on disk.
    """
    say = log or (lambda *_a, **_k: None)
    stats = collections.Counter()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            raise ValueError("project not found")
        books = session.exec(
            select(Book).where(Book.project_id == project_id)).all()
        order_map = {book.id: (book.book_order if book.book_order is not None
                               else 0) for book in books}
        side_stories = {book.id for book in books
                        if getattr(book, "is_side_story", False)}

        characters = session.exec(
            select(Character).where(Character.project_id == project_id)).all()
        for char in characters:
            if char.locked:
                stats["locked_skipped"] += 1
                continue
            events = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == char.id)).all()
            real_exists = any(
                event.book_id is not None and not _is_synthetic_book_start(event)
                for event in events)

            to_delete = set()
            if real_exists:
                for event in events:
                    if (event.book_id is not None
                            and event.source == "extraction"
                            and _is_synthetic_book_start(event)):
                        to_delete.add(event.id)

            kept = [event for event in events if event.id not in to_delete]
            manual_positions = {
                (event.chapter_num, event.shot_num)
                for event in kept
                if event.book_id is not None and event.source != "extraction"}
            for event in kept:
                if (event.book_id is not None
                        and event.source == "extraction"
                        and (event.chapter_num, event.shot_num) in manual_positions):
                    to_delete.add(event.id)
                    stats["manual_collisions_deleted"] += 1

            by_book = collections.defaultdict(list)
            for event in kept:
                if event.book_id is not None and event.id not in to_delete:
                    by_book[event.book_id].append(event)
            for book_events in by_book.values():
                book_events.sort(key=lambda event: (event.chapter_num,
                                                    event.shot_num,
                                                    event.id or 0))
                previous = None
                for event in book_events:
                    rendered = _rendered_tuple(event)
                    if (previous is not None and rendered == previous
                            and event.source == "extraction"):
                        to_delete.add(event.id)
                        stats["duplicates_deleted"] += 1
                        continue
                    previous = rendered

            for event in events:
                if event.id in to_delete:
                    session.delete(event)
                    if (event.book_id is not None
                            and event.source == "extraction"
                            and _is_synthetic_book_start(event)):
                        stats["synthetic_deleted"] += 1
            session.flush()
            if _sync_base_event(session, char, order_map, side_stories):
                stats["base_updated"] += 1
        session.commit()
    if save_json:
        save_project_characters_to_json(project_id)
    say(f"[Analysis] normalized {stats.get('base_updated', 0)} base events, "
        f"deleted {stats.get('synthetic_deleted', 0)} synthetic starts and "
        f"{stats.get('duplicates_deleted', 0) + stats.get('manual_collisions_deleted', 0)} "
        f"duplicates")
    return dict(stats)


def order_by_chapter_shot(chunks) -> Dict[Tuple[int, int], int]:
    """``{(chapter, group): first shot}`` from the chunk list."""
    out: Dict[Tuple[int, int], int] = {}
    for chunk in chunks:
        key = (chunk.chapter, chunk.group)
        if key not in out or chunk.shot < out[key]:
            out[key] = chunk.shot
    return out


def _bare_value(value: str) -> bool:
    words = [w for w in re.findall(r"[a-z']+", value.lower())
             if w not in {"a", "an", "the", "some", "his", "her", "their", "its",
                          "and", "or", "of", "with", "in", "on"} and len(w) > 1]
    return len(words) == 1


def _head_noun(value: str) -> str:
    words = [w for w in re.findall(r"[a-z']+", value.lower())
             if w not in {"a", "an", "the", "some", "his", "her", "their", "its",
                          "and", "or", "of", "with", "in", "on"} and len(w) > 1]
    return words[-1] if words else ""


def analysis_stage_status(book_dir) -> Dict[str, Dict]:
    """Per-stage provenance and whether each stage is still current.

    Answers the question the rollback dialog could not: *which* part of a book is
    out of date, and why. Digests only - nothing here needs a model call, so the
    dashboard can call it while rendering.
    """
    book_dir = Path(book_dir)
    transcript = book_dir / scene_map.TRANSCRIPT_FILENAME
    text = ""
    if transcript.exists():
        text = transcript.read_text(encoding="utf-8", errors="ignore")
    map_path = scene_map.scene_map_path(book_dir)
    gather_path = book_dir / "gather.json"

    checks: Dict[str, Dict] = {
        "segment": {"version": scene_map.SEG_VERSION,
                    "inputs": {"transcript_sha": scene_map.transcript_digest(text),
                               "window_words": scene_map.configured_window_words()}},
        "gather": {"version": 1,
                   "inputs": {"scene_map_sha": pipeline_manifest.digest(
                       map_path.read_text(encoding="utf-8"))
                       if map_path.exists() else ""}},
        "extract": {"version": ANALYSIS_VERSION,
                    "inputs": {"gather_sha": pipeline_manifest.digest(
                        (json.loads(gather_path.read_text(encoding="utf-8")) or {})
                        .get("quotes", [])) if gather_path.exists() else "",
                        "quote_cap": EXTRACT_QUOTE_CAP}},
    }
    status: Dict[str, Dict] = {}
    artifacts = {
        "segment": map_path.exists(),
        "gather": gather_path.exists(),
        "extract": (book_dir / "extract.json").exists(),
    }
    # Partial progress, so the re-run dialog can say "32 windows saved" rather
    # than looking identical to a stage that never started.
    partial = {
        "segment": pipeline_manifest.digest({
            "transcript_sha": scene_map.transcript_digest(text),
            "window_words": scene_map.configured_window_words(),
            "seg_version": scene_map.SEG_VERSION}),
        "gather": checks["gather"]["inputs"]["scene_map_sha"],
        "refname": checks["gather"]["inputs"]["scene_map_sha"],
        "extract": checks["extract"]["inputs"]["gather_sha"],
    }
    for stage, expected in checks.items():
        record = pipeline_manifest.stage_record(book_dir, stage)
        current, reasons = pipeline_manifest.stage_current(book_dir, stage, **expected)
        status[stage] = {"record": record, "current": current, "reasons": reasons,
                         "artifact": artifacts.get(stage, False),
                         "partial": checkpoints.summary(
                             book_dir, stage, partial.get(stage, ""))["done"]}
    return status


async def run_book_analysis(project_id: int, book_id: int, llm_url: str,
                            model: str, chunk_words: int = 350,
                            window_words: Optional[int] = None,
                            log=None, stages=None,
                            should_cancel=None) -> Dict[str, Any]:
    """Load or build the book's stage artifacts, then write the cast.

    ``stages`` selects what to re-run, from ``"gather"`` / ``"refname"`` /
    ``"extract"``. A stage that is not selected is loaded from its artifact, so
    re-extracting reuses the quotes the previous gather produced. The cast is
    written only when ``"extract"`` ran - a gather-only run must not touch the
    database, because it has no facts to write.

    Each stage records what it was built from in ``pipeline.json``, so the next
    run can tell what is stale instead of guessing. ``should_cancel`` is polled
    between units; a stop raises ``checkpoints.Cancelled``, which leaves the
    partials on disk and the database untouched.
    """
    say = log or (lambda *_a, **_k: None)
    stages = set(stages) if stages is not None else {"gather", "refname", "extract"}
    chosen = llm_config.overrides()

    def model_for(stage: str) -> str:
        return chosen.get(stage) or model

    with Session(engine) as session:
        project = session.get(Project, project_id)
        book = session.get(Book, book_id)
        if not project or not book:
            raise ValueError("project or book not found")
        base_dir = Path(get_setting("output_dir", "./output", session)).resolve()
        book_path = base_dir / project.name / book.name
        book_name = book.name
    transcript_path = book_path / scene_map.TRANSCRIPT_FILENAME
    if not transcript_path.exists():
        raise FileNotFoundError(transcript_path)
    text = transcript_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    sentences, para = scene_map.split_sentences(text)
    scenes = scene_map.load_scene_map(scene_map.scene_map_path(book_path), sentences,
                                      transcript=text)
    if not scenes:
        raise ValueError(
            f"'{book_name}' has no current scene map; run scene analysis first")
    chunks, source = scene_map.book_chunks(book_path, chunk_words, text=text)
    if source != "scene_map":
        raise ValueError(f"'{book_name}' is bound to the {source} chunker; "
                         f"analysis requires a scene map")
    window = window_words or scene_map.configured_window_words()
    scene_map_sha = pipeline_manifest.digest(
        scene_map.scene_map_path(book_path).read_text(encoding="utf-8"))

    gather_path = book_path / "gather.json"
    existing_gather = None
    if "gather" not in stages:
        if not gather_path.exists():
            raise ValueError(
                f"'{book_name}' has no gather.json; run quote gathering first")
        saved = json.loads(gather_path.read_text(encoding="utf-8"))
        # ``windows`` in the file is the window COUNT, not the plan itself (the
        # plan is what the gather stage produced and is not needed to re-sort
        # quotes). Older files are the only shape that exists, but accept either.
        saved_windows = saved.get("windows")
        window_count = (len(saved_windows) if isinstance(saved_windows, list)
                        else int(saved_windows or 0))
        existing_gather = {
            "quotes": saved.get("quotes", []),
            "dropped": saved.get("dropped", []),
            "counts": saved.get("counts", {}),
            "windows": [],
            "window_count": window_count,
        }
        say(f"[Analysis] {book_name}: reusing {len(existing_gather['quotes'])} "
            f"gathered quotes")

    payload = await analyse_book_text(book_name, sentences, para, scenes, chunks,
                                      llm_url, model, window, log=log,
                                      stages=stages, gather=existing_gather,
                                      book_dir=book_path,
                                      scene_map_sha=scene_map_sha,
                                      should_cancel=should_cancel)

    # Artifacts for inspection and for the next stages.
    if "gather" in stages or "refname" in stages or existing_gather is None:
        gather_path.write_text(json.dumps({
            "quotes": payload["quotes"], "dropped": payload["dropped"],
            "counts": payload["counts"], "windows": payload["windows"],
            "book": book_name}, indent=2, ensure_ascii=False), encoding="utf-8")
        # The artifact supersedes the partials that built it.
        checkpoints.discard(book_path, "gather")
        checkpoints.discard(book_path, "refname")

    # Provenance. Digests, not timestamps: "did the input change" is the question
    # a later run needs to answer. Recorded before the early return below, so a
    # quotes-only run still leaves a usable record of what it produced.
    try:
        transcript_sha = scene_map.transcript_digest(text)
        scene_map_sha = pipeline_manifest.digest(
            scene_map.scene_map_path(book_path).read_text(encoding="utf-8")) \
            if scene_map.scene_map_path(book_path).exists() else ""
        gather_sha = pipeline_manifest.digest(payload["quotes"])
        if "gather" in stages:
            pipeline_manifest.record_stage(
                book_path, "gather", version=1,
                inputs={"transcript_sha": transcript_sha,
                        "scene_map_sha": scene_map_sha,
                        "window_words": window},
                model=model_for("gather"),
                prompt_sha=pipeline_manifest.digest(GATHER_PROMPT))
        if "refname" in stages:
            pipeline_manifest.record_stage(
                book_path, "refname", version=1,
                inputs={"gather_sha": gather_sha},
                model=model_for("refname"),
                prompt_sha=pipeline_manifest.digest(REFNAME_PROMPT))
    except Exception as ex:  # provenance must never fail a run
        say(f"[Analysis] {book_name}: provenance not recorded ({ex})")

    if "extract" not in stages:
        say(f"[Analysis] {book_name}: quotes only; the cast was not rewritten.")
        return {"quotes": len(payload["quotes"]), "facts": 0, "characters": 0,
                "gather_only": True}

    (book_path / "extract.json").write_text(json.dumps({
        "facts": payload["facts"], "rejects": payload["rejects"],
        "subjects": payload["subjects"], "canonical": payload["canonical"],
        "counts": payload["counts"]}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    checkpoints.discard(book_path, "extract")

    try:
        pipeline_manifest.record_stage(
            book_path, "extract", version=ANALYSIS_VERSION,
            inputs={"gather_sha": gather_sha,
                    "scene_map_sha": scene_map_sha,
                    "quote_cap": EXTRACT_QUOTE_CAP},
            model=model_for("extract"),
            prompt_sha=pipeline_manifest.digest(
                EXTRACT_PROMPT + FIELDS_BLOCK + SINK_BLOCK))
        if (payload.get("counts", {}).get("sex-llm")
                or payload.get("counts", {}).get("sex-llm-abstained")):
            pipeline_manifest.record_stage(
                book_path, "sex", version=1,
                inputs={"gather_sha": gather_sha},
                model=model_for("sex"),
                prompt_sha=pipeline_manifest.digest(SEX_PROMPT))
    except Exception as ex:
        say(f"[Analysis] {book_name}: provenance not recorded ({ex})")

    stats = write_book_analysis(project_id, book_id, payload, chunks, log=log,
                                chunk_words=chunk_words)
    return {"quotes": len(payload["quotes"]), "facts": len(payload["facts"]),
            "characters": len(payload["subjects"]), **stats}
