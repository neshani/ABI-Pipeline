"""Scene map and chunker: transcript -> sentences -> scenes -> groups -> chunks.

This is the coordinate system every later stage keys off, so the rules are
deliberately narrow:

    sentence  the unit the scene map is expressed in (1-based, per book)
    scene     a narrative unit; starts on a sentence, never crosses a chapter
    group     one or more adjacent scenes, chosen so each can be cut into chunks
              that sit near the image target; never crosses a chapter either
    chunk     the image unit. One chunk renders to one image, and its index
              within its chapter is the ``shot`` number in ``prompts.csv``,
              ``shot_plan.csv`` and ``(chapter, shot)`` everywhere else.

The extractor's ``scene`` key is the **group's index within its chapter**, which
is why :func:`chunk_map` reports both the global group index and the per-chapter
one. Mapping a shot to a description is therefore chunk -> group -> (chapter,
scene), never chunk -> scene.

Everything here is pure code. Segmentation (the one LLM call that produces the
scene map) lives in a sibling module and writes the map this module reads.

Chapter numbering matches prompt generation exactly: the transcript is split on
``==CHAPTER==``, empty parts are dropped, and the parts are numbered from 1. On
transcripts that open with a marker - which most of this library's do - that
means the prologue is chapter 1 and a fictional chapter N is record N+1. The
probe's ``chapter_of_unit`` incremented on the marker itself, producing an empty
chapter 1 and shifting every label by one; that is corrected here because
``prompts.csv`` already uses the app's numbering.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from services import checkpoints

# --- defaults ---------------------------------------------------------------
#
# The chunk target is user-settable (it is how a user asks for more or fewer
# images). Every tolerance below is a RATIO of that target, never a word count,
# because "target +/- 100 words" is meaningless at a 150-word target.

TARGET_WORDS = 350
CHUNK_FLOOR_RATIO = 0.5     # a chunk below half the target is a stub
CHUNK_BAND = 0.20           # "on target" means within this share
CHUNK_MAX_RATIO = 1.75      # above this share only when one scene forces it
GROUP_MAX_RATIO = 3.0       # a group is never longer than this many targets...
GROUP_MAX_SCENES = 24       # ...nor this many scenes
SIZE_POWER = 2.0            # how sharply a size deviation is punished
STUB_COST = 50.0            # added to a chunk below the floor: effectively never
PARA_TOL = 0.15
PARA_COST = PARA_TOL ** SIZE_POWER
# A scene boundary carried INSIDE a chunk costs this much. Deliberately the cost
# of a scene that is itself below the minimum chunk size: "absorb a scene exactly
# when leaving it alone would produce a chunk smaller than we are willing to draw."
CROSS_COST = (1.0 - CHUNK_FLOOR_RATIO) ** SIZE_POWER

MERGE_SIMILARITY = 0.4      # Jaccard on content words, for merging scenes
MIN_SCENE_WORDS = 0         # the scene floor is OFF - see the findings doc
SEG_WINDOW_WORDS = 12000    # default segmentation budget, WITHIN one chapter
SEG_CONTEXT_WORDS = 700     # read-only lead-in for a chapter's later windows
SEG_VERSION = 8             # bump when anything changes the scene map's shape

SCENE_MAP_FILENAME = "scene_map.json"


# --- stage 1: sentences -----------------------------------------------------

# Sentence splitting must survive "Mr. Smith", initials ("J. R. R. Tolkien") and
# ellipses. This is a helper, not a requirement: without it we simply get more,
# smaller sentences, which the packer absorbs.
_ABBREV = re.compile(
    r"\b(?:mr|mrs|ms|dr|prof|st|sr|jr|vs|etc|inc|ltd|co|no|vol|ch|pp?|fig|"
    r"capt|sgt|lt|col|gen|rev|hon|messrs|mmes)\.$", re.I)
_INITIAL = re.compile(r"\b[A-Z]\.$")

# Closing marks that belong to the sentence that just ended, so ". " and '.” '
# both terminate. A plain lookbehind on [.!?] + whitespace misses the closing
# quote and silently merges every line of dialogue into the narration after it.
_CLOSERS = "\"'”’»)]}"
# A run of whitespace that is not a paragraph break: NBSP and the other exotic
# spaces these transcripts are full of, normalised to a plain space.
_ODD_SPACE = re.compile(r"[\u00a0\u2000-\u200b\u202f\u205f\u3000\t]")
# A fragment that is nothing but punctuation is not a sentence.
_BARE = re.compile(r"[^A-Za-z0-9]*")


def _split_block(block: str) -> List[str]:
    """Sentences in one paragraph. Splits only at real sentence ends."""
    out, buf, i, n = [], [], 0, len(block)
    while i < n:
        ch = block[i]
        if ch not in ".!?":
            buf.append(ch)
            i += 1
            continue
        if ch == ".":
            # An ellipsis never terminates a sentence. This transcript writes it
            # as ". " (dot, NBSP, dot); "...", ".." and "…" also occur in other
            # books. Consume the whole run.
            j, dots = i, 0
            while j < n:
                if block[j] == ".":
                    dots += 1
                    j += 1
                    continue
                if block[j] == " " and j + 1 < n and block[j + 1] == ".":
                    j += 1
                    continue
                break
            if dots >= 2:
                buf.append(block[i:j])
                i = j
                # before a capital it ends the sentence; before lowercase it is
                # a pause inside one
                m = re.match(r"\s+", block[i:])
                if m:
                    nxt = block[i + m.end():i + m.end() + 1]
                    if nxt.isupper() or nxt in "\"'“‘":
                        out.append("".join(buf).strip())
                        buf = []
                        i += m.end()
                continue
        # the terminator, plus any closing quotes or brackets that belong to it
        k = i + 1
        while k < n and block[k] in _CLOSERS:
            k += 1
        if k < n and not block[k].isspace():
            buf.append(block[i:k])
            i = k
            continue
        out.append(("".join(buf) + block[i:k]).strip())
        buf = []
        i = k + 1 if k < n else k
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    # A fragment of pure punctuation is not a sentence of its own - it is the
    # tail of the one before it, which is what a split ellipsis leaves behind.
    merged = []
    for p in out:
        if merged and _BARE.fullmatch(p):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    # a bare fragment at the very start of a block has nothing before it to join,
    # so it joins the sentence after instead - the invariant is that no sentence
    # is pure punctuation
    if len(merged) > 1 and _BARE.fullmatch(merged[0]):
        merged[1] = merged[0] + " " + merged[1]
        merged.pop(0)
    return merged


def split_sentences(text: str) -> Tuple[List[str], List[bool]]:
    """Sentences, plus a flag for whether a paragraph break preceded each one."""
    sentences, para_break = [], []
    for block in _ODD_SPACE.sub(" ", text).split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for piece in _split_block(block):
            piece = piece.strip()
            if not piece:
                continue
            if sentences and (_ABBREV.search(sentences[-1])
                              or _INITIAL.search(sentences[-1])):
                sentences[-1] = sentences[-1] + " " + piece
                continue
            sentences.append(piece)
            para_break.append(False)
        if sentences:
            para_break[-1] = True
    if para_break:
        para_break[0] = False       # do not treat the start as a break
    return sentences, para_break


# --- stage 2: chapters ------------------------------------------------------

CHAPTER_MARKER = "==CHAPTER=="


def chapter_of_unit(sentences: Sequence[str]) -> List[int]:
    """1-based chapter number for each sentence, matching ``prompts.csv``.

    A transcript with no markers is one chapter, which is correct for it: the
    rule is "a window never spans a chapter", not "there must be chapters".

    A marker OPENS the chapter that follows it. A leading marker (the transcript
    begins with one) opens chapter 1, and consecutive markers add no chapter,
    because an empty transcript block is not a block at all.
    """
    out: List[int] = []
    chapter = 1
    has_content = False
    for sentence in sentences:
        if sentence.strip().startswith(CHAPTER_MARKER):
            if has_content:
                chapter += 1
            has_content = False
            out.append(chapter)
            continue
        has_content = True
        out.append(chapter)
    return out


def chapter_breaks(sentences: Sequence[str]) -> set:
    """0-based indices of the sentences that begin a new chapter."""
    chapters = chapter_of_unit(sentences)
    return {i for i in range(1, len(chapters)) if chapters[i] != chapters[i - 1]}


def window_plan(units: Sequence[str], hard_breaks: Optional[set] = None,
                budget: Optional[int] = None) -> List[Tuple[int, List[int]]]:
    """``[(first_index, [unit indices])]`` - windows that never span a chapter.

    A unit is whatever :func:`split_sentences` numbers: a sentence. A chapter
    bigger than ``budget`` is split, and the later windows of it are
    continuations. ``budget`` defaults to :data:`SEG_WINDOW_WORDS`, read at call
    time so the constant stays tunable by settings.
    """
    hard_breaks = set(hard_breaks or ())
    budget = SEG_WINDOW_WORDS if budget is None else budget
    windows, cur, cur_words, start = [], [], 0, 0
    for i, s in enumerate(units):
        if cur and (i in hard_breaks or cur_words >= budget):
            windows.append((start, cur))
            start, cur, cur_words = i, [], 0
        cur.append(i)
        cur_words += len(s.split())
    if cur:
        windows.append((start, cur))
    return windows


# --- furniture: chapter machinery that is not prose -------------------------
#
# The transcript convention is a "==CHAPTER==" block separator, then a "CHAPTER n"
# line, then the chapter title on its own line. None of that is narrative, and all
# of it was being handed to the shot planner and the prompt model as though it
# were. Measured on the test book: 55 markers + 53 "CHAPTER n" lines + their
# titles, appearing inside chunk text and in the passage a prompt is built from.

HEADING_LINE = re.compile(r"^\s*(==\s*CHAPTER\s*==|CHAPTER\s*\d*)\s*$", re.I)


def heading_shaped(sentence: str, max_words: int = 12) -> bool:
    """A chapter title: short, and with no sentence-ending punctuation."""
    s = (sentence or "").strip()
    return (0 < len(s.split()) <= max_words
            and not re.search(r"[.!?][\"'\u2019\u201d]?$", s))


def furniture_indices(sentences: Sequence[str]) -> set:
    """1-based indices of sentences that are chapter machinery, not prose.

    A marker or a "CHAPTER n" line, plus the single following title line when it
    is heading-shaped. The shape test is what keeps this safe: a chapter that
    opens straight into prose loses nothing, because a prose sentence is either
    long or punctuated - and over-correction costs good data.
    """
    out = set()
    for i, s in enumerate(sentences):
        if HEADING_LINE.match(s):
            out.add(i + 1)
            if i + 1 < len(sentences) and heading_shaped(sentences[i + 1]):
                out.add(i + 2)
    return out


# --- stage 3: merge ---------------------------------------------------------

STOP = {"in", "on", "at", "the", "a", "an", "of", "and", "near", "inside",
        "outside", "into", "to", "with"}


def similar(a: str, b: str) -> bool:
    ta = {w for w in re.findall(r"[a-z]{3,}", (a or "").lower()) if w not in STOP}
    tb = {w for w in re.findall(r"[a-z]{3,}", (b or "").lower()) if w not in STOP}
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= MERGE_SIMILARITY


def merge_scenes(scenes, hard_breaks: Optional[set] = None):
    """Join neighbouring scenes whose settings match - never across a chapter.

    ``hard_breaks`` holds 0-based unit indices that begin a chapter; the boundary
    after unit L is a chapter break when L is in it.
    """
    hard_breaks = set(hard_breaks or ())
    out = []
    for f, l, st in scenes:
        if (out and similar(out[-1][2], st) and out[-1][1] not in hard_breaks):
            out[-1] = (out[-1][0], l, out[-1][2])
        else:
            out.append((f, l, st))
    return out


def merge_short(scenes, sentences, min_words: int = MIN_SCENE_WORDS,
                hard_breaks: Optional[set] = None):
    """Fold a scene too short to be worth its own shot into a neighbour.

    Sub-floor scenes come from micro-moves (common room -> hallway -> cellar and
    back); they are real moves but not worth a separate image. Never merges
    across a chapter boundary. The floor is 0 by default, see the findings doc.
    """
    hard_breaks = set(hard_breaks or ())
    out = []
    for f, l, st in scenes:
        w = sum(len(s.split()) for s in sentences[f - 1:l])
        if (out and w < min_words and out[-1][1] not in hard_breaks):
            out[-1] = (out[-1][0], l, out[-1][2])
        else:
            out.append((f, l, st))
    # a short FIRST scene merges forward instead - unless its own end is a
    # chapter boundary, in which case it stays as it is
    if len(out) > 1 and out[0][1] not in hard_breaks:
        w0 = sum(len(s.split()) for s in sentences[out[0][0] - 1:out[0][1]])
        if w0 < min_words:
            out[1] = (out[0][0], out[1][1], out[1][2])
            out.pop(0)
    return out


# --- stage 4: scene groups, then chunks -------------------------------------
#
# The old rule was "a chunk never leaves its scene", and because every scene was
# guaranteed a chunk that made every short scene a short image. The rule now is a
# priority ladder - close a chunk at a SCENE edge when the text is near the
# target, otherwise carry on past it and close at the next PARAGRAPH edge, and
# failing that at a SENTENCE edge. A chunk may span several scenes.


def size_cost(words: int, target: int, power: float = SIZE_POWER) -> float:
    """How bad a chunk of this many words is, given the target."""
    return abs(words / target - 1.0) ** power


def best_k(total: int, target: int) -> int:
    """How many chunks a ``total``-word run should be cut into.

    Each chunk should sit as near the target as possible, so this is the ``k``
    minimising ``|total/k - target|``. That is usually ``round(total/target)``
    and differs in one zone, where rounding is simply wrong: a 500-word run at a
    350-word target rounds to 1 - a single 500-word chunk - when two 250s are
    nearer to the target than one 500 is.
    """
    if total <= 0:
        return 1
    hi = max(1, int(total / target) + 2)
    return min(range(1, hi + 1), key=lambda k: (abs(total / k - target), k))


def group_scenes(scenes, sentences, target: int = TARGET_WORDS,
                 cross_cost: Optional[float] = None,
                 max_ratio: float = GROUP_MAX_RATIO,
                 max_scenes: int = GROUP_MAX_SCENES,
                 chapters: Optional[Sequence[int]] = None):
    """Partition consecutive scenes into groups, each cut into whole chunks.

    ``[(first_sentence, last_sentence, (scene_index, ...))]``.

    A group of one scene is a scene that can stand alone; a group of several is a
    run of scenes too small to be worth an image each. The partition is chosen by
    minimising a cost over the whole sequence, which is why it cannot leave a stub
    behind or glue one chunk the wrong way the way a forward-greedy pass does.

    A group never crosses a chapter.

    THIS IS ALSO THE DETAIL-EXTRACTION UNIT: one call per group, so a scene that
    shares a group is never extracted twice, and a scene split into three chunks
    is still one call.
    """
    cross_cost = CROSS_COST if cross_cost is None else cross_cost
    chapters = chapter_of_unit(sentences) if chapters is None else chapters
    n = len(scenes)
    if n == 0:
        return []
    scene_words = [sum(len(x.split()) for x in sentences[f - 1:l])
                   for f, l, _st in scenes]
    scene_chapter = [chapters[f - 1] for f, _l, _st in scenes]
    inf = float("inf")
    dp = [inf] * (n + 1)
    back = [0] * (n + 1)
    dp[0] = 0.0
    for j in range(1, n + 1):
        total = 0
        longest = 0
        for i in range(j - 1, max(-1, j - 1 - max_scenes), -1):
            if scene_chapter[i] != scene_chapter[j - 1]:
                break                        # a group never crosses a chapter
            total += scene_words[i]
            longest = max(longest, scene_words[i])
            if total - longest > max_ratio * target:
                # The cap is on the SMALL scenes a group may absorb, not on its
                # total. A long scene is allowed to take in a short neighbour -
                # that is the whole point - and a short scene at the end of a
                # chapter has no other neighbour it could take.
                break
            if dp[i] == inf:
                continue
            k = best_k(total, target)
            cost = k * size_cost(total / k, target) + cross_cost * (j - 1 - i)
            if dp[i] + cost < dp[j]:
                dp[j] = dp[i] + cost
                back[j] = i
    groups, j = [], n
    while j > 0:
        i = back[j]
        groups.append((scenes[i][0], scenes[j - 1][1],
                       tuple(range(i + 1, j + 1))))
        j = i
    return list(reversed(groups))


def split_group(words: Sequence[int], para_break: Sequence[bool],
                scene_start: Sequence[bool], target: int, k: int, floor: int,
                para_cost: Optional[float] = None,
                power: float = SIZE_POWER,
                max_words: Optional[float] = None) -> List[int]:
    """Cut ONE group into exactly ``k`` chunks. Returns ``k+1`` cut offsets.

    Scene edges outrank paragraph edges, which outrank sentence edges. Nothing
    here needs a special case for a stub or for the chunk count: the boundaries
    are chosen by minimising a cost across the whole group, so the failure mode
    a forward-greedy pass has - a 1-word trailing chunk - cannot arise.

    ``max_words`` bounds the search: a span wider than this is never the better
    answer, so the inner loop stops there. It changes no result, only the time -
    a chapter of 300 tiny paragraphs is otherwise ``k * n^2``.
    """
    para_cost = PARA_COST if para_cost is None else para_cost
    n = len(words)
    k = max(1, min(k, n))
    if k == 1 or n == 1:
        return [0, n]
    pre = [0] * (n + 1)
    for i, w in enumerate(words):
        pre[i + 1] = pre[i] + w
    inf = float("inf")

    def solve(bound: Optional[float]):
        dp = [[inf] * (n + 1) for _ in range(k + 1)]
        back = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 0.0
        for c in range(1, k + 1):
            for i in range(c, n - (k - c) + 1):  # leave a sentence per later chunk
                best, arg = inf, 0
                for j in range(i - 1, c - 2, -1):
                    if bound is not None and pre[i] - pre[j] > bound and j < i - 1:
                        break
                    if dp[c - 1][j] == inf:
                        continue
                    w = pre[i] - pre[j]
                    cost = size_cost(w, target, power)
                    if w < floor:
                        cost += STUB_COST       # a stub is never the better answer
                    if j > 0 and not (para_break[j - 1] or scene_start[j]):
                        cost += para_cost
                    v = dp[c - 1][j] + cost
                    if v < best:
                        best, arg = v, j
                dp[c][i], back[c][i] = best, arg
        return dp, back

    dp, back = solve(max_words)
    if dp[k][n] == inf:
        # The bound pruned every route to the last chunk. It is only ever an
        # optimisation, so pay the full cost rather than return a broken path.
        dp, back = solve(None)
    cuts, i = [n], n
    for c in range(k, 0, -1):
        i = back[c][i]
        cuts.append(i)
    return sorted(cuts)


def blind_chunks(sentences: Sequence[str], target: int = TARGET_WORDS
                 ) -> List[Tuple[int, int]]:
    """What a packer does when the text has no structure: pack to the target.

    ``[(first_sentence, last_sentence)]``. Used only as the comparison baseline.
    """
    out, first, w = [], 1, 0
    for i, s in enumerate(sentences, 1):
        w += len(s.split())
        if w >= target:
            out.append((first, i))
            first, w = i + 1, 0
    if first <= len(sentences):
        out.append((first, len(sentences)))
    return out


# --- stage 2b: segmentation (the only LLM call in this module) --------------
#
# One call per chapter window. Sentences are numbered ABSOLUTELY within the book
# and the model answers with the FIRST sentence of each scene only, so a boundary
# is exact and gaps or overlaps are impossible by construction.
#
# The reply is parsed field by field when it will not parse as a document: one
# unescaped quote inside a value (dialogue is full of them) or a token cap that
# cuts the last object in half must not cost the other forty scenes. A window that
# still yields nothing is halved and re-asked recursively.

SEG_SYS = """You are a continuity reader preparing an illustrated edition of a novel.

You receive the sentences of ONE chapter of a book, in story order, numbered.
A sentence's number is its position in the chapter. Lines reading <BR> mark a
paragraph break: a <BR> is not a sentence, never give its number, never quote it.

Divide the chapter into narrative scenes. A new scene begins when the story moves
to a genuinely different place, or when time or viewpoint clearly breaks.

- A character moving between adjacent rooms, or around one building, during a
  single continuous stretch of action is still ONE scene.
- Do not split on every change of topic, mood, or speaker.
- Do not make one- or two-sentence scenes. A scene is a sustained passage;
  if in doubt, keep it in the current scene.
- Do not invent boundaries to make scenes even in length.
- These boundaries become the illustrations of the book: one illustration per
  scene. So mark the places the story DWELLS in and shows you - the rooms, roads,
  courts and landscapes an illustration would be worth making. A place the story
  merely passes through, steps into for a moment, or glimpses - a doorway, a
  stair, a gateway, a street crossed on the way somewhere - belongs to the scene
  around it; fold it into its neighbour and spend no boundary on it.
- Aim for a scene to be roughly one to three pages: not a passing glimpse, and
  not a whole chapter.
- A setting phrase that needs "and" or "then" is a sign you have merged two
  places into one scene. Name ONE place per scene.

For EACH scene give exactly three things:
- "first": the number of the FIRST sentence of the scene. The first scene always
  begins at sentence 1.
- "setting": a PLACE, as a short noun phrase: "in the cellar of the inn", "on the
  Quarry Road". Never a movement or a verb ("walking into the cellar", "traveling
  through the forest") - a verb in a setting makes a still image move.
- "quote": the opening words of that first sentence, copied exactly.

Output ONLY this JSON and nothing else:
{"scenes": [{"first": 1, "setting": "in a candlelit sickroom", "quote": "..."},
            {"first": 44, "setting": "in a village square", "quote": "..."}]}"""

SEG_TIMEOUT = 900.0     # seconds; a 12k-word window can take minutes
SEG_MAX_TOKENS = 4096   # generous for a short list, but a repetition loop is
                        # unbounded without it, and a truncated reply used to
                        # yield NO onsets for the four largest chapters
SEG_MAX_ONSETS = 500    # more scenes than this in one window is a loop, not prose
SEG_RETRY_DEPTH = 3     # a window that parses to nothing is halved and re-asked
SEG_RETRY_MIN = 40      # sentences; below this, halving cannot help


def _norm(text: str) -> str:
    """Whitespace and curly-quote normalised, lowercased, for quote matching."""
    text = (text.replace("\u2019", "'").replace("\u2018", "'")
            .replace("\u201c", '"').replace("\u201d", '"').replace("\u2014", "--"))
    return re.sub(r"\s+", " ", text).strip().lower()


def locate_quote(quote: str, sentences: Sequence[str], lo: int, hi: int) -> Optional[int]:
    """The sentence number in [lo, hi] whose text opens with ``quote``.

    The model's number is an index over a list it may be hundreds of lines long,
    which is the one thing small models are reliably bad at. The quote is the
    ground truth: we can find the sentence ourselves. Returns None when no
    sentence matches, in which case the caller keeps the model's number.
    """
    q = _norm(quote)
    if len(q) < 12:
        return None
    for take in (len(q), 60, 40, 25, 15):
        frag = q[:take]
        if len(frag) < 12:
            continue
        for i in range(lo, hi + 1):
            if _norm(sentences[i - 1]).startswith(frag):
                return i
    for i in range(lo, hi + 1):                 # opens mid-sentence, or reworded
        if q[:40] in _norm(sentences[i - 1]):
            return i
    return None


# A partial JSON object at the head of a truncated reply, or one with a stray
# quote inside "quote", is still recoverable field by field.
_SCENE_OBJ = re.compile(
    r'\{\s*"first"\s*:\s*(\d+)\s*,\s*"setting"\s*:\s*"([^"]*)"\s*,'
    r'\s*"quote"\s*:\s*"(.*?)"\s*\}', re.DOTALL)


def _recover_objects(raw: str) -> List[dict]:
    """Pull complete scene objects out of a reply that will not parse as a whole.

    Two ways a good reply breaks as a document: one unescaped quote inside a
    quoted value, or the token cap cutting the last object in half. Neither
    should cost us the other forty scenes. Field order is assumed because the
    prompt specifies it.
    """
    out = []
    for m in _SCENE_OBJ.finditer(raw or ""):
        out.append({"first": int(m.group(1)),
                    "setting": m.group(2).replace("\\/", "/"),
                    "quote": m.group(3).replace('\\"', '"').replace("\\n", " ")})
    return out


def extract_json(text: str) -> Optional[dict]:
    """Parse a reply as JSON, recovering complete scene objects when it is broken."""
    t = text or ""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.DOTALL)
    raw = m.group(1) if m else None
    if raw is None:
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j > i:
            raw = t[i:j + 1]
    if raw is not None:
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        for strict in (True, False):      # strict=False allows raw control chars
            try:
                return json.loads(cleaned, strict=strict)
            except Exception:
                pass
    scenes = _recover_objects(raw if raw is not None else t)
    if scenes:
        return {"scenes": scenes}
    return None


async def _call_segmentation(prompt: str, llm_url: str, model: str,
                             timeout: float) -> str:
    """One segmentation request. Imported lazily: prompt_engine imports this
    module for its chunker, so a top-level import would be circular."""
    from services.prompt_engine import get_llm_response
    return await get_llm_response(
        prompt, llm_url, model, temperature=0.0,
        extra_payload={
            "max_tokens": SEG_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )


async def segment(sentences: Sequence[str], hard_breaks: Optional[set] = None,
                  log=None, budget: Optional[int] = None,
                  para_breaks: Optional[Sequence[bool]] = None,
                  llm_url: str = "", model: str = "",
                  timeout: float = SEG_TIMEOUT,
                  book_dir=None, checkpoint_sha: str = "",
                  should_cancel=None
                  ) -> Tuple[List[tuple], dict]:
    """``(scenes, stats)`` - ``[(first_sentence, last_sentence, setting)]``.

    Sentence-level. Every sentence is shown with its ABSOLUTE number, and the
    model is asked only for the FIRST sentence of each scene. One number per
    scene, strictly increasing, means a scene is "from here to the sentence
    before the next scene starts": gaps and overlaps are impossible by
    construction, so there is no stitch left to get wrong.

    ``para_breaks[i]`` True prints a <BR> after that sentence - a hint, never a
    requirement (many transcripts have no paragraphs at all).

    ``book_dir`` + ``checkpoint_sha`` turn on per-window checkpoints, so a run
    that is stopped or killed resumes at the next window instead of re-asking
    for 20 minutes of them. ``should_cancel`` is polled between windows and
    raises ``checkpoints.Cancelled``.
    """
    say = log or (lambda *_a, **_k: None)
    hard_breaks = set(hard_breaks or ())
    para_breaks = list(para_breaks or [False] * len(sentences))
    n_total = len(sentences)
    windows = window_plan(sentences, hard_breaks, budget)
    done = (checkpoints.load(book_dir, "segment", checkpoint_sha)
            if book_dir and checkpoint_sha else {})

    starts: List[Tuple[int, str]] = []
    stats = {"windows": len(windows), "asked": 0, "unparsed": 0, "relative": 0,
             "quote_agreed": 0, "quote_moved": 0, "quote_missed": 0,
             "over_cap": 0, "retried": 0, "retry_helped": 0,
             "still_empty": 0, "split_windows": 0, "onsets": 0, "scenes": 0,
             "reused": 0, "stopped_after": 0}

    async def ask(offset: int, idx: List[int]) -> Tuple[List[Tuple[int, str]], bool]:
        """One window -> its onsets. Returns ``(onsets, parsed_ok)``."""
        lo, hi = idx[0] + 1, idx[-1] + 1
        lines = []
        for i in idx:
            lines.append(f"{i + 1}: {sentences[i]}")
            if para_breaks[i]:
                lines.append("<BR>")
        lead = ""
        if offset and offset not in hard_breaks:
            # a continuation of the same chapter: hand back the text in front of
            # it, so a scene already under way is not re-opened at the window top.
            # It stops at the chapter's first unit - CONTEXT must not reach back
            # across a chapter boundary either.
            base = max([0] + [h for h in hard_breaks if h <= offset])
            take, w, j = [], 0, offset
            while j > base and w < SEG_CONTEXT_WORDS:
                j -= 1
                take.insert(0, sentences[j])
                w += len(sentences[j].split())
            lead = ("\n\nCONTEXT (the text immediately before sentence "
                    f"{lo}; do not extract from it, and do not open a scene "
                    "just because the numbered sentences begin):\n"
                    + " ".join(take))
            stats["split_windows"] += 1
        stats["asked"] += 1
        raw = await _call_segmentation(
            SEG_SYS + lead + "\n\nSENTENCES:\n" + "\n".join(lines),
            llm_url, model, timeout)
        got = extract_json(raw)
        if isinstance(got, list):
            got = {"scenes": got}
        if not isinstance(got, dict):
            stats["unparsed"] += 1
            say(f"    !! window {lo}-{hi}: reply did not parse ({len(raw)} chars, "
                f"{len(idx)} sentences) - suspect a generation loop")
            return [], False
        out: List[Tuple[int, str]] = []
        for scene in (got.get("scenes") or []):
            try:
                first = int(scene["first"])
            except Exception:
                continue
            if not (lo <= first <= hi):
                # a model that renumbers the window from 1 is common; accept it
                # and count how often, so the prompt can be judged on evidence
                if 1 <= first <= len(idx):
                    first = idx[first - 1] + 1
                    stats["relative"] += 1
                else:
                    continue
            quote = str(scene.get("quote", "")).strip()
            if quote:
                hit = locate_quote(quote, sentences, lo, hi)
                if hit is None:
                    stats["quote_missed"] += 1
                elif hit == first:
                    stats["quote_agreed"] += 1
                else:
                    out.append((hit, str(scene.get("setting", "")).strip()))
                    stats["quote_moved"] += 1
                    continue
            out.append((first, str(scene.get("setting", "")).strip()))
        if len(out) > SEG_MAX_ONSETS:
            stats["over_cap"] += 1
            say(f"    !! window {lo}-{hi}: {len(out)} onsets, over the cap - "
                f"keeping the first {SEG_MAX_ONSETS}")
            out = out[:SEG_MAX_ONSETS]
        return out, True

    for w_i, (offset, idx) in enumerate(windows, 1):
        lo, hi = idx[0] + 1, idx[-1] + 1
        if str(w_i) in done:
            starts += done[str(w_i)]
            stats["reused"] += 1
            continue
        checkpoints.stopped(should_cancel)
        say(f"  [segment] window {w_i}/{len(windows)}  sentences {lo}-{hi}")

        async def try_window(part: List[int], depth: int) -> List[Tuple[int, str]]:
            """Ask a window; if it yields nothing, halve and recurse.

            A shorter numbered list is a smaller ask and sidesteps whatever
            tripped the loop. Onsets are absolute, so the pieces concatenate.
            """
            ons, _ok = await ask(part[0], part)
            if ons or depth <= 0 or len(part) < SEG_RETRY_MIN:
                return ons
            stats["retried"] += 1
            mid = len(part) // 2
            say(f"    .. re-asking {part[0]+1}-{part[mid-1]+1} and "
                f"{part[mid]+1}-{part[-1]+1} separately")
            a = await try_window(part[:mid], depth - 1)
            b = await try_window(part[mid:], depth - 1)
            if a or b:
                stats["retry_helped"] += 1
            return a + b

        onsets = await try_window(idx, SEG_RETRY_DEPTH)
        if not onsets and len(idx) >= SEG_RETRY_MIN:
            stats["still_empty"] += 1
            say(f"    !! window {lo}-{hi} produced NO onsets after retries - "
                f"this stretch will carry a single label")
        starts += onsets
        stats["onsets"] += len(onsets)
        if book_dir and checkpoint_sha:
            checkpoints.add(book_dir, "segment", checkpoint_sha, str(w_i), onsets)

    # Onsets only: strictly increasing, in range. Deliberately NOT sorted - the
    # model's list is in narrative order, so an entry that is not greater than the
    # one before it is evidence of a bad answer, and dropping it merges two scenes
    # rather than inventing one in the wrong place.
    #
    # Each chapter is then tiled SEPARATELY, and its earliest onset is clamped to
    # the chapter's first sentence. Without that, a chapter opening the model did
    # not claim (a heading line, say) gets absorbed by the previous chapter's last
    # scene, and that scene crosses a chapter boundary.
    bounds, lo = [], 1
    for hb in sorted(hard_breaks):
        if hb >= lo:
            bounds.append((lo, hb))
        lo = hb + 1
    bounds.append((lo, n_total))

    out, prev, n_out_of_order, n_anchored, n_empty_chapter = [], 0, 0, 0, 0
    for clo, chi in bounds:
        mine = []
        for first, setting in starts:
            if first < clo or first > chi:
                continue
            if first <= prev:
                n_out_of_order += 1
                continue
            mine.append((first, setting))
            prev = first
        if not mine:
            n_empty_chapter += 1
            out.append((clo, chi, ""))        # keep the text; label is unknown
            continue
        if mine[0][0] > clo:
            mine[0] = (clo, mine[0][1])       # this chapter's first scene
            n_anchored += 1
        for k, (first, setting) in enumerate(mine):
            last = mine[k + 1][0] - 1 if k + 1 < len(mine) else chi
            out.append((first, last, setting))
    stats["scenes"] = len(out)
    stats.update({"out_of_order_dropped": n_out_of_order,
                  "chapter_first_onset_anchored": n_anchored,
                  "chapters_with_no_onset": n_empty_chapter})
    return out, stats


# --- the settings that decide how much text one call reads ------------------

CONTEXT_TOKEN_SHARE = 0.6   # of the model's context; the rest is prompt + output


def configured_window_words() -> int:
    """The segmentation/gather window in words, capped by the model's context.

    ``analysis_window_words`` is the measured, tunable value; the context cap
    exists because users bring their own model and a 12,000-word window does not
    fit an 8k context. 0.6 of the context allows for the instruction block, the
    reply, and tokenizer variance.
    """
    from database.connection import get_setting
    try:
        context = int(get_setting("llm_context_tokens", "32768") or 32768)
    except (TypeError, ValueError):
        context = 32768
    try:
        words = int(get_setting("analysis_window_words",
                                str(SEG_WINDOW_WORDS)) or SEG_WINDOW_WORDS)
    except (TypeError, ValueError):
        words = SEG_WINDOW_WORDS
    return max(500, min(words, int(context * CONTEXT_TOKEN_SHARE)))


# --- whole-book entry point -------------------------------------------------

TRANSCRIPT_FILENAME = "transcript.txt"


async def build_book_map(book_dir, llm_url: str, model: str, *, log=None,
                         window_words: Optional[int] = None,
                         floor_words: int = MIN_SCENE_WORDS,
                         force: bool = False, should_cancel=None) -> List[tuple]:
    """The book's scene map, from cache when it is current, otherwise segmented.

    Writes ``scene_map.json`` next to the transcript. The cache is keyed on the
    text, the segmentation version and the window budget, so changing any of
    them rebuilds the map instead of silently reusing a different one.

    A scene never crosses a chapter and the map always tiles the text; an
    invalid reply is repaired by halving the window, and a chapter that still
    yields nothing becomes one scene rather than a hole.

    Per-window checkpoints live beside the map, so a stopped run resumes at the
    next window. They are dropped once the map is written: the map supersedes
    them, and a stale partial next to a current map only invites confusion.
    """
    say = log or (lambda *_a, **_k: None)
    from services import pipeline_manifest

    book_dir = Path(book_dir)
    transcript_path = book_dir / TRANSCRIPT_FILENAME
    if not transcript_path.exists():
        raise FileNotFoundError(transcript_path)
    text = transcript_path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n")
    sentences, para_breaks = split_sentences(text)
    if not sentences:
        raise ValueError(f"no sentences in {transcript_path}")

    window = configured_window_words() if window_words is None else int(window_words)
    cache_path = scene_map_path(book_dir)
    # The checkpoint identity carries everything the window plan depends on, so
    # a partial can never be resumed into a changed transcript or budget.
    segment_sha = pipeline_manifest.digest({
        "transcript_sha": transcript_digest(text),
        "window_words": window,
        "seg_version": SEG_VERSION,
    })
    checkpoints.reset_if_stale(book_dir, "segment", segment_sha)
    checkpoints.stopped(should_cancel)

    def record_segment():
        # Local import: scene_map is imported by pipeline_manifest's callers.
        from services import pipeline_manifest
        try:
            pipeline_manifest.record_stage(
                book_dir, "segment", version=SEG_VERSION,
                inputs={"transcript_sha": transcript_digest(text),
                        "window_words": window,
                        "seg_version": SEG_VERSION},
                model=model)
        except Exception:
            pass  # provenance must never block a map

    if not force:
        cached = load_scene_map(cache_path, sentences, transcript=text,
                                window_words=window)
        if cached:
            words = sum(len(s.split()) for s in sentences)
            say(f"[Scene-Map] reusing the cached map for '{book_dir.name}' "
                f"({len(cached)} scenes, {words} words)")
            checkpoints.discard(book_dir, "segment")
            record_segment()
            return cached

    hard_breaks = chapter_breaks(sentences)
    say(f"[Scene-Map] segmenting '{book_dir.name}': {len(sentences)} sentences, "
        f"{len(hard_breaks) + 1} chapters, {window}-word windows")
    raw, stats = await segment(sentences, hard_breaks=hard_breaks, log=log,
                               budget=window, para_breaks=para_breaks,
                               llm_url=llm_url, model=model,
                               book_dir=book_dir, checkpoint_sha=segment_sha,
                               should_cancel=should_cancel)
    pre_floor = merge_scenes(raw, hard_breaks=hard_breaks)
    scenes = merge_short(pre_floor, sentences, min_words=floor_words,
                         hard_breaks=hard_breaks)
    validate_scene_map(scenes, sentences)
    save_scene_map(cache_path, sentences, scenes, pre_floor, transcript=text,
                   model=model, window_words=window, floor_words=floor_words)
    checkpoints.discard(book_dir, "segment")
    record_segment()
    say(f"[Scene-Map] '{book_dir.name}': {len(raw)} onsets -> "
        f"{len(pre_floor)} after merge -> {len(scenes)} scenes "
        f"({stats['windows']} windows, {stats['still_empty']} empty, "
        f"{stats['quote_moved']} corrected by quote)")
    return scenes


@dataclass(frozen=True)
class Chunk:
    """One image unit. ``shot`` is its 1-based index within its chapter."""
    chapter: int
    shot: int
    group: int              # 1-based group index WITHIN the chapter (extractor `scene`)
    group_global: int       # 1-based group index over the book (diagnostics)
    first_sentence: int
    last_sentence: int
    scenes: Tuple[Tuple[int, float], ...]   # (global scene index, word share)
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


def validate_scene_map(scenes, sentences: Sequence[str]) -> None:
    """Refuse a map that does not tile the text or crosses a chapter.

    A hole in the map means a stretch of the book can never be rendered; an
    overlap means text is rendered twice. Both are silent downstream, so they are
    an error here rather than a warning.
    """
    if not scenes:
        raise ValueError("the scene map is empty")
    if scenes[0][0] != 1:
        raise ValueError(f"the scene map starts at sentence {scenes[0][0]}, not 1")
    previous = 0
    chapters = chapter_of_unit(sentences)
    for first, last, _setting in scenes:
        if first != previous + 1:
            kind = "overlap" if first <= previous else "hole"
            raise ValueError(f"{kind} in the scene map at sentence {previous + 1} "
                             f"(next scene starts at {first})")
        if last < first:
            raise ValueError(f"scene {first}-{last} is inverted")
        if chapters[first - 1] != chapters[last - 1]:
            raise ValueError(f"scene {first}-{last} crosses a chapter boundary")
        previous = last
    if previous != len(sentences):
        raise ValueError(f"the scene map ends at sentence {previous}, but the "
                         f"transcript has {len(sentences)}")


def chunk_map(scenes, sentences: Sequence[str], para_breaks: Sequence[bool],
              target: int = TARGET_WORDS,
              floor: Optional[int] = None) -> List[Chunk]:
    """The image units of a book, in reading order.

    ``scenes`` is ``[(first_sentence, last_sentence, setting)]`` over the same
    sentence list. ``floor`` is the stub floor; ``None`` means half the target.

    Chapter machinery is left out of each chunk's TEXT but still counted in its
    SPAN, so the chunks tile the text exactly while the passage a model sees is
    prose only.
    """
    validate_scene_map(scenes, sentences)
    floor = max(1, round(target * CHUNK_FLOOR_RATIO)) if floor is None else floor
    chapters = chapter_of_unit(sentences)
    drop = furniture_indices(sentences)
    groups = group_scenes(scenes, sentences, target=target, chapters=chapters)

    owner: Dict[int, int] = {}
    for index, (first, last, _setting) in enumerate(scenes, 1):
        for i in range(first, last + 1):
            owner[i] = index

    groups_in_chapter: Counter = Counter()
    shots_in_chapter: Counter = Counter()
    out: List[Chunk] = []
    for group_index, (first, last, group_scene_ids) in enumerate(groups, 1):
        chapter = chapters[first - 1]
        groups_in_chapter[chapter] += 1
        group_in_chapter = groups_in_chapter[chapter]

        segment = sentences[first - 1:last]
        words = [len(s.split()) for s in segment]
        starts = {scenes[s - 1][0] - first for s in group_scene_ids}
        cuts = split_group(
            words, para_breaks[first - 1:last],
            [i in starts for i in range(len(segment))],
            target, best_k(sum(words), target), floor)
        for a, b in zip(cuts, cuts[1:]):
            span = list(range(a, b))
            keep = [x for x in span if (first + x) not in drop]
            if not keep:                     # a chunk of pure machinery
                keep = span
            share: Dict[int, int] = {}
            for x in span:
                scene_index = owner.get(first + x)
                if scene_index is None:
                    raise ValueError(
                        f"sentence {first + x} belongs to no scene - the map "
                        f"passed validation but cannot be attributed")
                share[scene_index] = share.get(scene_index, 0) + words[x]
            total = sum(share.values()) or 1
            shots_in_chapter[chapter] += 1
            out.append(Chunk(
                chapter=chapter,
                shot=shots_in_chapter[chapter],
                group=group_in_chapter,
                group_global=group_index,
                first_sentence=first + a,
                last_sentence=first + b - 1,
                scenes=tuple(sorted((s, round(w / total, 3))
                                    for s, w in share.items())),
                text=" ".join(sentences[first + x - 1] for x in keep),
            ))
    return out


# --- the linear chunker -----------------------------------------------------
#
# This is what numbers ``(chapter, shot)`` now: there is no scene map in the
# pipeline any more, so the image unit is cut from the transcript directly.
#
# The packer is the map chunker's size ladder without the scenes: the same
# ``best_k`` / ``size_cost`` / ``split_group`` cost model, the same paragraph
# preference, the same stub penalty. Only the cut PRIORITY changes - a paragraph
# edge outranks a sentence edge, and there is no scene edge above them.
#
# Two versions exist and a book is bound to one of them by
# ``pipeline_manifest.chunker_version``:
#
#   v1  the greedy paragraph packer this module has always shipped. Frozen: a
#       book that already has prompts.csv was numbered by it, and changing its
#       boundaries would silently re-point every later ``(chapter, shot)``.
#   v2  the even packer. Same rules, but the cut positions are chosen by
#       minimising size deviation across the whole chapter, so a short leftover
#       is absorbed instead of becoming a stub image.
#
# THE RULE WHEN THIS CHANGES: a new packer is a new version, and the old one's
# implementation stays frozen beside it - ``linear_chunks`` dispatches, it does
# not "improve in place". Otherwise every book already numbered by the old packer
# silently maps its shot numbers onto different text.

PARA_SLACK = 0.15   # paragraph window: target +/- 15%  (350 -> 298..402)
SENT_SLACK = 0.05   # sentence window:  target +/- 5%   (350 -> 333..367)

LINEAR_V1 = 1       # greedy paragraph packer, frozen
LINEAR_V2 = 2       # even packer
CHUNKER_VERSION = LINEAR_V2


def smart_chunk_text(text: str, max_chunk_words: int,
                     para_slack: float = PARA_SLACK,
                     sent_slack: float = SENT_SLACK) -> List[str]:
    """Paragraph-aware text chunking with a soft word window.

    Boundaries land on line breaks (``\\n\\n``) wherever possible: a whole
    paragraph is packed as long as it fits inside the paragraph window. Only a
    paragraph that alone exceeds that window is split by sentences. Pass
    ``para_slack=sent_slack=0`` for the old strict behaviour.
    """
    if not text.strip() or max_chunk_words <= 0:
        return []

    target = float(max_chunk_words)
    para_hi = target * (1 + para_slack)
    sent_lo, sent_hi = target * (1 - sent_slack), target * (1 + sent_slack)

    def split_into_sentences(paragraph_text: str) -> List[str]:
        # Split after sentence-ending punctuation + whitespace. Lossless: no
        # characters are dropped and a closing quote stays attached to its
        # sentence.
        if not paragraph_text:
            return []
        return [s for s in re.split(r'(?<=[.!?])\s+', paragraph_text) if s]

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks: List[str] = []
    current_chunk_words: List[str] = []

    def flush() -> None:
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
            current_chunk_words.clear()

    def pack_sentences(sentences: List[str]) -> None:
        for sentence in sentences:
            sentence_words = sentence.split()
            if not sentence_words:
                continue
            if (current_chunk_words
                    and len(current_chunk_words) + len(sentence_words) > sent_hi
                    and len(current_chunk_words) >= sent_lo):
                flush()
            current_chunk_words.extend(sentence_words)

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        if not paragraph_words:
            continue
        if len(paragraph_words) > para_hi:
            # A single oversized paragraph: start a fresh chunk and split it by
            # sentences inside the tighter window.
            flush()
            pack_sentences(split_into_sentences(paragraph))
            continue
        if current_chunk_words and len(current_chunk_words) + len(paragraph_words) > para_hi:
            flush()
        current_chunk_words.extend(paragraph_words)

    flush()
    return chunks


def _pack_chapter(indexes: Sequence[int], sentences: Sequence[str],
                  para_breaks: Sequence[bool], target: int) -> List[Tuple[int, int]]:
    """``[(first, last)]`` 0-based, inclusive, for one chapter's sentences.

    Units are paragraphs, so every cut this can make is a paragraph boundary; a
    paragraph fat enough to starve its neighbours is split into sentence units,
    which the cost model will only use when it has to. The number of chunks comes
    from ``best_k`` and the cut positions from ``split_group``, exactly as the map
    chunker decides them inside a group - which is what evens the sizes out and
    absorbs a short tail instead of leaving it as a stub.
    """
    n = len(indexes)
    if not n:
        return []
    words = [len(sentences[i].split()) for i in indexes]
    para_cap = max(1.0, target * (1.0 + PARA_SLACK))

    unit_words: List[int] = []
    unit_ends_para: List[bool] = []
    unit_first: List[int] = []
    start = 0
    for i in range(n):
        if not para_breaks[indexes[i]] and i < n - 1:
            continue
        span = range(start, i + 1)
        total = sum(words[k] for k in span)
        if total <= para_cap or len(span) == 1:
            unit_words.append(total)
            unit_ends_para.append(True)
            unit_first.append(start)
        else:
            for k in span:
                unit_words.append(words[k])
                unit_ends_para.append(k == i)
                unit_first.append(k)
        start = i + 1

    total = sum(words)
    floor = max(1, round(target * CHUNK_FLOOR_RATIO))
    cuts = split_group(unit_words, unit_ends_para, [False] * len(unit_words),
                       target, best_k(total, target), floor,
                       max_words=target * CHUNK_MAX_RATIO)
    out: List[Tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        first = unit_first[a]
        last = unit_first[b] - 1 if b < len(unit_first) else n - 1
        if out and first <= out[-1][1]:
            out[-1] = (out[-1][0], last)
        else:
            out.append((first, last))
    return out


def linear_chunks(text: str, target: int = TARGET_WORDS,
                  version: int = CHUNKER_VERSION) -> List[Chunk]:
    """Chunks for a book, numbered by the packer named in ``version``.

    ``group`` is 0: there is no group without a scene map. ``first_sentence`` /
    ``last_sentence`` are absolute sentence numbers in the transcript - the same
    coordinate ``chunk_map`` reports - and 0 for the frozen v1 packer, which
    works on text and never learned where its boundaries landed.
    """
    text = text.replace("\r\n", "\n")
    if version == LINEAR_V1:
        chapters = [c for c in text.split(CHAPTER_MARKER) if c.strip()]
        out: List[Chunk] = []
        for chapter_num, body in enumerate(chapters, 1):
            for shot_num, body_chunk in enumerate(smart_chunk_text(body, target), 1):
                out.append(Chunk(chapter=chapter_num, shot=shot_num, group=0,
                                 group_global=0, first_sentence=0, last_sentence=0,
                                 scenes=(), text=body_chunk))
        return out

    sentences, para_breaks = split_sentences(text)
    chapters = chapter_of_unit(sentences)
    drop = furniture_indices(sentences)
    by_chapter: Dict[int, List[int]] = {}
    for i, chapter in enumerate(chapters):
        by_chapter.setdefault(chapter, []).append(i)

    out = []
    for chapter_num in sorted(by_chapter):
        indexes = by_chapter[chapter_num]
        for shot_num, (a, b) in enumerate(
                _pack_chapter(indexes, sentences, para_breaks, target), 1):
            span = list(range(a, b + 1))
            keep = [k for k in span if (indexes[k] + 1) not in drop]
            if not keep:                    # a chunk of pure chapter machinery
                keep = span
            out.append(Chunk(
                chapter=chapter_num,
                shot=shot_num,
                group=0,
                group_global=0,
                first_sentence=indexes[a] + 1,
                last_sentence=indexes[b] + 1,
                scenes=(),
                text=" ".join(sentences[indexes[k]] for k in keep),
            ))
    return out


# --- caching ----------------------------------------------------------------

def transcript_digest(text: str) -> str:
    """Short digest of the transcript, whitespace-normalised."""
    normalised = " ".join(text.replace("\r\n", "\n").split())
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:16]


def scene_map_path(book_dir) -> Path:
    return Path(book_dir) / SCENE_MAP_FILENAME


def load_scene_map(path, sentences: Sequence[str],
                   transcript: Optional[str] = None,
                   window_words: Optional[int] = None) -> Optional[List[tuple]]:
    """The cached scene map, or ``None`` when it is missing or stale.

    A map is keyed on the sentence count, the segmentation version, the
    transcript digest (when the map records one) and the segmentation window
    (when the map records one). The window is part of the key because it decides
    whether a long chapter was split, which changes the map.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("seg_version") != SEG_VERSION:
        return None
    if payload.get("sents") != len(sentences):
        return None
    recorded_digest = payload.get("transcript_sha")
    if recorded_digest and transcript is not None \
            and recorded_digest != transcript_digest(transcript):
        return None
    recorded_window = payload.get("window_words")
    if (window_words is not None and recorded_window is not None
            and int(recorded_window) != int(window_words)):
        return None
    scenes = [tuple(s) for s in payload.get("scenes") or []]
    if not scenes:
        return None
    validate_scene_map(scenes, sentences)
    return scenes


def save_scene_map(path, sentences: Sequence[str], scenes, pre_floor,
                   transcript: str, model: str = "",
                   window_words: int = SEG_WINDOW_WORDS,
                   floor_words: int = MIN_SCENE_WORDS) -> Path:
    """Write the scene map. ``pre_floor`` is kept so the floor can be retuned
    offline with no model calls."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sents": len(sentences),
        "words": sum(len(s.split()) for s in sentences),
        "model": model,
        "seg_version": SEG_VERSION,
        "window_words": int(window_words),
        "floor_words": int(floor_words),
        "merge_similarity": MERGE_SIMILARITY,
        "transcript_sha": transcript_digest(transcript),
        "scenes": [list(s) for s in scenes],
        "pre_floor": [list(s) for s in pre_floor],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def scene_map_signature(book_dir) -> str:
    """Digest of the map's SENTENCE RANGES, for the pipeline manifest.

    A map rewritten with the same ranges (a retuned floor that changed nothing,
    say) is the same coordinate system; one with different ranges is not, and a
    prompts.csv built against it must be rebuilt.
    """
    path = scene_map_path(book_dir)
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        spans = ";".join(f"{int(s[0])}-{int(s[1])}"
                         for s in payload.get("scenes") or [])
    except Exception:
        return ""
    return hashlib.sha1(spans.encode("utf-8")).hexdigest()[:16]


class StaleSceneMap(Exception):
    """A map exists for this book but does not match its transcript any more."""


def chunks_by_chapter(chunks: Sequence[Chunk]) -> Dict[int, List[Chunk]]:
    """``{chapter: [chunk, ...]}``, in reading order."""
    out: Dict[int, List[Chunk]] = {}
    for chunk in chunks:
        out.setdefault(chunk.chapter, []).append(chunk)
    return out


def prompts_have_rows(book_dir) -> bool:
    """True when ``prompts.csv`` has at least one data row.

    A header-only file is a crashed first run, not a generated book, and must not
    bind the book to the linear chunker.
    """
    path = Path(book_dir) / "prompts.csv"
    if not path.exists():
        return False
    try:
        import csv as _csv
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = _csv.reader(fh, delimiter="|")
            next(reader, None)
            return any(row for row in reader)
    except Exception:
        return True     # unreadable: assume it holds work, and refuse to overwrite


def book_source(book_dir, prompts_exist: Optional[bool] = None) -> str:
    """The chunking that owns this book: ``"scene_map"`` or ``"linear"``.

    Retained for books already numbered by a scene map. The pipeline no longer
    builds maps, so every caller in the app passes ``source="linear"``
    explicitly; this remains as the read side for a legacy book and for the
    retired cast path, which positions its facts by group.

    Once a book has prompt rows, the pipeline manifest is authoritative: a book
    generated linearly stays linear even if a map is built later, because
    switching would silently redefine every shot. Before any rows exist nothing
    is at risk, so the project's analysis mode decides: ``fast`` keeps the linear
    chunker even when a map exists, and ``scenes``/``full`` prefer the map. A book
    with rows and no manifest is legacy and stays linear.
    """
    if prompts_exist is None:
        prompts_exist = prompts_have_rows(book_dir)
    if not prompts_exist:
        try:
            from database.connection import get_setting
            mode = str(get_setting("pipeline_mode", "full") or "full")
        except Exception:
            mode = "full"
        if mode == "fast":
            return "linear"
        return "scene_map" if scene_map_path(book_dir).exists() else "linear"
    from services import pipeline_manifest
    recorded = pipeline_manifest.chunking(book_dir)
    if recorded in ("scene_map", "linear"):
        return recorded
    return "linear"


def chunker_version(book_dir, prompts_exist: Optional[bool] = None) -> int:
    """Which linear packer owns this book's ``(chapter, shot)`` numbering.

    A recorded version wins. A book with prompt rows and no record was numbered
    by the greedy v1 packer, and must keep it: re-deriving its chunks under v2
    would point every existing shot at different text. A book with no rows is
    free to use the current packer, so a fresh project always gets v2.
    """
    from services import pipeline_manifest
    recorded = pipeline_manifest.read(book_dir).get("chunker_version")
    try:
        if recorded:
            return int(recorded)
    except (TypeError, ValueError):
        pass
    if prompts_exist is None:
        prompts_exist = prompts_have_rows(book_dir)
    return LINEAR_V1 if prompts_exist else CHUNKER_VERSION


def book_chunks(book_dir, target: int = TARGET_WORDS, log=None,
                text: Optional[str] = None,
                source: Optional[str] = None,
                version: Optional[int] = None) -> Tuple[List[Chunk], str]:
    """``(chunks, source)`` for a book, honoring the chunking it is bound to.

    ``source`` defaults to :func:`book_source`. A book bound to a scene map whose
    map is missing or no longer matches the transcript raises
    :class:`StaleSceneMap` instead of falling back to the linear chunker:
    silently changing what ``(chapter, shot)`` means is exactly the failure this
    exists to prevent. ``version`` picks the linear packer and defaults to the one
    the book is recorded with.
    """
    book_dir = Path(book_dir)
    if text is None:
        transcript = book_dir / TRANSCRIPT_FILENAME
        if not transcript.exists():
            raise FileNotFoundError(transcript)
        text = transcript.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n")
    source = source or book_source(book_dir)

    if source == "scene_map":
        sentences, para_breaks = split_sentences(text)
        path = scene_map_path(book_dir)
        if not path.exists():
            raise StaleSceneMap(
                f"'{book_dir.name}' is recorded as using a scene map, but the map "
                f"is missing; re-run scene analysis")
        scenes = None
        try:
            scenes = load_scene_map(path, sentences, transcript=text)
        except ValueError as ex:
            raise StaleSceneMap(
                f"the scene map in '{book_dir.name}' is invalid: {ex}") from ex
        if scenes is None:
            raise StaleSceneMap(
                f"the scene map in '{book_dir.name}' does not match the current "
                f"transcript or segmentation version; re-run scene analysis")
        chunks = chunk_map(scenes, sentences, para_breaks, target=target)
        if log:
            log(f"[Chunker] chunking from the map: {len(scenes)} scenes -> "
                f"{len(chunks)} chunks")
        return chunks, "scene_map"

    version = chunker_version(book_dir) if version is None else int(version)
    chunks = linear_chunks(text, target, version=version)
    if log:
        log(f"[Chunker] linear v{version} for '{book_dir.name}' -> "
            f"{len(chunks)} chunks")
    return chunks, "linear"
