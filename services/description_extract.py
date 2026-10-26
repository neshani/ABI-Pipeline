"""Traits from one reference block: the shared extraction step.

Every source that ends up as ``## Name`` + prose - a companion volume, a wiki
page, a pasted page, a transcript - goes through this, so the field shapes and
the trimming rules cannot drift between sources.

Two things are deliberate:

* ``face`` is not requested. Prose descriptions of it are transient ("cheeks
  turning pink") and an automated value there is noise, so only the web lookup -
  where the source states a face as a stable feature - fills it.
* ``height`` is requested but never stored. It is folded into ``build`` by
  ``merge_height``, which is the one field that renders.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Optional

from services.book_lookup import CUE, merge_height
from services.web_lookup import sanitise

EXTRACT_FIELDS = ("sex", "species", "age", "height", "build", "hair_color",
                  "hair_style", "eye_color")

PROMPT = (
    "You are given reference material about a single fictional character. "
    "Extract only what the material states about their physical appearance. "
    "Never guess, never add what is typical for the setting, never use your own "
    "memory of the story.\n"
    'Reply with JSON only, no prose and no code fence: {"sex","species","age",'
    '"height","build","hair_color","hair_style","eye_color"} - null for anything '
    "not stated.\n"
    "Shapes: sex = male|female. species = lowercase noun, null for an ordinary "
    "human. age = child|teen|young|adult|middle-aged|old (child and teen change the "
    "noun the image gets, so use them whenever the material says so). height = as "
    "the material says it, including comparisons. build = adjectives only. "
    "hair_color and eye_color = colour words. hair_style = a short phrase."
)


def trim(text: str, max_chars: int = 9000) -> str:
    """The appearance-bearing sentences of a long entry, in order.

    A companion entry for a principal can run to 3,000 words of biography with a
    paragraph of appearance buried in it, so the whole block is not worth sending
    when the first and last sentence usually are.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept, used = [], 0
    for index, sentence in enumerate(sentences):
        if index < 2 or CUE.search(sentence):
            kept.append(sentence)
            used += len(sentence)
            if used >= max_chars:
                break
    return " ".join(kept)


def parse(raw: str) -> Dict[str, str]:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    fields = {k: str(v) for k, v in data.items()
              if k in EXTRACT_FIELDS and v not in (None, "", "null", "None")}
    return merge_height(sanitise(fields))


async def extract_from_text(name: str, text: str, *, llm_url: str, model: str,
                            max_chars: int = 9000,
                            log: Optional[callable] = None) -> Dict:
    """``{ok, fields, error}`` for one block of reference material."""
    from services.prompt_engine import get_llm_response

    body = trim(text, max_chars=max_chars)
    if not body:
        return {"ok": False, "fields": {}, "error": "empty reference text"}
    try:
        raw = await get_llm_response(
            f"{PROMPT}\n\nCharacter: {name}\n\nMaterial:\n{body}",
            llm_url, model, temperature=0.0, timeout=180,
            extra_payload={"max_tokens": 500})
    except Exception as ex:
        return {"ok": False, "fields": {}, "error": f"extraction call failed: {ex}"}
    fields = parse(raw)
    if log:
        log(f"[Extract] {name}: {len(fields)} fields")
    return {"ok": bool(fields), "fields": fields,
            "error": "" if fields else "nothing stated in this material"}
