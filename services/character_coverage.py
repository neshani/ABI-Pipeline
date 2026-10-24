"""Which characters a bulk description step should spend calls on.

The target is a threshold, not "any empty field". Almost no character in a novel
has every trait stated - the last missing field is usually missing because the
book never says it - so chasing the final field for 300 characters buys nothing
and costs a call each. The threshold says "start with the characters who have
nothing, and only widen it deliberately".

``face`` does not count towards coverage: it is a manual field, so counting it
would make every character look one field short of the truth. ``notes`` is not a
trait at all.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

# The traits a source can actually supply, and the ones the renderer compiles.
# `face` is excluded on purpose: manual only.
COVERAGE_FIELDS = ("sex", "species", "age", "build", "hair_color", "hair_style",
                   "eye_color")


def known_fields(event) -> int:
    """How many of the extractable traits this character has."""
    if not event:
        return 0
    total = 0
    for field in COVERAGE_FIELDS:
        value = event.get(field) if isinstance(event, dict) else getattr(event, field, None)
        if value and str(value).strip().lower() not in ("", "null", "none", "unknown"):
            total += 1
    return total


def needs_work(event, threshold: int, *, require_sex: bool = True) -> bool:
    """Should a step look at this character?

    ``require_sex`` keeps a character in scope while their sex is unknown even if
    they are otherwise described: without it the renderer falls back to "a
    person", which is the one omission a viewer notices immediately.
    """
    if known_fields(event) <= max(0, int(threshold)):
        return True
    if require_sex and event:
        sex = event.get("sex") if isinstance(event, dict) else getattr(event, "sex", None)
        if not (sex or "").strip() if isinstance(sex, str) else not sex:
            return True
    return False


def select(rows: Iterable[Dict], threshold: int, *, require_sex: bool = True) -> List[Dict]:
    """Filter ``[{id, name, event}]`` down to the characters a step should take."""
    return [row for row in rows
            if needs_work(row.get("event"), threshold, require_sex=require_sex)]


def coverage_counts(rows: Sequence[Dict], thresholds: Sequence[int] = (0, 1, 2, 3),
                    *, require_sex: bool = True) -> Dict[int, int]:
    """``{threshold: character count}`` for the target selector.

    Shown in the UI next to each option so the cost of widening the net is
    visible before the run rather than afterwards.
    """
    return {int(t): len(select(rows, t, require_sex=require_sex)) for t in thresholds}
