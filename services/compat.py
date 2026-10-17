"""Read-compatibility shims for on-disk files written before the scene->shot rename.

Vocabulary:
    shot  -- the image unit (one chunk -> one image)
    scene -- the narrative unit (one or more shots)

On-disk artifacts are the *source of truth* in this project and are not migrated
when the code moves on, because doing so would rewrite files that belong to the
user and break older checkouts of the app. Instead every reader accepts both
spellings and every writer emits only the current one.

Use these helpers anywhere a ``prompts.csv``, ``shot_plan.csv`` or
``characters.json`` value is read. Do not hand-roll the lookup: the whole point
is that retired names are listed in exactly one place.
"""

import csv

LEGACY_KEYS = {
    "scene": "shot",
    "scene_num": "shot",
}

# Every spelling a shot index may appear under in a dict, newest first.
SHOT_KEYS = ("shot", "shot_num", "scene", "scene_num")

# POLICY
#   * Readers accept a retired spelling; every dict handed to application code
#     is keyed by the CURRENT name, so callers only ever see ``shot``.
#   * Writers emit only the current name.
#   * A file already on disk is not rewritten just to change its header. It
#     adopts the current spelling the next time the app writes it. An output
#     tree may therefore hold both spellings across different files; readers
#     handle either, which is what makes the rename backwards compatible.
#   * Older checkouts of the app cannot read a file that has already been
#     rewritten. That is accepted: upgrading the app is the expected path, and
#     output/ is per-user rather than shared.


def normalize_key(name: str) -> str:
    """Map one retired field name onto its current spelling."""
    return LEGACY_KEYS.get((name or "").strip().lower(), name)


def normalize_fieldnames(fieldnames) -> list:
    """Normalise a CSV header: lowercase, strip, and map retired names.

    Apply to ``reader.fieldnames`` immediately after constructing a
    ``csv.DictReader`` so that every subsequent dict is keyed consistently.
    """
    return [normalize_key(n) if n else "" for n in (fieldnames or [])]


def normalize_row(row: dict) -> dict:
    """Map a row/record written under retired header names onto current names."""
    return {LEGACY_KEYS.get(k, k): v for k, v in row.items()}


def read_rows(path, delimiter: str = "|"):
    """``csv.DictReader`` that transparently accepts legacy headers.

    Yields dicts keyed by the current column names, so callers can always use
    ``row["shot"]`` regardless of how old the file on disk is.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            yield normalize_row(row)


def get_shot(record: dict, default=0):
    """Pull a shot index out of a dict that may use any historic spelling."""
    for key in SHOT_KEYS:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default
