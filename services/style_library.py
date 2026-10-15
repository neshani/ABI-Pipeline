"""Style-set library: per-shot-role style definitions.

A style set is one JSON file under ``./styles``. Its top-level fields are the
**Character** style - the base - and an optional ``roles`` block carries
per-role deltas over that base::

    {
      "name": "WOT1",
      "workflow": "z_image_turbo_lora2.json",
      "prompt_prefix": "...",
      "prompt_suffix": "...",
      "negative_prompt": "...",
      "overrides": {"16": {"ckpt_name": "..."}},
      "roles": {
        "wide":   {"prompt_prefix": "... wide ..."},
        "insert": {"prompt_prefix": "... insert ...",
                   "overrides": {"20": {"strength_model": 0.6}}}
      }
    }

Character is the base because it is the most common shot; every other role
inherits every field it does not override. Files written before roles existed
carry no ``roles`` key and behave exactly as before.

Resolution is per field: ``effective(set, role)`` returns the base style with the
role's delta applied on top. ``overrides`` (ComfyUI node parameters) merge one
level deep - node by node, parameter by parameter - so a role can change a single
LoRA strength without restating the whole node.

The workflow is deliberately **not** per role: overrides are keyed by ComfyUI
node id, and a node id only means something relative to one graph.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Roles a style can specialise. "character" is the base; the others are deltas.
ROLES = ("character", "wide", "insert", "single")
BASE_ROLE = "character"

# Text fields a role may override, plus the node-override map.
TEXT_FIELDS = ("prompt_prefix", "prompt_suffix", "negative_prompt")

# Keys lifted straight off a style-set file onto the resolved style.
_SET_KEYS = ("workflow",) + TEXT_FIELDS + ("overrides",)


def styles_dir() -> Path:
    return Path("./styles")


def style_path(name: str) -> Path:
    return styles_dir() / f"{name}.json"


def list_style_sets() -> List[str]:
    """Style-set names on disk, plus the implicit ``default`` entry."""
    styles_dir().mkdir(parents=True, exist_ok=True)
    names = [f.stem for f in styles_dir().glob("*.json")]
    if "default" not in names:
        names.append("default")
    return sorted(names)


def load_style_set(name: str) -> Dict[str, Any]:
    """Parse a style set, or ``{}`` when the file is missing or unreadable."""
    path = style_path(name)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_style_set(name: str, data: Dict[str, Any]) -> Path:
    styles_dir().mkdir(parents=True, exist_ok=True)
    path = style_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


# --- role normalisation ----------------------------------------------------

def normalize_role(role: Optional[str]) -> str:
    """Map anything to a known role, defaulting to the base (Character)."""
    key = (role or "").strip().lower()
    return key if key in ROLES else BASE_ROLE


def role_for_row(row: Optional[Dict[str, Any]]) -> str:
    """Role recorded on a ``prompts.csv`` row.

    Books generated before the shot-role column existed have no value here, so
    they resolve to Character - which is the base style, i.e. the old behaviour.
    """
    return normalize_role((row or {}).get("shot_role"))


def shot_roles_in_file(csv_path: Path) -> List[str]:
    """Distinct roles recorded in a ``prompts.csv``, in canonical order.

    Returns ``[]`` for a legacy file with no role column, which the UI reads as
    "only the base style is meaningful for this volume".
    """
    present = set()
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            fieldnames = [n.strip().lower() if n else "" for n in (reader.fieldnames or [])]
            if "shot_role" not in fieldnames:
                return []
            for row in reader:
                value = (row.get("shot_role") or "").strip().lower()
                if value in ROLES:
                    present.add(value)
    except Exception:
        return []
    return [r for r in ROLES if r in present]


# --- resolution ------------------------------------------------------------

def base_style(style_set: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The Character/base half of a style set (everything but ``roles``)."""
    data = style_set or {}
    base = {k: data.get(k) for k in _SET_KEYS if k in data}
    base.setdefault("prompt_prefix", "")
    base.setdefault("prompt_suffix", "")
    base.setdefault("negative_prompt", "")
    base.setdefault("overrides", {})
    return base


def role_deltas(style_set: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The per-role delta map, with any unknown role keys dropped."""
    roles = (style_set or {}).get("roles") or {}
    if not isinstance(roles, dict):
        return {}
    clean: Dict[str, Dict[str, Any]] = {}
    for role, delta in roles.items():
        if role in ROLES and role != BASE_ROLE and isinstance(delta, dict):
            clean[role] = delta
    return clean


def merge_overrides(base: Optional[Dict[str, Any]],
                    delta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge two ``{node_id: {param: value}}`` maps, delta winning per parameter."""
    merged = {nid: dict(fields or {}) for nid, fields in (base or {}).items()}
    for nid, fields in (delta or {}).items():
        merged.setdefault(nid, {}).update(fields or {})
    return merged


def effective(style_set: Optional[Dict[str, Any]], role: Optional[str] = None) -> Dict[str, Any]:
    """The style actually applied for one shot role."""
    role = normalize_role(role)
    base = base_style(style_set)
    delta = role_deltas(style_set).get(role, {})

    out = {
        "workflow": base.get("workflow", ""),
        "prompt_prefix": base.get("prompt_prefix", ""),
        "prompt_suffix": base.get("prompt_suffix", ""),
        "negative_prompt": base.get("negative_prompt", ""),
        "overrides": merge_overrides(base.get("overrides"), delta.get("overrides")),
    }
    for field in TEXT_FIELDS:
        if field in delta:
            out[field] = delta[field]
    return out


def changed_fields(style_set: Optional[Dict[str, Any]], role: str) -> List[str]:
    """Fields a role overrides, for the UI's "overridden" markers."""
    delta = role_deltas(style_set).get(normalize_role(role), {})
    return [f for f in TEXT_FIELDS if f in delta] + (["overrides"] if delta.get("overrides") else [])
