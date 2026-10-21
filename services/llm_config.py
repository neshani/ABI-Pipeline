"""Which model each pipeline stage uses.

A full run is no longer one model doing everything. The stages differ in what they
repay: segmentation and trait extraction are the ones worth a large model, the
shot-prompt loop is hundreds of cheap calls, and the sex question is a one-word
answer. ``llm_model`` stays the default for every stage; ``llm_stage_models`` is a
JSON object of overrides, so an empty or missing value means "use the default" and
nothing about an existing install changes until an override is set.

This is also what makes a remote or externally hosted model useful: point the two
expensive stages at it and leave the rest local.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from database.connection import get_setting, set_setting

DEFAULT_SETTING = "llm_model"
OVERRIDE_SETTING = "llm_stage_models"

# Ordered as the pipeline runs them. The labels are for the settings UI.
STAGES: Tuple[Tuple[str, str], ...] = (
    ("segment", "Scene segmentation"),
    ("gather", "Quote gathering"),
    ("refname", "Reference naming"),
    ("extract", "Trait extraction"),
    ("sex", "Sex resolution"),
    ("plan", "Shot planning"),
    ("prompt", "Shot prompts"),
)
STAGE_KEYS: Tuple[str, ...] = tuple(key for key, _label in STAGES)
STAGE_LABELS: Dict[str, str] = dict(STAGES)


def default_model() -> str:
    return str(get_setting(DEFAULT_SETTING, "local-model") or "local-model")


def overrides() -> Dict[str, str]:
    """Per-stage overrides only, with blank values dropped."""
    raw = get_setting(OVERRIDE_SETTING, {}) or {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v).strip() for k, v in raw.items()
            if str(v or "").strip() and str(k) in STAGE_KEYS}


def stage_model(stage: str) -> str:
    """The model for one stage: its override, else the global default."""
    return overrides().get(stage) or default_model()


def stage_models() -> Dict[str, str]:
    """Every stage resolved to the model it will actually use."""
    base = default_model()
    chosen = overrides()
    return {stage: chosen.get(stage, base) for stage in STAGE_KEYS}


def set_stage_model(stage: str, model: str) -> None:
    """Set or clear one stage's override. A blank model clears it."""
    if stage not in STAGE_KEYS:
        raise ValueError(f"unknown stage: {stage}")
    current = dict(get_setting(OVERRIDE_SETTING, {}) or {})
    if not isinstance(current, dict):
        current = {}
    model = (model or "").strip()
    if model:
        current[stage] = model
    else:
        current.pop(stage, None)
    set_setting(OVERRIDE_SETTING, current)


def describe() -> List[str]:
    """One line per stage, for a console log or a tooltip."""
    chosen = overrides()
    base = default_model()
    out = []
    for stage, label in STAGES:
        model = chosen.get(stage, base)
        suffix = "" if stage not in chosen else "  (override)"
        out.append(f"{label}: {model}{suffix}")
    return out
