"""The model the pipeline sends its requests to.

A run used to be able to give each stage its own model: segmentation and trait
extraction were worth a large model, the shot-prompt loop is hundreds of cheap
calls, and the sex question is a one-word answer. In practice nobody split them -
one local server holds one model at a time, a run mixes stages, and an override
set months ago and forgotten became an invisible reason for output to change.
There is one model for every stage again.
"""
from __future__ import annotations

from database.connection import get_setting

DEFAULT_SETTING = "llm_model"


def default_model() -> str:
    return str(get_setting(DEFAULT_SETTING, "local-model") or "local-model")


def stage_model(stage: str = "") -> str:
    """The model used by every stage. ``stage`` is accepted but unused.

    Kept as a named function because it reads as intent at the call site ("give
    the planner the planner's model") and because it is the one place a future
    split would live. ``llm_stage_models`` may still exist in an older database;
    it is ignored, not read.
    """
    return default_model()
