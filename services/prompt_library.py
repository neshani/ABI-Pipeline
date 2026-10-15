"""Shot-role prompt library.

Prompts are plain-text files under ``./prompt_templates/shot_roles``:

    shared.txt             common block; contains a ``<role>`` marker
    shot_direction.txt     the Shot Planner prompt (standalone, variant-capable)
    single.txt             role fragment - used when the Shot Planner is off
    wide.txt               role fragment
    wide_with_figures.txt  a variant of ``wide``
    character.txt          role fragment
    insert.txt             role fragment

A role's effective prompt is ``shared.txt`` with its ``<role>`` marker replaced
by the chosen variant fragment, so the role rules always land ahead of the
``PASSAGE:`` block. ``<text>`` is substituted with the passage at run time.

The Shot Planner is not spliced: it is read standalone by ``shot_planner`` and
selected from its own active variant, like the roles.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from database.connection import get_setting, set_setting

LIB_DIR = Path("./prompt_templates/shot_roles")
_SHARED = "shared.txt"

# The Shot Planner prompt is a standalone slot: it is not spliced into the
# shared block and takes no ``<text>`` marker. Its file stem doubles as its
# variant-slot name, so it uses the same variant machinery as the roles.
PLANNER_SLOT = "shot_direction"

# Shot types the planner may return. NONE is no longer offered to the planner;
# it survives only as the internal sentinel for an unparsed or unanswered line
# (kept so a dead LLM still degrades to the single prompt instead of a guess).
SHOT_TYPES = ("WIDE", "CHARACTER", "INSERT", "NONE")

# Roles that have a prompt fragment. "single" is the planner-off fallback.
SHOT_ROLES = ("single", "wide", "character", "insert")

# Slots the editor shows and the run-time snapshot records: the common block,
# the standalone Shot Planner, then the role fragments.
EDIT_SLOTS = ("shared", PLANNER_SLOT) + SHOT_ROLES

# Planner modes: "auto" uses the planner's shot type; a role name forces one;
# "off" sends everything through the single prompt.
SHOT_MODES = ("auto", "wide", "character", "insert", "off")

_SETTING_ACTIVE = "shot_role_variant_{role}"  # one active variant per role
_SETTING_MODE = "shot_mode_default"           # project-level default mode

_SHOT_TO_ROLE = {"WIDE": "wide", "CHARACTER": "character", "INSERT": "insert"}


def lib_dir() -> Path:
    return LIB_DIR


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def shared_text(variant: Optional[str] = None) -> str:
    """The shared block, using the active shared variant unless one is named."""
    return read_variant("shared", variant or get_active_variant("shared"))


def shot_direction_text() -> str:
    """The active Shot Planner prompt (its own variant slot, no ``<role>``)."""
    return read_variant(PLANNER_SLOT, get_active_variant(PLANNER_SLOT))


def list_variants(role: str) -> List[str]:
    """Variant stems for a role: ``<role>.txt`` plus ``<role>_*.txt``.

    The base variant (``<role>``) is always present and always first.
    """
    if not LIB_DIR.exists():
        return [role]
    stems = {p.stem for p in LIB_DIR.glob(f"{role}*.txt")
             if p.stem == role or p.stem.startswith(role + "_")}
    stems.add(role)
    return [role] + sorted(stems - {role})


def read_variant(role: str, variant: Optional[str] = None) -> str:
    return _read(LIB_DIR / f"{variant or role}.txt")


def get_active_variant(role: str) -> str:
    """Active variant stem for a role; defaults to the base variant."""
    return get_setting(_SETTING_ACTIVE.format(role=role), role)


def set_active_variant(role: str, variant: str) -> None:
    set_setting(_SETTING_ACTIVE.format(role=role), variant)


def assemble(shared: str, fragment: str) -> str:
    """Splice a role fragment into the shared block at its ``<role>`` marker.

    Falls back to plain concatenation when the marker is absent, and to the
    bare fragment when the shared block is empty.
    """
    fragment = fragment.strip("\n")
    if not shared:
        return fragment
    if "<role>" in shared:
        return shared.replace("<role>", fragment)
    return shared.rstrip("\n") + "\n" + fragment + "\n"


def compose(role: str, variant: Optional[str] = None) -> str:
    """``shared.txt`` with ``<role>`` replaced by the active role fragment."""
    return assemble(shared_text(), read_variant(role, variant))


def render(role: str, passage: str, variant: Optional[str] = None) -> str:
    """The exact string sent to the LLM for one passage."""
    return compose(role, variant).replace("<text>", passage)


def render_from(shared: str, fragment: str, passage: str) -> str:
    """Compose from explicit text.

    Used by the editor so unsaved drafts can be tested without being saved,
    which is the sandbox behaviour the old playground had.
    """
    return assemble(shared, fragment).replace("<text>", passage)


def variant_stem(slot: str, name: str) -> str:
    """Turn a user-entered variant name into a file stem for ``slot``."""
    name = name.strip().replace(" ", "_")
    return name if name.startswith(slot) else f"{slot}_{name}"


def save_variant(slot: str, variant: str, text: str) -> Path:
    """Write a variant file for a slot (creating it if new)."""
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    path = LIB_DIR / f"{variant}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def delete_variant(slot: str, variant: str) -> bool:
    """Delete a variant. The base variant (``<slot>.txt``) is never deleted."""
    if variant == slot:
        return False
    path = LIB_DIR / f"{variant}.txt"
    if path.exists():
        path.unlink()
        return True
    return False


def resolve_role(shot_type: Optional[str], mode: str) -> Optional[str]:
    """Role to use for a scene, or ``None`` when the scene is skipped.

    ``mode`` is ``"auto"`` (use the planner's shot type), a forced role
    (``"wide"`` / ``"character"`` / ``"insert"``), or ``"off"`` (the single
    prompt for every chunk).
    """
    if mode == "off":
        return "single"
    if mode in ("wide", "character", "insert"):
        return mode
    shot = (shot_type or "").upper()
    if shot == "NONE":
        return None
    return _SHOT_TO_ROLE.get(shot, "single")


# --- shot mode: per book, with a project-level default ---------------------

def mode_marker_path(project_name: str, book_name: str) -> Path:
    base = Path(get_setting("output_dir", "./output")).resolve()
    return base / project_name / book_name / ".shot_mode"


def get_default_shot_mode() -> str:
    """Project-wide default shot mode."""
    mode = get_setting(_SETTING_MODE, "auto")
    return mode if mode in SHOT_MODES else "auto"


def set_default_shot_mode(mode: str) -> None:
    if mode not in SHOT_MODES:
        raise ValueError(f"unknown shot mode: {mode!r}")
    set_setting(_SETTING_MODE, mode)


def get_shot_mode_override(project_name: str, book_name: str) -> Optional[str]:
    """The per-book override, or ``None`` when the book inherits the default."""
    marker = mode_marker_path(project_name, book_name)
    if marker.exists():
        value = marker.read_text(encoding="utf-8").strip().lower()
        if value in SHOT_MODES:
            return value
    return None


def get_shot_mode(project_name: str, book_name: str) -> str:
    """Per-book mode from a ``.shot_mode`` marker, else the project default."""
    return get_shot_mode_override(project_name, book_name) or get_default_shot_mode()


def set_shot_mode(project_name: str, book_name: str, mode: str) -> None:
    """Write the per-book mode marker (mirrors ``.transcript_approved``)."""
    if mode not in SHOT_MODES:
        raise ValueError(f"unknown shot mode: {mode!r}")
    marker = mode_marker_path(project_name, book_name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(mode, encoding="utf-8")


def clear_shot_mode(project_name: str, book_name: str) -> None:
    """Drop a per-book override so the book inherits the project default."""
    marker = mode_marker_path(project_name, book_name)
    if marker.exists():
        marker.unlink()


# --- run-time selection snapshot -------------------------------------------

def active_selection() -> Dict[str, str]:
    """Snapshot of the active variant for every slot.

    Taken once at the start of a book's run so a variant change made mid-run
    cannot split that book between two prompt versions. Nothing is persisted -
    the next run picks up whatever is active then.
    """
    return {slot: get_active_variant(slot) for slot in EDIT_SLOTS}


def render_with_selection(role: str, passage: str, selection: Dict[str, str]) -> str:
    """Compose a scene prompt from a captured selection rather than live settings."""
    shared = read_variant("shared", selection.get("shared", "shared"))
    fragment = read_variant(role, selection.get(role, role))
    return assemble(shared, fragment).replace("<text>", passage)
