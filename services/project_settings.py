import json
from pathlib import Path
from sqlmodel import Session
from database.connection import engine, get_setting
from database.models import Project
from ui import state

def get_project_settings_path(project_name: str) -> Path:
    """Returns the path to the project's persistent settings file on disk."""
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    project_dir = base_output_dir / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir / "project_settings.json"


def save_project_settings_to_disk(project_id: int) -> None:
    """Serializes the project's non-style configuration into project_settings.json.

    Style values (prefix/suffix/negative, workflow, node overrides, and the
    per-role deltas) deliberately live only in ``styles/<name>.json``; this file
    records which style set the project points at, plus generation settings that
    are genuinely per project.
    """
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return
        project_name = project.name

    settings_path = get_project_settings_path(project_name)
    data = {
        "active_template": state.playground_selected_template,
        "active_style_preset": state.style_selected_preset,
        "style_use_random_image_seed": state.style_use_random_image_seed,
        "style_image_seed": state.style_image_seed,
        "playground_chunk_size": state.playground_chunk_size,
        "cast_enabled": bool(state.cast_enabled),
    }
    
    try:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        state.add_console_log(f"[FaST-Engine] Saved project configuration to disk: {settings_path.name}")
    except Exception as e:
        state.add_console_log(f"[FaST-Engine] Error saving project settings: {str(e)}")


def load_project_settings_from_disk(project_id: int) -> None:
    """Restores the project's active style set and generation settings.

    Legacy files may still carry the flat style keys written by earlier versions;
    they are ignored on purpose - the style set on disk is authoritative now.
    """
    from ui.pages.project.style_playground import load_style_preset_by_name

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return
        project_name = project.name

    settings_path = get_project_settings_path(project_name)
    if not settings_path.exists():
        # Fallback to defaults if no custom settings exist yet
        state.playground_selected_template = "default"
        state.style_selected_preset = "retro_graphic_novel"
        state.style_use_random_image_seed = True
        state.style_image_seed = 42
        state.playground_chunk_size = 350
        state.cast_enabled = True
        load_style_preset_by_name(state.style_selected_preset)
        return

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        state.playground_selected_template = data.get("active_template", "default")

        # Redirect the retired "default" preset name to the current default.
        preset_name = data.get("active_style_preset", "retro_graphic_novel")
        if preset_name == "default":
            preset_name = "retro_graphic_novel"
        state.style_selected_preset = preset_name

        state.style_use_random_image_seed = data.get("style_use_random_image_seed", True)
        state.style_image_seed = data.get("style_image_seed", 42)
        state.playground_chunk_size = data.get("playground_chunk_size", 350)
        state.cast_enabled = bool(data.get("cast_enabled", True))

        # Loads the Character base + role deltas, the workflow, and the node overrides.
        load_style_preset_by_name(preset_name)

        state.add_console_log(f"[FaST-Engine] Restored project configurations from: {settings_path.name}")
    except Exception as e:
        state.add_console_log(f"[FaST-Engine] Error loading project settings: {str(e)}")