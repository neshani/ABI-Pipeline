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
    """Serializes the active state configuration into project_settings.json on disk."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return
        project_name = project.name

    settings_path = get_project_settings_path(project_name)
    data = {
        "active_template": state.playground_selected_template,
        "active_style_preset": state.style_selected_preset,
        "active_workflow": state.style_selected_workflow,
        "style_prompt_prefix": state.style_prompt_prefix,
        "style_prompt_suffix": state.style_prompt_suffix,
        "style_negative_prompt": state.style_negative_prompt,
        "style_use_random_image_seed": state.style_use_random_image_seed,
        "style_image_seed": state.style_image_seed,
        "workflow_overrides": state.style_workflow_overrides,
        "playground_chunk_size": state.playground_chunk_size
    }
    
    try:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        state.add_console_log(f"[FaST-Engine] Saved project configuration to disk: {settings_path.name}")
    except Exception as e:
        state.add_console_log(f"[FaST-Engine] Error saving project settings: {str(e)}")


def load_project_settings_from_disk(project_id: int) -> None:
    """Deserializes project_settings.json and restores active configurations to state bindings with legacy redirection."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return
        project_name = project.name

    settings_path = get_project_settings_path(project_name)
    if not settings_path.exists():
        # Fallback to new default state values if no custom settings exist yet
        state.playground_selected_template = "default"
        state.style_selected_preset = "retro_graphic_novel"
        state.style_selected_workflow = ""
        state.style_prompt_prefix = "space opera adventure graphic novel illustration, sharp ink sketch, crisp outlines, retro-futuristic sci-fi aesthetic, cosmic wonder, detailed, "
        state.style_prompt_suffix = ", high-contrast shadows, selective color accents, bold ink-wash shading"
        state.style_negative_prompt = "blurry, bad quality, text, watermark, photorealistic, photography, dystopian, gritty, grimy, decay, cyberpunk"
        state.style_use_random_image_seed = True
        state.style_image_seed = 42
        state.style_workflow_overrides = {}
        state.style_discovered_params = {}
        state.playground_chunk_size = 350
        return

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        state.playground_selected_template = data.get("active_template", "default")
        
        # 1. Load the base Style Preset first. Redirect legacy "default" presets to the new "retro_graphic_novel" default.
        preset_name = data.get("active_style_preset", "retro_graphic_novel")
        if preset_name == "default":
            preset_name = "retro_graphic_novel"
            
        state.style_selected_preset = preset_name
        
        from ui.pages.project.style_playground import load_style_preset_by_name
        load_style_preset_by_name(preset_name)

        # 2. Apply/Overlay any customized project-level settings on top of the preset (only if they exist and are non-empty)
        if data.get("active_workflow"):
            state.style_selected_workflow = data.get("active_workflow")
        if data.get("style_prompt_prefix"):
            state.style_prompt_prefix = data.get("style_prompt_prefix")
        if data.get("style_prompt_suffix"):
            state.style_prompt_suffix = data.get("style_prompt_suffix")
        if data.get("style_negative_prompt"):
            state.style_negative_prompt = data.get("style_negative_prompt")
            
        state.style_use_random_image_seed = data.get("style_use_random_image_seed", True)
        state.style_image_seed = data.get("style_image_seed", 42)
        
        # Merge workflow overrides rather than fully overwriting them, preserving preset overrides if project has none
        project_overrides = data.get("workflow_overrides", {})
        if project_overrides:
            state.style_workflow_overrides.update(project_overrides)
            
        state.playground_chunk_size = data.get("playground_chunk_size", 350)
        
        # Re-analyze active workflow parameters to repopulate active sliders
        if state.style_selected_workflow:
            from ui.pages.project.style_playground import handle_style_workflow_change
            from ui.pages.project.style_playground import render_workflow_overrides_ui
            handle_style_workflow_change(state.style_selected_workflow, clear_overrides=False)
            try:
                render_workflow_overrides_ui.refresh()
            except Exception:
                pass

        state.add_console_log(f"[FaST-Engine] Restored project configurations from: {settings_path.name}")
    except Exception as e:
        state.add_console_log(f"[FaST-Engine] Error loading project settings: {str(e)}")