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
        "style_negative_prompt": state.style_negative_prompt,
        "style_use_random_image_seed": state.style_use_random_image_seed,
        "style_image_seed": state.style_image_seed,
        "workflow_overrides": state.style_workflow_overrides
    }
    
    try:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        state.add_console_log(f"[FaST-Engine] Saved project configuration to disk: {settings_path.name}")
    except Exception as e:
        state.add_console_log(f"[FaST-Engine] Error saving project settings: {str(e)}")


def load_project_settings_from_disk(project_id: int) -> None:
    """Deserializes project_settings.json and restores active configurations to state bindings."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return
        project_name = project.name

    settings_path = get_project_settings_path(project_name)
    if not settings_path.exists():
        # Fallback to default state values if no custom settings exist yet
        state.playground_selected_template = "default"
        state.style_selected_preset = "default"
        state.style_selected_workflow = ""
        state.style_prompt_prefix = "ArsMJStyle, 1890s Victorian illustration, detailed pen and ink with soft watercolor wash, Sidney Paget style. "
        state.style_negative_prompt = "blurry, bad quality, text, watermark, photorealistic, photography"
        state.style_use_random_image_seed = True
        state.style_image_seed = 42
        state.style_workflow_overrides = {}
        state.style_discovered_params = {}
        return

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        state.playground_selected_template = data.get("active_template", "default")
        state.style_selected_preset = data.get("active_style_preset", "default")
        state.style_selected_workflow = data.get("active_workflow", "")
        state.style_prompt_prefix = data.get("style_prompt_prefix", "")
        state.style_negative_prompt = data.get("style_negative_prompt", "")
        state.style_use_random_image_seed = data.get("style_use_random_image_seed", True)
        state.style_image_seed = data.get("style_image_seed", 42)
        state.style_workflow_overrides = data.get("workflow_overrides", {})
        
        # Re-analyze active workflow parameters to repopulate active sliders
        if state.style_selected_workflow:
            from ui.pages.project.style_playground import handle_style_workflow_change
            handle_style_workflow_change(state.style_selected_workflow, clear_overrides=False)

        state.add_console_log(f"[FaST-Engine] Restored project configurations from: {settings_path.name}")
    except Exception as e:
        state.add_console_log(f"[FaST-Engine] Error loading project settings: {str(e)}")