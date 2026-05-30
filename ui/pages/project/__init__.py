from ui.pages.project.dashboard import render_project_tabs as render_dashboard_tabs
from ui.pages.project.dashboard import STAGES, get_active_stage_idx

def render_project_tabs(
    project, 
    books, 
    start_transcribe_cb, 
    stop_transcribe_cb,
    start_prompt_gen_cb=None,
    start_image_gen_cb=None,
    save_project_settings_cb=None
):
    render_dashboard_tabs(
        project, 
        books, 
        start_transcribe_cb, 
        stop_transcribe_cb,
        start_prompt_gen_cb,
        start_image_gen_cb,
        save_project_settings_cb
    )