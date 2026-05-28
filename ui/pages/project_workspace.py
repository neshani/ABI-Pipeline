from nicegui import ui
from sqlmodel import Session
from database.models import Project
from ui import state

# Unified Pipeline Steps
STAGES = ["Imported", "Transcription", "Prompt Gen", "Image Gen", "Proofreading", "Finished"]

def get_active_stage_idx(status: str) -> int:
    mapping = {
        "Imported": 0,
        "Transcribing": 1,
        "Transcribed": 1,
        "Prompts Created": 2,
        "Images Created": 3,
        "Proofreading": 4,
        "Finished": 5
    }
    return mapping.get(status, 0)


@ui.refreshable
def render_stepper(status: str):
    """Isolated stepper layout container that refreshes extremely fast with zero DOM flashing."""
    current_stage_idx = get_active_stage_idx(status)
    
    with ui.row().classes('w-full justify-between items-center bg-white border rounded-xl p-4 shadow-sm mb-2'):
        for idx, stage_name in enumerate(STAGES):
            is_completed = idx < current_stage_idx
            is_active = idx == current_stage_idx
            
            with ui.row().classes('items-center gap-1.5'):
                if is_completed:
                    ui.icon('check_circle', color='emerald-500', size='18px')
                    ui.label(stage_name).classes('text-xs font-bold text-emerald-600')
                elif is_active:
                    ui.icon('radio_button_checked', color='blue-600', size='18px').classes('animate-pulse')
                    ui.label(stage_name).classes('text-xs font-black text-blue-700')
                else:
                    ui.icon('radio_button_unchecked', color='slate-300', size='18px')
                    ui.label(stage_name).classes('text-xs font-semibold text-slate-400')
                    
            if idx < len(STAGES) - 1:
                ui.icon('chevron_right', color='slate-300', size='xs')


def render_project_tabs(project: Project, books: list, start_transcribe_cb, stop_transcribe_cb):
    # Render Dynamic Stepper inside its container
    render_stepper(state.project_status)

    # Header Row
    ui.label(f'Project Settings: {project.name}').classes('text-lg font-bold text-slate-800 mt-2')
    
    with ui.tabs().classes('w-full border-b') as project_tabs:
        tab_dash = ui.tab('Dashboard', icon='dashboard')
        tab_style = ui.tab('Style & Workflows', icon='brush')
        tab_play = ui.tab('Prompt-Gen Playground', icon='science')
        
    project_tabs.bind_value(state, 'active_project_tab')
        
    with ui.tab_panels(project_tabs, value=state.active_project_tab).classes('w-full bg-transparent p-0'):
        with ui.tab_panel(tab_dash):
            # Dynamic stats panels utilizing stable reactive text bindings
            with ui.row().classes('w-full gap-4'):
                with ui.card().classes('flex-1 border p-4 shadow-sm bg-white'):
                    ui.label('Project Status').classes('text-xs font-semibold text-slate-500')
                    # One-way bind the text directly to global state so it updates in-place via websocket
                    ui.label('').classes('text-xl font-bold text-slate-800').bind_text_from(state, 'project_status')
                with ui.card().classes('flex-1 border p-4 shadow-sm bg-white'):
                    ui.label('Discovered Books').classes('text-xs font-semibold text-slate-500')
                    ui.label(str(len(books))).classes('text-xl font-bold text-slate-800')
                    
            # Process Control Card
            with ui.card().classes('w-full border p-5 shadow-sm bg-white mt-4'):
                ui.label('Sequential Process Orchestration').classes('text-sm font-bold text-slate-800 mb-2')
                ui.label('Configure styling rules and workflows in the settings tab prior to launching processing.').classes('text-xs text-slate-400 mb-4')
                
                # Render control buttons inside their own container to prevent rebuilding the logs
                @ui.refreshable
                def action_buttons():
                    with ui.row().classes('items-center gap-3'):
                        if state.project_status == "Transcribing":
                            ui.spinner(size='md', color='blue')
                            ui.button(
                                'Stop Execution', 
                                icon='stop', 
                                color='red', 
                                on_click=lambda: stop_transcribe_cb(project.id)
                            ).classes('px-4 font-semibold')
                        else:
                            ui.button(
                                'Start Processing', 
                                icon='play_arrow', 
                                color='green', 
                                on_click=lambda: start_transcribe_cb(project.id)
                            ).classes('px-4 font-semibold')
                
                action_buttons()
                # Expose refresh trigger to the global state
                state.action_buttons_refresh = action_buttons.refresh
            
            # 2. Stable Live Console Log Output Widget (Created ONCE, never torn down)
            with ui.card().classes('w-full border p-5 shadow-sm bg-white mt-4 gap-3'):
                with ui.row().classes('w-full justify-between items-center'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('terminal', size='sm', color='slate-700')
                        ui.label('Process Control Console').classes('text-sm font-bold text-slate-800')
                    ui.button('Clear Logs', on_click=lambda: (state.console_logs.clear(), log_widget.clear())).props('flat dense').classes('text-xs text-slate-500')
                
                # Instantiating stable log element
                log_widget = ui.log(max_lines=300).classes('w-full h-64 bg-slate-900 text-slate-100 font-mono text-xs p-3 rounded-lg leading-relaxed')
                
                # Populate existing memory logs
                for line in state.console_logs:
                    log_widget.push(line)
                
                # Register reference so that background threads can stream to it directly
                state.active_log_widget = log_widget
                state.logs_pushed_index = len(state.console_logs)
                        
        with ui.tab_panel(tab_style):
            with ui.card().classes('w-full border p-5 shadow-sm bg-white'):
                ui.label('Visual Style & Workflow Configurations').classes('text-sm font-bold text-slate-800')
                ui.label('Placeholder scaffolding for style selections and ComfyUI definitions. (Phase 2)').classes('text-xs text-slate-500')
                
        with ui.tab_panel(tab_play):
            with ui.card().classes('w-full border p-5 shadow-sm bg-white'):
                ui.label('Prompt Generator Playground').classes('text-sm font-bold text-slate-800')
                ui.label('Placeholder scaffolding for local LLM text chunk testing. (Phase 4)').classes('text-xs text-slate-500')

# Parent layout register callback
main_layout_ref = None
def register_main_layout(layout):
    global main_layout_ref
    main_layout_ref = layout