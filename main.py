import sys
import os
import asyncio
import subprocess
from nicegui import ui
from sqlmodel import Session, select
from database.connection import init_db, get_setting, set_setting, engine
from database.models import Project, Book, Chapter
from services.sync_engine import recover_from_temp_workspaces
from services.transcription import (
    start_project_transcription, 
    cancel_project_transcription, 
    active_projects
)
from ui.components.settings_modal import SettingsModal

# Modularized states and pages imports
from ui import state
from ui.pages import render_portal_view, render_project_tabs, render_book_tabs, register_main_layout

# --- Initialize SQLite Database & Recovery Engines ---
init_db()

def reset_stuck_transcriptions():
    """Finds and resets any projects, books, or chapters that were stuck in a 'Transcribing' or 'Generating Prompts' state on startup."""
    with Session(engine) as session:
        stuck_projects = session.exec(select(Project).where(Project.status.in_(["Transcribing", "Generating Prompts"]))).all()
        for p in stuck_projects:
            # Revert to Transcribed if it was generating prompts, otherwise revert to Imported
            p.status = "Transcribed" if p.status == "Generating Prompts" else "Imported"
            session.add(p)
            
        stuck_books = session.exec(select(Book).where(Book.status.in_(["Transcribing", "Generating Prompts"]))).all()
        for b in stuck_books:
            b.status = "Transcribed" if b.status == "Generating Prompts" else "Imported"
            session.add(b)
            
        stuck_chapters = session.exec(select(Chapter).where(Chapter.status == "Transcribing")).all()
        for c in stuck_chapters:
            c.status = "Pending"
            session.add(c)
            
        session.commit()

# Ensure stuck tasks are cleared
reset_stuck_transcriptions()

# Run workspace recovery to restore wiped database entries
with Session(engine) as session:
    recover_from_temp_workspaces(session)

# --- Programmatic App Restart Engine ---
def restart_app():
    """Kills the active Python web server and restarts a clean instance."""
    ui.notify("Restarting ABI-Pipeline...", type="warning")
    asyncio.create_task(async_restart())

async def async_restart():
    await asyncio.sleep(0.5)
    script_path = os.path.abspath(sys.argv[0])
    
    if os.name == 'nt':  # Windows
        cmd = f'timeout 2 && "{sys.executable}" "{script_path}"'
        subprocess.Popen(cmd, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:  # macOS / Linux
        cmd = f'sleep 1.5 && "{sys.executable}" "{script_path}"'
        subprocess.Popen(cmd, shell=True)
        
    os._exit(0)

# --- Default App Configurations ---
DEFAULT_SETTINGS = {
    "comfy_url": "http://127.0.0.1:8188",
    "comfy_path": "F:/AI/ComfyUI/ComfyUI",
    "llm_url": "http://127.0.0.1:11434",
    "llm_api_key": "",
    "llm_model": "unsloth/gemma-4-e4b-it",
    "stt_engine": "Parakeet ONNX",
    "stt_device": "GPU/CUDA",
    "batch_size": 30,
    "output_dir": "./output"
}

app_settings = {}
for key, default_val in DEFAULT_SETTINGS.items():
    db_val = get_setting(key)
    if db_val is None:
        set_setting(key, default_val)
        app_settings[key] = default_val
    else:
        app_settings[key] = db_val

# --- Navigation State Control Handlers ---
def select_project(project_id: int):
    state.active_project_id = project_id
    state.active_book_id = None
    state.active_project_tab = 'Dashboard'
    state.active_log_widget = None  # Clear log references on panel change
    main_layout.refresh()


def select_book(book_id: int):
    state.active_book_id = book_id
    state.active_book_tab = 'Dashboard'
    state.active_log_widget = None  # Clear log references on panel change
    main_layout.refresh()


def exit_to_portal():
    state.active_project_id = None
    state.active_book_id = None
    state.active_log_widget = None
    main_layout.refresh()


def refresh_dashboard():
    main_layout.refresh()


# --- Transcription Action Handlers ---
def start_transcribing(project_id: int):
    try:
        import onnx_asr
    except ImportError:
        ui.notify(
            "Required dependency 'onnx-asr' is not installed. Check the dynamic installer in Settings.", 
            type="negative",
            close_button=True
        )
        return

    start_project_transcription(project_id)
    ui.notify("Background audiobook transcription started!", type="positive")
    
    # Fast trigger to update action buttons and stepper layout
    state.project_status = "Transcribing"
    if hasattr(state, 'action_buttons_refresh'):
        state.action_buttons_refresh()
    from ui.pages.project_workspace import render_stepper
    render_stepper.refresh("Transcribing")


def stop_transcribing(project_id: int):
    if state.project_status == "Transcribing":
        cancel_project_transcription(project_id)
        ui.notify("Stopping transcription process...", type="warning")
        state.project_status = "Imported"
    elif state.project_status == "Generating Prompts":
        from services.prompt_engine import cancel_prompt_generation
        cancel_prompt_generation(project_id)
        ui.notify("Stopping prompt generation process...", type="warning")
        state.project_status = "Transcribed"
    
    # Fast trigger to update action buttons and stepper layout
    if hasattr(state, 'action_buttons_refresh'):
        state.action_buttons_refresh()
    from ui.pages.project_workspace import render_stepper
    render_stepper.refresh(state.project_status)


def start_prompt_generation(project_id: int):
    """
    Launches the asynchronous, interruptible, and resumable prompt generation process.
    """
    from services.prompt_engine import start_project_prompt_gen
    asyncio.create_task(start_project_prompt_gen(project_id))
    ui.notify("Background prompt generation sequences initiated!", type="positive")
    
    state.project_status = "Generating Prompts"
    if hasattr(state, 'action_buttons_refresh'):
        state.action_buttons_refresh()
    from ui.pages.project_workspace import render_stepper
    render_stepper.refresh("Generating Prompts")


def start_image_generation(project_id: int):
    """
    Temporary scaffolding callback to transition the project and books 
    from 'Prompts Created' to 'Images Created'.
    """
    state.add_console_log(f"[Image-Gen] Connecting to ComfyUI at {get_setting('comfy_url')}...")
    state.add_console_log("[Image-Gen] Queueing prompt generation sequences...")
    
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project:
            project.status = "Images Created"
            session.add(project)
            
            books = session.exec(select(Book).where(Book.project_id == project_id)).all()
            for b in books:
                b.status = "Images Created"
                session.add(b)
            session.commit()
            
    state.add_console_log("[Image-Gen] Process Complete: PNGs successfully rendered! (Scaffold Model)")
    ui.notify("Mock Image Generation completed!", type="success")
    
    # Refresh local state variables
    state.project_status = "Images Created"
    if hasattr(state, 'action_buttons_refresh'):
        state.action_buttons_refresh()
    from ui.pages.project_workspace import render_stepper
    render_stepper.refresh("Images Created")


# --- Dynamic WebSocket State-Updater (No full page refreshes!) ---
def check_for_active_transcriptions():
    if state.active_project_id is not None:
        with Session(engine) as session:
            project = session.get(Project, state.active_project_id)
            if project:
                # 1. Update bound labels/values dynamically
                state.project_status = project.status
                
                # Check if we need to update the action button spinner
                if hasattr(state, 'action_buttons_refresh'):
                    try:
                        state.action_buttons_refresh()
                    except Exception:
                        pass
                
                # Re-render ONLY the isolated Horizontal Stepper
                from ui.pages.project_workspace import render_stepper
                try:
                    render_stepper.refresh(project.status)
                except Exception:
                    pass

            # 2. Update bound Book statuses and progress values in-place
            books = session.exec(select(Book).where(Book.project_id == state.active_project_id)).all()
            for b in books:
                state.books_progress[b.id] = b.progress
                state.books_status[b.id] = b.status
                state.books_subtitle[b.id] = f"{b.status} • {int(b.progress * 100)}%"

        # 3. Stream newly added log lines to stable log widget (Leaves scrollbar untouched!)
        if state.active_log_widget:
            try:
                new_logs = state.console_logs[state.logs_pushed_index:]
                for line in new_logs:
                    state.active_log_widget.push(line)
                state.logs_pushed_index = len(state.console_logs)
            except Exception:
                pass

        # 4. Refresh the live Prompt Generation Feed dynamically
        if hasattr(state, 'recent_prompts_refresh'):
            try:
                state.recent_prompts_refresh()
            except Exception:
                pass


# --- Split-Panel Shell Renderer ---
def render_split_panel_shell(project_id: int):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            ui.notify("Project not found.", type="negative")
            exit_to_portal()
            return
        books = session.exec(select(Book).where(Book.project_id == project.id)).all()

    # Seed static binding dictionaries on initial layout rendering
    state.project_status = project.status

    with ui.grid(columns='250px 1fr').classes('w-full gap-6 items-start'):
        # LEFT NAVIGATION SIDEBAR
        with ui.column().classes('bg-white border rounded-xl p-4 gap-4 shadow-sm h-[calc(100vh-140px)] sticky top-24'):
            ui.button(
                'Back to Projects', 
                icon='arrow_back', 
                on_click=exit_to_portal
            ).props('flat dense').classes('text-slate-600 text-xs self-start -ml-2 mb-2')
            
            # Active Project Switcher Row
            project_bg = 'bg-blue-50 border border-blue-100' if state.active_book_id is None else 'hover:bg-slate-50'
            with ui.row().classes(f'w-full p-2 rounded-lg cursor-pointer items-center justify-between transition-colors {project_bg}') \
                    .on('click', lambda: select_project(project.id)):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('folder' if project.is_batch else 'menu_book', size='sm', color='slate-700')
                    ui.label(project.name).classes('text-sm font-bold text-slate-800 truncate max-w-[150px]')
                ui.icon('settings', size='xs', color='slate-400')
                
            ui.separator()
            ui.label('Books & Volumes').classes('text-[10px] font-bold text-slate-400 tracking-wider uppercase px-1')
            
            with ui.column().classes('w-full gap-1 overflow-y-auto flex-1'):
                for book in books:
                    # Seed bindings
                    if book.id not in state.books_progress:
                        state.books_progress[book.id] = book.progress
                    if book.id not in state.books_status:
                        state.books_status[book.id] = book.status
                    if book.id not in state.books_subtitle:
                        state.books_subtitle[book.id] = f"{book.status} • {int(book.progress * 100)}%"

                    book_bg = 'bg-blue-50/70 border border-blue-100/50 text-blue-700' if state.active_book_id == book.id else 'hover:bg-slate-50 text-slate-700'
                    with ui.row().classes(f'w-full p-2 rounded-lg cursor-pointer items-center justify-between transition-colors {book_bg}') \
                            .on('click', lambda b_id=book.id: select_book(b_id)):
                        with ui.row().classes('items-center gap-2 truncate'):
                            if book.cover_path:
                                ui.image(book.cover_path).classes('w-6 h-6 rounded object-cover shadow-xs border')
                            else:
                                ui.icon('library_books', size='xs', color='slate-400')
                            
                            with ui.column().classes('gap-0 truncate'):
                                ui.label(book.name).classes('text-xs font-semibold truncate max-w-[120px]')
                                # BIND TEXT REACTIVELY (Updates only this label, zero page redraws!)
                                ui.label('').classes('text-[9px] font-medium text-slate-500 truncate max-w-[120px]') \
                                    .bind_text_from(state.books_subtitle, book.id)
                        
                        # Compact sidebar visual status indicator
                        status_dot = ui.element('div').classes('w-2 h-2 rounded-full')
                        # Reactive style bindings: map class changes programmatically
                        status_dot.bind_visibility_from(state.books_status, book.id, backward=lambda val: val == "Transcribing")
                        status_dot.classes('bg-blue-500 animate-pulse')

        # RIGHT WORKSPACE ROUTER
        with ui.column().classes('w-full gap-4'):
            if state.active_book_id is None:
                render_project_tabs(
                    project, 
                    books, 
                    start_transcribing, 
                    stop_transcribing,
                    start_prompt_generation,
                    start_image_generation
                )
            else:
                render_book_tabs(state.active_book_id)


# --- Dynamic Main Page Wrapper ---
@ui.refreshable
def main_layout():
    if state.active_project_id is None:
        render_portal_view(select_project, refresh_dashboard)
    else:
        render_split_panel_shell(state.active_project_id)


# Register layout reference inside rendering module
register_main_layout(main_layout)

# Initialize settings modal
settings_modal = SettingsModal(app_settings, restart_app)

# --- Header & Top Bar Navigation ---
with ui.header(elevated=False).classes('bg-slate-800 text-white px-6 py-4 justify-between items-center'):
    with ui.row().classes('items-center gap-3'):
        ui.icon('auto_awesome', size='md').classes('text-blue-400')
        ui.label('ABI-Pipeline').classes('text-xl font-bold tracking-tight')

    with ui.row().classes('items-center gap-4 bg-slate-700/50 p-1 rounded-lg border border-slate-600/30'):
        ui.input(
            placeholder='Search projects...',
            on_change=lambda e: (setattr(state, 'search_query', e.value), refresh_dashboard())
        ).props('dark borderless dense').classes('w-48 px-2')
        
        ui.select(
            options=['All', 'Single', 'Batch'],
            value='All',
            on_change=lambda e: (setattr(state, 'selected_project_type', e.value), refresh_dashboard())
        ).props('dark borderless dense').classes('w-24 text-sm')

    with ui.row().classes('items-center gap-3'):
        with ui.button(icon='construction', color='slate-600') as tools_btn:
            tools_btn.classes('text-white text-sm capitalize rounded-lg')
            with ui.menu() as menu:
                ui.menu_item('Style Library', on_click=lambda: ui.notify('Style Library coming soon'))
                ui.menu_item('Prompt Templates', on_click=lambda: ui.notify('Templates coming soon'))
        ui.button(icon='settings', on_click=lambda: settings_modal.open()).props('flat round color=white')


# --- Main Application Page Frame ---
with ui.column().classes('w-full max-w-7xl mx-auto p-6 gap-6'):
    main_layout()


# --- Polling Update Refresh Timer ---
ui.timer(2.0, check_for_active_transcriptions)

# Launch our app
ui.run(title="ABI-Pipeline")