import sys
import os
import asyncio
from nicegui import ui
from sqlmodel import Session, select
from database.connection import init_db, get_setting, set_setting, engine
from database.models import Project, Book, Chapter
from services.scanner import scan_directory, ingest_project
from services.transcription import (
    start_project_transcription, 
    cancel_project_transcription, 
    active_projects
)
import subprocess

# --- Import modular UI Components ---
from ui.components.settings_modal import SettingsModal


# --- Initialize SQLite Database ---
init_db()

def reset_stuck_transcriptions():
    """Finds and resets any projects, books, or chapters that were stuck in a 'Transcribing' state on startup."""
    with Session(engine) as session:
        # Reset stuck projects
        stuck_projects = session.exec(select(Project).where(Project.status == "Transcribing")).all()
        for p in stuck_projects:
            p.status = "Imported"
            session.add(p)
            
        # Reset stuck books
        stuck_books = session.exec(select(Book).where(Book.status == "Transcribing")).all()
        for b in stuck_books:
            b.status = "Imported"
            session.add(b)
            
        # Reset stuck chapters
        stuck_chapters = session.exec(select(Chapter).where(Chapter.status == "Transcribing")).all()
        for c in stuck_chapters:
            c.status = "Pending"
            session.add(c)
            
        session.commit()

reset_stuck_transcriptions()

# --- Programmatic App Restart Engine ---
def restart_app():
    """Kills the active Python web server and restarts a clean instance."""
    ui.notify("Restarting ABI-Pipeline...", type="warning")
    # Wait briefly for the UI notification to hit the browser
    asyncio.create_task(async_restart())

async def async_restart():
    await asyncio.sleep(0.5) # Give the WebSocket a moment to transmit the notification
    
    script_path = os.path.abspath(sys.argv[0])
    
    if os.name == 'nt':  # Windows
        # Start a detached command prompt that waits 2 seconds (to let us die) and then restarts main.py
        cmd = f'timeout 2 && "{sys.executable}" "{script_path}"'
        subprocess.Popen(
            cmd, 
            shell=True, 
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:  # macOS / Linux
        # Wait 1.5 seconds in background and restart
        cmd = f'sleep 1.5 && "{sys.executable}" "{script_path}"'
        subprocess.Popen(cmd, shell=True)
        
    # Hard-kill our own process instantly. This forces Windows to release Port 8080 immediately!
    os._exit(0)

# --- Default App Configurations ---
DEFAULT_SETTINGS = {
    "comfy_url": "http://127.0.0.1:8188",
    "comfy_path": "F:/AI/ComfyUI/ComfyUI",
    "llm_provider": "Ollama",  # Ollama or LM Studio
    "llm_url": "http://127.0.0.1:11434",
    "stt_engine": "Parakeet ONNX",  # Parakeet ONNX or Whisper
    "stt_device": "GPU/CUDA",  # GPU/CUDA or CPU
    "batch_size": 34,  # Hardware batch size (usually 8 is optimized for RTX 3090/4090 vram)
    "output_dir": "./output"
}

# Load settings from SQLite, fallback to defaults
app_settings = {}
for key, default_val in DEFAULT_SETTINGS.items():
    db_val = get_setting(key)
    if db_val is None:
        set_setting(key, default_val)
        app_settings[key] = default_val
    else:
        app_settings[key] = db_val


search_query = ""
selected_project_type = "All"

# --- Ingestion Dialog State Variables ---
current_scan_result = None
scan_error = ""
custom_project_name_value = ""


# --- Helper: Status Badges ---
def get_status_badge(status: str):
    styles = {
        "Imported": "bg-slate-200 text-slate-700",
        "Transcribing": "bg-blue-100 text-blue-800 border-blue-200",
        "Transcribed": "bg-emerald-100 text-emerald-800 border-emerald-200",
        "Prompts Created": "bg-purple-100 text-purple-800 border-purple-200",
        "Images Created": "bg-amber-100 text-amber-800 border-amber-200",
        "Finished": "bg-emerald-100 text-emerald-800 border-emerald-200"
    }
    style = styles.get(status, "bg-slate-100 text-slate-800")
    return ui.badge(status).classes(f'px-3 py-1 text-xs rounded-full font-medium {style}')

# --- Settings Save Event Handlers ---
def save_settings_to_db():
    """Saves the current modified in-memory UI settings back to SQLite."""
    for key, val in app_settings.items():
        set_setting(key, val)
    ui.notify("System settings successfully updated!", type="positive")
    settings_dialog.close()

# --- Refresh Dashboard UI when filters change ---
def refresh_dashboard():
    dashboard_container.refresh()


# --- Transcription Action Handlers ---
def start_transcribing(project_id: int):
    """Triggers the background sequential transcription process."""
    try:
        import onnx_asr
    except ImportError:
        ui.notify(
            "Required dependency 'onnx-asr' is not installed in the active python environment. Check the dynamic installer in Settings.", 
            type="negative",
            close_button=True
        )
        return

    start_project_transcription(project_id)
    ui.notify("Background audiobook transcription started!", type="positive")
    refresh_dashboard()

def stop_transcribing(project_id: int):
    """Signals active process worker threads to wind down and cancel safely."""
    cancel_project_transcription(project_id)
    ui.notify("Stopping transcription process...", type="warning")
    refresh_dashboard()


# --- Background Polling Update Handler ---
def check_for_active_transcriptions():
    """Polls SQLite to update progress bars, capturing the final transition when threads shut down."""
    # Since SQLite is local and lightning-fast, we poll and refresh the UI to capture
    # all status updates (including the final transition from 'Transcribing' back to 'Imported' on stop).
    refresh_dashboard()


# --- Main Dashboard Component ---
@ui.refreshable
def dashboard_container():
    global search_query, selected_project_type
    
    # 1. Fetch real projects & books from SQLite database
    projects_data = []
    with Session(engine) as session:
        projects = session.exec(select(Project)).all()
        for p in projects:
            books = session.exec(select(Book).where(Book.project_id == p.id)).all()
            
            # Calculate project progress as average of book progresses
            avg_progress = (
                sum(b.progress for b in books) / len(books) if books else 0.0
            )
            
            projects_data.append({
                "id": p.id,
                "name": p.name,
                "status": p.status,
                "is_batch": p.is_batch,
                "progress": avg_progress,
                "books_count": len(books),
                "books": books
            })

    # 2. Apply search and type filters
    filtered = []
    for p in projects_data:
        name_match = search_query.lower() in p["name"].lower()
        
        type_match = True
        if selected_project_type == "Single" and p["is_batch"]:
            type_match = False
        elif selected_project_type == "Batch" and not p["is_batch"]:
            type_match = False
            
        if name_match and type_match:
            filtered.append(p)

    if not filtered:
        with ui.column().classes('w-full items-center justify-center p-12 text-slate-400'):
            ui.icon('search', size='lg')
            ui.label('No projects found. Use "+ New Project" to import audiobooks.').classes('text-lg text-center')
        return

    # 3. Render dashboard list
    for project in filtered:
        if project["is_batch"]:
            with ui.expansion().classes('w-full border rounded-xl shadow-sm bg-white overflow-hidden mb-4') as exp:
                with exp.add_slot('header'):
                    with ui.row().classes('w-full items-center justify-between py-2'):
                        with ui.row().classes('items-center gap-3'):
                            ui.icon('folder', size='md', color='amber-500')
                            with ui.column().classes('gap-0'):
                                ui.label(project["name"]).classes('text-base font-semibold text-slate-800')
                                ui.label(f'Batch • {project["books_count"]} books').classes('text-xs text-slate-400')
                        
                        with ui.row().classes('items-center gap-4'):
                            # Multi-book Play / Stop Orchestration Button
                            if project["status"] == "Transcribing":
                                ui.spinner(size='sm', color='blue')
                                ui.button(
                                    icon='stop', 
                                    color='red', 
                                    on_click=lambda p_id=project["id"]: stop_transcribing(p_id)
                                ).props('flat round dense')
                            elif project["status"] != "Transcribed":
                                ui.button(
                                    icon='play_arrow', 
                                    color='green', 
                                    on_click=lambda p_id=project["id"]: start_transcribing(p_id)
                                ).props('flat round dense').classes('hover:scale-105')

                            get_status_badge(project["status"])
                            ui.linear_progress(value=project["progress"], show_value=False).classes('w-24 h-2 rounded-full')
                
                with ui.column().classes('w-full p-4 bg-slate-50 border-t gap-3'):
                    for book in project["books"]:
                        with ui.row().classes('w-full justify-between items-center bg-white p-3 rounded-lg border shadow-xs'):
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('menu_book', size='sm', color='slate-400')
                                ui.label(book.name).classes('text-sm font-medium text-slate-700')
                            with ui.row().classes('items-center gap-4'):
                                get_status_badge(book.status)
                                ui.linear_progress(value=book.progress, show_value=False).classes('w-16 h-1 rounded-full')
        else:
            with ui.card().classes('w-full border rounded-xl shadow-sm hover:shadow-md transition-shadow p-5 mb-4 bg-white'):
                with ui.row().classes('w-full items-center justify-between'):
                    with ui.row().classes('items-center gap-3'):
                        ui.icon('menu_book', size='md', color='blue-500')
                        with ui.column().classes('gap-0'):
                            ui.label(project["name"]).classes('text-base font-semibold text-slate-800')
                            ui.label('Single Novel').classes('text-xs text-slate-400')
                    
                    with ui.row().classes('items-center gap-4'):
                        # Play / Stop Orchestration Button
                        if project["status"] == "Transcribing":
                            ui.spinner(size='sm', color='blue')
                            ui.button(
                                icon='stop', 
                                color='red', 
                                on_click=lambda p_id=project["id"]: stop_transcribing(p_id)
                            ).props('flat round dense')
                        elif project["status"] != "Transcribed":
                            ui.button(
                                icon='play_arrow', 
                                color='green', 
                                on_click=lambda p_id=project["id"]: start_transcribing(p_id)
                            ).props('flat round dense').classes('hover:scale-105')

                        get_status_badge(project["status"])
                        ui.linear_progress(value=project["progress"], show_value=False).classes('w-24 h-2 rounded-full')

# --- Async Scanning Event Handler ---
async def run_live_scan(e):
    global current_scan_result, scan_error, custom_project_name_value
    path_str = e.value.strip()
    
    if not path_str:
        current_scan_result = None
        scan_error = ""
        custom_project_name_value = ""
        scan_preview_container.refresh()
        return
        
    try:
        # Offload file probing to background thread to prevent freezing the NiceGUI UI thread
        result = await asyncio.to_thread(scan_directory, path_str)
        if result["type"] == "none":
            current_scan_result = None
            scan_error = "No supported audiobook files or subdirectories found."
            custom_project_name_value = ""
        else:
            current_scan_result = result
            scan_error = ""
            custom_project_name_value = result["project_name"]
    except Exception as ex:
        current_scan_result = None
        scan_error = f"Error scanning folder: {str(ex)}"
        custom_project_name_value = ""
        
    scan_preview_container.refresh()


# --- Database Ingestion Save Handler ---
def save_scanned_project():
    global current_scan_result, custom_project_name_value
    if not current_scan_result:
        ui.notify("No valid scanned project to save.", type="negative")
        return
        
    try:
        project_id = ingest_project(current_scan_result, custom_project_name_value)
        ui.notify(f"Successfully imported project ID: {project_id}!", type="positive")
        new_project_dialog.close()
        refresh_dashboard()
    except Exception as ex:
        ui.notify(f"Failed to save project: {str(ex)}", type="negative")


# --- Refreshable Dialog Preview Container ---
@ui.refreshable
def scan_preview_container():
    global current_scan_result, scan_error, custom_project_name_value
    
    if scan_error:
        with ui.row().classes('items-center gap-2 p-3 bg-red-50 text-red-700 rounded-lg border border-red-200 w-full'):
            ui.icon('error_outline', size='sm')
            ui.label(scan_error).classes('text-sm font-medium')
        return

    if not current_scan_result:
        with ui.column().classes('w-full items-center justify-center p-6 text-slate-400 border border-dashed rounded-lg bg-slate-50'):
            ui.icon('folder_open', size='lg')
            ui.label('Waiting for a valid local audiobook directory path...').classes('text-xs text-center')
        return

    # Valid results found
    with ui.column().classes('w-full gap-4'):
        # Input to customize the final project folder database entry
        ui.input(
            'Project Title', 
            value=custom_project_name_value,
            on_change=lambda e: globals().update(custom_project_name_value=e.value)
        ).classes('w-full')
        
        # Display type mapping
        with ui.row().classes('items-center gap-2'):
            if current_scan_result["type"] == "single":
                ui.icon('menu_book', color='blue-500', size='sm')
                ui.label('Structure: Single Novel').classes('text-sm font-semibold text-slate-700')
            else:
                ui.icon('folder', color='amber-500', size='sm')
                ui.label(f'Structure: Batch ({len(current_scan_result["books"])} audiobooks found)').classes('text-sm font-semibold text-slate-700')
        
        # Discovered books list
        with ui.column().classes('w-full gap-2 max-h-48 overflow-y-auto p-2 bg-slate-50 border rounded-lg'):
            for book in current_scan_result["books"]:
                with ui.row().classes('w-full justify-between items-center bg-white p-2 rounded border shadow-xs'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('library_books', size='xs', color='slate-400')
                        ui.label(book["name"]).classes('text-xs font-medium text-slate-700 truncate max-w-sm')
                    with ui.row().classes('items-center gap-2'):
                        if book["cover_path"]:
                            ui.badge('Cover Found', color='emerald-100').classes('text-emerald-800 text-[10px] px-1.5 py-0.5 rounded font-bold')
                        ui.badge(f'{len(book["files"])} tracks', color='slate-100').classes('text-slate-600 text-[10px] px-1.5 py-0.5 rounded')

        # Action bar
        with ui.row().classes('w-full justify-end gap-3 mt-2'):
            ui.button('Cancel', on_click=new_project_dialog.close).props('flat color=slate')
            ui.button(
                'Import & Create Project', 
                on_click=save_scanned_project
            ).classes('bg-blue-600 hover:bg-blue-700 text-white font-semibold')


# --- Clean Opener to Reset Ingestion State ---
def open_new_project_dialog():
    globals().update(current_scan_result=None, scan_error="", custom_project_name_value="")
    path_input.set_value("")
    scan_preview_container.refresh()
    new_project_dialog.open()


# Initialize settings modal in state
settings_modal = SettingsModal(app_settings, restart_app)

# --- Header & Top Bar Navigation ---
with ui.header(elevated=False).classes('bg-slate-800 text-white px-6 py-4 justify-between items-center'):
    with ui.row().classes('items-center gap-3'):
        ui.icon('auto_awesome', size='md').classes('text-blue-400')
        ui.label('ABI-Pipeline').classes('text-xl font-bold tracking-tight')

    with ui.row().classes('items-center gap-4 bg-slate-700/50 p-1 rounded-lg border border-slate-600/30'):
        search_input = ui.input(
            placeholder='Search projects...',
            on_change=lambda e: (globals().update(search_query=e.value), refresh_dashboard())
        ).props('dark borderless dense').classes('w-48 px-2')
        
        type_select = ui.select(
            options=['All', 'Single', 'Batch'],
            value='All',
            on_change=lambda e: (globals().update(selected_project_type=e.value), refresh_dashboard())
        ).props('dark borderless dense').classes('w-24 text-sm')

    with ui.row().classes('items-center gap-3'):
        with ui.button(icon='construction', color='slate-600') as tools_btn:
            tools_btn.classes('text-white text-sm capitalize rounded-lg')
            with ui.menu() as menu:
                ui.menu_item('Style Library', on_click=lambda: ui.notify('Style Library coming soon'))
                ui.menu_item('Prompt Templates', on_click=lambda: ui.notify('Templates coming soon'))
        ui.button(icon='settings', on_click=lambda: settings_modal.open()).props('flat round color=white')


# --- Main Application Page Frame ---
with ui.column().classes('w-full max-w-4xl mx-auto p-6 gap-6'):
    with ui.row().classes('w-full justify-between items-center mb-2'):
        with ui.column().classes('gap-0'):
            ui.label('Project Dashboard').classes('text-2xl font-bold text-slate-800')
            ui.label('Manage audiobooks, generate prompts, and render pipeline.').classes('text-sm text-slate-500')
        ui.button(
            '+ New Project', 
            on_click=open_new_project_dialog
        ).classes('bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg shadow-sm text-sm font-semibold capitalize')

    dashboard_container()


# --- Modal/Dialog: Add New Project ---
with ui.dialog() as new_project_dialog, ui.card().classes('w-full max-w-2xl p-6 rounded-xl'):
    ui.label('Create New Project').classes('text-xl font-bold text-slate-800 mb-2')
    ui.label('Enter a local audiobook directory path. We will analyze the structure and discover the covers automatically.').classes('text-sm text-slate-500 mb-4')
    
    with ui.column().classes('w-full gap-4'):
        path_input = ui.input(
            'Local Directory Path', 
            placeholder='e.g., F:/Audiobooks/Jack_Aubrey_Series',
            on_change=run_live_scan
        ).classes('w-full')
        
        scan_preview_container()

# --- Active Transcription UI Polling Refresh Timer ---
ui.timer(2.0, check_for_active_transcriptions)

# Launch our app
ui.run(title="ABI-Pipeline")