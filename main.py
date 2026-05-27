import sys
import os
import asyncio
from nicegui import ui
from database.connection import init_db, get_setting, set_setting
import subprocess
# --- Import modular UI Components ---
from ui.components.settings_modal import SettingsModal


# --- Initialize SQLite Database ---
init_db()

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

# --- Mock Data for Dashboard (Unchanged) ---
mock_projects = [
    {
        "id": 1,
        "name": "Master and Commander",
        "type": "single",
        "status": "Transcribed",
        "progress": 0.25,
        "books_count": 1
    },
    {
        "id": 2,
        "name": "Fantasy Series Batch 1",
        "type": "batch",
        "status": "Prompts Created",
        "progress": 0.50,
        "books_count": 3,
        "books": [
            {"name": "The Way of Kings", "status": "Prompts Created", "progress": 0.50},
            {"name": "Words of Radiance", "status": "Transcribed", "progress": 0.25},
            {"name": "Oathbringer", "status": "Imported", "progress": 0.10}
        ]
    }
]

search_query = ""
selected_project_type = "All"

# --- Helper: Status Badges ---
def get_status_badge(status: str):
    styles = {
        "Imported": "bg-slate-200 text-slate-700",
        "Transcribed": "bg-blue-100 text-blue-800 border-blue-200",
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

# --- Main Dashboard Component ---
@ui.refreshable
def dashboard_container():
    global search_query, selected_project_type
    filtered = []
    for p in mock_projects:
        name_match = search_query.lower() in p["name"].lower()
        type_match = (selected_project_type == "All" or 
                      (selected_project_type == "Single" and p["type"] == "single") or 
                      (selected_project_type == "Batch" and p["type"] == "batch"))
        if name_match and type_match:
            filtered.append(p)

    if not filtered:
        with ui.column().classes('w-full items-center justify-center p-12 text-slate-400'):
            ui.icon('search', size='lg')
            ui.label('No projects found matching the filters.').classes('text-lg')
        return

    for project in filtered:
        if project["type"] == "batch":
            with ui.expansion().classes('w-full border rounded-xl shadow-sm bg-white overflow-hidden mb-4') as exp:
                with exp.add_slot('header'):
                    with ui.row().classes('w-full items-center justify-between py-2'):
                        with ui.row().classes('items-center gap-3'):
                            ui.icon('folder', size='md', color='amber-500')
                            with ui.column().classes('gap-0'):
                                ui.label(project["name"]).classes('text-base font-semibold text-slate-800')
                                ui.label(f'Batch • {project["books_count"]} books').classes('text-xs text-slate-400')
                        with ui.row().classes('items-center gap-4'):
                            get_status_badge(project["status"])
                            ui.linear_progress(value=project["progress"]).classes('w-24 h-2 rounded-full')
                
                with ui.column().classes('w-full p-4 bg-slate-50 border-t gap-3'):
                    for book in project.get("books", []):
                        with ui.row().classes('w-full justify-between items-center bg-white p-3 rounded-lg border shadow-xs'):
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('menu_book', size='sm', color='slate-400')
                                ui.label(book["name"]).classes('text-sm font-medium text-slate-700')
                            with ui.row().classes('items-center gap-4'):
                                get_status_badge(book["status"])
                                ui.linear_progress(value=book["progress"]).classes('w-16 h-1 rounded-full')
        else:
            with ui.card().classes('w-full border rounded-xl shadow-sm hover:shadow-md transition-shadow p-5 mb-4 bg-white'):
                with ui.row().classes('w-full items-center justify-between'):
                    with ui.row().classes('items-center gap-3'):
                        ui.icon('menu_book', size='md', color='blue-500')
                        with ui.column().classes('gap-0'):
                            ui.label(project["name"]).classes('text-base font-semibold text-slate-800')
                            ui.label('Single Novel').classes('text-xs text-slate-400')
                    with ui.row().classes('items-center gap-4'):
                        get_status_badge(project["status"])
                        ui.linear_progress(value=project["progress"]).classes('w-24 h-2 rounded-full')

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
        ui.button('+ New Project', on_click=lambda: new_project_dialog.open()).classes('bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg shadow-sm text-sm font-semibold capitalize')

    dashboard_container()


# --- Modal/Dialog: Add New Project (Placeholder) ---
with ui.dialog() as new_project_dialog, ui.card().classes('w-full max-w-lg p-6 rounded-xl'):
    ui.label('Create New Project').classes('text-xl font-bold text-slate-800 mb-4')
    with ui.column().classes('w-full gap-4'):
        project_name_input = ui.input('Project Title', placeholder='e.g. Master and Commander').classes('w-full')
        ui.label('Project Structure').classes('text-sm font-medium text-slate-500')
        type_radio = ui.radio(options={'single': 'Single Novel', 'batch': 'Batch of Books'}, value='single').props('inline')
        ui.label('Ingestion Source').classes('text-sm font-medium text-slate-500')
        source_radio = ui.radio(options={'transcribe': 'Transcribe Audio', 'epub': 'Import Text/EPUB'}, value='transcribe').props('inline')
        ui.separator().classes('my-2')
        with ui.row().classes('w-full justify-end gap-3 mt-2'):
            ui.button('Cancel', on_click=new_project_dialog.close).props('flat color=slate')
            ui.button('Create', on_click=new_project_dialog.close).classes('bg-blue-600 hover:bg-blue-700 text-white')

# Launch our app
ui.run(title="ABI-Pipeline")