import asyncio
from typing import Callable
from nicegui import ui
from sqlmodel import Session, select
from database.connection import engine
from database.models import Project, Book
from services.scanner import scan_directory, ingest_project
from ui import state

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


@ui.refreshable
def scan_preview_container(new_project_dialog, refresh_parent: Callable):
    if state.scan_error:
        with ui.row().classes('items-center gap-2 p-3 bg-red-50 text-red-700 rounded-lg border border-red-200 w-full'):
            ui.icon('error_outline', size='sm')
            ui.label(state.scan_error).classes('text-sm font-medium')
        return

    if not state.current_scan_result:
        with ui.column().classes('w-full items-center justify-center p-6 text-slate-400 border border-dashed rounded-lg bg-slate-50'):
            ui.icon('folder_open', size='lg')
            ui.label('Waiting for a valid local audiobook directory path...').classes('text-xs text-center')
        return

    with ui.column().classes('w-full gap-4'):
        ui.input(
            'Project Title', 
            value=state.custom_project_name_value,
            on_change=lambda e: setattr(state, 'custom_project_name_value', e.value)
        ).classes('w-full')
        
        with ui.row().classes('items-center gap-2'):
            if state.current_scan_result["type"] == "single":
                ui.icon('menu_book', color='blue-500', size='sm')
                ui.label('Structure: Single Novel').classes('text-sm font-semibold text-slate-700')
            else:
                ui.icon('folder', color='amber-500', size='sm')
                ui.label(f'Structure: Batch ({len(state.current_scan_result["books"])} audiobooks found)').classes('text-sm font-semibold text-slate-700')
        
        with ui.column().classes('w-full gap-2 max-h-48 overflow-y-auto p-2 bg-slate-50 border rounded-lg'):
            for book in state.current_scan_result["books"]:
                with ui.row().classes('w-full justify-between items-center bg-white p-2 rounded border shadow-xs'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('library_books', size='xs', color='slate-400')
                        ui.label(book["name"]).classes('text-xs font-medium text-slate-700 truncate max-w-sm')
                    with ui.row().classes('items-center gap-2'):
                        if book["cover_path"]:
                            ui.badge('Cover Found', color='emerald-100').classes('text-emerald-800 text-[10px] px-1.5 py-0.5 rounded font-bold')
                        ui.badge(f'{len(book["files"])} tracks', color='slate-100').classes('text-slate-600 text-[10px] px-1.5 py-0.5 rounded')

        with ui.row().classes('w-full justify-end gap-3 mt-2'):
            ui.button('Cancel', on_click=new_project_dialog.close).props('flat color=slate')
            ui.button(
                'Import & Create Project', 
                on_click=lambda: save_scanned_project(new_project_dialog, refresh_parent)
            ).classes('bg-blue-600 hover:bg-blue-700 text-white font-semibold')


async def run_live_scan(e, new_project_dialog, refresh_parent: Callable):
    path_str = e.value.strip()
    if not path_str:
        state.current_scan_result = None
        state.scan_error = ""
        state.custom_project_name_value = ""
        scan_preview_container.refresh(new_project_dialog, refresh_parent)
        return
        
    try:
        result = await asyncio.to_thread(scan_directory, path_str)
        if result["type"] == "none":
            state.current_scan_result = None
            state.scan_error = "No supported audiobook files or subdirectories found."
            state.custom_project_name_value = ""
        else:
            state.current_scan_result = result
            state.scan_error = ""
            state.custom_project_name_value = result["project_name"]
    except Exception as ex:
        state.current_scan_result = None
        state.scan_error = f"Error scanning folder: {str(ex)}"
        state.custom_project_name_value = ""
        
    scan_preview_container.refresh(new_project_dialog, refresh_parent)


def save_scanned_project(new_project_dialog, refresh_parent: Callable):
    if not state.current_scan_result:
        ui.notify("No valid scanned project to save.", type="negative")
        return
        
    try:
        project_id = ingest_project(state.current_scan_result, state.custom_project_name_value)
        ui.notify(f"Successfully imported project ID: {project_id}!", type="positive")
        new_project_dialog.close()
        refresh_parent()
    except Exception as ex:
        ui.notify(f"Failed to save project: {str(ex)}", type="negative")


def render_portal_view(select_project_cb: Callable, refresh_parent: Callable):
    projects_data = []
    with Session(engine) as session:
        projects = session.exec(select(Project)).all()
        for p in projects:
            books = session.exec(select(Book).where(Book.project_id == p.id)).all()
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

    filtered = []
    for p in projects_data:
        name_match = state.search_query.lower() in p["name"].lower()
        
        type_match = True
        if state.selected_project_type == "Single" and p["is_batch"]:
            type_match = False
        elif state.selected_project_type == "Batch" and not p["is_batch"]:
            type_match = False
            
        if name_match and type_match:
            filtered.append(p)

    with ui.row().classes('w-full justify-between items-center mb-2'):
        with ui.column().classes('gap-0'):
            ui.label('Project Dashboard').classes('text-2xl font-bold text-slate-800')
            ui.label('Manage audiobooks, generate prompts, and render pipeline.').classes('text-sm text-slate-500')
        ui.button(
            '+ New Project', 
            on_click=lambda: open_new_project_dialog(new_project_dialog, path_input)
        ).classes('bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg shadow-sm text-sm font-semibold capitalize')

    if not filtered:
        with ui.column().classes('w-full items-center justify-center p-12 text-slate-400'):
            ui.icon('search', size='lg')
            ui.label('No projects found. Use "+ New Project" to import audiobooks.').classes('text-lg text-center')
    else:
        for project in filtered:
            with ui.card().classes('w-full border rounded-xl shadow-sm hover:shadow-md transition-all p-5 mb-4 bg-white cursor-pointer') \
                    .on('click', lambda p_id=project["id"]: select_project_cb(p_id)):
                with ui.row().classes('w-full items-center justify-between'):
                    with ui.row().classes('items-center gap-3'):
                        if project["is_batch"]:
                            ui.icon('folder', size='md', color='amber-500')
                            with ui.column().classes('gap-0'):
                                ui.label(project["name"]).classes('text-base font-semibold text-slate-800')
                                ui.label(f'Batch Workspace • {project["books_count"]} books').classes('text-xs text-slate-400')
                        else:
                            ui.icon('menu_book', size='md', color='blue-500')
                            with ui.column().classes('gap-0'):
                                ui.label(project["name"]).classes('text-base font-semibold text-slate-800')
                                ui.label('Single Novel Workspace').classes('text-xs text-slate-400')
                    
                    with ui.row().classes('items-center gap-4'):
                        get_status_badge(project["status"])
                        ui.linear_progress(value=project["progress"], show_value=False).classes('w-24 h-2 rounded-full')
                        ui.icon('chevron_right', size='sm', color='slate-400')

    with ui.dialog() as new_project_dialog, ui.card().classes('w-full max-w-2xl p-6 rounded-xl'):
        ui.label('Create New Project').classes('text-xl font-bold text-slate-800 mb-2')
        ui.label('Enter a local audiobook directory path. We will analyze the structure and discover the covers automatically.').classes('text-sm text-slate-500 mb-4')
        
        with ui.column().classes('w-full gap-4'):
            path_input = ui.input(
                'Local Directory Path', 
                placeholder='e.g., F:/Audiobooks/Jack_Aubrey_Series',
                on_change=lambda e: run_live_scan(e, new_project_dialog, refresh_parent)
            ).classes('w-full')
            
            scan_preview_container(new_project_dialog, refresh_parent)


def open_new_project_dialog(dialog, path_input):
    state.current_scan_result = None
    state.scan_error = ""
    state.custom_project_name_value = ""
    path_input.set_value("")
    dialog.open()