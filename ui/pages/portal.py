import asyncio
from typing import Callable
from nicegui import ui
from sqlmodel import Session, select
from database.connection import engine
from database.models import Project, Book
from services.scanner import scan_directory, ingest_project
from ui import state

def get_status_badge(status: str):
    display_mapping = {
        "Imported": "Transcription",
        "Transcribing": "Transcription",
        "Transcribed": "Prompt Gen",
        "Generating Prompts": "Prompt Gen",
        "Prompts Created": "Image Gen",
        "Rendering Images": "Image Gen",
        "Images Created": "Finished",
        "Proofreading": "Finished",
        "Finished": "Finished"
    }
    display_status = display_mapping.get(status, status)
    
    styles = {
        "Transcription": "bg-slate-200 text-slate-700",
        "Prompt Gen": "bg-blue-100 text-blue-800 border-blue-200",
        "Image Gen": "bg-purple-100 text-purple-800 border-purple-200",
        "Finished": "bg-emerald-100 text-emerald-800 border-emerald-200"
    }
    style = styles.get(display_status, "bg-slate-100 text-slate-800")
    return ui.badge(display_status).classes(f'px-3 py-1 text-xs rounded-full font-medium {style}')


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


def toggle_project_expansion(project_id: int, refresh_parent: Callable):
    """Toggles the expansion visibility state for a project card."""
    if project_id in state.expanded_projects:
        state.expanded_projects.remove(project_id)
    else:
        state.expanded_projects.add(project_id)
    refresh_parent()


def render_portal_view(select_project_cb: Callable, select_book_cb: Callable, refresh_parent: Callable):
    from services.sync_engine import get_book_stats_cached

    projects_data = []
    with Session(engine) as session:
        projects = session.exec(select(Project)).all()
        for p in projects:
            books = session.exec(select(Book).where(Book.project_id == p.id)).all()
            
            # Aggregate stats across all child volumes/books on-the-fly
            total_words = 0
            total_scenes = 0
            total_completed = 0
            books_data = []
            
            for b in books:
                stats = get_book_stats_cached(p.name, b.name)
                total_words += stats["word_count"]
                
                # Deduce total scenes either from existing prompts or fallback estimates
                b_scenes = stats["total_prompts"] if stats["total_prompts"] > 0 else stats["estimated_scenes"]
                total_scenes += b_scenes
                total_completed += stats["generated_images"]
                
                books_data.append({
                    "id": b.id,
                    "name": b.name,
                    "status": b.status,
                    "progress": b.progress,
                    "cover_path": b.cover_path,
                    "stats": stats
                })

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
                "books": books,
                "books_data": books_data,
                "total_words": total_words,
                "total_scenes": total_scenes,
                "total_completed": total_completed
            })

    # Deep-filtering logic: matching project titles OR nested book titles
    filtered = []
    for p in projects_data:
        project_name_match = state.search_query.lower() in p["name"].lower()
        book_name_match = any(state.search_query.lower() in b["name"].lower() for b in p["books_data"])
        name_match = project_name_match or book_name_match
        
        type_match = True
        if state.selected_project_type == "Single" and p["is_batch"]:
            type_match = False
        elif state.selected_project_type == "Batch" and not p["is_batch"]:
            type_match = False
            
        if name_match and type_match:
            # Polished UX: Auto-expand projects if the search query matches its nested book but not its name
            if state.search_query.strip() and book_name_match and not project_name_match:
                state.expanded_projects.add(p["id"])
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
            is_expanded = project["id"] in state.expanded_projects
            
            with ui.card().classes('w-full border rounded-xl shadow-sm p-5 mb-4 bg-white transition-all'):
                # 1. Main Dashboard Card Header
                with ui.row().classes('w-full items-center justify-between'):
                    # Clicking the title/icon area takes you directly to the project's workspace
                    with ui.row().classes('items-center gap-3 cursor-pointer') \
                            .on('click', lambda p_id=project["id"]: select_project_cb(p_id)):
                        if project["is_batch"]:
                            ui.icon('folder', size='md', color='amber-500')
                            with ui.column().classes('gap-0'):
                                ui.label(project["name"]).classes('text-base font-semibold text-slate-800 leading-tight')
                                ui.label(f'Batch Workspace • {project["books_count"]} volumes').classes('text-xs text-slate-400')
                        else:
                            ui.icon('menu_book', size='md', color='blue-500')
                            with ui.column().classes('gap-0'):
                                ui.label(project["name"]).classes('text-base font-semibold text-slate-800 leading-tight')
                                ui.label('Single Novel Workspace').classes('text-xs text-slate-400')
                    
                    # Performance stats, status badges, and expandable control actions
                    with ui.row().classes('items-center gap-4'):
                        get_status_badge(project["status"])
                        
                        # Aggregated numeric statistics
                        with ui.column().classes('items-end gap-0'):
                            ui.label(f'{project["total_words"]:,} words').classes('text-[11px] font-semibold text-slate-500')
                            ui.label(f'{project["total_completed"]}/{project["total_scenes"]} rendered').classes('text-[10px] text-slate-400 font-medium')
                        
                        ui.linear_progress(value=project["progress"], show_value=False).classes('w-24 h-2 rounded-full')
                        
                        # Workspace shortcut trigger
                        ui.button(
                            'Open Workspace', 
                            icon='launch',
                            on_click=lambda p_id=project["id"]: select_project_cb(p_id)
                        ).props('flat dense').classes('text-blue-600 text-xs font-bold capitalize')

                        # Collapsible Detail Toggle Chevron
                        chevron_icon = 'expand_less' if is_expanded else 'expand_more'
                        ui.button(
                            icon=chevron_icon,
                            on_click=lambda p_id=project["id"]: toggle_project_expansion(p_id, refresh_parent)
                        ).props('flat round dense').classes('text-slate-500')

                # 2. Collapsible Child Book Listings details
                if is_expanded:
                    ui.separator().classes('my-3')
                    with ui.column().classes('w-full gap-3 pl-8'):
                        ui.label('Volumes included').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wider')
                        
                        for b in project["books_data"]:
                            with ui.row().classes('w-full items-center justify-between bg-slate-50 border border-slate-100 rounded-lg p-3 hover:bg-slate-100/50 transition-colors'):
                                # Book thumbnail, title, and file counters
                                with ui.row().classes('items-center gap-3 flex-1 min-w-0'):
                                    if b["cover_path"]:
                                        ui.image(b["cover_path"]).classes('w-10 h-14 rounded object-cover shadow-sm border border-slate-200 flex-shrink-0')
                                    else:
                                        with ui.column().classes('w-10 h-14 bg-slate-100 border border-dashed border-slate-300 rounded items-center justify-center flex-shrink-0 text-slate-400'):
                                            ui.icon('library_books', size='16px')
                                            
                                    with ui.column().classes('gap-0.5 flex-1 min-w-0'):
                                        ui.label(b["name"]).classes('text-sm font-semibold text-slate-800 truncate')
                                        if b["stats"]["has_transcript"]:
                                            total_b_scenes = b["stats"]["total_prompts"] or b["stats"]["estimated_scenes"]
                                            ui.label(f'{b["stats"]["word_count"]:,} words • {b["stats"]["generated_images"]}/{total_b_scenes} rendered').classes('text-xs text-slate-500 font-medium')
                                        else:
                                            ui.label('Awaiting transcription or text import').classes('text-xs text-slate-400 italic')

                                # Direct Book Selection Navigation Trigger
                                with ui.row().classes('items-center gap-3'):
                                    ui.badge(b["status"]).classes('px-2.5 py-0.5 text-[10px] rounded bg-slate-200 text-slate-700 font-medium')
                                    ui.button(
                                        'Jump to Book',
                                        icon='chevron_right',
                                        on_click=lambda p_id=project["id"], b_id=b["id"]: select_book_cb(p_id, b_id)
                                    ).props('flat dense').classes('text-xs text-blue-600 font-bold capitalize')

    with ui.dialog() as new_project_dialog, ui.card().classes('w-full max-w-2xl p-6 rounded-xl'):
        ui.label('Import New Project').classes('text-xl font-bold text-slate-800 mb-2')
        
        # Elegant header navigation tabs
        with ui.tabs().classes('w-full border-b mb-4') as tabs:
            audiobook_tab = ui.tab('Audiobook Folder', icon='folder')
            txt_tab = ui.tab('Text Transcripts', icon='description')
            epub_tab = ui.tab('EPUB Novels', icon='book')
            
        with ui.tab_panels(tabs, value=audiobook_tab).classes('w-full bg-transparent p-0') as panels:
            
            # TAB 1: Standard Audiobook Directories Importer
            with ui.tab_panel(audiobook_tab).classes('p-0 gap-4 column w-full'):
                ui.label('Select or enter a local audiobook directory path. We will analyze the structure and discover the covers automatically.').classes('text-xs text-slate-500 mb-2')
                
                with ui.row().classes('w-full items-end gap-2'):
                    path_input = ui.input(
                        'Local Directory Path', 
                        placeholder='e.g., F:/Audiobooks/Jack_Aubrey_Series',
                        on_change=lambda e: run_live_scan(e, new_project_dialog, refresh_parent)
                    ).classes('flex-1')
                    
                    async def browse_folder():
                        from services.picker import run_directory_picker
                        selected_path = await asyncio.to_thread(run_directory_picker, "Select Audiobook Directory")
                        if selected_path:
                            path_input.set_value(selected_path)
                    
                    ui.button(
                        icon='folder_open', 
                        on_click=browse_folder
                    ).props('flat dense').classes('h-10 text-blue-600').tooltip('Browse Local Folders')
                
                scan_preview_container(new_project_dialog, refresh_parent)
                
            # TAB 2: Text Transcript Files Importer (.txt)
            with ui.tab_panel(txt_tab).classes('p-0 gap-4 column w-full'):
                ui.label('Select one or more plain text files containing transcriptions. We will split chapters dynamically on ==CHAPTER== tags.').classes('text-xs text-slate-500 mb-2')
                
                ui.input(
                    'Project Title', 
                    placeholder='e.g., Sherlock Holmes Collection',
                    value=state.import_project_name,
                    on_change=lambda e: setattr(state, 'import_project_name', e.value)
                ).classes('w-full')
                
                async def browse_txt_files():
                    from services.picker import run_file_picker
                    selected = await asyncio.to_thread(
                        run_file_picker, 
                        "Select Transcript TXT Files", 
                        [("Text Files", "*.txt")]
                    )
                    if selected:
                        state.selected_txt_files.extend(selected)
                        # Filter unique paths to avoid duplicates
                        state.selected_txt_files = list(set(state.selected_txt_files))
                        txt_files_list.refresh()
                
                ui.button(
                    'Browse TXT Files...', 
                    icon='add', 
                    on_click=browse_txt_files
                ).classes('w-full bg-blue-50 text-blue-600 border border-dashed border-blue-200 shadow-none hover:bg-blue-100')
                
                # Reactive listing of selected file paths
                @ui.refreshable
                def txt_files_list():
                    if not state.selected_txt_files:
                        with ui.column().classes('w-full items-center justify-center p-4 text-slate-400 border border-dashed rounded-lg bg-slate-50'):
                            ui.label('No text files selected yet.').classes('text-xs')
                        return
                    
                    with ui.column().classes('w-full gap-2 max-h-40 overflow-y-auto p-2 bg-slate-50 border rounded-lg'):
                        for path in state.selected_txt_files:
                            from pathlib import Path
                            filename = Path(path).name
                            with ui.row().classes('w-full justify-between items-center bg-white p-2 rounded border shadow-xs'):
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('description', size='xs', color='slate-400')
                                    ui.label(filename).classes('text-xs font-medium text-slate-700 truncate max-w-sm')
                                ui.button(
                                    icon='delete', 
                                    on_click=lambda p=path: (state.selected_txt_files.remove(p), txt_files_list.refresh())
                                ).props('flat dense round').classes('text-rose-500 h-6 w-6')
                
                txt_files_list()
                
                # Async Click Handler keeps client context active
                async def do_txt_import():
                    await trigger_text_import(new_project_dialog, refresh_parent)
                
                with ui.row().classes('w-full justify-end gap-3 mt-4'):
                    ui.button('Cancel', on_click=new_project_dialog.close).props('flat color=slate')
                    ui.button(
                        'Import Transcripts', 
                        on_click=do_txt_import
                    ).classes('bg-blue-600 hover:bg-blue-700 text-white font-semibold')
                    
            # TAB 3: EPUB Books Importer (.epub)
            with ui.tab_panel(epub_tab).classes('p-0 gap-4 column w-full'):
                ui.label('Select one or more EPUB novel files. We will automatically extract texts and structural chapters.').classes('text-xs text-slate-500 mb-2')
                
                ui.input(
                    'Project Title', 
                    placeholder='e.g., Harry Potter Collection',
                    value=state.import_project_name,
                    on_change=lambda e: setattr(state, 'import_project_name', e.value)
                ).classes('w-full')
                
                async def browse_epub_files():
                    from services.picker import run_file_picker
                    selected = await asyncio.to_thread(
                        run_file_picker, 
                        "Select EPUB Book Files", 
                        [("EPUB Books", "*.epub")]
                    )
                    if selected:
                        state.selected_epub_files.extend(selected)
                        state.selected_epub_files = list(set(state.selected_epub_files))
                        epub_files_list.refresh()
                
                ui.button(
                    'Browse EPUB Files...', 
                    icon='add', 
                    on_click=browse_epub_files
                ).classes('w-full bg-blue-50 text-blue-600 border border-dashed border-blue-200 shadow-none hover:bg-blue-100')
                
                # Reactive listing of selected file paths
                @ui.refreshable
                def epub_files_list():
                    if not state.selected_epub_files:
                        with ui.column().classes('w-full items-center justify-center p-4 text-slate-400 border border-dashed rounded-lg bg-slate-50'):
                            ui.label('No EPUB files selected yet.').classes('text-xs')
                        return
                    
                    with ui.column().classes('w-full gap-2 max-h-40 overflow-y-auto p-2 bg-slate-50 border rounded-lg'):
                        for path in state.selected_epub_files:
                            from pathlib import Path
                            filename = Path(path).name
                            with ui.row().classes('w-full justify-between items-center bg-white p-2 rounded border shadow-xs'):
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('book', size='xs', color='slate-400')
                                    ui.label(filename).classes('text-xs font-medium text-slate-700 truncate max-w-sm')
                                ui.button(
                                    icon='delete', 
                                    on_click=lambda p=path: (state.selected_epub_files.remove(p), epub_files_list.refresh())
                                ).props('flat dense round').classes('text-rose-500 h-6 w-6')
                
                epub_files_list()
                
                # Async Click Handler keeps client context active
                async def do_epub_import():
                    await trigger_epub_import(new_project_dialog, refresh_parent)
                
                with ui.row().classes('w-full justify-end gap-3 mt-4'):
                    ui.button('Cancel', on_click=new_project_dialog.close).props('flat color=slate')
                    ui.button(
                        'Import EPUBs', 
                        on_click=do_epub_import
                    ).classes('bg-blue-600 hover:bg-blue-700 text-white font-semibold')


async def trigger_text_import(dialog, refresh_parent):
    """Imports plain text transcript files asynchronously inside the active client session."""
    if not state.import_project_name.strip():
        ui.notify("Please enter a Project Title first.", type="warning")
        return
    if not state.selected_txt_files:
        ui.notify("Please select at least one TXT file to import.", type="warning")
        return
        
    from services.import_engine import import_text_transcripts
    ui.notify("Processing and importing text transcripts...", type="info")
    project_id = await asyncio.to_thread(
        import_text_transcripts, 
        state.import_project_name.strip(), 
        state.selected_txt_files
    )
    ui.notify(f"Successfully imported project ID: {project_id}!", type="positive")
    dialog.close()
    refresh_parent()


async def trigger_epub_import(dialog, refresh_parent):
    """Extracts, parses, and imports EPUB chapters asynchronously inside the active client session."""
    if not state.import_project_name.strip():
        ui.notify("Please enter a Project Title first.", type="warning")
        return
    if not state.selected_epub_files:
        ui.notify("Please select at least one EPUB file to import.", type="warning")
        return
        
    from services.import_engine import import_epub_novels
    ui.notify("Parsing EPUB spine and extracting text chapters...", type="info")
    project_id = await asyncio.to_thread(
        import_epub_novels, 
        state.import_project_name.strip(), 
        state.selected_epub_files
    )
    ui.notify(f"Successfully imported project ID: {project_id}!", type="positive")
    dialog.close()
    refresh_parent()


def open_new_project_dialog(dialog, path_input):
    """Resets scanner and format caches before presenting the active import modal."""
    state.current_scan_result = None
    state.scan_error = ""
    state.custom_project_name_value = ""
    state.selected_txt_files = []
    state.selected_epub_files = []
    state.import_project_name = ""
    path_input.set_value("")
    dialog.open()