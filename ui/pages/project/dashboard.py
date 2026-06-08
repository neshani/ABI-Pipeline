import asyncio
import os
from pathlib import Path
from nicegui import ui
from sqlmodel import Session, select
from database.models import Project, Book
from ui import state
from database.connection import get_setting, engine

# Shared Playground render methods
from ui.pages.project.style_playground import render_style_playground_tab, open_style_chooser_modal_globally
from ui.pages.project.prompt_playground import render_prompt_playground_tab, list_stored_templates, load_template_by_name

STAGES = ["Imported", "Transcription", "Prompt Gen", "Image Gen", "Finished"]

def get_active_stage_idx(status: str) -> int:
    mapping = {
        "Imported": 0,
        "Transcribing": 1,
        "Transcribed": 2,
        "Generating Prompts": 2,
        "Prompts Created": 3,
        "Rendering Images": 3,
        "Images Created": 4,
        "Proofreading": 4,
        "Finished": 4
    }
    return mapping.get(status, 0)


def execute_project_rollback_io(project_id: int, choice: str) -> None:
    """Safely renames and archives files on disk depending on the user's rollback selection."""
    import datetime
    from sqlmodel import Session
    from database.connection import engine
    from database.models import Project, Book
    
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            state.add_console_log("[Rollback-Engine] Error: Project not found.")
            return
        project_name = project.name
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(get_setting("output_dir", "./output")).resolve()
    
    state.add_console_log(f"[Rollback-Engine] Initiating file archiver for project: {project_name} (Choice: {choice})")
    
    for book in books:
        book_dir = output_base / project_name / book.name
        if not book_dir.exists():
            state.add_console_log(f"[Rollback-Engine] Directory not found for book: {book.name}")
            continue
            
        state.add_console_log(f"[Rollback-Engine] Processing on-disk assets for book: {book.name}")
            
        # Choices cascade logically to maintain functional data constraints:
        # - "transcripts": Archives transcript.txt, prompts.csv, and images/
        # - "prompts": Archives prompts.csv and images/
        # - "images": Archives images/
        
        # 1. Archive transcripts
        if choice == "transcripts":
            transcript_path = book_dir / "transcript.txt"
            if transcript_path.exists():
                new_transcript = book_dir / f"transcript_backup_{timestamp}.txt"
                try:
                    transcript_path.rename(new_transcript)
                    state.add_console_log(f"[Rollback-Engine] Archived transcript to: {new_transcript.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving transcript.txt: {str(e)}")
                    
        # 2. Archive prompts (Choice of either "transcripts" or "prompts" wipes prompt-level states)
        if choice in ("transcripts", "prompts"):
            prompts_path = book_dir / "prompts.csv"
            if prompts_path.exists():
                new_prompts = book_dir / f"prompts_backup_{timestamp}.csv"
                try:
                    prompts_path.rename(new_prompts)
                    state.add_console_log(f"[Rollback-Engine] Archived prompts to: {new_prompts.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving prompts.csv: {str(e)}")
                    
        # 3. Archive rendered images (All options archive generated visual records)
        images_dir = book_dir / "images"
        if images_dir.exists() and images_dir.is_dir():
            new_images = book_dir / f"images_backup_{timestamp}"
            try:
                images_dir.rename(new_images)
                state.add_console_log(f"[Rollback-Engine] Archived images folder to: {new_images.name}")
            except Exception as e:
                state.add_console_log(f"[Rollback-Engine] Error archiving images folder: {str(e)}")
                    
    state.add_console_log("[Rollback-Engine] State rollback file archival step complete.")


def execute_project_delete_io(project_id: int) -> bool:
    """Deletes a project, its child books and chapters from DB, and completely wipes its output directory."""
    import shutil
    from sqlmodel import Session
    from database.connection import engine, get_setting
    from database.models import Project, Book, Chapter
    
    try:
        with Session(engine) as session:
            project = session.get(Project, project_id)
            if not project:
                state.add_console_log(f"[Delete-Engine] Error: Project ID {project_id} not found.")
                return False
                
            project_name = project.name
            state.add_console_log(f"[Delete-Engine] Initiating full wipe for project: {project_name}")
            
            # Fetch all books in the project
            books = session.exec(select(Book).where(Book.project_id == project_id)).all()
            for book in books:
                # Delete chapters of each book
                chapters = session.exec(select(Chapter).where(Chapter.book_id == book.id)).all()
                for chapter in chapters:
                    session.delete(chapter)
                session.delete(book)
                
            session.delete(project)
            session.commit()
            
        # Wipe the output directory on disk
        output_base = Path(get_setting("output_dir", "./output")).resolve()
        project_dir = output_base / project_name
        if project_dir.exists() and project_dir.is_dir():
            shutil.rmtree(project_dir)
            state.add_console_log(f"[Delete-Engine] Successfully deleted directory: {project_dir}")
        else:
            state.add_console_log(f"[Delete-Engine] Directory not found or already deleted: {project_dir}")
            
        state.add_console_log(f"[Delete-Engine] Project {project_name} deleted from database and disk.")
        return True
    except Exception as e:
        state.add_console_log(f"[Delete-Engine] Error during project deletion: {str(e)}")
        return False


def rescan_project_database_state(project_id: int) -> None:
    """Invokes sync operations across all project books to dynamically align database indexes with flat files."""
    from services.sync_engine import sync_book_from_disk
    from database.connection import touch_project
    with Session(engine) as session:
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        for b in books:
            sync_book_from_disk(b.id, session)
            
        project = session.get(Project, project_id)
        if project:
            state.project_status = project.status
        session.commit()
        
    # Touch the project to synchronize modified_at both in database and disk
    touch_project(project_id)


def render_transcription_step_view(project, books, start_transcribe_cb, stop_transcribe_cb):
    with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-3'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('record_voice_over', size='sm', color='blue-500')
            ui.label('Phase 1: Speech-to-Text Transcription').classes('text-sm font-bold text-slate-800')
            
        ui.label('The local STT pipeline scans and transcribes audiobook track files into chapter-level transcript.txt files.').classes('text-xs text-slate-500')
        
        with ui.row().classes('items-center gap-4 bg-blue-50/50 p-3 rounded-lg border border-blue-100/50 w-full mt-1'):
            ui.icon('info', color='blue', size='sm')
            ui.label(f'Active STT Config: {get_setting("stt_engine", "Parakeet ONNX")} | Device: {get_setting("stt_device", "GPU/CUDA")}').classes('text-xs font-semibold text-slate-700')
            
        with ui.row().classes('w-full justify-between items-center mt-2 pt-2 border-t border-slate-100'):
            if state.project_status == "Transcribing":
                with ui.row().classes('items-center gap-2'):
                    ui.spinner(size='sm', color='blue')
                    ui.label('Transcribing audio files in background...').classes('text-xs font-semibold text-blue-700 animate-pulse')
                ui.button(
                    'Stop Transcription', 
                    icon='stop', 
                    color='red', 
                    on_click=lambda: stop_transcribe_cb(project.id)
                ).classes('px-4 font-semibold text-xs')
            else:
                ui.label('Awaiting transcription pipeline initiation.').classes('text-xs font-semibold text-slate-400')
                ui.button(
                    'Start Transcription', 
                    icon='play_arrow', 
                    on_click=lambda: start_transcribe_cb(project.id)
                ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-5')


def render_prompt_gen_step_view(project, books, start_prompt_gen_cb, stop_transcribe_cb):
    available_templates = list_stored_templates()
    
    # Check for unapproved transcripts across all books in the project
    unapproved_books = []
    for b in books:
        t_path = Path(f"./output/{project.name}/{b.name}/transcript.txt")
        app_path = Path(f"./output/{project.name}/{b.name}/.transcript_approved")
        if t_path.exists() and not app_path.exists():
            unapproved_books.append(b.name)
            
    with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-3'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('psychology', size='sm', color='purple-500')
            ui.label('Phase 2: LLM Prompt Generation & Extraction').classes('text-sm font-bold text-slate-800')
            
        ui.label('Transcription passages are processed through a local LLM to extract key quotes and generate visual rendering prompts.').classes('text-xs text-slate-500')
        
        # Unapproved Warning Banner
        if unapproved_books:
            with ui.row().classes('w-full items-center justify-between gap-3 bg-amber-50 p-3 rounded-lg border border-amber-200 mt-1'):
                with ui.row().classes('items-center gap-3 flex-1 min-w-0'):
                    ui.icon('warning', color='amber', size='sm')
                    with ui.column().classes('gap-0 flex-1 min-w-0'):
                        ui.label('Review Required').classes('text-xs font-bold text-amber-800')
                        ui.label(f'The following volume(s) have unapproved transcripts: {", ".join(unapproved_books)}. Please approve them inside the book workspace before generating prompts.').classes('text-[10px] text-amber-700 leading-normal')
                
                def handle_approve_all():
                    output_base = Path(get_setting("output_dir", "./output")).resolve()
                    approved_count = 0
                    for b in books:
                        t_path = output_base / project.name / b.name / "transcript.txt"
                        app_path = output_base / project.name / b.name / ".transcript_approved"
                        if t_path.exists() and not app_path.exists():
                            try:
                                app_path.touch()
                                approved_count += 1
                            except Exception as ex:
                                state.add_console_log(f"[Dashboard] Error approving transcript for {b.name}: {str(ex)}")
                    
                    if approved_count > 0:
                        ui.notify(f"Approved {approved_count} volume transcripts!", type="positive")
                        rescan_project_database_state(project.id)
                        render_dynamic_step_dashboard.refresh()
                        if hasattr(state, 'active_header_refresh') and state.active_header_refresh:
                            state.active_header_refresh()
                    else:
                        ui.notify("No transcripts were eligible for approval.", type="info")

                ui.button(
                    'Approve All', 
                    icon='done_all', 
                    on_click=handle_approve_all
                ).classes('bg-amber-600 hover:bg-amber-700 text-white font-bold text-xs px-3 py-1.5 rounded')

        with ui.row().classes('w-full items-center gap-3 bg-slate-50 p-3 rounded-lg border mt-1'):
            ui.icon('description', size='sm', color='slate-500')
            with ui.column().classes('gap-0 flex-1'):
                ui.label('Active Guidelines Prompt Template').classes('text-xs font-bold text-slate-700')
                ui.label('Select template rules. Customize rules or add templates inside the Playground tab.').classes('text-[9px] text-slate-500')
            
            def handle_dashboard_template_change(val: str):
                if not val:
                    return
                state.playground_selected_template = val
                loaded = load_template_by_name(val)
                if loaded:
                    state.playground_template = loaded
                    # Auto-persist chosen template layout settings immediately
                    from services.project_settings import save_project_settings_to_disk
                    if state.active_project_id:
                        save_project_settings_to_disk(state.active_project_id)
                    ui.notify(f"Active Prompt Template changed to: {val}", type="info")

            ui.select(
                options=available_templates,
                value=state.playground_selected_template,
                on_change=lambda e: handle_dashboard_template_change(e.value)
            ).classes('w-48 bg-white').props('outlined dense')

        # --- Dynamic Scene Chunk Size Setting ---
        with ui.row().classes('w-full items-center gap-3 bg-slate-50 p-3 rounded-lg border mt-1'):
            ui.icon('wrap_text', size='sm', color='slate-500')
            with ui.column().classes('gap-0 flex-1'):
                ui.label('Scene Chunk Size (Words per Image)').classes('text-xs font-bold text-slate-700')
                ui.label('Defines the text volume analyzed for each scene. Smaller = more images.').classes('text-[9px] text-slate-500')
            
            def handle_chunk_size_change(val: int):
                if not val or val < 50:
                    val = 50
                state.playground_chunk_size = int(val)
                from services.project_settings import save_project_settings_to_disk
                if state.active_project_id:
                    save_project_settings_to_disk(state.active_project_id)
                
                # Clear stats cache and call registered update handler safely
                state._stats_cache.clear()
                if hasattr(state, "stats_refresh_callback") and state.stats_refresh_callback:
                    try:
                        state.stats_refresh_callback()
                    except Exception as ex:
                        state.add_console_log(f"[Dashboard] Error refreshing stats callback: {str(ex)}")
                
                ui.notify(f"Scene chunk size updated to {val} words.", type="info")

            ui.number(
                value=state.playground_chunk_size,
                min=50, max=2000, step=50,
                on_change=lambda e: handle_chunk_size_change(e.value)
            ).classes('w-48 bg-white').props('outlined dense suffix="words"')

        with ui.row().classes('w-full justify-between items-center mt-2 pt-2 border-t border-slate-100'):
            if state.project_status == "Generating Prompts":
                with ui.row().classes('items-center gap-2'):
                    ui.spinner(size='sm', color='purple')
                    ui.label('Generating prompts in background...').classes('text-xs font-semibold text-purple-700 animate-pulse')
                ui.button(
                    'Stop Prompt Gen', 
                    icon='stop', 
                    color='red', 
                    on_click=lambda: stop_transcribe_cb(project.id)
                ).classes('px-4 font-semibold text-xs')
            else:
                ui.label('Guidelines configured. Ready to run local prompt generation.').classes('text-xs font-semibold text-slate-400')
                
                # Intercept logic to display confirmation dialog if unapproved transcripts exist
                def confirm_and_generate_prompts():
                    if unapproved_books:
                        with ui.dialog() as confirm_dialog, ui.card().classes('w-full max-w-sm p-5 rounded-xl gap-4'):
                            ui.label('Unapproved Transcripts Detected').classes('text-base font-bold text-slate-800')
                            ui.label(f'Some volumes have not been approved yet: {", ".join(unapproved_books)}. Generating prompts now might analyze boilerplate or TOC headers.').classes('text-xs text-slate-500 leading-relaxed')
                            with ui.row().classes('w-full justify-end gap-2 mt-2'):
                                ui.button('Cancel', on_click=confirm_dialog.close).props('flat').classes('text-xs text-slate-600')
                                ui.button('Proceed Anyway', on_click=lambda: (confirm_dialog.close(), start_prompt_gen_cb(project.id) if start_prompt_gen_cb else None)).classes('bg-purple-600 hover:bg-purple-700 text-white font-bold text-xs px-4')
                        confirm_dialog.open()
                    else:
                        if start_prompt_gen_cb:
                            start_prompt_gen_cb(project.id)

                ui.button(
                    'Generate Prompts', 
                    icon='bolt', 
                    on_click=confirm_and_generate_prompts
                ).classes('bg-purple-600 hover:bg-purple-700 text-white font-bold text-xs px-5')


def render_image_gen_step_view(project, books, start_image_gen_cb, stop_transcribe_cb):
    with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-3'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('image', size='sm', color='amber-500')
            ui.label('Phase 3: ComfyUI Image Generation').classes('text-sm font-bold text-slate-800')
            
        ui.label('Render image batches via ComfyUI. Visual style settings and base workflows can be adjusted inside the Style & Workflows tab.').classes('text-xs text-slate-500')
        
        # Style preset configuration split - Uses the visual StyleChooserModal directly!
        with ui.row().classes('w-full items-center gap-3 bg-slate-50 p-3 rounded-lg border mt-1'):
            ui.icon('brush', size='sm', color='slate-500')
            with ui.column().classes('gap-0 flex-1'):
                ui.label('Active Visual Style Preset').classes('text-xs font-bold text-slate-700')
                ui.label('Selected style controls prompt prefixes and generation workflows.').classes('text-[9px] text-slate-500')
            
            with ui.row().classes('items-center gap-2'):
                ui.label().classes('text-xs font-bold text-blue-600 bg-white border px-3 py-1.5 rounded').bind_text_from(state, 'style_selected_preset')
                ui.button(
                    'Choose Style', 
                    icon='search', 
                    on_click=lambda: open_style_chooser_modal_globally()
                ).classes('bg-slate-700 text-white text-xs h-9 font-semibold')

        with ui.row().classes('w-full justify-between items-center mt-2 pt-2 border-t border-slate-100'):
            if state.project_status == "Rendering Images":
                with ui.row().classes('items-center gap-2'):
                    ui.spinner(size='sm', color='amber')
                    ui.label('Rendering images in background...').classes('text-xs font-semibold text-amber-700 animate-pulse')
                ui.button(
                    'Stop Rendering', 
                    icon='stop', 
                    color='red', 
                    on_click=lambda: stop_transcribe_cb(project.id)
                ).classes('px-4 font-semibold text-xs')
            else:
                ui.label('Batch prompts ready. Initiates automated image renderings.').classes('text-xs font-semibold text-slate-400')
                ui.button(
                    'Render Images', 
                    icon='play_circle_filled', 
                    on_click=lambda: start_image_gen_cb(project.id) if start_image_gen_cb else None
                ).classes('bg-amber-600 hover:bg-amber-700 text-white font-bold text-xs px-5')


def render_completed_step_view(project, books):
    with ui.card().classes('w-full border p-5 shadow-sm bg-emerald-50/10 gap-3'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('task_alt', size='sm', color='emerald-500')
            ui.label('Automatic Execution Phases Completed!').classes('text-sm font-bold text-slate-800')
            
        ui.label('All automatic generation steps have completed. You can now proofread chapters, tune image prompts, and bake description metadata.').classes('text-xs text-slate-500')
        
        with ui.row().classes('w-full gap-4 items-center bg-white p-4 rounded-lg border mt-2'):
            ui.icon('info', color='emerald')
            with ui.column().classes('gap-0 flex-1'):
                ui.label('Next Step: Workspace Proofreader').classes('text-xs font-bold text-slate-700')
                ui.label('Select a specific book in the left sidebar to open the Proofreader & Interactive Editor Grid.').classes('text-[10px] text-slate-500')

def open_large_image(img_base64: str, title: str):
    """Opens a modal popup dialog displaying the full-size rendered image using stable global references."""
    state.preview_image_title = title
    state.preview_image_src = img_base64
    if hasattr(state, 'global_preview_dialog') and state.global_preview_dialog:
        state.global_preview_dialog.open()


@ui.refreshable
def render_recent_images_feed():
    """Renders a real-time horizontal strip of the 5 most recent generated images."""
    with ui.card().classes('w-full border p-5 shadow-sm bg-white mt-4 gap-3'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('photo_library', size='sm', color='blue-500')
            ui.label('Live Rendered Images Feed (Most Recent)').classes('text-sm font-bold text-slate-800')
            
        if not state.recent_rendered_images:
            with ui.column().classes('w-full items-center justify-center p-6 bg-slate-50 border border-dashed rounded-lg text-slate-400'):
                ui.icon('image', size='md', color='slate-300')
                ui.label('Images will stream here in real-time as they are completed by ComfyUI...').classes('text-xs text-center')
        else:
            with ui.row().classes('w-full gap-4 items-start overflow-x-auto pb-2 flex-nowrap'):
                for item in reversed(state.recent_rendered_images):
                    with ui.card().classes('w-44 border p-2 rounded-lg shadow-xs bg-slate-50 flex-shrink-0 cursor-pointer') \
                            .on('click', lambda _, img=item['base64'], title=f"Ch {item['chapter']}, Scene {item['scene']}": open_large_image(img, title)):
                        ui.image(item['base64']).classes('w-full h-28 rounded object-cover border')
                        with ui.column().classes('gap-0 mt-1'):
                            ui.label(f"Ch {item['chapter']}, Scene {item['scene']}").classes('text-[10px] font-bold text-slate-700')
                            ui.label(item['prompt'][:40] + "...").classes('text-[8px] text-slate-500 leading-tight')

    state.recent_images_refresh = render_recent_images_feed.refresh


@ui.refreshable
def render_recent_prompts_feed():
    """Renders the last 5 generated prompts dynamically in-place during pipeline execution."""
    with ui.card().classes('w-full border p-5 shadow-sm bg-white mt-4 gap-3'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('auto_awesome', size='sm', color='blue-500')
            ui.label('Live Prompt Generation Feed (Most Recent)').classes('text-sm font-bold text-slate-800')
            
        if state.project_status in ("Imported", "Transcribing"):
            with ui.column().classes('w-full items-center justify-center p-6 bg-slate-50 border border-dashed rounded-lg text-slate-400'):
                ui.icon('lock', size='md', color='slate-300')
                ui.label('Feed is currently locked. Complete transcription to unlock prompt generation.').classes('text-xs text-center')
        elif not state.recent_prompts:
            with ui.column().classes('w-full items-center justify-center p-6 bg-slate-50 border border-dashed rounded-lg text-slate-400'):
                ui.icon('science', size='md', color='slate-300')
                ui.label('Feed is currently empty. Start prompt generation to stream live prompts here...').classes('text-xs text-center')
        else:
            with ui.column().classes('w-full gap-2'):
                for item in reversed(state.recent_prompts):
                    is_refusal = item["status"] == "refusal" or item["prompt"] == "REFUSAL"
                    bg_color = "bg-red-50/40 border-red-100" if is_refusal else "bg-slate-50/50 border-slate-100"
                    badge_label = "Refused" if is_refusal else "Success"
                    badge_color = "rose" if is_refusal else "emerald"
                    
                    with ui.card().classes(f'w-full border p-3 rounded-lg shadow-xs {bg_color}'):
                        with ui.row().classes('w-full justify-between items-center'):
                            ui.label(f"{item['book']} — Ch {item['chapter']}, Scene {item['scene']}").classes('text-[10px] font-bold text-slate-500 uppercase')
                            ui.badge(badge_label, color=badge_color).classes('text-[9px]')
                        ui.label(f'Quote: "{item["quote"][:220]}..."' if len(item["quote"]) > 220 else f'Quote: "{item["quote"]}"').classes('text-xs italic text-slate-600 leading-normal')
                        ui.label(f'Prompt: {item["prompt"]}').classes('text-xs font-semibold text-blue-700 leading-normal')

    state.recent_prompts_refresh = render_recent_prompts_feed.refresh


def render_project_tabs(
    project: Project, 
    books: list, 
    start_transcribe_cb, 
    stop_transcribe_cb,
    start_prompt_gen_cb=None,
    start_image_gen_cb=None,
    save_project_settings_cb=None
):
    # Safeguard selected template checks
    available_templates = list_stored_templates()
    if not state.playground_selected_template or state.playground_selected_template not in available_templates:
        state.playground_selected_template = "default"
    
    if not state.playground_template:
        state.playground_template = load_template_by_name(state.playground_selected_template)

    book_names = [b.name for b in books]
    if books and (not state.playground_book_selection or state.playground_book_selection not in book_names):
        state.playground_book_selection = books[0].name

    with ui.dialog() as global_preview_dialog:
        with ui.card().classes('w-full max-w-3xl p-4 items-center bg-white rounded-xl shadow-lg'):
            ui.label().classes('text-sm font-bold text-slate-800 mb-3 uppercase tracking-wider').bind_text_from(state, 'preview_image_title')
            ui.image().props('fit=contain').classes('w-full rounded-lg max-h-[70vh] cursor-zoom-out bg-slate-50/30').bind_source_from(state, 'preview_image_src').on('click', global_preview_dialog.close)
            with ui.row().classes('w-full justify-end mt-3'):
                ui.button('Close', on_click=global_preview_dialog.close).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs')

    state.global_preview_dialog = global_preview_dialog

    # Reusable Rollback Configuration Radio Modal
    with ui.dialog() as rollback_dialog, ui.card().classes('w-full max-w-md p-6 rounded-xl gap-4'):
        ui.label('Confirm State Rollback').classes('text-lg font-bold text-slate-800')
        ui.label('Select how far back you want to roll this project. Your existing files will be safely archived on disk.').classes('text-xs text-slate-500 leading-normal')
        
        rollback_choice = ui.radio({
            'images': 'Archive Images and regenerate images (Archives images only)',
            'prompts': 'Archive prompts and regenerate prompts (Includes archiving images)',
            'transcripts': 'Archive transcripts and re-transcribe audio (Includes archiving prompts and images)'
        }, value='images').classes('text-xs w-full gap-2 p-2 border rounded bg-slate-50/50')
        
        async def confirm_action():
            choice = rollback_choice.value
            ui.notify("Executing rollback and archiving files...", type="info")
            await asyncio.to_thread(execute_project_rollback_io, project.id, choice)
            await asyncio.to_thread(rescan_project_database_state, project.id)
            
            render_dynamic_step_dashboard.refresh()
            if hasattr(state, 'active_header_refresh') and state.active_header_refresh:
                state.active_header_refresh()
            rollback_dialog.close()
            ui.notify("Project rolled back successfully!", type="positive")
            
        with ui.row().classes('w-full justify-end gap-3 mt-2'):
            ui.button('Cancel', on_click=rollback_dialog.close).props('flat color=slate').classes('text-xs font-semibold')
            ui.button('Confirm & Rollback', on_click=confirm_action, color='red').classes('text-xs font-bold text-white')

    # Reusable Delete Confirmation Modal
    with ui.dialog() as delete_dialog, ui.card().classes('w-full max-w-md p-6 rounded-xl gap-4'):
        ui.label('Delete Project?').classes('text-lg font-bold text-rose-800')
        ui.label('Warning: This action is permanent!').classes('text-xs font-bold text-slate-700')
        ui.label('This will completely delete the project from the database, along with all of its volumes, transcripts, prompts, and generated images on disk.').classes('text-xs text-slate-500 leading-normal')
        
        async def confirm_delete():
            ui.notify("Deleting project and wiping files...", type="info")
            success = await asyncio.to_thread(execute_project_delete_io, project.id)
            if success:
                ui.notify("Project deleted successfully!", type="positive")
                # Clean exit state to portal
                state.active_project_id = None
                state.active_book_id = None
                state.active_tool = None
                state.active_log_widget = None
                
                from ui.pages import main_layout_ref
                if main_layout_ref:
                    main_layout_ref.refresh()
                if hasattr(state, 'active_header_refresh') and state.active_header_refresh:
                    state.active_header_refresh()
            else:
                ui.notify("Error deleting project. Check the process logs.", type="negative")
            delete_dialog.close()
            
        with ui.row().classes('w-full justify-end gap-3 mt-2'):
            ui.button('Cancel', on_click=delete_dialog.close).props('flat color=slate').classes('text-xs font-semibold')
            ui.button('Confirm & Delete', on_click=confirm_delete, color='red').classes('text-xs font-bold text-white')

    # Workspace Navigation Layout with Folder, Rollback, and Delete Shortcuts
    with ui.row().classes('w-full justify-between items-center mb-1'):
        with ui.column().classes('gap-0'):
            ui.label('Project Workspace Controls').classes('text-base font-bold text-slate-800')
            ui.label('Configure orchestration guidelines and render dynamic style models.').classes('text-xs text-slate-500')
        
        def open_project_folder():
            import platform
            import subprocess
            base_dir = Path(get_setting("output_dir", "./output")).resolve()
            proj_dir = base_dir / project.name
            proj_dir.mkdir(parents=True, exist_ok=True)
            try:
                if platform.system() == "Windows":
                    os.startfile(proj_dir)
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", str(proj_dir)])
                else:
                    subprocess.Popen(["xdg-open", str(proj_dir)])
            except Exception as ex:
                ui.notify(f"Failed to open project folder: {str(ex)}", type="negative")

        with ui.row().classes('items-center gap-2'):
            ui.button(
                'Open Folder', 
                icon='folder_open', 
                on_click=open_project_folder
            ).props('flat dense').classes('text-xs text-slate-600')
            ui.button(
                'Rollback Project',
                icon='history',
                on_click=rollback_dialog.open
            ).props('flat dense').classes('text-xs text-red-600 hover:text-red-800')
            ui.button(
                'Delete Project',
                icon='delete_forever',
                on_click=delete_dialog.open
            ).props('flat dense').classes('text-xs text-rose-600 hover:text-rose-800')
    
    with ui.tabs().classes('w-full border-b') as project_tabs:
        tab_dash = ui.tab('Dashboard', icon='dashboard')
        tab_style = ui.tab('Style & Workflows', icon='brush')
        tab_play = ui.tab('Prompt-Gen Playground', icon='science')
        
    project_tabs.bind_value(state, 'active_project_tab')
        
    with ui.tab_panels(project_tabs, value=state.active_project_tab).classes('w-full bg-transparent p-0'):
        with ui.tab_panel(tab_dash):
            with ui.column().classes('w-full gap-4'):
                global render_dynamic_step_dashboard
                @ui.refreshable
                def render_dynamic_step_dashboard_local():
                    status = state.project_status
                    if status in ("Imported", "Transcribing"):
                        render_transcription_step_view(project, books, start_transcribe_cb, stop_transcribe_cb)
                    elif status in ("Transcribed", "Generating Prompts"):
                        render_prompt_gen_step_view(project, books, start_prompt_gen_cb, stop_transcribe_cb)
                    elif status in ("Prompts Created", "Rendering Images"):
                        render_image_gen_step_view(project, books, start_image_gen_cb, stop_transcribe_cb)
                    else:
                        render_completed_step_view(project, books)
                        
                render_dynamic_step_dashboard = render_dynamic_step_dashboard_local
                render_dynamic_step_dashboard()

            @ui.refreshable
            def render_conditional_feeds():
                status = state.project_status
                if status not in ("Imported", "Transcribing", "Transcribed", "Generating Prompts"):
                    render_recent_images_feed()
                    
            render_conditional_feeds()
            
            with ui.card().classes('w-full border p-5 shadow-sm bg-white mt-4 gap-3'):
                with ui.row().classes('w-full justify-between items-center'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('terminal', size='sm', color='slate-700')
                        ui.label('Process Control Console').classes('text-sm font-bold text-slate-800')
                    ui.button('Clear Logs', on_click=lambda: (state.console_logs.clear(), log_widget.clear())).props('flat dense').classes('text-xs text-slate-500')
                
                log_widget = ui.log(max_lines=300).classes('w-full h-64 bg-slate-900 text-slate-100 font-mono text-xs p-3 rounded-lg leading-relaxed')
                for line in state.console_logs:
                    log_widget.push(line)
                
                state.active_log_widget = log_widget
                state.logs_pushed_index = len(state.console_logs)

            @ui.refreshable
            def render_conditional_prompt_feed():
                status = state.project_status
                if status not in ("Imported", "Transcribing"):
                    render_recent_prompts_feed()
                    
            render_conditional_prompt_feed()

            state.action_buttons_refresh = lambda: (
                render_dynamic_step_dashboard.refresh(), 
                render_conditional_feeds.refresh(), 
                render_conditional_prompt_feed.refresh()
            )
                        
        with ui.tab_panel(tab_style):
            # Delegates rendering directly to the modular style_playground page!
            render_style_playground_tab(project, save_project_settings_cb)
                
        with ui.tab_panel(tab_play):
            # Delegates rendering directly to the modular prompt_playground page!
            render_prompt_playground_tab(project, books)