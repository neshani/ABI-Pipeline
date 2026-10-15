import asyncio
import os
from pathlib import Path
from nicegui import ui, app
from sqlmodel import Session, select
from database.models import Project, Book
from ui import state
from database.connection import get_setting, engine

# Shared Playground render methods
from ui.pages.project.style_playground import render_style_playground_tab, open_style_chooser_modal_globally
from ui.pages.project.prompt_playground import render_prompt_playground_tab, list_stored_templates, load_template_by_name

STAGES = ["Imported", "Transcription", "Prompt Gen", "Image Gen"]


def _refresh_cleanup_progress() -> None:
    """Refresh the transcript-cleanup dialog, if it still exists.

    A cleanup run is a long-lived async task. If the panel re-renders mid-run
    (e.g. the user switches tabs) the dialog's slot is deleted, so refreshing it
    raises ``RuntimeError``. The run itself is unaffected - progress state lives
    in ``ui.state`` and is re-rendered when the panel comes back.
    """
    callback = getattr(state, "cleanup_refresh", None)
    if callback is None:
        return
    try:
        callback()
    except RuntimeError:
        pass

def get_active_stage_idx(status: str) -> int:
    mapping = {
        "Imported": 0,
        "Transcribing": 1,
        "Transcribed": 2,
        "Generating Prompts": 2,
        "Prompts Created": 3,
        "Rendering Images": 3,
        "Images Created": 3,
        "Proofreading": 3,
        "Finished": 3
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


def rescan_project_database_state(project_id: int, force: bool = False) -> None:
    """Invokes sync operations across all project books to dynamically align database indexes with flat files."""
    from services.sync_engine import sync_book_from_disk
    from database.connection import touch_project
    from services.transcription import active_projects
    
    # Determine active tasks for this specific project to skip destructive disk syncs
    is_transcribing = False
    if active_projects:
        if isinstance(active_projects, dict):
            is_transcribing = project_id in active_projects
        elif isinstance(active_projects, (set, list)):
            is_transcribing = project_id in active_projects or str(project_id) in active_projects

    is_prompt_gen = getattr(state, "prompt_gen_active", False) and state.active_project_id == project_id
    is_image_gen = getattr(state, "image_gen_active", False) and state.active_project_id == project_id

    skip_sync = (is_transcribing or is_prompt_gen or is_image_gen) and not force

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            state.add_console_log(f"[Sync-Engine] Error: Project ID {project_id} not found.")
            return

        if not skip_sync:
            books = session.exec(select(Book).where(Book.project_id == project_id)).all()
            for b in books:
                sync_book_from_disk(b.id, session)
            session.refresh(project)
        else:
            state.add_console_log(f"[Sync-Engine] Active pipeline process detected. Bypassing disk rescan to preserve runtime progress.")
            
            # Re-force active workflow statuses in the DB in case of transient state mismatch
            if is_transcribing and project.status != "Transcribing":
                project.status = "Transcribing"
                session.add(project)
            elif is_prompt_gen and project.status != "Generating Prompts":
                project.status = "Generating Prompts"
                session.add(project)
            elif is_image_gen and project.status != "Rendering Images":
                project.status = "Rendering Images"
                session.add(project)

        # Synchronize local UI state
        state.project_status = project.status
        
        # Restore active execution properties so global timers can track progress transitions cleanly
        if is_transcribing and not force:
            state.active_task_type = "transcription"
        elif is_prompt_gen and not force:
            state.active_task_type = "prompt_gen"
        elif is_image_gen and not force:
            state.active_task_type = "image_gen"
        else:
            if state.active_project_id == project_id:
                state.active_task_type = None

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
            
        # Contextual warning card dynamically bound to the pending restart state
        with ui.card().classes('w-full bg-amber-50 border border-amber-200 p-4 rounded-xl gap-2 mt-1') \
                .bind_visibility_from(state, 'needs_restart'):
            with ui.row().classes('items-center gap-2 text-amber-900 font-bold text-xs'):
                ui.icon('warning', size='18px', color='amber')
                ui.label('Engine Ready — App Restart Required')
            ui.label(
                "Dynamic transcription libraries were successfully installed but cannot be loaded in this session. "
                "Please shut down and restart the application cleanly to initialize the transcription engine."
            ).classes('text-[11px] text-amber-800 leading-normal')
            with ui.row().classes('w-full justify-end mt-1'):
                async def do_shutdown():
                    ui.notify("Shutting down cleanly... Please restart from your launcher.", type="warning", timeout=5)
                    await asyncio.sleep(1.5)
                    app.shutdown()
                ui.button(
                    'Shutdown App Safely', 
                    icon='power_settings_new', 
                    on_click=do_shutdown
                ).classes('bg-amber-600 text-white font-semibold text-xs')
            
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
                
                start_btn = ui.button(
                    'Start Transcription', 
                    icon='play_arrow', 
                    color='positive',
                    on_click=lambda: start_transcribe_cb(project.id)
                ).classes('text-white font-bold text-xs px-5')
                
                # Safe-guard action execution by binding its availability inversely to needs_restart
                start_btn.bind_enabled_from(state, 'needs_restart', backward=lambda x: not x)


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

        # --- Shot Planner mode ---
        # NB: NiceGUI select takes dict options as {value: label}.
        shot_mode_options = {
            "auto": "Plan shots (auto)",
            "wide": "All WIDE",
            "character": "All CHARACTER",
            "insert": "All INSERT",
            "off": "Custom single prompt",
        }

        def handle_default_shot_mode(val: str):
            from services import prompt_library as prompt_lib
            prompt_lib.set_default_shot_mode(val)
            ui.notify(f"Shot Planner: {shot_mode_options.get(val, val)}", type="info")

        def handle_book_shot_mode(book_name: str, val: str):
            from services import prompt_library as prompt_lib
            if not val:
                prompt_lib.clear_shot_mode(project.name, book_name)
                ui.notify(f"{book_name}: inherits the project default", type="info")
            else:
                prompt_lib.set_shot_mode(project.name, book_name, val)
                ui.notify(f"{book_name}: {shot_mode_options.get(val, val)}", type="info")

        from services import prompt_library as _pl

        with ui.row().classes('w-full items-center gap-3 bg-slate-50 p-3 rounded-lg border mt-1'):
            ui.icon('movie_filter', size='sm', color='slate-500')
            with ui.column().classes('gap-0 flex-1'):
                ui.label('Shot Planner').classes('text-xs font-bold text-slate-700')
                ui.label('How each scene decides its prompt. Prompts themselves are edited in the Prompt-Gen Playground.').classes('text-[9px] text-slate-500')

            ui.select(
                options=shot_mode_options,
                value=_pl.get_default_shot_mode(),
                on_change=lambda e: handle_default_shot_mode(e.value),
            ).classes('w-56 bg-white').props('outlined dense')

        with ui.expansion('Per-volume overrides (optional)').classes('w-full border rounded bg-slate-50 text-xs'):
            for b in books:
                override = _pl.get_shot_mode_override(project.name, b.name)
                with ui.row().classes('w-full items-center gap-3 px-2 py-1'):
                    ui.label(b.name).classes('text-[11px] flex-1 truncate')
                    ui.select(
                        options={"": "Inherit project default", **shot_mode_options},
                        value=override or "",
                        on_change=lambda e, bn=b.name: handle_book_shot_mode(bn, e.value),
                    ).classes('w-60').props('dense outlined')

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
                    color='positive',
                    on_click=confirm_and_generate_prompts
                ).classes('text-white font-bold text-xs px-5')


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
                    color='positive',
                    on_click=lambda: start_image_gen_cb(project.id) if start_image_gen_cb else None
                ).classes('text-white font-bold text-xs px-5')


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

def render_source_text_tab(project, books):
    """Optional repair step: match volumes to EPUB source files and rewrite
    their transcripts from the clean book text (timings carried across)."""
    from services import source_text as st

    book_names = [b.name for b in books]

    def resolved_pairs():
        resolved = st.resolve_project_sources(project.name, book_names)
        return [(n, str(info["chosen"])) for n, info in resolved.items() if info["chosen"]]

    def unaligned_books():
        return [b.name for b in books
                if not st.is_cleaned(project.name, b.name)
                and (st.book_dir(project.name, b.name) / "transcript.txt").exists()]

    # --- dialogs are declared once per render (project convention) ---
    # Progress lives in ui.state so a run survives a panel re-render.
    with ui.dialog().props('persistent') as cleanup_dialog, \
            ui.card().classes('w-full max-w-2xl p-6 rounded-xl gap-3'):

        async def start_cleanup():
            pairs = resolved_pairs()
            if not pairs:
                ui.notify('Nothing to clean - no source files matched.', type='info')
                return
            apply_corrections = getattr(state, 'spellfix_from_corrections', True)
            state.cleanup_progress.update(
                phase="running", current="", summary="",
                rows=[{"book": n, "kind": "align", "status": "pending", "detail": ""}
                      for n, _ in pairs],
            )
            _refresh_cleanup_progress()

            def row_for(book_name):
                return next(r for r in state.cleanup_progress["rows"] if r["book"] == book_name)

            state.add_console_log(f'[Align] Cleaning {len(pairs)} transcript(s) in "{project.name}"...')
            failures = 0
            for name, epub in pairs:
                row = row_for(name)
                row.update(status="working", detail=f"aligning against {Path(epub).name}")
                state.cleanup_progress["current"] = name
                _refresh_cleanup_progress()
                state.add_console_log(f'[Align] {name}: aligning against {Path(epub).name}')
                try:
                    stats = await asyncio.to_thread(st.clean_book, project.name, name, epub)
                    row.update(status="done", detail=(
                        f"{stats['emitted_words']:,} words · "
                        f"{stats['safe_corrections']:,} corrections · "
                        f"{stats['timing_chunks']:,} chunks"))
                    state.add_console_log(f"[Align] {name}: {row['detail']}")
                    for warning in stats.get('warnings', []):
                        state.add_console_log(f'[Align] {name}: WARNING {warning}')
                except Exception as ex:
                    row.update(status="error", detail=str(ex))
                    failures += 1
                    state.add_console_log(f'[Align] {name} FAILED: {ex}')
                _refresh_cleanup_progress()

            # Volumes with no source: repair their spelling from the corrections
            # learned above (and from any previous runs).
            if apply_corrections:
                from services import corrections as corr
                remaining = corr.books_without_source(project.name, book_names)
                if remaining:
                    state.cleanup_progress["rows"].extend(
                        {"book": n, "kind": "corrections", "status": "pending", "detail": ""}
                        for n in remaining)
                    _refresh_cleanup_progress()
                    vocab = await asyncio.to_thread(corr.project_clean_vocab, project.name, book_names)
                    agg = await asyncio.to_thread(
                        corr.aggregate_corrections, project.name, book_names, 2, True, vocab)
                    mapping = {a['wrong']: a['right'] for a in agg}
                    state.add_console_log(
                        f'[Spellfix] Applying {len(mapping)} learned corrections to '
                        f'{len(remaining)} volume(s) without a source...')
                    for name in remaining:
                        row = row_for(name)
                        row.update(status="working", detail=f"{len(mapping):,} learned corrections")
                        state.cleanup_progress["current"] = name
                        _refresh_cleanup_progress()
                        try:
                            stats = await asyncio.to_thread(
                                corr.apply_to_book, project.name, name, mapping, False)
                            row.update(status="done", detail=f"{stats['hits']:,} corrections applied")
                            state.add_console_log(f"[Spellfix] {name}: {stats['hits']} corrections")
                        except Exception as ex:
                            row.update(status="error", detail=str(ex))
                            failures += 1
                            state.add_console_log(f'[Spellfix] {name} FAILED: {ex}')
                        _refresh_cleanup_progress()

            succeeded = sum(1 for r in state.cleanup_progress["rows"] if r["status"] == "done")
            state.cleanup_progress.update(
                phase="done", current="",
                summary=f"{succeeded} succeeded, {failures} failed")
            _refresh_cleanup_progress()
            try:
                ui.notify(f'Cleanup finished - {state.cleanup_progress["summary"]}',
                          type='positive' if not failures else 'warning')
            except RuntimeError:
                pass

        @ui.refreshable
        def render_dialog():
            if state.cleanup_progress["phase"] == "confirm":
                ui.label('Confirm Transcript Cleanup').classes('text-lg font-bold text-slate-800')
                ui.label(
                    "Each volume below has its transcript.txt replaced with the book's own text "
                    "(spelling, punctuation and quotes corrected) and transcript_timing.json rebuilt "
                    "so quote timestamps keep working. The current ASR text and timings are backed up "
                    "first, so this can be reverted."
                ).classes('text-xs text-slate-500 leading-relaxed')

                with ui.column().classes('w-full gap-1 max-h-72 overflow-y-auto border rounded-lg p-3 bg-slate-50'):
                    pairs = resolved_pairs()
                    matched_names = {name for name, _ in pairs}
                    if not pairs:
                        ui.label('No volumes are matched to a source file.').classes('text-xs text-slate-500 italic')
                    for name, epub in pairs:
                        with ui.row().classes('w-full items-center gap-2'):
                            ui.icon('menu_book', size='16px', color='teal-500')
                            ui.label(name).classes('text-[11px] font-semibold text-slate-700 flex-1 truncate')
                            ui.label(Path(epub).name).classes('text-[10px] text-slate-500 truncate max-w-[45%]')
                    if getattr(state, 'spellfix_from_corrections', True):
                        remaining = [n for n in unaligned_books() if n not in matched_names]
                        if remaining:
                            ui.separator()
                            ui.label('Without a source - repaired from learned corrections').classes(
                                'text-[10px] font-bold text-slate-500 uppercase pt-1')
                            for name in remaining:
                                with ui.row().classes('w-full items-center gap-2'):
                                    ui.icon('spellcheck', size='16px', color='amber-500')
                                    ui.label(name).classes('text-[11px] text-slate-600 flex-1 truncate')

                with ui.row().classes('w-full justify-end gap-2 mt-1'):
                    ui.button('Cancel', on_click=cleanup_dialog.close).props('flat').classes('text-xs text-slate-600')
                    ui.button('Clean Up', icon='auto_fix_high', on_click=start_cleanup) \
                        .classes('bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-xs px-4')
                return

            # Running / finished view.
            rows = state.cleanup_progress["rows"]
            total = len(rows)
            finished = sum(1 for r in rows if r["status"] in ("done", "error"))
            running = state.cleanup_progress["phase"] == "running"
            ui.label('Cleaning Transcripts' if running else 'Cleanup Complete') \
                .classes('text-lg font-bold text-slate-800')
            headline = state.cleanup_progress["current"] if running else state.cleanup_progress["summary"]
            ui.label(f'{finished} of {total}  ·  {headline}').classes('text-xs text-slate-500')
            ui.linear_progress(value=(finished / total) if total else 0.0, show_value=False) \
                .classes('w-full')

            icon_for = {
                "pending": ("radio_button_unchecked", "slate-300"),
                "working": ("autorenew", "blue-500"),
                "done": ("check_circle", "emerald-500"),
                "error": ("error", "red-500"),
            }
            with ui.column().classes('w-full gap-1 max-h-72 overflow-y-auto border rounded-lg p-3 bg-slate-50'):
                for row in rows:
                    icon, color = icon_for.get(row["status"], ("circle", "slate-400"))
                    with ui.row().classes('w-full items-start gap-2 flex-nowrap'):
                        ui.icon(icon, size='16px', color=color).classes('mt-0.5')
                        with ui.column().classes('gap-0 flex-1 min-w-0'):
                            ui.label(row["book"]).classes('text-[11px] font-semibold text-slate-700 truncate')
                            if row["detail"]:
                                ui.label(row["detail"]).classes('text-[10px] text-slate-500 truncate')

            with ui.row().classes('w-full justify-end gap-2 mt-1'):
                close_btn = ui.button('Close', icon='done',
                                      on_click=lambda: (cleanup_dialog.close(), panel.refresh())) \
                    .classes('bg-slate-700 hover:bg-slate-800 text-white text-xs px-4')
                if running:
                    close_btn.disable()

        render_dialog()

    # Let the running cleanup task refresh whichever dialog is currently mounted,
    # and bring the progress view back if a run was already in flight.
    state.cleanup_refresh = render_dialog.refresh
    if state.cleanup_progress.get("phase") == "running":
        cleanup_dialog.open()

    @ui.refreshable
    def panel():
        cfg = st.load_source_config(project.name)
        dirs = cfg.get('epub_dirs', [])
        resolved = st.resolve_project_sources(project.name, book_names)
        matched = sum(1 for info in resolved.values() if info['chosen'])
        cleaned = sum(1 for n in book_names if st.is_cleaned(project.name, n))
        matched_names = {n for n, info in resolved.items() if info['chosen']}
        no_source_books = [n for n in unaligned_books() if n not in matched_names]

        with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-3'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('menu_book', size='sm', color='teal-500')
                ui.label('Source Text (EPUB) Cleanup').classes('text-sm font-bold text-slate-800')
                ui.badge(f'{matched}/{len(book_names)} matched', color='teal' if matched else 'slate').classes('text-[10px]')
                if cleaned:
                    ui.badge(f'{cleaned} cleaned', color='emerald').classes('text-[10px]')

            ui.label(
                'Optional. Point this project at a folder of EPUBs to replace the Parakeet transcripts '
                'with the publisher text. Word timings are carried across, so quote timestamps keep '
                'working, and volumes without a source file are simply skipped.'
            ).classes('text-xs text-slate-500 leading-normal')

            with ui.row().classes('w-full items-center gap-2'):
                folder_input = ui.input(placeholder=r'e.g. G:\Books\My Series\ePubs') \
                    .classes('flex-1 bg-white').props('outlined dense')

                def add_folder():
                    val = (folder_input.value or '').strip().strip('"')
                    if not val or not Path(val).is_dir():
                        ui.notify('Folder not found. Paste an existing folder path.', type='negative')
                        return
                    if st.add_epub_dir(project.name, val):
                        folder_input.value = ''
                        ui.notify('Source folder added.', type='positive')
                        panel.refresh()
                    else:
                        ui.notify('That folder is already listed.', type='info')

                ui.button('Add Folder', icon='create_new_folder', on_click=add_folder) \
                    .classes('bg-slate-700 hover:bg-slate-800 text-white text-xs h-9 font-semibold')

            if not dirs:
                with ui.row().classes('w-full items-center gap-2 bg-slate-50 border border-dashed rounded-lg p-3'):
                    ui.icon('info', size='16px', color='slate-400')
                    ui.label('No source folders configured. Add one above to scan for EPUBs.').classes('text-[11px] text-slate-500')

            for directory in dirs:
                with ui.row().classes('w-full items-center gap-2 bg-slate-50 border rounded-lg px-3 py-1'):
                    ui.icon('folder', size='16px', color='amber-500')
                    ui.label(directory).classes('text-[11px] text-slate-600 flex-1 truncate')
                    ui.button(icon='close',
                              on_click=lambda _, d=directory: (st.remove_epub_dir(project.name, d), panel.refresh())) \
                        .props('flat dense round size=sm').classes('text-slate-400')

            if dirs:
                with ui.column().classes('w-full gap-0 mt-1'):
                    for b in books:
                        info = resolved.get(b.name, {})
                        candidates = info.get('candidates', [])
                        chosen = info.get('chosen')
                        with ui.row().classes('w-full items-center gap-2 border-b border-slate-100 py-1'):
                            ui.label(b.name).classes('text-[11px] font-semibold text-slate-700 w-52 truncate')
                            if st.is_cleaned(project.name, b.name):
                                ui.badge('cleaned', color='emerald').classes('text-[9px]')
                            if chosen is None:
                                ui.label('no source found').classes('text-[10px] text-slate-400 italic flex-1')
                            elif len(candidates) > 1:
                                def on_pick(e, book=b.name):
                                    st.set_attachment(project.name, book, e.value)
                                    panel.refresh()
                                ui.select(options={str(p): Path(p).name for p, _ in candidates},
                                          value=str(chosen), on_change=on_pick) \
                                    .classes('flex-1').props('dense outlined')
                                ui.badge(f'{len(candidates)} options', color='amber').classes('text-[9px]')
                            else:
                                ui.label(Path(chosen).name).classes('text-[10px] text-slate-500 flex-1 truncate')
                            if st.can_revert(project.name, b.name):
                                def do_revert(name=b.name):
                                    if st.revert_book(project.name, name):
                                        ui.notify(f'{name}: restored the ASR transcript.', type='info')
                                        panel.refresh()
                                ui.button(icon='undo', on_click=do_revert) \
                                    .props('flat dense round size=sm').classes('text-slate-400')

            with ui.row().classes('w-full justify-between items-center mt-2 pt-2 border-t border-slate-100'):
                ui.label(f'{matched} volume(s) ready to clean.' if matched else 'No volumes matched yet.'
                         ).classes('text-xs font-semibold text-slate-400')

                def open_confirm():
                    if state.cleanup_progress.get("phase") != "running":
                        state.cleanup_progress.update(phase="confirm", current="", rows=[], summary="")
                    # Dialog content renders lazily; open it before refreshing
                    # its body (same order as style_playground.py).
                    cleanup_dialog.open()
                    _refresh_cleanup_progress()

                with ui.row().classes('items-center gap-3'):
                    # Only meaningful once the project has at least one source
                    # (otherwise there are no corrections to learn from).
                    if matched and no_source_books:
                        ui.checkbox(
                            f'Also fix {len(no_source_books)} volume(s) without a source, '
                            'using learned corrections',
                            value=state.spellfix_from_corrections,
                            on_change=lambda e: setattr(state, 'spellfix_from_corrections', e.value),
                        ).classes('text-[11px] text-slate-600').props('dense')

                    clean_btn = ui.button('Clean Up Transcripts', icon='auto_fix_high', on_click=open_confirm) \
                        .classes('bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-xs px-5')
                    if not matched or state.cleanup_progress.get("phase") == "running":
                        clean_btn.disable()

    panel()


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
            await asyncio.to_thread(rescan_project_database_state, project.id, True)
            
            render_active_panel.refresh()
            if hasattr(state, 'active_header_refresh') and state.active_header_refresh:
                state.active_header_refresh()
            if hasattr(state, 'action_buttons_refresh') and state.action_buttons_refresh:
                state.action_buttons_refresh()
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

        async def force_sync_project():
            ui.notify("Scanning project files on disk...", type="info")
            await asyncio.to_thread(rescan_project_database_state, project.id, True)
            ui.notify("Sync complete!", type="positive")
            render_active_panel.refresh()
            if hasattr(state, 'active_header_refresh') and state.active_header_refresh:
                state.active_header_refresh()
            if hasattr(state, 'action_buttons_refresh') and state.action_buttons_refresh:
                state.action_buttons_refresh()

        with ui.row().classes('items-center gap-2'):
            ui.button(
                'Open Folder', 
                icon='folder_open', 
                on_click=open_project_folder
            ).props('flat dense').classes('text-xs text-slate-600')
            ui.button(
                'Sync with Disk',
                icon='sync',
                on_click=force_sync_project
            ).props('flat dense').classes('text-xs text-blue-600 hover:text-blue-800')
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
    
    # Refreshable Tab Router - Only renders the selected tab panel to achieve instant loading performance!
    @ui.refreshable
    def render_active_panel():
        active = state.active_project_tab
        if active == 'Dashboard':
            with ui.column().classes('w-full gap-4'):
                global render_dynamic_step_dashboard
                @ui.refreshable
                def render_dynamic_step_dashboard_local():
                    status = state.project_status
                    if status in ("Imported", "Transcribing"):
                        render_transcription_step_view(project, books, start_transcribe_cb, stop_transcribe_cb)
                    elif status in ("Transcribed", "Generating Prompts"):
                        render_prompt_gen_step_view(project, books, start_prompt_gen_cb, stop_transcribe_cb)
                    else:
                        render_image_gen_step_view(project, books, start_image_gen_cb, stop_transcribe_cb)
                        
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

                def safe_refresh():
                    try:
                        render_dynamic_step_dashboard.refresh()
                    except Exception:
                        pass
                    try:
                        render_conditional_feeds.refresh()
                    except Exception:
                        pass
                    try:
                        render_conditional_prompt_feed.refresh()
                    except Exception:
                        pass

                state.action_buttons_refresh = safe_refresh
                            
        elif active == 'Prompt-Gen Playground':
            from ui.pages.project.prompt_lab import render_prompt_lab
            render_prompt_lab(project, books)

        elif active == 'Characters':
            from ui.pages.project.characters_tab import render_characters_tab
            render_characters_tab(project, books)

        elif active == 'Source Text':
            render_source_text_tab(project, books)

        elif active == 'Style & Workflows':
            render_style_playground_tab(project, save_project_settings_cb)

        elif active == 'OIS Packager':
            from ui.pages.project.packager_playground import PackagerPlayground
            if not hasattr(state, 'packager_playground') or state.packager_playground.project_id != project.id:
                state.packager_playground = PackagerPlayground(project.id, project.name)
            state.packager_playground.render(project, books)

    def handle_tab_change(e):
        state.active_project_tab = e.value
        if e.value == 'Style & Workflows':
            from ui.pages.project.style_playground import sync_style_playground_state
            sync_style_playground_state(project.name)
        render_active_panel.refresh()

    with ui.tabs(on_change=handle_tab_change).classes('w-full border-b') as project_tabs:
        tab_dash = ui.tab('Dashboard', icon='dashboard')
        tab_play = ui.tab('Prompt-Gen Playground', icon='science')
        tab_char = ui.tab('Characters', icon='face')
        tab_src = ui.tab('Source Text', icon='menu_book')
        tab_style = ui.tab('Style & Workflows', icon='brush')
        tab_pack = ui.tab('OIS Packager', icon='inventory_2')
        
    project_tabs.bind_value(state, 'active_project_tab')
        
    # Render active panel container
    with ui.column().classes('w-full mt-4'):
        render_active_panel()
