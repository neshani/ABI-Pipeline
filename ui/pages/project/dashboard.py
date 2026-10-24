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


# --- stage-scoped re-run ----------------------------------------------------
# The dialog is built in render_project_tabs (which is not a refresh path) but
# opened from the analysis card, which is refreshable. Module state is the bridge:
# a closure would capture the first render's dialog, and a dialog built inside the
# card would be duplicated on every refresh.
_rerun_dialog = None
_rerun_body = None
_rerun_checks: dict = {}
_rerun_book_id = None
_rerun_start_cb = None
_rerun_project_id = None

STAGE_ROWS = (
    ("gather", "Quotes",
     "Finds and attributes the descriptions. Needs a current scene map."),
    ("extract", "Traits and cast",
     "Re-sorts the existing quotes and rewrites the timeline events."),
)


def open_stage_rerun(book_id: int, book_dir) -> None:
    """Populate the stage picker for one book and open it."""
    global _rerun_book_id, _rerun_checks
    from services import character_analysis

    if _rerun_dialog is None or _rerun_body is None:
        # The dialog is built at page load, before this card renders, so this is
        # defensive only: return rather than raise from an event handler.
        return
    status = character_analysis.analysis_stage_status(book_dir)
    _rerun_book_id = book_id
    _rerun_checks = {}
    _rerun_body.clear()
    with _rerun_body:
        for stage, label, hint in STAGE_ROWS:
            info = status.get(stage, {})
            record = info.get("record") or {}
            partial = int(info.get("partial") or 0)
            if info.get("current"):
                detail = "up to date"
            elif partial:
                detail = f"{partial} unit(s) saved - resumes where it stopped"
            elif not record and info.get("artifact"):
                detail = "exists, but built before stage records"
            elif not record:
                detail = "never run"
            else:
                detail = "stale: " + ", ".join(info.get("reasons") or ["changed"])
            box = ui.checkbox(label, value=not info.get("current"))
            box.classes('text-xs').tooltip(hint)
            _rerun_checks[stage] = box
            ui.label(detail).classes('text-[9px] text-slate-500 pl-6 -mt-1')
    _rerun_dialog.open()


def confirm_stage_rerun() -> None:
    """Launch the selected stages for the book the picker is pointing at."""
    chosen = [stage for stage, box in _rerun_checks.items()
              if getattr(box, "value", False)]
    if not chosen:
        ui.notify("Pick at least one stage to re-run.", type="warning")
        return
    if _rerun_start_cb and _rerun_book_id is not None:
        _rerun_start_cb(_rerun_project_id, [_rerun_book_id], chosen)
    if _rerun_dialog is not None:
        _rerun_dialog.close()


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
        # - "transcripts": Archives transcript.txt, prompts.csv, shot_plan.csv, and images/
        # - "prompts": Archives prompts.csv, shot_plan.csv, and images/
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
                    
        # 3. Archive the shot plan wherever the prompts go. It is LLM output
        # derived from the same transcript and planner prompt, so a regenerated
        # prompts.csv must not inherit the previous run's shot choices. (Only
        # chunk size and the planner prompt digest are checked for reuse, so a
        # re-transcribed book would otherwise keep a plan for the old text.)
        if choice in ("transcripts", "prompts"):
            plan_path = book_dir / "shot_plan.csv"
            if plan_path.exists():
                new_plan = book_dir / f"shot_plan_backup_{timestamp}.csv"
                try:
                    plan_path.rename(new_plan)
                    state.add_console_log(f"[Rollback-Engine] Archived shot plan to: {new_plan.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving shot_plan.csv: {str(e)}")

        # 4. Archive rendered images (All options archive generated visual records)
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


PIPELINE_STAGES = ("prompts",)

_STAGE_TITLE = {
    "prompts": "Prompts",
}

_STAGE_HINT = {
    "prompts": "One prompt per shot, planned and written in one pass. Resumes at the first "
               "shot that has no row yet.",
}


def _fmt_minutes(total: float) -> str:
    """``~1h 40m`` for an estimate. Always approximate, always prefixed."""
    if not total or total <= 0:
        return "-"
    if total < 55:
        return f"~{int(round(total))}m"
    return f"~{int(total // 60)}h {int(round(total % 60)):02d}m"


# Cell presentation. `ok` is the only state that needs no work; everything else
# is unfinished (missing/partial), out of date (stale) or cannot run (blocked).
# The colour carries that, the tooltip carries the reason.
_CELL_STYLE = {
    "ok": ("check_circle", "text-emerald-500"),
    "unrecorded": ("help", "text-amber-500"),
    "stale": ("warning", "text-amber-500"),
    "partial": ("incomplete_circle", "text-amber-500"),
    "missing": ("radio_button_unchecked", "text-slate-300"),
    "blocked": ("block", "text-rose-400"),
    "off": ("remove", "text-slate-300"),
}


def _cell_text(stage: str, row: dict) -> str:
    """The few characters a status cell shows. The reason lives in the tooltip."""
    info = row.get(stage) or {}
    kind = info.get("state", "missing")
    if stage == "prompts" and kind in ("ok", "partial"):
        total = info.get("total") or 0
        return f"{info.get('rows', 0)}/{total}" if total else str(info.get("rows", 0))
    if info.get("short"):
        # A blocked/stale stage with a two-word reason ("map changed", "size
        # changed"): the cell says why, the tooltip says the whole sentence.
        return info["short"]
    if stage == "images":
        done, expected = info.get("done", 0), info.get("expected", 0)
        return f"{done}/{expected}" if expected else str(done)
    return {
        "ok": "ready", "unrecorded": "no record", "stale": "stale",
        "partial": "in progress", "missing": "-", "blocked": "blocked",
        "off": "off",
    }.get(kind, "-")


def render_transcript_approval_banner(project, ordered_books, output_base, refresh):
    """Warn about transcripts nobody has approved, and offer to approve them.

    Kept from the old Phase 2 card: an unapproved transcript may still contain
    boilerplate or a table of contents, and prompt generation would read it as
    narrative.
    """
    unapproved = [b for b in ordered_books
                  if (output_base / project.name / b.name / "transcript.txt").exists()
                  and not (output_base / project.name / b.name / ".transcript_approved").exists()]
    if not unapproved:
        return

    def approve_all():
        count = 0
        for b in unapproved:
            marker = output_base / project.name / b.name / ".transcript_approved"
            try:
                marker.touch()
                count += 1
            except Exception as ex:
                state.add_console_log(f"[Dashboard] Could not approve {b.name}: {ex}")
        if count:
            ui.notify(f"Approved {count} volume transcript(s).", type="positive")
            rescan_project_database_state(project.id)
            refresh()
        else:
            ui.notify("Nothing could be approved.", type="info")

    with ui.row().classes('w-full items-center gap-3 bg-amber-50 p-2 rounded-lg border border-amber-200'):
        ui.icon('warning', color='amber', size='sm')
        with ui.column().classes('gap-0 flex-1 min-w-0'):
            ui.label('Unapproved transcripts').classes('text-[11px] font-bold text-amber-800')
            ui.label(', '.join(b.name for b in unapproved)).classes(
                'text-[10px] text-amber-700 truncate')
        ui.button('Approve All', icon='done_all', on_click=approve_all) \
            .classes('bg-amber-600 hover:bg-amber-700 text-white font-bold text-[10px] px-3 py-1 rounded')


def render_pipeline_table(project, ordered_books, rows_by_id, dirs_by_id,
                          *, start_pipeline_cb, start_analysis_cb, refresh):
    """One row per volume: reading order, side flag, and the four stages.

    This replaces the old status-gated cards, which chose between phases on
    ``Project.status``. A status word cannot represent "15 books, each at its own
    point", and the old cards hid the cast controls exactly when re-running them
    was most useful.
    """
    from services import pipeline_status

    def move_book(book_id: int, delta: int) -> None:
        ids = [b.id for b in ordered_books]
        index = ids.index(book_id)
        target = index + delta
        if target < 0 or target >= len(ids):
            return
        ids[index], ids[target] = ids[target], ids[index]
        # Renumber contiguously: reading order is the axis that drives cross-book
        # description inheritance, and a duplicate or a zero is a silent bug there.
        with Session(engine) as session:
            for position, bid in enumerate(ids, start=1):
                db_book = session.get(Book, bid)
                if db_book:
                    db_book.book_order = position
                    session.add(db_book)
            session.commit()
        refresh()

    def toggle_side(book_id: int, value: bool) -> None:
        with Session(engine) as session:
            db_book = session.get(Book, book_id)
            if db_book:
                db_book.is_side_story = bool(value)
                session.add(db_book)
                session.commit()
        ui.notify("Side-story flag updated.", type="info")

    def toggle_selected(book_id: int, value: bool) -> None:
        selected = set(state.pipeline_selected_books or [])
        if value:
            selected.add(book_id)
        else:
            selected.discard(book_id)
        state.pipeline_selected_books = sorted(selected)
        refresh()

    def toggle_all(value: bool) -> None:
        state.pipeline_selected_books = [b.id for b in ordered_books] if value else []
        refresh()

    all_selected = bool(ordered_books) and all(
        b.id in (state.pipeline_selected_books or []) for b in ordered_books)

    with ui.card().classes('w-full border shadow-sm bg-white gap-0 p-0 overflow-x-auto'):
        with ui.row().classes('w-full items-center gap-2 px-3 py-1.5 bg-slate-50 border-b'):
            ui.checkbox(value=all_selected, on_change=lambda e: toggle_all(bool(e.value))) \
                .props('dense size=sm').classes('w-5').tooltip(
                'Tick volumes to narrow Run to just those; tick none to run the '
                'whole project.')
            ui.label('').classes('w-5')
            ui.label('Volume').classes(
                'flex-1 text-[9px] font-bold uppercase tracking-wider text-slate-400')
            ui.label('Prompts').classes(
                'w-20 text-[9px] font-bold uppercase tracking-wider text-slate-400')
            ui.label('Images').classes(
                'w-20 text-[9px] font-bold uppercase tracking-wider text-slate-400')
            ui.label('').classes('w-16')

        for position, b in enumerate(ordered_books, start=1):
            row = rows_by_id[b.id]
            b_dir = dirs_by_id[b.id]
            with ui.row().classes('w-full items-center gap-2 px-3 py-1 border-b '
                                  'border-slate-100 hover:bg-slate-50/60'):
                ui.checkbox(
                    value=b.id in (state.pipeline_selected_books or []),
                    on_change=lambda e, bid=b.id: toggle_selected(bid, bool(e.value)),
                ).props('dense size=sm').classes('w-5').tooltip(
                    'Include this volume in Run')

                with ui.column().classes('gap-0 w-5'):
                    ui.button(icon='keyboard_arrow_up',
                              on_click=lambda _, bid=b.id: move_book(bid, -1)) \
                        .props('flat dense round size=xs').classes(
                            'text-slate-300 hover:text-slate-700 h-3')
                    ui.button(icon='keyboard_arrow_down',
                              on_click=lambda _, bid=b.id: move_book(bid, 1)) \
                        .props('flat dense round size=xs').classes(
                            'text-slate-300 hover:text-slate-700 h-3')

                with ui.row().classes('flex-1 items-center gap-2 min-w-0'):
                    ui.label(f"{position}.").classes('text-[9px] text-slate-400 w-4 text-right')
                    ui.label(b.name).classes('text-[11px] font-medium text-slate-700 truncate')
                    if getattr(b, "is_side_story", False):
                        ui.label('side').classes(
                            'text-[8px] font-bold uppercase tracking-wide bg-violet-100 '
                            'text-violet-700 px-1.5 py-0.5 rounded').tooltip(
                            'Side story: inherits descriptions from earlier volumes and is '
                            'skipped by later ones. Click to clear.').on(
                            'click', lambda _=None, bid=b.id: toggle_side(bid, False))
                    if row.get("chunker_version") == 1:
                        ui.label('old packer').classes(
                            'text-[8px] font-bold uppercase tracking-wide bg-slate-200 '
                            'text-slate-600 px-1.5 py-0.5 rounded').tooltip(
                            'Generated by the greedy linear packer that shipped before '
                            'the scene map was removed. Its prompts can only be extended by '
                            'that packer until the volume is archived and rebuilt.')

                for stage in ("prompts", "images"):
                    info = row.get(stage) or {}
                    kind = info.get("state", "missing")
                    icon, colour = _CELL_STYLE.get(kind, _CELL_STYLE["missing"])
                    with ui.row().classes('w-20 items-center gap-1').tooltip(
                            info.get("detail") or _STAGE_TITLE.get(stage, stage.title())):
                        ui.icon(icon, size='14px').classes(colour)
                        ui.label(_cell_text(stage, row)).classes(
                            'text-[10px] '
                            + ('text-slate-400' if kind in ('missing', 'off') else 'text-slate-600'))

                with ui.row().classes('w-16 items-center justify-end gap-0'):
                    missing_stages = [
                        stage for stage in PIPELINE_STAGES
                        if pipeline_status.needs_work(row, stage)
                    ]
                    run_btn = ui.button(
                        icon='play_arrow',
                        on_click=lambda _, bid=b.id, stages=missing_stages:
                            start_pipeline_cb(project.id, [bid], stages),
                    ).props('flat dense round size=sm').classes('text-blue-600')
                    run_btn.tooltip(f"Run {', '.join(_STAGE_TITLE[s].lower() for s in missing_stages)}"
                                    if missing_stages else 'This volume is up to date')
                    if not missing_stages:
                        run_btn.disable()

                    with ui.button(icon='more_horiz').props('flat dense round size=sm') \
                            .classes('text-slate-400'):
                        with ui.menu().classes('p-1 w-64 shadow-lg rounded-xl border'):
                            ui.button(
                                'Re-run stages...',
                                icon='replay',
                                on_click=lambda _=None, bid=b.id, bdir=b_dir:
                                    open_stage_rerun(bid, bdir),
                            ).props('flat dense align=left').classes(
                                'text-xs text-slate-700 w-full justify-start').tooltip(
                                'Pick individual stages of the retired cast pass; '
                                'everything else is reused. Needs a scene map, so it '
                                'only applies to volumes built before maps were removed.')
                            if not getattr(b, "is_side_story", False):
                                ui.button(
                                    'Mark as side story',
                                    icon='call_split',
                                    on_click=lambda _=None, bid=b.id: toggle_side(bid, True),
                                ).props('flat dense align=left').classes(
                                    'text-xs text-slate-700 w-full justify-start').tooltip(
                                    'Inherits descriptions from earlier volumes; later '
                                    'volumes skip it.')


def render_run_bar(project, ordered_books, rows_by_id, *, start_pipeline_cb,
                   stop_cb, refresh):
    """The one action: what to run, for which volumes, and roughly how long.

    Scope is the checkbox column and nothing else: ticked volumes narrow the run,
    nothing ticked means the whole project. An earlier version also had a
    "selected volumes" dropdown, which meant ticking a box changed nothing until
    you noticed the second control - the same two-controls-one-decision mistake
    as the old mode select.
    """
    from services import pipeline_status

    running = bool(getattr(state, "pipeline_active", False)
                   or getattr(state, "prompt_gen_active", False)
                   or getattr(state, "analysis_active", False))
    selected_ids = set(state.pipeline_selected_books or [])

    def clear_selection():
        state.pipeline_selected_books = []
        refresh()

    with ui.card().classes('w-full border shadow-sm bg-white p-3 gap-2'):
        with ui.row().classes('w-full items-center gap-3 flex-wrap min-w-0'):
            ui.icon('play_circle', size='sm', color='blue-500')
            ui.label('Run').classes('text-xs font-bold text-slate-700')
            if selected_ids:
                ui.label(f'{len(selected_ids)} of {len(ordered_books)} volumes selected') \
                    .classes('text-[10px] font-semibold text-blue-700')
                ui.button('Clear', icon='close', on_click=lambda: clear_selection()) \
                    .props('flat dense').classes('text-[10px] text-slate-500 -ml-1')
            else:
                ui.label(f'all {len(ordered_books)} volumes').classes(
                    'text-[10px] text-slate-500')

            def toggle_force(value):
                state.pipeline_run_force = bool(value)

            ui.checkbox('Even if current', value=bool(getattr(state, "pipeline_run_force", False)),
                        on_change=lambda e: toggle_force(e.value)) \
                .props('dense').classes('text-[10px] text-slate-600').tooltip(
                'Re-run stages that are already up to date. Off (the default) only '
                'does outstanding work.')

            ui.label().classes('flex-1 min-w-2')
            if running:
                ui.spinner(size='sm', color='blue')
                ui.label('Working - progress is in the console.').classes(
                    'text-[10px] font-semibold text-blue-700 animate-pulse')
                ui.button('Stop', icon='stop', on_click=lambda: stop_cb(project.id)) \
                    .classes('bg-rose-600 text-white text-xs font-bold px-4')

        pool = ([b for b in ordered_books if b.id in selected_ids]
                if selected_ids else list(ordered_books))
        force = bool(getattr(state, "pipeline_run_force", False))

        needed = {}
        for stage in PIPELINE_STAGES:
            # ``needed`` is what the button promises: everything that would do work
            # in this run, for the volumes in scope.
            needed[stage] = [b for b in pool
                             if force or pipeline_status.needs_work(rows_by_id[b.id], stage)]

        wanted = [s for s in (state.pipeline_run_stages or list(PIPELINE_STAGES))
                  if s in needed]

        def toggle_stage(stage: str, value: bool) -> None:
            chosen = set(state.pipeline_run_stages or PIPELINE_STAGES)
            if value:
                chosen.add(stage)
            else:
                chosen.discard(stage)
            state.pipeline_run_stages = [s for s in PIPELINE_STAGES if s in chosen]
            refresh()

        with ui.row().classes('w-full items-center gap-3 flex-wrap min-w-0'):
            for stage in PIPELINE_STAGES:
                group = needed.get(stage, [])
                minutes = sum(pipeline_status.estimate_minutes(
                    stage, rows_by_id[b.id]["words"]) for b in group)
                box = ui.checkbox(_STAGE_TITLE[stage], value=(stage in wanted),
                                  on_change=lambda e, s=stage: toggle_stage(s, bool(e.value)))
                box.classes('text-[11px] text-slate-700').props('dense')
                box.tooltip(_STAGE_HINT[stage])
                if group:
                    ui.label(f"{len(group)} · {_fmt_minutes(minutes)}").classes(
                        'text-[9px] -ml-2 text-slate-500')
                else:
                    ui.label("up to date").classes('text-[9px] -ml-2 text-emerald-600')

            ui.label().classes('flex-1 min-w-2')

            total = sum(pipeline_status.estimate_minutes(
                stage, rows_by_id[b.id]["words"])
                for stage in wanted for b in needed.get(stage, []))
            volume_count = len({b.id for stage in wanted
                                for b in needed.get(stage, [])})
            resuming = any(
                (rows_by_id[b.id].get(stage) or {}).get("state") == "partial"
                for stage in wanted for b in needed.get(stage, []))

            def start_chosen():
                if not volume_count:
                    ui.notify("Nothing in scope needs running.", type="info")
                    return
                # The selection *is* the scope, so it is always passed explicitly
                # when non-empty; None means "the whole project".
                scope_ids = [b.id for b in pool] if selected_ids else None
                start_pipeline_cb(project.id, scope_ids, wanted)

            label = ('Resume' if resuming else 'Run')
            if volume_count:
                label += f" · {volume_count} volume{'s' if volume_count != 1 else ''} · {_fmt_minutes(total)}"
            button = ui.button(label, icon='play_arrow', on_click=start_chosen)
            button.classes('bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold px-3')
            if running or not wanted or volume_count == 0:
                button.disable()

        if selected_ids and not volume_count:
            ui.label('The selected volumes have nothing outstanding. Tick "Even if '
                     'current" to re-run a stage anyway.').classes(
                'text-[9px] text-amber-600')


def render_pipeline_settings(project, ordered_books, refresh):
    """Overrides: the shot planner, the chunk size, and their blast radius."""
    from services import pipeline_manifest
    from services import pipeline_status
    from services import prompt_library as _pl

    shot_mode_options = {
        "auto": "Plan shots (auto)",
        "wide": "All WIDE",
        "character": "All CHARACTER",
        "insert": "All INSERT",
        "off": "Custom single prompt",
    }

    def handle_default_shot_mode(val: str):
        _pl.set_default_shot_mode(val)
        ui.notify(f"Shot Planner: {shot_mode_options.get(val, val)}", type="info")

    def handle_book_shot_mode(book_name: str, val: str):
        if not val:
            _pl.clear_shot_mode(project.name, book_name)
            ui.notify(f"{book_name}: inherits the project default", type="info")
        else:
            _pl.set_shot_mode(project.name, book_name, val)
            ui.notify(f"{book_name}: {shot_mode_options.get(val, val)}", type="info")

    with ui.expansion('Overrides').classes('w-full border rounded bg-white text-xs'):
        with ui.column().classes('w-full p-3 gap-3'):
            with ui.row().classes('w-full items-center gap-3'):
                ui.icon('movie_filter', size='sm', color='slate-400')
                with ui.column().classes('gap-0 flex-1'):
                    ui.label('Shot Planner').classes('text-xs font-bold text-slate-700')
                    ui.label('How each shot chooses its prompt. The prompts themselves are '
                             'edited in the Prompt-Gen Playground.').classes(
                        'text-[9px] text-slate-500')
                ui.select(shot_mode_options, value=_pl.get_default_shot_mode(),
                          on_change=lambda e: handle_default_shot_mode(e.value)) \
                    .classes('w-56 bg-white').props('outlined dense')

            with ui.expansion('Per-volume shot modes').classes(
                    'w-full border rounded bg-slate-50 text-xs').props('dense'):
                for b in ordered_books:
                    override = _pl.get_shot_mode_override(project.name, b.name)
                    with ui.row().classes('w-full items-center gap-3 px-2 py-1'):
                        ui.label(b.name).classes('text-[11px] flex-1 truncate')
                        ui.select({"": "Inherit project default", **shot_mode_options},
                                  value=override or "",
                                  on_change=lambda e, bn=b.name:
                                      handle_book_shot_mode(bn, e.value)) \
                            .classes('w-60').props('dense outlined')

            def handle_chunk_size_change(val: int):
                if not val or val < 50:
                    val = 50
                state.playground_chunk_size = int(val)
                from services.project_settings import save_project_settings_to_disk
                if state.active_project_id:
                    save_project_settings_to_disk(state.active_project_id)
                state._stats_cache.clear()
                pipeline_status.clear_cache()
                if hasattr(state, "stats_refresh_callback") and state.stats_refresh_callback:
                    try:
                        state.stats_refresh_callback()
                    except Exception as ex:
                        state.add_console_log(f"[Dashboard] Error refreshing stats: {ex}")
                refresh()
                ui.notify(f"Words per shot: {val}.", type="info")

            # The blast radius is real (pipeline_manifest.check refuses a mismatch
            # rather than mixing two coordinate systems), so it is stated here
            # instead of being discovered as a red cell later.
            affected = 0
            for b in ordered_books:
                manifest = pipeline_manifest.read(
                    Path(get_setting("output_dir", "./output")).resolve()
                    / project.name / b.name)
                if manifest.get("chunk_words") not in (None, state.playground_chunk_size):
                    affected += 1

            with ui.row().classes('w-full items-center gap-3'):
                ui.icon('wrap_text', size='sm', color='slate-400')
                with ui.column().classes('gap-0 flex-1'):
                    ui.label('Words per shot').classes('text-xs font-bold text-slate-700')
                    detail = ('How much text one image covers. Smaller means more images.')
                    if affected:
                        detail += (f' Changing it invalidates the shot plan and the '
                                   f'prompts for {affected} volume(s).')
                    ui.label(detail).classes('text-[9px] text-slate-500 leading-normal')
                ui.number(value=state.playground_chunk_size, min=50, max=2000, step=50,
                          on_change=lambda e: handle_chunk_size_change(e.value)) \
                    .classes('w-48 bg-white').props('outlined dense suffix="words"')


def render_dashboard_view(project, books, *, start_transcribe_cb, stop_cb,
                          start_analysis_cb, start_pipeline_cb, start_image_gen_cb):
    """The Dashboard tab: transcripts, the volume table, the run bar, overrides."""
    from services import pipeline_status
    from services.character_manager import ensure_book_orders

    ensure_book_orders(project.id)

    output_base = Path(get_setting("output_dir", "./output")).resolve()
    ordered_books = sorted(
        books,
        key=lambda b: (b.book_order if b.book_order is not None else 10 ** 6, b.name))

    def build_rows():
        rows_by_id, dirs_by_id = {}, {}
        for b in ordered_books:
            b_dir = output_base / project.name / b.name
            dirs_by_id[b.id] = b_dir
            rows_by_id[b.id] = pipeline_status.snapshot(
                b_dir,
                chunk_words=state.playground_chunk_size,
                expected_prompts=getattr(b, "total_images", 0) or 0,
            )
        return rows_by_id, dirs_by_id

    missing = [b for b in ordered_books
               if not (output_base / project.name / b.name / "transcript.txt").exists()]
    if missing or state.project_status == "Transcribing":
        render_transcription_step_view(project, ordered_books, start_transcribe_cb, stop_cb)
    else:
        with ui.row().classes('w-full items-center gap-2 px-1'):
            ui.icon('check_circle', size='14px', color='emerald-500')
            ui.label(f'{len(ordered_books)} volumes transcribed.').classes(
                'text-[10px] text-slate-500')

    @ui.refreshable
    def dashboard_body():
        rows_by_id, dirs_by_id = build_rows()

        total_words = sum(rows_by_id[b.id]["words"] for b in ordered_books)
        # Same sum the Run button promises: every stage, for every volume that
        # still needs it.
        outstanding = sum(
            pipeline_status.estimate_minutes(stage, rows_by_id[b.id]["words"])
            for stage in PIPELINE_STAGES
            for b in ordered_books
            if pipeline_status.needs_work(rows_by_id[b.id], stage))

        with ui.row().classes('w-full items-center gap-3 flex-wrap min-w-0'):
            with ui.column().classes('gap-0 flex-1 min-w-0'):
                ui.label(f'{len(ordered_books)} volumes · {total_words:,} words').classes(
                    'text-sm font-bold text-slate-800')
                ui.label(_fmt_minutes(outstanding) + ' of work outstanding'
                         if outstanding else 'Everything is generated.').classes(
                    'text-[10px] ' + ('text-amber-600 font-semibold' if outstanding
                                      else 'text-emerald-600'))

        render_transcript_approval_banner(project, ordered_books, output_base,
                                          dashboard_body.refresh)

        # The action sits above the table: with a long series the table runs past
        # the fold, and a Run button nobody can see is the problem this dashboard
        # is replacing.
        render_run_bar(project, ordered_books, rows_by_id,
                       start_pipeline_cb=start_pipeline_cb, stop_cb=stop_cb,
                       refresh=dashboard_body.refresh)

        render_pipeline_table(project, ordered_books, rows_by_id, dirs_by_id,
                              start_pipeline_cb=start_pipeline_cb,
                              start_analysis_cb=start_analysis_cb,
                              refresh=dashboard_body.refresh)

        render_pipeline_settings(project, ordered_books, dashboard_body.refresh)

    dashboard_body()

    # Phase 3: rendering. Always present - it is a thing to do, not a phase to
    # arrive at, and the resume behaviour lives in the filenames.
    render_image_gen_step_view(project, ordered_books, start_image_gen_cb, stop_cb)


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
                            .on('click', lambda _, img=item['base64'], title=f"Ch {item['chapter']}, Scene {item['shot']}": open_large_image(img, title)):
                        ui.image(item['base64']).classes('w-full h-28 rounded object-cover border')
                        with ui.column().classes('gap-0 mt-1'):
                            ui.label(f"Ch {item['chapter']}, Scene {item['shot']}").classes('text-[10px] font-bold text-slate-700')
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
                            ui.label(f"{item['book']} — Ch {item['chapter']}, Scene {item['shot']}").classes('text-[10px] font-bold text-slate-500 uppercase')
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
    save_project_settings_cb=None,
    start_analysis_cb=None,
    start_pipeline_cb=None
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
    # --- Stage-scoped re-run -------------------------------------------------
    # Separate from the rollback dialog below: rollback archives files to force a
    # rebuild, while this re-runs *named stages* and reuses every artifact that is
    # still current. The two coexist because re-importing a transcript or
    # re-rendering images is not an analysis stage.
    #
    # The button lives in the dashboard table, which runs inside a @ui.refreshable,
    # so the dialog cannot be built there (it would be rebuilt on every refresh).
    # It is built once here and driven through module state.
    global _rerun_dialog, _rerun_body, _rerun_checks, _rerun_start_cb, _rerun_project_id
    _rerun_start_cb = start_analysis_cb
    _rerun_project_id = project.id
    _rerun_checks = {}

    with ui.dialog() as _rerun_dialog, ui.card().classes('w-full max-w-md p-6 rounded-xl gap-3'):
        ui.label('Re-run selected stages').classes('text-lg font-bold text-slate-800')
        ui.label('Only the stages you pick make model calls. Everything else is '
                 'reused from disk.').classes('text-xs text-slate-500 leading-normal')
        _rerun_body = ui.column().classes('w-full gap-1')
        with ui.row().classes('w-full justify-end gap-3 mt-2'):
            ui.button('Cancel', on_click=_rerun_dialog.close)\
                .props('flat color=slate').classes('text-xs font-semibold')
            ui.button('Re-run', on_click=confirm_stage_rerun)\
                .classes('bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold')

    with ui.dialog() as rollback_dialog, ui.card().classes('w-full max-w-md p-6 rounded-xl gap-4'):
        ui.label('Confirm State Rollback').classes('text-lg font-bold text-slate-800')
        ui.label('Select how far back you want to roll this project. Your existing files will be safely archived on disk.').classes('text-xs text-slate-500 leading-normal')
        
        rollback_choice = ui.radio({
            'images': 'Archive Images and regenerate images (Archives images only)',
            'prompts': 'Archive prompts and regenerate prompts (Includes images and the shot plan)',
            'transcripts': 'Archive transcripts and re-transcribe audio (Includes prompts, images and the shot plan)'
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
            with ui.column().classes('w-full gap-4 min-w-0'):
                global render_dynamic_step_dashboard
                @ui.refreshable
                def render_dynamic_step_dashboard_local():
                    render_dashboard_view(
                        project, books,
                        start_transcribe_cb=start_transcribe_cb,
                        stop_cb=stop_transcribe_cb,
                        start_analysis_cb=start_analysis_cb,
                        start_pipeline_cb=start_pipeline_cb or (
                            lambda pid, plan: None),
                        start_image_gen_cb=start_image_gen_cb,
                    )

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
