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


def backup_and_cleanup_files(project_id: int, target_status: str):
    """Safely renames directories and csv/txt files to prevent loss and reset pipeline states on disk."""
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
    
    state.add_console_log(f"[Rollback-Engine] Initiating file archiver for project: {project_name}")
    
    for book in books:
        book_dir = output_base / project_name / book.name
        if not book_dir.exists():
            state.add_console_log(f"[Rollback-Engine] Directory not found for book: {book.name}")
            continue
            
        state.add_console_log(f"[Rollback-Engine] Archiving files in book directory: {book.name}")
            
        if target_status == "Imported":
            transcript_path = book_dir / "transcript.txt"
            if transcript_path.exists():
                new_transcript = book_dir / f"transcript_backup_{timestamp}.txt"
                try:
                    transcript_path.rename(new_transcript)
                    state.add_console_log(f"[Rollback-Engine] Archived transcript to: {new_transcript.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving transcript.txt: {str(e)}")
                    
            prompts_path = book_dir / "prompts.csv"
            if prompts_path.exists():
                new_prompts = book_dir / f"prompts_backup_{timestamp}.csv"
                try:
                    prompts_path.rename(new_prompts)
                    state.add_console_log(f"[Rollback-Engine] Archived prompts to: {new_prompts.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving prompts.csv: {str(e)}")
                    
            images_dir = book_dir / "images"
            if images_dir.exists() and images_dir.is_dir():
                new_images = book_dir / f"images_backup_{timestamp}"
                try:
                    images_dir.rename(new_images)
                    state.add_console_log(f"[Rollback-Engine] Archived images folder to: {new_images.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving images folder: {str(e)}")
                    
        elif target_status == "Transcribed":
            prompts_path = book_dir / "prompts.csv"
            if prompts_path.exists():
                new_prompts = book_dir / f"prompts_backup_{timestamp}.csv"
                try:
                    prompts_path.rename(new_prompts)
                    state.add_console_log(f"[Rollback-Engine] Archived prompts to: {new_prompts.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving prompts.csv: {str(e)}")
                    
            images_dir = book_dir / "images"
            if images_dir.exists() and images_dir.is_dir():
                new_images = book_dir / f"images_backup_{timestamp}"
                try:
                    images_dir.rename(new_images)
                    state.add_console_log(f"[Rollback-Engine] Archived images folder to: {new_images.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving images folder: {str(e)}")
                    
        elif target_status == "Prompts Created":
            images_dir = book_dir / "images"
            if images_dir.exists() and images_dir.is_dir():
                new_images = book_dir / f"images_backup_{timestamp}"
                try:
                    images_dir.rename(new_images)
                    state.add_console_log(f"[Rollback-Engine] Archived images folder to: {new_images.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving images folder: {str(e)}")
                    
    state.add_console_log("[Rollback-Engine] State rollback file archival step complete.")


def rollback_project_status(project_id: int, target_status: str, refresh_callback) -> None:
    """Safely shifts project and book statuses backwards to allow pipeline step re-runs."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return
        project.status = target_status
        session.add(project)
        
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        for b in books:
            b.status = target_status
            if target_status in ("Imported", "Transcribed", "Prompts Created"):
                b.progress = 0.0
            session.add(b)
            
        session.commit()
        
    state.project_status = target_status
    ui.notify(f"Project rolled back to: {target_status}", type="warning")
    refresh_callback()
    
    if hasattr(state, 'active_header_refresh') and state.active_header_refresh:
        state.active_header_refresh()


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


def render_prompt_gen_step_view(project, books, start_prompt_gen_cb, stop_transcribe_cb, trigger_rollback_cb):
    available_templates = list_stored_templates()
    
    with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-3'):
        with ui.row().classes('w-full justify-between items-center'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('psychology', size='sm', color='purple-500')
                ui.label('Phase 2: LLM Prompt Generation & Extraction').classes('text-sm font-bold text-slate-800')
            
            ui.button(
                'Rollback to Transcription',
                icon='history',
                on_click=lambda: trigger_rollback_cb("Imported")
            ).classes('text-[10px] text-slate-500 hover:text-red-600').props('flat dense')
            
        ui.label('Transcription passages are processed through a local LLM to extract key quotes and generate visual rendering prompts.').classes('text-xs text-slate-500')
        
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
                ui.button(
                    'Generate Prompts', 
                    icon='bolt', 
                    on_click=lambda: start_prompt_gen_cb(project.id) if start_prompt_gen_cb else None
                ).classes('bg-purple-600 hover:bg-purple-700 text-white font-bold text-xs px-5')


def render_image_gen_step_view(project, books, start_image_gen_cb, stop_transcribe_cb, trigger_rollback_cb):
    with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-3'):
        with ui.row().classes('w-full justify-between items-center'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('image', size='sm', color='amber-500')
                ui.label('Phase 3: ComfyUI Image Generation').classes('text-sm font-bold text-slate-800')
            
            ui.button(
                'Rollback to Prompt Gen',
                icon='history',
                on_click=lambda: trigger_rollback_cb("Transcribed")
            ).classes('text-[10px] text-slate-500 hover:text-red-600').props('flat dense')
            
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


def render_completed_step_view(project, books, trigger_rollback_cb):
    with ui.card().classes('w-full border p-5 shadow-sm bg-emerald-50/10 gap-3'):
        with ui.row().classes('w-full justify-between items-center'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('task_alt', size='sm', color='emerald-500')
                ui.label('Automatic Execution Phases Completed!').classes('text-sm font-bold text-slate-800')
            
            ui.button(
                'Rollback to Image Gen',
                icon='history',
                on_click=lambda: trigger_rollback_cb("Prompts Created")
            ).classes('text-[10px] text-slate-500 hover:text-red-600').props('flat dense')
            
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
            ui.image().classes('w-full rounded-lg max-h-[75vh] object-contain cursor-zoom-out').bind_source_from(state, 'preview_image_src').on('click', global_preview_dialog.close)
            with ui.row().classes('w-full justify-end mt-3'):
                ui.button('Close', on_click=global_preview_dialog.close).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs')
                
    state.global_preview_dialog = global_preview_dialog

    # Reusable rollback confirmation dialog
    with ui.dialog() as rollback_dialog, ui.card().classes('w-full max-w-md p-6 rounded-xl gap-4'):
        ui.label('Confirm State Rollback').classes('text-lg font-bold text-slate-800')
        warning_msg = ui.label().classes('text-xs text-slate-600 leading-relaxed')
        backup_details = ui.markdown().classes('text-[10px] text-slate-500 bg-slate-50 p-2.5 rounded border border-slate-200 font-mono leading-normal w-full')
        
        rollback_target = {"status": "Imported"}
        
        async def confirm_action():
            target = rollback_target["status"]
            ui.notify("Archiving and renaming on-disk folders...", type="info")
            await asyncio.to_thread(backup_and_cleanup_files, project.id, target)
            rollback_project_status(project.id, target, render_dynamic_step_dashboard.refresh)
            rollback_dialog.close()
            
        with ui.row().classes('w-full justify-end gap-3 mt-2'):
            ui.button('Cancel', on_click=rollback_dialog.close).props('flat color=slate').classes('text-xs font-semibold')
            ui.button('Confirm & Rollback', on_click=confirm_action, color='red').classes('text-xs font-bold text-white')

    def trigger_rollback_prompt(target_status: str):
        rollback_target["status"] = target_status
        if target_status == "Imported":
            warning_msg.set_text("You are rolling back to the Transcription phase. This will archive active transcripts, prompt files, and rendered images so transcription can be executed fresh.")
            backup_details.set_content(
                "**Action items:**\n"
                "- Rename `transcript.txt` ➔ `transcript_backup_*.txt`\n"
                "- Rename `prompts.csv` ➔ `prompts_backup_*.csv`\n"
                "- Rename `images/` ➔ `images_backup_*`"
            )
        elif target_status == "Transcribed":
            warning_msg.set_text("You are rolling back to the Prompt Generation phase. This will keep transcripts intact, but archive your active prompt list and rendered images so scene prompts can be generated fresh.")
            backup_details.set_content(
                "**Action items:**\n"
                "- Rename `prompts.csv` ➔ `prompts_backup_*.csv`\n"
                "- Rename `images/` ➔ `images_backup_*`"
            )
        elif target_status == "Prompts Created":
            warning_msg.set_text("You are rolling back to the Image Generation phase. This will keep transcripts and prompt files intact, but archive your rendered images folder so you can clean render the entire visual list.")
            backup_details.set_content(
                "**Action items:**\n"
                "- Rename `images/` ➔ `images_backup_*`"
            )
        rollback_dialog.open()

    # Workspace Navigation Layout with Folder Shortcut
    with ui.row().classes('w-full justify-between items-center mb-1'):
        with ui.row().classes('items-center gap-3'):
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

            ui.button(
                'Open Folder', 
                icon='folder_open', 
                on_click=open_project_folder
            ).props('flat dense').classes('text-xs text-slate-600')
    
    with ui.tabs().classes('w-full border-b') as project_tabs:
        tab_dash = ui.tab('Dashboard', icon='dashboard')
        tab_style = ui.tab('Style & Workflows', icon='brush')
        tab_play = ui.tab('Prompt-Gen Playground', icon='science')
        
    project_tabs.bind_value(state, 'active_project_tab')
        
    with ui.tab_panels(project_tabs, value=state.active_project_tab).classes('w-full bg-transparent p-0'):
        with ui.tab_panel(tab_dash):
            with ui.column().classes('w-full gap-4'):
                @ui.refreshable
                def render_dynamic_step_dashboard():
                    status = state.project_status
                    if status in ("Imported", "Transcribing"):
                        render_transcription_step_view(project, books, start_transcribe_cb, stop_transcribe_cb)
                    elif status in ("Transcribed", "Generating Prompts"):
                        render_prompt_gen_step_view(project, books, start_prompt_gen_cb, stop_transcribe_cb, trigger_rollback_prompt)
                    elif status in ("Prompts Created", "Rendering Images"):
                        render_image_gen_step_view(project, books, start_image_gen_cb, stop_transcribe_cb, trigger_rollback_prompt)
                    else:
                        render_completed_step_view(project, books, trigger_rollback_prompt)
                        
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