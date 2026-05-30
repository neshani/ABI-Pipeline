import sys
import os
import asyncio
import subprocess
import random
import re
import csv
import json
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import io
from typing import Optional, List, Dict, Any

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

# --- Project-Level Settings Persistence (FaST Engine) ---

def get_project_settings_path(project_name: str) -> Path:
    """Returns the path to the project's persistent settings file on disk."""
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    project_dir = base_output_dir / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir / "project_settings.json"


def save_project_settings_to_disk(project_id: int) -> None:
    """Serializes the active state configuration into project_settings.json on disk."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return
        project_name = project.name

    settings_path = get_project_settings_path(project_name)
    data = {
        "active_template": state.playground_selected_template,
        "active_style_preset": state.style_selected_preset,
        "active_workflow": state.style_selected_workflow,
        "style_prompt_prefix": state.style_prompt_prefix,
        "style_negative_prompt": state.style_negative_prompt,
        "style_use_random_image_seed": state.style_use_random_image_seed,
        "style_image_seed": state.style_image_seed,
        "workflow_overrides": state.style_workflow_overrides
    }
    
    try:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        state.add_console_log(f"[FaST-Engine] Saved project configuration to disk: {settings_path.name}")
    except Exception as e:
        state.add_console_log(f"[FaST-Engine] Error saving project settings: {str(e)}")


def load_project_settings_from_disk(project_id: int) -> None:
    """Deserializes project_settings.json and restores active configurations to state bindings."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return
        project_name = project.name

    settings_path = get_project_settings_path(project_name)
    if not settings_path.exists():
        # Fallback to default state values if no custom settings exist yet
        state.playground_selected_template = "default"
        state.style_selected_preset = "default"
        state.style_selected_workflow = ""
        state.style_prompt_prefix = "ArsMJStyle, 1890s Victorian illustration, detailed pen and ink with soft watercolor wash, Sidney Paget style. "
        state.style_negative_prompt = "blurry, bad quality, text, watermark, photorealistic, photography"
        state.style_use_random_image_seed = True
        state.style_image_seed = 42
        state.style_workflow_overrides = {}
        state.style_discovered_params = {}
        return

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        state.playground_selected_template = data.get("active_template", "default")
        state.style_selected_preset = data.get("active_style_preset", "default")
        state.style_selected_workflow = data.get("active_workflow", "")
        state.style_prompt_prefix = data.get("style_prompt_prefix", "")
        state.style_negative_prompt = data.get("style_negative_prompt", "")
        state.style_use_random_image_seed = data.get("style_use_random_image_seed", True)
        state.style_image_seed = data.get("style_image_seed", 42)
        state.style_workflow_overrides = data.get("workflow_overrides", {})
        
        # Re-analyze active workflow parameters to repopulate active sliders
        if state.style_selected_workflow:
            from ui.pages.project_workspace import handle_style_workflow_change
            handle_style_workflow_change(state.style_selected_workflow, clear_overrides=False)

        state.add_console_log(f"[FaST-Engine] Restored project configurations from: {settings_path.name}")
    except Exception as e:
        state.add_console_log(f"[FaST-Engine] Error loading project settings: {str(e)}")

# --- Navigation State Control Handlers ---
def select_project(project_id: int):
    """Sets the active project in memory and restores its saved settings from disk."""
    state.active_project_id = project_id
    state.active_book_id = None
    state.active_project_tab = 'Dashboard'
    state.active_log_widget = None  # Clear log references on panel change
    
    # Fast load settings from project folder
    load_project_settings_from_disk(project_id)
    
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
    """Interrupts active pipeline subprocesses and state tasks gracefully."""
    if state.project_status == "Transcribing":
        cancel_project_transcription(project_id)
        ui.notify("Stopping transcription process...", type="warning")
        state.project_status = "Imported"
    elif state.project_status == "Generating Prompts":
        from services.prompt_engine import cancel_prompt_generation
        cancel_prompt_generation(project_id)
        ui.notify("Stopping prompt generation process...", type="warning")
        state.project_status = "Transcribed"
    elif state.project_status == "Rendering Images":
        state.cancel_image_gen_flag = True
        ui.notify("Stopping image generation process...", type="warning")
        state.project_status = "Prompts Created"
    
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


# --- Image Generation Pipeline & Metadata Baker Utilities ---

def make_slug(prompt: str) -> str:
    """Creates a clean descriptive lowercase slug from the first few words of a prompt."""
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', prompt).lower()
    words = cleaned.split()[:4]
    return "_".join(words)


def save_image_with_metadata(img_bytes: bytes, output_path: Path, quote: str):
    """Uses Pillow to write the target quote text into description metadata chunk of the PNG."""
    image = Image.open(io.BytesIO(img_bytes))
    metadata = PngInfo()
    metadata.add_text("Description", quote)
    metadata.add_text("TargetQuote", quote)
    image.save(output_path, "PNG", pnginfo=metadata)


def find_prompts_csv(project_name: str, book_name: str) -> Optional[Path]:
    """Finds the prompts.csv file across standard folder structures."""
    csv_paths = [
        Path(f"./output/{project_name}/{book_name}/prompts.csv"),
        Path(f"./output/{project_name}/{book_name}_prompts.csv"),
        Path(f"./output/{project_name}/{book_name}/{book_name}_prompts.csv"),
        Path(f"./output/{project_name}/{project_name}_prompts.csv"),
        Path(f"./output/{project_name}_prompts.csv")
    ]
    for path in csv_paths:
        if path.exists():
            return path
    return None


def read_prompts_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Reads prompts CSV separated by pipes safely and standardizes headers."""
    rows = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        # Standardize column headers to lowercase
        reader.fieldnames = [name.lower().strip() if name else "" for name in reader.fieldnames]
        for row in reader:
            rows.append({k: v.strip() if v else "" for k, v in row.items()})
    return rows


async def async_run_project_image_gen_logic(project_id: int):
    """Background rendering processor using the selected style settings & ComfyUI API."""
    comfy_url = get_setting("comfy_url", "127.0.0.1:8188")
    if "http" in comfy_url:
        comfy_url = comfy_url.replace("http://", "").replace("https://", "").strip("/")
        
    if not state.style_selected_workflow:
        state.add_console_log("[Image-Gen] Error: No ComfyUI base workflow selected. Choose one in the Style tab.")
        return

    wf_path = Path("./workflows") / state.style_selected_workflow
    if not wf_path.exists():
        wf_path = Path("./Comfy_Workflows") / state.style_selected_workflow
        
    if not wf_path.exists():
        state.add_console_log(f"[Image-Gen] Error: Workflow '{state.style_selected_workflow}' not found.")
        return
        
    try:
        with open(wf_path, "r") as f:
            workflow_json = json.load(f)
    except Exception as e:
        state.add_console_log(f"[Image-Gen] Error parsing workflow JSON: {str(e)}")
        return

    # Extract clean, independent primitive data to avoid SQLAlchemy detachment issues
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return
        project_name = project.name
        
        books_db = session.exec(select(Book).where(Book.project_id == project_id)).all()
        books_info = []
        for b in books_db:
            b.status = "Rendering Images"
            session.add(b)
            books_info.append({"id": b.id, "name": b.name})
        session.commit()

    from services.comfy_client import ComfyClient
    client = ComfyClient(comfy_url)

    state.add_console_log(f"[Image-Gen] Starting generation for project '{project_name}'...")
    state.add_console_log(f"[Image-Gen] Target ComfyUI API Address: {comfy_url}")

    for b_info in books_info:
        book_id = b_info["id"]
        book_name = b_info["name"]

        if state.cancel_image_gen_flag:
            state.add_console_log("[Image-Gen] Image generation cancelled by user.")
            break
            
        csv_path = find_prompts_csv(project_name, book_name)
        if not csv_path:
            state.add_console_log(f"[Image-Gen] Warning: No prompts.csv found for volume '{book_name}'. Skipping.")
            continue
            
        state.add_console_log(f"[Image-Gen] Processing volume: {book_name}")
        
        try:
            rows = read_prompts_from_csv(csv_path)
        except Exception as e:
            state.add_console_log(f"[Image-Gen] Error reading prompts for {book_name}: {str(e)}")
            continue

        # Filter valid rows
        valid_rows = []
        for r in rows:
            p_text = r.get("prompt", "").strip()
            if p_text and p_text.lower() != "none" and p_text.lower() != "refusal":
                valid_rows.append(r)

        total_prompts = len(valid_rows)
        if total_prompts == 0:
            state.add_console_log(f"[Image-Gen] No valid prompt entries found in {csv_path.name}.")
            with Session(engine) as session:
                db_book = session.get(Book, book_id)
                if db_book:
                    db_book.status = "Images Created"
                    db_book.progress = 1.0
                    session.add(db_book)
                    session.commit()
            continue

        state.add_console_log(f"[Image-Gen] Discovered {total_prompts} scenes to process.")
        
        # Nested target directory initialization
        parent_dir = Path(f"./output/{project_name}/{book_name}")
        out_dir = parent_dir / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        completed_prompts = 0
        
        for idx, row in enumerate(valid_rows):
            if state.cancel_image_gen_flag:
                break
                
            prompt_text = row.get("prompt", "").strip()
            quote_text = row.get("quote", "").strip()
            chapter_str = row.get("chapter", "1")
            scene_str = row.get("scene", str(idx + 1))
            
            try:
                chapter = int(float(chapter_str))
            except ValueError:
                chapter = 1
            try:
                scene = int(float(scene_str))
            except ValueError:
                scene = idx + 1

            # Check both folders to keep backward compatibility and maintain resumability
            existing_files = list(out_dir.glob(f"{chapter:02d}_{scene:02d}_*.png")) or list(parent_dir.glob(f"{chapter:02d}_{scene:02d}_*.png"))
            if not existing_files:
                existing_files = list(out_dir.glob(f"{chapter:02d}_{scene:02d}.png")) or list(parent_dir.glob(f"{chapter:02d}_{scene:02d}.png"))

            if existing_files:
                state.add_console_log(f"[Image-Gen] Resume Skip: Ch {chapter}, Scene {scene} already rendered.")
                completed_prompts += 1
                
                progress_val = completed_prompts / total_prompts
                with Session(engine) as session:
                    db_book = session.get(Book, book_id)
                    if db_book:
                        db_book.progress = progress_val
                        session.add(db_book)
                        session.commit()
                continue

            # Determine seed
            if state.style_use_random_image_seed:
                seed = random.randint(1, 4294967294)
            else:
                seed = state.style_image_seed

            state.add_console_log(f"[Image-Gen] Rendering Ch {chapter}, Scene {scene} with seed {seed}...")
            
            # Execute synchronous workflow API block inside background worker thread
            def render_block():
                return client.generate_image_sync(
                    workflow_json=workflow_json,
                    prompt_text=prompt_text,
                    neg_prompt_text=state.style_negative_prompt,
                    seed=seed,
                    overrides=state.style_workflow_overrides,
                    prefix=state.style_prompt_prefix
                )

            img_bytes, logs = await asyncio.to_thread(render_block)
            
            for log_line in logs.split("\n"):
                if log_line.strip():
                    state.add_console_log(log_line)

            if img_bytes:
                slug = make_slug(prompt_text)
                target_filename = f"{chapter:02d}_{scene:02d}_{slug}.png"
                target_path = out_dir / target_filename
                
                try:
                    def save_and_bake():
                        save_image_with_metadata(img_bytes, target_path, quote_text)
                    await asyncio.to_thread(save_and_bake)
                    state.add_console_log(f"[Image-Gen] Saved and baked metadata into: {target_filename}")
                    
                    # Convert to base64 data string and inject directly to real-time feed
                    import base64
                    encoded = base64.b64encode(img_bytes).decode("utf-8")
                    base64_str = f"data:image/png;base64,{encoded}"
                    
                    state.recent_rendered_images.append({
                        "filename": target_filename,
                        "base64": base64_str,
                        "chapter": chapter,
                        "scene": scene,
                        "quote": quote_text,
                        "prompt": prompt_text
                    })
                    if len(state.recent_rendered_images) > 5:
                        state.recent_rendered_images.pop(0)
                        
                    # Live trigger UI frame refresh
                    if state.recent_images_refresh:
                        try:
                            state.recent_images_refresh()
                        except Exception:
                            pass
                except Exception as save_err:
                    state.add_console_log(f"[Image-Gen] Error saving image file: {str(save_err)}")
            else:
                state.add_console_log(f"[Image-Gen] Failed to retrieve image for Ch {chapter}, Scene {scene}.")

            completed_prompts += 1
            progress_val = completed_prompts / total_prompts
            
            with Session(engine) as session:
                db_book = session.get(Book, book_id)
                if db_book:
                    db_book.progress = progress_val
                    session.add(db_book)
                    session.commit()

        if not state.cancel_image_gen_flag:
            with Session(engine) as session:
                db_book = session.get(Book, book_id)
                if db_book:
                    db_book.status = "Images Created"
                    db_book.progress = 1.0
                    session.add(db_book)
                    session.commit()


async def run_project_image_gen(project_id: int):
    """Wrapper task handling state cleanup, database transitions, and step updating."""
    try:
        await async_run_project_image_gen_logic(project_id)
    except Exception as e:
        state.add_console_log(f"[Image-Gen] Fatal error: {str(e)}")
    finally:
        state.image_gen_active = False
        
        # Determine ending status based on active cancellation flags
        with Session(engine) as session:
            project = session.get(Project, project_id)
            if project:
                if state.cancel_image_gen_flag:
                    project.status = "Prompts Created"
                    session.add(project)
                    
                    books = session.exec(select(Book).where(Book.project_id == project_id)).all()
                    for b in books:
                        if b.status == "Rendering Images":
                            b.status = "Prompts Created"
                            session.add(b)
                else:
                    if project.status == "Rendering Images":
                        project.status = "Images Created"
                        session.add(project)
                        
                        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
                        for b in books:
                            if b.status == "Rendering Images":
                                b.status = "Images Created"
                                session.add(b)
                session.commit()
                
                state.project_status = project.status
        
        # Clear the cancellation flag
        state.cancel_image_gen_flag = False
        
        state.add_console_log("[Image-Gen] Rendering task finished.")
        if hasattr(state, 'action_buttons_refresh'):
            state.action_buttons_refresh()
        from ui.pages.project_workspace import render_stepper
        render_stepper.refresh(state.project_status)


def start_image_generation(project_id: int):
    """Launches the asynchronous, interruptible, and resumable image generation process."""
    if state.image_gen_active:
        ui.notify("An image generation task is already active.", type="warning")
        return

    # Auto-save current configurations to disk before starting
    save_project_settings_to_disk(project_id)

    state.image_gen_active = True
    state.cancel_image_gen_flag = False
    state.project_status = "Rendering Images"
    
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project:
            project.status = "Rendering Images"
            session.add(project)
            
            books = session.exec(select(Book).where(Book.project_id == project_id)).all()
            for b in books:
                b.status = "Rendering Images"
                session.add(b)
            session.commit()
    
    if hasattr(state, 'action_buttons_refresh'):
        state.action_buttons_refresh()
    from ui.pages.project_workspace import render_stepper
    render_stepper.refresh("Rendering Images")

    asyncio.create_task(run_project_image_gen(project_id))
    ui.notify("Background image rendering sequences initiated!", type="positive")


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
                    start_image_generation,
                    save_project_settings_to_disk  # Pass callback here
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