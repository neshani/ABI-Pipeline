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

from nicegui import ui, app

# --- Configure WebSockets for Large Payloads ---
# This overrides the default 1MB Socket.IO limit to prevent connection drops on large transcript.txt files.
from nicegui import core
core.sio.max_http_buffer_size = 50 * 1024 * 1024  # Raise limit to 50 MB
if hasattr(core.sio, 'eio'):
    core.sio.eio.max_http_buffer_size = 50 * 1024 * 1024
    
from sqlmodel import Session, select
from database.connection import init_db, get_setting, set_setting, engine
from database.models import Project, Book, Chapter
from services.sync_engine import recover_from_temp_workspaces, get_book_stats, get_book_stats_cached
from services.transcription import (
    start_project_transcription, 
    cancel_project_transcription, 
    active_projects
)

from ui.components.settings_modal import SettingsModal

# Modularized states and pages imports
from ui import state
from ui.pages import render_portal_view, render_project_tabs, render_book_tabs, render_lora_contact_sheet, register_main_layout


# --- Cache-Busted On-Disk Volume Statistics Engine ---
# Cache dictionary is now located safely inside state._stats_cache to prevent circular imports


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



# --- Initialize SQLite Database Schema ---
# Running this globally right after imports ensures that the setting tables exist 
# before DEFAULT_SETTINGS and any global database reads are evaluated.
init_db()


# --- Initialize SQLite Database & Recovery Engines (One-time Startup Event) ---

def run_startup_recovery():
    """Clears stuck tasks and recovers workspaces exactly once on startup."""
    # Ensure stuck tasks are cleared
    reset_stuck_transcriptions()

    # Run workspace recovery to restore wiped database entries
    with Session(engine) as session:
        recover_from_temp_workspaces(session)

# Registers the callback so it only executes once in the active child worker process
app.on_startup(run_startup_recovery)

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
    "comfy_args": "--windows-standalone-build",
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
from services.project_settings import (
    get_project_settings_path,
    save_project_settings_to_disk,
    load_project_settings_from_disk
)

# --- Navigation State Control Handlers ---
def select_project(project_id: int):
    """Sets the active project in memory and restores its saved settings from disk."""
    state.active_project_id = project_id
    state.active_book_id = None
    state.active_project_tab = 'Dashboard'
    state.active_log_widget = None  # Clear log references on panel change
    
    # Fast load settings from project folder
    load_project_settings_from_disk(project_id)
    header_controls.refresh()
    main_layout.refresh()


async def async_select_book(book_id: int, client):
    # Capture the current scroll position of the sidebar books list container
    try:
        scroll_pos = await client.run_javascript("document.getElementById('sidebar-books-list')?.scrollTop || 0")
    except Exception:
        scroll_pos = 0
    
    state.active_book_id = book_id
    state.active_book_tab = 'Dashboard'
    state.active_log_widget = None  # Clear log references on panel change
    
    # Instantly scroll the browser window back to the top of the page
    try:
        await client.run_javascript('window.scrollTo(0, 0)')
    except Exception:
        pass
    
    main_layout.refresh()
    
    # Restore the scroll position after DOM updates render
    await asyncio.sleep(0.15)
    try:
        await client.run_javascript(f"var el = document.getElementById('sidebar-books-list'); if (el) el.scrollTop = {scroll_pos};")
    except Exception:
        pass

def select_book(book_id: int):
    # Capture client context synchronously in the active event handler thread
    client = ui.context.client
    asyncio.create_task(async_select_book(book_id, client))


def select_book_from_portal(project_id: int, book_id: int):
    """Sets the active project, loads its settings from disk, and opens the target book workspace."""
    state.active_project_id = project_id
    load_project_settings_from_disk(project_id)
    state.active_book_id = book_id
    state.active_book_tab = 'Dashboard'
    state.active_log_widget = None
    ui.run_javascript('window.scrollTo(0, 0)')
    header_controls.refresh()
    main_layout.refresh()


def exit_to_portal():
    state.active_project_id = None
    state.active_book_id = None
    state.active_tool = None
    state.active_log_widget = None
    header_controls.refresh()
    main_layout.refresh()

def open_tool(tool_name: str):
    state.active_project_id = None
    state.active_book_id = None
    state.active_tool = tool_name
    header_controls.refresh()
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
    
    # Touch project modification timestamp
    from database.connection import touch_project
    touch_project(project_id)
    
    # Fast trigger to update action buttons and stepper layout
    state.project_status = "Transcribing"
    if hasattr(state, 'action_buttons_refresh') and state.action_buttons_refresh:
        state.action_buttons_refresh()
    header_controls.refresh()


def interrupt_comfy_execution():
    """Tells ComfyUI to immediately abort active generation and clear the pending queue."""
    comfy_url = get_setting("comfy_url", "127.0.0.1:8188")
    if "http" in comfy_url:
        comfy_url = comfy_url.replace("http://", "").replace("https://", "").strip("/")
    
    import requests
    try:
        # Interrupt the active workflow execution
        requests.post(f"http://{comfy_url}/interrupt", timeout=5.0)
    except Exception as e:
        state.add_console_log(f"[Comfy-API] Failed to send interrupt command: {str(e)}")
        
    try:
        # Clear the queue of pending jobs
        requests.post(f"http://{comfy_url}/queue", json={"clear": True}, timeout=5.0)
    except Exception as e:
        state.add_console_log(f"[Comfy-API] Failed to send clear queue command: {str(e)}")


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
        ui.notify("Stopping after the current image finishes...", type="info")
        state.project_status = "Prompts Created"
    
    # Fast trigger to update action buttons and stepper layout
    if hasattr(state, 'action_buttons_refresh') and state.action_buttons_refresh:
        state.action_buttons_refresh()
    header_controls.refresh()


def start_prompt_generation(project_id: int):
    """Launches the asynchronous, interruptible, and resumable prompt generation process."""
    from services.prompt_engine import start_project_prompt_gen
    
    # Reset all books to start prompt generation at 0% progress
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project:
            project.status = "Generating Prompts"
            project.modified_at = time.time()  # Touch modification timestamp directly inside the session
            session.add(project)
            
            books = session.exec(select(Book).where(Book.project_id == project_id)).all()
            for b in books:
                b.status = "Generating Prompts"
                b.progress = 0.0
                session.add(b)
            session.commit()
            
    asyncio.create_task(start_project_prompt_gen(project_id))
    ui.notify("Background prompt generation sequences initiated!", type="positive")
    
    state.project_status = "Generating Prompts"
    if hasattr(state, 'action_buttons_refresh') and state.action_buttons_refresh:
        state.action_buttons_refresh()
    header_controls.refresh()


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
    metadata.add_text("Quote", quote)
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

def format_eta(seconds: float) -> str:
    """Formats a remaining duration in seconds into a human-readable telemetry string."""
    if seconds <= 0:
        return "Completed"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


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

    # --- Pre-scan all books to establish Global Batch Progress metrics ---
    global_total_prompts = 0
    global_completed_prompts = 0
    actual_render_durations = []

    for b_info in books_info:
        csv_path = find_prompts_csv(project_name, b_info["name"])
        if csv_path:
            try:
                rows = read_prompts_from_csv(csv_path)
                valid_count = 0
                for r in rows:
                    p_text = r.get("prompt", "").strip()
                    if p_text and p_text.lower() != "none" and p_text.lower() != "refusal":
                        valid_count += 1
                global_total_prompts += valid_count
            except Exception:
                pass

    state.add_console_log(f"[Image-Gen] Global Project Batch Master Prompt Count: {global_total_prompts}")

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

        state.add_console_log(f"[Image-Gen] Discovered {total_prompts} scenes to process inside {book_name}.")
        
        # Nested target directory initialization
        parent_dir = Path(f"./output/{project_name}/{book_name}")
        out_dir = parent_dir / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pre-index existing files on disk exactly once to prevent thousands of O(N^2) directory globs
        existing_coords = set()
        def index_existing_images(directory: Path):
            if not directory.exists():
                return
            try:
                for filename in os.listdir(directory):
                    if filename.lower().endswith('.png'):
                        stem, _ = os.path.splitext(filename)
                        parts = stem.split('_')
                        if len(parts) >= 2:
                            try:
                                ch = int(parts[0])
                                sc = int(parts[1])
                                existing_coords.add((ch, sc))
                            except ValueError:
                                pass
            except Exception as e:
                state.add_console_log(f"[Image-Gen] Error pre-indexing folder {directory.name}: {e}")

        index_existing_images(parent_dir)
        index_existing_images(out_dir)

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

            # High performance coordinate set lookup
            if (chapter, scene) in existing_coords:
                state.add_console_log(f"[Image-Gen] Resume Skip: Ch {chapter}, Scene {scene} already rendered.")
                completed_prompts += 1
                global_completed_prompts += 1
                
                # Update Global project ETA dynamically on resume skips
                global_remaining = max(0, global_total_prompts - global_completed_prompts)
                if actual_render_durations:
                    avg_render_time = sum(actual_render_durations) / len(actual_render_durations)
                    eta_sec = avg_render_time * global_remaining
                else:
                    eta_sec = 10.0 * global_remaining  # fallback guess

                state.batch_eta_label = f"ETA: {format_eta(eta_sec)}"

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
            
            # Start timer for actual render block execution
            import time
            render_start_t = time.time()

            # Execute synchronous workflow API block inside background worker thread
            def render_block():
                return client.generate_image_sync(
                    workflow_json=workflow_json,
                    prompt_text=prompt_text,
                    neg_prompt_text=state.style_negative_prompt,
                    seed=seed,
                    overrides=state.style_workflow_overrides,
                    prefix=state.style_prompt_prefix,
                    suffix=getattr(state, "style_prompt_suffix", "")
                )

            img_bytes, logs = await asyncio.to_thread(render_block)
            
            # Save rendering speed duration
            render_dur = time.time() - render_start_t
            actual_render_durations.append(render_dur)

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
                    
                    # Store dynamic coordinate timestamp to selectively cache-bust only this card
                    if not hasattr(state, 'custom_image_timestamps'):
                        state.custom_image_timestamps = {}
                    state.custom_image_timestamps[(chapter, scene)] = int(time.time() * 1000)
                    
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
            global_completed_prompts += 1
            
            # Update dynamic Global Project Batch ETA predictions
            global_remaining = max(0, global_total_prompts - global_completed_prompts)
            avg_render_time = sum(actual_render_durations) / len(actual_render_durations)
            eta_sec = avg_render_time * global_remaining

            state.batch_eta_label = f"ETA: {format_eta(eta_sec)}"

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
        if hasattr(state, 'action_buttons_refresh') and state.action_buttons_refresh:
            state.action_buttons_refresh()
        header_controls.refresh()


def start_image_generation(project_id: int):
    """Launches the asynchronous, interruptible, and resumable image generation process."""
    if state.image_gen_active:
        ui.notify("An image generation task is already active.", type="warning")
        return

    # Auto-save current configurations to disk before starting
    save_project_settings_to_disk(project_id)

    # Initialize Telemetry Clock
    import time
    state.batch_start_time = time.time()
    state.batch_elapsed_sec = 0.0
    state.batch_eta_label = "ETA: Estimating..."

    state.image_gen_active = True
    state.cancel_image_gen_flag = False
    state.project_status = "Rendering Images"
    
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project:
            project.status = "Rendering Images"
            project.modified_at = time.time()  # Touch modification timestamp directly inside the session
            session.add(project)
            
            books = session.exec(select(Book).where(Book.project_id == project_id)).all()
            for b in books:
                b.status = "Rendering Images"
                b.progress = 0.0  # Reset progress to start rendering fresh visually
                session.add(b)
            session.commit()
    
    if hasattr(state, 'action_buttons_refresh') and state.action_buttons_refresh:
        state.action_buttons_refresh()
    header_controls.refresh()

    asyncio.create_task(run_project_image_gen(project_id))
    ui.notify("Background image rendering sequences initiated!", type="positive")

# Register global pipeline control callbacks onto state to prevent circular imports
state.start_image_generation_cb = start_image_generation
state.stop_image_generation_cb = stop_transcribing


# --- Dynamic WebSocket State-Updater (No full page refreshes!) ---
async def check_for_active_transcriptions():
    if state.active_project_id is not None:
        # Offload database queries and heavy file statistics checking to a background thread
        def perform_background_sync():
            with Session(engine) as session:
                project = session.get(Project, state.active_project_id)
                if not project:
                    return None
                    
                books = session.exec(select(Book).where(Book.project_id == state.active_project_id)).all()
                
                # Pre-fetch and assemble stats off-thread
                updates = {
                    "project_status": project.status,
                    "books": []
                }
                
                for b in books:
                    stats = get_book_stats_cached(project.name, b.name)
                    updates["books"].append({
                        "id": b.id,
                        "progress": b.progress,
                        "status": b.status,
                        "stats": stats
                    })
                return updates

        res = await asyncio.to_thread(perform_background_sync)
        if not res:
            return
            
        status_changed = (res["project_status"] != state.project_status)
        state.project_status = res["project_status"]
        
        if status_changed and hasattr(state, 'action_buttons_refresh') and state.action_buttons_refresh:
            try:
                state.action_buttons_refresh()
            except Exception:
                pass
        
        try:
            header_controls.refresh()
        except Exception:
            pass

        # Update timing statistics
        import time
        if state.image_gen_active and state.batch_start_time:
            state.batch_elapsed_sec = time.time() - state.batch_start_time

        # Update UI state binders on the main thread safely
        for b_data in res["books"]:
            b_id = b_data["id"]
            state.books_progress[b_id] = b_data["progress"]
            state.books_status[b_id] = b_data["status"]
            stats = b_data["stats"]
            
            # Dynamic subtitle based on project step status
            if state.project_status in ("Imported", "Transcribing"):
                if b_data["status"] == "Transcribing":
                    state.books_subtitle[b_id] = f"Transcribing... {int(b_data['progress'] * 100)}%"
                elif stats["has_transcript"]:
                    state.books_subtitle[b_id] = f"{stats['word_count']:,} words • {stats['estimated_scenes']} est. scenes"
                else:
                    state.books_subtitle[b_id] = "Awaiting transcription"
                    
            elif state.project_status in ("Transcribed", "Generating Prompts"):
                if b_data["status"] == "Generating Prompts":
                    state.books_subtitle[b_id] = f"Generating prompts... {int(b_data['progress'] * 100)}%"
                elif stats["total_prompts"] > 0:
                    state.books_subtitle[b_id] = f"{stats['total_prompts']} scene prompts ready"
                elif stats["has_transcript"]:
                    state.books_subtitle[b_id] = f"{stats['word_count']:,} words • {stats['estimated_scenes']} est. images"
                else:
                    state.books_subtitle[b_id] = "Awaiting prompts"
                    
            else:  # Image Generation, Finished, or Proofreading states
                total_scenes = stats["total_prompts"] or stats["estimated_scenes"] or 1
                if b_data["status"] == "Rendering Images":
                    state.books_subtitle[b_id] = f"Rendering: {stats['generated_images']} / {total_scenes} ({int(b_data['progress'] * 100)}%)"
                elif stats["approved_prompts"] == total_scenes and total_scenes > 0:
                    state.books_subtitle[b_id] = f"All Approved! • {total_scenes} scenes"
                else:
                    state.books_subtitle[b_id] = f"Rendered: {stats['generated_images']}/{total_scenes} ({stats['approved_prompts']} approved)"

        books_count = len(res["books"])
        if books_count > 0:
            avg_progress = sum(b["progress"] for b in res["books"]) / books_count
        else:
            avg_progress = 0.0
        state.project_progress = avg_progress
        state.project_progress_label = f"Batch Progress ({int(avg_progress * 100)}%)"

        if state.active_log_widget:
            try:
                new_logs = state.console_logs[state.logs_pushed_index:]
                for line in new_logs:
                    state.active_log_widget.push(line)
                state.logs_pushed_index = len(state.console_logs)
            except Exception:
                pass

        if hasattr(state, 'recent_prompts_refresh'):
            try:
                state.recent_prompts_refresh()
            except Exception:
                pass

# Register the statistics refresh callback on the safe state module
state.stats_refresh_callback = check_for_active_transcriptions


# --- Split-Panel Shell Renderer ---
def render_split_panel_shell(project_id: int):
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            ui.notify("Project not found.", type="negative")
            exit_to_portal()
            return
        books = session.exec(select(Book).where(Book.project_id == project.id)).all()

    # Seed static binding dictionaries and initial progress on initial layout rendering
    state.project_status = project.status
    if books:
        initial_avg = sum(b.progress for b in books) / len(books)
    else:
        initial_avg = 0.0
    state.project_progress = initial_avg
    state.project_progress_label = f"Batch Progress ({int(initial_avg * 100)}%)"

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

    with ui.grid(columns='260px 1fr').classes('w-full gap-6 items-start'):
        # LEFT NAVIGATION SIDEBAR
        with ui.column().classes('bg-white border rounded-xl p-4 gap-4 shadow-sm h-[calc(100vh-140px)] sticky top-24 w-full overflow-x-hidden'):
            ui.button(
                'Back to Projects', 
                icon='arrow_back', 
                on_click=exit_to_portal
            ).props('flat dense').classes('text-slate-600 text-xs self-start -ml-2 mb-2')
            
            # --- Project Global Header Card ---
            project_card_bg = 'bg-blue-50/70 border-blue-100/50 text-blue-700 font-bold' if state.active_book_id is None else 'bg-slate-50/50 hover:bg-slate-100'

            with ui.card().classes(f'w-full border p-3 rounded-lg shadow-xs gap-2 cursor-pointer transition-all overflow-hidden {project_card_bg}') \
                    .on('click', lambda: select_project(project.id)):
                with ui.row().classes('items-center gap-2 w-full justify-between'):
                    ui.icon('folder' if project.is_batch else 'menu_book', size='sm', color='slate-700')
                    # Dynamic Project Status Badge mapped to active step
                    ui.badge().classes('px-2 py-0.5 text-[10px] font-bold rounded-full bg-blue-100 text-blue-800') \
                        .bind_text_from(state, 'project_status', backward=lambda val: display_mapping.get(val, val))
                    
                # Applying break-words and whitespace-normal to prevent overflows
                ui.label(project.name).classes('text-sm font-bold text-slate-800 leading-tight break-words whitespace-normal')
                
                with ui.column().classes('w-full gap-0.5 mt-1'):
                    ui.label('').classes('text-[9px] font-bold text-slate-500 uppercase tracking-wide') \
                        .bind_text_from(state, 'project_progress_label')
                    ui.linear_progress(show_value=False).classes('w-full h-1.5 rounded-full') \
                        .bind_value_from(state, 'project_progress')

            # --- Project Real-Time Telemetry telemetry ---
            with ui.card().classes('w-full bg-slate-50 border p-2.5 rounded-lg gap-1') \
                    .bind_visibility_from(state, 'image_gen_active'):
                with ui.row().classes('w-full justify-between items-center text-[9px] font-black text-slate-500 uppercase tracking-wider'):
                    ui.label('Batch Telemetry')
                    ui.label().classes('text-blue-600 font-bold').bind_text_from(state, 'batch_eta_label')
                with ui.row().classes('w-full justify-between items-center text-[9px] font-semibold text-slate-400'):
                    ui.label().bind_text_from(state, 'batch_elapsed_sec', backward=lambda s: f"Elapsed: {int(s//60)}m {int(s%60)}s")
            
            ui.separator()
            ui.label('Books & Volumes').classes('text-[10px] font-bold text-slate-400 tracking-wider uppercase px-1')
            
            with ui.column().classes('w-full gap-2 overflow-y-auto flex-1').props('id="sidebar-books-list"'):
                for book in books:
                    # Seed bindings
                    if book.id not in state.books_progress:
                        state.books_progress[book.id] = book.progress
                    if book.id not in state.books_status:
                        state.books_status[book.id] = book.status
                    if book.id not in state.books_subtitle:
                        # Fetch initial stats
                        stats = get_book_stats_cached(project.name, book.name)
                        total_scenes = stats["total_prompts"] or stats["estimated_scenes"] or 1
                        state.books_subtitle[book.id] = f"Rendered: {stats['generated_images']}/{total_scenes}"

                    book_bg = 'bg-blue-50/70 border border-blue-100/50 text-blue-700 font-bold' if state.active_book_id == book.id else 'hover:bg-slate-50 text-slate-700'
                    
                    # Use items-start to allow the row card to expand vertically for multi-line content
                    with ui.row().classes(f'w-full p-2 rounded-lg cursor-pointer items-start justify-between transition-colors {book_bg}') \
                            .on('click', lambda b_id=book.id: select_book(b_id)):
                        with ui.row().classes('items-start gap-3 flex-1 min-w-0'):
                            # Upgrade cover sizes to standard 2:3 aspect ratio (w-10 h-14)
                            if book.cover_path:
                                ui.image(book.cover_path).classes('w-10 h-14 rounded object-cover shadow-sm border flex-shrink-0')
                            else:
                                with ui.column().classes('w-10 h-14 bg-slate-50 border border-dashed rounded items-center justify-center flex-shrink-0 text-slate-400'):
                                    ui.icon('library_books', size='16px')
                            
                            # Multi-line column container: allows subtitles and wrapped titles to flow downwards naturally
                            with ui.column().classes('gap-0.5 flex-1 min-w-0'):
                                ui.label(book.name).classes('text-xs font-semibold leading-tight break-words whitespace-normal')
                                # BIND TEXT REACTIVELY (Supports multi-line wraps cleanly)
                                ui.label('').classes('text-[9px] font-medium text-slate-500 leading-normal break-words whitespace-normal') \
                                    .bind_text_from(state.books_subtitle, book.id)
                                

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
                    save_project_settings_to_disk
                )
            else:
                render_book_tabs(state.active_book_id)


# --- Dynamic Main Page Wrapper ---
@ui.refreshable
def main_layout():
    if state.active_tool == "lora_contact_sheet":
        render_lora_contact_sheet(exit_to_portal)
    elif state.active_project_id is None:
        render_portal_view(select_project, select_book_from_portal, refresh_dashboard)
    else:
        render_split_panel_shell(state.active_project_id)


# Register layout reference inside rendering module
register_main_layout(main_layout)

# Initialize settings modal
settings_modal = SettingsModal(app_settings, restart_app)


# --- TOPBAR HEADERS & CONTROLS ---

def render_topbar_stepper(status: str):
    """Compacts the horizontal pipeline progress stepper to fit beautifully inside the global sticky header."""
    from ui.pages.project import STAGES, get_active_stage_idx
    current_stage_idx = get_active_stage_idx(status)
    
    with ui.row().classes('items-center gap-3 bg-slate-700/40 px-4 py-1.5 rounded-lg border border-slate-600/30 text-xs'):
        for idx, stage_name in enumerate(STAGES):
            is_completed = idx < current_stage_idx
            is_active = idx == current_stage_idx
            
            with ui.row().classes('items-center gap-1'):
                if is_completed:
                    ui.icon('check_circle', color='emerald-400', size='15px')
                    ui.label(stage_name).classes('text-emerald-300 font-bold')
                elif is_active:
                    ui.icon('radio_button_checked', color='blue-400', size='15px').classes('animate-pulse')
                    ui.label(stage_name).classes('text-blue-300 font-black')
                else:
                    ui.icon('radio_button_unchecked', color='slate-400', size='15px')
                    ui.label(stage_name).classes('text-slate-400 font-medium')
                    
            if idx < len(STAGES) - 1:
                ui.icon('chevron_right', color='slate-500', size='12px')


@ui.refreshable
def header_controls():
    """Reactive layout container that swaps search bars for pipeline steppers based on navigation state."""
    if state.active_project_id is None:
        with ui.row().classes('items-center gap-4 bg-slate-700/50 p-1 rounded-lg border border-slate-600/30'):
            ui.input(
                placeholder='Search projects...',
                value=state.search_query,
                on_change=lambda e: (setattr(state, 'search_query', e.value), refresh_dashboard())
            ).props('dark borderless dense').classes('w-48 px-2')
            
            ui.select(
                options=['Most Recent', 'Alphabetical'],
                value=state.selected_sort,
                on_change=lambda e: (setattr(state, 'selected_sort', e.value), refresh_dashboard())
            ).props('dark borderless dense').classes('w-32 text-sm')
    else:
        render_topbar_stepper(state.project_status)

# Export callback so workspace settings can trigger reactive header updates
state.active_header_refresh = header_controls.refresh


# --- Process Control and VRAM Clearing Utilities (Phase B) ---

def launch_comfyui():
    """Launches ComfyUI in a non-blocking background subprocess, auto-detecting standalone portable installations."""
    comfy_path_str = get_setting("comfy_path", "F:/AI/ComfyUI/ComfyUI")
    if not comfy_path_str:
        ui.notify("ComfyUI path is not configured in settings.", type="warning")
        return
        
    comfy_dir = Path(comfy_path_str).resolve()
    
    # Auto-resolve path: if they pointed to the inner 'ComfyUI' folder, promote to the parent standalone folder
    if comfy_dir.name.lower() == "comfyui" and (comfy_dir.parent / "run_nvidia_gpu.bat").exists():
        comfy_dir = comfy_dir.parent
        state.add_console_log(f"[Comfy-Launcher] Auto-resolved ComfyUI root directory to: {comfy_dir}")

    # If the folder still does not exist
    if not comfy_dir.exists():
        ui.notify(f"ComfyUI directory not found: {comfy_dir}", type="negative")
        return

    bat_file = comfy_dir / "run_nvidia_gpu.bat"
    sh_file = comfy_dir / "run_nvidia_gpu.sh"
    python_file = comfy_dir / "main.py"
    inner_python_file = comfy_dir / "ComfyUI" / "main.py"
    
    # Locate embedded python installations in portable packages
    embedded_py_win = comfy_dir / "python_embeded" / "python.exe"
    embedded_py_unix = comfy_dir / "python_embeded" / "bin" / "python"
    
    # Retrieve startup launch arguments safely using shlex to parse string elements
    import shlex
    comfy_args_str = get_setting("comfy_args", "--windows-standalone-build") or ""
    comfy_args_list = shlex.split(comfy_args_str)
    
    try:
        if os.name == 'nt':  # Windows
            # Prioritize direct embedded Python execution to ensure all custom parameters
            # (like --port, --highvram) are forwarded properly. Standalone ComfyUI .bat files
            # do not forward command-line parameters natively because they lack a `%*` suffix.
            if embedded_py_win.exists() and inner_python_file.exists():
                subprocess.Popen([str(embedded_py_win), "-s", "ComfyUI\\main.py"] + comfy_args_list, cwd=str(comfy_dir), creationflags=subprocess.CREATE_NEW_CONSOLE)
                ui.notify("Launching ComfyUI via embedded python...", type="info")
            elif bat_file.exists():
                subprocess.Popen([str(bat_file)] + comfy_args_list, cwd=str(comfy_dir), creationflags=subprocess.CREATE_NEW_CONSOLE)
                ui.notify("Launching ComfyUI via run_nvidia_gpu.bat...", type="info")
            elif python_file.exists():
                venv_python = comfy_dir / "venv" / "Scripts" / "python.exe"
                py_exec = str(venv_python) if venv_python.exists() else sys.executable
                subprocess.Popen([py_exec, "main.py"] + comfy_args_list, cwd=str(comfy_dir), creationflags=subprocess.CREATE_NEW_CONSOLE)
                ui.notify(f"Launching ComfyUI via Python interpreter ({py_exec})...", type="info")
            else:
                ui.notify("Could not locate run_nvidia_gpu.bat, embedded python, or main.py.", type="warning")
        else:  # Linux / macOS
            # Prioritize direct embedded or local Python execution to bypass static .sh constraints
            if embedded_py_unix.exists() and inner_python_file.exists():
                subprocess.Popen([str(embedded_py_unix), "-s", "ComfyUI/main.py"] + comfy_args_list, cwd=str(comfy_dir))
                ui.notify("Launching ComfyUI via embedded python...", type="info")
            elif sh_file.exists():
                subprocess.Popen(["bash", str(sh_file)] + comfy_args_list, cwd=str(comfy_dir))
                ui.notify("Launching ComfyUI via shell script...", type="info")
            elif python_file.exists():
                venv_python = comfy_dir / "venv" / "bin" / "python"
                py_exec = str(venv_python) if venv_python.exists() else sys.executable
                subprocess.Popen([py_exec, "main.py"] + comfy_args_list, cwd=str(comfy_dir))
                ui.notify("Launching ComfyUI via Python interpreter...", type="info")
            else:
                ui.notify("Could not locate shell script, embedded python, or main.py.", type="warning")
    except Exception as e:
        ui.notify(f"Failed to launch ComfyUI process: {str(e)}", type="negative")


def free_all_memory():
    """Unloads model weights and clears memory caches from both ComfyUI and LM Studio/Ollama."""
    ui.notify("Dispatching VRAM/RAM clearance commands...", type="info")
    
    # 1. Clear ComfyUI Cache and Unload Models
    comfy_url = get_setting("comfy_url", "127.0.0.1:8188")
    if "http" in comfy_url:
        comfy_url = comfy_url.replace("http://", "").replace("https://", "").strip("/")
        
    import requests
    try:
        resp = requests.post(
            f"http://{comfy_url}/free", 
            json={"unload_models": True, "free_memory": True}, 
            timeout=4.0
        )
        if resp.status_code == 200:
            state.add_console_log("[Memory-Engine] Successfully requested ComfyUI model unload and cache clear.")
        else:
            state.add_console_log(f"[Memory-Engine] ComfyUI free returned status code: {resp.status_code}")
    except Exception as e:
        state.add_console_log(f"[Memory-Engine] ComfyUI clearance skipped or host offline: {str(e)}")

    # 2. Clear LM Studio / Ollama Model Weights
    llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
    model_name = get_setting("llm_model", "local-model")
    
    # Resolve the root host address to avoid trailing paths (e.g., "http://127.0.0.1:1234/v1" -> "http://127.0.0.1:1234")
    import urllib.parse
    try:
        parsed_url = urllib.parse.urlparse(llm_url)
        base_host_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    except Exception:
        base_host_url = llm_url

    if "11434" in llm_url:  # Ollama Server
        try:
            resp = requests.post(
                f"{base_host_url}/api/generate", 
                json={"model": model_name, "keep_alive": 0, "prompt": ""}, 
                timeout=4.0
            )
            if resp.status_code == 200:
                state.add_console_log(f"[Memory-Engine] Successfully requested Ollama to unload model '{model_name}'.")
        except Exception as e:
            state.add_console_log(f"[Memory-Engine] Ollama clearance skipped or host offline: {str(e)}")
    else:  # LM Studio Server
        try:
            # Query models to fetch active model instance IDs
            models_resp = requests.get(f"{base_host_url}/api/v1/models", timeout=4.0)
            if models_resp.status_code == 200:
                models_data = models_resp.json()
                models_list = models_data.get("models", [])
                if not models_list and "data" in models_data:
                    models_list = models_data.get("data", [])
                
                unloaded_count = 0
                for model in models_list:
                    # In LM Studio v1 REST API, loaded instances reside inside 'loaded_instances' list of each model
                    loaded_instances = model.get("loaded_instances", [])
                    for instance in loaded_instances:
                        instance_id = instance.get("id")
                        if instance_id:
                            unload_resp = requests.post(
                                f"{base_host_url}/api/v1/models/unload",
                                json={"instance_id": instance_id},
                                timeout=4.0
                            )
                            if unload_resp.status_code == 200:
                                state.add_console_log(f"[Memory-Engine] Successfully unloaded LM Studio model instance: {instance_id}")
                                unloaded_count += 1
                            else:
                                state.add_console_log(f"[Memory-Engine] Unload command failed for instance {instance_id} with status: {unload_resp.status_code}")
                                
                if unloaded_count == 0:
                    state.add_console_log("[Memory-Engine] No active model instances discovered in LM Studio's loaded cache.")
            else:
                # Direct fallback unload trigger using configured setting name as a generic ID
                requests.post(
                    f"{base_host_url}/api/v1/models/unload", 
                    json={"instance_id": model_name}, 
                    timeout=4.0
                )
                state.add_console_log(f"[Memory-Engine] Dispatched generic LM Studio unload request for model: {model_name}")
        except Exception as e:
            state.add_console_log(f"[Memory-Engine] LM Studio clearance skipped or host offline: {str(e)}")

    ui.notify("VRAM and RAM clearance commands successfully sent!", type="positive")


# --- Header & Top Bar Navigation Layout ---
with ui.header(elevated=False).classes('bg-slate-800 text-white px-6 py-4 justify-between items-center'):
    with ui.row().classes('items-center gap-3'):
        ui.icon('auto_awesome', size='md').classes('text-blue-400')
        ui.label('ABI-Pipeline').classes('text-xl font-bold tracking-tight cursor-pointer').on('click', exit_to_portal)

    header_controls()

    with ui.row().classes('items-center gap-3'):
        # Launch Comfy Button (Universal Shortcut)
        ui.button(
            'Launch Comfy', 
            icon='bolt', 
            on_click=launch_comfyui
        ).props('flat dense').classes('text-xs text-blue-400 hover:text-blue-300 font-bold px-2 py-1 bg-slate-700/50 rounded border border-slate-600/30')
        
        # Free VRAM Button (Universal Shortcut)
        ui.button(
            'Free VRAM', 
            icon='memory', 
            on_click=free_all_memory
        ).props('flat dense').classes('text-xs text-rose-400 hover:text-rose-300 font-bold px-2 py-1 bg-slate-700/50 rounded border border-slate-600/30')

        with ui.button(icon='construction', color='slate-600') as tools_btn:
            tools_btn.classes('text-white text-sm capitalize rounded-lg')
            with ui.menu() as menu:
                ui.menu_item('LoRA Contact Sheets', on_click=lambda: open_tool('lora_contact_sheet'))
                ui.menu_item('Style Library', on_click=lambda: ui.notify('Style Library coming soon'))
                ui.menu_item('Prompt Templates', on_click=lambda: ui.notify('Templates coming soon'))
        ui.button(icon='settings', on_click=lambda: settings_modal.open()).props('flat round color=white')


# --- Main Application Page Frame ---
with ui.column().classes('w-full max-w-7xl mx-auto p-6 gap-6'):
    main_layout()


# --- Polling Update Refresh Timer ---
ui.timer(2.0, check_for_active_transcriptions)

# Launch our app
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="ABI-Pipeline")