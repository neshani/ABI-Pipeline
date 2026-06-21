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

# --- Register CUDA & cuDNN DLL Directories (Windows Isolation) ---
def setup_cuda_dll_directories():
    """
    Dynamically registers Pip-installed CUDA and cuDNN DLL directories on Windows.
    Resolves cudnn64_9.dll errors by appending to os.environ["PATH"] (for transitive C++ loads),
    calling os.add_dll_directory() (for Python 3.8+ direct imports), and triggering ort.preload_dlls().
    """
    if os.name != 'nt':
        return
        
    # 1. Process-level singleton guard to prevent redundant execution on duplicate module imports
    if getattr(sys, "_abi_cuda_setup_done", False):
        return
    sys._abi_cuda_setup_done = True
    
    try:
        venv_base = Path(sys.executable).parent.parent
        nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
        
        candidate_dirs = []
        if nvidia_base_path.exists():
            # Scan all pip-installed NVIDIA wheel folders and locate any /bin subdirectories
            for item in nvidia_base_path.iterdir():
                if item.is_dir():
                    bin_dir = item / "bin"
                    if bin_dir.exists() and bin_dir.is_dir():
                        candidate_dirs.append(bin_dir)
                        
        # Check and include PyTorch's complementary CUDA library directory if present
        torch_lib_path = venv_base / 'Lib' / 'site-packages' / 'torch' / 'lib'
        if torch_lib_path.exists() and torch_lib_path.is_dir():
            candidate_dirs.append(torch_lib_path)
            
        for path in candidate_dirs:
            path_str = str(path.resolve())
            
            # Prepend to Windows PATH environment variable.
            # Critical for C++ libraries (like onnxruntime_providers_cuda.dll) performing transitive
            # LoadLibrary calls internally for dependency DLLs (e.g. cudnn64_9.dll loading cudnn_ops64_9.dll).
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
            
            # Register with Python's direct DLL search pathways
            try:
                os.add_dll_directory(path_str)
                print(f"[CUDA-Loader] Registered DLL directory: {path_str}")
            except Exception as e:
                print(f"[CUDA-Loader] Warning: Failed to add DLL directory {path.name}: {e}")
                
        # 2. Call ONNX Runtime's native preloader ONLY if local DLL paths were found and registered.
        # This completely avoids ugly console warning logs on non-CUDA / CPU-only environments.
        if candidate_dirs:
            try:
                import onnxruntime as ort
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()
                    print("[CUDA-Loader] Successfully executed onnxruntime.preload_dlls()")
            except Exception:
                pass
            
    except Exception as ex:
        print(f"[CUDA-Loader] Error setting up DLL directories: {ex}")

# Execute DLL path setups immediately before project submodules are imported
setup_cuda_dll_directories()

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
from ui.components.onboarding_wizard import OnboardingWizard

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
init_db()

# --- Mount Static Directory & Auto-Create on Startup ---
static_dir = Path("./static")
static_dir.mkdir(exist_ok=True)
app.add_static_files('/static', 'static')

# --- Inject Client-Side Focus Reset Actions ---
ui.add_head_html('''
<script>
window.addEventListener('focus', () => {
    const fav = document.querySelector("link[rel~='icon']");
    if (fav && fav.href.includes('favicon_alert.png')) {
        fav.href = '/static/favicon.png';
    }
    if (document.title.startsWith('(✓)') || document.title.includes('Process Finished')) {
        document.title = "ABI-Pipeline";
    }
});
</script>
''')


# --- Initialize SQLite Database & Recovery Engines (One-time Startup Event) ---

def run_startup_recovery():
    """Clears stuck tasks, initializes GPU telemetry, and recovers workspaces on startup."""
    # Ensure stuck tasks are cleared
    reset_stuck_transcriptions()

    # Safely initialize telemetry bindings
    init_gpu_telemetry()

    # Run workspace recovery to restore wiped database entries
    with Session(engine) as session:
        recover_from_temp_workspaces(session)

# Registers the callback so it only executes once in the active child worker process
app.on_startup(run_startup_recovery)

# --- Programmatic App Restart Engine ---
should_restart = False

def restart_app():
    """
    Triggers a clean app reload. Sets the restart flag and shuts down the NiceGUI app cleanly.
    The batch file start.bat intercepts exit code 123 to loop and restart the process instantly.
    """
    global should_restart
    should_restart = True
    ui.notify("Refreshing application modules...", type="warning")
    asyncio.create_task(async_restart())

async def async_restart():
    # Allow the notification to display before shutting down
    await asyncio.sleep(1.0)
    app.shutdown()

# Register shutdown hook to cleanly handle restarts vs normal closures
app.on_shutdown(lambda: os._exit(123) if should_restart else os._exit(0))

# --- Default App Configurations ---
DEFAULT_SETTINGS = {
    "comfy_url": "http://127.0.0.1:8188",
    "comfy_path": "",
    "comfy_args": "--windows-standalone-build",
    "llm_url": "http://127.0.0.1:11434",
    "llm_api_key": "",
    "llm_model": "unsloth/gemma-4-e4b-it",
    "llm_launch_path": "",
    "llm_launch_args": "",
    "stt_engine": "Parakeet ONNX",
    "stt_device": "GPU/CUDA",
    "batch_size": 30,
    "output_dir": "./output",
    "enable_desktop_notifications": False,
    "notification_threshold": 30,
    "wizard_completed": False
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

def cleanup_book_workspace_resources():
    """Aggressively stops and deallocates book-specific timers and keyboard listeners to prevent VRAM/RAM leakage."""
    if getattr(state, 'book_scroll_timer', None):
        try:
            state.book_scroll_timer.cancel()
            state.book_scroll_timer.delete()
        except Exception:
            pass
        state.book_scroll_timer = None

    if getattr(state, 'book_update_timer', None):
        try:
            state.book_update_timer.cancel()
            state.book_update_timer.delete()
        except Exception:
            pass
        state.book_update_timer = None

    if getattr(state, 'book_keyboard', None):
        try:
            state.book_keyboard.delete()
        except Exception:
            pass
        state.book_keyboard = None


# --- Navigation State Control Handlers ---
def select_project(project_id: int):
    """Sets the active project in memory and restores its saved settings from disk."""
    cleanup_book_workspace_resources()
    
    state.active_project_id = project_id
    state.active_book_id = None
    state.active_project_tab = 'Dashboard'
    state.active_log_widget = None  # Clear log references on panel change
    
    # Fast load settings from project folder
    load_project_settings_from_disk(project_id)
    
    # Set the initial baseline for transition checking to prevent spurious complete alerts
    with Session(engine) as session:
        proj = session.get(Project, project_id)
        if proj:
            state.last_known_status = proj.status
    
    # Dynamically rescan project database state on load to align database with disk
    try:
        from ui.pages.project.dashboard import rescan_project_database_state
        rescan_project_database_state(project_id)
    except Exception as ex:
        print(f"[ProjectLoad-Sync] Error during load synchronization: {ex}")

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
    cleanup_book_workspace_resources()
    
    state.active_project_id = project_id
    load_project_settings_from_disk(project_id)
    
    # Set the initial baseline for transition checking to prevent spurious complete alerts
    with Session(engine) as session:
        proj = session.get(Project, project_id)
        if proj:
            state.last_known_status = proj.status

    # Dynamically rescan project database state on load to align database with disk
    try:
        from ui.pages.project.dashboard import rescan_project_database_state
        rescan_project_database_state(project_id)
    except Exception as ex:
        print(f"[ProjectLoad-Sync] Error during load synchronization: {ex}")

    state.active_book_id = book_id
    state.active_book_tab = 'Dashboard'
    state.active_log_widget = None
    ui.run_javascript('window.scrollTo(0, 0)')
    header_controls.refresh()
    main_layout.refresh()


def exit_to_portal():
    cleanup_book_workspace_resources()
    
    state.active_project_id = None
    state.active_book_id = None
    state.active_tool = None
    state.active_log_widget = None
    header_controls.refresh()
    main_layout.refresh()

def open_tool(tool_name: str):
    cleanup_book_workspace_resources()
    
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

    # Clear cancellation tracking status
    state.was_manually_cancelled = False
    state.active_task_type = "transcription"

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
    # Set the cancellation tracking status
    state.was_manually_cancelled = True

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
    import time
    
    # Clear cancellation tracking status
    state.was_manually_cancelled = False
    state.active_task_type = "prompt_gen"

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
                    
                    # Increment count of newly rendered images
                    if hasattr(state, 'newly_generated_count'):
                        state.newly_generated_count += 1
                    else:
                        state.newly_generated_count = 1
                    
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

    # Initialize actual generation counter to bypass cached skips in notification metrics
    state.newly_generated_count = 0
    state.active_task_type = "image_gen"

    state.image_gen_active = True
    state.cancel_image_gen_flag = False
    
    # Clear cancellation tracking status
    state.was_manually_cancelled = False

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
                        "duration": b.duration or 0.0,  # Grab book duration
                        "stats": stats
                    })
                return updates

        res = await asyncio.to_thread(perform_background_sync)
        if not res:
            return
            
        status_changed = (res["project_status"] != state.project_status)
        
        # State transition tracking for task completions
        if not hasattr(state, 'last_known_status'):
            state.last_known_status = None
            
        old_status = state.last_known_status
        new_status = res["project_status"]
        state.project_status = new_status
        
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

        # --- CORRECTION: Duration-Weighted Batch Progress Calculation ---
        total_duration = sum(b["duration"] for b in res["books"])
        if total_duration > 0:
            completed_duration = sum(b["progress"] * b["duration"] for b in res["books"])
            avg_progress = completed_duration / total_duration
        else:
            books_count = len(res["books"])
            avg_progress = sum(b["progress"] for b in res["books"]) / books_count if books_count > 0 else 0.0

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

        # --- Dynamic Browser Tab Progress and Notification Orchestrator ---
        active_statuses = {"Transcribing", "Generating Prompts", "Rendering Images"}
        was_cancelled = getattr(state, "was_manually_cancelled", False)
        
        # 1. Update Title During Active Process Runs
        if new_status in active_statuses:
            progress_pct = int(state.project_progress * 100)
            status_text = {
                "Transcribing": "Transcribing",
                "Generating Prompts": "Prompting",
                "Rendering Images": "Rendering"
            }.get(new_status, new_status)
            
            ui.run_javascript(f'document.title = "[{progress_pct}%] {status_text}... | ABI-Pipeline";')
            
        # 2. State Transition Completion Event Detector (Process Just Ended)
        elif (old_status in active_statuses or getattr(state, "active_task_type", None) is not None) and new_status not in active_statuses:
            if was_cancelled:
                state.was_manually_cancelled = False
                ui.run_javascript('''
                    document.title = "ABI-Pipeline";
                    const fav = document.querySelector("link[rel~='icon']");
                    if (fav) fav.href = "/static/favicon.png";
                ''')
            else:
                ui.run_javascript('''
                    document.title = "(✓) Process Finished | ABI-Pipeline";
                    const fav = document.querySelector("link[rel~='icon']");
                    if (fav) fav.href = "/static/favicon_alert.png";
                ''')
                
                enable_notif = get_setting("enable_desktop_notifications") in ("True", True)
                notif_threshold = int(get_setting("notification_threshold", 30))
                
                # Check actual work accomplished rather than the static project total
                task_type = getattr(state, "active_task_type", None)
                if task_type == "image_gen" or old_status == "Rendering Images":
                    actual_processed = getattr(state, "newly_generated_count", 0)
                    notif_body = f"Successfully generated {actual_processed} new images."
                else:
                    # Fallback metric for transcription/prompting runs
                    actual_processed = 0
                    for b_data in res["books"]:
                        stats = b_data["stats"]
                        actual_processed += stats.get("total_prompts", 0) or stats.get("estimated_scenes", 0) or 0
                    notif_body = f"The background process has finished successfully ({actual_processed} items processed)."
                    
                if enable_notif and actual_processed >= notif_threshold:
                    ui.run_javascript(f'''
                        if ("Notification" in window && Notification.permission === "granted") {{
                            new Notification("ABI-Pipeline", {{
                                body: "{notif_body}",
                                icon: "/static/favicon_alert.png"
                            }});
                        }}
                    ''')
            
            # Clear active task type once the completion event has been processed
            state.active_task_type = None

        state.last_known_status = new_status

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
        # --- CORRECTION: Duration-Weighted Initial Progress Evaluation ---
        total_duration = sum(b.duration or 0.0 for b in books)
        if total_duration > 0:
            completed_duration = sum((b.progress or 0.0) * (b.duration or 0.0) for b in books)
            initial_avg = completed_duration / total_duration
        else:
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
        "Images Created": "Image Gen",
        "Proofreading": "Image Gen",
        "Finished": "Image Gen"
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
                    # Dynamic Project Status Badge using ui.label to avoid default background overrides
                    ui.label().classes('px-2 py-0.5 text-[10px] font-bold rounded-full bg-blue-50 text-blue-700 border border-blue-200/60') \
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
    # The onboarding wizard is now launched via an app.on_connect event handler.

    if state.active_tool == "lora_contact_sheet":
        render_lora_contact_sheet(exit_to_portal)
    elif state.active_project_id is None:
        render_portal_view(select_project, select_book_from_portal, refresh_dashboard)
    else:
        render_split_panel_shell(state.active_project_id)


# Register layout reference inside rendering module
register_main_layout(main_layout)

# Initialize settings modal & onboarding wizard
settings_modal = SettingsModal(app_settings, restart_app)
onboarding_wizard = OnboardingWizard(
    app_settings, 
    on_complete_callback=refresh_dashboard,
    launch_comfy_callback=lambda: launch_comfyui()
)

# Register onboarding wizard trigger globally to prevent circular imports
state.show_onboarding_wizard = onboarding_wizard.open



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


def launch_llm_host():
    """Launches the configured local LLM host in a non-blocking background subprocess."""
    llm_path_str = get_setting("llm_launch_path", "")
    if not llm_path_str:
        ui.notify("LLM launch path is not configured in settings.", type="warning")
        return
        
    llm_dir_or_file = Path(llm_path_str).resolve()
    if not llm_dir_or_file.exists():
        ui.notify(f"LLM executable/directory not found: {llm_dir_or_file}", type="negative")
        return

    import shlex
    llm_args_str = get_setting("llm_launch_args", "") or ""
    llm_args_list = shlex.split(llm_args_str)
    
    if llm_dir_or_file.is_dir():
        ui.notify("LLM Launch Path points to a directory. Please specify the executable file path.", type="warning")
        return
        
    working_dir = llm_dir_or_file.parent
    
    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen([str(llm_dir_or_file)] + llm_args_list, cwd=str(working_dir), creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Linux / macOS
            subprocess.Popen([str(llm_dir_or_file)] + llm_args_list, cwd=str(working_dir))
        ui.notify("Launching LLM Host process...", type="info")
        state.add_console_log(f"[LLM-Launcher] Process launched: {llm_dir_or_file.name} {llm_args_str}")
    except Exception as e:
        ui.notify(f"Failed to launch LLM process: {str(e)}", type="negative")
        state.add_console_log(f"[LLM-Launcher] Error launching LLM process: {str(e)}")


def find_comfy_output_dir() -> Optional[Path]:
    """Finds the ComfyUI output directory based on current settings."""
    comfy_path_str = get_setting("comfy_path", "")
    if not comfy_path_str:
        return None
        
    base_dir = Path(comfy_path_str).resolve()
    
    # Check possible output locations
    paths_to_check = [
        base_dir / "output",
        base_dir / "ComfyUI" / "output"
    ]
    
    for path in paths_to_check:
        if path.exists() and path.is_dir():
            return path
            
    return None


def clear_comfy_outputs():
    """Cleans up ABI-specific files inside the ComfyUI output directory safely."""
    output_dir = find_comfy_output_dir()
    if not output_dir:
        ui.notify("Could not resolve ComfyUI output folder. Verify ComfyUI Path in Settings.", type="negative")
        return
        
    try:
        count = 0
        for f in output_dir.glob("abi_*.png"):
            if f.is_file():
                f.unlink()
                count += 1
                
        if count > 0:
            ui.notify(f"Successfully cleaned up {count} ABI-generated file(s) from ComfyUI output folder.", type="positive")
            state.add_console_log(f"[Housekeeping] Deleted {count} 'abi_*' images from {output_dir}")
        else:
            ui.notify("No ABI-generated files found to clear.", type="info")
    except Exception as e:
        ui.notify(f"Error during cleanup: {str(e)}", type="negative")


# --- Declarative confirmation dialog for clearing ComfyUI output files ---
with ui.dialog() as clear_comfy_dialog, ui.card().classes('w-full max-w-md p-6 rounded-xl'):
    ui.label('Clear ComfyUI Outputs').classes('text-xl font-bold text-slate-800 mb-2')
    ui.label('This will delete all temporary or cache files starting with "abi_" in your ComfyUI output folder. Unrelated files will not be touched.').classes('text-sm text-slate-500 mb-4')
    
    with ui.row().classes('w-full justify-end gap-3'):
        ui.button('Cancel', on_click=clear_comfy_dialog.close).props('flat color=slate')
        ui.button(
            'Confirm Clear', 
            on_click=lambda: (clear_comfy_outputs(), clear_comfy_dialog.close())
        ).classes('bg-rose-600 text-white font-semibold')


async def free_all_memory():
    """Unloads model weights and clears memory caches from ComfyUI and active LLM providers asynchronously to prevent UI freeze."""
    ui.notify("Dispatching VRAM/RAM clearance commands...", type="info")
    
    # 1. Clear ComfyUI Cache and Unload Models
    comfy_url = get_setting("comfy_url", "127.0.0.1:8188")
    if "http" in comfy_url:
        comfy_url = comfy_url.replace("http://", "").replace("https://", "").strip("/")
        
    llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
    model_name = get_setting("llm_model", "local-model")
    
    # Resolve the root host address to avoid trailing paths (e.g., "http://127.0.0.1:1234/v1" -> "http://127.0.0.1:1234")
    import urllib.parse
    try:
        parsed_url = urllib.parse.urlparse(llm_url)
        base_host_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    except Exception:
        base_host_url = llm_url

    state.add_console_log(f"[Memory-Engine] Dispatching asynchronous target VRAM clear requests for configured model '{model_name}' to: {base_host_url}")

    import httpx
    async with httpx.AsyncClient() as client:
        # --- TARGET 1: ComfyUI Model Unload & Cache Clear ---
        try:
            resp = await client.post(
                f"http://{comfy_url}/free", 
                json={"unload_models": True, "free_memory": True}, 
                timeout=2.0
            )
            if resp.status_code == 200:
                state.add_console_log("[Memory-Engine] Successfully requested ComfyUI model unload and cache clear.")
            else:
                state.add_console_log(f"[Memory-Engine] ComfyUI free returned status code: {resp.status_code}")
        except Exception as e:
            state.add_console_log(f"[Memory-Engine] ComfyUI clearance skipped or host offline: {str(e)}")

        # --- TARGET 2: Target-Specific LM Studio Unload ---
        try:
            # Send targeted unload specifically for the model name saved in settings
            await client.post(
                f"{base_host_url}/api/v1/models/unload", 
                json={"instance_id": model_name}, 
                timeout=1.5
            )
            state.add_console_log(f"[Memory-Engine] Sent LM Studio targeted unload command for: {model_name}")
        except Exception:
            pass

        # --- TARGET 3: Target-Specific llama-server Unload ---
        try:
            # Send targeted unload specifically for the model name saved in settings
            await client.post(
                f"{base_host_url}/models/unload", 
                json={"model": model_name}, 
                timeout=1.5
            )
            state.add_console_log(f"[Memory-Engine] Sent llama-server targeted unload command for: {model_name}")
        except Exception:
            pass

    ui.notify("VRAM and RAM clearance commands successfully sent!", type="positive")


async def quit_app():
    """Performs a clean shutdown sequence, automatically closing the browser-initiated process console."""
    ui.notify("Shutting down cleanly...", type="warning", timeout=3)
    await asyncio.sleep(1.0)
    app.shutdown()

# Initialize Onboarding Wizard safely if not already done earlier in the file
if 'onboarding_wizard' not in globals():
    onboarding_wizard = OnboardingWizard(
        app_settings, 
        on_complete_callback=refresh_dashboard,
        launch_comfy_callback=launch_comfyui
    )

# --- GPU/VRAM Telemetry Helper & Widget ---

def init_gpu_telemetry():
    """Attempts to initialize NVML and detect the primary NVIDIA GPU safely."""
    try:
        from pynvml import (
            nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName,
            nvmlDeviceGetPowerManagementLimit
        )
        nvmlInit()
        # Get handle for first GPU (index 0)
        handle = nvmlDeviceGetHandleByIndex(0)
        
        # Get card model name
        name_raw = nvmlDeviceGetName(handle)
        state.gpu_name = name_raw.decode('utf-8') if isinstance(name_raw, bytes) else str(name_raw)
        
        # Get max design power limit (returned in milliwatts, convert to Watts)
        try:
            limit_mw = nvmlDeviceGetPowerManagementLimit(handle)
            state.gpu_power_limit = limit_mw / 1000.0
        except Exception:
            state.gpu_power_limit = 0.0
            
        state.gpu_telemetry_supported = True
        state.add_console_log(f"[NVML-Telemetry] Detected NVIDIA GPU: {state.gpu_name} (Max Power Limit: {state.gpu_power_limit:.1f}W)")
    except Exception as e:
        state.gpu_telemetry_supported = False
        state.add_console_log(f"[NVML-Telemetry] NVIDIA Management Library (NVML) initialization bypassed: {str(e)}")


def update_gpu_telemetry():
    """Polls NVML at regular intervals for temperature, memory, utilization, and power stats."""
    if not state.gpu_telemetry_supported:
        return
        
    try:
        from pynvml import (
            nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
            nvmlDeviceGetMemoryInfo, nvmlDeviceGetTemperature,
            nvmlDeviceGetPowerUsage, NVML_TEMPERATURE_GPU
        )
        handle = nvmlDeviceGetHandleByIndex(0)
        
        # 1. Utilization percentage (GPU core load)
        util = nvmlDeviceGetUtilizationRates(handle)
        state.gpu_utilization = util.gpu
        
        # 2. VRAM allocation stats (Convert bytes to GB)
        mem = nvmlDeviceGetMemoryInfo(handle)
        state.gpu_vram_used = mem.used / (1024**3)
        state.gpu_vram_total = mem.total / (1024**3)
        state.gpu_vram_pct = mem.used / mem.total if mem.total > 0 else 0.0
        
        # 3. Core Temperature in °C
        state.gpu_temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        
        # 4. Power usage (Convert milliwatts to Watts)
        try:
            power_mw = nvmlDeviceGetPowerUsage(handle)
            state.gpu_power_used = power_mw / 1000.0
        except Exception:
            state.gpu_power_used = 0.0
            
        # Refresh the UI widget to display new readings
        gpu_telemetry_widget.refresh()
            
    except Exception as e:
        state.add_console_log(f"[NVML-Telemetry] Error polling GPU statistics: {str(e)}")


@ui.refreshable
def gpu_telemetry_widget():
    """Renders a compact, color-coded dual-bar telemetry layout with detailed hover card."""
    if not state.gpu_telemetry_supported:
        return

    # Color threshold evaluations matching load states
    vram_color = 'emerald-500'
    if state.gpu_vram_pct > 0.85:
        vram_color = 'rose-500'
    elif state.gpu_vram_pct > 0.60:
        vram_color = 'amber-500'

    gpu_color = 'blue-400'
    if state.gpu_utilization > 85:
        gpu_color = 'rose-400'
    elif state.gpu_utilization > 50:
        gpu_color = 'amber-400'

    with ui.row().classes('items-center gap-2 bg-slate-700/30 px-2 py-1 rounded border border-slate-600/30 text-xs').style('height: 32px; cursor: help;'):
        # Dynamic Hover Breakdown Card
        with ui.tooltip().classes('bg-slate-950 text-slate-200 p-3 rounded-lg border border-slate-700 shadow-2xl gap-1 text-[11px] min-w-[210px]'):
            ui.label(state.gpu_name or "NVIDIA GPU").classes('font-bold text-blue-400 text-xs')
            ui.separator().classes('my-1 bg-slate-800')
            with ui.grid(columns=2).classes('w-full gap-x-2 gap-y-1 font-mono'):
                ui.label('Core Load:')
                ui.label(f"{state.gpu_utilization}%").classes('text-right font-bold')
                
                ui.label('VRAM Allocated:')
                ui.label(f"{state.gpu_vram_used:.1f} / {state.gpu_vram_total:.1f} GB").classes('text-right font-bold')
                
                ui.label('VRAM Pct:')
                ui.label(f"{int(state.gpu_vram_pct * 100)}%").classes('text-right font-bold')
                
                ui.label('Temperature:')
                ui.label(f"{state.gpu_temp}°C").classes('text-right font-bold')
                
                ui.label('Power Draw:')
                ui.label(f"{state.gpu_power_used:.1f}W / {state.gpu_power_limit:.1f}W").classes('text-right font-bold')

        # Thermometer display (always visible)
        with ui.row().classes('items-center gap-0.5'):
            ui.icon('thermostat', size='15px', color='orange-400')
            ui.label(f"{state.gpu_temp}°C").classes('font-mono font-black text-[11px] text-slate-300')

        # Vertical alignment of core and allocation micro bars
        with ui.column().classes('gap-1 w-14 justify-center py-0.5'):
            # Core GPU utilization (0.0 to 1.0 representation)
            ui.linear_progress(value=state.gpu_utilization / 100.0, color=gpu_color, show_value=False, size='4px').classes('rounded-full')
            # VRAM allocated saturation (0.0 to 1.0 representation)
            ui.linear_progress(value=state.gpu_vram_pct, color=vram_color, show_value=False, size='4px').classes('rounded-full')

# --- Header & Top Bar Navigation Layout ---
with ui.header(elevated=False).classes('bg-slate-800 text-white px-6 py-4 justify-between items-center'):
    with ui.row().classes('items-center gap-3'):
        ui.icon('auto_awesome', size='md').classes('text-blue-400')
        ui.label('ABI-Pipeline').classes('text-xl font-bold tracking-tight cursor-pointer').on('click', exit_to_portal)

    header_controls()

    with ui.row().classes('items-center gap-3'):
        # Launch Comfy Button (Visible only if local comfy path is populated)
        ui.button(
            'Launch Comfy', 
            icon='bolt', 
            on_click=launch_comfyui
        ).props('flat dense').classes('text-xs text-blue-400 hover:text-blue-300 font-bold px-2 py-1 bg-slate-700/50 rounded border border-slate-600/30') \
            .bind_visibility_from(app_settings, 'comfy_path', backward=lambda val: bool(val and val.strip()))
        
        # Launch LLM Button (Visible only if path is populated)
        ui.button(
            'Launch LLM',
            icon='psychology',
            on_click=launch_llm_host
        ).props('flat dense').classes('text-xs text-emerald-400 hover:text-emerald-300 font-bold px-2 py-1 bg-slate-700/50 rounded border border-slate-600/30') \
            .bind_visibility_from(app_settings, 'llm_launch_path', backward=lambda val: bool(val and val.strip()))
        
        # Free VRAM Button (Universal Shortcut)
        ui.button(
            'Free VRAM', 
            icon='memory', 
            on_click=free_all_memory
        ).props('flat dense').classes('text-xs text-rose-400 hover:text-rose-300 font-bold px-2 py-1 bg-slate-700/50 rounded border border-slate-600/30')

        # GPU Telemetry Indicator Widget
        gpu_telemetry_widget()

        with ui.button(icon='construction', color='slate-600') as tools_btn:
            tools_btn.classes('text-white text-sm capitalize rounded-lg')
            with ui.menu() as menu:
                ui.menu_item('LoRA Contact Sheets', on_click=lambda: open_tool('lora_contact_sheet'))
                ui.menu_item('Rerun Setup Wizard', on_click=onboarding_wizard.open)
                ui.menu_item('Clear Comfy Outputs (abi_*)', on_click=clear_comfy_dialog.open) \
                    .bind_visibility_from(app_settings, 'comfy_path', backward=lambda val: bool(val and val.strip()))
                ui.separator()
                ui.menu_item('Quit App', on_click=quit_app)
        ui.button(icon='settings', on_click=lambda: settings_modal.open()).props('flat round color=white')


# --- Main Application Page Frame ---
with ui.column().classes('w-full max-w-7xl mx-auto p-6 gap-6'):
    main_layout()


# --- Polling Update Refresh Timer ---
ui.timer(2.0, check_for_active_transcriptions)
ui.timer(3.0, update_gpu_telemetry)

# Launch our app
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="ABI-Pipeline", 
        favicon="static/favicon.png", 
        port=8910,
        # Restricts the reload watcher to monitor only active code directories.
        # This completely ignores the .venv folder, keeping startup under 3 seconds.
        uvicorn_reload_dirs="services, ui, static",
        uvicorn_reload_excludes="*.db, *.db-journal, workflows/**/*.json"
    )