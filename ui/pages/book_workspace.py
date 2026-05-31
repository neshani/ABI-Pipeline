import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from nicegui import ui, app
from sqlmodel import Session
from database.connection import engine
from database.models import Book
from ui import state

# Ensure standard FastAPI static file serving is mounted once
try:
    app.add_static_files('/output_media', './output')
except Exception:
    pass


# --- Robust CSV Loading & Saving Utilities ---

def load_prompts_csv(project_name: str, book_name: str) -> List[Dict[str, Any]]:
    """Discovers and parses prompts.csv from output folders, self-healing missing approved columns."""
    csv_paths = [
        Path(f"./output/{project_name}/{book_name}/prompts.csv"),
        Path(f"./output/{project_name}/{book_name}_prompts.csv"),
        Path(f"./output/{project_name}/{book_name}/{book_name}_prompts.csv"),
        Path(f"./output/{project_name}/{project_name}_prompts.csv"),
        Path(f"./output/{project_name}_prompts.csv")
    ]
    
    csv_path = None
    for p in csv_paths:
        if p.exists():
            csv_path = p
            break
            
    if not csv_path or not csv_path.exists():
        return []
        
    rows = []
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            fieldnames = [name.strip().lower() if name else "" for name in reader.fieldnames]
            
            for row in reader:
                cleaned_row = {}
                for k, v in row.items():
                    if k:
                        cleaned_row[k.strip().lower()] = v.strip() if v else ""
                
                if "approved" not in cleaned_row:
                    cleaned_row["approved"] = "False"
                rows.append(cleaned_row)
    except Exception as e:
        print(f"[Proofreader-CSV] Error reading prompts: {e}")
        return []
        
    return rows


def save_prompts_csv(project_name: str, book_name: str, rows: List[Dict[str, Any]]) -> None:
    """Serializes rows back to prompts.csv using pipes, keeping the approved column structured."""
    target_path = Path(f"./output/{project_name}/{book_name}/prompts.csv")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not rows:
        return
        
    headers = ["chapter", "scene", "prompt", "quote", "approved"]
    
    for row in rows:
        for k in row.keys():
            if k not in headers and k:
                headers.append(k)
                
    try:
        with open(target_path, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter='|')
            writer.writeheader()
            for r in rows:
                cleaned_row = {h: r.get(h, "") for h in headers}
                writer.writerow(cleaned_row)
    except Exception as e:
        print(f"[Proofreader-CSV] Error saving prompts: {e}")


# --- High Performance Directory Caching & Parsing ---

def get_book_images_cache(project_name: str, book_name: str) -> Dict[tuple, str]:
    """Scans directories once. Returns a fast lookup dict mapping {(chapter, scene): static_url}."""
    cache = {}
    out_dirs = [
        (Path(f"./output/{project_name}/{book_name}/images"), f"/output_media/{project_name}/{book_name}/images"),
        (Path(f"./output/{project_name}/{book_name}"), f"/output_media/{project_name}/{book_name}")
    ]
    
    for d, url_prefix in out_dirs:
        if not d.exists():
            continue
        try:
            for filename in os.listdir(d):
                if filename.lower().endswith('.png'):
                    item_path = d / filename
                    stem, _ = os.path.splitext(filename)
                    parts = stem.split('_')
                    if len(parts) >= 2:
                        try:
                            ch = int(parts[0])
                            sc = int(parts[1])
                            mtime = int(item_path.stat().st_mtime)
                            cache[(ch, sc)] = f"{url_prefix}/{filename}?t={mtime}"
                        except ValueError:
                            pass
        except Exception as ex:
            print(f"[Proofreader-Cache] Error indexing folder {d}: {ex}")
            
    return cache


def delete_scene_image_file(project_name: str, book_name: str, chapter_str: str, scene_str: str) -> bool:
    """Removes standard image patterns from disk, preparing the scene for ComfyUI regenerations."""
    try:
        chapter = int(float(chapter_str))
        scene = int(float(scene_str))
    except ValueError:
        return False
        
    out_dirs = [
        Path(f"./output/{project_name}/{book_name}/images"),
        Path(f"./output/{project_name}/{book_name}")
    ]
    
    deleted = False
    for d in out_dirs:
        if not d.exists():
            continue
        matches = list(d.glob(f"{chapter:02d}_{scene:02d}_*.png")) or list(d.glob(f"{chapter:02d}_{scene:02d}.png"))
        for m in matches:
            try:
                m.unlink()
                deleted = True
            except Exception as e:
                print(f"[Proofreader] Error deleting file {m.name}: {e}")
    return deleted


# --- Lazy Context Finder ---

def find_quote_context(project_name: str, book_name: str, quote: str) -> str:
    """Lazy parses transcript.txt, extracting surrounding paragraphs to highlight narration context."""
    transcript_path = Path(f"./output/{project_name}/{book_name}/transcript.txt")
    if not transcript_path.exists():
        return "Transcript file not found on disk. Run Transcription to load context narration."
        
    try:
        text = transcript_path.read_text(encoding='utf-8')
    except Exception as e:
        return f"Error reading transcript: {str(e)}"
        
    if not quote:
        return "Narrative quote is empty."
        
    text_lower = text.lower()
    quote_lower = quote.lower()
    
    idx = text_lower.find(quote_lower)
    match_len = len(quote)
    
    if idx == -1 and len(quote_lower) > 25:
        idx = text_lower.find(quote_lower[:25])
        match_len = 25
        
    if idx == -1:
        return "Narrative segment context match could not be found inside transcript.txt."
        
    start_idx = max(0, idx - 400)
    end_idx = min(len(text), idx + match_len + 400)
    snippet = text[start_idx:end_idx]
    
    snippet_lower = snippet.lower()
    snippet_idx = snippet_lower.find(quote_lower[:match_len])
    
    if snippet_idx != -1:
        highlighted = (
            snippet[:snippet_idx] +
            '<mark class="bg-yellow-200 text-slate-900 font-semibold px-1 rounded">' +
            snippet[snippet_idx:snippet_idx + match_len] +
            '</mark>' +
            snippet[snippet_idx + match_len:]
        )
        return f"... {highlighted} ..."
        
    return f"... {snippet} ..."


# --- Main Proofing UI Component ---

def render_book_tabs(book_id: int):
    import platform
    import subprocess
    
    with Session(engine) as session:
        book = session.get(Book, book_id)
        if not book:
            ui.label("Book details not available.").classes('text-slate-400 text-sm')
            return
            
        from database.models import Project
        project = session.get(Project, book.project_id)
        if not project:
            ui.label("Project workspace details not available.").classes('text-slate-400 text-sm')
            return

    project_name = project.name
    book_name = book.name

    # Initialize shortcuts and key tracking in state
    if not hasattr(state, 'key_approve'):
        state.key_approve = 'a'
    if not hasattr(state, 'key_delete'):
        state.key_delete = 'd'
    if not hasattr(state, 'key_next'):
        state.key_next = 'f'
    if not hasattr(state, 'key_prev'):
        state.key_prev = 's'
    if not hasattr(state, 'book_active_chapter'):
        state.book_active_chapter = 1
    if not hasattr(state, 'book_active_scene'):
        state.book_active_scene = 1

    # Lazy-load prompts and scan directories once at load
    prompts = load_prompts_csv(project_name, book_name)
    images_cache = get_book_images_cache(project_name, book_name)

    if not hasattr(state, 'book_active_scene_idx'):
        state.book_active_scene_idx = 0

    # --- 1. NESTED HANDLERS DEFINED FIRST (Prevents UnboundLocalErrors) ---

    def open_directory(path: Path):
        abs_path = path.resolve()
        if not abs_path.exists():
            abs_path.mkdir(parents=True, exist_ok=True)
        try:
            if platform.system() == "Windows":
                os.startfile(abs_path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(abs_path)])
            else:
                subprocess.Popen(["xdg-open", str(abs_path)])
        except Exception as e:
            ui.notify(f"Failed to open directory: {str(e)}", type="negative")

    def get_filtered_prompts():
        filtered = []
        for p in prompts:
            is_approved = p.get("approved", "False").strip().lower() == "true"
            try:
                ch = int(float(p.get("chapter", "1")))
                sc = int(float(p.get("scene", "1")))
            except ValueError:
                ch, sc = 1, 1
            has_image = (ch, sc) in images_cache
            
            if filter_mode.value == "Unapproved Only":
                if has_image and not is_approved:
                    filtered.append(p)
            elif filter_mode.value == "Missing Only":
                if not has_image:
                    filtered.append(p)
            else:
                filtered.append(p)
        return filtered

    def update_prompt_text(val: str, row_dict: dict):
        if row_dict:
            row_dict["prompt"] = val.strip()
            save_prompts_csv(project_name, book_name, prompts)
            try:
                ch = int(float(row_dict.get("chapter", "1")))
                sc = int(float(row_dict.get("scene", "1")))
                if (ch, sc) in grid_card_references:
                    grid_card_references[(ch, sc)]["item"]["prompt"] = val.strip()
            except ValueError:
                pass

    def get_current_filtered_index(filtered_list: list) -> int:
        for idx, p in enumerate(filtered_list):
            try:
                ch = int(float(p.get("chapter", "1")))
                sc = int(float(p.get("scene", "1")))
            except ValueError:
                ch, sc = 1, 1
            if ch == state.book_active_chapter and sc == state.book_active_scene:
                return idx
        return 0

    def update_active_scene_ui():
        filtered = get_filtered_prompts()
        if not filtered:
            return
            
        current_scene = None
        for p in filtered:
            try:
                ch = int(float(p.get("chapter", "1")))
                sc = int(float(p.get("scene", "1")))
            except ValueError:
                ch, sc = 1, 1
            if ch == state.book_active_chapter and sc == state.book_active_scene:
                current_scene = p
                break
                
        # Fallback if the active item was filtered out (e.g. approved)
        if not current_scene:
            current_scene = filtered[0]
            try:
                state.book_active_chapter = int(float(current_scene.get("chapter", "1")))
                state.book_active_scene = int(float(current_scene.get("scene", "1")))
            except ValueError:
                state.book_active_chapter, state.book_active_scene = 1, 1
                
        active_row_ref[0] = current_scene
        current_idx = get_current_filtered_index(filtered)
        
        try:
            ch = int(float(current_scene.get("chapter", "1")))
            sc = int(float(current_scene.get("scene", "1")))
        except ValueError:
            ch, sc = 1, 1
            
        img_url = images_cache.get((ch, sc))
        is_approved = current_scene.get("approved", "False").strip().lower() == "true"
        
        # Update Modal Content In-place
        if modal_img_el:
            if img_url:
                modal_img_el.set_source(img_url)
                modal_img_el.visible = True
            else:
                modal_img_el.visible = False
                
        if modal_placeholder:
            modal_placeholder.visible = not img_url
            
        if modal_quote_el:
            modal_quote_el.set_text(f'"{current_scene.get("quote", "")}"')
            
        if modal_prompt_input:
            modal_prompt_input.set_value(current_scene.get("prompt", ""))
            
        if modal_context_html:
            modal_context_html.set_content(find_quote_context(project_name, book_name, current_scene.get("quote", "")))
            
        if modal_title_el:
            modal_title_el.set_text(f"Chapter {current_scene.get('chapter')}, Scene {current_scene.get('scene')}")
            
        if modal_subtitle_el:
            modal_subtitle_el.set_text(f"Review Scene: {current_idx + 1} of {len(filtered)}")
            
        # Modal Badge updates
        if modal_badge_missing:
            modal_badge_missing.visible = not img_url
        if modal_badge_approved:
            modal_badge_approved.visible = bool(img_url and is_approved)
        if modal_badge_review:
            modal_badge_review.visible = bool(img_url and not is_approved)

        # Highlight background Grid Card (Keeping highlight, but removed the scrap-scrolling JS call)
        for (grid_ch, grid_sc), ref in grid_card_references.items():
            ref["card"].classes(remove="ring-4 ring-blue-500 ring-offset-2")
            
        if (ch, sc) in grid_card_references:
            target_ref = grid_card_references[(ch, sc)]
            target_ref["card"].classes(add="ring-4 ring-blue-500 ring-offset-2")

    def next_scene():
        filtered = get_filtered_prompts()
        if not filtered:
            return
        idx = get_current_filtered_index(filtered)
        next_idx = min(idx + 1, len(filtered) - 1)
        next_scene_obj = filtered[next_idx]
        try:
            state.book_active_chapter = int(float(next_scene_obj.get("chapter", "1")))
            state.book_active_scene = int(float(next_scene_obj.get("scene", "1")))
        except ValueError:
            pass
        update_active_scene_ui()

    def prev_scene():
        filtered = get_filtered_prompts()
        if not filtered:
            return
        idx = get_current_filtered_index(filtered)
        prev_idx = max(idx - 1, 0)
        prev_scene_obj = filtered[prev_idx]
        try:
            state.book_active_chapter = int(float(prev_scene_obj.get("chapter", "1")))
            state.book_active_scene = int(float(prev_scene_obj.get("scene", "1")))
        except ValueError:
            pass
        update_active_scene_ui()

    def approve_current():
        filtered = get_filtered_prompts()
        if not filtered:
            return
            
        row = None
        for p in filtered:
            try:
                ch = int(float(p.get("chapter", "1")))
                sc = int(float(p.get("scene", "1")))
            except ValueError:
                ch, sc = 1, 1
            if ch == state.book_active_chapter and sc == state.book_active_scene:
                row = p
                break
                
        if not row:
            return
            
        row["approved"] = "True"
        save_prompts_csv(project_name, book_name, prompts)
        
        try:
            ch = int(float(row.get("chapter", "1")))
            sc = int(float(row.get("scene", "1")))
            if (ch, sc) in grid_card_references:
                grid_card_references[(ch, sc)]["item"]["approved"] = "True"
        except ValueError:
            pass
            
        update_grid_views_in_place()
        ui.notify(f"Ch {row.get('chapter')}, Sc {row.get('scene')} Approved!", type="positive", timeout=1.0)
        next_scene()

    def delete_current():
        filtered = get_filtered_prompts()
        if not filtered:
            return
            
        row = None
        for p in filtered:
            try:
                ch = int(float(p.get("chapter", "1")))
                sc = int(float(p.get("scene", "1")))
            except ValueError:
                ch, sc = 1, 1
            if ch == state.book_active_chapter and sc == state.book_active_scene:
                row = p
                break
                
        if not row:
            return
            
        row["approved"] = "False"
        save_prompts_csv(project_name, book_name, prompts)
        
        was_deleted = delete_scene_image_file(project_name, book_name, row.get("chapter", "1"), row.get("scene", "1"))
        
        try:
            ch = int(float(row.get("chapter", "1")))
            sc = int(float(row.get("scene", "1")))
            images_cache.pop((ch, sc), None)
            if (ch, sc) in grid_card_references:
                grid_card_references[(ch, sc)]["item"]["approved"] = "False"
        except ValueError:
            pass
            
        update_grid_views_in_place()
        
        if was_deleted:
            ui.notify(f"Deleted image for Ch {row.get('chapter')}, Sc {row.get('scene')}!", type="warning", timeout=1.0)
        else:
            ui.notify(f"Ch {row.get('chapter')}, Sc {row.get('scene')} file was already missing.", type="info", timeout=1.0)
        
        next_scene()

    def handle_modal_close():
        pass

    async def trigger_batch_restart():
        start_fn = getattr(state, 'start_image_generation_cb', None)
        if not start_fn:
            ui.notify("Pipeline process control callbacks are not fully registered in state.", type="negative")
            return

        if state.project_status == "Rendering Images" or state.image_gen_active:
            ui.notify("Waiting for current image to finish rendering before restarting...", type="info", timeout=2.0)
            state.cancel_image_gen_flag = True
            
            while state.image_gen_active:
                await asyncio.sleep(0.5)
                
        state.cancel_image_gen_flag = False
        state.image_gen_active = False
        start_fn(project.id)

    def trigger_stop_rendering():
        stop_fn = getattr(state, 'stop_image_generation_cb', None)
        if stop_fn:
            stop_fn(project.id)
            ui.notify("Stop signal dispatched. Halting after active image completes...", type="warning")
        else:
            ui.notify("Stop callback is not registered.", type="negative")

    def update_grid_views_in_place():
        """Updates rendering frames, cards, and dots in-place with zero layout shifts."""
        for (ch, sc), ref in grid_card_references.items():
            img_url = images_cache.get((ch, sc))
            is_approved = ref["item"].get("approved", "False").strip().lower() == "true"
            has_image = bool(img_url)
            
            # Determine if this card matches the active filter criteria
            should_be_visible = True
            if filter_mode.value == "Unapproved Only":
                # Show only if it has a rendered image but hasn't been approved yet
                should_be_visible = bool(has_image and not is_approved)
            elif filter_mode.value == "Missing Only":
                # Show only if it is missing its image file
                should_be_visible = not has_image
                
            ref["card"].visible = should_be_visible
            
            # If the card is filtered out, skip heavy element manipulation to save cycles
            if not should_be_visible:
                continue
            
            if not img_url:
                border_style = "border-red-300 bg-red-50/10"
                dot_color = "bg-red-500"
            elif is_approved:
                border_style = "border-emerald-300 bg-emerald-50/10"
                dot_color = "bg-emerald-500"
            else:
                border_style = "border-amber-300 bg-amber-50/10"
                dot_color = "bg-amber-500"
                
            ref["card"].classes(replace=f"border rounded-lg shadow-sm p-2 cursor-pointer hover:shadow-md transition-all {border_style}")
            ref["dot"].classes(replace=f"w-2 h-2 rounded-full {dot_color}")
            
            if img_url:
                ref["image"].set_source(img_url)
                ref["image"].visible = True
                ref["placeholder"].visible = False
            else:
                ref["image"].visible = False
                ref["placeholder"].visible = True

    def check_for_image_updates():
        if state.active_book_id is None:
            return
            
        img_dir = Path(f"./output/{project_name}/{book_name}/images")
        parent_dir = Path(f"./output/{project_name}/{book_name}")
        
        count = 0
        if img_dir.exists():
            count += len(os.listdir(img_dir))
        if parent_dir.exists():
            count += len(os.listdir(parent_dir))
            
        if count != last_file_count[0]:
            last_file_count[0] = count
            
            nonlocal images_cache
            images_cache = get_book_images_cache(project_name, book_name)
            
            update_grid_views_in_place()
            if theater_dialog.value:
                update_active_scene_ui()

    # --- 2. LAYOUT RENDERING COMPONENT DECLARATIONS ---

    # Awaiting Generation Panel (Only loaded if prompts are completely missing)
    if not prompts:
        transcript_path = Path(f"./output/{project_name}/{book_name}/transcript.txt")
        has_transcript = transcript_path.exists()
        char_count = 0
        word_count = 0
        if has_transcript:
            try:
                text_content = transcript_path.read_text(encoding="utf-8")
                char_count = len(text_content)
                word_count = len(text_content.split())
            except Exception:
                pass

        with ui.card().classes('w-full border p-6 shadow-sm bg-white gap-4'):
            with ui.row().classes('items-center gap-2 border-b pb-3 w-full'):
                ui.icon('pending_actions', size='md', color='amber-500')
                with ui.column().classes('gap-0'):
                    ui.label('Awaiting Generation Pipeline').classes('text-base font-bold text-slate-800')
                    ui.label('Complete the initial setup phases to start image proofing.').classes('text-xs text-slate-500')
            
            with ui.grid(columns='1fr 1fr').classes('w-full gap-4'):
                with ui.column().classes('gap-3 bg-slate-50 p-4 rounded-xl border border-dashed'):
                    ui.label('Volume Statistics').classes('text-xs font-bold text-slate-700 uppercase tracking-wide')
                    
                    with ui.row().classes('items-center justify-between w-full text-xs'):
                        ui.label('Transcript File:').classes('text-slate-500')
                        if has_transcript:
                            ui.badge('Found', color='emerald').classes('px-2 py-0.5 text-[10px]')
                        else:
                            ui.badge('Missing', color='red').classes('px-2 py-0.5 text-[10px]')
                            
                    with ui.row().classes('items-center justify-between w-full text-xs'):
                        ui.label('Character Count:').classes('text-slate-500')
                        ui.label(f"{char_count:,} characters").classes('font-bold text-slate-700')
                        
                    with ui.row().classes('items-center justify-between w-full text-xs'):
                        ui.label('Word Count:').classes('text-slate-500')
                        ui.label(f"{word_count:,} words").classes('font-bold text-slate-700')

                    with ui.row().classes('items-center justify-between w-full text-xs'):
                        ui.label('Estimated Scenes:').classes('text-slate-500')
                        est_scenes = max(1, char_count // 1500) if char_count > 0 else 0
                        ui.label(f"~ {est_scenes} scenes").classes('font-bold text-slate-700')

                with ui.column().classes('gap-3 bg-slate-50 p-4 rounded-xl border border-dashed'):
                    ui.label('Orchestration Guide').classes('text-xs font-bold text-slate-700 uppercase tracking-wide')
                    
                    with ui.row().classes('items-center gap-2 text-xs'):
                        ui.icon('check_circle' if has_transcript else 'radio_button_unchecked', color='emerald' if has_transcript else 'slate', size='16px')
                        ui.label('Step 1: Transcription').classes('font-bold ' + ('text-slate-400 line-through' if has_transcript else 'text-slate-700'))
                        
                    with ui.row().classes('items-center gap-2 text-xs'):
                        is_step_2_active = has_transcript and not prompts
                        icon_color = 'purple' if is_step_2_active else 'slate'
                        ui.icon('radio_button_checked' if is_step_2_active else 'radio_button_unchecked', color=icon_color, size='16px')
                        ui.label('Step 2: Generate Prompts').classes('font-bold ' + ('text-purple-700 animate-pulse' if is_step_2_active else 'text-slate-700'))
                        
                    with ui.row().classes('items-center gap-2 text-xs'):
                        ui.icon('radio_button_unchecked', color='slate', size='16px')
                        ui.label('Step 3: Render Images').classes('font-bold text-slate-500')
                        
            with ui.row().classes('w-full justify-end mt-2 border-t pt-3'):
                ui.label("Switch to the 'Dashboard' tab on the project workspace to run these steps.").classes('text-[11px] text-slate-500 italic')
        return

    # Hidden filter reference
    filter_mode = ui.select(
        options=["All", "Unapproved Only", "Missing Only"], 
        value="All"
    ).classes('hidden')

    # --- Top Interface Toolbar (Placed at the TOP of the Workspace) ---
    with ui.row().classes('w-full justify-between items-center bg-white p-3 border rounded-xl shadow-xs mb-4'):
        with ui.row().classes('items-center gap-4'):
            # Filtering selector
            ui.select(
                options=["All", "Unapproved Only", "Missing Only"],
                label="Filter Scenes",
                on_change=lambda e: (setattr(filter_mode, 'value', e.value), render_content.refresh())
            ).classes('w-44 bg-white').props('outlined dense').bind_value_to(filter_mode, 'value')
            
            # Direct batch reboot action (Hoisted trigger_batch_restart is fully bound safely!)
            ui.button(
                'Restart Batch / Regen', 
                icon='refresh', 
                on_click=trigger_batch_restart
            ).classes('bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold px-4 h-10')

            # Stop image generation button (appears dynamically if generation active)
            ui.button(
                'Stop Rendering',
                icon='stop',
                on_click=trigger_stop_rendering
            ).classes('bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold px-4 h-10') \
             .bind_visibility_from(state, 'image_gen_active')
            
        # Shortcuts Reminder Label (Active inside theater modal)
        with ui.row().classes('items-center gap-2 bg-slate-50 px-3 py-1.5 rounded-lg border text-[11px] font-semibold text-slate-500'):
            ui.icon('keyboard', size='xs')
            ui.label(
                f"Shortcuts: [{state.key_approve.upper()}] Approve  |  "
                f"[{state.key_delete.upper()}] Delete  |  "
                f"[{state.key_next.upper()}] Next  |  "
                f"[{state.key_prev.upper()}] Prev"
            )

    # --- Persistent References for Modal In-place Updates ---
    modal_img_el = None
    modal_placeholder = None
    modal_quote_el = None
    modal_prompt_input = None
    modal_context_html = None
    modal_title_el = None
    modal_subtitle_el = None
    
    modal_badge_missing = None
    modal_badge_approved = None
    modal_badge_review = None
    
    active_row_ref = [None]
    grid_card_references: Dict[tuple, Dict[str, Any]] = {}

    # --- High Performance Theater Modal ---
    with ui.dialog() as theater_dialog:
        # Tighter card padding (p-4), maximum height (90vh), and full-width stretch to prioritize image size
        with ui.card().classes('w-full max-w-[95vw] lg:max-w-7xl h-[90vh] p-4 rounded-xl bg-white flex flex-col items-stretch overflow-hidden gap-0'):
            
            # Left: Full-Height Image Viewport (1fr), Right: Informative Sidebar (380px)
            with ui.grid(columns='1fr 380px').classes('w-full h-full gap-4 items-stretch overflow-hidden min-h-0'):
                
                # LEFT IMAGE VIEWPORT (Occupies 100% of vertical height, completely maximized)
                with ui.column().classes('w-full h-full justify-center min-h-0 relative'):
                    with ui.card().classes('w-full h-full border rounded-xl overflow-hidden shadow-sm flex items-center justify-center bg-slate-900 relative p-0 m-0'):
                        modal_placeholder = ui.column().classes('items-center justify-center text-slate-400 w-full h-full')
                        with modal_placeholder:
                            ui.icon('photo_library', size='lg').classes('mb-2 text-slate-500 animate-pulse')
                            ui.label("Awaiting ComfyUI Generation...").classes('text-xs font-semibold text-slate-400')
                            
                        # High-performance fit=contain representation
                        modal_img_el = ui.image("").classes('w-full h-full bg-transparent').props('fit=contain')
                        
                        # overlays
                        modal_badge_missing = ui.badge("Missing", color="red").classes('absolute top-4 left-4 font-bold text-xs')
                        modal_badge_approved = ui.badge("Approved", color="emerald").classes('absolute top-4 left-4 font-bold text-xs')
                        modal_badge_review = ui.badge("Needs Review", color="amber").classes('absolute top-4 left-4 font-bold text-xs')
                        
                # RIGHT DETAILS COLUMN (Holds header details, action grid, prompt inputs)
                with ui.column().classes('w-full h-full gap-4 overflow-y-auto min-h-0 flex-nowrap pr-1'):
                    
                    # Consolidated Top Header & Close Row with hover shortcut discovery helper
                    with ui.row().classes('w-full items-center justify-between border-b pb-2 flex-shrink-0'):
                        with ui.column().classes('gap-0'):
                            modal_title_el = ui.label("").classes('text-base font-bold text-slate-800 leading-none')
                            modal_subtitle_el = ui.label("").classes('text-[11px] text-slate-400 mt-1')
                        with ui.row().classes('items-center gap-1'):
                            # Help Button with hovering keyboard shortcuts tooltip
                            with ui.button(icon='help_outline').props('flat round dense').classes('text-slate-400'):
                                with ui.tooltip().classes('bg-slate-800 text-white text-xs p-3 rounded-lg gap-1 flex flex-col shadow-lg'):
                                    ui.label('Keyboard Shortcuts').classes('font-bold border-b pb-1 text-blue-400')
                                    ui.label(f'[{state.key_approve.upper()}] Approve Scene')
                                    ui.label(f'[{state.key_delete.upper()}] Delete Image')
                                    ui.label(f'[{state.key_next.upper()}] Next Scene')
                                    ui.label(f'[{state.key_prev.upper()}] Prev Scene')
                            ui.button(icon='close', on_click=theater_dialog.close).props('flat round dense').classes('text-slate-400')
                    
                    # Condensed Navigation and State Actions
                    with ui.column().classes('w-full gap-2 bg-slate-100 p-3 rounded-lg border border-dashed flex-shrink-0'):
                        with ui.row().classes('w-full gap-2 items-center justify-between'):
                            ui.button('Prev', icon='chevron_left', on_click=prev_scene).props('flat dense').classes('text-xs font-bold text-slate-600 flex-1 py-1.5 bg-white border rounded')
                            ui.button('Next', icon='chevron_right', on_click=next_scene).props('flat dense').classes('text-xs font-bold text-slate-600 flex-1 py-1.5 bg-white border rounded')
                        with ui.row().classes('w-full gap-2'):
                            ui.button('Delete Image', icon='delete', on_click=delete_current).classes('bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold flex-1 py-2')
                            ui.button('Approve', icon='check', on_click=approve_current).classes('bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold flex-1 py-2')

                    # Target Narration Quote
                    with ui.column().classes('w-full gap-2 bg-slate-50 p-3 rounded border border-dashed flex-shrink-0'):
                        ui.label("Target Narration Quote").classes('text-[9px] font-black text-slate-400 uppercase tracking-wider')
                        modal_quote_el = ui.label("").classes('text-xs italic text-slate-700 leading-relaxed font-serif')
                        
                    # Style-Ready Visual Prompt Textarea
                    modal_prompt_input = ui.textarea(
                        label="Style-Ready Visual Prompt"
                    ).classes('w-full h-32 text-xs leading-relaxed flex-shrink-0').props('outlined')
                    
                    modal_prompt_input.on('blur', lambda: update_prompt_text(modal_prompt_input.value, active_row_ref[0]))
                    
                    # Narrative Context expansion panel
                    with ui.expansion('Narrative Context (transcript.txt)').classes('w-full border rounded bg-slate-50 text-xs flex-shrink-0'):
                        modal_context_html = ui.html("").classes('p-3 leading-relaxed text-slate-700 bg-white font-serif')

    theater_dialog.on('close', handle_modal_close)

    # Keyboard shortcut listener (Fires only when the active theater dialog is open, and ignores inputs naturally)
    def handle_key(e):
        if not theater_dialog.value:
            return
        if e.action.keydown and not e.action.repeat:
            k = e.key.name.lower()
            if k == state.key_approve:
                approve_current()
            elif k == state.key_delete:
                delete_current()
            elif k == state.key_next:
                next_scene()
            elif k == state.key_prev:
                prev_scene()

    ui.keyboard(on_key=handle_key)

    # --- Gallery Grid View Rendering ---
    def render_grid_view(filtered_list: list):
        """Renders the gallery layout once, establishing DOM references for smooth updates."""
        grid_card_references.clear()
        
        def launch_theater(ch_val: int, sc_val: int):
            state.book_active_chapter = ch_val
            state.book_active_scene = sc_val
            theater_dialog.open()
            update_active_scene_ui()
            
        with ui.grid(columns='repeat(auto-fill, minmax(180px, 1fr))').classes('w-full gap-4'):
            for idx, item in enumerate(filtered_list):
                is_approved = item.get("approved", "False").strip().lower() == "true"
                
                try:
                    ch = int(float(item.get("chapter", "1")))
                    sc = int(float(item.get("scene", "1")))
                except ValueError:
                    ch, sc = 1, 1
                    
                img_url = images_cache.get((ch, sc))
                
                if not img_url:
                    border_style = "border-red-300 bg-red-50/10"
                    status_color = "bg-red-500"
                elif is_approved:
                    border_style = "border-emerald-300 bg-emerald-50/10"
                    status_color = "bg-emerald-500"
                else:
                    border_style = "border-amber-300 bg-amber-50/10"
                    status_color = "bg-amber-500"
                    
                with ui.card().classes(f'border rounded-lg shadow-sm p-2 cursor-pointer hover:shadow-md transition-all {border_style}') \
                        .on('click', lambda _, ch_val=ch, sc_val=sc: launch_theater(ch_val, sc_val)) as card_el:
                    
                    img_el = ui.image(img_url or "").classes('w-full aspect-square rounded object-cover border')
                    img_el.visible = bool(img_url)
                    
                    placeholder_el = ui.column().classes('w-full aspect-square items-center justify-center bg-slate-100 rounded border border-dashed text-slate-400')
                    with placeholder_el:
                        ui.icon('photo_library', size='sm')
                        ui.label('Missing').classes('text-[9px]')
                    placeholder_el.visible = not img_url
                            
                    with ui.row().classes('w-full justify-between items-center mt-1 px-1'):
                        ui.label(f"Ch {item.get('chapter')}, Sc {item.get('scene')}").classes('text-[10px] font-bold text-slate-700')
                        dot_el = ui.element('div').classes(f'w-2 h-2 rounded-full {status_color}')
                        
                grid_card_references[(ch, sc)] = {
                    "card": card_el,
                    "image": img_el,
                    "placeholder": placeholder_el,
                    "dot": dot_el,
                    "item": item
                }

    # --- Parent Workspace Loader ---
    @ui.refreshable
    def render_content():
        filtered = get_filtered_prompts()
        if not filtered:
            with ui.column().classes('w-full items-center justify-center p-12 text-slate-400 border border-dashed rounded-xl bg-slate-50'):
                ui.icon('info', size='lg', color='slate-300')
                ui.label("No scenes match your active filter.").classes('text-sm text-center font-semibold')
            return
            
        render_grid_view(filtered)

    # Initial render
    render_content()

    # --- Real-Time Background Image Pop-in Timer (Optimized In-Place!) ---
    last_file_count = [len(images_cache)]
    
    # Check for newly generated images every 3 seconds
    ui.timer(3.0, check_for_image_updates)