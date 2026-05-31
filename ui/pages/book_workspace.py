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
    pass # Already registered globally


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
                
                # Back-fill approved column if it was never created
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
    """
    Scans directories once using fast system listings.
    Returns a fast lookup dictionary mapping {(chapter, scene): static_url_path_with_timestamp}.
    """
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
                            # Perfect cache busting: append modification time to target URL
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
        # Standard glob matches are safe during singular user action deletion
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
    
    # Primary lookup
    idx = text_lower.find(quote_lower)
    match_len = len(quote)
    
    # Fallback to fuzzy substring matches
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

    # Initialize shortcuts configuration in state
    if not hasattr(state, 'key_approve'):
        state.key_approve = 'a'
    if not hasattr(state, 'key_delete'):
        state.key_delete = 'd'
    if not hasattr(state, 'key_next'):
        state.key_next = 'f'
    if not hasattr(state, 'key_prev'):
        state.key_prev = 's'

    # Lazy-load prompts
    prompts = load_prompts_csv(project_name, book_name)

    # Scans directory ONCE at page load
    images_cache = get_book_images_cache(project_name, book_name)

    # Scoped book index
    if not hasattr(state, 'book_active_scene_idx'):
        state.book_active_scene_idx = 0

    # Cross-Platform Directory Opening Helper
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

    # Clean Header Panel with Explorer Button
    with ui.row().classes('w-full justify-between items-center mb-2 border-b pb-2'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('library_books', size='sm', color='slate-700')
            ui.label(f'Volume: {book_name}').classes('text-lg font-bold text-slate-800')
        ui.button(
            'Open Folder', 
            icon='folder_open', 
            on_click=lambda: open_directory(Path(f"./output/{project_name}/{book_name}"))
        ).props('flat dense').classes('text-xs text-slate-600')

    # Awaiting Generation Panel (Shown only when prompts are missing)
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
                # Stats Card
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

                # Instructions card
                with ui.column().classes('gap-3 bg-slate-50 p-4 rounded-xl border border-dashed'):
                    ui.label('Orchestration Guide').classes('text-xs font-bold text-slate-700 uppercase tracking-wide')
                    
                    # Step 1
                    with ui.row().classes('items-center gap-2 text-xs'):
                        ui.icon('check_circle' if has_transcript else 'radio_button_unchecked', color='emerald' if has_transcript else 'slate', size='16px')
                        ui.label('Step 1: Transcription').classes('font-bold ' + ('text-slate-400 line-through' if has_transcript else 'text-slate-700'))
                        
                    # Step 2
                    with ui.row().classes('items-center gap-2 text-xs'):
                        is_step_2_active = has_transcript and not prompts
                        icon_color = 'purple' if is_step_2_active else 'slate'
                        ui.icon('radio_button_checked' if is_step_2_active else 'radio_button_unchecked', color=icon_color, size='16px')
                        ui.label('Step 2: Generate Prompts').classes('font-bold ' + ('text-purple-700 animate-pulse' if is_step_2_active else 'text-slate-700'))
                        
                    # Step 3
                    with ui.row().classes('items-center gap-2 text-xs'):
                        ui.icon('radio_button_unchecked', color='slate', size='16px')
                        ui.label('Step 3: Render Images').classes('font-bold text-slate-500')
                        
            with ui.row().classes('w-full justify-end mt-2 border-t pt-3'):
                ui.label("Switch to the 'Dashboard' tab on the project workspace to run these steps.").classes('text-[11px] text-slate-500 italic')
        return

    # Proofreader Panel elements
    filter_mode = ui.select(
        options=["All", "Unapproved Only", "Missing Only"], 
        value="All"
    ).classes('hidden')
    
    view_mode = ui.select(
        options=["Theatre", "Gallery Grid"], 
        value="Theatre"
    ).classes('hidden')

    # --- Persistent References for In-place Updates ---
    large_image = None
    placeholder_frame = None
    floating_link = None
    quote_label = None
    prompt_input = None
    context_html = None
    header_label = None
    position_label = None
    badge_missing = None
    badge_approved = None
    badge_review = None
    
    # Mutable wrapper holding active row data reference
    active_row_ref = [None]

    # --- Internal Helper Layout Functions ---
    
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

    # --- Smooth In-Place Updater (No Page Shifting!) ---
    def update_active_scene_ui():
        filtered = get_filtered_prompts()
        if not filtered:
            return
        current_idx = min(max(0, state.book_active_scene_idx), len(filtered) - 1)
        state.book_active_scene_idx = current_idx
        current_scene = filtered[current_idx]
        active_row_ref[0] = current_scene
        
        try:
            ch = int(float(current_scene.get("chapter", "1")))
            sc = int(float(current_scene.get("scene", "1")))
        except ValueError:
            ch, sc = 1, 1
            
        img_url = images_cache.get((ch, sc))
        is_approved = current_scene.get("approved", "False").strip().lower() == "true"
        
        # In-place value updates
        if large_image:
            if img_url:
                large_image.set_source(img_url)
                large_image.visible = True
            else:
                large_image.visible = False
                
        if placeholder_frame:
            placeholder_frame.visible = not img_url
                
        if floating_link:
            if img_url:
                floating_link._props['href'] = img_url
                floating_link.update()
                floating_link.visible = True
            else:
                floating_link.visible = False
                
        if quote_label:
            quote_label.set_text(f'"{current_scene.get("quote", "")}"')
            
        if prompt_input:
            prompt_input.set_value(current_scene.get("prompt", ""))
            
        if context_html:
            context_html.set_content(find_quote_context(project_name, book_name, current_scene.get("quote", "")))
            
        if header_label:
            header_label.set_text(f"Chapter {current_scene.get('chapter')}, Scene {current_scene.get('scene')}")
            
        if position_label:
            position_label.set_text(f"Position: {current_idx + 1} of {len(filtered)}")
            
        # Update badges
        if badge_missing:
            badge_missing.visible = not img_url
        if badge_approved:
            badge_approved.visible = bool(img_url and is_approved)
        if badge_review:
            badge_review.visible = bool(img_url and not is_approved)
            
        # Refresh filmstrip card cleanly without shifting page viewport
        render_filmstrip.refresh(filtered, current_idx)

    def next_scene():
        filtered = get_filtered_prompts()
        if not filtered:
            return
        state.book_active_scene_idx = min(state.book_active_scene_idx + 1, len(filtered) - 1)
        update_active_scene_ui()

    def prev_scene():
        state.book_active_scene_idx = max(state.book_active_scene_idx - 1, 0)
        update_active_scene_ui()

    def approve_current():
        filtered = get_filtered_prompts()
        if not filtered or state.book_active_scene_idx >= len(filtered):
            return
        row = filtered[state.book_active_scene_idx]
        row["approved"] = "True"
        save_prompts_csv(project_name, book_name, prompts)
        ui.notify(f"Ch {row.get('chapter')}, Sc {row.get('scene')} Marked Approved!", type="positive", timeout=1.0)
        next_scene()

    def delete_current():
        filtered = get_filtered_prompts()
        if not filtered or state.book_active_scene_idx >= len(filtered):
            return
        row = filtered[state.book_active_scene_idx]
        row["approved"] = "False"
        save_prompts_csv(project_name, book_name, prompts)
        
        was_deleted = delete_scene_image_file(project_name, book_name, row.get("chapter", "1"), row.get("scene", "1"))
        
        # Instantly remove file from cached directory mapping
        try:
            ch = int(float(row.get("chapter", "1")))
            sc = int(float(row.get("scene", "1")))
            images_cache.pop((ch, sc), None)
        except ValueError:
            pass
            
        if was_deleted:
            ui.notify(f"Deleted image file for Ch {row.get('chapter')}, Sc {row.get('scene')}!", type="warning", timeout=1.0)
        else:
            ui.notify(f"Ch {row.get('chapter')}, Sc {row.get('scene')} file was already missing.", type="info", timeout=1.0)
        
        next_scene()

    # --- Isolated Filmstrip Refresh Container ---
    
    @ui.refreshable
    def render_filmstrip(filtered_list: list, current_idx: int):
        start = max(0, current_idx - 2)
        end = min(len(filtered_list), current_idx + 4)
        
        with ui.row().classes('w-full gap-2 items-center overflow-x-auto pb-2 flex-nowrap mt-4 justify-center bg-slate-50 p-2.5 rounded-lg border border-dashed'):
            for idx in range(start, end):
                item = filtered_list[idx]
                is_active = (idx == current_idx)
                is_approved = item.get("approved", "False").strip().lower() == "true"
                
                try:
                    ch = int(float(item.get("chapter", "1")))
                    sc = int(float(item.get("scene", "1")))
                except ValueError:
                    ch, sc = 1, 1
                    
                thumb_url = images_cache.get((ch, sc))
                
                if is_active:
                    border_style = "border-blue-500 bg-blue-50 text-blue-700 ring-2 ring-blue-400"
                elif not thumb_url:
                    border_style = "border-red-300 text-red-500 bg-white hover:bg-slate-50"
                elif is_approved:
                    border_style = "border-emerald-300 text-emerald-600 bg-white hover:bg-slate-50"
                else:
                    border_style = "border-amber-300 text-amber-600 bg-white hover:bg-slate-50"
                    
                def set_active(idx_val=idx):
                    state.book_active_scene_idx = idx_val
                    update_active_scene_ui()
                    
                with ui.card().classes(f'w-24 border p-1 rounded cursor-pointer flex-shrink-0 transition-all {border_style}') \
                        .on('click', lambda _, idx_val=idx: set_active(idx_val)):
                    if thumb_url:
                        ui.image(thumb_url).classes('w-full h-14 rounded object-cover mb-1')
                    else:
                        with ui.column().classes('w-full h-14 items-center justify-center bg-slate-100 rounded text-slate-400 mb-1'):
                            ui.icon('photo_library', size='xs')
                    ui.label(f"Ch {item.get('chapter')}, Sc {item.get('scene')}").classes('text-[9px] font-bold text-center truncate w-full')

    # --- Theatre View Rendering ---
    
    def render_theatre_view(filtered_list: list, current_idx: int):
        nonlocal large_image, placeholder_frame, floating_link, quote_label, prompt_input, context_html
        nonlocal header_label, position_label, badge_missing, badge_approved, badge_review
        
        current_scene = filtered_list[current_idx]
        active_row_ref[0] = current_scene
        
        try:
            ch = int(float(current_scene.get("chapter", "1")))
            sc = int(float(current_scene.get("scene", "1")))
        except ValueError:
            ch, sc = 1, 1
            
        img_url = images_cache.get((ch, sc))
        is_approved = current_scene.get("approved", "False").strip().lower() == "true"
        
        with ui.grid(columns='1fr 350px').classes('w-full gap-6 items-start'):
            # LEFT: Image Viewport
            with ui.column().classes('w-full gap-2 items-center'):
                with ui.card().classes('w-full aspect-square border rounded-xl overflow-hidden shadow-sm flex items-center justify-center bg-slate-900 relative p-0'):
                    with ui.column().classes('items-center justify-center text-slate-400 w-full h-full') as placeholder_frame:
                        ui.icon('photo_library', size='lg').classes('mb-2 text-slate-500 animate-pulse')
                        ui.label("Awaiting ComfyUI Generation...").classes('text-xs font-semibold text-slate-400')
                        ui.label("Scene will load automatically once rendered.").classes('text-[10px] text-slate-500')
                        
                    large_image = ui.image(img_url or "").classes('w-full h-full object-contain')
                    large_image.visible = bool(img_url)
                    placeholder_frame.visible = not img_url
                    
                    # Float overlay badges
                    badge_missing = ui.badge("Missing", color="red").classes('absolute top-4 left-4 font-bold text-xs')
                    badge_missing.visible = not img_url
                    
                    badge_approved = ui.badge("Approved", color="emerald").classes('absolute top-4 left-4 font-bold text-xs')
                    badge_approved.visible = bool(img_url and is_approved)
                    
                    badge_review = ui.badge("Needs Review", color="amber").classes('absolute top-4 left-4 font-bold text-xs')
                    badge_review.visible = bool(img_url and not is_approved)
                    
                    with ui.link(target=img_url or "", new_tab=True).classes('absolute top-3 right-3') as floating_link:
                        ui.button(icon='zoom_in').props('flat fab-mini color=white').classes('bg-slate-900/60 hover:bg-slate-900/80')
                    floating_link.visible = bool(img_url)
                        
                # Action Row
                with ui.row().classes('w-full justify-between items-center bg-slate-100 p-3 rounded-lg border border-dashed mt-2'):
                    ui.button(
                        'Prev', 
                        icon='chevron_left', 
                        on_click=prev_scene
                    ).props('flat dense').classes('text-xs font-bold text-slate-600')
                    
                    with ui.row().classes('gap-3'):
                        ui.button(
                            'Delete Image', 
                            icon='delete', 
                            on_click=delete_current
                        ).classes('bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold px-4')
                        
                        ui.button(
                            'Approve', 
                            icon='check', 
                            on_click=approve_current
                        ).classes('bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold px-5')
                        
                    ui.button(
                        'Next', 
                        icon='chevron_right', 
                        on_click=next_scene
                    ).props('flat dense').classes('text-xs font-bold text-slate-600')
                    
                # Embed isolated refreshed filmstrip
                render_filmstrip(filtered_list, current_idx)

            # RIGHT: Sidebar Details Panel
            with ui.card().classes('w-full border p-4 shadow-sm bg-white gap-4'):
                with ui.row().classes('w-full justify-between items-center border-b pb-2'):
                    with ui.column().classes('gap-0'):
                        header_label = ui.label(f"Chapter {current_scene.get('chapter')}, Scene {current_scene.get('scene')}").classes('text-sm font-bold text-slate-800')
                        position_label = ui.label(f"Position: {current_idx + 1} of {len(filtered_list)}").classes('text-[10px] text-slate-400')
                        
                with ui.column().classes('w-full gap-2 bg-slate-50 p-3 rounded border border-dashed'):
                    ui.label("Target Narration Quote").classes('text-[9px] font-black text-slate-400 uppercase tracking-wider')
                    quote_label = ui.label(f'"{current_scene.get("quote", "")}"').classes('text-xs italic text-slate-700 leading-relaxed font-serif')
                    
                prompt_input = ui.textarea(
                    label="Style-Ready Visual Prompt",
                    value=current_scene.get("prompt", "")
                ).classes('w-full h-36 text-xs leading-relaxed').props('outlined')
                
                prompt_input.on('blur', lambda: update_prompt_text(prompt_input.value, active_row_ref[0]))
                
                with ui.expansion('Narrative Context (transcript.txt)').classes('w-full border rounded bg-slate-50 text-xs'):
                    context_html = ui.html(
                        find_quote_context(project_name, book_name, current_scene.get("quote", ""))
                    ).classes('p-3 leading-relaxed text-slate-700 bg-white font-serif')

    # --- Gallery Grid View Rendering ---
    
    # Memory dictionary to cache DOM references of grid cards for fluid in-place updates
    grid_card_references: Dict[tuple, Dict[str, Any]] = {}

    def render_grid_view(filtered_list: list):
        """Renders the gallery layout once, establishing DOM references for smooth updates."""
        grid_card_references.clear()
        
        def switch_to_theatre(idx_val: int):
            view_mode.value = "Theatre"
            state.book_active_scene_idx = idx_val
            render_content.refresh()
            
        with ui.grid(columns='repeat(auto-fill, minmax(180px, 1fr))').classes('w-full gap-4'):
            for idx, item in enumerate(filtered_list):
                is_approved = item.get("approved", "False").strip().lower() == "true"
                
                try:
                    ch = int(float(item.get("chapter", "1")))
                    sc = int(float(item.get("scene", "1")))
                except ValueError:
                    ch, sc = 1, 1
                    
                img_url = images_cache.get((ch, sc))
                
                # Standardize initial styles
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
                        .on('click', lambda _, idx_val=idx: switch_to_theatre(idx_val)) as card_el:
                    
                    # Both elements are built once; we toggle visibility in-place
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
                        
                # Store references for fluid updates
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
            
        current_idx = min(max(0, state.book_active_scene_idx), len(filtered) - 1)
        state.book_active_scene_idx = current_idx
        
        if view_mode.value == "Theatre":
            render_theatre_view(filtered, current_idx)
        else:
            render_grid_view(filtered)


    # --- Centralized Key Bindings Handler ---
    
    def handle_key(e):
        if state.active_book_id is None:
            return
        if view_mode.value != "Theatre":
            return
            
        if e.action.keydown and not e.action.repeat:
            key_name = e.key.name.lower()
            
            if key_name == state.key_approve:
                approve_current()
            elif key_name == state.key_delete:
                delete_current()
            elif key_name == state.key_next:
                next_scene()
            elif key_name == state.key_prev:
                prev_scene()

    # Register key bindings
    ui.keyboard(on_key=handle_key)

    # --- Pipeline Restart Control Action ---
    
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

    # --- Top Interface Toolbar ---
    
    with ui.row().classes('w-full justify-between items-center bg-white p-3 border rounded-xl shadow-xs mb-4'):
        with ui.row().classes('items-center gap-4'):
            # View toggle (Bi-directionally bound to sync top buttons on grid-click!)
            ui.toggle(
                options=["Theatre", "Gallery Grid"],
                on_change=lambda e: render_content.refresh()
            ).classes('text-xs').bind_value(view_mode, 'value')
            
            # Filtering selector
            ui.select(
                options=["All", "Unapproved Only", "Missing Only"],
                label="Filter Scenes",
                on_change=lambda e: (setattr(filter_mode, 'value', e.value), render_content.refresh())
            ).classes('w-44 bg-white').props('outlined dense').bind_value_to(filter_mode, 'value')
            
            # Direct batch reboot action
            ui.button(
                'Restart Batch / Regen', 
                icon='refresh', 
                on_click=trigger_batch_restart
            ).classes('bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold px-4 h-10')

            # --- A3: Stop image generation button ---
            # Automatically appears if ComfyUI renders are currently running in the background
            def trigger_stop_rendering():
                stop_fn = getattr(state, 'stop_image_generation_cb', None)
                if stop_fn:
                    stop_fn(project.id)
                    ui.notify("Stop signal dispatched. Halting after active image completes...", type="warning")
                else:
                    ui.notify("Stop callback is not registered.", type="negative")

            ui.button(
                'Stop Rendering',
                icon='stop',
                on_click=trigger_stop_rendering
            ).classes('bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold px-4 h-10') \
             .bind_visibility_from(state, 'image_gen_active')
            
        # Shortcuts Reminder Label
        with ui.row().classes('items-center gap-2 bg-slate-50 px-3 py-1.5 rounded-lg border text-[11px] font-semibold text-slate-500'):
            ui.icon('keyboard', size='xs')
            ui.label(
                f"Shortcuts: [{state.key_approve.upper()}] Approve  |  "
                f"[{state.key_delete.upper()}] Delete  |  "
                f"[{state.key_next.upper()}] Next  |  "
                f"[{state.key_prev.upper()}] Prev"
            )

    # Initial render
    render_content()


    # --- Real-Time Background Image Pop-in Timer (Optimized In-Place!) ---
    
    last_file_count = [len(images_cache)]
    
    def update_grid_views_in_place():
        """Updates rendering frames, cards, and dots in-place with zero layout shifts."""
        for (ch, sc), ref in grid_card_references.items():
            img_url = images_cache.get((ch, sc))
            is_approved = ref["item"].get("approved", "False").strip().lower() == "true"
            
            # Formulate class lists dynamically
            if not img_url:
                border_style = "border-red-300 bg-red-50/10"
                dot_color = "bg-red-500"
            elif is_approved:
                border_style = "border-emerald-300 bg-emerald-50/10"
                dot_color = "bg-emerald-500"
            else:
                border_style = "border-amber-300 bg-amber-50/10"
                dot_color = "bg-amber-500"
                
            # Perform targeted, non-destructive WebSockets updates
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
            
            # Reload only the fast memory lookup dict
            nonlocal images_cache
            images_cache = get_book_images_cache(project_name, book_name)
            
            # Update active view using non-destructive, zero-shift refreshes
            if view_mode.value == "Theatre":
                update_active_scene_ui()
            else:
                update_grid_views_in_place()
            
    # Check for newly generated images every 3 seconds
    ui.timer(3.0, check_for_image_updates)