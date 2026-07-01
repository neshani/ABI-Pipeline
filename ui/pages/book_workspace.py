import csv
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from nicegui import ui, app
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Book, Project, Character, CharacterAlias
from services.character_manager import compile_character_visual_prompt, save_project_characters_to_json
from ui.pages.project.characters_tab import get_character_frequency_map
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

        # Sync update directly into SQLite database cache to maintain consistency
        from services.sync_engine import sync_prompts_csv_to_db_cache
        with Session(engine) as session:
            book = session.exec(select(Book).where(Book.name == book_name)).first()
            if book:
                sync_prompts_csv_to_db_cache(book.id, session)
                session.commit()
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
    
    # Establish load-time static timestamp if none exists in state
    load_time = getattr(state, 'workspace_load_time', 0)
    if not load_time:
        load_time = int(time.time())
        state.workspace_load_time = load_time
        
    custom_versions = getattr(state, 'custom_image_timestamps', {})
    
    for d, url_prefix in out_dirs:
        if not d.exists():
            continue
        try:
            for filename in os.listdir(d):
                if filename.lower().endswith('.png'):
                    stem, _ = os.path.splitext(filename)
                    parts = stem.split('_')
                    if len(parts) >= 2:
                        try:
                            ch = int(parts[0])
                            sc = int(parts[1])
                            # Use custom targeted coordinate timestamp if present, otherwise fallback to book load time
                            t_val = custom_versions.get((ch, sc), load_time)
                            cache[(ch, sc)] = f"{url_prefix}/{filename}?t={t_val}"
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
                
    # Register dynamic version bump timestamp for this coordinate to cache-bust it on generation
    if not hasattr(state, 'custom_image_timestamps'):
        state.custom_image_timestamps = {}
    state.custom_image_timestamps[(chapter, scene)] = int(time.time() * 1000)
                
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
    
    # Coordinate cache to prevent resetting the user's active typing/prompt edit input
    loaded_scene_coords = [-1, -1]

    def extract_prompt_character_tags(prompt_text: str) -> List[str]:
        """Parses bracketed tags from the prompt text area."""
        if not prompt_text:
            return []
        bracket_regex = re.compile(r"\[(.*?)\]")
        return [t.strip() for t in bracket_regex.findall(prompt_text) if t.strip()]

    def open_character_edit_dialog(char_id: int):
        """Spawns an independent edit dialog that remains open and functional during image rendering updates."""
        with Session(engine) as session:
            char = session.get(Character, char_id)
            if not char:
                ui.notify("Character not found.", type="negative")
                return
            
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
            
            # Count character mentions in this book's prompts.csv
            frequencies = get_character_frequency_map(project_name, [book])
            total_hits = sum(frequencies.get(a.alias.lower(), 0) for a in aliases)
            if not aliases:
                total_hits = frequencies.get(char.name.lower(), 0)

        # Force mounting the dialog to the page's root client context to prevent 
        # it from being destroyed when its triggering container clears during background refreshes.
        with ui.context.client:
            with ui.dialog() as char_dialog, ui.card().classes('w-[520px] max-w-[95vw] p-5 rounded-xl flex flex-col gap-3 overflow-hidden'):
                
                # Header
                with ui.row().classes('w-full justify-between items-center border-b pb-2 flex-shrink-0'):
                    with ui.column().classes('gap-0'):
                        ui.label(f"Edit Profile: {char.name}").classes('text-sm font-bold text-slate-800')
                        ui.label(f"{total_hits} mentions in {book_name}").classes('text-[11px] text-slate-400 font-medium')
                    
                    with ui.row().classes('items-center gap-2'):
                        def toggle_modal_lock(e):
                            with Session(engine) as session:
                                db_char = session.get(Character, char_id)
                                if db_char:
                                    db_char.locked = e.value
                                    session.add(db_char)
                                    session.commit()
                            save_project_characters_to_json(project.id)
                            ui.notify(f"Profile {'Locked' if e.value else 'Unlocked'}", type="info")

                        lock_switch = ui.switch(value=char.locked, on_change=toggle_modal_lock).props('dense')
                        ui.label('Locked').classes('text-xs font-semibold text-slate-500 mr-2')

                # Scrollable Body Panel
                with ui.column().classes('w-full flex-1 overflow-y-auto pr-1 gap-3 max-h-[55vh] min-h-0'):
                    ui.label('Physical Traits').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wide')
                    
                    demo_input = ui.input(label="Demographics", value=char.demographics or "").classes('w-full bg-white').props('outlined dense')
                    hair_input = ui.input(label="Hair & Face", value=char.hair_and_face or "").classes('w-full bg-white').props('outlined dense')
                    build_input = ui.input(label="Physical Build", value=char.physical_build or "").classes('w-full bg-white').props('outlined dense')
                    marks_input = ui.input(label="Distinguishing Marks", value=char.distinguishing_marks or "").classes('w-full bg-white').props('outlined dense')

                    def get_compiled_preview() -> str:
                        mock_char = Character(
                            name=char.name,
                            demographics=demo_input.value,
                            physical_build=build_input.value,
                            hair_and_face=hair_input.value,
                            distinguishing_marks=marks_input.value
                        )
                        return compile_character_visual_prompt(mock_char)

                    # Compiled Description Textarea
                    ui.label('Compiled Visual Prompt Override').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wide mt-1')
                    
                    desc_textarea = ui.textarea(
                        value=char.visual_description or get_compiled_preview()
                    ).classes('w-full font-mono text-xs').props('outlined dense autogrow')

                    def on_trait_change():
                        if not lock_switch.value:
                            desc_textarea.value = get_compiled_preview()

                    for field_el in [demo_input, hair_input, build_input, marks_input]:
                        field_el.on('blur', on_trait_change)

                    # Aliases chip manager
                    ui.label('Mapped Alias Tags').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wide mt-1')
                    
                    @ui.refreshable
                    def render_modal_aliases():
                        with Session(engine) as session:
                            curr_aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char_id)).all()
                        
                        with ui.row().classes('w-full gap-1.5 flex-wrap items-center bg-slate-50 p-2 rounded-lg border border-dashed'):
                            if not curr_aliases:
                                ui.label('No aliases mapped.').classes('text-[11px] text-slate-400 italic')
                            for a in curr_aliases:
                                def remove_alias(alias_id=a.id):
                                    with Session(engine) as session:
                                        db_a = session.get(CharacterAlias, alias_id)
                                        if db_a:
                                            session.delete(db_a)
                                            session.commit()
                                    save_project_characters_to_json(project.id)
                                    render_modal_aliases.refresh()
                                    ui.notify("Alias tag removed.", type="info")

                                ui.chip(a.alias, removable=True, on_value_change=lambda e, aid=a.id: remove_alias(aid) if not e.value else None).classes('bg-white text-xs')
                            
                            def add_alias():
                                txt = alias_add_input.value.strip()
                                if not txt:
                                    return
                                with Session(engine) as session:
                                    dup = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char_id).where(CharacterAlias.alias == txt)).first()
                                    if not dup:
                                        new_a = CharacterAlias(character_id=char_id, alias=txt)
                                        session.add(new_a)
                                        session.commit()
                                save_project_characters_to_json(project.id)
                                alias_add_input.value = ""
                                render_modal_aliases.refresh()
                                ui.notify(f"Added alias: {txt}", type="positive")

                            alias_add_input = ui.input(placeholder="Add Tag...").classes('w-24 text-xs').props('dense borderless')
                            alias_add_input.on('keydown.enter', add_alias)
                            ui.button(icon='add', on_click=add_alias).props('flat dense round').classes('text-slate-500 text-xs')

                    render_modal_aliases()

                # Footer
                with ui.row().classes('w-full justify-end gap-2 border-t pt-2 mt-2 flex-shrink-0'):
                    def cancel():
                        char_dialog.close()

                    def save():
                        with Session(engine) as session:
                            db_char = session.get(Character, char_id)
                            if db_char:
                                db_char.demographics = demo_input.value.strip() if demo_input.value.strip() else None
                                db_char.physical_build = build_input.value.strip() if build_input.value.strip() else None
                                db_char.hair_and_face = hair_input.value.strip() if hair_input.value.strip() else None
                                db_char.distinguishing_marks = marks_input.value.strip() if marks_input.value.strip() else None
                                db_char.visual_description = desc_textarea.value.strip() if desc_textarea.value.strip() else None
                                db_char.locked = lock_switch.value
                                session.add(db_char)
                                session.commit()
                        save_project_characters_to_json(project.id)
                        ui.notify("Character profile updated successfully!", type="positive")
                        char_dialog.close()
                        if modal_prompt_input:
                            render_scene_character_chips(modal_prompt_input.value)

                    ui.button('Cancel', on_click=cancel, color='slate').props('flat').classes('text-xs font-semibold')
                    ui.button('Save Profile', on_click=save).classes('bg-blue-600 text-white font-bold text-xs px-4 py-2 rounded-lg shadow-sm')

            char_dialog.open()

    # Dynamic Scene Chips Container clear and draw handler (Robust, context-safe)
    def render_scene_character_chips(prompt_text: str):
        """Safely clears and populates matched character tags on the scene."""
        scene_chips_container.clear()
        tags = extract_prompt_character_tags(prompt_text)
        if not tags:
            return
        
        with Session(engine) as session:
            chars = session.exec(select(Character).where(Character.project_id == project.id)).all()
            
            matched_chars = []
            for char in chars:
                aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
                alias_texts = {a.alias.lower() for a in aliases}
                alias_texts.add(char.name.lower())
                
                if any(t.lower() in alias_texts for t in tags):
                    matched_chars.append((char, aliases))

        if matched_chars:
            with scene_chips_container:
                with ui.row().classes('w-full items-center gap-1 bg-slate-50 p-2 rounded-lg border border-dashed mt-1 flex-wrap'):
                    ui.label('Scene Characters:').classes('text-[9px] font-black text-slate-400 uppercase tracking-wider')
                    for char, _ in matched_chars:
                        ui.chip(
                            f"👤 {char.name}",
                            on_click=lambda _, c_id=char.id: open_character_edit_dialog(c_id)
                        ).classes('text-[11px] bg-white hover:bg-blue-50 hover:text-blue-700 cursor-pointer border py-0.5 px-2.5 rounded-md')

    # Aggressively deallocate and clear any stale workspace timers & keyboards
    # to prevent background execution leakage and active binding propagation bloat.
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

    # Initialize shortcuts, infinite scroll parameters, and key tracking in state
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
    
    # Establish dynamic load times and clear coordinate timestamps
    import time
    state.workspace_load_time = int(time.time())
    state.custom_image_timestamps = {}
    
    # Infinite gallery paging limit
    state.book_gallery_limit = 24

    # Synchronize scenes to the SQLite DB cache
    from services.sync_engine import sync_prompts_csv_to_db_cache
    with Session(engine) as session:
        sync_prompts_csv_to_db_cache(book_id, session)
        session.commit()

    def get_all_prompts_as_dicts() -> List[Dict[str, Any]]:
        """Loads and converts ScenePrompts from SQLite to standard dicts compatible with the workspace."""
        from database.models import ScenePrompt
        with Session(engine) as session:
            query = select(ScenePrompt).where(ScenePrompt.book_id == book_id).order_by(ScenePrompt.chapter_num, ScenePrompt.scene_num)
            results = session.exec(query).all()
        
        # Convert each ScenePrompt database instance into standard lower-case dicts
        return [
            {
                "chapter": str(r.chapter_num),
                "scene": str(r.scene_num),
                "prompt": r.prompt,
                "quote": r.quote,
                "approved": "True" if r.approved else "False",
                "timestamp": r.timestamp or "00:00:00"
            }
            for r in results
        ]

    # Populate active prompts from SQLite Cache instantly
    prompts = get_all_prompts_as_dicts()
    images_cache = get_book_images_cache(project_name, book_name)

    if not hasattr(state, 'book_active_scene_idx'):
        state.book_active_scene_idx = 0

    # --- 1. NESTED HANDLERS DEFINED FIRST ---

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
                
        # Fallback if the active item was filtered out
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
                if modal_img_el.source != img_url:
                    modal_img_el.set_source(img_url)
                modal_img_el.visible = True
            else:
                modal_img_el.visible = False
                
        if modal_placeholder:
            modal_placeholder.visible = not img_url
            
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

        # Highlight background Grid Card
        for (grid_ch, grid_sc), ref in grid_card_references.items():
            ref["card"].classes(remove="ring-4 ring-blue-500 ring-offset-2")
            
        if (ch, sc) in grid_card_references:
            target_ref = grid_card_references[(ch, sc)]
            target_ref["card"].classes(add="ring-4 ring-blue-500 ring-offset-2")

        # Only overwrite inputs and text if coordinates changed (prevent resetting cursor / active edits)
        if loaded_scene_coords[0] != ch or loaded_scene_coords[1] != sc:
            loaded_scene_coords[0] = ch
            loaded_scene_coords[1] = sc

            if modal_quote_el:
                modal_quote_el.set_text(f'"{current_scene.get("quote", "")}"')
                
            if modal_prompt_input:
                modal_prompt_input.set_value(current_scene.get("prompt", ""))
                
            if modal_context_html:
                modal_context_html.set_content(find_quote_context(project_name, book_name, current_scene.get("quote", "")))

        # Update character chips beneath prompt input
        if modal_prompt_input:
            render_scene_character_chips(modal_prompt_input.value)

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
                should_be_visible = bool(has_image and not is_approved)
            elif filter_mode.value == "Missing Only":
                should_be_visible = not has_image
                
            ref["card"].visible = should_be_visible
            
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
                # Only tell the client to set the image source if the URL string has actually changed.
                # This prevents unchanged images on the grid from flashing and reloading.
                if ref["image"].source != img_url:
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

    # Awaiting Generation Panel / Transcript Editor Portal
    if not prompts:
        transcript_path = Path(f"./output/{project_name}/{book_name}/transcript.txt")
        has_transcript = transcript_path.exists()
        
        if has_transcript:
            approved_marker_path = Path(f"./output/{project_name}/{book_name}/.transcript_approved")
            
            @ui.refreshable
            def render_transcript_editor():
                is_approved = approved_marker_path.exists()
                try:
                    text_content = transcript_path.read_text(encoding="utf-8")
                except Exception as e:
                    text_content = f"Error reading transcript: {str(e)}"
                    
                char_count = len(text_content)
                word_count = len(text_content.split())
                
                with ui.card().classes('w-full border p-6 shadow-sm bg-white gap-4'):
                    with ui.row().classes('items-center justify-between border-b pb-3 w-full'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('description', size='md', color='blue-500')
                            with ui.column().classes('gap-0'):
                                ui.label('Step 1.5: Transcript Review & Formatting').classes('text-base font-bold text-slate-800')
                                ui.label('Review text, prune publishing boilerplate, and mark as approved to unlock prompt generation.').classes('text-xs text-slate-500')
                        
                        if is_approved:
                            ui.badge('Approved & Ready', color='emerald').classes('px-3 py-1 text-xs font-semibold rounded-full')
                        else:
                            ui.badge('Awaiting Review', color='amber').classes('px-3 py-1 text-xs font-semibold rounded-full')
                    
                    # Volume Stats Bar
                    with ui.row().classes('w-full justify-start gap-6 bg-slate-50 p-3 rounded-lg border border-dashed text-xs'):
                        with ui.column().classes('gap-0'):
                            ui.label('Character Count').classes('text-slate-400 font-medium')
                            ui.label(f"{char_count:,} characters").classes('font-bold text-slate-700')
                        with ui.column().classes('gap-0'):
                            ui.label('Word Count').classes('text-slate-400 font-medium')
                            ui.label(f"{word_count:,} words").classes('font-bold text-slate-700')
                        with ui.column().classes('gap-0'):
                            ui.label('Estimated Scenes').classes('text-slate-400 font-medium')
                            ui.label(f"~ {max(1, word_count // 350)} scenes").classes('font-bold text-slate-700')
                    
                    def save_transcript():
                        try:
                            transcript_path.write_text(editor.value, encoding="utf-8")
                            ui.notify("Transcript file saved successfully!", type="positive")
                            render_transcript_editor.refresh()
                        except Exception as e:
                            ui.notify(f"Failed to save: {str(e)}", type="negative")
                            
                    def open_in_system_editor():
                        import platform
                        import subprocess
                        abs_path = transcript_path.resolve()
                        try:
                            if platform.system() == "Windows":
                                os.startfile(abs_path)
                            elif platform.system() == "Darwin":
                                subprocess.Popen(["open", str(abs_path)])
                            else:
                                subprocess.Popen(["xdg-open", str(abs_path)])
                            ui.notify("Opening in native text editor...", type="info")
                        except Exception as e:
                            ui.notify(f"Could not open editor: {str(e)}", type="negative")
                            
                    def reload_from_disk():
                        render_transcript_editor.refresh()

                    def toggle_approval():
                        try:
                            # Auto-save changes on approval toggle so users do not lose typed edits
                            transcript_path.write_text(editor.value, encoding="utf-8")
                        except Exception as e:
                            state.add_console_log(f"[Workspace] Failed to auto-save transcript on approval: {str(e)}")

                        if approved_marker_path.exists():
                            approved_marker_path.unlink()
                            ui.notify("Transcript approval revoked.", type="warning")
                        else:
                            approved_marker_path.touch()
                            ui.notify("Transcript approved! Ready for Phase 2 prompt generation.", type="positive")
                        
                        render_transcript_editor.refresh()
                        
                        # Trigger main project dashboard stats check
                        if state.stats_refresh_callback:
                            asyncio.create_task(state.stats_refresh_callback())
                    
                    # Action Buttons (Positioned above the text area)
                    with ui.row().classes('w-full justify-between items-center border-b pb-3 mb-1'):
                        with ui.row().classes('gap-2'):
                            ui.button('Save Changes', icon='save', on_click=save_transcript).classes('bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold px-4')
                            ui.button('Reload from Disk', icon='refresh', on_click=reload_from_disk).classes('bg-slate-500 hover:bg-slate-600 text-white text-xs font-semibold px-4')
                            ui.button('Open in External Editor', icon='open_in_new', on_click=open_in_system_editor).props('flat').classes('text-xs text-slate-600')
                            
                        if is_approved:
                            ui.button('Revoke Approval', icon='cancel', on_click=toggle_approval).classes('bg-red-600 hover:bg-red-700 text-white text-xs font-bold px-5')
                        else:
                            ui.button('Approve Transcript', icon='check_circle', on_click=toggle_approval).classes('bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold px-5')

                    # Text Editor Component
                    editor = ui.textarea(
                        label="Editable transcript.txt", 
                        value=text_content
                    ).classes('w-full text-xs') \
                     .props('outlined input-style="height: 500px; font-family: monospace; overflow-y: auto;"') \
                     .style('height: 520px;')

            render_transcript_editor()
            return

        else:
            # Original Awaiting Generation Panel (No transcript available yet)
            char_count = 0
            word_count = 0
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
                            ui.badge('Missing', color='red').classes('px-2 py-0.5 text-[10px]')
                                
                        with ui.row().classes('items-center justify-between w-full text-xs'):
                            ui.label('Character Count:').classes('text-slate-500')
                            ui.label("0 characters").classes('font-bold text-slate-700')
                            
                        with ui.row().classes('items-center justify-between w-full text-xs'):
                            ui.label('Word Count:').classes('text-slate-500')
                            ui.label("0 words").classes('font-bold text-slate-700')

                        with ui.row().classes('items-center justify-between w-full text-xs'):
                            ui.label('Estimated Scenes:').classes('text-slate-500')
                            ui.label("~ 0 scenes").classes('font-bold text-slate-700')

                    with ui.column().classes('gap-3 bg-slate-50 p-4 rounded-xl border border-dashed'):
                        ui.label('Orchestration Guide').classes('text-xs font-bold text-slate-700 uppercase tracking-wide')
                        
                        with ui.row().classes('items-center gap-2 text-xs'):
                            ui.icon('radio_button_unchecked', color='slate', size='16px')
                            ui.label('Step 1: Transcription').classes('font-bold text-slate-700')
                            
                        with ui.row().classes('items-center gap-2 text-xs'):
                            ui.icon('radio_button_unchecked', color='slate', size='16px')
                            ui.label('Step 2: Generate Prompts').classes('font-bold text-slate-700')
                            
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

    # --- Top Interface Toolbar ---
    with ui.row().classes('w-full justify-between items-center bg-white p-3 border rounded-xl shadow-xs mb-4'):
        with ui.row().classes('items-center gap-4'):
            # Icon-based Filter Selector Segment
            with ui.row().classes('items-center gap-1 bg-slate-100 p-1 rounded-lg border'):
                btn_all = ui.button(icon='grid_view').props('flat dense').classes('px-3 py-1.5 rounded-md')
                btn_unapproved = ui.button(icon='rate_review').props('flat dense').classes('px-3 py-1.5 rounded-md')
                btn_missing = ui.button(icon='image_not_supported').props('flat dense').classes('px-3 py-1.5 rounded-md')
                
                with btn_all:
                    ui.tooltip('Show All Scenes')
                with btn_unapproved:
                    ui.tooltip('Show Unapproved Scenes Only')
                with btn_missing:
                    ui.tooltip('Show Missing Scenes Only')
                    
                def update_button_styles(mode: str):
                    for m, btn in [('All', btn_all), ('Unapproved Only', btn_unapproved), ('Missing Only', btn_missing)]:
                        if mode == m:
                            btn.classes(replace='px-3 py-1.5 rounded-md bg-white text-blue-600 shadow-sm font-bold')
                        else:
                            btn.classes(replace='px-3 py-1.5 rounded-md text-slate-500 hover:text-slate-700 hover:bg-slate-200/50')
                            
                def set_filter(mode: str):
                    filter_mode.value = mode
                    update_button_styles(mode)
                    render_content.refresh()
                    
                btn_all.on('click', lambda: set_filter('All'))
                btn_unapproved.on('click', lambda: set_filter('Unapproved Only'))
                btn_missing.on('click', lambda: set_filter('Missing Only'))
                
                # Apply styles without evaluating render_content.refresh() too early
                update_button_styles(filter_mode.value)
            
            # Direct batch reboot action
            ui.button(
                'Restart Batch / Regen', 
                icon='refresh', 
                on_click=trigger_batch_restart
            ).classes('bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold px-4 h-10')

            # Stop image generation button
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
        with ui.card().classes('w-full max-w-[95vw] lg:max-w-7xl h-[90vh] p-4 rounded-xl bg-white flex flex-col items-stretch overflow-hidden gap-0'):
            with ui.grid(columns='1fr 380px').classes('w-full h-full gap-4 items-stretch overflow-hidden min-h-0'):
                with ui.column().classes('w-full h-full justify-center min-h-0 relative'):
                    with ui.card().classes('w-full h-full border rounded-xl overflow-hidden shadow-sm flex items-center justify-center bg-slate-900 relative p-0 m-0'):
                        modal_placeholder = ui.column().classes('items-center justify-center text-slate-400 w-full h-full')
                        with modal_placeholder:
                            ui.icon('photo_library', size='lg').classes('mb-2 text-slate-500 animate-pulse')
                            ui.label("Awaiting ComfyUI Generation...").classes('text-xs font-semibold text-slate-400')
                            
                        modal_img_el = ui.image("").classes('w-full h-full bg-transparent').props('fit=contain')
                        
                        modal_badge_missing = ui.badge("Missing", color="red").classes('absolute top-4 left-4 font-bold text-xs')
                        modal_badge_approved = ui.badge("Approved", color="emerald").classes('absolute top-4 left-4 font-bold text-xs')
                        modal_badge_review = ui.badge("Needs Review", color="amber").classes('absolute top-4 left-4 font-bold text-xs')
                        
                with ui.column().classes('w-full h-full gap-4 overflow-y-auto min-h-0 flex-nowrap pr-1'):
                    with ui.row().classes('w-full items-center justify-between border-b pb-2 flex-shrink-0'):
                        with ui.column().classes('gap-0'):
                            modal_title_el = ui.label("").classes('text-base font-bold text-slate-800 leading-none')
                            modal_subtitle_el = ui.label("").classes('text-[11px] text-slate-400 mt-1')
                        with ui.row().classes('items-center gap-1'):
                            with ui.button(icon='help_outline').props('flat round dense').classes('text-slate-400'):
                                with ui.tooltip().classes('bg-slate-800 text-white text-xs p-3 rounded-lg gap-1 flex flex-col shadow-lg'):
                                    ui.label('Keyboard Shortcuts').classes('font-bold border-b pb-1 text-blue-400')
                                    ui.label(f'[{state.key_approve.upper()}] Approve Scene')
                                    ui.label(f'[{state.key_delete.upper()}] Delete Image')
                                    ui.label(f'[{state.key_next.upper()}] Next Scene')
                                    ui.label(f'[{state.key_prev.upper()}] Prev Scene')
                            ui.button(icon='close', on_click=theater_dialog.close).props('flat round dense').classes('text-slate-400')
                    
                    with ui.column().classes('w-full gap-2 bg-slate-100 p-3 rounded-lg border border-dashed flex-shrink-0'):
                        with ui.row().classes('w-full gap-2 items-center justify-between'):
                            ui.button('Prev', icon='chevron_left', on_click=prev_scene).props('flat dense').classes('text-xs font-bold text-slate-600 flex-1 py-1.5 bg-white border rounded')
                            ui.button('Next', icon='chevron_right', on_click=next_scene).props('flat dense').classes('text-xs font-bold text-slate-600 flex-1 py-1.5 bg-white border rounded')
                        with ui.row().classes('w-full gap-2'):
                            ui.button('Delete Image', icon='delete', on_click=delete_current).classes('bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold flex-1 py-2')
                            ui.button('Approve', icon='check', on_click=approve_current).classes('bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold flex-1 py-2')

                    # Dynamic scene chips row (fixed below the buttons block)
                    scene_chips_container = ui.row().classes('w-full flex-shrink-0')

                    with ui.column().classes('w-full gap-2 bg-slate-50 p-3 rounded border border-dashed flex-shrink-0'):
                        ui.label("Target Narration Quote").classes('text-[9px] font-black text-slate-400 uppercase tracking-wider')
                        modal_quote_el = ui.label("").classes('text-xs italic text-slate-700 leading-relaxed font-serif')
                        
                    modal_prompt_input = ui.textarea(
                        label="Style-Ready Visual Prompt"
                    ).classes('w-full h-32 text-xs leading-relaxed flex-shrink-0').props('outlined')
                    
                    modal_prompt_input.on('blur', lambda: update_prompt_text(modal_prompt_input.value, active_row_ref[0]))
                    
                    # Bind chip updates to live keystrokes / value adjustments
                    modal_prompt_input.on('update:value', lambda e: render_scene_character_chips(e.sender.value))
                    
                    # Narrative Context drawer (restored)
                    with ui.expansion('Narrative Context (transcript.txt)').classes('w-full border rounded bg-slate-50 text-xs flex-shrink-0'):
                        modal_context_html = ui.html("").classes('p-3 leading-relaxed text-slate-700 bg-white font-serif')

    theater_dialog.on('close', handle_modal_close)

    # Keyboard shortcut listener
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

    state.book_keyboard = ui.keyboard(on_key=handle_key)

    # --- Gallery Grid View Rendering (Dynamic In-Place Appending) ---
    
    # Reference boxes to track pagination and active containers across callbacks safely
    grid_el_ref = [None]
    load_more_spinner_ref = [None]
    rendered_count_ref = [0]

    def render_batch(batch_list: list):
        """Builds and mounts card elements dynamically into the active grid context container."""
        def launch_theater(ch_val: int, sc_val: int):
            state.book_active_chapter = ch_val
            state.book_active_scene = sc_val
            theater_dialog.open()
            update_active_scene_ui()
            
        for item in batch_list:
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
            
        # Reset counters and references on fresh renders (like changing filters)
        grid_card_references.clear()
        rendered_count_ref[0] = min(24, len(filtered))
        
        # Instantiate active grid container and capture reference
        grid_el = ui.grid(columns='repeat(auto-fill, minmax(180px, 1fr))').classes('w-full gap-4')
        grid_el_ref[0] = grid_el
        
        with grid_el:
            render_batch(filtered[:rendered_count_ref[0]])
            
        # Instantiate spinner element below the grid
        load_more_spinner_ref[0] = ui.row().classes('w-full justify-center p-4')
        with load_more_spinner_ref[0]:
            ui.spinner(size='md', color='blue')
            ui.label('Scrolling to load more scenes...').classes('text-xs text-slate-400 font-medium')
        
        load_more_spinner_ref[0].visible = len(filtered) > rendered_count_ref[0]

    # Initial render of the workspace view
    render_content()

    # --- Viewport-Aware Scroll Listener (In-place append lazy loading) ---
    async def check_scroll():
        # Exit scroll checking if active details modal is open
        if theater_dialog.value:
            return
        try:
            # Check browser scroll heights on the client side
            is_near_bottom = await ui.run_javascript(
                'window.pageYOffset >= document.body.offsetHeight - 1.5 * window.innerHeight'
            )
            if is_near_bottom:
                filtered = get_filtered_prompts()
                current_count = rendered_count_ref[0]
                
                # Append next batch of cards dynamically if more exist
                if current_count < len(filtered):
                    next_count = min(current_count + 24, len(filtered))
                    next_batch = filtered[current_count:next_count]
                    rendered_count_ref[0] = next_count
                    
                    if grid_el_ref[0]:
                        with grid_el_ref[0]:
                            render_batch(next_batch)
                            
                        # Toggle loader state
                        if load_more_spinner_ref[0]:
                            load_more_spinner_ref[0].visible = len(filtered) > rendered_count_ref[0]
        except Exception:
            pass

    # Non-blocking scroll checking timer
    state.book_scroll_timer = ui.timer(0.3, check_scroll)

 # --- Real-Time Background Image Pop-in Timer (Offloaded!) ---
    last_file_count = [len(images_cache)]
    
    async def check_for_image_updates():
        if state.active_book_id is None:
            return
            
        img_dir = Path(f"./output/{project_name}/{book_name}/images")
        parent_dir = Path(f"./output/{project_name}/{book_name}")
        
        # Offload file-system counting checks to a background thread
        def count_files():
            count = 0
            if img_dir.exists():
                try:
                    count += len(os.listdir(img_dir))
                except Exception:
                    pass
            if parent_dir.exists():
                try:
                    count += len(os.listdir(parent_dir))
                except Exception:
                    pass
            return count
            
        count = await asyncio.to_thread(count_files)
        
        if count != last_file_count[0]:
            last_file_count[0] = count
            
            nonlocal images_cache
            # Offload heavy folder parsing cache building to a background thread
            images_cache = await asyncio.to_thread(get_book_images_cache, project_name, book_name)
            
            # Use smooth swap on "All" view; refresh grid container for filter changes
            if filter_mode.value == "All":
                update_grid_views_in_place()
            else:
                render_content.refresh()
                
            if theater_dialog.value:
                update_active_scene_ui()

    # Check for newly generated images every 3 seconds
    state.book_update_timer = ui.timer(3.0, check_for_image_updates)