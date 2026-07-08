import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from nicegui import ui
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Book, Project, Character, CharacterAlias, CharacterTimelineEvent
from services.character_manager import compile_character_visual_prompt, save_project_characters_to_json
from ui.components.autocomplete import PromptAutocompleteManager
from ui import state


# --- Quote Formatting Clean Helper ---
def clean_quote_text(text: str) -> str:
    """Removes standard LLM extraction artifacts from quotes."""
    if not text:
        return ""
    t = text.strip()
    
    # Strip global bracket envelopes, e.g. [ "Hello world" ] -> "Hello world"
    if t.startswith('[') and t.endswith(']'):
        t = t[1:-1].strip()
        
    # Drop outer redundant quotation markers if they envelop the entire block
    if (t.startswith('"') and t.endswith('"')) or (t.startswith('“') and t.endswith('”')):
        t = t[1:-1].strip()
        
    # Trim interior line breaks, carriage returns, and double-spaces
    t = re.sub(r'\s+', ' ', t)
    return t


def format_narrative_context(html_content: str) -> str:
    """Splits dense paragraphs into highly legible spaced lines on terminal punctuation."""
    if not html_content:
        return ""
    # Split sentences cleanly when followed by a capital letter, quote, or bracket (or tag start)
    formatted = re.sub(r'(\.|\!|\?)\s+(?=[A-Z“"\'\[<])', r'\1<br><br>', html_content)
    return formatted


def render_prompt_editor_view(
    project_id: int, 
    book_id: int, 
    images_cache: Dict[tuple, str], 
    prompts_list: List[Dict[str, Any]], 
    on_refresh_grid: callable,
    open_character_edit_dialog_fn: callable,
    open_theater_modal_fn: callable
):
    """
    Renders the complete split-screen prompt, quote, and character detail editor.
    Utilizes permanent sidebar list and split columns with zero layout shift deallocations.
    """
    # Active selected scene coordinates tracked locally in editor scope
    active_coords = [1, 1]
    if prompts_list:
        try:
            active_coords[0] = int(float(prompts_list[0].get("chapter", "1")))
            active_coords[1] = int(float(prompts_list[0].get("scene", "1")))
        except ValueError:
            pass

    # Tracks original loaded values to prevent redundant blur saving lag
    original_loaded_values = {"prompt": "", "quote": ""}

    # Compile character vocabulary (needed for Find & Replace filters)
    book_characters: List[Character] = []
    
    with Session(engine) as session:
        book_characters = session.exec(
            select(Character).where(Character.project_id == project_id)
        ).all()

    # --- Interactive Element References ---
    sidebar_container_ref = [None]
    right_pane_ref = [None]
    
    # Detail fields
    image_viewer = [None]
    image_placeholder = [None]
    character_badges_container = [None]
    quote_textarea = [None]
    prompt_textarea = [None]

    # Dynamic filter binding
    search_filter = ui.input(placeholder="Search scene text...").classes('hidden')

    def get_filtered_editor_prompts():
        query = (search_filter.value or "").strip().lower()
        if not query:
            return prompts_list
            
        filtered = []
        for p in prompts_list:
            p_text = p.get("prompt", "").lower()
            q_text = p.get("quote", "").lower()
            ch_str = f"ch {p.get('chapter', '')}"
            sc_str = f"sc {p.get('scene', '')}"
            if query in p_text or query in q_text or query in ch_str or query in sc_str:
                filtered.append(p)
        return filtered

    def get_active_scene_dict() -> Optional[Dict[str, Any]]:
        for p in prompts_list:
            try:
                ch = int(float(p.get("chapter", "1")))
                sc = int(float(p.get("scene", "1")))
            except ValueError:
                ch, sc = 1, 1
            if ch == active_coords[0] and sc == active_coords[1]:
                return p
        return prompts_list[0] if prompts_list else None

    # --- Power Tool 1: Find & Replace Dialog ---
    with ui.dialog() as find_replace_dialog, ui.card().classes('w-[460px] p-5 rounded-xl gap-3 outline-none focus:outline-none'):
        ui.label('Find & Replace prompts').classes('text-base font-bold text-slate-800')
        
        find_input = ui.input('Find Text', placeholder='e.g. [Dino]').classes('w-full').props('outlined dense')
        replace_input = ui.input('Replace With', placeholder='e.g. [Dino Captain]').classes('w-full').props('outlined dense')
        
        # Scope filters
        char_options = ["Any Character"] + [c.name for c in book_characters]
        scope_char = ui.select(options=char_options, value="Any Character", label="Character Filter").classes('w-full').props('outlined dense')
        
        with ui.row().classes('w-full gap-2'):
            ch_start = ui.number('Ch Start', value=None).classes('flex-1').props('outlined dense')
            ch_end = ui.number('Ch End', value=None).classes('flex-1').props('outlined dense')

        preview_label = ui.label('No actions queued.').classes('text-xs text-slate-400 mt-1 italic')

        def evaluate_find_replace_matches() -> List[Dict[str, Any]]:
            target_find = find_input.value.strip()
            if not target_find:
                return []
                
            matches = []
            for p in prompts_list:
                try:
                    ch_num = int(float(p.get("chapter", "1")))
                except ValueError:
                    ch_num = 1
                    
                if ch_start.value is not None and ch_num < ch_start.value:
                    continue
                if ch_end.value is not None and ch_num > ch_end.value:
                    continue
                    
                if scope_char.value != "Any Character":
                    char_tag = f"[{scope_char.value}]"
                    if char_tag.lower() not in p.get("prompt", "").lower():
                        continue
                        
                if target_find.lower() in p.get("prompt", "").lower():
                    matches.append(p)
            return matches

        def update_preview():
            matches = evaluate_find_replace_matches()
            preview_label.set_text(f"Will modify {len(matches)} matching scenes.")

        for input_el in [find_input, scope_char, ch_start, ch_end]:
            input_el.on('change', update_preview)

        with ui.row().classes('w-full justify-end gap-2 mt-2'):
            ui.button('Cancel', on_click=find_replace_dialog.close).props('flat').classes('text-xs font-semibold text-slate-500')
            
            def apply_find_replace():
                target_find = find_input.value.strip()
                target_replace = replace_input.value.strip()
                if not target_find:
                    ui.notify("Find pattern cannot be empty.", type="negative")
                    return
                    
                matches = evaluate_find_replace_matches()
                if not matches:
                    ui.notify("No matching scenes found to replace.", type="warning")
                    return
                    
                from ui.pages.book_workspace import save_prompts_csv
                with Session(engine) as session:
                    project = session.get(Project, project_id)
                    book = session.get(Book, book_id)
                    if not project or not book:
                        return
                    
                    for p in matches:
                        p["prompt"] = re.sub(re.escape(target_find), target_replace, p.get("prompt", ""), flags=re.IGNORECASE)
                        
                save_prompts_csv(project.name, book.name, prompts_list)
                find_replace_dialog.close()
                ui.notify(f"Replaced text across {len(matches)} scenes!", type="positive")
                
                register_new_tags_from_text(target_replace)
                
                on_refresh_grid()
                render_sidebar_list()
                load_active_scene_details()
                
            ui.button('Apply Replace', on_click=apply_find_replace).classes('bg-blue-600 text-white text-xs font-bold')

    # --- Power Tool 2: Bulk Format Cleanup Dialogue ---
    with ui.dialog() as bulk_clean_dialog, ui.card().classes('w-[420px] p-5 rounded-xl gap-3 outline-none focus:outline-none'):
        ui.label('Bulk clean quotes').classes('text-base font-bold text-slate-800')
        ui.label('Removes quotation envelopes, stripping bad brackets, interior spaces, and carriage returns across all scenes in this book.').classes('text-xs text-slate-500')
        
        # Checkbox allowing project-wide batch cleanup
        all_books_checkbox = ui.checkbox('Fix quotes in all books in this project?').classes('text-xs font-semibold text-slate-700 mt-1')
        
        with ui.row().classes('w-full justify-end gap-2 mt-2'):
            ui.button('Cancel', on_click=bulk_clean_dialog.close).props('flat').classes('text-xs font-semibold text-slate-500')
            
            def apply_bulk_clean():
                from ui.pages.book_workspace import save_prompts_csv, load_prompts_csv
                
                with Session(engine) as session:
                    project = session.get(Project, project_id)
                    current_book = session.get(Book, book_id)
                    if not project or not current_book:
                        return
                    
                    # Determine list of books to clean
                    if all_books_checkbox.value:
                        books_to_process = session.exec(
                            select(Book).where(Book.project_id == project_id)
                        ).all()
                    else:
                        books_to_process = [current_book]
                        
                changed_count = 0
                books_affected_count = 0
                
                for b in books_to_process:
                    if b.id == book_id:
                        # Process current active book from in-memory prompts_list to sync the immediate UI
                        book_changed = 0
                        for p in prompts_list:
                            orig = p.get("quote", "")
                            cleaned = clean_quote_text(orig)
                            if cleaned != orig:
                                p["quote"] = cleaned
                                book_changed += 1
                        if book_changed > 0:
                            save_prompts_csv(project.name, b.name, prompts_list)
                            changed_count += book_changed
                            books_affected_count += 1
                    else:
                        # Process background books by loading directly from their CSV file
                        other_prompts = load_prompts_csv(project.name, b.name)
                        if other_prompts:
                            book_changed = 0
                            for p in other_prompts:
                                orig = p.get("quote", "")
                                cleaned = clean_quote_text(orig)
                                if cleaned != orig:
                                    p["quote"] = cleaned
                                    book_changed += 1
                            if book_changed > 0:
                                save_prompts_csv(project.name, b.name, other_prompts)
                                changed_count += book_changed
                                books_affected_count += 1
                                
                if changed_count > 0:
                    if all_books_checkbox.value:
                        scope_msg = f"across {changed_count} scenes in {books_affected_count} books"
                    else:
                        scope_msg = f"across {changed_count} scenes in this book"
                    ui.notify(f"Successfully cleaned quotes {scope_msg}!", type="positive")
                    
                    # Refresh editor workspace elements for the active book
                    on_refresh_grid()
                    render_sidebar_list()
                    load_active_scene_details()
                else:
                    ui.notify("No quotes required formatting cleanup.", type="info")
                    
                bulk_clean_dialog.close()
                
            ui.button('Clean All Quotes', on_click=apply_bulk_clean).classes('bg-purple-600 text-white text-xs font-bold')

    # Stateful reference values for the Prompt Editor Large Reader Modal
    context_reader_state = {"window_before": 2500, "window_after": 2500}
    context_stats_label = [None]
    prev_chunk_btn = [None]
    next_chunk_btn = [None]
    context_reader_html = [None]

    async def update_context_reader_view(scroll_to_quote=True):
        scene = get_active_scene_dict()
        if not scene:
            return
        
        # Inline import breaks circular dependencies at module load-time
        from ui.pages.book_workspace import find_quote_context_paginated
        
        with Session(engine) as session:
            project = session.get(Project, project_id)
            book = session.get(Book, book_id)
            if project and book:
                wb = context_reader_state["window_before"]
                wa = context_reader_state["window_after"]
                res = find_quote_context_paginated(
                    project.name, 
                    book.name, 
                    scene.get("quote", ""), 
                    window_before=wb, 
                    window_after=wa
                )
                
                formatted_html = format_narrative_context(res["html"])
                
                # Resolve character names and aliases for green context highlights
                chars = session.exec(
                    select(Character).where(Character.project_id == project_id)
                ).all()
                all_aliases = []
                for char in chars:
                    aliases = session.exec(
                        select(CharacterAlias).where(CharacterAlias.character_id == char.id)
                    ).all()
                    all_aliases.extend([a.alias.lower() for a in aliases] + [char.name.lower()])
                    
                unique_aliases = sorted(list(set(a.strip() for a in all_aliases if a.strip())), key=len, reverse=True)
                
                if unique_aliases and formatted_html:
                    combined_pattern = r'\b(' + '|'.join(re.escape(alias) for alias in unique_aliases) + r')\b'
                    combined_regex = re.compile(combined_pattern, flags=re.IGNORECASE)
                    
                    tokens = re.split(r'(<[^>]+>)', formatted_html)
                    highlighted_tokens = []
                    
                    for token in tokens:
                        if token.startswith('<') and token.endswith('>'):
                            highlighted_tokens.append(token)
                        else:
                            text_segment = token
                            text_segment = combined_regex.sub(
                                lambda m: f'<mark class="bg-green-100 text-green-800 px-1 rounded font-semibold">{m.group(1)}</mark>',
                                text_segment
                            )
                            highlighted_tokens.append(text_segment)
                    formatted_html = "".join(highlighted_tokens)

                if context_reader_html[0]:
                    context_reader_html[0].set_content(formatted_html)
                
                if context_stats_label[0]:
                    context_stats_label[0].set_text(f"Showing {wb:,} chars before, {wa:,} chars after target quote")
                if prev_chunk_btn[0]:
                    prev_chunk_btn[0].set_visibility(res["has_prev"])
                if next_chunk_btn[0]:
                    next_chunk_btn[0].set_visibility(res["has_next"])
                
                if scroll_to_quote:
                    await asyncio.sleep(0.15)
                    ui.run_javascript('document.getElementById("quote-target")?.scrollIntoView({ block: "center", behavior: "smooth" })')


    def load_adjusted_context(before_delta, after_delta):
        if before_delta is None and after_delta is None:
            context_reader_state["window_before"] = 2500
            context_reader_state["window_after"] = 2500
        else:
            if before_delta:
                context_reader_state["window_before"] = max(500, context_reader_state["window_before"] - before_delta)
            if after_delta:
                context_reader_state["window_after"] = max(500, context_reader_state["window_after"] + after_delta)
        
        asyncio.create_task(update_context_reader_view(scroll_to_quote=False))

    # --- Dynamic Context Book-Style Reader Modal ---
    with ui.dialog() as context_reader_dialog:
        with ui.card().classes('w-full max-w-3xl p-6 rounded-xl bg-white flex flex-col gap-4 outline-none focus:outline-none'):
            with ui.row().classes('w-full justify-between items-center border-b pb-2 flex-shrink-0'):
                with ui.row().classes('items-center gap-2 flex-nowrap'):
                    ui.label('📖 Narrative Passage Context').classes('text-base font-bold text-slate-800')
                    context_stats_label[0] = ui.label("").classes('text-xs text-slate-400 font-semibold')
                ui.button(icon='close', on_click=context_reader_dialog.close).props('flat round dense').classes('text-slate-400')
                
            # Navigation / Paginate actions bar at top
            with ui.row().classes('w-full justify-between items-center bg-slate-50 p-2 rounded-lg border gap-2 flex-nowrap'):
                prev_chunk_btn[0] = ui.button(
                    '← Read Further Back (+2,500 chars)', 
                    on_click=lambda: load_adjusted_context(-2500, 0)
                ).props('dense flat').classes('text-xs font-semibold text-blue-600 hover:bg-blue-50 px-2 rounded-md')
                
                reset_chunk_btn = ui.button(
                    'Reset Context Window', 
                    on_click=lambda: load_adjusted_context(None, None)
                ).props('dense flat').classes('text-xs font-semibold text-slate-500 hover:bg-slate-100 px-2 rounded-md')
                
                next_chunk_btn[0] = ui.button(
                    'Read Further Ahead (+2,500 chars) →', 
                    on_click=lambda: load_adjusted_context(0, 2500)
                ).props('dense flat').classes('text-xs font-semibold text-blue-600 hover:bg-blue-50 px-2 rounded-md')

            # Scroll container mapped dynamically to auto-position the quote anchor element
            scroll_wrapper = ui.card().classes('w-full h-96 border bg-white p-4 rounded-lg overflow-y-auto relative outline-none focus:outline-none')
            with scroll_wrapper:
                context_reader_html[0] = ui.html("").classes('text-sm leading-relaxed font-serif text-slate-800 tracking-wide block')
                
            with ui.row().classes('w-full justify-end border-t pt-2 mt-1'):
                ui.button('Close Reader', on_click=context_reader_dialog.close).classes('bg-slate-600 text-white text-xs font-semibold px-4 py-2')

    async def open_context_reader():
        """Formats large context surrounding quotes, resets view state, and centers scroll into view."""
        context_reader_state["window_before"] = 2500
        context_reader_state["window_after"] = 2500
        context_reader_dialog.open()
        await update_context_reader_view(scroll_to_quote=True)


    # --- Autocomplete Tag Insertion & Registration ---
    def register_new_tags_from_text(text: str):
        """Scans saved prompts for brand-new bracketed tags, automatically registering them in the DB."""
        if not text:
            return
        bracket_regex = re.compile(r"\[(.*?)\]")
        tags = [t.strip() for t in bracket_regex.findall(text) if t.strip()]
        
        new_registrations = False
        with Session(engine) as session:
            for tag in tags:
                alias_match = session.exec(
                    select(CharacterAlias)
                    .join(Character)
                    .where(CharacterAlias.alias == tag)
                    .where(Character.project_id == project_id)
                ).first()
                if alias_match:
                    continue
                    
                char_match = session.exec(
                    select(Character)
                    .where(Character.project_id == project_id)
                    .where(Character.name == tag)
                ).first()
                if char_match:
                    new_alias = CharacterAlias(character_id=char_match.id, alias=tag)
                    session.add(new_alias)
                    new_registrations = True
                    continue
                    
                new_char = Character(project_id=project_id, name=tag)
                session.add(new_char)
                session.commit()
                
                base_ev = CharacterTimelineEvent(
                    character_id=new_char.id,
                    book_id=None,
                    chapter_num=0,
                    scene_num=0,
                    label="Base State"
                )
                session.add(base_ev)
                
                new_alias = CharacterAlias(character_id=new_char.id, alias=tag)
                session.add(new_alias)
                new_registrations = True
                
            if new_registrations:
                session.commit()
                save_project_characters_to_json(project_id)
                ui.notify("Registered new scene character tags automatically!", type="positive")
                
                nonlocal book_characters
                book_characters = session.exec(
                    select(Character).where(Character.project_id == project_id)
                ).all()

    # --- Live Detail Panel Renderer ---
    def render_scene_character_chips(prompt_text: str):
        """Safely clears and populates matched character tags on the scene with descriptive hover tooltips."""
        character_badges_container[0].clear()
        bracket_regex = re.compile(r"\[(.*?)\]")
        
        raw_tags = [t.strip() for t in bracket_regex.findall(prompt_text or "") if t.strip()]
        scene_tags = []
        seen_tags = set()
        
        for t in raw_tags:
            if t.lower() not in seen_tags:
                seen_tags.add(t.lower())
                scene_tags.append(t)
        
        scene = get_active_scene_dict()
        try:
            ch = int(float(scene.get("chapter", "1"))) if scene else 1
            sc = int(float(scene.get("scene", "1"))) if scene else 1
        except ValueError:
            ch, sc = 1, 1

        with character_badges_container[0]:
            if not scene_tags:
                ui.label('No characters tagged in this scene.').classes('text-[11px] text-slate-400 italic')
            else:
                with Session(engine) as session:
                    for tag in scene_tags:
                        char = session.exec(
                            select(Character)
                            .join(CharacterAlias)
                            .where(CharacterAlias.alias == tag)
                            .where(Character.project_id == project_id)
                        ).first()
                        if not char:
                            char = session.exec(
                                select(Character)
                                .where(Character.name == tag)
                                .where(Character.project_id == project_id)
                            ).first()
                            
                        if char:
                            events = session.exec(
                                select(CharacterTimelineEvent)
                                .where(CharacterTimelineEvent.character_id == char.id)
                            ).all()
                            
                            books_list = session.exec(select(Book).where(Book.project_id == project_id)).all()
                            book_order_map = {b.id: (b.book_order or 0) for b in books_list}
                            target_book_order = book_order_map.get(book_id, 0)
                            
                            matched_evs = []
                            base_ev = None
                            for ev in events:
                                if ev.book_id is None:
                                    base_ev = ev
                                    continue
                                ev_order = book_order_map.get(ev.book_id, 0)
                                if ev_order < target_book_order:
                                    matched_evs.append((ev, ev_order))
                                elif ev_order == target_book_order:
                                    if ev.chapter_num < ch:
                                        matched_evs.append((ev, ev_order))
                                    elif ev.chapter_num == ch and ev.scene_num <= sc:
                                        matched_evs.append((ev, ev_order))
                                        
                            active_ev = base_ev
                            if matched_evs:
                                matched_evs.sort(key=lambda x: (x[1], x[0].chapter_num, x[0].scene_num))
                                active_ev = matched_evs[-1][0]
                                
                            if not active_ev:
                                active_ev = base_ev

                            demo_text = active_ev.demographics or "unspecified demographics"
                            build_text = active_ev.physical_build or "unspecified build"
                            tooltip_desc = f"👤 {char.name} ({demo_text}, {build_text})"
                            
                            btn = ui.button(
                                f"👤 {char.name}", 
                                on_click=lambda _, cid=char.id: open_character_edit_dialog_fn(cid)
                            ).props('dense outline size=sm color=primary').classes('text-[10px] bg-white hover:bg-blue-50/50 normal-case rounded px-2.5 py-0.5 outline-none focus:outline-none')
                            with btn:
                                ui.tooltip(tooltip_desc).classes('bg-slate-800 text-white text-[10px] p-2.5 rounded-md')

    def load_active_scene_details():
        scene = get_active_scene_dict()
        if not scene:
            if right_pane_ref[0]:
                right_pane_ref[0].visible = False
            return
            
        if right_pane_ref[0]:
            right_pane_ref[0].visible = True
            
        try:
            ch = int(float(scene.get("chapter", "1")))
            sc = int(float(scene.get("scene", "1")))
        except ValueError:
            ch, sc = 1, 1
            
        img_url = images_cache.get((ch, sc))
        if img_url:
            image_viewer[0].set_source(img_url)
            image_viewer[0].visible = True
            image_placeholder[0].visible = False
        else:
            image_viewer[0].visible = False
            image_placeholder[0].visible = True

        quote_textarea[0].set_value(scene.get("quote", ""))
        prompt_textarea[0].set_value(scene.get("prompt", ""))
        
        original_loaded_values["prompt"] = scene.get("prompt", "")
        original_loaded_values["quote"] = scene.get("quote", "")

        render_scene_character_chips(scene.get("prompt", ""))

    def save_active_scene_changes():
        scene = get_active_scene_dict()
        if not scene:
            return
            
        prompt_val = prompt_textarea[0].value.strip()
        quote_val = quote_textarea[0].value.strip()
        
        if prompt_val == original_loaded_values["prompt"] and quote_val == original_loaded_values["quote"]:
            return

        from ui.pages.book_workspace import save_prompts_csv
        with Session(engine) as session:
            project = session.get(Project, project_id)
            book = session.get(Book, book_id)
            if not project or not book:
                return
                
        scene["prompt"] = prompt_val
        scene["quote"] = quote_val
        
        save_prompts_csv(project.name, book.name, prompts_list)
        register_new_tags_from_text(prompt_val)
        
        original_loaded_values["prompt"] = prompt_val
        original_loaded_values["quote"] = quote_val
        
        on_refresh_grid()
        load_active_scene_details()

    # --- Sidebar Row Renderer ---
    def select_sidebar_scene(ch_num: int, sc_num: int):
        active_coords[0] = ch_num
        active_coords[1] = sc_num
        
        for row in sidebar_container_ref[0].default_slot.children:
            try:
                elem_key = row.key_coords
                row.classes(replace="p-2 cursor-pointer rounded-lg border flex flex-col gap-1 w-full transition-all border-slate-200 hover:bg-slate-50/50 bg-white outline-none focus:outline-none")
                if elem_key == (ch_num, sc_num):
                    row.classes(replace="p-2 cursor-pointer rounded-lg border flex flex-col gap-1 w-full transition-all border-blue-500 border-l-4 border-l-blue-600 bg-blue-50/40 shadow-sm font-semibold outline-none focus:outline-none")
            except Exception:
                pass
                
        load_active_scene_details()

    def render_sidebar_list():
        sidebar_container_ref[0].clear()
        filtered = get_filtered_editor_prompts()
        
        with sidebar_container_ref[0]:
            if not filtered:
                ui.label('No matching scenes.').classes('text-xs text-slate-400 italic p-4 text-center')
                return
                
            for item in filtered:
                try:
                    ch = int(float(item.get("chapter", "1")))
                    sc = int(float(item.get("scene", "1")))
                except ValueError:
                    ch, sc = 1, 1
                    
                img_url = images_cache.get((ch, sc))
                is_approved = item.get("approved", "False").strip().lower() == "true"
                is_selected = (ch == active_coords[0] and sc == active_coords[1])
                
                border_style = "border-slate-200 hover:bg-slate-50/50 bg-white outline-none focus:outline-none"
                if is_selected:
                    border_style = "border-blue-500 border-l-4 border-l-blue-600 bg-blue-50/40 shadow-sm font-semibold outline-none focus:outline-none"
                    
                dot_color = "bg-amber-500"
                if not img_url:
                    dot_color = "bg-red-500"
                elif is_approved:
                    dot_color = "bg-emerald-500"
                    
                bracket_regex = re.compile(r"\[(.*?)\]")
                tags = [t.strip() for t in bracket_regex.findall(item.get("prompt", "")) if t.strip()]

                with ui.card().classes(f'p-2 cursor-pointer rounded-lg border flex flex-col gap-1 w-full transition-all {border_style}') as card_el:
                    card_el.key_coords = (ch, sc)
                    
                    card_el.on('click', lambda _, ch_val=ch, sc_val=sc: select_sidebar_scene(ch_val, sc_val))
                    
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label(f"Ch {item.get('chapter')}, Sc {item.get('scene')}").classes('text-xs font-bold text-slate-800')
                        ui.element('div').classes(f'w-2 h-2 rounded-full {dot_color}')
                        
                    quote_prev = item.get("quote", "")
                    if len(quote_prev) > 55:
                        quote_prev = quote_prev[:52] + "..."
                    ui.label(f'"{quote_prev}"').classes('text-[10px] text-slate-500 italic font-serif leading-none')
                    
                    if tags:
                        with ui.row().classes('w-full gap-1 flex-wrap mt-0.5'):
                            for tag in tags[:2]:
                                ui.badge(tag, color='slate').classes('text-[8px] font-medium py-0 px-1 rounded-sm')
                            if len(tags) > 2:
                                ui.badge(f"+{len(tags)-2}", color='blue').classes('text-[8px] font-semibold py-0 px-1 rounded-sm')

    # --- Main Split-Screen Workspace Builder ---
    with ui.row().classes('w-full justify-between items-center bg-white p-3 border rounded-xl shadow-2xs mb-4 outline-none focus:outline-none'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('rate_review', size='sm', color='blue-600')
            ui.label('Prompt & Quote Editor').classes('text-sm font-black text-slate-800')
            
        with ui.row().classes('items-center gap-2'):
            ui.button('Find & Replace', icon='find_replace', on_click=find_replace_dialog.open).props('flat dense').classes('text-xs font-semibold text-slate-600 border bg-white rounded px-3 py-1.5 outline-none focus:outline-none')
            ui.button('Bulk Clean Quotes', icon='cleaning_services', on_click=bulk_clean_dialog.open).props('flat dense').classes('text-xs font-semibold text-slate-600 border bg-white rounded px-3 py-1.5 outline-none focus:outline-none')

    # Parallel Layout panels
    with ui.grid(columns='320px 1fr').classes('w-full gap-4 h-[calc(100vh-210px)] overflow-hidden items-stretch'):
        
        # --- LEFT SIDEBAR INDEX COLUMN ---
        with ui.column().classes('w-full h-full gap-2 overflow-hidden flex-nowrap border-r pr-3'):
            ui.input(
                placeholder='Search chapters, quotes, prompts...', 
                on_change=lambda e: [search_filter.set_value(e.value), render_sidebar_list()]
            ).classes('w-full bg-slate-50 rounded-lg').props('outlined dense clearable')
            
            sidebar_scroller = ui.scroll_area().classes('w-full flex-1')
            with sidebar_scroller:
                sidebar_container_ref[0] = ui.column().classes('w-full gap-2 pr-2.5 py-1')

        # --- RIGHT MAIN DETAIL SCREEN ---
        with ui.grid(columns='1fr 1.2fr').classes('w-full h-full gap-4 items-stretch overflow-hidden') as right_pane:
            right_pane_ref[0] = right_pane
            
            # --- DETAILED PANEL COLUMN 1 (READ-ONLY VIEWER) ---
            with ui.column().classes('w-full h-full gap-3 overflow-y-auto min-h-0 flex-nowrap border rounded-xl p-4 bg-slate-50/50 outline-none focus:outline-none'):
                
                ui.label("Rendered Visual").classes('text-[10px] font-black text-slate-400 uppercase tracking-wider')
                with ui.card().classes('w-full h-96 border rounded-lg overflow-hidden flex items-center justify-center bg-slate-900 relative p-0 cursor-pointer outline-none focus:outline-none') \
                        .on('click', lambda: open_theater_modal_fn(active_coords[0], active_coords[1])):
                    
                    image_placeholder[0] = ui.column().classes('items-center justify-center text-slate-500 w-full h-full')
                    with image_placeholder[0]:
                        ui.icon('photo_library', size='md', color='slate-600')
                        ui.label("Awaiting Image...").classes('text-[11px] text-slate-500 mt-1')
                        
                    image_viewer[0] = ui.image("").classes('w-full h-full object-contain')
                    
                # Spacious Context Reader trigger (Completely replaced old preview pane)
                ui.button(
                    '📖 Open Context Reader', 
                    on_click=open_context_reader
                ).classes('w-full py-2.5 bg-blue-50 text-blue-600 border border-blue-200 hover:bg-blue-100/80 rounded-lg outline-none focus:outline-none font-bold text-xs mt-1')

            # --- DETAILED PANEL COLUMN 2 (INTERACTIVE EDITOR) ---
            with ui.column().classes('w-full h-full gap-3 overflow-y-auto min-h-0 flex-nowrap border rounded-xl p-4 bg-slate-50/50 outline-none focus:outline-none'):
                
                # Top: Compact Horizontal Badges Row
                ui.label("Character Profiles In Scene").classes('text-[10px] font-black text-slate-400 uppercase tracking-wider')
                character_badges_container[0] = ui.row().classes('w-full gap-1.5 flex-wrap items-center bg-white p-2.5 rounded-lg border outline-none focus:outline-none')
                
                # Middle: Target Quote Block
                ui.label("Target Narration Quote").classes('text-[10px] font-black text-slate-400 uppercase tracking-wider mt-2')
                quote_textarea[0] = ui.textarea().classes('w-full text-xs bg-white font-serif').props('outlined dense autogrow rows=2')
                quote_textarea[0].on('blur', save_active_scene_changes)
                
                # Bottom: Style Visual Prompt
                ui.label("Visual Rendering Prompt").classes('text-[10px] font-black text-slate-400 uppercase tracking-wider mt-2')
                
                with ui.element('div').classes('relative w-full'):
                    # Note: Added 'visual-prompt-editor' class so JS can focus it!
                    prompt_textarea[0] = ui.textarea().classes('w-full text-xs bg-white visual-prompt-editor').props('outlined dense autogrow rows=4')
                    prompt_textarea[0].on('blur', save_active_scene_changes)
                    
                    # Set up our unified autocomplete manager on the target input
                    PromptAutocompleteManager(
                        textarea=prompt_textarea[0],
                        project_id=project_id,
                        on_change_callback=lambda val: render_scene_character_chips(val)
                    )

    # Initial side-by-side loaders
    render_sidebar_list()
    load_active_scene_details()