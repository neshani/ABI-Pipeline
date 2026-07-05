import asyncio
import re
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from nicegui import ui
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Project, Book, Character, CharacterAlias, CharacterStateModifier, CharacterTimelineEvent
from services.character_manager import (
    extract_characters_from_prompts,
    save_project_characters_to_json,
    merge_character_aliases,
    run_stateful_character_profiling,
    get_character_mention_chunks,
    get_character_book_mentions,
    save_setting,
    auto_merge_project_characters,
    compile_character_visual_prompt,
    ensure_book_orders,
    get_matching_source_projects,
    get_character_import_matches,
    execute_character_import
)

# Active local state trackers
selected_book_id: Optional[int] = None
is_profiling_all: bool = False
cancel_profiling_all: bool = False
currently_profiling_char_id: Optional[int] = None
profiling_progress: str = ""
profiler_scan_depth: int = 5
workspace_was_empty: bool = True

# Dynamic Filter and Interactive Selection states
search_query: str = ""
sort_by: str = "mentions_desc"
filter_status: str = "all"
selected_character_id: Optional[int] = None
selected_event_id: Optional[int] = None

# High-density caching tracker
row_elements: Dict[int, ui.row] = {}


def get_character_frequency_map(project_name: str, books: List[Book]) -> Dict[str, int]:
    """Scans prompts.csv files to build a fast map of bracket tag occurrences."""
    frequencies = {}
    bracket_regex = re.compile(r"\[(.*?)\]")
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    for b in books:
        csv_path = base_output_dir / project_name / b.name / "prompts.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="|")
                for row in reader:
                    prompt_text = row.get("prompt", "")
                    for match in bracket_regex.findall(prompt_text):
                        clean_tag = match.strip().lower()
                        frequencies[clean_tag] = frequencies.get(clean_tag, 0) + 1
        except Exception:
            pass
    return frequencies


def open_add_event_dialog(project_id: int, character_id: int, books: List[Book], refresh_callback: Any):
    """Spawns modal asking coordinate parameters to add a timeline transition event."""
    with ui.dialog() as dialog, ui.card().classes('w-[450px] p-5 rounded-xl flex flex-col gap-3'):
        ui.label('Add Timeline Override Event').classes('text-sm font-bold text-slate-800')
        
        book_opts = {b.id: b.name for b in books}
        if not book_opts:
            ui.label('No books imported in this project yet.').classes('text-xs text-red-500')
            ui.button('Close', on_click=dialog.close).props('flat')
            dialog.open()
            return
            
        book_select = ui.select(options=book_opts, value=books[0].id if books else None, label="Target Book").classes('w-full bg-white')
        chapter_input = ui.number(label="Chapter Number", value=1, min=1, step=1).classes('w-full').props('outlined dense')
        scene_input = ui.number(label="Scene Number", value=1, min=1, step=1).classes('w-full').props('outlined dense')
        label_input = ui.input(label="Event Label", placeholder="e.g. Gandalf the White").classes('w-full').props('outlined dense')
        
        copied_status = ui.label("").classes('text-[11px] text-green-600 font-semibold')
        copied_traits = {}
        
        def copy_previous():
            """Queries chronological state machine up to current coordinate to clone values."""
            with Session(engine) as session:
                events = session.exec(
                    select(CharacterTimelineEvent)
                    .where(CharacterTimelineEvent.character_id == character_id)
                ).all()
                
                tgt_book = session.get(Book, book_select.value)
                tgt_order = tgt_book.book_order if tgt_book else 0
                tgt_ch = int(chapter_input.value or 1)
                tgt_sc = int(scene_input.value or 1)
                
                all_books = session.exec(select(Book).where(Book.project_id == project_id)).all()
                book_order_map = {b.id: (b.book_order or 0) for b in all_books}
                
                matched = []
                base_ev = None
                for ev in events:
                    if ev.book_id is None:
                        base_ev = ev
                        continue
                    ev_order = book_order_map.get(ev.book_id, 0)
                    if ev_order < tgt_order:
                        matched.append((ev, ev_order))
                    elif ev_order == tgt_order:
                        if ev.chapter_num < tgt_ch:
                            matched.append((ev, ev_order))
                        elif ev.chapter_num == tgt_ch and ev.scene_num <= tgt_sc:
                            matched.append((ev, ev_order))
                            
                resolved = base_ev
                if matched:
                    matched.sort(key=lambda x: (x[1], x[0].chapter_num, x[0].scene_num))
                    resolved = matched[-1][0]
                
                if resolved:
                    copied_traits["demographics"] = resolved.demographics
                    copied_traits["physical_build"] = resolved.physical_build
                    copied_traits["hair_and_face"] = resolved.hair_and_face
                    copied_traits["distinguishing_marks"] = resolved.distinguishing_marks
                    copied_traits["visual_description"] = resolved.visual_description
                    copied_status.text = f"Copied description from: '{resolved.label or 'Base State'}'!"
                else:
                    copied_status.text = "No previous states found to copy."

        ui.button('Copy Previous Description', icon='content_copy', on_click=copy_previous).classes('text-xs text-slate-700 bg-slate-100 hover:bg-slate-200 border w-full')
        
        def save_new_event():
            b_id = book_select.value
            ch = int(chapter_input.value or 1)
            sc = int(scene_input.value or 1)
            lbl = label_input.value.strip() or "Override State"
            
            with Session(engine) as session:
                dup = session.exec(
                    select(CharacterTimelineEvent)
                    .where(CharacterTimelineEvent.character_id == character_id)
                    .where(CharacterTimelineEvent.book_id == b_id)
                    .where(CharacterTimelineEvent.chapter_num == ch)
                    .where(CharacterTimelineEvent.scene_num == sc)
                ).first()
                
                if dup:
                    ui.notify("An override event already exists at this scene coordinate.", type="warning")
                    return
                
                new_ev = CharacterTimelineEvent(
                    character_id=character_id,
                    book_id=b_id,
                    chapter_num=ch,
                    scene_num=sc,
                    label=lbl,
                    demographics=copied_traits.get("demographics"),
                    physical_build=copied_traits.get("physical_build"),
                    hair_and_face=copied_traits.get("hair_and_face"),
                    distinguishing_marks=copied_traits.get("distinguishing_marks"),
                    visual_description=copied_traits.get("visual_description")
                )
                session.add(new_ev)
                session.commit()
                
                global selected_event_id
                selected_event_id = new_ev.id
                
            save_project_characters_to_json(project_id)
            ui.notify("Timeline override event added!", type="positive")
            dialog.close()
            refresh_callback()

        with ui.row().classes('w-full justify-end gap-2 border-t pt-3 mt-2'):
            ui.button('Cancel', on_click=dialog.close).props('flat')
            ui.button('Create Event', on_click=save_new_event).classes('bg-blue-600 text-white font-bold text-xs px-3 py-1.5 rounded-lg')

    dialog.open()


def open_batch_profiler_dialog(
    project: Project, 
    books: List[Book], 
    refresh_ui_callback: Any, 
    refresh_toolbar_callback: Any,
    refresh_details_callback: Any
):
    """Opens options settings before launching character batch profiling runs."""
    global is_profiling_all, currently_profiling_char_id, profiling_progress, cancel_profiling_all, profiler_scan_depth

    async def safe_refresh(callback_fn):
        try:
            if asyncio.iscoroutinefunction(callback_fn):
                await callback_fn()
            else:
                callback_fn()
        except RuntimeError as e:
            if "The parent element this slot belongs to" in str(e):
                pass
            else:
                raise

    with ui.dialog() as dialog, ui.card().classes('w-[520px] max-w-[95vw] p-6 rounded-xl flex flex-col gap-4 overflow-hidden'):
        
        with ui.row().classes('w-full justify-between items-center border-b pb-3 shrink-0'):
            with ui.column().classes('gap-0.5'):
                ui.label('Batch Profiler Options').classes('text-base font-bold text-slate-800')
                ui.label('Configure rules for the automated batch sequence.').classes('text-xs text-slate-500')
            ui.button(icon='close', on_click=dialog.close).props('flat dense').classes('text-slate-400')

        with ui.row().classes('w-full items-center justify-between gap-3 bg-slate-50 p-3 rounded-lg border shrink-0'):
            with ui.column().classes('gap-0.5'):
                ui.label('Minimum Mentions Limit').classes('text-xs font-semibold text-slate-700')
                ui.label('Skips low-frequency background characters.').classes('text-[10px] text-slate-400')
            min_mentions_input = ui.number(value=5, min=1, step=1).classes('w-16 bg-white').props('outlined dense')

        with ui.tabs().classes('w-full border-b shrink-0') as tabs:
            factual_tab = ui.tab('Factual').classes('text-xs font-bold')
            creative_tab = ui.tab('Creative').classes('text-xs font-bold')

        async def run_configured_batch(start_mode: str):
            global is_profiling_all, currently_profiling_char_id, profiling_progress, cancel_profiling_all, profiler_scan_depth
            
            clear_existing = clear_existing_cb.value
            min_mentions = int(min_mentions_input.value or 1)
            run_creative_after = run_creative_after_cb.value if start_mode == "factual" else False
            creative_target = speculate_criteria.value

            stopping_traits = []
            if stop_demo.value: stopping_traits.append("demographics")
            if stop_build.value: stopping_traits.append("physical_build")
            if stop_hair.value: stopping_traits.append("hair_and_face")
            if stop_marks.value: stopping_traits.append("distinguishing_marks")

            dialog.close()

            is_profiling_all = True
            cancel_profiling_all = False
            await safe_refresh(refresh_toolbar_callback)

            client = ui.context.client

            def get_base_queue():
                frequencies = get_character_frequency_map(project.name, books)
                with Session(engine) as session:
                    unlocked_chars = session.exec(
                        select(Character).where(Character.project_id == project.id).where(Character.locked == False)
                    ).all()
                    
                    char_aliases = {}
                    for char in unlocked_chars:
                        aliases = session.exec(
                            select(CharacterAlias).where(CharacterAlias.character_id == char.id)
                        ).all()
                        char_aliases[char.id] = aliases

                def get_char_mentions(char_obj, aliases_list):
                    total = 0
                    for a in aliases_list:
                        total += frequencies.get(a.alias.lower(), 0)
                    if not aliases_list:
                        total = frequencies.get(char_obj.name.lower(), 0)
                    return total

                queue = []
                for char in unlocked_chars:
                    aliases_list = char_aliases.get(char.id, [])
                    mentions = get_char_mentions(char, aliases_list)
                    if mentions >= min_mentions:
                        queue.append(char)
                return queue

            # Phase 1: Factual Extraction Pass
            if start_mode == "factual":
                factual_queue = get_base_queue()
                with client:
                    ui.notify(f"Starting factual batch for {len(factual_queue)} characters...", type="info")

                for idx, char in enumerate(factual_queue):
                    if cancel_profiling_all:
                        break

                    currently_profiling_char_id = char.id
                    profiling_progress = f"[Factual] Profiling {char.name} ({idx + 1}/{len(factual_queue)})..."
                    await safe_refresh(refresh_toolbar_callback)
                    
                    if char.id == selected_character_id:
                        await safe_refresh(refresh_details_callback)

                    def make_progress_callback(char_obj=char, char_idx=idx, total_chars=len(factual_queue)):
                        def progress_callback(c_id, scanned, total, state_checklist):
                            global profiling_progress
                            found_traits = [v for k, v in state_checklist.items() if v]
                            traits_str = ", ".join(found_traits)[:40]
                            
                            if traits_str:
                                profiling_progress = f"[Factual] {char_obj.name} ({char_idx + 1}/{total_chars}) [{scanned}/{total}] - {traits_str}..."
                            else:
                                profiling_progress = f"[Factual] {char_obj.name} ({char_idx + 1}/{total_chars}) [{scanned}/{total}]..."
                            
                            asyncio.run_coroutine_threadsafe(safe_refresh(refresh_toolbar_callback), asyncio.get_event_loop())
                            if char_obj.id == selected_character_id:
                                asyncio.run_coroutine_threadsafe(safe_refresh(refresh_details_callback), asyncio.get_event_loop())
                        return progress_callback
                    
                    try:
                        await run_stateful_character_profiling(
                            project_id=project.id, 
                            character_id=char.id, 
                            book_id=selected_book_id, 
                            max_chunks_to_scan=profiler_scan_depth,
                            clear_existing=clear_existing,
                            early_stopping_traits=stopping_traits if stopping_traits else None,
                            is_cancelled_fn=lambda: cancel_profiling_all,
                            progress_callback=make_progress_callback(char, idx, len(factual_queue)),
                            speculate=False
                        )
                    except Exception as ex:
                        print(f"[Profiler] Error scanning {char.name}: {str(ex)}")

            # Phase 2: Creative / Speculation Pass
            if (start_mode == "creative" or (start_mode == "factual" and run_creative_after)) and not cancel_profiling_all:
                base_queue = get_base_queue()
                
                def get_trait_count(ev_obj) -> int:
                    fields = [
                        ev_obj.demographics, ev_obj.physical_build, 
                        ev_obj.hair_and_face, ev_obj.distinguishing_marks
                    ]
                    return sum(1 for f in fields if f and str(f).strip() and str(f).lower() != "null")

                creative_queue = []
                for char in base_queue:
                    with Session(engine) as session:
                        db_char = session.get(Character, char.id)
                        if not db_char or db_char.locked:
                            continue
                        # Query Base State event to check completion
                        base_ev = session.exec(
                            select(CharacterTimelineEvent)
                            .where(CharacterTimelineEvent.character_id == char.id)
                            .where(CharacterTimelineEvent.book_id == None)
                        ).first()
                        traits_count = get_trait_count(base_ev) if base_ev else 0
                    
                    if creative_target == "0" and traits_count == 0:
                        creative_queue.append(char)
                    elif creative_target == "1" and traits_count <= 1:
                        creative_queue.append(char)
                    elif creative_target == "all":
                        creative_queue.append(char)

                if creative_queue:
                    with client:
                        ui.notify(f"Starting creative speculation batch for {len(creative_queue)} characters...", type="info")

                    for idx, char in enumerate(creative_queue):
                        if cancel_profiling_all:
                            break

                        currently_profiling_char_id = char.id
                        profiling_progress = f"[Creative] Casting {char.name} ({idx + 1}/{len(creative_queue)})..."
                        await safe_refresh(refresh_toolbar_callback)
                        
                        if char.id == selected_character_id:
                            await safe_refresh(refresh_details_callback)

                        def make_progress_callback(char_obj=char, char_idx=idx, total_chars=len(creative_queue)):
                            def progress_callback(c_id, scanned, total, state_checklist):
                                global profiling_progress
                                found_traits = [v for k, v in state_checklist.items() if v]
                                traits_str = ", ".join(found_traits)[:40]
                                
                                if traits_str:
                                    profiling_progress = f"[Creative] {char_obj.name} ({char_idx + 1}/{total_chars}) [{scanned}/{total}] - {traits_str}..."
                                else:
                                    profiling_progress = f"[Creative] {char_obj.name} ({char_idx + 1}/{total_chars}) [{scanned}/{total}]..."
                                
                                asyncio.run_coroutine_threadsafe(safe_refresh(refresh_toolbar_callback), asyncio.get_event_loop())
                                if char_obj.id == selected_character_id:
                                    asyncio.run_coroutine_threadsafe(safe_refresh(refresh_details_callback), asyncio.get_event_loop())
                            return progress_callback

                        try:
                            await run_stateful_character_profiling(
                                project_id=project.id, 
                                character_id=char.id, 
                                book_id=selected_book_id, 
                                max_chunks_to_scan=profiler_scan_depth,
                                clear_existing=False,
                                early_stopping_traits=None,
                                is_cancelled_fn=lambda: cancel_profiling_all,
                                progress_callback=make_progress_callback(char, idx, len(creative_queue)),
                                speculate=True
                            )
                        except Exception as ex:
                            print(f"[Profiler] Error speculating {char.name}: {str(ex)}")
                else:
                    with client:
                        ui.notify("No characters met creative target criteria.", type="info")

            is_profiling_all = False
            currently_profiling_char_id = None
            profiling_progress = ""
            cancel_profiling_all = False
            
            with client:
                ui.notify("Batch profiling sequence completed.", type="info")
            
            await safe_refresh(refresh_toolbar_callback)
            await safe_refresh(refresh_ui_callback)

        with ui.tab_panels(tabs, value=factual_tab).classes('w-full flex-1 min-h-0 bg-transparent'):
            
            # FACTUAL PANEL
            with ui.tab_panel(factual_tab).classes('p-0 flex flex-col gap-4 h-full justify-between'):
                with ui.column().classes('w-full gap-4'):
                    clear_existing_cb = ui.checkbox(
                        'Wipe existing profile traits before profiling', 
                        value=False
                    ).tooltip("If checked, completely clears active event traits before researching.")

                    with ui.column().classes('w-full gap-1.5 bg-slate-50 p-3 rounded-lg border'):
                        ui.label('Early Stopping Criteria').classes('text-xs font-semibold text-slate-700')
                        ui.label('Stop scanning a character as soon as selected traits are found:').classes('text-[10px] text-slate-400 mb-1')
                        
                        with ui.grid().classes('grid-cols-2 gap-2 w-full'):
                            stop_demo = ui.checkbox('Demographics', value=True)
                            stop_build = ui.checkbox('Physical Build', value=True)
                            stop_hair = ui.checkbox('Hair & Face', value=False)
                            stop_marks = ui.checkbox('Distinguishing Marks', value=False)

                    run_creative_after_cb = ui.checkbox(
                        'Run Creative Casting after Factual pass?', 
                        value=False
                    ).tooltip("If checked, characters will undergo speculation following factual passes.")

                with ui.row().classes('w-full justify-end gap-2 border-t pt-3 shrink-0 mt-auto'):
                    ui.button('Cancel', on_click=dialog.close).props('flat').classes('text-xs text-slate-500 font-semibold')
                    ui.button(
                        'Run Factual', 
                        icon='science', 
                        on_click=lambda: run_configured_batch('factual')
                    ).classes('bg-blue-600 text-white font-bold text-xs px-4 py-2 rounded-lg shadow-sm')

            # CREATIVE PANEL
            with ui.tab_panel(creative_tab).classes('p-0 flex flex-col gap-4 h-full justify-between'):
                with ui.column().classes('w-full gap-4'):
                    ui.markdown(
                        "Creative Casting uses local LLM **speculation** to fill in sparse descriptions."
                    ).classes('text-xs text-slate-500 leading-relaxed bg-slate-50 p-3 rounded-lg border w-full')

                    speculate_criteria = ui.select(
                        options={
                            "0": "Empty profiles only (0/4 traits)",
                            "1": "Sparse profiles only (0 or 1/4 traits)",
                            "all": "All unlocked characters"
                        },
                        value="0",
                        label="Generate speculative details for characters with:"
                    ).classes('w-full bg-white').props('outlined dense')

                with ui.row().classes('w-full justify-end gap-2 border-t pt-3 shrink-0 mt-auto'):
                    ui.button('Cancel', on_click=dialog.close).props('flat').classes('text-xs text-slate-500 font-semibold')
                    ui.button(
                        'Run Creative', 
                        icon='theater_comedy', 
                        on_click=lambda: run_configured_batch('creative')
                    ).classes('bg-indigo-600 text-white font-bold text-xs px-4 py-2 rounded-lg shadow-sm')

    dialog.open()


def open_prompt_editor_dialog():
    """Renders a modal to customize the LLM profiler template instructions."""
    current_template = get_setting("character_profiler_template", "")
    if not current_template:
        from services.character_manager import get_default_character_template
        current_template = get_default_character_template()

    with ui.dialog() as dialog, ui.card().classes('w-[750px] max-w-[95vw] h-[650px] max-h-[90vh] p-6 rounded-xl flex flex-col overflow-hidden'):
        
        def reset():
            from services.character_manager import get_default_character_template
            editor.value = get_default_character_template()
            ui.notify("Template reset to system default.", type="info")

        def save():
            save_setting("character_profiler_template", editor.value)
            ui.notify("Custom profiler prompt template saved!", type="positive")
            dialog.close()

        with ui.row().classes('w-full justify-between items-center border-b pb-3 mb-3 shrink-0'):
            with ui.column().classes('gap-0.5'):
                ui.label('Customize Character Profiler Prompt').classes('text-base font-bold text-slate-800')
                ui.label('Configure system instructions sent to the local LLM.').classes('text-xs text-slate-500')
            
            with ui.row().classes('gap-2 items-center'):
                ui.button('Reset', on_click=reset, color='amber').props('flat').classes('text-xs font-semibold')
                ui.button('Cancel', on_click=dialog.close, color='slate').props('flat').classes('text-xs font-semibold')
                ui.button('Save Template', on_click=save).classes('bg-blue-600 text-white font-bold text-xs px-3 py-1.5 rounded-lg shadow-sm')

        with ui.column().classes('w-full flex-1 overflow-y-auto overflow-x-hidden gap-4 pr-1 min-w-0'):
            ui.markdown(
                "Configure the instructions sent to the LLM during character research. Placeholders:\n"
                "- `{character_name}`, `{aliases}`, `{known_traits}`, `{unknown_traits}`"
            ).classes('text-xs text-slate-500 leading-relaxed bg-slate-50 p-3 rounded-lg border w-full')

            editor = ui.textarea(
                label='System Instructions Template', 
                value=current_template
            ).classes('w-full font-mono text-xs').props('outlined autogrow')

    dialog.open()

def open_alias_explorer_dialog(project_id: int, alias: CharacterAlias, parent_char_id: int, refresh_callback: Any):
    """Opens a modal displaying where the selected alias occurs within transcripts."""
    from services.character_manager import get_alias_occurrences, save_project_characters_to_json
    
    occurrences = get_alias_occurrences(project_id, alias.alias)
    current_index = 0

    with ui.dialog() as dialog, ui.card().classes('w-[600px] max-w-[95vw] p-6 rounded-xl flex flex-col gap-4 overflow-hidden'):
        
        with ui.row().classes('w-full justify-between items-center border-b pb-3 shrink-0'):
            with ui.column().classes('gap-0.5'):
                ui.label(f'Context Explorer: "{alias.alias}"').classes('text-base font-bold text-slate-800')
                book_label = ui.label('Loading context...').classes('text-xs text-slate-500')
            ui.button(icon='close', on_click=dialog.close).props('flat dense').classes('text-slate-400')

        with ui.column().classes('w-full flex-1 justify-center items-center py-6 min-h-[160px] bg-slate-50 border rounded-lg px-4 overflow-y-auto'):
            context_html = ui.html('').classes('text-sm text-slate-700 leading-relaxed text-center')

        # Bottom Controls & Actions Row
        with ui.row().classes('w-full justify-between items-center pt-2 shrink-0 border-t mt-2'):
            with Session(engine) as session:
                char = session.get(Character, parent_char_id)
                char_name = char.name if char else ""
            
            show_promote = char_name.lower() != alias.alias.lower()
            
            if show_promote:
                def handle_promote():
                    with Session(engine) as session:
                        db_char = session.get(Character, parent_char_id)
                        db_alias = session.get(CharacterAlias, alias.id)
                        if db_char and db_alias:
                            old_name = db_char.name
                            new_name = db_alias.alias
                            
                            # 1. Update the character's canonical name
                            db_char.name = new_name
                            session.add(db_char)
                            
                            # 2. Check if the old name is already registered as an alias
                            exists = session.exec(
                                select(CharacterAlias)
                                .where(CharacterAlias.character_id == parent_char_id)
                                .where(CharacterAlias.alias == old_name)
                            ).first()
                            
                            # Swap old name into this alias record, or discard it if old_name is already a redundant alias
                            if not exists:
                                db_alias.alias = old_name
                                session.add(db_alias)
                            else:
                                session.delete(db_alias)
                                
                            session.commit()
                    
                    save_project_characters_to_json(project_id)
                    ui.notify(f"Promoted '{new_name}' to canonical character name!", type="positive")
                    dialog.close()
                    refresh_callback()

                ui.button(
                    'Promote to Character Name', 
                    icon='upgrade', 
                    on_click=handle_promote
                ).classes('bg-amber-600 hover:bg-amber-700 text-white font-bold text-xs px-3 py-1.5 rounded-lg')
            else:
                ui.label('Canonical Character Name').classes('text-xs text-slate-400 italic font-semibold')

            with ui.row().classes('gap-3 items-center'):
                prev_btn = ui.button(icon='chevron_left', on_click=lambda: navigate(-1)).props('flat dense').classes('bg-slate-100 p-1 rounded-lg')
                counter_label = ui.label('0 of 0').classes('text-xs font-bold text-slate-600')
                next_btn = ui.button(icon='chevron_right', on_click=lambda: navigate(1)).props('flat dense').classes('bg-slate-100 p-1 rounded-lg')

        def navigate(direction: int):
            nonlocal current_index
            new_idx = current_index + direction
            if 0 <= new_idx < len(occurrences):
                current_index = new_idx
                update_display()

        def update_display():
            if not occurrences:
                context_html.content = "<span class='text-slate-400 italic'>No literal transcript occurrences found for this alias.</span>"
                counter_label.text = "0 of 0"
                book_label.text = "No matches found"
                prev_btn.disable()
                next_btn.disable()
                return
            
            occ = occurrences[current_index]
            context_html.content = occ["html_context"]
            counter_label.text = f"{current_index + 1} of {len(occurrences)}"
            book_label.text = f"Source: {occ['book_name']}"
            
            if current_index > 0:
                prev_btn.enable()
            else:
                prev_btn.disable()
                
            if current_index < len(occurrences) - 1:
                next_btn.enable()
            else:
                next_btn.disable()

        update_display()

    dialog.open()

def get_character_prompt_occurrences(project_name: str, book_name: str, alias_texts: Set[str]) -> List[Dict[str, Any]]:
    """Reads prompts.csv and extracts all rows where this character or their aliases are mentioned in brackets."""
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    csv_path = base_output_dir / project_name / book_name / "prompts.csv"
    if not csv_path.exists():
        return []
    
    bracket_regex = re.compile(r"\[(.*?)\]")
    occurrences = []
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            rows = list(reader)
    except Exception as e:
        print(f"[Map] Error reading prompts.csv: {e}")
        return []

    for idx, row in enumerate(rows):
        prompt_text = row.get("prompt", "")
        matched = False
        
        # Check bracketed tags explicitly
        for match in bracket_regex.findall(prompt_text):
            if match.strip().lower() in alias_texts:
                matched = True
                break
        
        if matched:
            occurrences.append({
                "global_row_index": idx,
                "chapter": row.get("chapter") or row.get("chapter_num") or row.get("Chapter") or "N/A",
                "scene": row.get("scene") or row.get("scene_num") or row.get("Scene") or "N/A",
                "quote": row.get("quote") or row.get("Quote") or "",
                "prompt": prompt_text
            })
            
    return occurrences


def save_prompt_occurrence(project_name: str, book_name: str, global_row_index: int, new_quote: str, new_prompt: str) -> bool:
    """Saves edited quote and prompt values back to prompts.csv at the correct row index."""
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    csv_path = base_output_dir / project_name / book_name / "prompts.csv"
    if not csv_path.exists():
        return False
        
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            fieldnames = reader.fieldnames or []
            rows = list(reader)
            
        if 0 <= global_row_index < len(rows):
            # Normalize column headers
            quote_key = next((k for k in ["quote", "Quote"] if k in rows[global_row_index]), None) or "quote"
            prompt_key = next((k for k in ["prompt", "Prompt"] if k in rows[global_row_index]), None) or "prompt"
            
            rows[global_row_index][quote_key] = new_quote
            rows[global_row_index][prompt_key] = new_prompt
            
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|")
                writer.writeheader()
                writer.writerows(rows)
            return True
    except Exception as e:
        print(f"[Map] Error saving to prompts.csv: {e}")
    return False


def open_appearance_prompt_modal(project: Project, character: Character, book_name: str):
    """Spawns an interactive sliding/clicking dialog to view and edit book prompts where this character is tagged."""
    with Session(engine) as session:
        aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == character.id)).all()
        alias_texts = {a.alias.lower().strip() for a in aliases}
        if not alias_texts:
            alias_texts = {character.name.lower().strip()}

    occurrences = get_character_prompt_occurrences(project.name, book_name, alias_texts)
    current_index = 0

    with ui.dialog() as dialog, ui.card().classes('w-[700px] max-w-[95vw] p-6 rounded-xl flex flex-col gap-4 overflow-hidden'):
        
        with ui.row().classes('w-full justify-between items-center border-b pb-3 shrink-0'):
            with ui.column().classes('gap-0.5'):
                ui.label(f'Prompt Map: {character.name} in "{book_name}"').classes('text-base font-bold text-slate-800')
                status_label = ui.label('Loading prompts...').classes('text-xs text-slate-500')
            ui.button(icon='close', on_click=dialog.close).props('flat dense').classes('text-slate-400')

        if not occurrences:
            with ui.column().classes('w-full flex-1 justify-center items-center py-12 bg-slate-50 border rounded-lg px-4'):
                ui.icon('search_off', size='lg', color='slate-300')
                ui.label("No bracketed occurrences found in this book's prompts.csv.").classes('text-sm text-slate-500 mt-2 text-center')
                ui.button('Close', on_click=dialog.close).classes('mt-4 text-xs font-bold bg-slate-100 text-slate-700 border')
            dialog.open()
            return

        with ui.column().classes('w-full flex-1 gap-3 overflow-y-auto pr-1 min-h-[300px]'):
            with ui.row().classes('w-full bg-blue-50/40 p-3 rounded-lg border border-blue-100 items-center justify-between'):
                coordinate_label = ui.label('').classes('text-xs font-extrabold text-blue-700')
                counter_label = ui.label('').classes('text-xs font-bold text-blue-700 bg-blue-100/50 px-2 py-0.5 rounded-full')
            
            quote_input = ui.textarea(
                label="Scene Quote (Context Material)",
                placeholder="The spoken context..."
            ).classes('w-full bg-white').props('outlined dense autogrow')

            prompt_input = ui.textarea(
                label="Image Generation Prompt",
                placeholder="Descriptive text..."
            ).classes('w-full bg-white font-mono text-xs').props('outlined dense autogrow')

        with ui.row().classes('w-full justify-between items-center border-t pt-3 shrink-0 mt-2'):
            async def save_active_changes():
                occ = occurrences[current_index]
                new_q = quote_input.value.strip()
                new_p = prompt_input.value.strip()
                
                success = await asyncio.to_thread(
                    save_prompt_occurrence, 
                    project.name, 
                    book_name, 
                    occ["global_row_index"], 
                    new_q, 
                    new_p
                )
                if success:
                    occurrences[current_index]["quote"] = new_q
                    occurrences[current_index]["prompt"] = new_p
                    ui.notify("Prompt and Quote changes saved to disk!", type="positive")
                else:
                    ui.notify("Failed to save changes.", type="negative")

            ui.button(
                'Save Changes', 
                icon='save', 
                on_click=save_active_changes
            ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-4 py-2 rounded-lg shadow-sm')

            with ui.row().classes('gap-2 items-center'):
                prev_btn = ui.button(icon='chevron_left', on_click=lambda: navigate(-1)).props('flat dense').classes('bg-slate-100 p-1 rounded-lg')
                counter_label_nav = ui.label('').classes('text-xs font-bold text-slate-600 px-2')
                next_btn = ui.button(icon='chevron_right', on_click=lambda: navigate(1)).props('flat dense').classes('bg-slate-100 p-1 rounded-lg')

        def navigate(direction: int):
            nonlocal current_index
            new_idx = current_index + direction
            if 0 <= new_idx < len(occurrences):
                current_index = new_idx
                update_display()

        def update_display():
            occ = occurrences[current_index]
            coordinate_label.text = f"Coordinates: Chapter {occ['chapter']}, Scene {occ['scene']}"
            counter_label.text = f"{current_index + 1} of {len(occurrences)}"
            counter_label_nav.text = f"{current_index + 1} / {len(occurrences)}"
            status_label.text = f"Showing mentions in: {book_name}"
            
            quote_input.value = occ["quote"]
            prompt_input.value = occ["prompt"]
            
            if current_index > 0:
                prev_btn.enable()
            else:
                prev_btn.disable()
                
            if current_index < len(occurrences) - 1:
                next_btn.enable()
            else:
                next_btn.disable()

        update_display()

    dialog.open()

def open_import_profiles_dialog(project: Project, matching_projects: List[Dict[str, Any]], refresh_callback: Any):
    """Spawns an interactive reconciliation grid to copy matching character profiles from prior projects."""
    selected_source_project_id = matching_projects[0]["id"]
    
    # Track selection states
    checked_pairs: Dict[int, bool] = {}
    active_pairings: Dict[int, Dict[str, Any]] = {}

    def truncate_desc(text: str, max_len: int = 65) -> str:
        if not text or not text.strip():
            return "No traits profiled yet."
        clean = text.strip()
        if len(clean) > max_len:
            return clean[:max_len] + "..."
        return clean

    def format_aliases(aliases_list: List[str], max_len: int = 30) -> str:
        if not aliases_list:
            return ""
        joined = ", ".join(aliases_list)
        if len(joined) > max_len:
            return f" ({joined[:max_len]}...)"
        return f" ({joined})"

    with ui.dialog() as dialog, ui.card().classes('w-[700px] max-w-[95vw] h-[600px] max-h-[90vh] p-6 rounded-xl flex flex-col overflow-hidden'):
        
        with ui.row().classes('w-full justify-between items-center border-b pb-3 shrink-0'):
            with ui.column().classes('gap-0.5'):
                ui.label('Import Character Profiles').classes('text-base font-bold text-slate-800')
                ui.label('Reconcile and sync baseline traits from completed projects.').classes('text-xs text-slate-500')
            ui.button(icon='close', on_click=dialog.close).props('flat dense').classes('text-slate-400')

        # Dropdowns & Config Toggles
        with ui.row().classes('w-full items-center justify-between bg-slate-50 p-3 rounded-lg border gap-3 shrink-0'):
            with ui.column().classes('gap-0.5 flex-1'):
                ui.label('Source Project').classes('text-[10px] font-bold text-slate-400 uppercase')
                
                project_options = {p["id"]: p["name"] for p in matching_projects}
                def on_source_project_change(e):
                    nonlocal selected_source_project_id
                    selected_source_project_id = e.value
                    draw_matching_rows.refresh()

                ui.select(
                    options=project_options,
                    value=selected_source_project_id,
                    on_change=on_source_project_change
                ).classes('w-full bg-white').props('outlined dense')

            with ui.column().classes('gap-1 shrink-0 items-start'):
                lock_checkbox = ui.checkbox('Lock profiles after import', value=True).classes('text-xs')
                merge_aliases_checkbox = ui.checkbox('Import & Merge Aliases (Rename target)', value=True).classes('text-xs')

        @ui.refreshable
        def draw_matching_rows():
            checked_pairs.clear()
            active_pairings.clear()
            
            matches = get_character_import_matches(project.id, selected_source_project_id)
            if not matches:
                with ui.column().classes('w-full justify-center items-center py-12 gap-2'):
                    ui.icon('check_circle', size='lg', color='green-400')
                    ui.label('No unmatched characters match between these projects.').classes('text-xs text-slate-500 italic text-center')
                return

            with ui.column().classes('w-full gap-2'):
                for match in matches:
                    tgt_id = match["target_char_id"]
                    checked_pairs[tgt_id] = True  # Checked by default
                    active_pairings[tgt_id] = match
                    
                    with ui.row().classes('w-full items-center justify-between p-3 bg-white border rounded-xl gap-3 shadow-sm hover:border-slate-300 transition-colors'):
                        cb = ui.checkbox(value=True).classes('shrink-0')
                        cb.bind_value_to(checked_pairs, tgt_id)
                        
                        # Source display (Left)
                        with ui.column().classes('flex-1 min-w-0 gap-0.5'):
                            alias_suffix = format_aliases(match["source_aliases"])
                            ui.label(f"{match['source_name']}{alias_suffix}").classes('text-xs font-bold text-slate-800 truncate w-full')
                            ui.label(truncate_desc(match["source_desc"])).classes('text-[10px] text-slate-500 italic truncate w-full')
                            
                        # Separator arrow
                        ui.icon('arrow_forward', size='xs', color='slate-400').classes('shrink-0')
                        
                        # Target display (Right)
                        with ui.column().classes('flex-1 min-w-0 gap-0.5'):
                            t_alias_suffix = format_aliases(match["target_aliases"])
                            with ui.row().classes('items-center gap-2 w-full'):
                                ui.label(f"{match['target_name']}{t_alias_suffix}").classes('text-xs font-semibold text-slate-700 truncate')
                                ui.badge(f"{match['target_mentions']} hits", color='blue-50').classes('text-[9px] font-bold text-blue-700 px-1 py-0.5 rounded')
                            ui.label(truncate_desc(match["target_desc"])).classes('text-[10px] text-slate-400 italic truncate w-full')

        # Reconciliation table area
        with ui.column().classes('w-full flex-1 overflow-y-auto min-h-0 bg-slate-50/50 p-3 rounded-lg border gap-2'):
            ui.label('Verify Matches & Overrides').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1 shrink-0')
            draw_matching_rows()

        # Modal Action Row
        with ui.row().classes('w-full justify-end gap-3 border-t pt-3 shrink-0 mt-2'):
            async def run_import():
                pairings_to_execute = []
                for tgt_id, checked in checked_pairs.items():
                    if checked and tgt_id in active_pairings:
                        pairings_to_execute.append(active_pairings[tgt_id])
                
                if not pairings_to_execute:
                    ui.notify("No pairings selected for import.", type="warning")
                    return
                    
                await asyncio.to_thread(
                    execute_character_import,
                    tgt_project_id=project.id,
                    pairings=pairings_to_execute,
                    lock_after_import=lock_checkbox.value,
                    import_merge_aliases=merge_aliases_checkbox.value
                )
                
                ui.notify(f"Successfully imported {len(pairings_to_execute)} character profiles!", type="positive")
                dialog.close()
                await refresh_callback()

            ui.button('Cancel', on_click=dialog.close).props('flat').classes('text-xs font-semibold text-slate-500')
            ui.button(
                'Import Selected',
                icon='cloud_download',
                on_click=run_import
            ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-4 py-2 rounded-lg shadow-sm')

    dialog.open()


def render_characters_tab(project: Project, books: List[Book], refresh_parent: Optional[Any] = None):
    # Enforce chronological self-healing index alignment on draw
    ensure_book_orders(project.id)

    async def restore_scroll_position():
        """Scrolls the currently active/selected character row into view within the container."""
        js_code = """
        (() => {
            let attempts = 0;
            let timer = setInterval(() => {
                let active = document.querySelector('.char-scroll-list .bg-blue-50');
                if (active) {
                    active.scrollIntoView({ block: 'nearest', behavior: 'auto' });
                    clearInterval(timer);
                }
                attempts++;
                if (attempts > 20) {
                    clearInterval(timer);
                }
            }, 20);
        })();
        """
        try:
            await ui.run_javascript(js_code, timeout=1.0)
        except Exception:
            pass

    async def refresh_workspace_with_scroll():
        """Refreshes individual active components in-place and aligns the active selection."""
        global workspace_was_empty
        with Session(engine) as session:
            any_characters = session.exec(
                select(Character).where(Character.project_id == project.id)
                .where(Character.locked == False)
            ).first()

        current_empty = (any_characters is None)
        if current_empty != workspace_was_empty:
            draw_workspace_layout.refresh()
        else:
            draw_header_toolbar.refresh()
            draw_stats_bar.refresh()
            draw_character_list.refresh()
            draw_details_panel.refresh()
        
        await restore_scroll_position()

    async def refresh_list_with_scroll():
        """Refreshes only the list view and aligns the active selection."""
        draw_character_list.refresh()
        await restore_scroll_position()

    def select_char(c_id):
        """Changes focus and toggles selection styles without rebuilding the scrolling list element."""
        global selected_character_id, selected_event_id
        old_id = selected_character_id
        selected_character_id = c_id
        
        # Reset the selected event state to the baseline event of the newly focused character
        with Session(engine) as session:
            base_ev = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == c_id)
                .where(CharacterTimelineEvent.book_id == None)
            ).first()
            if base_ev:
                selected_event_id = base_ev.id
            else:
                selected_event_id = None
        
        if old_id in row_elements and row_elements[old_id]:
            try:
                row_elements[old_id].classes(
                    add='hover:bg-slate-50 text-slate-700 border-l border-slate-100',
                    remove='bg-blue-50 border-l-4 border-blue-600 font-semibold text-blue-900'
                )
            except Exception:
                pass
                
        if c_id in row_elements and row_elements[c_id]:
            try:
                row_elements[c_id].classes(
                    add='bg-blue-50 border-l-4 border-blue-600 font-semibold text-blue-900',
                    remove='hover:bg-slate-50 text-slate-700 border-l border-slate-100'
                )
            except Exception:
                pass
                
        draw_details_panel.refresh()

    @ui.refreshable
    def draw_header_toolbar():
        global selected_book_id, is_profiling_all, currently_profiling_char_id, profiling_progress, profiler_scan_depth, cancel_profiling_all
        
        with ui.column().classes('w-full bg-slate-50 border p-4 rounded-xl mb-4 gap-3'):
            # Line 1: Automated LLM Batch Profiling Suite
            with ui.row().classes('w-full items-center justify-between flex-wrap gap-3'):
                with ui.row().classes('items-center gap-3'):
                    ui.label('Batch Profiler Suite').classes('text-xs font-bold text-slate-400 uppercase tracking-wider')
                    
                    ui.label('Source:').classes('text-xs font-semibold text-slate-500 ml-2')
                    book_options = {None: "All Books (Project-wide)"}
                    for b in books:
                        book_options[b.id] = b.name
                    
                    def handle_book_change(val):
                        global selected_book_id
                        selected_book_id = val
                        
                    ui.select(
                        options=book_options,
                        value=selected_book_id,
                        on_change=lambda e: handle_book_change(e.value)
                    ).classes('w-48 bg-white').props('outlined dense')

                    ui.label('Depth:').classes('text-xs font-semibold text-slate-500 ml-1')
                    def handle_depth_change(e):
                        global profiler_scan_depth
                        profiler_scan_depth = int(e.value)

                    ui.number(
                        value=profiler_scan_depth,
                        min=1,
                        max=100,
                        step=1,
                        on_change=handle_depth_change
                    ).classes('w-14 bg-white').props('outlined dense')

                # Render "Profile All" button only if NOT actively profiling to keep controls neat
                if not is_profiling_all:
                    ui.button(
                        'Profile All', 
                        icon='bolt', 
                        on_click=lambda: open_batch_profiler_dialog(
                            project, 
                            books, 
                            refresh_workspace_with_scroll, 
                            draw_header_toolbar.refresh,
                            draw_details_panel.refresh
                        )
                    ).classes('bg-purple-600 hover:bg-purple-700 text-white font-bold text-xs px-4 py-2 rounded-lg shadow-sm')

            # Line 1.5: Dedicated Active Profiling Row (Visible only during execution)
            if is_profiling_all:
                with ui.row().classes('w-full items-center gap-3 bg-purple-50/40 border border-purple-200 p-3 rounded-lg mb-1 shrink-0'):
                    def stop_profiling():
                        global cancel_profiling_all, profiling_progress
                        cancel_profiling_all = True
                        profiling_progress = "Stopping..."
                        draw_header_toolbar.refresh()
                        ui.notify("Stop requested...", type="warning")

                    # The Stop button is anchored completely to the left, keeping its position absolute and immutable
                    ui.button(
                        'Stop Batch', 
                        icon='stop', 
                        on_click=stop_profiling
                    ).classes('bg-red-600 hover:bg-red-700 text-white font-bold text-xs px-3 py-1.5 rounded-lg shadow-sm shrink-0')

                    ui.spinner(size='xs', color='purple').classes('shrink-0')
                    ui.label(profiling_progress).classes('text-xs font-bold text-purple-700 animate-pulse truncate flex-1')

            # Line 2: Manual Curation Controls
            with ui.row().classes('w-full items-center justify-between border-t pt-3 border-slate-200 flex-wrap gap-2'):
                with ui.row().classes('items-center gap-3'):
                    ui.label('Curation Tools').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wider')
                    
                    async def run_prompt_scan():
                        client = ui.context.client
                        with client:
                            ui.notify("Scanning prompts.csv for character tags...", type="info")
                        tags = await asyncio.to_thread(extract_characters_from_prompts, project.id)
                        with client:
                            if tags:
                                ui.notify(f"Discovered and indexed {len(tags)} character tags!", type="positive")
                            else:
                                ui.notify("No new bracketed character tags found in prompts.csv.", type="info")
                        await refresh_workspace_with_scroll()

                    ui.button(
                        'Scan Tags', 
                        icon='tag', 
                        on_click=run_prompt_scan
                    ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-3 py-1.5 rounded-lg shadow-sm')\
                    .tooltip("Scan prompts.csv for bracketed character tags")

                    async def run_auto_merge():
                        client = ui.context.client
                        with client:
                            ui.notify("Running smart auto-merge of character tags...", type="info")
                        merged_log = await asyncio.to_thread(auto_merge_project_characters, project.id)
                        with client:
                            if merged_log:
                                ui.notify(f"Auto-merged {len(merged_log)} duplicate tags!", type="positive")
                            else:
                                ui.notify("No matching alias tags to merge found.", type="info")
                        await refresh_workspace_with_scroll()

                    ui.button(
                        'Auto-Merge',
                        icon='merge_type',
                        on_click=run_auto_merge
                    ).classes('bg-teal-600 hover:bg-teal-700 text-white font-bold text-xs px-3 py-1.5 rounded-lg shadow-sm')\
                    .tooltip("Fuzzy-merge common names, titles, and possessives")

                    # "Import Profiles" tool button is permanently visible to prevent layout shift
                    def try_open_import():
                        matching_projects = get_matching_source_projects(project.id)
                        if not matching_projects:
                            ui.notify("No completed projects with matching character tags found.", type="info")
                        else:
                            open_import_profiles_dialog(project, matching_projects, refresh_workspace_with_scroll)

                    ui.button(
                        'Import Profiles',
                        icon='cloud_download',
                        on_click=try_open_import
                    ).classes('bg-amber-600 hover:bg-amber-700 text-white font-bold text-xs px-3 py-1.5 rounded-lg shadow-sm')\
                    .tooltip("Import curated profiles from overlapping project(s)")

                # Settings/Prompt Button aligned right
                ui.button(
                    'Prompt Template',
                    icon='edit_note',
                    on_click=open_prompt_editor_dialog
                ).classes('bg-slate-600 hover:bg-slate-700 text-white font-bold text-xs px-3 py-1.5 rounded-lg shadow-sm')\
                .tooltip("Customize visual profiler LLM instructions")


    @ui.refreshable
    def draw_stats_bar():
        with Session(engine) as session:
            all_characters = session.exec(
                select(Character).where(Character.project_id == project.id)
            ).all()
        
        total_chars = len(all_characters)
        fully_profiled = 0
        locked_count = 0
        
        with Session(engine) as session:
            for char in all_characters:
                if char.locked:
                    locked_count += 1
                base_ev = session.exec(
                    select(CharacterTimelineEvent)
                    .where(CharacterTimelineEvent.character_id == char.id)
                    .where(CharacterTimelineEvent.book_id == None)
                ).first()
                if base_ev:
                    fields = [
                        base_ev.demographics, base_ev.physical_build, 
                        base_ev.hair_and_face, base_ev.distinguishing_marks
                    ]
                    if sum(1 for f in fields if f and str(f).strip()) == 4:
                        fully_profiled += 1

        with ui.row().classes('w-full items-center gap-4 bg-blue-50/50 border border-blue-100 p-3 rounded-xl mb-4 text-xs font-semibold text-blue-700'):
            ui.icon('info', size='xs')
            ui.label(f"Database Stats: {total_chars} total characters discovered.")
            ui.label(f"|  {fully_profiled} fully profiled baseline (4/4 traits)")
            ui.label(f"|  {locked_count} locked/manually curated")

    @ui.refreshable
    def draw_character_list():
        global selected_character_id, search_query, sort_by, filter_status
        row_elements.clear()
        frequencies = get_character_frequency_map(project.name, books)

        with Session(engine) as session:
            all_characters = session.exec(
                select(Character).where(Character.project_id == project.id)
            ).all()
            
            char_aliases: Dict[int, List[CharacterAlias]] = {}
            for char in all_characters:
                aliases = session.exec(
                    select(CharacterAlias).where(CharacterAlias.character_id == char.id)
                ).all()
                char_aliases[char.id] = aliases

        def get_char_mentions(char_obj, aliases_list):
            total = 0
            for a in aliases_list:
                total += frequencies.get(a.alias.lower(), 0)
            if not aliases_list:
                total = frequencies.get(char_obj.name.lower(), 0)
            return total

        char_data_list = []
        with Session(engine) as session:
            for char in all_characters:
                aliases_list = char_aliases.get(char.id, [])
                mentions = get_char_mentions(char, aliases_list)
                
                base_ev = session.exec(
                    select(CharacterTimelineEvent)
                    .where(CharacterTimelineEvent.character_id == char.id)
                    .where(CharacterTimelineEvent.book_id == None)
                ).first()
                
                if base_ev:
                    fields = [
                        base_ev.demographics, base_ev.physical_build,
                        base_ev.hair_and_face, base_ev.distinguishing_marks
                    ]
                    completion_count = sum(1 for f in fields if f and str(f).strip())
                    summary_pieces = []
                    if base_ev.demographics: summary_pieces.append(base_ev.demographics)
                    if base_ev.hair_and_face: summary_pieces.append(base_ev.hair_and_face)
                    if base_ev.physical_build: summary_pieces.append(base_ev.physical_build)
                else:
                    completion_count = 0
                    summary_pieces = []
                    
                char_data_list.append((char, aliases_list, mentions, completion_count, summary_pieces))

        filtered_list = []
        q = search_query.lower().strip()
        for char, aliases_list, mentions, completion_count, summary_pieces in char_data_list:
            if q:
                alias_texts = [a.alias.lower() for a in aliases_list]
                name_match = q in char.name.lower()
                alias_match = any(q in t for t in alias_texts)
                if not (name_match or alias_match):
                    continue
            
            if filter_status == "incomplete" and completion_count == 4:
                continue
            elif filter_status == "locked" and not char.locked:
                continue
            elif filter_status == "unlocked" and char.locked:
                continue
                
            filtered_list.append((char, aliases_list, mentions, completion_count, summary_pieces))

        if sort_by == "mentions_desc":
            filtered_list.sort(key=lambda x: x[2], reverse=True)
        elif sort_by == "mentions_asc":
            filtered_list.sort(key=lambda x: x[2])
        elif sort_by == "name_asc":
            filtered_list.sort(key=lambda x: x[0].name.lower())
        elif sort_by == "name_desc":
            filtered_list.sort(key=lambda x: x[0].name.lower(), reverse=True)
        elif sort_by == "completion_desc":
            filtered_list.sort(key=lambda x: x[3], reverse=True)

        if selected_character_id is None and filtered_list:
            selected_character_id = filtered_list[0][0].id

        if not filtered_list:
            ui.label('No characters match filters.').classes('text-xs text-slate-400 text-center py-8 w-full')
        else:
            for char, aliases_list, mentions, completion_count, summary_pieces in filtered_list:
                is_selected = char.id == selected_character_id
                bg_class = "bg-blue-50 border-l-4 border-blue-600 font-semibold text-blue-900" if is_selected else "hover:bg-slate-50 text-slate-700"
                border_class = "" if is_selected else "border-l border-slate-100"
                
                row_el = ui.row().classes(f'w-full p-2.5 rounded-lg cursor-pointer transition-colors justify-between items-center {bg_class} {border_class}')
                row_elements[char.id] = row_el
                
                with row_el.on('click', lambda _, c_id=char.id: select_char(c_id)):
                    with ui.column().classes('gap-0.5 flex-1 min-w-0'):
                        with ui.row().classes('items-center gap-1.5 min-w-0 w-full'):
                            if char.locked:
                                ui.icon('lock', size='12px', color='rose-500').tooltip('Locked')
                            else:
                                ui.icon('face', size='14px', color='slate-400')
                            ui.label(char.name).classes('text-xs truncate font-semibold')
                        
                        summary_text = " • ".join(summary_pieces) if summary_pieces else "No traits profiled yet"
                        ui.label(summary_text).classes('text-[10px] text-slate-400 truncate w-full')
                    
                    with ui.column().classes('items-end gap-1'):
                        ui.label(f"{mentions} hits").classes('text-[10px] font-bold bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded')
                        bar_color = "text-green-600 font-bold" if completion_count == 4 else "text-purple-600" if completion_count >= 2 else "text-slate-400"
                        ui.label(f"{completion_count}/4 traits").classes(f'text-[9px] font-bold {bar_color}')

    @ui.refreshable
    def draw_details_panel():
        global selected_character_id, selected_event_id, selected_book_id, profiler_scan_depth, currently_profiling_char_id
        
        if selected_character_id is None:
            with ui.column().classes('w-full h-full items-center justify-center text-slate-400 gap-4'):
                ui.icon('person_search', size='xl', color='slate-300')
                ui.label('No Character Selected').classes('text-sm font-bold text-slate-500')
                ui.label('Choose a character from the left panel list to view and edit details.').classes('text-xs text-slate-400 max-w-xs text-center')
            return

        with Session(engine) as session:
            char = session.get(Character, selected_character_id)
            if not char:
                ui.label('Character not found.').classes('text-xs text-slate-400 text-center py-8 w-full')
                return

            # Resolve/Safeguard Active Selected Event State
            if selected_event_id is None:
                base_ev = session.exec(
                    select(CharacterTimelineEvent)
                    .where(CharacterTimelineEvent.character_id == char.id)
                    .where(CharacterTimelineEvent.book_id == None)
                ).first()
                if base_ev:
                    selected_event_id = base_ev.id
                else:
                    base_ev = CharacterTimelineEvent(
                        character_id=char.id,
                        book_id=None,
                        chapter_num=0,
                        scene_num=0,
                        label="Base State"
                    )
                    session.add(base_ev)
                    session.commit()
                    selected_event_id = base_ev.id

            active_event = session.get(CharacterTimelineEvent, selected_event_id)
            if not active_event or active_event.character_id != char.id:
                base_ev = session.exec(
                    select(CharacterTimelineEvent)
                    .where(CharacterTimelineEvent.character_id == char.id)
                    .where(CharacterTimelineEvent.book_id == None)
                ).first()
                selected_event_id = base_ev.id if base_ev else None
                active_event = base_ev

            aliases = session.exec(
                select(CharacterAlias).where(CharacterAlias.character_id == char.id)
            ).all()

            frequencies = get_character_frequency_map(project.name, books)
            def get_char_mentions(char_obj, aliases_list):
                total = 0
                for a in aliases_list:
                    total += frequencies.get(a.alias.lower(), 0)
                if not aliases_list:
                    total = frequencies.get(char_obj.name.lower(), 0)
                return total
            mentions = get_char_mentions(char, aliases)

            fields_list = [
                active_event.demographics, active_event.physical_build,
                active_event.hair_and_face, active_event.distinguishing_marks
            ]
            completion_count = sum(1 for f in fields_list if f and str(f).strip())

            all_characters = session.exec(
                select(Character).where(Character.project_id == project.id)
            ).all()

        with ui.row().classes('w-full justify-between items-center pb-3 border-b flex-wrap gap-3'):
            with ui.row().classes('items-center gap-3'):
                ui.icon('face', size='md', color='blue-600')
                
                async def handle_name_blur(e, char_id=char.id):
                    new_name = e.sender.value.strip()
                    if not new_name:
                        return
                    with Session(engine) as session:
                        db_char = session.get(Character, char_id)
                        if db_char:
                            db_char.name = new_name
                            session.add(db_char)
                            session.commit()
                    save_project_characters_to_json(project.id)
                    ui.notify(f"Renamed profile to: {new_name}", type="info")
                    await refresh_list_with_scroll()
                    draw_details_panel.refresh()

                ui.input(
                    value=char.name
                ).classes('w-64 font-extrabold text-lg text-slate-800').props('dense borderless').on('blur', handle_name_blur)
                
                ui.badge(f'{mentions} total mentions', color='blue-50').classes('text-blue-700 text-xs font-bold px-2.5 py-1 rounded-full')
                
            # Flex-grow right-side container to stretch and support absolute right-aligned items
            with ui.row().classes('items-center gap-2 flex-grow justify-end'):
                # Sub-group to keep standard actions clustered together
                with ui.row().classes('items-center gap-2'):
                    async def scan_single_char():
                        global currently_profiling_char_id, profiler_scan_depth, selected_event_id
                        client = ui.context.client
                        
                        if char.locked:
                            with client:
                                ui.notify("Character is locked from being profiled, unlock and try again.", type="warning")
                            return

                        currently_profiling_char_id = char.id
                        draw_details_panel.refresh()
                        
                        try:
                            with client:
                                ui.notify(f"Running LLM research pipeline for {char.name}...", type="info")
                            await run_stateful_character_profiling(
                                project.id, char.id, selected_book_id, 
                                max_chunks_to_scan=profiler_scan_depth, event_id=selected_event_id
                            )
                            with client:
                                ui.notify("Profiling completed successfully!", type="positive")
                        except Exception as ex:
                            with client:
                                ui.notify(f"Profiling failed: {str(ex)}", type="negative")
                        
                        currently_profiling_char_id = None
                        await refresh_workspace_with_scroll()

                    async def speculate_single_char():
                        global currently_profiling_char_id, profiler_scan_depth, selected_event_id
                        client = ui.context.client
                        
                        if char.locked:
                            with client:
                                ui.notify("Character is locked from being profiled, unlock and try again.", type="warning")
                            return

                        currently_profiling_char_id = char.id
                        draw_details_panel.refresh()
                        
                        try:
                            with client:
                                ui.notify(f"Speculating character casting vibe for {char.name}...", type="info")
                            await run_stateful_character_profiling(
                                project.id, char.id, selected_book_id, 
                                max_chunks_to_scan=profiler_scan_depth, speculate=True, event_id=selected_event_id
                            )
                            with client:
                                ui.notify("Casting speculation completed!", type="positive")
                        except Exception as ex:
                            with client:
                                ui.notify(f"Speculation failed: {str(ex)}", type="negative")
                        
                        currently_profiling_char_id = None
                        await refresh_workspace_with_scroll()

                    is_card_profiling = currently_profiling_char_id == char.id
                    if is_card_profiling:
                        with ui.row().classes('items-center gap-1.5 bg-purple-50 px-3 py-1.5 rounded-lg border border-purple-200'):
                            ui.spinner(size='xs', color='purple')
                            ui.label('LLM Active...').classes('text-xs text-purple-700 font-bold')
                    else:
                        ui.button(
                            'Research (LLM)', 
                            icon='science', 
                            on_click=scan_single_char
                        ).classes('text-white font-bold text-xs bg-purple-600 hover:bg-purple-700').tooltip("Scan actual, written physical descriptions.")
                        
                        ui.button(
                            'Deduce Vibe', 
                            icon='theater_comedy', 
                            on_click=speculate_single_char
                        ).classes('text-white font-bold text-xs bg-indigo-600 hover:bg-indigo-700').tooltip("Deduce characteristics when details are unwritten.")

                    def toggle_locked(c_id=char.id, val=not char.locked):
                        with Session(engine) as session:
                            db_char = session.get(Character, c_id)
                            if db_char:
                                db_char.locked = val
                                session.add(db_char)
                                session.commit()
                        save_project_characters_to_json(project.id)
                        draw_character_list.refresh()
                        draw_details_panel.refresh()
                        draw_stats_bar.refresh()
                        ui.notify(f"Profile {'Locked' if val else 'Unlocked'}!", type="info")

                    lock_icon = "lock" if char.locked else "lock_open"
                    lock_color = "bg-rose-50 text-rose-600 hover:bg-rose-100" if char.locked else "bg-slate-100 text-slate-600 hover:bg-slate-200"
                    ui.button(
                        icon=lock_icon, 
                        on_click=lambda c_id=char.id: toggle_locked(c_id)
                    ).props('flat dense').classes(f'p-1.5 rounded-lg {lock_color}').tooltip('Toggle manual editing lock')

                # Fills the middle space, pushing the delete action to the absolute right
                ui.space()

                async def delete_profile(c_id=char.id):
                    global selected_character_id
                    with Session(engine) as session:
                        db_char = session.get(Character, c_id)
                        if db_char:
                            aliases_to_del = session.exec(
                                select(CharacterAlias).where(CharacterAlias.character_id == c_id)
                            ).all()
                            for a in aliases_to_del:
                                session.delete(a)

                            evs_to_del = session.exec(
                                select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id == c_id)
                            ).all()
                            for ev in evs_to_del:
                                session.delete(ev)

                            session.delete(db_char)
                            session.commit()
                    save_project_characters_to_json(project.id)
                    selected_character_id = None
                    await refresh_workspace_with_scroll()
                    ui.notify("Character profile deleted.", type="warning")

                # Standout red button to prevent accidental misclicks
                ui.button(
                    icon='delete', 
                    on_click=delete_profile,
                    color='red'
                ).props('unelevated dense').classes('p-2 rounded-lg text-white').tooltip('Delete Character Profile')

        with ui.column().classes('w-full flex-1 overflow-y-auto gap-4 pr-1'):
            
            # --- Row 1.5: Timeline State Switcher Row (New) ---
            with ui.row().classes('w-full items-center justify-between bg-slate-50 border p-3 rounded-xl gap-3'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('Timeline State:').classes('text-xs font-bold text-slate-500')
                    
                    with Session(engine) as session:
                        all_evs = session.exec(
                            select(CharacterTimelineEvent)
                            .where(CharacterTimelineEvent.character_id == char.id)
                        ).all()
                    
                    dropdown_options = {}
                    for ev in all_evs:
                        if ev.book_id is None:
                            dropdown_options[ev.id] = "Base State (Initial)"
                        else:
                            with Session(engine) as session:
                                b = session.get(Book, ev.book_id)
                                b_name = b.name if b else f"Book {ev.book_id}"
                            dropdown_options[ev.id] = f"{b_name} - Ch {ev.chapter_num}, Sc {ev.scene_num} ('{ev.label or 'Override'}')"
                    
                    def handle_event_change(e):
                        global selected_event_id
                        selected_event_id = e.value
                        draw_details_panel.refresh()
                        
                    ui.select(
                        options=dropdown_options,
                        value=selected_event_id,
                        on_change=handle_event_change
                    ).classes('w-72 bg-white').props('outlined dense')
                    
                with ui.row().classes('items-center gap-1.5'):
                    ui.button(
                        icon='add_circle', 
                        on_click=lambda: open_add_event_dialog(project.id, char.id, books, draw_details_panel.refresh)
                    ).props('flat dense').classes('p-1 text-blue-600 hover:bg-blue-50').tooltip('Add custom timeline override event')
                    
                    if active_event and active_event.book_id is not None:
                        async def delete_timeline_event():
                            global selected_event_id
                            with Session(engine) as session:
                                db_ev = session.get(CharacterTimelineEvent, active_event.id)
                                if db_ev:
                                    session.delete(db_ev)
                                    session.commit()
                            save_project_characters_to_json(project.id)
                            ui.notify("Timeline override event deleted.", type="warning")
                            selected_event_id = None
                            draw_details_panel.refresh()
                            
                        ui.button(
                            icon='remove_circle',
                            on_click=delete_timeline_event
                        ).props('flat dense').classes('p-1 text-red-500 hover:bg-red-50').tooltip('Delete selected timeline event')

            # --- Row 2.0: Compiled Visual Description Prompt (Brought to the top!) ---
            with ui.column().classes('w-full bg-blue-50/20 p-4 rounded-xl border border-blue-100 gap-2'):
                ui.label('Compiled Visual Description Prompt').classes('text-[11px] font-bold text-blue-600 uppercase tracking-wider')
                
                if not active_event.visual_description:
                    active_event.visual_description = compile_character_visual_prompt(active_event)
                    with Session(engine) as session:
                        db_ev = session.get(CharacterTimelineEvent, active_event.id)
                        if db_ev:
                            db_ev.visual_description = active_event.visual_description
                            session.add(db_ev)
                            session.commit()
                    save_project_characters_to_json(project.id)

                def handle_desc_blur(e, ev_id=active_event.id):
                    new_val = e.sender.value.strip()
                    with Session(engine) as session:
                        db_ev = session.get(CharacterTimelineEvent, ev_id)
                        if db_ev:
                            db_ev.visual_description = new_val if new_val else None
                            session.add(db_ev)
                            session.commit()
                    save_project_characters_to_json(project.id)
                    ui.notify("Visual Prompt overridden!", type="info")

                compiled_desc_input = ui.textarea(
                    value=active_event.visual_description
                ).classes('w-full bg-white font-mono text-xs').props('outlined dense autogrow')\
                 .on('blur', handle_desc_blur)\
                 .tooltip("Injected into your prompt template")

            # --- Row 3.0: Physical parameters grid (Brought to the top!) ---
            ui.label('Physical Description Parameters').classes('text-[11px] font-bold text-slate-500 uppercase tracking-wider mt-1')
            with ui.grid().classes('w-full grid-cols-1 md:grid-cols-2 gap-3'):
                fields = [
                    ("demographics", "Demographics (Age, Race, Gender)"),
                    ("hair_and_face", "Hair & Face Details"),
                    ("physical_build", "Physical Build (Height/Weight/Posture)"),
                    ("distinguishing_marks", "Distinguishing Marks & Key Accessories")
                ]
                
                def make_update_handler(ev_id, key, text_area_el):
                    def handler(e):
                        val = e.sender.value.strip()
                        with Session(engine) as session:
                            db_ev = session.get(CharacterTimelineEvent, ev_id)
                            if db_ev:
                                setattr(db_ev, key, val if val != "" else None)
                                
                                char_obj = session.get(Character, db_ev.character_id)
                                if char_obj and not char_obj.locked:
                                    new_prompt = compile_character_visual_prompt(db_ev)
                                    db_ev.visual_description = new_prompt
                                    text_area_el.set_value(new_prompt)
                                    
                                session.add(db_ev)
                                session.commit()
                        save_project_characters_to_json(project.id)
                        ui.notify("Trait saved.", type="positive", position="bottom-right", timeout=1000)
                    return handler

                for key, label in fields:
                    val = getattr(active_event, key) or ""
                    ui.input(
                        label=label, 
                        value=val
                    ).classes('w-full bg-white').props('outlined dense').on('blur', make_update_handler(active_event.id, key, compiled_desc_input))

            # --- Row 4.0: Mapped Aliases ---
            with ui.column().classes('w-full bg-slate-50 p-4 rounded-xl border gap-3 mt-1'):
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Assigned Aliases & Target Tags').classes('text-[11px] font-bold text-slate-500 uppercase tracking-wider')
                    ui.label(f'{completion_count}/4 traits populated').classes('text-[10px] font-bold text-purple-700 bg-purple-50 px-2 py-0.5 rounded-full')
                
                with ui.row().classes('w-full gap-2 flex-wrap items-center'):
                    for alias in aliases:
                        def make_delete_handler(alias_obj=alias, char_id=char.id):
                            async def handle():
                                global selected_character_id
                                with Session(engine) as session:
                                    db_alias = session.get(CharacterAlias, alias_obj.id)
                                    if db_alias:
                                        alias_name = db_alias.alias
                                        session.delete(db_alias)
                                        session.commit()
                                        
                                        if alias_name.lower() != char.name.lower():
                                            new_char = Character(project_id=project.id, name=alias_name)
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
                                            session.commit()
                                            
                                            new_alias = CharacterAlias(character_id=new_char.id, alias=alias_name)
                                            session.add(new_alias)
                                            session.commit()
                                            
                                            base_ev.visual_description = compile_character_visual_prompt(base_ev)
                                            session.add(base_ev)
                                            session.commit()
                                            
                                            ui.notify(f"Spun off '{alias_name}' into standalone profile!", type="positive")

                                    rem = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char_id)).all()
                                    if not rem:
                                        db_char = session.get(Character, char_id)
                                        if db_char:
                                            # Clean timeline events
                                            evs_to_del = session.exec(select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id == char_id)).all()
                                            for ev in evs_to_del:
                                                session.delete(ev)
                                            session.delete(db_char)
                                            session.commit()
                                            selected_character_id = None
                                
                                save_project_characters_to_json(project.id)
                                await refresh_workspace_with_scroll()
                                ui.notify("Alias removed.", type="info")
                            return handle

                        with ui.row().classes(
                            'items-center gap-1.5 bg-white border border-slate-200 px-2.5 py-1 rounded-full text-xs text-slate-800 hover:bg-slate-50 transition-colors shadow-sm'
                        ):
                            ui.label(alias.alias).classes('cursor-pointer font-medium').on(
                                'click', 
                                lambda _, a=alias, c_id=char.id: open_alias_explorer_dialog(
                                    project.id, a, c_id, refresh_workspace_with_scroll
                                )
                            ).tooltip("Click to view transcript occurrences")
                            
                            ui.icon('cancel', size='14px', color='slate-400').classes(
                                'cursor-pointer hover:text-red-500 transition-colors'
                            ).on('click', make_delete_handler(alias, char.id))

                other_chars = [c for c in all_characters if c.id != char.id]
                if other_chars:
                    with ui.row().classes('w-full items-center gap-2 mt-1'):
                        merge_options = {c.id: c.name for c in other_chars}
                        
                        merge_select = ui.select(
                            options=merge_options,
                            label='Merge another character into this one...',
                            with_input=True
                        ).classes('flex-1 bg-white').props('dense outlined clearable')

                        async def handle_merge_click(c_id=char.id, sel=merge_select):
                            client = ui.context.client
                            src_id = sel.value
                            if not src_id:
                                with client:
                                    ui.notify("Please select a character to merge.", type="warning")
                                return
                            
                            with Session(engine) as session:
                                source_aliases = session.exec(
                                    select(CharacterAlias).where(CharacterAlias.character_id == src_id)
                                ).all()
                                alias_ids = [a.id for a in source_aliases]

                            await asyncio.to_thread(merge_character_aliases, project.id, c_id, alias_ids)
                            with client:
                                ui.notify("Merged successfully!", type="positive")
                            await refresh_workspace_with_scroll()

                        ui.button(
                            'Merge',
                            icon='call_merge', 
                            on_click=handle_merge_click
                        ).classes('bg-blue-600 text-white font-bold text-xs px-3 py-2 rounded-lg')

            # --- Row 5.0: Appearance Map across Series ---
            book_mentions = get_character_book_mentions(project.id, char.id)
            if book_mentions:
                with ui.column().classes('w-full bg-slate-50 p-4 rounded-xl border gap-2 mt-1'):
                    ui.label('Appearance Map across Series (Click a book to edit prompts)').classes('text-[11px] font-bold text-slate-500 uppercase tracking-wider')
                    with ui.row().classes('w-full gap-2 flex-wrap'):
                        for b_name, m_count in book_mentions.items():
                            ui.badge(
                                f"{b_name} ({m_count} hits)", 
                                color='purple-50'
                            ).classes('text-purple-700 text-xs font-semibold px-2 py-1.5 rounded cursor-pointer hover:bg-purple-100 transition-colors')\
                             .on('click', lambda _, b=b_name: open_appearance_prompt_modal(project, char, b))\
                             .tooltip(f"Click to audit and edit {m_count} matching prompts in {b_name}")


    @ui.refreshable
    def draw_workspace_layout():
        global workspace_was_empty
        with Session(engine) as session:
            any_characters = session.exec(
                select(Character).where(Character.project_id == project.id)
            ).first()

        if not any_characters:
            workspace_was_empty = True
            with ui.column().classes('w-full items-center justify-center p-12 text-slate-400 border border-dashed rounded-xl bg-slate-50 gap-4'):
                ui.icon('face', size='xl', color='slate-300')
                ui.label('No characters detected or generated in this project yet.').classes('text-sm font-semibold text-slate-500')
                ui.label('The system needs bracketed names like [Dino] to exist in prompts.csv first.').classes('text-xs text-slate-400 max-w-sm text-center leading-normal')
                
                async def run_prompt_scan_empty():
                    client = ui.context.client
                    with client:
                        ui.notify("Scanning prompts.csv for character tags...", type="info")
                    tags = await asyncio.to_thread(extract_characters_from_prompts, project.id)
                    with client:
                        if tags:
                            ui.notify(f"Discovered and indexed {len(tags)} character tags!", type="positive")
                        else:
                            ui.notify("No new bracketed character tags found in prompts.csv.", type="info")
                    await refresh_workspace_with_scroll()

                ui.button(
                    'Scan for Bracketed Prompt Tags', 
                    icon='tag', 
                    on_click=run_prompt_scan_empty
                ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-5 py-2.5 rounded-lg shadow-sm')
            return

        workspace_was_empty = False
        draw_stats_bar()

        with ui.grid().classes('w-full grid-cols-1 lg:grid-cols-12 gap-4 items-start'):
            # --- LEFT PANEL: Searchable List (col-span-4) ---
            with ui.card().classes('col-span-4 p-4 border rounded-xl bg-white h-[650px] flex flex-col gap-3'):
                ui.label('Characters List').classes('text-sm font-bold text-slate-800 border-b pb-1.5')
                
                def on_search_change(e):
                    global search_query
                    search_query = e.value or ""
                    draw_character_list.refresh()
                
                ui.input(
                    placeholder='Search name or alias...',
                    value=search_query,
                    on_change=on_search_change
                ).props('dense outlined clearable').classes('w-full text-xs')
                
                with ui.row().classes('w-full gap-2 items-center'):
                    def on_sort_change(e):
                        global sort_by
                        sort_by = e.value
                        draw_character_list.refresh()
                        
                    def on_filter_change(e):
                        global filter_status
                        filter_status = e.value
                        draw_character_list.refresh()
                        
                    ui.select(
                        options={
                            "mentions_desc": "Most Mentions",
                            "mentions_asc": "Least Mentions",
                            "name_asc": "Name A-Z",
                            "name_desc": "Name Z-A",
                            "completion_desc": "Highest Completion"
                        },
                        value=sort_by,
                        on_change=on_sort_change
                    ).props('dense outlined').classes('flex-1 text-xs')
                    
                    ui.select(
                        options={
                            "all": "All",
                            "incomplete": "Incomplete",
                            "locked": "Locked",
                            "unlocked": "Auto-Profile"
                        },
                        value=filter_status,
                        on_change=on_filter_change
                    ).props('dense outlined').classes('w-32 text-xs')
                
                with ui.column().classes('w-full flex-1 overflow-y-auto gap-1 pr-1 char-scroll-list'):
                    draw_character_list()

            # --- RIGHT PANEL: Selected Curation Workspace Card (col-span-8) ---
            with ui.card().classes('col-span-8 p-6 border rounded-xl bg-white h-[650px] flex flex-col gap-4'):
                draw_details_panel()

    draw_header_toolbar()
    draw_workspace_layout()