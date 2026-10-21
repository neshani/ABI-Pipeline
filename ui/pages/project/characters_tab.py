import asyncio
import re
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
from nicegui import ui
from sqlmodel import Session, select
from sqlalchemy import func
from database.connection import engine, get_setting
from database.models import Project, Book, Character, CharacterAlias, CharacterTimelineEvent, CharacterBookLink, ScenePrompt
from ui import state
from services.character_manager import (
    save_project_characters_to_json,
    run_stateful_character_profiling,
    get_character_book_mentions,
    save_setting,
    compile_character_visual_prompt,
    compile_character_description_from_event,
    ensure_book_orders,
    get_matching_source_projects,
    get_character_import_matches,
    execute_character_import,
    merge_character_into_target,
    link_character_to_book,
    detach_alias_to_character,
    character_field_stats,
    project_known_counts,
    changed_trait_fields,
    propagate_trait_change,
    character_trait_summary,
    normalize_character_text,
    normalize_character_key,
    clean_character_tag,
    clean_character_key,
    get_pinned_fields,
    set_pinned_fields,
    resolve_event_at,
    NEW_TRAIT_FIELDS,
    RENDERED_TRAIT_FIELDS,
)
from services.character_organizer import (
    extract_characters_from_prompts,
    get_character_frequency_map_db,
    get_book_character_frequency_map,
    commit_raw_seeded_characters,
    merge_character_aliases,
    prune_unused_seeded_characters,
    get_suggested_alias_merges
)
from services import compat

# Active state trackers
selected_book_id: Optional[int] = None
active_view_mode: str = "merge"  # "merge" or "profiles"
is_profiling_all: bool = False
cancel_profiling_all: bool = False

# Fields the old profiler wrote. Kept visible under "legacy fields" so a project
# profiled by an older build can still be edited, but they are not the model.
LEGACY_TRAIT_FIELDS = ("demographics", "physical_build", "hair_and_face",
                       "distinguishing_marks")
_BLANK_TRAIT_VALUES = {"", "null", "none", "unknown", "not stated", "n/a", "unstated"}


def _trait_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _BLANK_TRAIT_VALUES else text


def event_trait_count(event) -> int:
    """How many qualities this event describes, new fields first."""
    if event is None:
        return 0
    new = sum(1 for key in RENDERED_TRAIT_FIELDS if _trait_text(getattr(event, key, None)))
    if new:
        return new
    return sum(1 for key in LEGACY_TRAIT_FIELDS if _trait_text(getattr(event, key, None)))


def event_trait_target(event) -> int:
    """How many fields a complete description has, for the badge and filters."""
    if event is not None and any(
            _trait_text(getattr(event, key, None)) for key in RENDERED_TRAIT_FIELDS):
        return len(RENDERED_TRAIT_FIELDS)
    return len(LEGACY_TRAIT_FIELDS)
currently_profiling_char_id: Optional[int] = None
profiling_progress: str = ""
profiler_scan_depth: int = 5
workspace_was_empty: bool = True

# Filter & Selection states
search_query: str = ""
sort_by: str = "mentions_desc"
filter_status: str = "all"
# Hide characters the analysis created from transcript mentions but that never
# appear as a bracketed tag in prompts.csv. A toggle, not a filter type.
hide_zero_hits: bool = True
selected_character_id: Optional[int] = None
selected_event_id: Optional[int] = None
# Detail-view scope. The list has its own book filter (selected_book_id); these
# two control what the field colours summarise and which book the timeline shows.
time_scope_project: bool = False
timeline_book_id: Optional[int] = None

# High-density caching tracker
row_elements: Dict[int, ui.row] = {}


def get_character_frequency_map(project_name: str, books: List[Book]) -> Dict[str, int]:
    """Helper returning project-wide tag occurrences."""
    with Session(engine) as session:
        return get_character_frequency_map_db(project_name, session)


def open_seed_cast_dialog(project_id: int, default_book_id: Optional[int], books: List[Book], refresh_callback: Any):
    """Spawns modal to quickly paste character names into the selected book."""
    with ui.dialog() as dialog, ui.card().classes('w-[500px] max-w-[95vw] p-5 rounded-xl flex flex-col gap-3'):
        ui.label('Seed Character Cast').classes('text-base font-bold text-slate-800')
        ui.label('Paste character names separated by commas or line breaks.').classes('text-xs text-slate-500')

        book_opts = {None: "Project Base (All Books)"}
        for b in books:
            book_opts[b.id] = b.name

        book_select = ui.select(
            options=book_opts,
            value=default_book_id,
            label="Target Book"
        ).classes('w-full bg-white').props('outlined dense')

        names_input = ui.textarea(
            label="Character Names",
            placeholder="e.g. John Smith, Mary Jane, Dino Bacchetti\nor one name per line..."
        ).classes('w-full bg-white font-mono text-xs').props('outlined autogrow')

        async def handle_seed():
            text = names_input.value.strip()
            if not text:
                ui.notify("Please enter at least one character name.", type="warning")
                return

            tgt_b_id = book_select.value
            await asyncio.to_thread(commit_raw_seeded_characters, project_id, tgt_b_id, text)
            ui.notify("Characters seeded successfully!", type="positive")
            dialog.close()
            await refresh_callback()

        with ui.row().classes('w-full justify-end gap-2 border-t pt-3 mt-2'):
            ui.button('Cancel', on_click=dialog.close).props('flat').classes('text-xs')
            ui.button('Seed Characters', icon='person_add', on_click=handle_seed).classes('bg-blue-600 text-white font-bold text-xs px-4 py-2 rounded-lg')

    dialog.open()


def open_become_alias_dialog(project_id: int, source_char: Character, books: List[Book], refresh_callback: Any):
    """Spawns modal allowing source character to be merged into an established target character ('Become Alias')."""
    with Session(engine) as session:
        all_chars = session.exec(
            select(Character).where(Character.project_id == project_id).where(Character.id != source_char.id)
        ).all()

        book_order_map = {b.id: (b.book_order or 0) for b in books}
        char_book_order = {}
        for c in all_chars:
            links = session.exec(select(CharacterBookLink).where(CharacterBookLink.character_id == c.id)).all()
            if links:
                char_book_order[c.id] = min(book_order_map.get(lk.book_id, 999) for lk in links)
            else:
                char_book_order[c.id] = 999

        sorted_targets = sorted(all_chars, key=lambda c: (char_book_order.get(c.id, 999), c.name.lower()))
        target_options = {c.id: f"{c.name} ({c.hit_count or 0} hits)" for c in sorted_targets}

    with ui.dialog() as dialog, ui.card().classes('w-[500px] max-w-[95vw] p-6 rounded-xl flex flex-col gap-4'):
        with ui.row().classes('w-full justify-between items-center border-b pb-2'):
            with ui.column().classes('gap-0.5'):
                ui.label(f'Merge "{source_char.name}" into Target').classes('text-base font-bold text-slate-800')
                ui.label('Converts this profile into an alias of the chosen character.').classes('text-xs text-slate-500')
            ui.button(icon='close', on_click=dialog.close).props('flat dense').classes('text-slate-400')

        if not target_options:
            ui.label('No other characters available in this project to merge into.').classes('text-xs text-slate-400 italic py-4')
            ui.button('Close', on_click=dialog.close).props('flat')
            dialog.open()
            return

        target_select = ui.select(
            options=target_options,
            label='Select Target Character',
            with_input=True
        ).classes('w-full bg-white').props('outlined dense')

        async def handle_merge():
            tgt_id = target_select.value
            if not tgt_id:
                ui.notify("Please select a target character.", type="warning")
                return

            global selected_character_id
            success = await asyncio.to_thread(merge_character_into_target, source_char.id, tgt_id, selected_book_id)
            if success:
                selected_character_id = tgt_id
                ui.notify(f"Merged '{source_char.name}' into target character!", type="positive")
                dialog.close()
                await refresh_callback()
            else:
                ui.notify("Failed to merge character.", type="negative")

        with ui.row().classes('w-full justify-end gap-2 border-t pt-3 mt-2'):
            ui.button('Cancel', on_click=dialog.close).props('flat').classes('text-xs')
            ui.button('Confirm Merge', icon='call_merge', on_click=handle_merge).classes('bg-amber-600 hover:bg-amber-700 text-white font-bold text-xs px-4 py-2 rounded-lg')

    dialog.open()


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
                        elif ev.chapter_num == tgt_ch and ev.shot_num <= tgt_sc:
                            matched.append((ev, ev_order))
                            
                resolved = base_ev
                if matched:
                    matched.sort(key=lambda x: (x[1], x[0].chapter_num, x[0].shot_num))
                    resolved = matched[-1][0]
                
                if resolved:
                    copied_traits["demographics"] = resolved.demographics
                    copied_traits["physical_build"] = resolved.physical_build
                    copied_traits["hair_and_face"] = resolved.hair_and_face
                    copied_traits["distinguishing_marks"] = resolved.distinguishing_marks
                    for field in NEW_TRAIT_FIELDS:
                        copied_traits[field] = getattr(resolved, field, None)
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
                    .where(CharacterTimelineEvent.shot_num == sc)
                ).first()
                
                if dup:
                    ui.notify("An override event already exists at this scene coordinate.", type="warning")
                    return
                
                new_ev = CharacterTimelineEvent(
                    character_id=character_id,
                    book_id=b_id,
                    chapter_num=ch,
                    shot_num=sc,
                    label=lbl,
                    source="manual",
                    demographics=copied_traits.get("demographics"),
                    physical_build=copied_traits.get("physical_build"),
                    hair_and_face=copied_traits.get("hair_and_face"),
                    distinguishing_marks=copied_traits.get("distinguishing_marks"),
                    **{field: copied_traits.get(field) for field in NEW_TRAIT_FIELDS},
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
            min_mentions_input = ui.number(value=1, min=1, step=1).classes('w-16 bg-white').props('outlined dense')

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
                    query = select(Character).where(Character.project_id == project.id).where(Character.locked == False)
                    if selected_book_id:
                        query = query.join(CharacterBookLink).where(CharacterBookLink.book_id == selected_book_id)
                    unlocked_chars = session.exec(query).all()
                    
                    char_aliases = {}
                    for char in unlocked_chars:
                        aliases = session.exec(
                            select(CharacterAlias).where(CharacterAlias.character_id == char.id)
                        ).all()
                        char_aliases[char.id] = aliases

                def get_char_mentions(char_obj, aliases_list):
                    total = sum(frequencies.get(clean_character_key(a.alias), 0) for a in aliases_list)
                    if not aliases_list:
                        total = frequencies.get(clean_character_key(char_obj.name), 0)
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
                    return event_trait_count(ev_obj)

                creative_queue = []
                for char in base_queue:
                    with Session(engine) as session:
                        db_char = session.get(Character, char.id)
                        if not db_char or db_char.locked:
                            continue
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

            is_profiling_all = False
            currently_profiling_char_id = None
            profiling_progress = ""
            cancel_profiling_all = False
            
            with client:
                ui.notify("Batch profiling sequence completed.", type="info")
            
            await safe_refresh(refresh_toolbar_callback)
            await safe_refresh(refresh_ui_callback)

        with ui.tab_panels(tabs, value=factual_tab).classes('w-full flex-1 min-h-0 bg-transparent'):
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


def open_prompt_editor_dialog(mode: str = "factual"):
    """Renders a modal to customize the LLM profiler template instructions."""
    from services.character_manager import load_prompt_template_from_file, save_prompt_template_to_file
    
    title = "Factual Profiler Prompt" if mode == "factual" else "Speculative Profiler Prompt"
    sub_title = (
        "Configure system instructions sent to the local LLM for factual extraction." 
        if mode == "factual" else 
        "Configure system instructions used when speculating character descriptions."
    )
    
    current_template = load_prompt_template_from_file(mode)

    with ui.dialog() as dialog, ui.card().classes('w-[750px] max-w-[95vw] h-[650px] max-h-[90vh] p-6 rounded-xl flex flex-col overflow-hidden'):
        
        def reset():
            if mode == "factual":
                from services.character_manager import get_default_character_template
                editor.value = get_default_character_template()
            else:
                from services.character_manager import get_speculative_character_template
                editor.value = get_speculative_character_template()
            ui.notify("Template reset to default.", type="info")

        def save():
            save_prompt_template_to_file(mode, editor.value)
            ui.notify(f"{title} template saved to disk!", type="positive")
            dialog.close()

        with ui.row().classes('w-full justify-between items-center border-b pb-3 mb-3 shrink-0'):
            with ui.column().classes('gap-0.5'):
                ui.label(f'Customize {title}').classes('text-base font-bold text-slate-800')
                ui.label(sub_title).classes('text-xs text-slate-500')
            
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

        with ui.row().classes('w-full justify-between items-center pt-2 shrink-0 border-t mt-2'):
            with Session(engine) as session:
                char = session.get(Character, parent_char_id)
                char_name = char.name if char else ""
            
            show_promote = clean_character_key(char_name) != clean_character_key(alias.alias)
            
            if show_promote:
                def handle_promote():
                    with Session(engine) as session:
                        db_char = session.get(Character, parent_char_id)
                        db_alias = session.get(CharacterAlias, alias.id)
                        if db_char and db_alias:
                            old_name = db_char.name
                            new_name = db_alias.alias
                            
                            db_char.name = new_name
                            session.add(db_char)
                            
                            exists = session.exec(
                                select(CharacterAlias)
                                .where(CharacterAlias.character_id == parent_char_id)
                                .where(CharacterAlias.alias == old_name)
                            ).first()
                            
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
            reader.fieldnames = compat.normalize_fieldnames(reader.fieldnames)
            rows = list(reader)
    except Exception as e:
        print(f"[Map] Error reading prompts.csv: {e}")
        return []

    for idx, row in enumerate(rows):
        prompt_text = row.get("prompt", "")
        matched = False
        
        for match in bracket_regex.findall(prompt_text):
            if clean_character_key(match) in alias_texts:
                matched = True
                break
        
        if matched:
            occurrences.append({
                "global_row_index": idx,
                "chapter": row.get("chapter") or row.get("chapter_num") or row.get("Chapter") or "N/A",
                "shot": compat.get_shot(row, "N/A"),
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
            reader.fieldnames = compat.normalize_fieldnames(reader.fieldnames)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
            
        if 0 <= global_row_index < len(rows):
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
    """Spawns dialog to view and edit book prompts where this character is tagged."""
    with Session(engine) as session:
        aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == character.id)).all()
        alias_texts = {clean_character_key(a.alias) for a in aliases}
        alias_texts.add(clean_character_key(character.name))

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
            coordinate_label.text = f"Coordinates: Chapter {occ['chapter']}, Scene {occ['shot']}"
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
    """Spawns reconciliation grid to copy matching character profiles from prior projects."""
    selected_source_project_id = matching_projects[0]["id"]
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
                    checked_pairs[tgt_id] = True
                    active_pairings[tgt_id] = match
                    
                    with ui.row().classes('w-full items-center justify-between p-3 bg-white border rounded-xl gap-3 shadow-sm hover:border-slate-300 transition-colors'):
                        cb = ui.checkbox(value=True).classes('shrink-0')
                        cb.bind_value_to(checked_pairs, tgt_id)
                        
                        with ui.column().classes('flex-1 min-w-0 gap-0.5'):
                            alias_suffix = format_aliases(match["source_aliases"])
                            ui.label(f"{match['source_name']}{alias_suffix}").classes('text-xs font-bold text-slate-800 truncate w-full')
                            ui.label(truncate_desc(match["source_desc"])).classes('text-[10px] text-slate-500 italic truncate w-full')
                            
                        ui.icon('arrow_forward', size='xs', color='slate-400').classes('shrink-0')
                        
                        with ui.column().classes('flex-1 min-w-0 gap-0.5'):
                            t_alias_suffix = format_aliases(match["target_aliases"])
                            with ui.row().classes('items-center gap-2 w-full'):
                                ui.label(f"{match['target_name']}{t_alias_suffix}").classes('text-xs font-semibold text-slate-700 truncate')
                                ui.badge(f"{match['target_mentions']} hits", color='blue-50').classes('text-[9px] font-bold text-blue-700 px-1 py-0.5 rounded')
                            ui.label(truncate_desc(match["target_desc"])).classes('text-[10px] text-slate-400 italic truncate w-full')

        with ui.column().classes('w-full flex-1 overflow-y-auto min-h-0 bg-slate-50/50 p-3 rounded-lg border gap-2'):
            ui.label('Verify Matches & Overrides').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1 shrink-0')
            draw_matching_rows()

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


def open_add_character_dialog(project_id: int, default_book_id: Optional[int], refresh_callback: Any):
    """Spawns modal to manually add a new character with baseline state and default alias."""
    with ui.dialog() as dialog, ui.card().classes('w-[400px] p-5 rounded-xl flex flex-col gap-3'):
        ui.label('Add New Character').classes('text-sm font-bold text-slate-800')
        name_input = ui.input(label="Character Name", placeholder="e.g. Molly").classes('w-full').props('outlined dense')
        
        def save_character():
            name = clean_character_tag(name_input.value)
            if not name:
                ui.notify("Character name cannot be empty.", type="warning")
                return
            name_key = clean_character_key(name)

            with Session(engine) as session:
                existing = next(
                    (char for char in session.exec(
                        select(Character)
                        .where(Character.project_id == project_id)).all()
                     if clean_character_key(char.name) == name_key),
                    None)
                if existing:
                    ui.notify(f"A character named '{name}' already exists.", type="warning")
                    return
                
                new_char = Character(project_id=project_id, name=name)
                session.add(new_char)
                session.commit()
                
                base_ev = CharacterTimelineEvent(
                    character_id=new_char.id,
                    book_id=None,
                    chapter_num=0,
                    shot_num=0,
                    label="Base State"
                )
                session.add(base_ev)
                session.commit()
                
                new_alias = CharacterAlias(character_id=new_char.id, alias=name)
                session.add(new_alias)
                session.commit()
                
                if default_book_id:
                    link = CharacterBookLink(character_id=new_char.id, book_id=default_book_id)
                    session.add(link)
                    session.commit()
                
                base_ev.visual_description = compile_character_visual_prompt(base_ev)
                session.add(base_ev)
                session.commit()
                
                global selected_character_id, selected_event_id
                selected_character_id = new_char.id
                selected_event_id = base_ev.id

            save_project_characters_to_json(project_id)
            ui.notify(f"Character '{name}' added successfully!", type="positive")
            dialog.close()
            refresh_callback()

        with ui.row().classes('w-full justify-end gap-2 border-t pt-3 mt-2'):
            ui.button('Cancel', on_click=dialog.close).props('flat')
            ui.button('Add Character', on_click=save_character).classes('bg-blue-600 text-white font-bold text-xs px-3 py-1.5 rounded-lg')
    dialog.open()


def load_hair_options() -> Dict[str, List[str]]:
    """Loads hair styles and colors from static/hair_options.json."""
    import json
    options_path = Path("static/hair_options.json")
    default_options = {
        "colors": [
            "blonde", "platinum blonde", "strawberry blonde", "golden blonde",
            "dark brown", "chestnut brown", "light brown", "auburn", "ginger", "red",
            "jet black", "charcoal gray", "silver-gray", "white",
            "pastel pink", "emerald green", "cobalt blue", "lavender", "neon purple", "teal"
        ],
        "female_styles": [
            "long straight hair", "pixie cut", "bob haircut", "shoulder-length wavy bob",
            "messy bun", "chignon", "high ponytail", "braided crown", "fishtail braid",
            "blunt bangs with long straight hair", "layered curls", "undercut pixie",
            "french braid", "long flowing waves", "twin braided tails", "curled bob"
        ],
        "male_styles": [
            "classic pompadour", "slicked-back undercut", "buzz cut", "shaggy layered cut",
            "crew cut", "side-part comb-over with high fade", "messy textured crop",
            "dreadlocks", "curly afro", "man bun with shaved sides", "taper fade",
            "long middle-parted hair"
        ],
        "unisex_styles": [
            "short messy hair", "curly mop-top", "shoulder-length straight hair",
            "shaved bald head", "asymmetrical crop", "bowl cut"
        ]
    }
    
    if not options_path.exists():
        try:
            options_path.parent.mkdir(parents=True, exist_ok=True)
            with open(options_path, "w", encoding="utf-8") as f:
                json.dump(default_options, f, indent=2, ensure_ascii=False)
        except Exception:
            return default_options
            
    try:
        with open(options_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_options


def open_hair_picker_modal(project_id: int, event_id: int, compiled_desc_input: Any, hair_input_el: Any):
    """Spawns helper to select, randomize, and apply hairstyle formulas."""
    import random
    options = load_hair_options()
    
    color_select = None
    style_select = None
    preview_text_input = None
    gender_toggle = None
    
    def update_preview():
        if not color_select or not style_select or not preview_text_input:
            return
        col = color_select.value
        sty = style_select.value
        if not col or not sty:
            return
        
        if "bald" in sty.lower() or "shaved head" in sty.lower():
            preview_text_input.set_value("with a shaved bald head")
        else:
            preview_text_input.set_value(f"with {col} hair styled in a {sty}")
            
    def on_gender_change(e):
        gender = e.value
        styles_pool = options[f"{gender}_styles"]
        style_select.set_options(styles_pool)
        style_select.set_value(styles_pool[0])
        update_preview()
        
    def randomize():
        gender = gender_toggle.value
        styles_pool = options[f"{gender}_styles"]
        rand_color = random.choice(options["colors"])
        rand_style = random.choice(styles_pool)
        color_select.set_value(rand_color)
        style_select.set_value(rand_style)
        update_preview()

    with ui.dialog() as dialog, ui.card().classes('w-[460px] p-6 rounded-xl flex flex-col gap-4'):
        with ui.row().classes('w-full justify-between items-center border-b pb-2'):
            ui.label('Hairstyle Vibe Helper').classes('text-sm font-bold text-slate-800')
            ui.button(icon='close', on_click=dialog.close).props('flat dense').classes('text-slate-400')
            
        gender_toggle = ui.radio(
            options={"female": "Female", "male": "Male", "unisex": "Unisex"},
            value="female",
            on_change=on_gender_change
        ).props('inline dense').classes('text-xs mb-1')
        
        color_select = ui.select(
            options=options["colors"],
            value=options["colors"][0],
            label="Hair Color",
            on_change=update_preview
        ).classes('w-full bg-white').props('outlined dense')
        
        style_select = ui.select(
            options=options["female_styles"],
            value=options["female_styles"][0],
            label="Hair Style",
            on_change=update_preview
        ).classes('w-full bg-white').props('outlined dense')
        
        preview_text_input = ui.input(
            label="Generated Preview Fragment",
            value=""
        ).classes('w-full font-mono text-xs bg-white').props('outlined dense')
        
        update_preview()
        
        with ui.row().classes('w-full gap-2 items-center bg-slate-50 p-3 rounded-lg border justify-between'):
            ui.label('Feeling lucky?').classes('text-xs font-semibold text-slate-500')
            ui.button(
                'Randomize Combo',
                icon='casino',
                on_click=randomize
            ).classes('bg-purple-600 hover:bg-purple-700 text-white font-bold text-xs px-3 py-1.5 rounded-lg shadow-sm')

        with ui.row().classes('w-full justify-end gap-2 border-t pt-3 mt-2'):
            def copy_clipboard():
                ui.run_javascript(f"navigator.clipboard.writeText('{preview_text_input.value}');")
                ui.notify("Copied description fragment to clipboard!", type="info")

            async def apply_to_profile():
                final_text = preview_text_input.value.strip()
                if not final_text or not hair_input_el:
                    return
                
                hair_input_el.set_value(final_text)
                
                with Session(engine) as session:
                    db_ev = session.get(CharacterTimelineEvent, event_id)
                    if db_ev:
                        db_ev.hair_and_face = final_text
                        if db_ev.source == "extraction":
                            db_ev.source = "manual"
                        char_obj = session.get(Character, db_ev.character_id)
                        if char_obj and not char_obj.locked:
                            new_prompt = compile_character_visual_prompt(db_ev)
                            db_ev.visual_description = new_prompt
                            compiled_desc_input.set_value(new_prompt)
                        session.add(db_ev)
                        session.commit()
                
                save_project_characters_to_json(project_id)
                ui.notify("Applied hair style description!", type="positive")
                dialog.close()

            ui.button('Copy', icon='content_copy', on_click=copy_clipboard).props('flat').classes('text-xs text-slate-600')
            ui.button('Cancel', on_click=dialog.close).props('flat')
            ui.button('Apply to Character', icon='check', on_click=apply_to_profile).classes('bg-blue-600 text-white font-bold text-xs px-3 py-1.5 rounded-lg shadow-sm')
            
    dialog.open()

def get_default_ai_check_template(mode: str) -> str:
    """Returns the default prompt template depending on the active view mode."""
    if mode == "merge":
        return (
            'I am organizing character profiles from an audiobook transcript for the book "{book_name}" '
            'in the "{project_name}" series. The transcription likely contains phonetic spelling errors, misheard names, and speech recognition artifacts.\n\n'
            'For the character "{character_name}" (Aliases/Variations: {assigned_aliases}):\n'
            '- Potential Unmerged Names Found in Transcript: {candidate_names}\n\n'
            'Please answer:\n'
            '1. Are all current assigned aliases canonically associated with {character_name}?\n'
            '2. Are any of the potential unmerged names actually aliases, nicknames, titles, or misspellings of {character_name}?\n'
            '3. Are there any distinct characters in this list that should NOT be merged?'
        )
    else:
        return (
            'An LLM has extracted/generated this visual description for the character "{character_name}" '
            '(Also known as / transcript variations: {assigned_aliases}) in the book "{book_name}" of the "{project_name}" series:\n\n'
            '- Compiled Visual Description: {visual_description}\n'
            '- Demographics: {demographics}\n'
            '- Hair & Face: {hair_and_face}\n'
            '- Physical Build: {physical_build}\n'
            '- Distinguishing Marks: {distinguishing_marks}\n'
            '- Extracted Traits: {traits}\n\n'
            'Based on the official book canon and lore:\n'
            '1. Is this visual description accurate for {character_name}?\n'
            '2. Are there any canon inaccuracies (wrong eye/hair color, wrong age, mismatched traits)?\n'
            '3. Are there any iconic visual details or signature accessories missing?'
        )


def generate_ai_check_prompt(
    project_id: int, 
    character_id: int, 
    book_id: Optional[int] = None, 
    mode: str = "merge", 
    event_id: Optional[int] = None,
    template_override: Optional[str] = None
) -> str:
    """Renders all available character and book context tags into the prompt template."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        char = session.get(Character, character_id)
        if not project or not char:
            return ""

        book_name = "the entire series"
        if book_id:
            book = session.get(Book, book_id)
            if book:
                book_name = book.name

        template = template_override
        if not template:
            setting_key = f"ai_check_template_{mode}"
            template = get_setting(setting_key, get_default_ai_check_template(mode), session)

        # 1. Alias & Candidate details
        aliases = session.exec(
            select(CharacterAlias).where(CharacterAlias.character_id == character_id)
        ).all()
        alias_names = [a.alias for a in aliases if clean_character_key(a.alias) != clean_character_key(char.name)]
        alias_str = ", ".join(f'"{a}"' for a in alias_names) if alias_names else "None assigned yet"

        sugs = get_suggested_alias_merges(project_id, character_id, book_id)
        sug_names = [s["name"] for s in sugs[:8]]
        sug_str = ", ".join(f'"{s}"' for s in sug_names) if sug_names else "None detected"

        # 2. Visual & Trait details
        active_ev = None
        if event_id:
            active_ev = session.get(CharacterTimelineEvent, event_id)
        if not active_ev:
            active_ev = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == character_id)
                .where(CharacterTimelineEvent.book_id == None)
            ).first()

        # All tags unified across both modes
        traits_text = ", ".join(
            f"{field.replace('_', ' ')}: {getattr(active_ev, field)}"
            for field in RENDERED_TRAIT_FIELDS
            if active_ev is not None and getattr(active_ev, field, None)
            and str(getattr(active_ev, field)).strip().lower()
            not in ("null", "none", "unknown", "not stated"))
        has_new_traits = bool(traits_text)
        compiled_desc = ""
        if active_ev is not None:
            compiled_desc = compile_character_description_from_event(
                char, active_ev, {}, True)
            if compiled_desc == char.name:
                compiled_desc = ""
        replacements = {
            "{project_name}": project.name,
            "{book_name}": book_name,
            "{character_name}": char.name,
            "{assigned_aliases}": alias_str,
            "{candidate_names}": sug_str,
            "{visual_description}": (
                (active_ev.visual_description if active_ev and active_ev.visual_description
                 else compiled_desc) or "None generated yet"),
            # The legacy quartet may be empty for an event written by the new
            # extractor; point the reviewer at the trait line rather than lying.
            "{demographics}": ((active_ev.demographics if active_ev and active_ev.demographics
                                else ("See extracted traits" if has_new_traits else "Unspecified"))),
            "{hair_and_face}": ((active_ev.hair_and_face if active_ev and active_ev.hair_and_face
                                 else ("See extracted traits" if has_new_traits else "Unspecified"))),
            "{physical_build}": ((active_ev.physical_build if active_ev and active_ev.physical_build
                                  else ("See extracted traits" if has_new_traits else "Unspecified"))),
            "{distinguishing_marks}": ((active_ev.distinguishing_marks
                                        if active_ev and active_ev.distinguishing_marks
                                        else "None recorded")),
            "{traits}": traits_text or "None extracted",
        }

        rendered = template
        for key, val in replacements.items():
            rendered = rendered.replace(key, str(val))
        return rendered


def open_ai_check_editor_dialog(
    project_id: int, 
    character_id: int, 
    book_id: Optional[int] = None, 
    mode: str = "merge", 
    event_id: Optional[int] = None
):
    """Spawns the AI Check Template Editor with live character preview and all available tags."""
    setting_key = f"ai_check_template_{mode}"
    with Session(engine) as session:
        saved_template = get_setting(setting_key, get_default_ai_check_template(mode), session)

    title = "Edit AI Merge Check Template" if mode == "merge" else "Edit AI Sanity Check Template"
    placeholders_doc = (
        "- **General:** `{project_name}`, `{book_name}`, `{character_name}`, `{assigned_aliases}`, `{candidate_names}`\n"
        "- **Visual Traits:** `{visual_description}`, `{traits}`, `{demographics}`, `{hair_and_face}`, `{physical_build}`, `{distinguishing_marks}`"
    )

    with ui.dialog() as dialog, ui.card().classes('w-[750px] max-w-[95vw] h-[650px] max-h-[90vh] p-6 rounded-xl flex flex-col gap-3 overflow-hidden'):
        with ui.row().classes('w-full justify-between items-center border-b pb-2 shrink-0'):
            with ui.column().classes('gap-0.5'):
                ui.label(title).classes('text-base font-bold text-slate-800')
                ui.label('Customize the prompt template once; all tags are available in both modes.').classes('text-xs text-slate-500')
            ui.button(icon='close', on_click=dialog.close).props('flat dense').classes('text-slate-400')

        with ui.column().classes('w-full flex-1 overflow-y-auto gap-3 pr-1'):
            ui.markdown(f"**Available Placeholders:**\n{placeholders_doc}").classes(
                'text-xs text-slate-600 bg-slate-50 p-2.5 rounded-lg border w-full leading-relaxed'
            )

            template_textarea = ui.textarea(
                label="Prompt Template Instructions",
                value=saved_template
            ).classes('w-full font-mono text-xs bg-white').props('outlined autogrow')

            ui.label('Live Preview (Current Character):').classes('text-[11px] font-bold text-slate-400 uppercase tracking-wider')
            
            initial_preview = generate_ai_check_prompt(
                project_id, character_id, book_id, mode=mode, event_id=event_id, template_override=saved_template
            )
            preview_textarea = ui.textarea(
                value=initial_preview
            ).classes('w-full font-mono text-xs bg-slate-50 text-slate-700').props('outlined autogrow readonly')

            def on_template_change():
                preview_textarea.value = generate_ai_check_prompt(
                    project_id, character_id, book_id, mode=mode, event_id=event_id, template_override=template_textarea.value
                )

            template_textarea.on('input', on_template_change)

        with ui.row().classes('w-full justify-between items-center border-t pt-3 shrink-0 mt-1'):
            def reset_default():
                default_tmpl = get_default_ai_check_template(mode)
                template_textarea.value = default_tmpl
                on_template_change()
                ui.notify("Template reset to default.", type="info")

            ui.button('Reset to Default', color='amber', icon='restart_alt', on_click=reset_default)\
                .props('flat').classes('text-xs font-semibold')

            with ui.row().classes('gap-2 items-center'):
                def save_template_only():
                    save_setting(setting_key, template_textarea.value.strip())
                    ui.notify("Template saved!", type="positive")
                    dialog.close()

                def copy_live_preview():
                    save_setting(setting_key, template_textarea.value.strip())
                    ui.clipboard.write(preview_textarea.value)
                    ui.notify("Rendered prompt copied to clipboard!", type="positive", icon="content_copy")
                    dialog.close()

                ui.button('Save Template', on_click=save_template_only)\
                    .classes('bg-slate-100 text-slate-700 hover:bg-slate-200 font-bold text-xs px-3 py-2 rounded-lg border')

                ui.button('Copy Live Prompt', icon='content_copy', on_click=copy_live_preview)\
                    .classes('bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs px-4 py-2 rounded-lg shadow-sm')

    dialog.open()

def render_characters_tab(project: Project, books: List[Book], refresh_parent: Optional[Any] = None):
    # Enforce chronological book order indexing
    ensure_book_orders(project.id)

    async def restore_scroll_position():
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
        global workspace_was_empty
        with Session(engine) as session:
            any_characters = session.exec(
                select(Character).where(Character.project_id == project.id)
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
        draw_character_list.refresh()
        await restore_scroll_position()

    async def refresh_after_trait_save():
        """Re-render the panels without losing your place in the profile.

        The trait grid lives in its own scrollable column. A plain refresh
        rebuilds it at scrollTop 0, which throws you back to the top of a long
        profile after every field edit - so the offset is read before the
        rebuild and put back after it lands.
        """
        try:
            top = await ui.run_javascript(
                "(() => { const el = document.querySelector('.char-details-scroll');"
                " return el ? el.scrollTop : 0; })()", timeout=1.0)
        except Exception:
            top = 0
        draw_stats_bar.refresh()
        draw_character_list.refresh()
        draw_details_panel.refresh()
        try:
            # The rebuilt column arrives over the socket, so the offset has to be
            # re-applied for a few frames rather than once.
            await ui.run_javascript(
                "(() => {"
                f"  const target = {int(top or 0)}; let tries = 0;"
                "  const put = () => {"
                "    const el = document.querySelector('.char-details-scroll');"
                "    if (el) el.scrollTop = target;"
                "    if (tries++ < 10) setTimeout(put, 30);"
                "  };"
                "  put();"
                "})()", timeout=1.0)
        except Exception:
            pass
        await restore_scroll_position()

    def select_char(c_id):
        global selected_character_id, selected_event_id, timeline_book_id
        old_id = selected_character_id
        selected_character_id = c_id
        timeline_book_id = selected_book_id
        
        with Session(engine) as session:
            base_ev = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == c_id)
                .where(CharacterTimelineEvent.book_id == None)
            ).first()
            selected_event_id = base_ev.id if base_ev else None
        
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

    def step_book(direction: int):
        global selected_book_id, selected_character_id
        book_ids = [None] + [b.id for b in books]
        current_idx = book_ids.index(selected_book_id) if selected_book_id in book_ids else 0
        new_idx = (current_idx + direction) % len(book_ids)
        selected_book_id = book_ids[new_idx]
        selected_character_id = None
        draw_header_toolbar.refresh()
        draw_stats_bar.refresh()
        draw_character_list.refresh()
        draw_details_panel.refresh()

    @ui.refreshable
    def draw_header_toolbar():
        global selected_book_id, active_view_mode, is_profiling_all, currently_profiling_char_id, profiling_progress, cancel_profiling_all
        
        with ui.column().classes('w-full bg-slate-50 border p-3 rounded-xl mb-3 gap-2.5'):
            # TOP ROW: Book Stepper, View Mode Toggle, Batch Profiler, and Overflow Menu
            with ui.row().classes('w-full items-center justify-between flex-wrap gap-2'):
                
                # 1. Book Navigation Stepper
                with ui.row().classes('items-center gap-1 bg-white border p-1 rounded-lg shadow-sm'):
                    ui.label('List filter').classes('text-[9px] font-black text-slate-400 uppercase tracking-wider px-1')
                    ui.button(icon='chevron_left', on_click=lambda: step_book(-1)).props('flat dense size=sm')\
                        .classes('text-slate-600 hover:bg-slate-100 rounded').tooltip("Previous Book")
                    
                    book_options = {None: "All Books (Project Wide)"}
                    for b in books:
                        book_options[b.id] = b.name

                    def on_book_select(e):
                        global selected_book_id, selected_character_id
                        selected_book_id = e.value
                        selected_character_id = None
                        draw_header_toolbar.refresh()
                        draw_stats_bar.refresh()
                        draw_character_list.refresh()
                        draw_details_panel.refresh()

                    ui.select(
                        options=book_options,
                        value=selected_book_id,
                        on_change=on_book_select
                    ).classes('w-52 text-xs font-semibold').props('dense borderless')

                    ui.button(icon='chevron_right', on_click=lambda: step_book(1)).props('flat dense size=sm')\
                        .classes('text-slate-600 hover:bg-slate-100 rounded').tooltip("Next Book")

                # 2. View Mode Toggle (Segmented Buttons)
                with ui.row().classes('items-center bg-slate-200/70 p-1 rounded-lg gap-1 border'):
                    def set_mode(m: str):
                        global active_view_mode
                        active_view_mode = m
                        draw_header_toolbar.refresh()
                        draw_character_list.refresh()
                        draw_details_panel.refresh()

                    is_merge = (active_view_mode == "merge")
                    merge_style = 'bg-white text-blue-700 shadow-sm font-bold' if is_merge else 'text-slate-600 font-semibold hover:text-slate-900'
                    prof_style = 'bg-white text-blue-700 shadow-sm font-bold' if not is_merge else 'text-slate-600 font-semibold hover:text-slate-900'

                    ui.button('👥 Clean & Merge', on_click=lambda: set_mode("merge")).props('flat dense')\
                        .classes(f'text-xs px-3 py-1 rounded-md transition-all {merge_style}')\
                        .tooltip("Deduplicate singletons, manage aliases, and resolve collisions")

                    ui.button('🎨 Visual Profiles', on_click=lambda: set_mode("profiles")).props('flat dense')\
                        .classes(f'text-xs px-3 py-1 rounded-md transition-all {prof_style}')\
                        .tooltip("Edit physical traits, craft visual prompts, and manage timeline overrides")

                # 3. Project-wide rendering policy for the `carried` trait. It is a
                # setting, not a per-character decision, because the problem is the
                # trait itself: a prop the author mentions once decorates every later
                # image, and there is no sentence saying it was put down.
                carried_options = {
                    "scene": "Scene only",
                    "off": "Never rendered",
                    "always": "Carry forward",
                }

                def set_carried_mode(value):
                    mode = value if value in carried_options else "scene"
                    state.carried_mode = mode
                    from services.project_settings import save_project_settings_to_disk
                    save_project_settings_to_disk(project.id)
                    ui.notify(f"Carried props: {carried_options[mode]}", type="info")
                    draw_header_toolbar.refresh()
                    draw_details_panel.refresh()

                with ui.row().classes('items-center gap-1 bg-white border px-2 py-0.5 rounded-lg'):
                    ui.icon('backpack', size='14px', color='slate-400')
                    ui.label('Carried').classes(
                        'text-[9px] font-bold text-slate-400 uppercase tracking-wide')
                    ui.select(carried_options,
                              value=state.carried_mode if state.carried_mode in carried_options
                              else "scene",
                              on_change=lambda e: set_carried_mode(e.value))\
                        .classes('w-32 text-[10px]').props('dense borderless')\
                        .tooltip('Scene only: a carried prop appears only where the text '
                                 'recorded it. Never rendered: stored and editable, but '
                                 'left out of prompts like notes. Carry forward: the old '
                                 'behaviour - it persists until something changes it. '
                                 'Pinning a character\'s carried field overrides this.')

                # 4. Action Buttons & Overflow Menu
                with ui.row().classes('items-center gap-2'):
                    if not is_profiling_all:
                        ui.button(
                            'Batch Profile', 
                            icon='bolt', 
                            on_click=lambda: open_batch_profiler_dialog(
                                project, 
                                books, 
                                refresh_workspace_with_scroll, 
                                draw_header_toolbar.refresh,
                                draw_details_panel.refresh
                            )
                        ).classes('bg-purple-600 hover:bg-purple-700 text-white font-bold text-xs px-3 py-1.5 rounded-lg shadow-sm')

                    # Consolidated "More Tools" Dropdown Menu
                    with ui.button(icon='more_vert').props('flat dense').classes('text-slate-600 hover:bg-slate-200 p-1.5 rounded-lg'):
                        with ui.menu().classes('p-2 flex flex-col gap-1 w-64 shadow-lg rounded-xl border'):
                            ui.label('Cast Tools').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wider px-2 py-1')

                            ui.button(
                                'Add Character Profile...',
                                icon='person_add',
                                on_click=lambda: open_add_character_dialog(project.id, selected_book_id, refresh_workspace_with_scroll)
                            ).props('flat dense align=left').classes('text-xs text-slate-700 w-full justify-start')

                            ui.button(
                                'Seed Cast from Text...',
                                icon='format_list_bulleted_add',
                                on_click=lambda: open_seed_cast_dialog(project.id, selected_book_id, books, refresh_workspace_with_scroll)
                            ).props('flat dense align=left').classes('text-xs text-slate-700 w-full justify-start')

                            async def run_prompt_scan():
                                client = ui.context.client
                                with client:
                                    ui.notify("Scanning prompts.csv for character tags...", type="info")
                                tags = await asyncio.to_thread(extract_characters_from_prompts, project.id)
                                with client:
                                    if tags:
                                        ui.notify(f"Discovered and indexed {len(tags)} character tags!", type="positive")
                                    else:
                                        ui.notify("No new bracketed character tags found.", type="info")
                                await refresh_workspace_with_scroll()

                            ui.button(
                                'Scan Prompts for Tags',
                                icon='tag',
                                on_click=run_prompt_scan
                            ).props('flat dense align=left').classes('text-xs text-slate-700 w-full justify-start')

                            def try_open_import():
                                matching_projects = get_matching_source_projects(project.id)
                                if not matching_projects:
                                    ui.notify("No completed projects with matching character tags found.", type="info")
                                else:
                                    open_import_profiles_dialog(project, matching_projects, refresh_workspace_with_scroll)

                            ui.button(
                                'Import Profiles from Project...',
                                icon='cloud_download',
                                on_click=try_open_import
                            ).props('flat dense align=left').classes('text-xs text-slate-700 w-full justify-start')

                            async def run_prune_unused():
                                client = ui.context.client
                                with client:
                                    ui.notify("Pruning unused characters (0 hits)...", type="info")
                                await asyncio.to_thread(prune_unused_seeded_characters, project.id)
                                with client:
                                    ui.notify("Pruning complete!", type="positive")
                                await refresh_workspace_with_scroll()

                            ui.button(
                                'Prune 0-Hit Characters',
                                icon='cleaning_services',
                                on_click=run_prune_unused
                            ).props('flat dense align=left').classes('text-xs text-slate-700 w-full justify-start')

                            ui.separator().classes('my-1')
                            ui.label('LLM Templates').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wider px-2 py-1')

                            ui.button(
                                'Factual Profiler Template...',
                                icon='science',
                                on_click=lambda: open_prompt_editor_dialog("factual")
                            ).props('flat dense align=left').classes('text-xs text-purple-700 w-full justify-start')

                            ui.button(
                                'Speculative Profiler Template...',
                                icon='psychology',
                                on_click=lambda: open_prompt_editor_dialog("speculative")
                            ).props('flat dense align=left').classes('text-xs text-indigo-700 w-full justify-start')

                            ui.separator().classes('my-1')

                            def confirm_reset():
                                with ui.dialog() as dialog, ui.card().classes('w-[400px] p-6 rounded-xl flex flex-col gap-4'):
                                    ui.label('Reset Character Database?').classes('text-base font-bold text-red-600')
                                    ui.markdown(
                                        "This will **permanently delete** all characters, aliases, and descriptions "
                                        f"for **{project.name}**."
                                    ).classes('text-xs text-slate-600')
                                    
                                    with ui.row().classes('w-full justify-end gap-2'):
                                        ui.button('Cancel', on_click=dialog.close).props('flat')
                                        async def handle_reset():
                                            from services.character_manager import reset_project_characters
                                            await asyncio.to_thread(reset_project_characters, project.id)
                                            global selected_character_id
                                            selected_character_id = None
                                            dialog.close()
                                            ui.notify("Character database reset successfully.", type="positive")
                                            await refresh_workspace_with_scroll()
                                        
                                        ui.button('Reset Everything', on_click=handle_reset).classes('bg-red-600 text-white font-bold text-xs px-3 py-1.5 rounded-lg')
                                dialog.open()

                            ui.button(
                                'Reset Character Database',
                                icon='restart_alt',
                                on_click=confirm_reset
                            ).props('flat dense align=left').classes('text-xs text-red-600 hover:bg-red-50 w-full justify-start font-semibold')

            # Batch Progress Spinner (when running)
            if is_profiling_all:
                with ui.row().classes('w-full items-center gap-3 bg-purple-50/40 border border-purple-200 p-2 rounded-lg shrink-0 mt-1'):
                    def stop_profiling():
                        global cancel_profiling_all, profiling_progress
                        cancel_profiling_all = True
                        profiling_progress = "Stopping..."
                        draw_header_toolbar.refresh()
                        ui.notify("Stop requested...", type="warning")

                    ui.button(
                        'Stop Batch', 
                        icon='stop', 
                        on_click=stop_profiling
                    ).classes('bg-red-600 hover:bg-red-700 text-white font-bold text-xs px-2.5 py-1 rounded-lg shadow-sm shrink-0')

                    ui.spinner(size='xs', color='purple').classes('shrink-0')
                    ui.label(profiling_progress).classes('text-xs font-bold text-purple-700 animate-pulse truncate flex-1')

    @ui.refreshable
    def draw_stats_bar():
        with Session(engine) as session:
            if selected_book_id:
                chars = session.exec(
                    select(Character)
                    .join(CharacterBookLink)
                    .where(CharacterBookLink.book_id == selected_book_id)
                ).all()
                selected_book = session.get(Book, selected_book_id)
                scope_name = selected_book.name if selected_book else "Selected Book"
            else:
                chars = session.exec(
                    select(Character).where(Character.project_id == project.id)
                ).all()
                scope_name = "All Books (Project Wide)"
        
        total_chars = len(chars)
        locked_count = sum(1 for char in chars if char.locked)
        with Session(engine) as session:
            known_counts = project_known_counts(session, project.id)
        fully_profiled = sum(
            1 for char in chars
            if known_counts.get(char.id, 0) >= len(RENDERED_TRAIT_FIELDS))

        with ui.row().classes('w-full items-center gap-4 bg-blue-50/50 border border-blue-100 p-2.5 rounded-xl mb-3 text-xs font-semibold text-blue-700'):
            ui.icon('info', size='xs')
            ui.label(f"Scope: {scope_name} ({total_chars} active)")
            ui.label(f"|  {fully_profiled} fully known")
            ui.label(f"|  {locked_count} locked")

    @ui.refreshable
    def draw_character_list():
        global selected_character_id, search_query, sort_by, filter_status, selected_book_id, active_view_mode, hide_zero_hits
        row_elements.clear()
        if filter_status not in ("all", "needs_description", "fully_known"):
            filter_status = "all"

        with Session(engine) as session:
            all_project_chars = session.exec(
                select(Character).where(Character.project_id == project.id)
            ).all()

            all_aliases = session.exec(
                select(CharacterAlias)
                .join(Character)
                .where(Character.project_id == project.id)
            ).all()

            alias_owner_counts = defaultdict(set)
            for a in all_aliases:
                alias_owner_counts[clean_character_key(a.alias)].add(a.character_id)
            for c in all_project_chars:
                alias_owner_counts[clean_character_key(c.name)].add(c.id)

            collision_names = {k for k, owners in alias_owner_counts.items() if len(owners) > 1}

            if selected_book_id:
                scoped_chars = session.exec(
                    select(Character)
                    .join(CharacterBookLink)
                    .where(CharacterBookLink.book_id == selected_book_id)
                ).all()
                selected_book = session.get(Book, selected_book_id)
                book_name = selected_book.name if selected_book else ""
                book_freq_map = get_book_character_frequency_map(project.name, book_name, session)
            else:
                scoped_chars = all_project_chars
                book_freq_map = {}

            series_freq_map = get_character_frequency_map_db(project.name, session)
            char_aliases: Dict[int, List[CharacterAlias]] = defaultdict(list)
            for a in all_aliases:
                char_aliases[a.character_id].append(a)
            known_counts = project_known_counts(session, project.id)

        def calc_hits(char_obj, aliases_list, freq_dict):
            total = sum(freq_dict.get(clean_character_key(a.alias), 0) for a in aliases_list)
            if not total:
                total = freq_dict.get(clean_character_key(char_obj.name), 0)
            return total

        char_data_list = []
        with Session(engine) as session:
            for char in scoped_chars:
                aliases_list = char_aliases.get(char.id, [])
                series_hits = calc_hits(char, aliases_list, series_freq_map)
                book_hits = calc_hits(char, aliases_list, book_freq_map) if selected_book_id else series_hits

                char_all_terms = {clean_character_key(char.name)}
                char_all_terms.update(clean_character_key(a.alias) for a in aliases_list)
                has_collision = bool(char_all_terms.intersection(collision_names))

                base_ev = session.exec(
                    select(CharacterTimelineEvent)
                    .where(CharacterTimelineEvent.character_id == char.id)
                    .where(CharacterTimelineEvent.book_id == None)
                ).first()

                if base_ev:
                    completion_count = known_counts.get(char.id, 0)
                    completion_target = len(RENDERED_TRAIT_FIELDS)
                    summary_pieces = []
                    if any(_trait_text(getattr(base_ev, key, None))
                           for key in RENDERED_TRAIT_FIELDS):
                        summary = compile_character_visual_prompt(base_ev)
                        if summary:
                            summary_pieces.append(summary)
                    else:
                        if base_ev.demographics: summary_pieces.append(base_ev.demographics)
                        if base_ev.hair_and_face: summary_pieces.append(base_ev.hair_and_face)
                        if base_ev.physical_build: summary_pieces.append(base_ev.physical_build)
                else:
                    completion_count = known_counts.get(char.id, 0)
                    completion_target = len(RENDERED_TRAIT_FIELDS)
                    summary_pieces = []

                char_data_list.append((char, aliases_list, book_hits, series_hits, completion_count, completion_target, summary_pieces, has_collision))

        filtered_list = []
        q = normalize_character_key(search_query)
        for char, aliases_list, b_hits, s_hits, completion_count, completion_target, summary_pieces, has_col in char_data_list:
            active_hits = b_hits if selected_book_id else s_hits
            if hide_zero_hits and active_hits <= 0:
                continue
            if q:
                alias_texts = [clean_character_key(a.alias) for a in aliases_list]
                name_match = q in clean_character_key(char.name)
                alias_match = any(q in t for t in alias_texts)
                if not (name_match or alias_match):
                    continue

            if filter_status == "needs_description":
                # In the prompts but barely described: the review-me shortlist.
                if active_hits <= 0 or completion_count > 2:
                    continue
            elif filter_status == "fully_known":
                if completion_count < completion_target:
                    continue
                
            filtered_list.append((char, aliases_list, b_hits, s_hits, completion_count, completion_target, summary_pieces, has_col))

        if sort_by == "mentions_desc":
            filtered_list.sort(key=lambda x: x[2], reverse=True)
        elif sort_by == "mentions_asc":
            filtered_list.sort(key=lambda x: x[2])
        elif sort_by == "name_asc":
            filtered_list.sort(key=lambda x: x[0].name.lower())
        elif sort_by == "name_desc":
            filtered_list.sort(key=lambda x: x[0].name.lower(), reverse=True)
        elif sort_by == "completion_desc":
            filtered_list.sort(key=lambda x: x[4], reverse=True)

        if selected_character_id is None and filtered_list:
            selected_character_id = filtered_list[0][0].id

        if not filtered_list:
            ui.label(f'0 shown').classes('text-[9px] text-slate-400 px-1 py-0.5')
            ui.label('No matching characters found.').classes('text-xs text-slate-400 text-center py-8 w-full')
        else:
            ui.label(f'{len(filtered_list)} shown'
                     + (' · 0-hit hidden' if hide_zero_hits else '')).classes(
                'text-[9px] text-slate-400 px-1 py-0.5')\
                .tooltip('Prompt hits only hides characters with no bracketed tag in prompts.csv')
            for char, aliases_list, b_hits, s_hits, completion_count, completion_target, summary_pieces, has_col in filtered_list:
                is_selected = char.id == selected_character_id
                bg_class = "bg-blue-50 border-l-4 border-blue-600 font-semibold text-blue-900" if is_selected else "hover:bg-slate-50 text-slate-700"
                border_class = "" if is_selected else "border-l border-slate-100"
                
                row_padding = 'p-2' if active_view_mode == "merge" else 'p-2.5'
                row_el = ui.row().classes(f'w-full {row_padding} rounded-lg cursor-pointer transition-colors justify-between items-center {bg_class} {border_class}')
                row_elements[char.id] = row_el
                
                with row_el.on('click', lambda _, c_id=char.id: select_char(c_id)):
                    with ui.column().classes('gap-0.5 flex-1 min-w-0'):
                        with ui.row().classes('items-center gap-1.5 min-w-0 w-full'):
                            if char.locked:
                                ui.icon('lock', size='12px', color='rose-500').tooltip('Locked')
                            else:
                                ui.icon('face', size='14px', color='slate-400')
                            
                            ui.label(char.name).classes('text-xs truncate font-semibold')

                            if has_col:
                                ui.badge('🟡 Collision', color='amber-50').classes('text-[9px] font-bold text-amber-700 px-1 py-0.5 rounded')\
                                .tooltip('Another character shares a name or alias with this profile')
                        
                        if active_view_mode == "profiles":
                            summary_text = (" • ".join(summary_pieces) if summary_pieces
                                            else ("No traits known yet" if completion_count == 0
                                                  else "Description available in the timeline"))
                            ui.label(summary_text).classes('text-[10px] text-slate-400 truncate w-full')
                    
                    with ui.column().classes('items-end gap-0.5'):
                        if selected_book_id:
                            ui.label(f"{b_hits} in book").classes('text-[9px] font-bold bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded')
                        else:
                            ui.label(f"{s_hits} hits").classes('text-[9px] font-bold bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded')

                        bar_color = ("text-green-600 font-bold"
                                     if (completion_target == 4 and completion_count == 4)
                                     or completion_count >= 6
                                     else "text-purple-600" if completion_count >= 2
                                     else "text-slate-400")
                        ui.label(f"{completion_count}/{completion_target} known").classes(f'text-[9px] font-bold {bar_color}')\
                            .tooltip("Rendered traits known anywhere in the project (Base + all books)")

    @ui.refreshable
    def draw_details_panel():
        global selected_character_id, selected_event_id, selected_book_id, profiler_scan_depth, currently_profiling_char_id, active_view_mode, time_scope_project, timeline_book_id
        
        if selected_character_id is None:
            with ui.column().classes('w-full h-full items-center justify-center text-slate-400 gap-4'):
                ui.icon('person_search', size='xl', color='slate-300')
                ui.label('No Character Selected').classes('text-sm font-bold text-slate-500')
                ui.label('Choose a character from the list on the left to view details.').classes('text-xs text-slate-400 max-w-xs text-center')
            return

        with Session(engine) as session:
            char = session.get(Character, selected_character_id)
            if not char:
                ui.label('Character not found.').classes('text-xs text-slate-400 text-center py-8 w-full')
                return

            all_evs = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == char.id)).all()
            base_ev = next((ev for ev in all_evs if ev.book_id is None), None)
            books_by_id = {b.id: b for b in books}
            order_map = {b.id: (b.book_order or 0) for b in books}
            char_book_ids = sorted(
                {ev.book_id for ev in all_evs if ev.book_id is not None},
                key=lambda bid: order_map.get(bid, 0))
            if timeline_book_id not in char_book_ids:
                timeline_book_id = (
                    selected_book_id if selected_book_id in char_book_ids
                    else (char_book_ids[0] if char_book_ids else None))
            book_max_chapter = 1
            if timeline_book_id is not None:
                latest_prompt_chapter = session.exec(
                    select(func.max(ScenePrompt.chapter_num))
                    .where(ScenePrompt.book_id == timeline_book_id)).one()
                book_max_chapter = int(latest_prompt_chapter or 1)

            if selected_event_id is None:
                if base_ev:
                    selected_event_id = base_ev.id
                else:
                    base_ev = CharacterTimelineEvent(
                        character_id=char.id,
                        book_id=None,
                        chapter_num=0,
                        shot_num=0,
                        label="Base State"
                    )
                    session.add(base_ev)
                    session.commit()
                    selected_event_id = base_ev.id

            active_event = session.get(CharacterTimelineEvent, selected_event_id)
            if not active_event or active_event.character_id != char.id:
                selected_event_id = base_ev.id if base_ev else None
                active_event = base_ev

            aliases = session.exec(
                select(CharacterAlias).where(CharacterAlias.character_id == char.id)
            ).all()

            frequencies = get_character_frequency_map(project.name, books)
            def get_char_mentions(char_obj, aliases_list):
                total = sum(frequencies.get(clean_character_key(a.alias), 0) for a in aliases_list)
                if not total:
                    total = frequencies.get(clean_character_key(char_obj.name), 0)
                return total
            mentions = get_char_mentions(char, aliases)

            all_characters = session.exec(
                select(Character).where(Character.project_id == project.id)
            ).all()
            known_counts_all = project_known_counts(session, project.id)
            known_count = known_counts_all.get(char.id, 0)
            field_stats = character_field_stats(
                session, char,
                book_id=None if time_scope_project else timeline_book_id,
                order_map=order_map)
            pinned_fields = set(get_pinned_fields(char))
            display_event = active_event
            if (pinned_fields and active_event is not None
                    and active_event.book_id is not None):
                display_event, _ = resolve_event_at(
                    session, char, active_event.book_id,
                    active_event.chapter_num, active_event.shot_num,
                    book_order_map=order_map)

        # Helper: compact trait value for the merge view.
        def format_trait(info):
            base = info.get("base") or ""
            latest = info.get("latest") or ""
            if base and latest and base.lower() != latest.lower():
                return f"{base} → {latest}"
            return latest or base or "—"

        # Helper: Render Assigned Aliases & Inline Adder
        def render_alias_board():
            with ui.column().classes('w-full bg-slate-50 p-3.5 rounded-xl border gap-2.5'):
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Assigned Aliases & Target Tags').classes('text-[11px] font-bold text-slate-500 uppercase tracking-wider')
                    ui.label(f'{len(aliases)} aliases linked').classes('text-[10px] font-semibold text-slate-400')

                with ui.row().classes('w-full gap-2 flex-wrap items-center'):
                    for alias in aliases:
                        is_canonical = normalize_character_key(alias.alias) == normalize_character_key(char.name)

                        def make_detach_alias(alias_obj=alias, char_id=char.id):
                            async def handle():
                                new_id = await asyncio.to_thread(
                                    detach_alias_to_character,
                                    project.id,
                                    char_id,
                                    alias_obj.id,
                                    selected_book_id
                                )
                                if new_id:
                                    await refresh_workspace_with_scroll()
                                    ui.notify(f"Detached '{alias_obj.alias}' into a standalone character!", type="positive")
                                else:
                                    ui.notify("Failed to detach alias.", type="warning")
                            return handle

                        with ui.row().classes(
                            'items-center gap-1.5 bg-white border border-slate-200 px-3 py-1 rounded-full text-xs text-slate-800 shadow-sm'
                        ):
                            ui.label(alias.alias).classes('cursor-pointer font-semibold').on(
                                'click', 
                                lambda _, a=alias, c_id=char.id: open_alias_explorer_dialog(
                                    project.id, a, c_id, refresh_workspace_with_scroll
                                )
                            ).tooltip("Click to view transcript occurrences")

                            if not is_canonical:
                                ui.icon('cancel', size='14px', color='slate-400').classes(
                                    'cursor-pointer hover:text-red-500 transition-colors'
                                ).on('click', make_detach_alias(alias, char.id)).tooltip("Detach into standalone character")
                            else:
                                ui.badge('Primary', color='blue-50').classes('text-[9px] text-blue-600 font-bold px-1 rounded')

                # Quick Add Alias Input
                with ui.row().classes('w-full items-center gap-2 mt-0.5'):
                    new_alias_input = ui.input(placeholder="Add new alias tag...").classes('flex-1 text-xs bg-white').props('dense outlined')
                    
                    async def handle_add_alias():
                        val = clean_character_tag(new_alias_input.value)
                        if not val:
                            return
                        val_key = clean_character_key(val)
                        with Session(engine) as session:
                            existing_aliases = session.exec(
                                select(CharacterAlias)
                                .where(CharacterAlias.character_id == char.id)).all()
                            dup = any(clean_character_key(a.alias) == val_key
                                      for a in existing_aliases)
                            if not dup:
                                session.add(CharacterAlias(character_id=char.id, alias=val))
                                session.commit()
                        new_alias_input.set_value("")
                        save_project_characters_to_json(project.id)
                        await refresh_workspace_with_scroll()
                        ui.notify(f"Added alias '{val}'!", type="positive")

                    ui.button('Add Alias', icon='add', on_click=handle_add_alias)\
                        .classes('bg-blue-600 text-white font-bold text-xs px-3 py-2 rounded-lg')

        # Helper: Render Manual Search & Merge Dropdown
        def render_manual_merge():
            other_chars = [c for c in all_characters if c.id != char.id]
            if other_chars:
                with ui.column().classes('w-full bg-slate-50 p-3.5 rounded-xl border gap-2'):
                    ui.label('Manual Search & Merge').classes('text-[11px] font-bold text-slate-500 uppercase tracking-wider')
                    preview_container = ui.column().classes('w-full gap-1')

                    def render_merge_preview(src_id):
                        preview_container.clear()
                        if not src_id:
                            return
                        try:
                            src_id = int(src_id)
                        except (TypeError, ValueError):
                            return
                        with Session(engine) as session:
                            source = session.get(Character, src_id)
                            target = session.get(Character, char.id)
                            if not source or not target:
                                return
                            target_summary = character_trait_summary(session, target)
                            source_summary = character_trait_summary(session, source)
                            target_name = target.name
                            source_name = source.name
                        with preview_container:
                            with ui.row().classes('w-full items-center justify-between gap-2'):
                                ui.label(f'Merge preview · {source_name} → {target_name}').classes(
                                    'text-[10px] font-bold text-slate-600')
                                ui.label('Target wins shared positions').classes(
                                    'text-[9px] font-semibold text-slate-400')
                            with ui.grid().classes('w-full grid-cols-3 gap-x-3 gap-y-0.5'):
                                ui.label('Trait').classes('text-[9px] font-black text-slate-400 uppercase')
                                ui.label(target_name).classes('text-[9px] font-bold text-slate-600 truncate')
                                ui.label(source_name).classes('text-[9px] font-bold text-slate-600 truncate')
                                for field in RENDERED_TRAIT_FIELDS:
                                    t_info = target_summary[field]
                                    s_info = source_summary[field]
                                    ui.label(field.replace('_', ' ')).classes(
                                        'text-[9px] text-slate-500 truncate')
                                    ui.label(format_trait(t_info)).classes(
                                        'text-[9px] text-slate-600 truncate')\
                                        .tooltip(f'Target: {format_trait(t_info)}')
                                    source_text = format_trait(s_info)
                                    source_class = 'text-[9px] text-slate-600 truncate'
                                    if not t_info["latest"] and s_info["latest"]:
                                        source_text += '  (adds)'
                                        source_class = 'text-[9px] font-semibold text-amber-600 truncate'
                                    ui.label(source_text).classes(source_class)\
                                        .tooltip(f'Source: {format_trait(s_info)}')
                            ui.label('Source changes and traits the target lacks are added. '
                                     'Where both describe the same position, the target value wins.'
                                     ).classes('text-[9px] text-slate-400 italic')

                    with ui.row().classes('w-full items-center gap-2'):
                        merge_options = {c.id: f"{c.name} ({c.hit_count or 0} hits)" for c in other_chars}
                        manual_select = ui.select(
                            options=merge_options,
                            label='Select any character to merge into this one...',
                            with_input=True,
                            on_change=lambda e: render_merge_preview(e.value)
                        ).classes('flex-1 bg-white').props('dense outlined clearable')

                        async def handle_manual_merge():
                            src_id = manual_select.value
                            if not src_id:
                                ui.notify("Please select a character to merge.", type="warning")
                                return
                            success = await asyncio.to_thread(merge_character_into_target, src_id, char.id, selected_book_id)
                            if success:
                                ui.notify("Merged successfully!", type="positive")
                                await refresh_workspace_with_scroll()
                            else:
                                ui.notify("Merge failed.", type="negative")

                        ui.button('Merge In', icon='call_merge', on_click=handle_manual_merge)\
                            .classes('bg-blue-600 text-white font-bold text-xs px-3 py-2 rounded-lg')

        # --- COMMON HEADER BAR ---
        with ui.row().classes('w-full justify-between items-center pb-2.5 border-b flex-wrap gap-2'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('face', size='md', color='blue-600')
                
                async def handle_name_blur(e, char_id=char.id):
                    new_name = clean_character_tag(e.sender.value)
                    if not new_name:
                        e.sender.set_value(char.name)
                        return
                    new_key = clean_character_key(new_name)
                    with Session(engine) as session:
                        clash = next(
                            (other for other in session.exec(
                                select(Character)
                                .where(Character.project_id == project.id)
                                .where(Character.id != char_id)).all()
                             if clean_character_key(other.name) == new_key),
                            None)
                        if clash:
                            ui.notify(f"'{new_name}' is already used by another profile.",
                                      type="warning")
                            e.sender.set_value(char.name)
                            return
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
                ).classes('w-56 font-extrabold text-lg text-slate-800').props('dense borderless').on('blur', handle_name_blur)
                
                ui.badge(f'{mentions} mentions', color='blue-50').classes('text-blue-700 text-xs font-bold px-2 py-0.5 rounded-full')

                char_trait_count = known_count
                char_trait_target = len(RENDERED_TRAIT_FIELDS)
                char_badge_color = 'green-100' if char_trait_count >= char_trait_target else 'purple-100' if char_trait_count >= 2 else 'slate-100'
                char_badge_text = 'text-green-800' if char_trait_count >= char_trait_target else 'text-purple-800' if char_trait_count >= 2 else 'text-slate-600'
                ui.badge(f'{char_trait_count}/{char_trait_target} known', color=char_badge_color).classes(f'{char_badge_text} text-xs font-bold px-2 py-0.5 rounded-full')\
                    .tooltip("Rendered traits known anywhere in the project (Base + all books)")
                
            with ui.row().classes('items-center gap-2 flex-grow justify-end'):
                # Mode-Aware AI Check Button (Left-Click: Copy Rendered Prompt, Right-Click: Template Editor)
                def copy_ai_prompt():
                    prompt_text = generate_ai_check_prompt(
                        project.id, char.id, selected_book_id, mode=active_view_mode, event_id=selected_event_id
                    )
                    if prompt_text:
                        ui.clipboard.write(prompt_text)
                        msg_mode = "Alias check" if active_view_mode == "merge" else "Visual sanity check"
                        ui.notify(f"{msg_mode} prompt for '{char.name}' copied to clipboard!", type="positive", icon="content_copy")

                btn_label = 'AI Merge Check' if active_view_mode == "merge" else 'AI Sanity Check'
                btn_tooltip = (
                    'Left-click: Copy alias verification prompt | Right-click: Edit template'
                    if active_view_mode == "merge" else
                    'Left-click: Copy visual description sanity check | Right-click: Edit template'
                )

                ui.button(btn_label, icon='auto_awesome')\
                    .classes('bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs px-2.5 py-1.5 rounded-lg shadow-sm')\
                    .on('click', copy_ai_prompt)\
                    .on('contextmenu.prevent', lambda: open_ai_check_editor_dialog(
                        project.id, char.id, selected_book_id, mode=active_view_mode, event_id=selected_event_id
                    ))\
                    .tooltip(btn_tooltip)

                # Lock Button
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

                # "Become Alias" (Reverse Merge) Action
                ui.button(
                    'Become Alias...',
                    icon='call_merge',
                    on_click=lambda: open_become_alias_dialog(project.id, char, books, refresh_workspace_with_scroll)
                ).classes('bg-amber-600 hover:bg-amber-700 text-white font-bold text-xs px-2.5 py-1.5 rounded-lg shadow-sm')\
                .tooltip("Merge this profile into an established character")

                # Delete Button
                async def delete_profile(c_id=char.id):
                    global selected_character_id
                    with Session(engine) as session:
                        db_char = session.get(Character, c_id)
                        if db_char:
                            aliases_to_del = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == c_id)).all()
                            for a in aliases_to_del: session.delete(a)
                            links_to_del = session.exec(select(CharacterBookLink).where(CharacterBookLink.character_id == c_id)).all()
                            for lk in links_to_del: session.delete(lk)
                            evs_to_del = session.exec(select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id == c_id)).all()
                            for ev in evs_to_del: session.delete(ev)
                            session.delete(db_char)
                            session.commit()
                    save_project_characters_to_json(project.id)
                    selected_character_id = None
                    await refresh_workspace_with_scroll()
                    ui.notify("Character profile deleted.", type="warning")

                ui.button(icon='delete', on_click=delete_profile, color='red')\
                    .props('unelevated dense').classes('p-1.5 rounded-lg text-white').tooltip('Delete Character Profile')

        # =========================================================================
        # WORKFLOW MODE 1: CLEAN & MERGE (Fast Deduplication & Alias Management)
        # =========================================================================
        if active_view_mode == "merge":
            with ui.column().classes('w-full flex-1 overflow-y-auto gap-4 pr-1'):

                # Target traits, so merging does not require a trip to Visual Profiles.
                with Session(engine) as session:
                    target_summary = character_trait_summary(session, char)
                with ui.column().classes('w-full bg-white border rounded-xl p-3 gap-2'):
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label(f'{char.name} traits').classes(
                            'text-[11px] font-bold text-slate-500 uppercase tracking-wider')
                        ui.badge(f'{known_count}/{len(RENDERED_TRAIT_FIELDS)} known',
                                 color='slate-100').classes(
                                     'text-[9px] font-bold text-slate-600 px-2 py-0.5 rounded')
                    with ui.row().classes('w-full flex-wrap gap-1.5'):
                        shown = 0
                        for field, info in target_summary.items():
                            if info["populated"]:
                                ui.badge(f'{field.replace("_", " ")}: {format_trait(info)}',
                                         color='slate-50').classes(
                                             'text-[9px] font-semibold text-slate-600 '
                                             'px-1.5 py-0.5 rounded border border-slate-200')\
                                    .tooltip('Base → latest in the project. Shared positions '
                                             'keep this profile’s value when merging.')
                                shown += 1
                        if not shown:
                            ui.label('No traits known yet.').classes(
                                'text-[10px] text-slate-400 italic')

                # Assigned Aliases Section
                render_alias_board()

                # 💡 Smart Suggested Merges Section
                suggested_matches = get_suggested_alias_merges(project.id, char.id, selected_book_id)
                sug_traits = {}
                if suggested_matches:
                    for sug in suggested_matches:
                        sug_traits[sug["character_id"]] = (
                            known_counts_all.get(sug["character_id"], 0),
                            len(RENDERED_TRAIT_FIELDS))

                with ui.column().classes('w-full bg-amber-50/40 border border-amber-200/80 p-4 rounded-xl gap-3'):
                    with ui.row().classes('w-full items-center justify-between'):
                        with ui.row().classes('items-center gap-1.5'):
                            ui.icon('lightbulb', size='16px', color='amber-600')
                            ui.label('Suggested Merges').classes('text-[11px] font-bold text-amber-900 uppercase tracking-wider')
                        ui.label('Possible unmerged variations in your text').classes('text-[10px] text-amber-700 italic')

                    if not suggested_matches:
                        ui.label('No automatic matching candidates detected for this name.').classes('text-xs text-amber-700/70 italic py-2')
                    else:
                        with ui.column().classes('w-full gap-2'):
                            for sug in suggested_matches:
                                t_count, t_target = sug_traits.get(sug["character_id"], (0, 9))
                                badge_color = 'green-100' if t_count >= t_target else 'purple-100' if t_count >= 2 else 'slate-100'
                                badge_text_color = 'text-green-800' if t_count >= t_target else 'text-purple-800' if t_count >= 2 else 'text-slate-600'

                                with ui.row().classes('w-full items-center justify-between bg-white border border-amber-200 p-2.5 rounded-lg shadow-sm'):
                                    with ui.column().classes('gap-0.5 flex-1 min-w-0 cursor-pointer group')\
                                        .on('click', lambda _, s_id=sug["character_id"]: select_char(s_id))\
                                        .tooltip("Click to view this character's profile"):
                                        with ui.row().classes('items-center gap-2'):
                                            ui.label(sug["name"]).classes('text-xs font-bold text-slate-800 group-hover:text-blue-600 transition-colors truncate')
                                            ui.badge(f"{sug['hits']} hits", color='amber-100').classes('text-amber-800 text-[9px] font-bold px-1.5 py-0.5 rounded')
                                            ui.badge(f"{t_count}/{t_target} known", color=badge_color).classes(f'{badge_text_color} text-[9px] font-bold px-1.5 py-0.5 rounded')
                                        ui.label(f"Match reason: {sug['reason']}").classes('text-[10px] text-slate-400')

                                    async def merge_suggestion(sug_id=sug["character_id"], sug_name=sug["name"]):
                                        success = await asyncio.to_thread(merge_character_into_target, sug_id, char.id, selected_book_id)
                                        if success:
                                            ui.notify(f"Merged '{sug_name}' into {char.name}!", type="positive")
                                            await refresh_workspace_with_scroll()
                                        else:
                                            ui.notify("Merge failed.", type="negative")

                                    ui.button('Merge In', icon='call_merge', on_click=merge_suggestion)\
                                        .classes('bg-amber-600 hover:bg-amber-700 text-white font-bold text-xs px-3 py-1.5 rounded-lg shadow-sm shrink-0')

                # Manual Search & Merge Fallback
                render_manual_merge()

        # =========================================================================
        # WORKFLOW MODE 2: VISUAL PROFILES (Creative Prompting & Trait Curation)
        # =========================================================================
        else:
            with ui.column().classes('w-full flex-1 overflow-y-auto gap-3.5 pr-1 '
                                      'char-details-scroll'):
                
                # Research / Vibe Dedication Row
                with ui.row().classes('w-full items-center justify-between bg-purple-50/40 border border-purple-100 p-3 rounded-xl gap-2'):
                    with ui.row().classes('items-center gap-2'):
                        async def scan_single_char():
                            global currently_profiling_char_id, profiler_scan_depth, selected_event_id
                            client = ui.context.client
                            if char.locked:
                                with client:
                                    ui.notify("Character is locked. Unlock to re-profile.", type="warning")
                                return
                            currently_profiling_char_id = char.id
                            draw_details_panel.refresh()
                            try:
                                with client:
                                    ui.notify(f"Scanning written descriptions for {char.name}...", type="info")
                                await run_stateful_character_profiling(
                                    project.id, char.id, selected_book_id, 
                                    max_chunks_to_scan=profiler_scan_depth, event_id=selected_event_id
                                )
                                with client:
                                    ui.notify("Profiling completed!", type="positive")
                            except Exception as ex:
                                with client:
                                    ui.notify(f"Profiling failed: {ex}", type="negative")
                            finally:
                                currently_profiling_char_id = None
                                await refresh_workspace_with_scroll()

                        async def speculate_single_char():
                            global currently_profiling_char_id, profiler_scan_depth, selected_event_id
                            client = ui.context.client
                            if char.locked:
                                with client:
                                    ui.notify("Character is locked. Unlock to re-profile.", type="warning")
                                return
                            currently_profiling_char_id = char.id
                            draw_details_panel.refresh()
                            try:
                                with client:
                                    ui.notify(f"Speculating character concept for {char.name}...", type="info")
                                await run_stateful_character_profiling(
                                    project.id, char.id, selected_book_id, 
                                    max_chunks_to_scan=profiler_scan_depth, speculate=True, event_id=selected_event_id
                                )
                                with client:
                                    ui.notify("Casting speculation completed!", type="positive")
                            except Exception as ex:
                                with client:
                                    ui.notify(f"Speculation failed: {ex}", type="negative")
                            finally:
                                currently_profiling_char_id = None
                                await refresh_workspace_with_scroll()

                        is_card_profiling = currently_profiling_char_id == char.id
                        if is_card_profiling:
                            with ui.row().classes('items-center gap-1.5 bg-purple-100 px-3 py-1.5 rounded-lg'):
                                ui.spinner(size='xs', color='purple')
                                ui.label('LLM Active...').classes('text-xs text-purple-700 font-bold')
                        else:
                            ui.button('Research (LLM)', icon='science', on_click=scan_single_char)\
                                .classes('text-white font-bold text-xs bg-purple-600 hover:bg-purple-700 shadow-sm')\
                                .tooltip("Scan transcript for written physical descriptions")
                            
                            ui.button('Deduce Vibe', icon='theater_comedy', on_click=speculate_single_char)\
                                .classes('text-white font-bold text-xs bg-indigo-600 hover:bg-indigo-700 shadow-sm')\
                                .tooltip("Deduce characteristics when details are unwritten")

                    # Hairstyle Helper Button
                    ui.button(
                        'Hairstyle Helper',
                        icon='face',
                        on_click=lambda: open_hair_picker_modal(project.id, active_event.id, compiled_desc_input, inputs_dict.get("hair_and_face"))
                    ).classes('bg-purple-50 hover:bg-purple-100 text-purple-700 font-bold text-xs px-3 py-1.5 rounded-lg border border-purple-200 shadow-sm')

                # --- Timeline: book-scoped, change points only ----------------
                book_events = sorted(
                    [ev for ev in all_evs
                     if timeline_book_id is not None
                     and ev.book_id == timeline_book_id],
                    key=lambda ev: (ev.chapter_num, ev.shot_num, ev.id or 0))
                sequence = ([base_ev] if base_ev else []) + book_events
                max_chapter = max([ev.chapter_num for ev in book_events] + [book_max_chapter, 1])

                def select_event(ev_id):
                    global selected_event_id
                    selected_event_id = ev_id
                    draw_details_panel.refresh()

                def step_event(direction):
                    ids = [ev.id for ev in sequence]
                    if not ids:
                        return
                    try:
                        index = ids.index(selected_event_id)
                    except ValueError:
                        index = 0
                    select_event(ids[max(0, min(len(ids) - 1, index + direction))])

                def switch_timeline_book(e):
                    global timeline_book_id, selected_event_id
                    timeline_book_id = e.value
                    evs = sorted(
                        [ev for ev in all_evs
                         if timeline_book_id is not None
                         and ev.book_id == timeline_book_id],
                        key=lambda ev: (ev.chapter_num, ev.shot_num, ev.id or 0))
                    selected_event_id = (evs[0].id if evs
                                         else (base_ev.id if base_ev else None))
                    draw_details_panel.refresh()

                def set_time_scope(value):
                    global time_scope_project
                    time_scope_project = (value == "project")
                    draw_details_panel.refresh()

                with ui.column().classes('w-full bg-slate-50 border p-2.5 rounded-xl gap-2'):
                    with ui.row().classes('w-full items-center gap-2 flex-wrap'):
                        ui.label('Time scope:').classes('text-[10px] font-bold text-slate-500 uppercase tracking-wider')
                        scope_options = ({'book': 'This book', 'project': 'Whole project'}
                                         if char_book_ids else
                                         {'project': 'Whole project'})
                        ui.toggle(scope_options,
                                  value=('project' if time_scope_project or not char_book_ids
                                         else 'book'),
                                  on_change=lambda e: set_time_scope(e.value))\
                            .props('dense unelevated').classes('text-xs')
                        if char_book_ids:
                            ui.label('Timeline book:').classes('text-[10px] font-bold text-slate-500 uppercase tracking-wider')
                            ui.select(
                                options={bid: books_by_id[bid].name for bid in char_book_ids},
                                value=timeline_book_id,
                                on_change=switch_timeline_book
                            ).classes('w-56 text-xs').props('dense outlined')
                        ui.label(f"{len(book_events)} changes").classes('text-[10px] font-bold text-slate-400')
                        with ui.row().classes('items-center gap-1 ml-auto'):
                            ui.button(icon='chevron_left', on_click=lambda: step_event(-1))\
                                .props('flat dense round size=sm').tooltip('Previous change')
                            ui.button(icon='chevron_right', on_click=lambda: step_event(1))\
                                .props('flat dense round size=sm').tooltip('Next change')
                            ui.button(icon='add_circle',
                                      on_click=lambda: open_add_event_dialog(
                                          project.id, char.id, books,
                                          draw_details_panel.refresh))\
                                .props('flat dense').classes('p-1 text-blue-600 hover:bg-blue-50')\
                                .tooltip('Add custom timeline override')
                            if active_event and active_event.book_id is not None:
                                async def delete_timeline_event():
                                    global selected_event_id
                                    with Session(engine) as session:
                                        db_ev = session.get(CharacterTimelineEvent, active_event.id)
                                        if db_ev:
                                            session.delete(db_ev)
                                            session.commit()
                                    save_project_characters_to_json(project.id)
                                    ui.notify("Timeline event deleted.", type="warning")
                                    selected_event_id = base_ev.id if base_ev else None
                                    draw_details_panel.refresh()

                                ui.button(icon='remove_circle', on_click=delete_timeline_event)\
                                    .props('flat dense').classes('p-1 text-red-500 hover:bg-red-50')\
                                    .tooltip('Delete selected timeline event')

                    scope_name = ('whole project'
                                  if time_scope_project or timeline_book_id is None
                                  else (books_by_id[timeline_book_id].name
                                        if timeline_book_id in books_by_id
                                        else 'this book'))
                    ui.label(f"Field colours: {scope_name}").classes('text-[9px] font-semibold text-slate-400')

                    # Chapter axis with one marker per change.
                    with ui.element('div').classes('relative w-full h-9'):
                        ui.element('div').classes('absolute left-2 right-2 top-1/2 h-0.5 bg-slate-200')
                        if base_ev:
                            ui.button(icon='home', on_click=lambda: select_event(base_ev.id))\
                                .props('flat dense round size=sm')\
                                .classes('absolute text-slate-500 bg-white border border-slate-300')\
                                .style('left: 0%; top: 50%; transform: translateY(-50%);')\
                                .tooltip('Base (first appearance)')
                        previous = base_ev
                        for ev in book_events:
                            pct = 2 + 96 * ((ev.chapter_num - 1) / max(1, max_chapter - 1))
                            fields = changed_trait_fields(ev, previous)
                            if any(f in ("sex", "species") for f in fields):
                                marker_class = 'text-red-500 bg-red-50 border-red-300'
                            elif any(f in ("age", "hair_color", "hair_style",
                                           "eye_color", "build") for f in fields):
                                marker_class = 'text-amber-500 bg-amber-50 border-amber-300'
                            else:
                                marker_class = 'text-blue-500 bg-blue-50 border-blue-300'
                            selected = (ev.id == selected_event_id)
                            with ui.button(icon='circle',
                                           on_click=lambda _, eid=ev.id: select_event(eid))\
                                    .props('flat dense round size=sm')\
                                    .classes(f"absolute {marker_class} border"
                                             + (" ring-2 ring-blue-400" if selected else ""))\
                                    .style(f'left: {pct}%; top: 50%; transform: translate(-50%, -50%);'):
                                ui.tooltip(f"Ch {ev.chapter_num} · shot {ev.shot_num} — "
                                           + (", ".join(f.replace('_', ' ') for f in fields)
                                              if fields else "first described"))
                            previous = ev

                    # The precise, readable form of the axis.
                    with ui.column().classes('w-full max-h-48 overflow-y-auto gap-1'):
                        if base_ev:
                            selected = (selected_event_id == base_ev.id)
                            with ui.row().classes(
                                    'w-full items-center gap-2 p-1.5 rounded cursor-pointer '
                                    + ('bg-blue-50 border border-blue-200' if selected
                                       else 'hover:bg-white'))\
                                    .on('click', lambda: select_event(base_ev.id)):
                                ui.label('Base').classes('text-[10px] font-bold text-slate-600 w-16')
                                ui.label('first appearance').classes('text-[10px] text-slate-400')
                        previous = base_ev
                        for ev in book_events:
                            fields = changed_trait_fields(ev, previous)
                            diff = []
                            for field in fields[:3]:
                                after = (getattr(ev, field, None) or '—')
                                if previous is not None:
                                    before = (getattr(previous, field, None) or '—')
                                    diff.append(f"{field.replace('_', ' ')}: {before} → {after}")
                                else:
                                    diff.append(f"{field.replace('_', ' ')}: {after}")
                            selected = (ev.id == selected_event_id)
                            with ui.row().classes(
                                    'w-full items-center gap-2 p-1.5 rounded cursor-pointer '
                                    + ('bg-blue-50 border border-blue-200' if selected
                                       else 'hover:bg-white'))\
                                    .on('click', lambda _, eid=ev.id: select_event(eid)):
                                ui.label(f"Ch {ev.chapter_num} · {ev.shot_num}")\
                                    .classes('text-[10px] font-bold text-slate-600 w-16 shrink-0')
                                fallback = ("first described"
                                            if book_events and ev.id == book_events[0].id
                                            else "changed")
                                ui.label(" · ".join(diff) or fallback)\
                                    .classes('text-[10px] text-slate-500 truncate')
                            previous = ev


                # Compiled Visual Description Prompt Box
                with ui.column().classes('w-full bg-blue-50/20 p-3.5 rounded-xl border border-blue-100 gap-1.5'):
                    ui.label('Compiled Visual Description Prompt').classes('text-[11px] font-bold text-blue-600 uppercase tracking-wider')
                    
                    # Show the computed description, but do NOT persist it just by
                    # opening the panel: visual_description is a manual override,
                    # and the render path compiles from the fields when it is empty.
                    computed_desc = display_event.visual_description or (
                        compile_character_visual_prompt(display_event)
                        if event_trait_count(display_event) else "")
                    if pinned_fields:
                        ui.label(
                            "Pinned traits render from the Base value; the fields below "
                            "show the effective value."
                        ).classes('text-[9px] text-blue-500')

                    def handle_desc_blur(e, ev_id=active_event.id):
                        new_val = e.sender.value.strip()
                        with Session(engine) as session:
                            db_ev = session.get(CharacterTimelineEvent, ev_id)
                            if db_ev:
                                db_ev.visual_description = new_val if new_val else None
                                if db_ev.source == "extraction":
                                    db_ev.source = "manual"  # protect the edit from re-analysis
                                session.add(db_ev)
                                session.commit()
                        save_project_characters_to_json(project.id)
                        ui.notify("Visual Prompt updated!", type="info")

                    compiled_desc_input = ui.textarea(
                        value=computed_desc
                    ).classes('w-full bg-white font-mono text-xs').props('outlined dense autogrow')\
                     .on('blur', handle_desc_blur)\
                     .tooltip("Injected into your rendering prompts. Clear it to let the fields compile it.")
                    if pinned_fields and active_event.book_id is not None:
                        compiled_desc_input.props('readonly').tooltip(
                            "Pinned traits render from Base; select Base to edit the canonical description.")

                # Physical Parameters Grid
                def make_update_handler(ev_id, key, text_area_el):
                    async def handler(e):
                        val = e.sender.value.strip()
                        with Session(engine) as session:
                            db_ev = session.get(CharacterTimelineEvent, ev_id)
                            if db_ev:
                                old_val = getattr(db_ev, key, None)
                                setattr(db_ev, key, val if val != "" else None)
                                if db_ev.source == "extraction":
                                    db_ev.source = "manual"  # protect the edit from re-analysis
                                # Sex and species are character-level, not scene-level:
                                # apply one edit to every event for this character so
                                # the user does not have to touch 20+ rows.
                                if key in ("sex", "species") and val:
                                    for other in session.exec(
                                            select(CharacterTimelineEvent)
                                            .where(CharacterTimelineEvent.character_id
                                                   == db_ev.character_id)).all():
                                        setattr(other, key, val)
                                        if other.book_id is None:
                                            # Character-level traits are canonical:
                                            # protect the Base from a re-analysis vote.
                                            other.source = "manual"
                                        session.add(other)
                                char_obj = session.get(Character, db_ev.character_id)
                                if char_obj and key not in ("sex", "species"):
                                    # Everything else is scene-level, but the user means
                                    # "from here on": carry it through the events that
                                    # still repeat the old value and stop where the book
                                    # changes it. Without this, a Base edit never reaches
                                    # a position that already has an event.
                                    propagate_trait_change(
                                        session, char_obj, key, val, old_val,
                                        book_id=db_ev.book_id,
                                        chapter_num=db_ev.chapter_num,
                                        shot_num=db_ev.shot_num,
                                        log=state.add_console_log)
                                if char_obj and not char_obj.locked:
                                    if event_trait_count(db_ev):
                                        new_prompt = compile_character_visual_prompt(db_ev)
                                        # Only refresh an override that exists; leave
                                        # an uncompiled event dynamic for the renderer.
                                        if db_ev.visual_description:
                                            db_ev.visual_description = new_prompt
                                        text_area_el.set_value(new_prompt)
                                    elif db_ev.visual_description:
                                        db_ev.visual_description = None
                                        text_area_el.set_value("")
                                session.add(db_ev)
                                session.commit()
                        save_project_characters_to_json(project.id)
                        # The status beside each field ("missing" / "1 change") is
                        # computed at render time, so it has to be re-rendered or it
                        # keeps describing the state before the edit - and the
                        # re-render must not move the scroll position.
                        await refresh_after_trait_save()
                        ui.notify("Trait saved.", type="positive", position="bottom-right", timeout=1000)
                    return handler

                # Trait editors: "is it populated?" and "does it change?" are
                # shown as a status next to the label; the pin keeps the Base
                # value for fields that should never change.
                def spans_tooltip(key):
                    info = field_stats.get(key, {})
                    lines = []
                    for span in info.get("spans", []):
                        book = books_by_id.get(span["book_id"])
                        where = (f"{book.name} · Ch {span['start_chapter']}"
                                 if book else
                                 f"Base · Ch {span['start_chapter']}")
                        if span["end_chapter"] != span["start_chapter"]:
                            where += f"–{span['end_chapter']}"
                        lines.append(f"{span['value']}  ({where})")
                    return "\n".join(lines) or "No value in this scope."

                def status_parts(key):
                    info = field_stats.get(key, {})
                    if key in pinned_fields:
                        ignored = info.get("ignored_changes", 0)
                        text = "pinned" + (f" · {ignored} ignored" if ignored else "")
                        return "🔒", text, "text-blue-600", spans_tooltip(key)
                    if not info.get("populated"):
                        return "○", "missing", "text-slate-400", "No value in this scope."
                    within = info.get("within_book_changes", 0)
                    across = info.get("across_book_changes", 0)
                    if within:
                        text = f"{within} change" + ("s" if within != 1 else "")
                        if across:
                            text += f" · {across} across"
                        color = ("text-red-600" if key in ("sex", "species")
                                 else "text-amber-600")
                        return "●", text, color, spans_tooltip(key)
                    if across:
                        text = f"{across} across books"
                        return "●", text, "text-blue-600", spans_tooltip(key)
                    return "●", "stable", "text-green-600", spans_tooltip(key)

                def toggle_pin(key):
                    with Session(engine) as session:
                        db_char = session.get(Character, char.id)
                        if not db_char:
                            return
                        pins = set(get_pinned_fields(db_char))
                        if key in pins:
                            pins.discard(key)
                        else:
                            base = session.exec(
                                select(CharacterTimelineEvent)
                                .where(CharacterTimelineEvent.character_id == char.id)
                                .where(CharacterTimelineEvent.book_id == None)
                            ).first()
                            if base is None:
                                base = CharacterTimelineEvent(
                                    character_id=char.id, book_id=None,
                                    chapter_num=0, shot_num=0, label="Base State")
                                session.add(base)
                            value = (getattr(base, key, None) or "").strip()
                            if not value:
                                value = (getattr(active_event, key, None) or "").strip()
                                if value:
                                    setattr(base, key, value)
                            if not value:
                                ui.notify(
                                    "Nothing to pin for this trait.",
                                    type="warning", position="bottom-right")
                                return
                            # A pin is a manual canonical decision: protect the
                            # Base snapshot from being overwritten by re-analysis.
                            if base.source != "manual":
                                base.source = "manual"
                            session.add(base)
                            pins.add(key)
                        set_pinned_fields(session, db_char, sorted(pins))
                        session.commit()
                    save_project_characters_to_json(project.id)
                    draw_details_panel.refresh()

                inputs_dict = {}
                trait_fields = [
                    ("sex", "Sex (male / female)"),
                    ("species", "Species (human unless stated)"),
                    ("age", "Age (young / adult / middle-aged / old)"),
                    ("hair_color", "Hair Colour"),
                    ("hair_style", "Hair Style"),
                    ("eye_color", "Eye Colour"),
                    ("build", "Build (Height / Frame)"),
                    ("attire", "Attire (Wearing Now)"),
                    ("carried", "Carried (Held or Worn)"),
                    ("notes", "Notes (extracted, never rendered)"),
                ]
                with ui.grid().classes('w-full grid-cols-1 md:grid-cols-2 gap-3'):
                    for key, label in trait_fields:
                        is_rendered = key in RENDERED_TRAIT_FIELDS
                        is_pinned = is_rendered and key in pinned_fields
                        val = (getattr(base_ev, key, None) if is_pinned and base_ev
                               else getattr(active_event, key, None)) or ""
                        with ui.column().classes('w-full gap-0.5'):
                            with ui.row().classes('w-full items-center gap-1'):
                                ui.label(label).classes(
                                    'text-[10px] font-bold text-slate-500 uppercase tracking-wide')
                                if is_rendered:
                                    dot, text, color, tip = status_parts(key)
                                    ui.label(f"{dot} {text}").classes(
                                        f'text-[9px] font-semibold {color}')\
                                        .tooltip(tip)
                                else:
                                    ui.label('note').classes('text-[9px] text-slate-400')
                                if is_rendered:
                                    ui.button(
                                        icon='push_pin',
                                        on_click=lambda _, k=key: toggle_pin(k)
                                    ).props('flat dense round size=sm').classes(
                                        'ml-auto ' + ('text-blue-600' if is_pinned
                                                      else 'text-slate-300 hover:text-blue-500'))\
                                     .tooltip('Pinned to the Base value; click to unpin'
                                              if is_pinned else
                                              'Pin this trait to the Base value')
                            input_el = ui.input(
                                value=val, placeholder=label
                            ).classes('w-full bg-white').props('outlined dense')\
                             .on('blur', make_update_handler(active_event.id, key,
                                                             compiled_desc_input))
                            if is_pinned and active_event.book_id is not None:
                                input_el.props('readonly').tooltip(
                                    'Pinned to Base; select Base to edit the canonical value.')
                            inputs_dict[key] = input_el

                legacy_fields = [
                    ("demographics", "Demographics (Age, Race, Gender)"),
                    ("hair_and_face", "Hair & Face Details"),
                    ("physical_build", "Physical Build (Height/Weight/Posture)"),
                    ("distinguishing_marks", "Distinguishing Marks & Key Accessories"),
                ]
                with ui.expansion('Legacy description fields', icon='history', value=False)\
                        .classes('w-full border rounded-xl bg-slate-50/40 text-xs'):
                    ui.label(
                        'The four fields the older profiler wrote. Kept editable so '
                        'existing projects lose nothing; new descriptions use the '
                        'fields above.'
                    ).classes('text-[10px] text-slate-500 leading-tight')
                    with ui.grid().classes('w-full grid-cols-1 md:grid-cols-2 gap-3'):
                        for key, label in legacy_fields:
                            val = getattr(active_event, key) or ""
                            inputs_dict[key] = ui.input(label=label, value=val)\
                                .classes('w-full bg-white').props('outlined dense')\
                                .on('blur', make_update_handler(active_event.id, key, compiled_desc_input))

                # Appearance Map across Series
                book_mentions = get_character_book_mentions(project.id, char.id)
                if book_mentions:
                    with ui.column().classes('w-full bg-slate-50 p-3 rounded-xl border gap-1.5'):
                        ui.label('Appearance Map across Series (Click a book to audit prompts)').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wider')
                        with ui.row().classes('w-full gap-1.5 flex-wrap'):
                            for b_name, m_count in book_mentions.items():
                                ui.badge(f"{b_name} ({m_count} hits)", color='purple-50')\
                                    .classes('text-purple-700 text-xs font-semibold px-2 py-1 rounded cursor-pointer hover:bg-purple-100 transition-colors')\
                                    .on('click', lambda _, b=b_name: open_appearance_prompt_modal(project, char, b))\
                                    .tooltip(f"Click to audit and edit matching prompts in {b_name}")

                # Assigned Aliases Tag Board
                render_alias_board()

                # Manual Search & Merge Fallback
                render_manual_merge()

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
                ui.label('No characters detected in this project yet.').classes('text-sm font-semibold text-slate-500')
                ui.label('Scan your prompts.csv for bracketed character tags like [Dino], or seed from plaintext.').classes('text-xs text-slate-400 max-w-sm text-center leading-normal')
                
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

                with ui.row().classes('items-center gap-3'):
                    ui.button(
                        'Scan Prompts for Tags', 
                        icon='tag', 
                        on_click=run_prompt_scan_empty
                    ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-5 py-2.5 rounded-lg shadow-sm')

                    ui.button(
                        'Seed Cast from Text',
                        icon='format_list_bulleted_add',
                        on_click=lambda: open_seed_cast_dialog(project.id, selected_book_id, books, refresh_workspace_with_scroll)
                    ).classes('bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-xs px-5 py-2.5 rounded-lg shadow-sm')
            return

        workspace_was_empty = False
        draw_stats_bar()

        with ui.grid().classes('w-full grid-cols-1 lg:grid-cols-12 gap-4 items-start'):
            # --- LEFT PANEL: Searchable Character List ---
            with ui.card().classes('col-span-4 p-3.5 border rounded-xl bg-white h-[650px] flex flex-col gap-2.5'):
                ui.label('Characters List').classes('text-xs font-bold text-slate-800 border-b pb-1')
                
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
                            "needs_description": "Needs Description (0–2)",
                            "fully_known": "Fully Known (9/9)"
                        },
                        value=filter_status,
                        on_change=on_filter_change
                    ).props('dense outlined').classes('w-44 text-xs')

                def on_hide_zero_change(e):
                    global hide_zero_hits
                    hide_zero_hits = bool(e.value)
                    draw_character_list.refresh()

                ui.switch('Prompt hits only', value=hide_zero_hits,
                          on_change=on_hide_zero_change)\
                    .classes('text-xs text-slate-600')\
                    .tooltip('Hide characters the analysis created that never appear '
                             'as a bracketed tag in prompts.csv')
                
                with ui.column().classes('w-full flex-1 overflow-y-auto gap-1 pr-1 char-scroll-list'):
                    draw_character_list()

            # --- RIGHT PANEL: Focused Details Workspace ---
            with ui.card().classes('col-span-8 p-5 border rounded-xl bg-white h-[650px] flex flex-col gap-3'):
                draw_details_panel()

    draw_header_toolbar()
    def render_analysis_entry():
        """One line pointing at the analysis pass, since this is where the gaps show.

        The dashboard owns the run controls; this is the door from "this character
        has 0/9 fields" to "re-extract the book". It is deliberately a banner and
        not a second copy of the control panel.
        """
        def turn_on(value: bool):
            state.cast_enabled = bool(value)
            from services.project_settings import save_project_settings_to_disk
            save_project_settings_to_disk(project.id)
            if hasattr(state, "action_buttons_refresh") and state.action_buttons_refresh:
                state.action_buttons_refresh()
            draw_workspace_layout.refresh()

        def open_rerun():
            from ui.pages.project.dashboard import open_stage_rerun
            target = selected_book_id or (books[0].id if books else None)
            if target is None:
                ui.notify("No volume to analyse.", type="info")
                return
            book = next((b for b in books if b.id == target), None)
            if book is None:
                return
            from database.connection import get_setting
            from pathlib import Path
            book_dir = (Path(get_setting("output_dir", "./output")).resolve()
                        / project.name / book.name)
            open_stage_rerun(target, book_dir)

        with ui.row().classes('w-full items-center gap-3 bg-blue-50/60 border border-blue-100 '
                              'rounded-lg px-3 py-1.5'):
            if not state.cast_enabled:
                ui.icon('face', size='16px', color='blue-400')
                ui.label('Character descriptions are off for this project. Existing data is '
                         'kept but nothing here is being updated.').classes(
                    'text-[10px] text-slate-600 flex-1')
                ui.button('Turn on', icon='toggle_on', on_click=lambda: turn_on(True)) \
                    .props('flat dense').classes('text-[10px] text-blue-700 font-bold')
            else:
                ui.icon('person_search', size='16px', color='blue-400')
                ui.label('Names and descriptions come from the analysis pass over the book. '
                         'It can run before or after prompts - tags compile at render time.').classes(
                    'text-[10px] text-slate-600 flex-1')
                target_name = next(
                    (b.name for b in books
                     if b.id == (selected_book_id or (books[0].id if books else None))), "")
                btn = ui.button('Analyse cast...', icon='replay', on_click=open_rerun) \
                    .props('flat dense').classes('text-[10px] text-blue-700 font-bold')
                if target_name:
                    btn.tooltip(f"Pick the stages for {target_name}")

    render_analysis_entry()
    draw_workspace_layout()
