import asyncio
import re
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from nicegui import ui
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Project, Book, Character, CharacterAlias, CharacterStateModifier
from services.character_manager import (
    extract_characters_from_prompts,
    save_project_characters_to_json,
    merge_character_aliases,
    run_stateful_character_profiling,
    get_character_mention_chunks,
    get_character_book_mentions,
    save_setting,
    auto_merge_project_characters,
    compile_character_visual_prompt
)

# Active local state trackers
selected_book_id: Optional[int] = None  # None translates cleanly to "All Books"
is_profiling_all: bool = False
cancel_profiling_all: bool = False
currently_profiling_char_id: Optional[int] = None
profiling_progress: str = ""
profiler_scan_depth: int = 5  # Scan depth (how many text chunks LLM reads)

# Dynamic Filter and Interactive Selection states
search_query: str = ""
sort_by: str = "mentions_desc"
filter_status: str = "all"
selected_character_id: Optional[int] = None

# High-density caching tracker to prevent list scroll position resets
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


def open_batch_profiler_dialog(
    project: Project, 
    books: List[Book], 
    refresh_ui_callback: Any, 
    refresh_toolbar_callback: Any,
    refresh_details_callback: Any
):
    """Opens options settings before launching character batch profiling runs."""
    global is_profiling_all, currently_profiling_char_id, profiling_progress, cancel_profiling_all, profiler_scan_depth

    # Safe refresh helper to bypass stale slot stack warnings during async tasks
    async def safe_refresh(callback_fn):
        try:
            if asyncio.iscoroutinefunction(callback_fn):
                await callback_fn()
            else:
                callback_fn()
        except RuntimeError as e:
            if "The parent element this slot belongs to" in str(e):
                pass  # Discard stale slot lookup errors from closed modal tasks
            else:
                raise

    with ui.dialog() as dialog, ui.card().classes('w-[520px] max-w-[95vw] p-6 rounded-xl flex flex-col gap-4 overflow-hidden'):
        
        # Header Row
        with ui.row().classes('w-full justify-between items-center border-b pb-3 shrink-0'):
            with ui.column().classes('gap-0.5'):
                ui.label('Batch Profiler Options').classes('text-base font-bold text-slate-800')
                ui.label('Configure rules for the automated batch sequence.').classes('text-xs text-slate-500')
            ui.button(icon='close', on_click=dialog.close).props('flat dense').classes('text-slate-400')

        # Shared Global Settings (Mentions limit)
        with ui.row().classes('w-full items-center justify-between gap-3 bg-slate-50 p-3 rounded-lg border shrink-0'):
            with ui.column().classes('gap-0.5'):
                ui.label('Minimum Mentions Limit').classes('text-xs font-semibold text-slate-700')
                ui.label('Skips low-frequency background characters.').classes('text-[10px] text-slate-400')
            min_mentions_input = ui.number(value=5, min=1, step=1).classes('w-16 bg-white').props('outlined dense')

        # Tab Navigation
        with ui.tabs().classes('w-full border-b shrink-0') as tabs:
            factual_tab = ui.tab('Factual').classes('text-xs font-bold')
            creative_tab = ui.tab('Creative').classes('text-xs font-bold')

        # Trigger batch execution
        async def run_configured_batch(start_mode: str):
            global is_profiling_all, currently_profiling_char_id, profiling_progress, cancel_profiling_all, profiler_scan_depth
            
            clear_existing = clear_existing_cb.value
            min_mentions = int(min_mentions_input.value or 1)
            run_creative_after = run_creative_after_cb.value if start_mode == "factual" else False
            creative_target = speculate_criteria.value  # "0", "1", "all"

            # Map selected stopping conditions
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

            # Helper to gather base unlocked queue matching minimum mentions limit
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
                    ui.notify(f"Starting factual batch for {len(factual_queue)} characters (Min. {min_mentions} mentions)...", type="info")

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
                            
                            # Using run_coroutine to cleanly run safe_refresh asynchronously from inner sync callbacks
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
                
                # Helper to count non-null and non-empty traits
                def get_trait_count(char_obj) -> int:
                    fields = [
                        char_obj.demographics, char_obj.physical_build, 
                        char_obj.hair_and_face, char_obj.distinguishing_marks
                    ]
                    return sum(1 for f in fields if f and str(f).strip() and str(f).lower() != "null")

                creative_queue = []
                for char in base_queue:
                    # Re-verify character record from DB to get the newly generated Factual traits
                    with Session(engine) as session:
                        db_char = session.get(Character, char.id)
                        if not db_char or db_char.locked:
                            continue
                        traits_count = get_trait_count(db_char)
                    
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
                            # Run profiling with speculation enabled
                            await run_stateful_character_profiling(
                                project_id=project.id, 
                                character_id=char.id, 
                                book_id=selected_book_id, 
                                max_chunks_to_scan=profiler_scan_depth,
                                clear_existing=False,  # Preserve existing traits and only fill the missing ones
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

        # Tab Content Panels
        with ui.tab_panels(tabs, value=factual_tab).classes('w-full flex-1 min-h-0 bg-transparent'):
            
            # FACTUAL PANEL
            with ui.tab_panel(factual_tab).classes('p-0 flex flex-col gap-4 h-full justify-between'):
                with ui.column().classes('w-full gap-4'):
                    clear_existing_cb = ui.checkbox(
                        'Wipe existing profile traits before profiling', 
                        value=False
                    ).tooltip("If checked, completely clears demographics, build, etc. before researching. If unchecked, preserves completed fields and searches only for missing traits.")

                    # Early Stopping Rules
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
                    ).tooltip("If checked, characters meeting Creative Tab criteria will immediately undergo speculation following the Factual pass.")

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
                        "Creative Casting uses local LLM **speculation** to fill in sparse descriptions. "
                        "It deduces age, demographic profiles, hair, build, or permanent accessories "
                        "even if they are never explicitly written in the transcript."
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

    # Added a stable height 'h-[650px]' to prevent flex-1 height collapse
    with ui.dialog() as dialog, ui.card().classes('w-[750px] max-w-[95vw] h-[650px] max-h-[90vh] p-6 rounded-xl flex flex-col overflow-hidden'):
        
        def reset():
            from services.character_manager import get_default_character_template
            editor.value = get_default_character_template()
            ui.notify("Template reset to system default.", type="info")

        def save():
            save_setting("character_profiler_template", editor.value)
            ui.notify("Custom profiler prompt template saved!", type="positive")
            dialog.close()

        # Fixed Header row with Actions on Top (no floating overlap, non-shrinkable)
        with ui.row().classes('w-full justify-between items-center border-b pb-3 mb-3 shrink-0'):
            with ui.column().classes('gap-0.5'):
                ui.label('Customize Character Profiler Prompt').classes('text-base font-bold text-slate-800')
                ui.label('Configure system instructions sent to the local LLM.').classes('text-xs text-slate-500')
            
            with ui.row().classes('gap-2 items-center'):
                ui.button('Reset', on_click=reset, color='amber').props('flat').classes('text-xs font-semibold')
                ui.button('Cancel', on_click=dialog.close, color='slate').props('flat').classes('text-xs font-semibold')
                ui.button('Save Template', on_click=save).classes('bg-blue-600 text-white font-bold text-xs px-3 py-1.5 rounded-lg shadow-sm')

        # Scrollable column with a robust minimum height context
        with ui.column().classes('w-full flex-1 overflow-y-auto overflow-x-hidden gap-4 pr-1 min-w-0'):
            ui.markdown(
                "Configure the instructions sent to the LLM during character research. "
                "You can use these dynamic placeholder tags:\n"
                "- `{character_name}`: Canonical name of the character\n"
                "- `{aliases}`: Comma-separated list of known aliases\n"
                "- `{known_traits}`: Attributes already discovered\n"
                "- `{unknown_traits}`: Attributes still needing discovery"
            ).classes('text-xs text-slate-500 leading-relaxed bg-slate-50 p-3 rounded-lg border w-full')

            editor = ui.textarea(
                label='System Instructions Template', 
                value=current_template
            ).classes('w-full font-mono text-xs').props('outlined autogrow')

    dialog.open()

def open_alias_explorer_dialog(project_id: int, alias: CharacterAlias, parent_char_id: int, refresh_callback: Any):
    """Opens a modal displaying where the selected alias occurs within project audiobook transcripts."""
    from services.character_manager import get_alias_occurrences
    
    occurrences = get_alias_occurrences(project_id, alias.alias)
    current_index = 0

    with ui.dialog() as dialog, ui.card().classes('w-[600px] max-w-[95vw] p-6 rounded-xl flex flex-col gap-4 overflow-hidden'):
        
        with ui.row().classes('w-full justify-between items-center border-b pb-3 shrink-0'):
            with ui.column().classes('gap-0.5'):
                ui.label(f'Context Explorer: "{alias.alias}"').classes('text-base font-bold text-slate-800')
                book_label = ui.label('Loading context...').classes('text-xs text-slate-500')
            ui.button(icon='close', on_click=dialog.close).props('flat dense').classes('text-slate-400')

        # Highlighted context container
        with ui.column().classes('w-full flex-1 justify-center items-center py-6 min-h-[160px] bg-slate-50 border rounded-lg px-4 overflow-y-auto'):
            context_html = ui.html('').classes('text-sm text-slate-700 leading-relaxed text-center')

        # Footer Actions & Pagination Row (Simplified: delete button removed)
        with ui.row().classes('w-full justify-between items-center pt-2 shrink-0'):
            # Empty element keeps pagination elements pushed to the right
            ui.label('').classes('flex-grow')

            # Navigators
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
            
            # Disable buttons at boundaries
            if current_index > 0:
                prev_btn.enable()
            else:
                prev_btn.disable()
                
            if current_index < len(occurrences) - 1:
                next_btn.enable()
            else:
                next_btn.disable()

        # Load first occurrence
        update_display()

    dialog.open()

def render_characters_tab(project: Project, books: List[Book], refresh_parent: Optional[Any] = None):
    # Keep selected_book_id as None to default to Project-wide All Books scans

    # --- SCROLL PRESERVATION UTILITIES ---
    async def refresh_workspace_with_scroll():
        """Refreshes the entire workspace while maintaining the exact scroll offset of the character list."""
        try:
            scroll_pos = await ui.run_javascript("document.querySelector('.char-scroll-list')?.scrollTop || 0")
        except Exception:
            scroll_pos = 0
        
        draw_workspace_layout.refresh()
        
        await asyncio.sleep(0.1)  # Allow DOM nodes to be fully created
        if scroll_pos > 0:
            ui.run_javascript(f"const el = document.querySelector('.char-scroll-list'); if (el) el.scrollTop = {scroll_pos};")

    async def refresh_list_with_scroll():
        """Refreshes only the list view while maintaining the exact scroll offset."""
        try:
            scroll_pos = await ui.run_javascript("document.querySelector('.char-scroll-list')?.scrollTop || 0")
        except Exception:
            scroll_pos = 0
        
        draw_character_list.refresh()
        
        await asyncio.sleep(0.05)
        if scroll_pos > 0:
            ui.run_javascript(f"const el = document.querySelector('.char-scroll-list'); if (el) el.scrollTop = {scroll_pos};")

    def select_char(c_id):
        """Changes focus and toggles selection styles without rebuilding the scrolling list element."""
        global selected_character_id
        old_id = selected_character_id
        selected_character_id = c_id
        
        # Style transition (pure Python element class mutation without full pane rebuilds)
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
        
        with ui.row().classes('w-full items-center justify-between bg-slate-50 border p-4 rounded-xl mb-4 gap-4'):
            with ui.column().classes('gap-0'):
                ui.label('Character Visual Profiles').classes('text-base font-bold text-slate-800')
                ui.label('Tag aliases, generate physical characteristics using local LLMs, and track character changes.').classes('text-xs text-slate-500')
            
            with ui.row().classes('items-center gap-3'):
                ui.label('Source:').classes('text-xs font-semibold text-slate-500')
                
                # Default to dynamic Project-wide mapping
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
                ).classes('w-56 bg-white').props('outlined dense')

                ui.label('Depth (Chunks):').classes('text-xs font-semibold text-slate-500 ml-1')
                def handle_depth_change(e):
                    global profiler_scan_depth
                    profiler_scan_depth = int(e.value)

                ui.number(
                    value=profiler_scan_depth,
                    min=1,
                    max=100,
                    step=1,
                    on_change=handle_depth_change
                ).classes('w-16 bg-white').props('outlined dense')

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
                ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-3 py-2 rounded-lg').tooltip("Scan prompts.csv for bracketed character tags")

                async def run_auto_merge():
                    client = ui.context.client
                    with client:
                        ui.notify("Running smart auto-merge of character tags...", type="info")
                    merged_log = await asyncio.to_thread(auto_merge_project_characters, project.id)
                    with client:
                        if merged_log:
                            ui.notify(f"Auto-merged {len(merged_log)} duplicate tags!", type="positive")
                            for log in merged_log[:5]:
                                ui.notify(f"Merged {log['merged_name']} -> {log['target_name']}", type="info")
                        else:
                            ui.notify("No matching alias tags to merge found.", type="info")
                    await refresh_workspace_with_scroll()

                ui.button(
                    'Auto-Merge',
                    icon='merge_type',
                    on_click=run_auto_merge
                ).classes('bg-teal-600 hover:bg-teal-700 text-white font-bold text-xs px-3 py-2 rounded-lg').tooltip("Fuzzy-merge common names, titles, and possessives")

                ui.button(
                    'Prompt',
                    icon='edit_note',
                    on_click=open_prompt_editor_dialog
                ).classes('bg-slate-600 hover:bg-slate-700 text-white font-bold text-xs px-3 py-2 rounded-lg').tooltip("Customize visual profiler LLM instructions")

                if is_profiling_all:
                    with ui.row().classes('items-center gap-2 bg-purple-50 border border-purple-200 px-3 py-1.5 rounded-lg'):
                        ui.spinner(size='xs', color='purple')
                        ui.label(profiling_progress).classes('text-xs font-semibold text-purple-700 animate-pulse')
                        
                        def stop_profiling():
                            global cancel_profiling_all, profiling_progress
                            cancel_profiling_all = True
                            profiling_progress = "Stopping after current character..."
                            draw_header_toolbar.refresh()
                            ui.notify("Stop requested. Halting sequence after active run...", type="warning")

                        ui.button(
                            'Stop', 
                            icon='stop', 
                            on_click=stop_profiling
                        ).classes('bg-red-600 hover:bg-red-700 text-white font-bold text-xs px-2.5 py-1 rounded-lg')
                else:
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
                    ).classes('bg-purple-600 hover:bg-purple-700 text-white font-bold text-xs px-3 py-2 rounded-lg')


    @ui.refreshable
    def draw_stats_bar():
        with Session(engine) as session:
            all_characters = session.exec(
                select(Character).where(Character.project_id == project.id)
            ).all()
        
        total_chars = len(all_characters)
        fully_profiled = 0
        locked_count = 0
        
        for char in all_characters:
            if char.locked:
                locked_count += 1
            fields = [
                char.demographics, char.physical_build, 
                char.hair_and_face, char.distinguishing_marks
            ]
            if sum(1 for f in fields if f and str(f).strip()) == 4:
                fully_profiled += 1

        with ui.row().classes('w-full items-center gap-4 bg-blue-50/50 border border-blue-100 p-3 rounded-xl mb-4 text-xs font-semibold text-blue-700'):
            ui.icon('info', size='xs')
            ui.label(f"Database Stats: {total_chars} total characters discovered.")
            ui.label(f"|  {fully_profiled} fully profiled (4/4 traits)")
            ui.label(f"|  {locked_count} locked/manually curated")

    @ui.refreshable
    def draw_character_list():
        global selected_character_id, search_query, sort_by, filter_status
        row_elements.clear()  # Purge pointer maps to rebuild lists
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
        for char in all_characters:
            aliases_list = char_aliases.get(char.id, [])
            mentions = get_char_mentions(char, aliases_list)
            fields = [
                char.demographics, char.physical_build,
                char.hair_and_face, char.distinguishing_marks
            ]
            completion_count = sum(1 for f in fields if f and str(f).strip())
            char_data_list.append((char, aliases_list, mentions, completion_count))

        filtered_list = []
        q = search_query.lower().strip()
        for char, aliases_list, mentions, completion_count in char_data_list:
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
                
            filtered_list.append((char, aliases_list, mentions, completion_count))

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

        # Added the 'char-scroll-list' class here to easily preserve the scroll offset on refreshes
        with ui.column().classes('w-full flex-1 overflow-y-auto gap-1 pr-1 char-scroll-list'):
            if not filtered_list:
                ui.label('No characters match filters.').classes('text-xs text-slate-400 text-center py-8 w-full')
            else:
                for char, aliases_list, mentions, completion_count in filtered_list:
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
                            
                            summary_pieces = []
                            if char.demographics: summary_pieces.append(char.demographics)
                            if char.hair_and_face: summary_pieces.append(char.hair_and_face)
                            if char.physical_build: summary_pieces.append(char.physical_build)
                            
                            summary_text = " • ".join(summary_pieces) if summary_pieces else "No traits profiled yet"
                            ui.label(summary_text).classes('text-[10px] text-slate-400 truncate w-full')
                        
                        with ui.column().classes('items-end gap-1'):
                            ui.label(f"{mentions} hits").classes('text-[10px] font-bold bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded')
                            bar_color = "text-green-600 font-bold" if completion_count == 4 else "text-purple-600" if completion_count >= 2 else "text-slate-400"
                            ui.label(f"{completion_count}/4 traits").classes(f'text-[9px] font-bold {bar_color}')

    @ui.refreshable
    def draw_details_panel():
        global selected_character_id, selected_book_id, profiler_scan_depth, currently_profiling_char_id
        
        if selected_character_id is None:
            with ui.column().classes('w-full h-full items-center justify-center text-slate-400 gap-4'):
                ui.icon('person_search', size='xl', color='slate-300')
                ui.label('No Character Selected').classes('text-sm font-bold text-slate-500')
                ui.label('Choose a character from the left panel list to view and edit their details.').classes('text-xs text-slate-400 max-w-xs text-center')
            return

        with Session(engine) as session:
            char = session.get(Character, selected_character_id)
            if not char:
                ui.label('Character not found.').classes('text-xs text-slate-400 text-center py-8 w-full')
                return

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
                char.demographics, char.physical_build,
                char.hair_and_face, char.distinguishing_marks
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
                
            with ui.row().classes('items-center gap-2'):
                async def scan_single_char():
                    global currently_profiling_char_id, profiler_scan_depth
                    client = ui.context.client
                    currently_profiling_char_id = char.id
                    draw_details_panel.refresh()
                    
                    try:
                        with client:
                            ui.notify(f"Running LLM research pipeline for {char.name}...", type="info")
                        await run_stateful_character_profiling(project.id, char.id, selected_book_id, max_chunks_to_scan=profiler_scan_depth)
                        with client:
                            ui.notify("Profiling completed successfully!", type="positive")
                    except Exception as ex:
                        with client:
                            ui.notify(f"Profiling failed: {str(ex)}", type="negative")
                    
                    currently_profiling_char_id = None
                    await refresh_workspace_with_scroll()

                async def speculate_single_char():
                    global currently_profiling_char_id, profiler_scan_depth
                    client = ui.context.client
                    currently_profiling_char_id = char.id
                    draw_details_panel.refresh()
                    
                    try:
                        with client:
                            ui.notify(f"Speculating character casting vibe for {char.name}...", type="info")
                        await run_stateful_character_profiling(
                            project.id, 
                            char.id, 
                            selected_book_id, 
                            max_chunks_to_scan=profiler_scan_depth,
                            speculate=True
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
                    ).classes('text-white font-bold text-xs bg-purple-600 hover:bg-purple-700').tooltip("Scan the text for actual, written physical descriptions of this character.")
                    
                    ui.button(
                        'Deduce Vibe', 
                        icon='theater_comedy', 
                        on_click=speculate_single_char
                    ).classes('text-white font-bold text-xs bg-indigo-600 hover:bg-indigo-700').tooltip("Deduce gender, age, job, and cast a plausible visual description when no physical descriptions are written in the text.")

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

                            mods_to_del = session.exec(
                                select(CharacterStateModifier).where(CharacterStateModifier.character_id == c_id)
                            ).all()
                            for m in mods_to_del:
                                session.delete(m)

                            session.delete(db_char)
                            session.commit()
                    save_project_characters_to_json(project.id)
                    selected_character_id = None
                    await refresh_workspace_with_scroll()
                    ui.notify("Character profile deleted.", type="warning")

                ui.button(
                    icon='delete', 
                    on_click=delete_profile
                ).props('flat dense').classes('bg-red-50 text-red-500 hover:bg-red-100 p-1.5 rounded-lg').tooltip('Delete Character Profile')

        with ui.column().classes('w-full flex-1 overflow-y-auto gap-4 pr-1'):
            with ui.column().classes('w-full bg-slate-50 p-4 rounded-xl border gap-3'):
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Assigned Aliases & Target Tags').classes('text-[11px] font-bold text-slate-500 uppercase tracking-wider')
                    ui.label(f'{completion_count}/4 attributes populated').classes('text-[10px] font-bold text-purple-700 bg-purple-50 px-2 py-0.5 rounded-full')
                
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
                                        
                                        # Spin off a standalone character if it was not the primary self-alias
                                        if alias_name.lower() != char.name.lower():
                                            new_char = Character(project_id=project.id, name=alias_name)
                                            session.add(new_char)
                                            session.commit()
                                            
                                            new_alias = CharacterAlias(character_id=new_char.id, alias=alias_name)
                                            session.add(new_alias)
                                            session.commit()
                                            
                                            new_char.visual_description = compile_character_visual_prompt(new_char)
                                            session.add(new_char)
                                            session.commit()
                                            
                                            ui.notify(f"Spun off '{alias_name}' into standalone profile!", type="positive")

                                    # Clear out parent character if empty of aliases
                                    rem = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char_id)).all()
                                    if not rem:
                                        db_char = session.get(Character, char_id)
                                        if db_char:
                                            session.delete(db_char)
                                            session.commit()
                                            selected_character_id = None
                                
                                save_project_characters_to_json(project.id)
                                await refresh_workspace_with_scroll()
                                ui.notify("Alias removed.", type="info")
                            return handle

                        # Interactive Custom Chip: click text for Context Modal, click cancel icon to delete
                        with ui.row().classes(
                            'items-center gap-1.5 bg-white border border-slate-200 px-2.5 py-1 rounded-full text-xs text-slate-800 hover:bg-slate-50 transition-colors shadow-sm'
                        ):
                            # Clickable Text to trigger Dialog Context Explorer
                            ui.label(alias.alias).classes('cursor-pointer font-medium').on(
                                'click', 
                                lambda _, a=alias, c_id=char.id: open_alias_explorer_dialog(
                                    project.id, a, c_id, draw_workspace_layout.refresh
                                )
                            ).tooltip("Click to view transcript occurrences")
                            
                            # Clean click action to trigger deletion/spin-off
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

            # Dynamic Appearance Map across series
            book_mentions = get_character_book_mentions(project.id, char.id)
            if book_mentions:
                with ui.column().classes('w-full bg-slate-50 p-4 rounded-xl border gap-2 mt-1'):
                    ui.label('Appearance Map across Series').classes('text-[11px] font-bold text-slate-500 uppercase tracking-wider')
                    with ui.row().classes('w-full gap-2 flex-wrap'):
                        for b_name, m_count in book_mentions.items():
                            ui.badge(f"{b_name} ({m_count} hits)", color='purple-50').classes('text-purple-700 text-xs font-semibold px-2 py-1 rounded')

            # Row 2.5: Composite prompt editor card
            with ui.column().classes('w-full bg-blue-50/20 p-4 rounded-xl border border-blue-100 gap-2'):
                ui.label('Compiled Visual Description Prompt').classes('text-[11px] font-bold text-blue-600 uppercase tracking-wider')
                
                if not char.visual_description:
                    char.visual_description = compile_character_visual_prompt(char)
                    with Session(engine) as session:
                        db_char = session.get(Character, char.id)
                        if db_char:
                            db_char.visual_description = char.visual_description
                            session.add(db_char)
                            session.commit()
                    save_project_characters_to_json(project.id)

                def handle_desc_blur(e, char_id=char.id):
                    new_val = e.sender.value.strip()
                    with Session(engine) as session:
                        db_char = session.get(Character, char_id)
                        if db_char:
                            db_char.visual_description = new_val if new_val else None
                            session.add(db_char)
                            session.commit()
                    save_project_characters_to_json(project.id)
                    ui.notify("Visual Prompt overridden!", type="info")

                compiled_desc_input = ui.textarea(
                    value=char.visual_description
                ).classes('w-full bg-white font-mono text-xs').props('outlined dense autogrow')\
                 .on('blur', handle_desc_blur)\
                 .tooltip("This string is what actually replaces the [bracketed] tags inside your ComfyUI prompting pipeline")

            ui.label('Physical Description Parameters').classes('text-[11px] font-bold text-slate-500 uppercase tracking-wider mt-1')
            with ui.grid().classes('w-full grid-cols-1 md:grid-cols-2 gap-3'):
                fields = [
                    ("demographics", "Demographics (Age, Race, Gender)"),
                    ("hair_and_face", "Hair & Face Details"),
                    ("physical_build", "Physical Build (Height/Weight/Posture)"),
                    ("distinguishing_marks", "Distinguishing Marks & Key Accessories")
                ]
                
                # Upgraded: Directly update the visual prompt textarea's value on-the-fly to stop layout redraw lag
                def make_update_handler(char_id, key, text_area_el):
                    def handler(e):
                        val = e.sender.value.strip()
                        with Session(engine) as session:
                            db_char = session.get(Character, char_id)
                            if db_char:
                                setattr(db_char, key, val if val != "" else None)
                                
                                if not db_char.locked:
                                    new_prompt = compile_character_visual_prompt(db_char)
                                    db_char.visual_description = new_prompt
                                    text_area_el.set_value(new_prompt)
                                    
                                session.add(db_char)
                                session.commit()
                        save_project_characters_to_json(project.id)
                        ui.notify("Trait saved.", type="positive", position="bottom-right", timeout=1000)
                    return handler

                for key, label in fields:
                    val = getattr(char, key) or ""
                    ui.input(
                        label=label, 
                        value=val
                    ).classes('w-full bg-white').props('outlined dense').on('blur', make_update_handler(char.id, key, compiled_desc_input))

    @ui.refreshable
    def draw_workspace_layout():
        with Session(engine) as session:
            any_characters = session.exec(
                select(Character).where(Character.project_id == project.id)
            ).first()

        if not any_characters:
            with ui.column().classes('w-full items-center justify-center p-12 text-slate-400 border border-dashed rounded-xl bg-slate-50 gap-4'):
                ui.icon('face', size='xl', color='slate-300')
                ui.label('No characters detected or generated in this project yet.').classes('text-sm font-semibold text-slate-500')
                ui.label('The system needs bracketed names like [Dino] to exist in your prompts.csv files first.').classes('text-xs text-slate-400 max-w-sm text-center leading-normal')
                
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

        draw_stats_bar()

        with ui.grid().classes('w-full grid-cols-1 lg:grid-cols-12 gap-4 items-start'):
            # --- LEFT PANEL: High Density Searchable List (col-span-4) ---
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
                
                draw_character_list()

            # --- RIGHT PANEL: Selected Curation Workspace Card (col-span-8) ---
            with ui.card().classes('col-span-8 p-6 border rounded-xl bg-white h-[650px] flex flex-col gap-4'):
                draw_details_panel()

    # Layout container hierarchy
    draw_header_toolbar()
    draw_workspace_layout()