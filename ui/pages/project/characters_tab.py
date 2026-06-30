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
    save_setting,
    auto_merge_project_characters
)

# Active local state trackers
selected_book_id: Optional[int] = None
is_profiling_all: bool = False
currently_profiling_char_id: Optional[int] = None
profiling_progress: str = ""
profiler_scan_depth: int = 5  # Scan depth (how many text chunks LLM reads)

# Dynamic Filter and Interactive Selection states
search_query: str = ""
sort_by: str = "mentions_desc"
filter_status: str = "all"
selected_character_id: Optional[int] = None


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


def open_prompt_editor_dialog():
    """Renders a modal to customize the LLM profiler template instructions."""
    current_template = get_setting("character_profiler_template", "")
    if not current_template:
        from services.character_manager import get_default_character_template
        current_template = get_default_character_template()

    with ui.dialog() as dialog, ui.card().classes('w-[800px] max-w-full p-6 rounded-xl'):
        ui.label('Customize Character Profiler Prompt').classes('text-lg font-bold text-slate-800 mb-1')
        ui.markdown(
            "Configure the instructions sent to the LLM during character research. "
            "You can use these dynamic placeholder tags:\n"
            "- `{character_name}`: Canonical name of the character\n"
            "- `{aliases}`: Comma-separated list of known aliases\n"
            "- `{known_traits}`: Attributes already discovered\n"
            "- `{unknown_traits}`: Attributes still needing discovery"
        ).classes('text-xs text-slate-500 mb-4 leading-relaxed')

        editor = ui.textarea(
            label='System Instructions Template', 
            value=current_template
        ).classes('w-full h-96 font-mono text-xs').props('outlined autogrow')

        with ui.row().classes('w-full justify-end gap-3 mt-4'):
            def reset():
                from services.character_manager import get_default_character_template
                editor.value = get_default_character_template()
                ui.notify("Template reset to system default.", type="info")

            def save():
                save_setting("character_profiler_template", editor.value)
                ui.notify("Custom profiler prompt template saved!", type="positive")
                dialog.close()

            ui.button('Reset to Default', on_click=reset, color='amber').classes('text-xs')
            ui.button('Cancel', on_click=dialog.close, color='slate').props('flat').classes('text-xs')
            ui.button('Save Template', on_click=save).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-4 py-2 rounded-lg')
    dialog.open()


def render_characters_tab(project: Project, books: List[Book], refresh_parent: Optional[Any] = None):
    global selected_book_id
    if selected_book_id is None and books:
        selected_book_id = books[0].id

    # Master refreshable container
    @ui.refreshable
    def draw_characters_view():
        global selected_book_id, is_profiling_all, currently_profiling_char_id, profiling_progress, profiler_scan_depth
        global search_query, sort_by, filter_status, selected_character_id

        # Load raw frequencies to enable dynamic sorting
        frequencies = get_character_frequency_map(project.name, books)

        with Session(engine) as session:
            all_characters = session.exec(
                select(Character).where(Character.project_id == project.id)
            ).all()
            
            # Fetch aliases for all characters to enable tag grouping
            char_aliases: Dict[int, List[CharacterAlias]] = {}
            for char in all_characters:
                aliases = session.exec(
                    select(CharacterAlias).where(CharacterAlias.character_id == char.id)
                ).all()
                char_aliases[char.id] = aliases

        # Helper to compute total mentions of a character based on its alias tags
        def get_char_mentions(char_obj, aliases_list):
            total = 0
            for a in aliases_list:
                total += frequencies.get(a.alias.lower(), 0)
            if not aliases_list:
                total = frequencies.get(char_obj.name.lower(), 0)
            return total

        # Build list of character tuple metadata
        char_data_list = []
        for char in all_characters:
            aliases_list = char_aliases.get(char.id, [])
            mentions = get_char_mentions(char, aliases_list)
            
            # Compute total trait completion (0 to 8 attributes filled)
            fields = [
                char.sex_or_gender, char.approximate_age, char.ethnicity_or_race,
                char.height_or_stature, char.weight_or_build, char.hair_color_and_style,
                char.facial_features, char.distinguishing_marks
            ]
            completion_count = sum(1 for f in fields if f and str(f).strip())
            char_data_list.append((char, aliases_list, mentions, completion_count))

        # Filter the lists matching Search Queries and Filters
        filtered_list = []
        q = search_query.lower().strip()
        for char, aliases_list, mentions, completion_count in char_data_list:
            if q:
                alias_texts = [a.alias.lower() for a in aliases_list]
                name_match = q in char.name.lower()
                alias_match = any(q in t for t in alias_texts)
                if not (name_match or alias_match):
                    continue
            
            if filter_status == "incomplete" and completion_count == 8:
                continue
            elif filter_status == "locked" and not char.locked:
                continue
            elif filter_status == "unlocked" and char.locked:
                continue
                
            filtered_list.append((char, aliases_list, mentions, completion_count))

        # Apply Sort preferences
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

        # Enforce Auto-selection fallback of first list item on boot
        if selected_character_id is None and filtered_list:
            selected_character_id = filtered_list[0][0].id

        # 1. Header Toolbar Panel
        with ui.row().classes('w-full items-center justify-between bg-slate-50 border p-4 rounded-xl mb-4 gap-4'):
            with ui.column().classes('gap-0'):
                ui.label('Character Visual Profiles').classes('text-base font-bold text-slate-800')
                ui.label('Tag aliases, generate physical characteristics using local LLMs, and track character changes.').classes('text-xs text-slate-500')
            
            with ui.row().classes('items-center gap-3'):
                # Selected Book Dropdown for profiling scans
                ui.label('Source:').classes('text-xs font-semibold text-slate-500')
                book_options = {b.id: b.name for b in books}
                
                def handle_book_change(val):
                    global selected_book_id
                    selected_book_id = val
                    
                ui.select(
                    options=book_options,
                    value=selected_book_id,
                    on_change=lambda e: handle_book_change(e.value)
                ).classes('w-44 bg-white').props('outlined dense')

                # Depth Selector (Scan depth)
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

                # Interactive Actions
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
                    draw_characters_view.refresh()

                ui.button(
                    'Scan Tags', 
                    icon='tag', 
                    on_click=run_prompt_scan
                ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-3 py-2 rounded-lg').tooltip("Scan prompts.csv for bracketed character tags")

                # Auto-Merge Action
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
                    draw_characters_view.refresh()

                ui.button(
                    'Auto-Merge',
                    icon='merge_type',
                    on_click=run_auto_merge
                ).classes('bg-teal-600 hover:bg-teal-700 text-white font-bold text-xs px-3 py-2 rounded-lg').tooltip("Fuzzy-merge common names, titles, and possessives")

                # Custom Prompt Settings Button
                ui.button(
                    'Prompt',
                    icon='edit_note',
                    on_click=open_prompt_editor_dialog
                ).classes('bg-slate-600 hover:bg-slate-700 text-white font-bold text-xs px-3 py-2 rounded-lg').tooltip("Customize visual profiler LLM instructions")

                # Profiler batch action runner
                async def profile_all_unlocked():
                    global is_profiling_all, currently_profiling_char_id, profiling_progress, profiler_scan_depth
                    client = ui.context.client
                    if not selected_book_id:
                        with client:
                            ui.notify("Please select a profiling source book first.", type="warning")
                        return
                    
                    is_profiling_all = True
                    draw_characters_view.refresh()
                    
                    with Session(engine) as session:
                        unlocked_chars = session.exec(
                            select(Character).where(Character.project_id == project.id).where(Character.locked == False)
                        ).all()

                    with client:
                        ui.notify(f"Starting batch profiling for {len(unlocked_chars)} characters...", type="info")
                    
                    for idx, char in enumerate(unlocked_chars):
                        currently_profiling_char_id = char.id
                        profiling_progress = f"Profiling {char.name} ({idx + 1}/{len(unlocked_chars)})..."
                        draw_characters_view.refresh()
                        
                        try:
                            await run_stateful_character_profiling(project.id, char.id, selected_book_id, max_chunks_to_scan=profiler_scan_depth)
                        except Exception as ex:
                            print(f"[Profiler] Error scanning {char.name}: {str(ex)}")

                    is_profiling_all = False
                    currently_profiling_char_id = None
                    profiling_progress = ""
                    with client:
                        ui.notify("Batch profiling complete!", type="positive")
                    draw_characters_view.refresh()

                if is_profiling_all:
                    with ui.row().classes('items-center gap-2 bg-purple-50 border border-purple-200 px-3 py-1.5 rounded-lg'):
                        ui.spinner(size='xs', color='purple')
                        ui.label(profiling_progress).classes('text-xs font-semibold text-purple-700 animate-pulse')
                else:
                    ui.button(
                        'Profile All', 
                        icon='bolt', 
                        on_click=profile_all_unlocked
                    ).classes('bg-purple-600 hover:bg-purple-700 text-white font-bold text-xs px-3 py-2 rounded-lg')

        # Empty State fallback if no characters are loaded
        if not all_characters:
            with ui.column().classes('w-full items-center justify-center p-12 text-slate-400 border border-dashed rounded-xl bg-slate-50 gap-4'):
                ui.icon('face', size='xl', color='slate-300')
                ui.label('No characters detected or generated in this project yet.').classes('text-sm font-semibold text-slate-500')
                ui.label('The system needs bracketed names like [Dino] to exist in your prompts.csv files first.').classes('text-xs text-slate-400 max-w-sm text-center leading-normal')
                ui.button(
                    'Scan for Bracketed Prompt Tags', 
                    icon='tag', 
                    on_click=run_prompt_scan
                ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-5 py-2.5 rounded-lg shadow-sm')
            return

        # 2. Dynamic Database Metadata summary stat row
        total_chars = len(all_characters)
        fully_profiled = sum(1 for x in char_data_list if x[3] == 8)
        locked_count = sum(1 for x in char_data_list if x[0].locked)
        
        with ui.row().classes('w-full items-center gap-4 bg-blue-50/50 border border-blue-100 p-3 rounded-xl mb-4 text-xs font-semibold text-blue-700'):
            ui.icon('info', size='xs')
            ui.label(f"Database Stats: {total_chars} total characters discovered.")
            ui.label(f"|  {fully_profiled} fully profiled (8/8 traits)")
            ui.label(f"|  {locked_count} locked/manually curated")

        # 3. Split-Screen Workspace Grid
        with ui.grid().classes('w-full grid-cols-1 lg:grid-cols-12 gap-4 items-start'):
            
            # --- LEFT PANEL: High Density Searchable List (col-span-4) ---
            with ui.card().classes('col-span-4 p-4 border rounded-xl bg-white h-[650px] flex flex-col gap-3'):
                ui.label('Characters List').classes('text-sm font-bold text-slate-800 border-b pb-1.5')
                
                # Search Bar with dynamic filter reload hook
                def on_search_change(e):
                    global search_query
                    search_query = e.value or ""
                    draw_characters_view.refresh()
                
                ui.input(
                    placeholder='Search name or alias...',
                    value=search_query,
                    on_change=on_search_change
                ).props('dense outlined clearable').classes('w-full text-xs')
                
                # Sort and Filter Selectors
                with ui.row().classes('w-full gap-2 items-center'):
                    def on_sort_change(e):
                        global sort_by
                        sort_by = e.value
                        draw_characters_view.refresh()
                        
                    def on_filter_change(e):
                        global filter_status
                        filter_status = e.value
                        draw_characters_view.refresh()
                        
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
                
                # Compact list container (scrolling)
                with ui.column().classes('w-full flex-1 overflow-y-auto gap-1 pr-1'):
                    if not filtered_list:
                        ui.label('No characters match filters.').classes('text-xs text-slate-400 text-center py-8 w-full')
                    else:
                        for char, aliases_list, mentions, completion_count in filtered_list:
                            is_selected = char.id == selected_character_id
                            bg_class = "bg-blue-50 border-l-4 border-blue-600 font-semibold text-blue-900" if is_selected else "hover:bg-slate-50 text-slate-700"
                            border_class = "" if is_selected else "border-l border-slate-100"
                            
                            def select_char(c_id=char.id):
                                global selected_character_id
                                selected_character_id = c_id
                                draw_characters_view.refresh()
                                
                            with ui.row().classes(f'w-full p-2.5 rounded-lg cursor-pointer transition-colors justify-between items-center {bg_class} {border_class}').on('click', lambda _, c_id=char.id: select_char(c_id)):
                                with ui.column().classes('gap-0.5 flex-1 min-w-0'):
                                    with ui.row().classes('items-center gap-1.5 min-w-0 w-full'):
                                        if char.locked:
                                            ui.icon('lock', size='12px', color='rose-500').tooltip('Locked')
                                        else:
                                            ui.icon('face', size='14px', color='slate-400')
                                        ui.label(char.name).classes('text-xs truncate font-semibold')
                                    
                                    # Formulate a clean trait preview string under the name
                                    summary_pieces = []
                                    if char.sex_or_gender: summary_pieces.append(char.sex_or_gender)
                                    if char.approximate_age: summary_pieces.append(char.approximate_age)
                                    if char.ethnicity_or_race: summary_pieces.append(char.ethnicity_or_race)
                                    if char.hair_color_and_style: summary_pieces.append(char.hair_color_and_style)
                                    
                                    summary_text = " • ".join(summary_pieces) if summary_pieces else "No traits profiled yet"
                                    ui.label(summary_text).classes('text-[10px] text-slate-400 truncate w-full')
                                
                                with ui.column().classes('items-end gap-1'):
                                    ui.label(f"{mentions} hits").classes('text-[10px] font-bold bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded')
                                    bar_color = "text-green-600 font-bold" if completion_count == 8 else "text-purple-600" if completion_count >= 4 else "text-slate-400"
                                    ui.label(f"{completion_count}/8 traits").classes(f'text-[9px] font-bold {bar_color}')

            # --- RIGHT PANEL: Selected Curation Workspace Card (col-span-8) ---
            with ui.card().classes('col-span-8 p-6 border rounded-xl bg-white h-[650px] flex flex-col gap-4'):
                # Load the currently active character selection details
                selected_char_tuple = next((x for x in char_data_list if x[0].id == selected_character_id), None)
                
                if not selected_char_tuple:
                    with ui.column().classes('w-full h-full items-center justify-center text-slate-400 gap-4'):
                        ui.icon('person_search', size='xl', color='slate-300')
                        ui.label('No Character Selected').classes('text-sm font-bold text-slate-500')
                        ui.label('Choose a character from the left panel list to view and edit their details.').classes('text-xs text-slate-400 max-w-xs text-center')
                else:
                    char, aliases, mentions, completion_count = selected_char_tuple
                    
                    # Row 1: Focused Profile Header
                    with ui.row().classes('w-full justify-between items-center pb-3 border-b flex-wrap gap-3'):
                        with ui.row().classes('items-center gap-3'):
                            ui.icon('face', size='md', color='blue-600')
                            
                            def handle_name_blur(e, char_id=char.id):
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
                                draw_characters_view.refresh()

                            ui.input(
                                value=char.name
                            ).classes('w-64 font-extrabold text-lg text-slate-800').props('dense borderless').on('blur', handle_name_blur)
                            
                            ui.badge(f'{mentions} total mentions', color='blue-50').classes('text-blue-700 text-xs font-bold px-2.5 py-1 rounded-full')
                            
                        # Contextual Action buttons
                        with ui.row().classes('items-center gap-2'):
                            # Single profile scanner
                            async def scan_single_char():
                                global currently_profiling_char_id, profiler_scan_depth
                                client = ui.context.client
                                if not selected_book_id:
                                    with client:
                                        ui.notify("Please select a profiling source book first.", type="warning")
                                    return
                                currently_profiling_char_id = char.id
                                draw_characters_view.refresh()
                                
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
                                draw_characters_view.refresh()

                            is_card_profiling = currently_profiling_char_id == char.id
                            if is_card_profiling:
                                with ui.row().classes('items-center gap-1.5 bg-purple-50 px-3 py-1.5 rounded-lg border border-purple-200'):
                                    ui.spinner(size='xs', color='purple')
                                    ui.label('LLM Researching...').classes('text-xs text-purple-700 font-bold')
                            else:
                                ui.button(
                                    'Research (LLM)', 
                                    icon='science', 
                                    on_click=scan_single_char
                                ).classes('text-white font-bold text-xs bg-purple-600 hover:bg-purple-700')

                            # Toggle Locked status
                            def toggle_locked(c_id=char.id, val=not char.locked):
                                with Session(engine) as session:
                                    db_char = session.get(Character, c_id)
                                    if db_char:
                                        db_char.locked = val
                                        session.add(db_char)
                                        session.commit()
                                save_project_characters_to_json(project.id)
                                draw_characters_view.refresh()
                                ui.notify(f"Profile {'Locked' if val else 'Unlocked'}!", type="info")

                            lock_icon = "lock" if char.locked else "lock_open"
                            lock_color = "bg-rose-50 text-rose-600 hover:bg-rose-100" if char.locked else "bg-slate-100 text-slate-600 hover:bg-slate-200"
                            ui.button(
                                icon=lock_icon, 
                                on_click=lambda c_id=char.id: toggle_locked(c_id)
                            ).props('flat dense').classes(f'p-1.5 rounded-lg {lock_color}').tooltip('Toggle manual editing lock')

                            # Delete profile action
                            def delete_profile(c_id=char.id):
                                global selected_character_id
                                with Session(engine) as session:
                                    db_char = session.get(Character, c_id)
                                    if db_char:
                                        # Wipe associated aliases via ORM
                                        aliases_to_del = session.exec(
                                            select(CharacterAlias).where(CharacterAlias.character_id == c_id)
                                        ).all()
                                        for a in aliases_to_del:
                                            session.delete(a)

                                        # Wipe state modifiers via ORM
                                        mods_to_del = session.exec(
                                            select(CharacterStateModifier).where(CharacterStateModifier.character_id == c_id)
                                        ).all()
                                        for m in mods_to_del:
                                            session.delete(m)

                                        session.delete(db_char)
                                        session.commit()
                                save_project_characters_to_json(project.id)
                                selected_character_id = None  # Reset selection
                                draw_characters_view.refresh()
                                ui.notify("Character profile deleted.", type="warning")

                            ui.button(
                                icon='delete', 
                                on_click=delete_profile
                            ).props('flat dense').classes('bg-red-50 text-red-500 hover:bg-red-100 p-1.5 rounded-lg').tooltip('Delete Character Profile')

                    # Scrolling card workspace content
                    with ui.column().classes('w-full flex-1 overflow-y-auto gap-4 pr-1'):
                        
                        # Row 2: Aliases Tag Card Row
                        with ui.column().classes('w-full bg-slate-50 p-4 rounded-xl border gap-3'):
                            with ui.row().classes('w-full justify-between items-center'):
                                ui.label('Assigned Aliases & Target Tags').classes('text-[11px] font-bold text-slate-500 uppercase tracking-wider')
                                ui.label(f'{completion_count}/8 attributes populated').classes('text-[10px] font-bold text-purple-700 bg-purple-50 px-2 py-0.5 rounded-full')
                            
                            with ui.row().classes('w-full gap-2 flex-wrap items-center'):
                                for alias in aliases:
                                    def delete_alias(a_id=alias.id, char_id=char.id):
                                        global selected_character_id
                                        with Session(engine) as session:
                                            db_alias = session.get(CharacterAlias, a_id)
                                            if db_alias:
                                                session.delete(db_alias)
                                                session.commit()
                                            
                                            # Clean up if Character is now empty of aliases
                                            rem = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char_id)).all()
                                            if not rem:
                                                db_char = session.get(Character, char_id)
                                                if db_char:
                                                    session.delete(db_char)
                                                    session.commit()
                                                    selected_character_id = None
                                        save_project_characters_to_json(project.id)
                                        draw_characters_view.refresh()
                                        ui.notify("Alias removed.", type="info")

                                    ui.chip(
                                        alias.alias, 
                                        removable=True, 
                                        on_value_change=lambda e, a_id=alias.id: delete_alias(a_id) if not e.value else None
                                    ).classes('text-xs bg-white border border-slate-200 text-slate-800')

                            # Contextual Autocomplete Merge Dropdown
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
                                        draw_characters_view.refresh()

                                    ui.button(
                                        'Merge',
                                        icon='call_merge', 
                                        on_click=handle_merge_click
                                    ).classes('bg-blue-600 text-white font-bold text-xs px-3 py-2 rounded-lg')

                        # Row 3: 8-Field Editable Profile Parameters Grid
                        ui.label('Physical Description Parameters').classes('text-[11px] font-bold text-slate-500 uppercase tracking-wider mt-1')
                        with ui.grid().classes('w-full grid-cols-1 md:grid-cols-2 gap-3'):
                            fields = [
                                ("sex_or_gender", "Gender (man/woman/boy/girl)"),
                                ("approximate_age", "Approximate Age"),
                                ("ethnicity_or_race", "Race or Ethnicity"),
                                ("height_or_stature", "Height or Stature"),
                                ("weight_or_build", "Weight or Build"),
                                ("hair_color_and_style", "Hair Color & Style"),
                                ("facial_features", "Facial Features"),
                                ("distinguishing_marks", "Distinguishing Marks (Misc)")
                            ]
                            
                            def make_update_handler(char_id, key):
                                def handler(e):
                                    val = e.sender.value.strip()
                                    with Session(engine) as session:
                                        db_char = session.get(Character, char_id)
                                        if db_char:
                                            setattr(db_char, key, val if val != "" else None)
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
                                ).classes('w-full bg-white').props('outlined dense').on('blur', make_update_handler(char.id, key))

    draw_characters_view()