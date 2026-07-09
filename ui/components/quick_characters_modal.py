# ui/components/quick_characters_modal.py

import asyncio
import re
from typing import List, Dict, Any, Optional
from nicegui import ui
from sqlmodel import Session, select
from database.connection import engine
from database.models import Book, Project, Character, CharacterAlias, CharacterTimelineEvent
from services.character_manager import (
    compile_character_visual_prompt,
    save_project_characters_to_json,
    run_stateful_character_profiling,
    get_alias_occurrences,
)
from ui.pages.project.characters_tab import get_character_frequency_map

try:
    from services.character_manager import merge_character_aliases
except ImportError:
    def merge_character_aliases(project_id: int, dest_char_id: int, source_alias_ids: List[int]):
        with Session(engine) as session:
            for alias_id in source_alias_ids:
                alias = session.get(CharacterAlias, alias_id)
                if alias:
                    existing = session.exec(
                        select(CharacterAlias)
                        .where(CharacterAlias.character_id == dest_char_id)
                        .where(CharacterAlias.alias == alias.alias)
                    ).first()
                    if existing:
                        session.delete(alias)
                    else:
                        alias.character_id = dest_char_id
                        session.add(alias)
            session.commit()


class QuickCharactersModal:
    # ui/components/quick_characters_modal.py

    def __init__(self, project_id: int, book_id: int, initial_char_id: Optional[int] = None, on_change_callback: Optional[callable] = None):
        self.project_id = project_id
        self.book_id = book_id
        self.on_change_callback = on_change_callback
        
        # State tracking
        self.selected_char_id: Optional[int] = initial_char_id
        self.selected_event_id: Optional[int] = None  # Tracks which chronological event state is being edited
        self.active_row_id: Optional[int] = None
        self.is_profiling = False
        self.search_query = ""
        
        # Load core data
        with Session(engine) as session:
            self.project = session.get(Project, self.project_id)
            self.book = session.get(Book, self.book_id)
            self.books_list = session.exec(select(Book).where(Book.project_id == self.project_id)).all()
            
        self.frequencies = get_character_frequency_map(self.project.name, self.books_list)
        
        # Force the dialog to be built globally in the persistent page-layout slot
        # This prevents it from being destroyed when dynamic sub-containers (like chips) are cleared.
        with ui.context.client:
            self.dialog = ui.dialog()

    def open(self):
        with self.dialog, ui.card().classes('w-full max-w-[95vw] lg:max-w-5xl h-[650px] p-5 rounded-xl flex flex-col gap-3 outline-none bg-white'):
            # --- MODAL HEADER ---
            with ui.row().classes('w-full justify-between items-center border-b pb-2 flex-shrink-0'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('people', size='sm', color='blue-600')
                    ui.label('Project Characters Quick Manager').classes('text-sm font-bold text-slate-800')
                ui.button(icon='close', on_click=self.dialog.close).props('flat round dense').classes('text-slate-400')

            # --- SPLIT BODY CONTAINER ---
            with ui.grid(columns='300px 1fr').classes('w-full flex-1 overflow-hidden gap-4 h-full min-h-0'):
                
                # --- LEFT COLUMN (SEARCH & LIST) ---
                with ui.column().classes('h-full flex flex-col gap-2 overflow-hidden border-r pr-3'):
                    search_in = ui.input(
                        placeholder='Search name or alias...',
                        on_change=lambda e: self.update_search(e.value)
                    ).classes('w-full text-xs').props('outlined dense clearable')
                    
                    self.left_list_container = ui.scroll_area().classes('w-full flex-1')
                    with self.left_list_container:
                        self.draw_left_list()

                    ui.button(
                        'Add Character', 
                        icon='person_add', 
                        on_click=self.quick_add_new_character
                    ).classes('bg-blue-600 text-white font-bold text-xs w-full py-2 shrink-0')

                # --- RIGHT COLUMN (DETAILS) ---
                self.right_details_container = ui.column().classes('h-full flex-1 overflow-y-auto pr-1 gap-4 min-h-0')
                with self.right_details_container:
                    self.draw_right_details()

        self.dialog.open()

        # Smooth auto-scroll viewport adjustment after client connection settled
        async def initial_scroll():
            await asyncio.sleep(0.2)
            self.scroll_to_active()
        asyncio.create_task(initial_scroll())

    def scroll_to_active(self):
        """Dispatches a micro-task client query to scroll the highlighted card into viewport context."""
        if self.active_row_id and hasattr(self, 'client') and self.client:
            self.client.run_javascript(f"""
                const el = document.getElementById('c{self.active_row_id}');
                if (el) {{
                    el.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                }}
            """)

    def update_search(self, val: str):
        self.search_query = val.strip().lower() if val else ""
        self.draw_left_list.refresh()

    def trigger_workspace_refresh(self):
        if self.on_change_callback:
            self.on_change_callback()

    def get_sorted_events(self, char_id: int) -> List[CharacterTimelineEvent]:
        """Loads and chronologically orders all timeline state events for a character."""
        with Session(engine) as session:
            events = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == char_id)
            ).all()
            
        book_order_map = {b.id: (b.book_order or 0) for b in self.books_list}
        
        def event_sort_key(ev):
            if ev.book_id is None:
                return (-1, -1, -1)  # Base state is always first
            order = book_order_map.get(ev.book_id, 9999)
            return (order, ev.chapter_num or 0, ev.scene_num or 0)
            
        return sorted(events, key=event_sort_key)

    def get_active_event_for_scene(self, char_id: int) -> Optional[CharacterTimelineEvent]:
        """Calculates which timeline event is active at the current workspace scene coordinate."""
        from ui import state
        target_ch = int(float(getattr(state, 'book_active_chapter', 1)))
        target_sc = int(float(getattr(state, 'book_active_scene', 1)))
        
        sorted_evs = self.get_sorted_events(char_id)
        book_order_map = {b.id: (b.book_order or 0) for b in self.books_list}
        target_book_order = book_order_map.get(self.book_id, 0)
        
        active_ev = None
        for ev in sorted_evs:
            if ev.book_id is None:
                active_ev = ev
                continue
            ev_order = book_order_map.get(ev.book_id, 0)
            if ev_order < target_book_order:
                active_ev = ev
            elif ev_order == target_book_order:
                if ev.chapter_num < target_ch:
                    active_ev = ev
                elif ev.chapter_num == target_ch and ev.scene_num <= target_sc:
                    active_ev = ev
        return active_ev

    @ui.refreshable
    def draw_left_list(self):
        with Session(engine) as session:
            chars = session.exec(
                select(Character).where(Character.project_id == self.project_id)
            ).all()
            
        char_data = []
        for c in chars:
            with Session(engine) as session:
                aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == c.id)).all()
            
            c_mentions = sum(self.frequencies.get(a.alias.lower(), 0) for a in aliases)
            if not aliases:
                c_mentions = self.frequencies.get(c.name.lower(), 0)
                
            if self.search_query:
                alias_texts = [a.alias.lower() for a in aliases]
                if self.search_query not in c.name.lower() and not any(self.search_query in t for t in alias_texts):
                    continue
                    
            char_data.append((c, len(aliases), c_mentions))
            
        char_data.sort(key=lambda x: x[2], reverse=True)
        
        valid_ids = [item[0].id for item in char_data]
        if self.selected_char_id not in valid_ids:
            self.selected_char_id = valid_ids[0] if valid_ids else None
            self.selected_event_id = None  # Reset state tracking on selection drift
            self.draw_right_details.refresh()
            
        if not char_data:
            ui.label('No characters found.').classes('text-xs text-slate-400 italic p-4 text-center')
            return
            
        self.active_row_id = None
        for c, alias_count, c_mentions in char_data:
            is_selected = c.id == self.selected_char_id
            bg_class = "bg-blue-50 border-l-4 border-blue-600 font-semibold text-blue-900" if is_selected else "hover:bg-slate-50 text-slate-700"
            
            row_el = ui.row().classes(f'w-full p-2.5 rounded-lg cursor-pointer transition-colors justify-between items-center {bg_class}')
            
            if is_selected:
                self.active_row_id = row_el.id
                
            with row_el:
                with ui.column().classes('gap-0 flex-1 min-w-0'):
                    ui.label(c.name).classes('text-xs font-semibold truncate')
                    ui.label(f"{alias_count} alias{'es' if alias_count != 1 else ''}").classes('text-[9px] text-slate-400')
                ui.badge(f"{c_mentions} hits", color='blue-50').classes('text-blue-700 text-[10px] font-bold shrink-0 px-1.5 py-0.5 rounded')
                
            def select_this(cid=c.id):
                self.selected_char_id = cid
                self.selected_event_id = None  # Allow chronological auto-calculation on change
                self.draw_left_list.refresh()
                self.draw_right_details.refresh()
                
            row_el.on('click', select_this)

    @ui.refreshable
    def draw_right_details(self):
        cid = self.selected_char_id
        if cid is None:
            with ui.column().classes('w-full h-full items-center justify-center text-slate-400 gap-2 py-24'):
                ui.icon('person_search', size='lg', color='slate-300')
                ui.label('No Character Selected').classes('text-xs font-bold text-slate-500')
            return
            
        with Session(engine) as session:
            char = session.get(Character, cid)
            if not char:
                return
            
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
            all_project_chars = session.exec(
                select(Character)
                .where(Character.project_id == self.project_id)
                .where(Character.id != char.id)
            ).all()

        # Load chronologically sorted list of states
        sorted_evs = self.get_sorted_events(char.id)
        
        # Self-heal/Calculate chronological active event state
        if self.selected_event_id is None:
            active_ev = self.get_active_event_for_scene(char.id)
            self.selected_event_id = active_ev.id if active_ev else None
            
        # Select active event reference
        current_ev = None
        for ev in sorted_evs:
            if ev.id == self.selected_event_id:
                current_ev = ev
                break
                
        if not current_ev and sorted_evs:
            current_ev = sorted_evs[0]
            self.selected_event_id = current_ev.id

        with Session(engine) as session:
            c_mentions = sum(self.frequencies.get(a.alias.lower(), 0) for a in aliases)
            if not aliases:
                c_mentions = self.frequencies.get(char.name.lower(), 0)

        with ui.row().classes('w-full justify-between items-center border-b pb-2 flex-shrink-0 gap-2'):
            with ui.row().classes('items-center gap-2 flex-1 min-w-0'):
                async def save_quick_name(e):
                    val = e.sender.value.strip()
                    if not val:
                        return
                    with Session(engine) as session:
                        db_char = session.get(Character, char.id)
                        if db_char:
                            db_char.name = val
                            session.add(db_char)
                            session.commit()
                    save_project_characters_to_json(self.project_id)
                    ui.notify(f"Renamed character to: {val}", type="info")
                    self.draw_left_list.refresh()
                    self.draw_right_details.refresh()
                    self.trigger_workspace_refresh()
                    
                ui.input(value=char.name).classes('text-base font-bold text-slate-800 flex-1').props('dense borderless').on('blur', save_quick_name)
                ui.badge(f"{c_mentions} mentions", color='blue-50').classes('text-blue-700 font-bold text-xs')
            
            with ui.row().classes('items-center gap-1 flex-shrink-0'):
                def toggle_lock_quick():
                    with Session(engine) as session:
                        db_char = session.get(Character, char.id)
                        if db_char:
                            db_char.locked = not db_char.locked
                            session.add(db_char)
                            session.commit()
                    save_project_characters_to_json(self.project_id)
                    self.draw_left_list.refresh()
                    self.draw_right_details.refresh()
                    ui.notify("Lock toggled!", type="info")
                    self.trigger_workspace_refresh()
                    
                lock_icon = "lock" if char.locked else "lock_open"
                lock_color = "text-rose-600 bg-rose-50 hover:bg-rose-100" if char.locked else "text-slate-600 bg-slate-100 hover:bg-slate-200"
                ui.button(icon=lock_icon, on_click=toggle_lock_quick).props('flat dense').classes(f'p-1.5 rounded-lg {lock_color}').tooltip('Toggle Curated Edit Lock')

                def delete_quick_char():
                    with Session(engine) as session:
                        db_char = session.get(Character, char.id)
                        if db_char:
                            for a in session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all():
                                session.delete(a)
                            for ev in session.exec(select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id == char.id)).all():
                                session.delete(ev)
                            session.delete(db_char)
                            session.commit()
                    save_project_characters_to_json(self.project_id)
                    self.selected_char_id = None
                    self.selected_event_id = None
                    ui.notify("Character profile deleted.", type="warning")
                    self.draw_left_list.refresh()
                    self.draw_right_details.refresh()
                    self.trigger_workspace_refresh()
                    
                ui.button(icon='delete', on_click=delete_quick_char, color='red').props('unelevated dense').classes('p-1.5 rounded-lg text-white').tooltip('Delete Character Profile')

        # --- CHRONOLOGICAL TIMELINE STATE SELECTOR BAR ---
        with ui.row().classes('w-full items-center justify-between bg-purple-50 border border-purple-100 p-2.5 rounded-lg flex-shrink-0 gap-2'):
            def get_ev_label(ev):
                if ev.book_id is None:
                    return "Base Character State"
                b_name = next((b.name for b in self.books_list if b.id == ev.book_id), "Unknown Book")
                return f"Ch {ev.chapter_num}, Sc {ev.scene_num} State Overide ({ev.label or 'State'})"
                
            opts = {ev.id: get_ev_label(ev) for ev in sorted_evs}
            
            def on_state_change(e):
                self.selected_event_id = e.value
                self.draw_right_details.refresh()
                
            ui.select(
                options=opts, 
                value=self.selected_event_id, 
                on_change=on_state_change,
                label="Timeline Active State"
            ).classes('flex-1 bg-white').props('dense outlined clearable=false')
            
            ui.button(
                icon='add_circle',
                on_click=lambda: self.open_add_timeline_dialog(char.id)
            ).classes('bg-purple-600 text-white font-bold text-xs p-2 rounded-lg').tooltip('Add Timeline Override State')

        with ui.row().classes('w-full items-center gap-2 flex-shrink-0'):
            async def run_research_quick(speculate: bool = False):
                if char.locked:
                    ui.notify("Character is locked from being profiled.", type="warning")
                    return
                self.is_profiling = True
                self.draw_right_details.refresh()
                try:
                    ui.notify(f"Running LLM {'speculation' if speculate else 'factual research'}...", type="info")
                    await run_stateful_character_profiling(
                        self.project_id, char.id, book_id=self.book_id, 
                        max_chunks_to_scan=5, speculate=speculate
                    )
                    ui.notify("LLM complete!", type="positive")
                except Exception as e:
                    ui.notify(f"Research failed: {e}", type="negative")
                self.is_profiling = False
                self.draw_left_list.refresh()
                self.draw_right_details.refresh()
                self.trigger_workspace_refresh()

            if self.is_profiling:
                with ui.row().classes('items-center gap-2 bg-purple-50 px-3 py-1.5 rounded-lg border border-purple-200'):
                    ui.spinner(size='xs', color='purple')
                    ui.label('LLM Active...').classes('text-xs text-purple-700 font-bold')
            else:
                ui.button('Research (LLM)', icon='science', on_click=lambda: run_research_quick(False)).classes('text-white font-bold text-xs bg-purple-600 hover:bg-purple-700')
                ui.button('Deduce Vibe', icon='theater_comedy', on_click=lambda: run_research_quick(True)).classes('text-white font-bold text-xs bg-indigo-600 hover:bg-indigo-700')

        # --- PHYSICAL TRAITS INPUT GRID (BOUND TO SELECTED EVENT) ---
        with ui.grid().classes('w-full grid-cols-1 md:grid-cols-2 gap-2 mt-1'):
            trait_fields = [
                ("demographics", "Demographics (Age, Race, Gender)"),
                ("hair_and_face", "Hair & Face Details"),
                ("physical_build", "Physical Build"),
                ("distinguishing_marks", "Distinguishing Marks & Accessories")
            ]
            
            def make_quick_trait_handler(ev_id, key):
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
                                compiled_desc_area.set_value(new_prompt)
                            session.add(db_ev)
                            session.commit()
                    save_project_characters_to_json(self.project_id)
                    ui.notify("Trait saved.", type="positive", position="bottom-right", timeout=1000)
                    self.trigger_workspace_refresh()
                return handler

            for key, label in trait_fields:
                val = getattr(current_ev, key) or ""
                ui.input(label=label, value=val).classes('w-full bg-white').props('outlined dense').on('blur', make_quick_trait_handler(current_ev.id, key))

        ui.label('Compiled Visual Prompt').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wide mt-1')
        
        def handle_quick_desc_blur(e, ev_id=current_ev.id):
            new_val = e.sender.value.strip()
            with Session(engine) as session:
                db_ev = session.get(CharacterTimelineEvent, ev_id)
                if db_ev:
                    db_ev.visual_description = new_val if new_val else None
                    session.add(db_ev)
                    session.commit()
            save_project_characters_to_json(self.project_id)
            ui.notify("Visual Prompt overridden!", type="info")
            self.trigger_workspace_refresh()

        compiled_desc_area = ui.textarea(value=current_ev.visual_description or "").classes('w-full bg-white font-mono text-xs').props('outlined dense autogrow').on('blur', handle_quick_desc_blur)

        with ui.column().classes('w-full bg-slate-50 p-3 rounded-lg border gap-2 mt-1'):
            ui.label('Assigned Aliases & Target Tags').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wider')
            
            def remove_alias_and_spin_off(alias_obj):
                with Session(engine) as session:
                    db_alias = session.get(CharacterAlias, alias_obj.id)
                    if db_alias:
                        alias_name = db_alias.alias
                        session.delete(db_alias)
                        session.commit()
                        
                        if alias_name.lower() != char.name.lower():
                            new_char = Character(project_id=self.project_id, name=alias_name)
                            session.add(new_char)
                            session.commit()
                            
                            b_ev = CharacterTimelineEvent(character_id=new_char.id, book_id=None, chapter_num=0, scene_num=0, label="Base State")
                            session.add(b_ev)
                            session.commit()
                            
                            new_alias = CharacterAlias(character_id=new_char.id, alias=alias_name)
                            session.add(new_alias)
                            session.commit()
                            
                            b_ev.visual_description = compile_character_visual_prompt(b_ev)
                            session.add(b_ev)
                            session.commit()
                            
                            ui.notify(f"Spun off '{alias_name}' into standalone profile!", type="positive")
                
                save_project_characters_to_json(self.project_id)
                self.draw_left_list.refresh()
                self.draw_right_details.refresh()
                self.trigger_workspace_refresh()

            with ui.row().classes('w-full gap-2 flex-wrap items-center'):
                for a in aliases:
                    with ui.row().classes('items-center gap-1 bg-white border border-slate-200 px-2 py-0.5 rounded-full text-xs text-slate-800'):
                        alias_el = ui.label(a.alias).classes('font-medium px-1 cursor-pointer hover:text-blue-600 hover:underline')
                        alias_el.on('click', lambda _, x=a: self.open_alias_explorer_dialog(x, char.id))
                        
                        ui.icon('cancel', size='14px', color='slate-400').classes('cursor-pointer hover:text-red-500 transition-colors').on('click', lambda _, x=a: remove_alias_and_spin_off(x))

            if all_project_chars:
                with ui.row().classes('w-full items-center gap-2 mt-1'):
                    merge_opts = {c.id: c.name for c in all_project_chars}
                    merge_select = ui.select(options=merge_opts, label='Merge another character into this...', with_input=True).classes('flex-1 bg-white').props('dense outlined clearable')
                    
                    async def handle_merge_quick_click():
                        src_id = merge_select.value
                        if not src_id:
                            ui.notify("Select a character to merge.", type="warning")
                            return
                        with Session(engine) as session:
                            source_aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == src_id)).all()
                            alias_ids = [a.id for a in source_aliases]
                        if not alias_ids:
                            ui.notify("No aliases found on target character to merge.", type="warning")
                            return
                        await asyncio.to_thread(merge_character_aliases, self.project_id, char.id, alias_ids)
                        
                        with Session(engine) as session:
                            src_char = session.get(Character, src_id)
                            if src_char:
                                session.delete(src_char)
                                session.commit()
                        
                        save_project_characters_to_json(self.project_id)
                        ui.notify("Characters successfully merged!", type="positive")
                        self.selected_char_id = char.id
                        self.selected_event_id = None
                        self.draw_left_list.refresh()
                        self.draw_right_details.refresh()
                        self.trigger_workspace_refresh()
                        
                    ui.button('Merge', icon='call_merge', on_click=handle_merge_quick_click).classes('bg-blue-600 text-white font-bold text-xs px-3 py-1.5 rounded-lg')

    def open_alias_explorer_dialog(self, alias: CharacterAlias, parent_char_id: int):
        occurrences = get_alias_occurrences(self.project_id, alias.alias)
        current_index = 0

        with ui.dialog() as dialog, ui.card().classes('w-[600px] max-w-[95vw] p-6 rounded-xl flex flex-col gap-4 overflow-hidden'):
            
            with ui.row().classes('w-full justify-between items-center border-b pb-3 shrink-0'):
                with ui.column().classes('gap-0.5'):
                    ui.label(f'Context Explorer: "{alias.alias}"').classes('text-base font-bold text-slate-800')
                    book_label = ui.label('Loading context...').classes('text-xs text-slate-500')
                ui.button(icon='close', on_click=dialog.close).props('flat round dense').classes('text-slate-400')

            with ui.column().classes('w-full flex-1 justify-center items-center py-6 min-h-[160px] bg-slate-50 border rounded-lg px-4 overflow-y-auto'):
                context_html = ui.html('').classes('text-sm text-slate-700 leading-relaxed text-center')

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
                        
                        save_project_characters_to_json(self.project_id)
                        ui.notify(f"Promoted '{new_name}' to canonical character name!", type="positive")
                        dialog.close()
                        
                        self.draw_left_list.refresh()
                        self.draw_right_details.refresh()
                        self.trigger_workspace_refresh()

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

    def open_add_timeline_dialog(self, character_id: int):
        """Spawns an inline dialogue to append coordinate-based timeline transitions to the active cast member."""
        from ui import state
        default_ch = int(float(getattr(state, 'book_active_chapter', 1)))
        default_sc = int(float(getattr(state, 'book_active_scene', 1)))

        with ui.dialog() as dialog, ui.card().classes('w-[450px] p-5 rounded-xl flex flex-col gap-3'):
            ui.label('Add Timeline Override Event').classes('text-sm font-bold text-slate-800')
            
            book_opts = {b.id: b.name for b in self.books_list}
            if not book_opts:
                ui.label('No books imported in this project yet.').classes('text-xs text-red-500')
                ui.button('Close', on_click=dialog.close).props('flat')
                dialog.open()
                return
                
            book_select = ui.select(options=book_opts, value=self.book_id, label="Target Book").classes('w-full bg-white').props('dense outlined')
            chapter_input = ui.number(label="Chapter Number", value=default_ch, min=1, step=1).classes('w-full').props('outlined dense')
            scene_input = ui.number(label="Scene Number", value=default_sc, min=1, step=1).classes('w-full').props('outlined dense')
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
                    
                    book_order_map = {b.id: (b.book_order or 0) for b in self.books_list}
                    
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
                        copied_status.text = f"Cloned traits from existing: '{resolved.label or 'Base State'}'!"
                    else:
                        copied_status.text = "No previous chronological state to clone."

            ui.button('Clone Previous State Traits', icon='content_copy', on_click=copy_previous).classes('text-xs text-slate-700 bg-slate-100 hover:bg-slate-200 border w-full')
            
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
                    
                    # Redirect active editing selector target to this new event state
                    self.selected_event_id = new_ev.id
                    
                save_project_characters_to_json(self.project_id)
                ui.notify("Timeline override event added!", type="positive")
                dialog.close()
                self.draw_right_details.refresh()
                self.trigger_workspace_refresh()

            with ui.row().classes('w-full justify-end gap-2 border-t pt-3 mt-2'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                ui.button('Create Event', on_click=save_new_event).classes('bg-blue-600 text-white font-bold text-xs px-3 py-1.5 rounded-lg')

        dialog.open()
        # Prepopulate cloning on window mount
        copy_previous()

    def quick_add_new_character(self):
        with ui.dialog() as add_dialog, ui.card().classes('w-[350px] p-4 rounded-xl flex flex-col gap-3'):
            ui.label('Add New Character').classes('text-sm font-bold text-slate-800')
            name_in = ui.input('Character Name', placeholder='e.g. Stone Barrington').classes('w-full').props('outlined dense')
            
            def do_add():
                name = name_in.value.strip()
                if not name:
                    return
                with Session(engine) as session:
                    new_char = Character(project_id=self.project_id, name=name)
                    session.add(new_char)
                    session.commit()
                    
                    base_ev = CharacterTimelineEvent(character_id=new_char.id, book_id=None, chapter_num=0, scene_num=0, label="Base State")
                    session.add(base_ev)
                    session.commit()
                    
                    new_alias = CharacterAlias(character_id=new_char.id, alias=name)
                    session.add(new_alias)
                    session.commit()
                    
                save_project_characters_to_json(self.project_id)
                ui.notify(f"Added character '{name}'!", type="positive")
                add_dialog.close()
                self.selected_char_id = new_char.id
                self.selected_event_id = None
                self.draw_left_list.refresh()
                self.draw_right_details.refresh()
                self.trigger_workspace_refresh()
                    
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Cancel', on_click=add_dialog.close).props('flat').classes('text-xs')
                ui.button('Add', on_click=do_add).classes('bg-blue-600 text-white text-xs font-semibold px-3 py-1.5 rounded-lg')
        add_dialog.open()