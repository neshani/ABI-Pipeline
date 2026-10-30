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
    is_placeholder_description,
    save_project_characters_to_json,
    get_alias_occurrences,
    resolve_event_at,
    get_pinned_fields,
    set_pinned_fields,
    character_field_stats,
    propagate_trait_change,
    NEW_TRAIT_FIELDS,
)
from ui.pages.project.characters_tab import get_character_frequency_map, event_trait_count
from ui.components.description_lookup_modal import open_lookup_dialog
from ui.components.trait_grid import render_trait_grid, render_legacy_fields

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


def retime_timeline_event(event_id: int, book_id: int, chapter_num: int,
                          shot_num: int, label: str) -> Optional[str]:
    """Move one change to a new (book, chapter, shot) and/or relabel it.

    Returns None when the row was written, otherwise the reason it was not:
    the caller shows that message and leaves the form open.
    """
    with Session(engine) as session:
        ev = session.get(CharacterTimelineEvent, event_id)
        if ev is None:
            return "This change no longer exists."
        if ev.book_id is None:
            return "The Base state has no position on the timeline."
        dup = session.exec(
            select(CharacterTimelineEvent)
            .where(CharacterTimelineEvent.character_id == ev.character_id)
            .where(CharacterTimelineEvent.book_id == book_id)
            .where(CharacterTimelineEvent.chapter_num == chapter_num)
            .where(CharacterTimelineEvent.shot_num == shot_num)
            .where(CharacterTimelineEvent.id != event_id)
        ).first()
        if dup:
            return "Another change already sits at that coordinate."
        ev.book_id = book_id
        ev.chapter_num = chapter_num
        ev.shot_num = shot_num
        ev.label = label
        if ev.source == "extraction":
            # Re-timing is a manual decision. Left as extraction, a re-analysis
            # clears the row and the change snaps back to where it came from.
            ev.source = "manual"
        session.add(ev)
        session.commit()
    return None


def delete_timeline_event(event_id: int) -> bool:
    """Remove one change. The Base state is a snapshot and is not deletable."""
    with Session(engine) as session:
        ev = session.get(CharacterTimelineEvent, event_id)
        if ev is None or ev.book_id is None:
            return False
        session.delete(ev)
        session.commit()
    return True


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
            return (order, ev.chapter_num or 0, ev.shot_num or 0)
            
        return sorted(events, key=event_sort_key)

    def get_active_event_for_scene(self, char_id: int) -> Optional[CharacterTimelineEvent]:
        """The state in force at the current workspace scene.

        ``resolve_event_at`` is the resolver the renderer itself uses, so the panel
        cannot show a different state than the one that renders: the previous
        inline walk ignored side stories and pins, and picked its own "latest".
        """
        from ui import state
        target_ch = int(float(getattr(state, 'book_active_chapter', 1)))
        target_sc = int(float(getattr(state, 'book_active_scene', 1)))

        with Session(engine) as session:
            char = session.get(Character, char_id)
            if not char:
                return None
            events = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == char_id)
            ).all()
            effective, _base = resolve_event_at(
                session, char, self.book_id, target_ch, target_sc, events=events)
        return effective

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
                return f"Ch {ev.chapter_num}, Sc {ev.shot_num} State Overide ({ev.label or 'State'})"
                
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

            # Re-time or delete the selected change. The Base state is a
            # snapshot, not a point on the axis, so it has nothing to edit.
            if current_ev is not None and current_ev.book_id is not None:
                ui.button(
                    icon='edit',
                    on_click=lambda _, eid=current_ev.id: self.open_edit_timeline_dialog(eid)
                ).props('flat dense').classes(
                    'p-2 text-purple-700 bg-purple-50 hover:bg-purple-100 rounded-lg')\
                 .tooltip('Edit when this change happens, or delete it')
            else:
                ui.button(icon='edit').props('flat dense disable').classes(
                    'p-2 text-slate-300 bg-slate-100 rounded-lg')\
                 .tooltip('The Base state is not on the timeline; select a change to edit it')
            
            ui.button(
                icon='add_circle',
                on_click=lambda: self.open_add_timeline_dialog(char.id)
            ).classes('bg-purple-600 text-white font-bold text-xs p-2 rounded-lg').tooltip('Add Timeline Override State')

        with ui.row().classes('w-full items-center gap-2 flex-shrink-0'):
            def open_find_dialog(source: str):
                """Open the shared lookup dialog, then rebuild this panel on close.

                The work happens in the dialog, not in a handler inside this grid.
                That is deliberate: refreshing the grid from a handler *inside* it
                deletes the element the handler is running on, which is why these
                two buttons used to leave the form frozen until the modal was
                reopened. The dialog refreshes the host once, when it closes.
                """
                if char.locked:
                    ui.notify("Character is locked from being profiled.", type="warning")
                    return
                open_lookup_dialog(self.project, char,
                                   refresh_callback=refresh_after_lookup,
                                   initial_source=source)

            def refresh_after_lookup():
                self.draw_left_list.refresh()
                self.draw_right_details.refresh()
                self.trigger_workspace_refresh()

            ui.button('Research (LLM)', icon='science',
                      on_click=lambda: open_find_dialog("book")) \
                .classes('text-white font-bold text-xs bg-purple-600 hover:bg-purple-700') \
                .tooltip('Read the book text and keep only what it states. Nothing '
                         'is invented and nothing you have set is replaced.')
            ui.button('Deduce Vibe', icon='theater_comedy',
                      on_click=lambda: open_find_dialog("deduce")) \
                .classes('text-white font-bold text-xs bg-indigo-600 hover:bg-indigo-700') \
                .tooltip('Research first, then give the empty traits a look that '
                         'fits their role, age and bearing. Values it invents are '
                         'noted as not coming from the text.')

        # --- PHYSICAL TRAITS INPUT GRID (BOUND TO SELECTED EVENT) ---
        # These are the fields the renderer compiles, at this scene's position. The
        # legacy quartet sits below them, collapsed: a project profiled by an older
        # build has only those four, and the Characters tab keeps them for exactly
        # that reason.
        pinned_fields = set(get_pinned_fields(char))
        editing_base = current_ev.book_id is None
        display_ev = current_ev
        if pinned_fields and not editing_base:
            # Pinned traits render from the Base value, so show what actually
            # renders here while the inputs keep writing to the row being edited.
            with Session(engine) as session:
                resolved, _base = resolve_event_at(
                    session, char, current_ev.book_id,
                    current_ev.chapter_num, current_ev.shot_num)
            if resolved is not None:
                display_ev = resolved

        # The compiled box shows what the renderer would use, but an event with no
        # override has to stay dynamic: a blur that changes nothing must not freeze
        # the computed text into ``visual_description``.
        shown_desc = (display_ev.visual_description or "").strip()
        # The no-trait placeholder was stamped automatically; show the compiled
        # fields instead so the box reflects what the renderer will use.
        if is_placeholder_description(shown_desc):
            shown_desc = ""
        desc_state = {"shown": shown_desc, "is_override": bool(shown_desc)}
        if not shown_desc and event_trait_count(display_ev):
            desc_state["shown"] = compile_character_visual_prompt(
                display_ev, name=char.name)

        def save_trait(key, val):
            """Write one trait to the row being edited.

            ``val`` arrives already collapsed to a single line by trait_grid, so
            an empty string is the only "no value" case. The rules themselves are
            unchanged: character-level for sex/species, "from here on" for the
            rest, and a locked profile keeps its compiled prompt.
            """
            with Session(engine) as session:
                db_ev = session.get(CharacterTimelineEvent, current_ev.id)
                if not db_ev:
                    return
                old_val = getattr(db_ev, key, None)
                setattr(db_ev, key, val if val != "" else None)
                if db_ev.source == "extraction":
                    db_ev.source = "manual"  # protect the edit from re-analysis
                char_obj = session.get(Character, db_ev.character_id)
                if key in ("sex", "species") and val:
                    # Character-level, not scene-level: one edit applies to every
                    # state so the user does not have to touch 20+ rows.
                    for other in session.exec(
                            select(CharacterTimelineEvent)
                            .where(CharacterTimelineEvent.character_id
                                   == db_ev.character_id)).all():
                        setattr(other, key, val)
                        if other.book_id is None:
                            other.source = "manual"
                        session.add(other)
                elif char_obj:
                    # Scene-level, but the user means "from here on": carry the
                    # edit through the states that still repeat the old value and
                    # stop where the book changes it. A legacy field is a no-op.
                    propagate_trait_change(
                        session, char_obj, key, val, old_val,
                        book_id=db_ev.book_id, chapter_num=db_ev.chapter_num,
                        shot_num=db_ev.shot_num)
                if char_obj and not char_obj.locked:
                    if event_trait_count(db_ev):
                        new_prompt = compile_character_visual_prompt(
                            db_ev, name=char_obj.name)
                        # Only refresh an override that already exists; an event
                        # with none stays dynamic, so pins and the carried scope
                        # still reach the render.
                        if db_ev.visual_description:
                            db_ev.visual_description = new_prompt
                        compiled_desc_area.set_value(new_prompt)
                        desc_state["shown"] = new_prompt
                    elif db_ev.visual_description:
                        db_ev.visual_description = None
                        compiled_desc_area.set_value("")
                        desc_state["shown"] = ""
                        desc_state["is_override"] = False
                session.add(db_ev)
                session.commit()
            save_project_characters_to_json(self.project_id)
            ui.notify("Trait saved.", type="positive", position="bottom-right", timeout=1000)
            self.trigger_workspace_refresh()
        # Per-field provenance for the tooltips. Whole-project spans, because the
        # value in force here may have been set volumes earlier; the Characters tab
        # owns the per-book, axis-shaped view of the same data.
        with Session(engine) as session:
            order_map = {b.id: (b.book_order or 0) for b in self.books_list}
            field_stats = character_field_stats(session, char, book_id=None,
                                                order_map=order_map)
        book_names = {b.id: b.name for b in self.books_list}

        def span_line(span):
            """One recorded value and where it was in force."""
            if span["book_id"] is None:
                # The Base precedes every volume, so a span that starts there and
                # merges into the first book is "unchanged through" that chapter -
                # there is no book name to show, and "Base-Ch 48" read as a range
                # on the Base itself.
                if not span["end_chapter"]:
                    return f"{span['value']}  (Base)"
                return (f"{span['value']}  (Base, unchanged through "
                        f"Ch {span['end_chapter']})")
            where = f"{book_names.get(span['book_id'])} · Ch {span['start_chapter']}"
            if span["end_chapter"] != span["start_chapter"]:
                where += f"–{span['end_chapter']}"
            return f"{span['value']}  ({where})"

        def field_tooltip(info, here):
            """Every value this trait has held, in reading order."""
            spans = info.get("spans", [])
            if not spans:
                return "No value anywhere in this project."
            lines = [span_line(span) for span in spans[:12]]
            if len(spans) > 12:
                lines.append(f"…and {len(spans) - 12} more")
            if info.get("pinned"):
                lines.insert(0, "Pinned: the Base value always renders here. "
                                "The changes below are the extraction's, ignored.")
            elif not here:
                # The field adds nothing to this render even though the project
                # knows a value - the commonest reason an image looks wrong.
                lines.insert(0, "Not set at this scene; these values are elsewhere.")
            return "\n".join(lines)

        def field_badge(key, info, here):
            """(text, colour) for the change count beside a field's label."""
            if info.get("pinned"):
                return "", ""  # the pin button already says so
            if not here:
                return "missing", "text-slate-400"
            within = info.get("within_book_changes", 0)
            across = info.get("across_book_changes", 0)
            if within:
                text = f"{within} change" + ("s" if within != 1 else "")
                if across:
                    text += f" · {across} across"
                colour = ("text-red-600" if key in ("sex", "species")
                          else "text-amber-600")
                return text, colour
            if across:
                return f"{across} across books", "text-blue-600"
            return "stable", "text-green-600"

        def toggle_pin(key):
            """Pin a trait to the Base value, or release it.

            Same rule as the Characters tab: a pin makes the Base value the one
            that renders. A Base with no value for this field takes the value in
            force here, which is what a user looking at a scene means by pinning.
            """
            message = ""
            with Session(engine) as session:
                db_char = session.get(Character, char.id)
                if not db_char:
                    return
                pins = set(get_pinned_fields(db_char))
                if key in pins:
                    pins.discard(key)
                    message = f"Unpinned '{key.replace('_', ' ')}'; the book's changes apply again."
                else:
                    base = session.exec(
                        select(CharacterTimelineEvent)
                        .where(CharacterTimelineEvent.character_id == char.id)
                        .where(CharacterTimelineEvent.book_id == None)  # noqa: E711
                    ).first()
                    if base is None:
                        base = CharacterTimelineEvent(
                            character_id=char.id, book_id=None, chapter_num=0,
                            shot_num=0, label="Base State")
                        session.add(base)
                    base_value = (getattr(base, key, None) or "").strip()
                    value = base_value
                    from_here = False
                    if not value:
                        value = (getattr(current_ev, key, None) or "").strip()
                        if value:
                            setattr(base, key, value)
                            from_here = True
                    if not value:
                        ui.notify("Nothing to pin for this trait.", type="warning",
                                  position="bottom-right")
                        return
                    if base.source != "manual":
                        base.source = "manual"  # a pin is a manual canonical decision
                    session.add(base)
                    pins.add(key)
                    message = (f"Pinned '{value}' (taken from this scene)." if from_here
                               else f"Pinned to the Base value: '{base_value}'.")
                set_pinned_fields(session, db_char, sorted(pins))
                session.commit()
            save_project_characters_to_json(self.project_id)
            ui.notify(message, type="info", position="bottom-right", timeout=1800)
            self.draw_right_details.refresh()
            self.trigger_workspace_refresh()

        def trait_status(key):
            """The badge beside a trait's label, or None to draw none."""
            info = field_stats.get(key, {})
            here = bool((getattr(display_ev, key, None) or "").strip())
            text, colour = field_badge(key, info, here)
            return (text, colour) if text else None

        def trait_tooltip(key):
            return field_tooltip(field_stats.get(key, {}),
                                 getattr(display_ev, key, None) or "")

        def readonly_reason(key):
            # The value shown is the Base's, so the input would write to a row
            # that is not the one supplying the text. Editing happens on Base.
            if key in pinned_fields and not editing_base:
                return 'Pinned to Base; select Base to edit the canonical value.'
            return None

        render_trait_grid(
            value_for=lambda key: getattr(display_ev, key, None) or "",
            save=save_trait,
            status_for=trait_status,
            tooltip_for=trait_tooltip,
            pinned_for=lambda key: key in pinned_fields,
            pin_for=toggle_pin,
            readonly_for=readonly_reason,
        )
        render_legacy_fields(
            value_for=lambda key: getattr(current_ev, key, None) or "",
            save=save_trait,
        )

        ui.label('Compiled Visual Prompt').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wide mt-1')

        def handle_quick_desc_blur(e, ev_id=current_ev.id):
            new_val = e.sender.value.strip()
            if not desc_state["is_override"] and new_val == desc_state["shown"].strip():
                return
            with Session(engine) as session:
                db_ev = session.get(CharacterTimelineEvent, ev_id)
                if db_ev:
                    db_ev.visual_description = new_val if new_val else None
                    if db_ev.source == "extraction" and new_val:
                        db_ev.source = "manual"
                    session.add(db_ev)
                    session.commit()
            desc_state["shown"] = new_val
            desc_state["is_override"] = bool(new_val)
            save_project_characters_to_json(self.project_id)
            ui.notify("Visual Prompt saved." if new_val
                      else "Visual Prompt cleared; the fields compile it.", type="info")
            self.trigger_workspace_refresh()

        compiled_desc_area = ui.textarea(value=desc_state["shown"])\
            .classes('w-full bg-white font-mono text-xs').props('outlined dense autogrow')\
            .on('blur', handle_quick_desc_blur)\
            .tooltip('Injected into the render prompt. Clear it to let the fields compile it.')

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
                            
                            b_ev = CharacterTimelineEvent(character_id=new_char.id, book_id=None, chapter_num=0, shot_num=0, label="Base State")
                            session.add(b_ev)
                            session.commit()
                            
                            new_alias = CharacterAlias(character_id=new_char.id, alias=alias_name)
                            session.add(new_alias)
                            session.commit()
                            
                            initial_desc = compile_character_visual_prompt(
                                b_ev, name=alias_name)
                            if not is_placeholder_description(initial_desc):
                                b_ev.visual_description = initial_desc
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

    def open_edit_timeline_dialog(self, event_id: int):
        """Edit where an existing change happens, or delete it.

        The add dialog only ever creates. Before this, a change that landed on
        the wrong shot could only be moved by duplicating the state at the right
        position and then finding a way to remove the old one.
        """
        with Session(engine) as session:
            ev = session.get(CharacterTimelineEvent, event_id)
            if ev is None or ev.book_id is None:
                ui.notify("The Base state has no position on the timeline.",
                          type="warning")
                return
            start_book = ev.book_id
            start_chapter = int(ev.chapter_num or 0)
            start_shot = int(ev.shot_num or 0)
            start_label = (ev.label or "").strip()

        with ui.dialog() as dialog, ui.card().classes('w-[450px] p-5 rounded-xl flex flex-col gap-3'):
            ui.label('Timing of This Change').classes('text-sm font-bold text-slate-800')
            ui.label(
                'Move the change to the chapter and shot where it actually starts. '
                'The traits stay with the state; nothing is copied.'
            ).classes('text-[10px] text-slate-500 leading-tight')

            book_opts = {b.id: b.name for b in self.books_list}
            book_select = ui.select(options=book_opts, value=start_book, label="Book")\
                .classes('w-full bg-white').props('dense outlined')
            # min=0: most change rows sit on shot 0, and an edit form must not
            # silently clamp a coordinate the user never touched.
            chapter_input = ui.number(label="Chapter Number", value=start_chapter,
                                      min=0, step=1).classes('w-full').props('outlined dense')
            scene_input = ui.number(label="Scene Number", value=start_shot,
                                    min=0, step=1).classes('w-full').props('outlined dense')
            label_input = ui.input(label="Event Label", value=start_label,
                                   placeholder="e.g. Gandalf the White")\
                .classes('w-full').props('outlined dense')

            def save_timing():
                b_id = book_select.value
                if b_id is None or chapter_input.value is None or scene_input.value is None:
                    ui.notify("Choose a book and give both coordinates.",
                              type="warning")
                    return
                ch = int(float(chapter_input.value))
                sc = int(float(scene_input.value))
                typed_label = (label_input.value or "").strip()
                moved = (b_id != start_book or ch != start_chapter
                         or sc != start_shot or typed_label != start_label)
                if not moved:
                    dialog.close()
                    return
                error = retime_timeline_event(event_id, b_id, ch, sc,
                                              typed_label or "Override State")
                if error:
                    ui.notify(error, type="warning")
                    return
                save_project_characters_to_json(self.project_id)
                ui.notify("Change timing updated.", type="positive")
                dialog.close()
                self.draw_right_details.refresh()
                self.trigger_workspace_refresh()

            def delete_change():
                if not delete_timeline_event(event_id):
                    ui.notify("This change no longer exists.", type="warning")
                else:
                    save_project_characters_to_json(self.project_id)
                    # Let the panel self-heal to the state in force at the scene
                    # the workspace is showing, rather than a row that is gone.
                    self.selected_event_id = None
                    ui.notify("Change deleted.", type="warning")
                dialog.close()
                self.draw_right_details.refresh()
                self.trigger_workspace_refresh()

            with ui.row().classes('w-full justify-between items-center border-t pt-3 mt-2 gap-2'):
                ui.button('Delete Change', icon='delete', on_click=lambda: confirm_dialog.open())\
                    .props('flat dense').classes('text-red-600 text-xs')
                with ui.row().classes('gap-2'):
                    ui.button('Cancel', on_click=dialog.close).props('flat').classes('text-xs')
                    ui.button('Save Timing', on_click=save_timing)\
                        .classes('bg-blue-600 text-white font-bold text-xs px-3 py-1.5 rounded-lg')

        # Built after the editor, opened from inside it. Deleting a change has no
        # undo, so it takes a second click that names the change being removed.
        with ui.dialog() as confirm_dialog, ui.card().classes('w-[380px] p-5 rounded-xl flex flex-col gap-3'):
            ui.label('Delete this change?').classes('text-sm font-bold text-slate-800')
            ui.label(
                f"Ch {start_chapter} · {start_shot} ({start_label or 'Override State'}). "
                "The state is removed for this character and cannot be undone."
            ).classes('text-xs text-slate-500 leading-tight')
            with ui.row().classes('w-full justify-end gap-2 border-t pt-3 mt-1'):
                ui.button('Keep It', on_click=confirm_dialog.close).props('flat').classes('text-xs')
                ui.button('Delete', on_click=delete_change)\
                    .classes('bg-red-600 text-white font-bold text-xs px-3 py-1.5 rounded-lg')

        dialog.open()

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
                    
                    base_ev = CharacterTimelineEvent(character_id=new_char.id, book_id=None, chapter_num=0, shot_num=0, label="Base State")
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