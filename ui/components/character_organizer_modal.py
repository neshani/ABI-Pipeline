from typing import List, Callable, Dict, Set, Any, Optional
from nicegui import ui
from sqlmodel import Session, select
from database.connection import engine
from database.models import Project, Book, Character, CharacterBookLink

from services.character_organizer import (
    parse_goodreads_dump,
    find_matching_project_book,
    update_book_metadata,
    commit_seeded_characters,
    extract_characters_from_prompts,
    recalculate_project_character_hits,
    prune_unused_seeded_characters,
    # New Zap Engine additions
    get_unresolved_singletons,
    get_master_profiles,
    promote_singleton_to_master,
    zap_singleton_into_master,
    remove_alias_from_master_by_text,
    is_loose_match
)

class CharacterOrganizerModal(ui.dialog):
    def __init__(self, project: Project, books: List[Book], refresh_callback: Callable[[], None]):
        super().__init__()
        self.project = project
        self.books = books
        self.refresh_callback = refresh_callback
        
        # Staged characters dict: { character_name: {"book_ids": set(), "count": int} }
        self.staged_characters: Dict[str, Dict[str, Any]] = {}
        # Track metadata parsed for books: { book_id: {"title": str, "author": str} }
        self.staged_book_metadata: Dict[int, Dict[str, str]] = {}
        
        # Zap Active Targets State
        self.active_master_id: Optional[int] = None
        self.magnet_filter_enabled = True
        self.hide_populated_enabled = False
        self.fuzz_threshold = 0.60
        
        # Programmatic local chip containers reference
        self.active_chips_container = None
        
        self.current_view = 'gr'
        
        with self, ui.card().classes('w-[980px] max-w-[98vw] h-[820px] max-h-[94vh] p-6 rounded-xl flex flex-col overflow-hidden'):
            # Header Block
            with ui.row().classes('w-full justify-between items-center border-b pb-3 shrink-0'):
                with ui.column().classes('gap-0.5'):
                    ui.label('Smart Character Organizer').classes('text-base font-bold text-slate-800')
                    ui.label('Seed, scan, resolve, and consolidate character entities systematically.').classes('text-xs text-slate-500')
                ui.button(icon='close', on_click=self.close).props('flat dense').classes('text-slate-400')

            # Main Body Layout
            with ui.row().classes('w-full flex-1 min-h-0 gap-4 mt-4 items-stretch flex-nowrap'):
                # Left Navigation Bar
                with ui.column().classes('w-52 shrink-0 border-r pr-4 gap-2'):
                    self.tab_gr = ui.button(
                        '1. Goodreads Seeds', 
                        icon='list_alt', 
                        on_click=lambda: self.switch_view('gr')
                    ).props('flat align=left').classes('w-full text-xs font-semibold justify-start text-slate-700 bg-slate-50')
                    
                    self.tab_scan = ui.button(
                        '2. Scan Tags', 
                        icon='tag', 
                        on_click=lambda: self.switch_view('scan')
                    ).props('flat align=left').classes('w-full text-xs font-semibold justify-start text-slate-700')
                    
                    self.tab_merge = ui.button(
                        '3. Resolve Merges', 
                        icon='call_merge', 
                        on_click=lambda: self.switch_view('merge')
                    ).props('flat align=left').classes('w-full text-xs font-semibold justify-start text-slate-700')
                    
                    ui.space()
                    ui.button(
                        'Reset Steps', 
                        icon='refresh', 
                        on_click=self.reset_steps
                    ).classes('w-full text-xs text-red-500 border hover:bg-red-50 mt-auto')

                # Right Workspace Container
                with ui.column().classes('flex-1 min-h-0 overflow-y-auto gap-4') as self.workspace:
                    self.draw_view()

            # Footer Actions
            with ui.row().classes('w-full justify-end gap-3 border-t pt-3 shrink-0 mt-2'):
                ui.button('Close Organizer', on_click=self.close).props('flat').classes('text-xs font-semibold text-slate-500')
                self.action_btn = ui.button('Next Step', on_click=self.next_step).classes('bg-blue-600 text-white font-bold text-xs px-4 py-2 rounded-lg shadow-sm')

    def switch_view(self, view_name: str):
        self.current_view = view_name
        
        self.tab_gr.classes(add='bg-slate-50' if view_name == 'gr' else '', remove='bg-slate-50' if view_name != 'gr' else '')
        self.tab_scan.classes(add='bg-slate-50' if view_name == 'scan' else '', remove='bg-slate-50' if view_name != 'scan' else '')
        self.tab_merge.classes(add='bg-slate-50' if view_name == 'merge' else '', remove='bg-slate-50' if view_name != 'merge' else '')
        
        if view_name == 'merge':
            self.action_btn.set_text('Finish Consolidation')
        elif view_name == 'gr':
            self.action_btn.set_text('Import Staged Seeds' if self.staged_characters else 'Next Step')
        else:
            self.action_btn.set_text('Next Step')
            
        self.draw_view()

    def set_active_master(self, master_id: Optional[int]):
        self.active_master_id = master_id
        self.draw_view()

    def toggle_magnet(self, e):
        self.magnet_filter_enabled = e.value
        self.draw_view()

    def toggle_hide_populated(self, e):
        self.hide_populated_enabled = e.value
        self.draw_view()

    def handle_fuzz_change(self, e):
        self.fuzz_threshold = e.value
        self.draw_view()

    def refresh_active_chips_only(self):
        """Re-draws active target alias chips in-place without triggering a full screen scroll-reset."""
        if not self.active_master_id or self.active_chips_container is None:
            return
            
        with Session(engine) as session:
            tgt_char = session.get(Character, self.active_master_id)
            if not tgt_char:
                return
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == self.active_master_id)).all()
            alias_list = [a.alias for a in aliases]
            tgt_name = tgt_char.name
            
        self.active_chips_container.clear()
        
        with self.active_chips_container:
            other_aliases = [al for al in alias_list if al.lower() != tgt_name.lower()]
            if not other_aliases:
                ui.label('No custom aliases yet. Click any left-pane [->] buttons to zap them here!').classes('text-xs text-green-600/70 italic m-auto text-center')
            else:
                for alias in other_aliases:
                    with ui.row().classes('items-center bg-green-100 border border-green-300 text-green-800 px-2 py-0.5 rounded-full text-xs gap-1 font-semibold shadow-sm'):
                        ui.label(alias)
                        
                        def make_rollback_cb(al=alias, tid=self.active_master_id):
                            def rollback_alias():
                                remove_alias_from_master_by_text(self.project.id, tid, al)
                                # Redraw everything on reversion to clean and rebuild unresolved lists
                                self.draw_view()
                            return rollback_alias
                            
                        ui.button(icon='close', on_click=make_rollback_cb()).props('flat round dense').classes('text-green-400 hover:text-green-600 text-[9px] w-4 h-4 p-0 min-h-0')

    def draw_view(self):
        self.workspace.clear()
        with self.workspace:
            if self.current_view == 'gr':
                ui.label('Step 1: Seed Characters from Goodreads').classes('text-sm font-bold text-slate-800')
                ui.markdown(
                    "To seed character lists, perform a `Ctrl+A` -> `Ctrl+C` copy of the Goodreads novel page, "
                    "then click and paste (`Ctrl+V`) into the text box below. The engine will automatically process, "
                    "deduplicate, and link the characters to the correct book."
                ).classes('text-xs text-slate-500 leading-normal')
                
                with ui.row().classes('w-full items-center gap-2 bg-slate-50 p-4 border rounded-xl'):
                    ui.icon('bolt', color='amber-500', size='sm')
                    self.gobbler_input = ui.textarea(
                        placeholder="Click here and paste (Ctrl+V) Goodreads page text...",
                        on_change=self.handle_paste
                    ).classes('flex-1 bg-white border rounded px-3 py-1 text-xs h-10 resize-none overflow-hidden').props('outlined dense autofocus')

                with ui.column().classes('w-full gap-2 mt-2'):
                    ui.label('Staged Characters for Import').classes('text-xs font-semibold text-slate-700')
                    self.chip_board = ui.element('div').classes('w-full flex flex-wrap gap-2 p-3 bg-slate-50 rounded-lg min-h-[100px] max-h-[250px] overflow-y-auto border border-dashed')
                    self.draw_chipboard()
                
            elif self.current_view == 'scan':
                ui.label('Step 2: Scan Prompts for Bracketed Tags').classes('text-sm font-bold text-slate-800')
                ui.markdown(
                    "This step scans the project's `prompts.csv` files for bracketed tags like `[Dino]`. "
                    "Any newly found tags will be cross-referenced with your Goodreads seed list."
                ).classes('text-xs text-slate-500 leading-normal')
                
                with ui.row().classes('items-center gap-2 bg-slate-50 p-4 border rounded-xl w-full'):
                    ui.icon('info', color='blue', size='sm')
                    ui.label("This will search all books belonging to the active project.").classes('text-xs text-slate-600')
                    
            elif self.current_view == 'merge':
                # --- STEP 3: ZAP & MAGNET MERGING BOARD ---
                singletons = get_unresolved_singletons(self.project.id)
                masters = get_master_profiles(self.project.id)
                
                # Header control bar
                with ui.row().classes('w-full items-center bg-slate-50 border p-3 rounded-xl gap-4 flex-nowrap'):
                    ui.checkbox('Auto-Filter Fuzzy Magnet', value=self.magnet_filter_enabled, on_change=self.toggle_magnet).props('dense').classes('text-xs font-bold text-slate-700')
                    ui.checkbox('Hide Populated Targets', value=self.hide_populated_enabled, on_change=self.toggle_hide_populated).props('dense').classes('text-xs font-bold text-slate-700')
                    
                    ui.label('Fuzz Threshold:').classes('text-xs font-extrabold text-slate-500 ml-2')
                    ui.slider(min=0.3, max=0.9, step=0.05, value=self.fuzz_threshold, on_change=self.handle_fuzz_change).classes('w-28').props('dense')
                    ui.label(f"{int(self.fuzz_threshold*100)}%").classes('text-xs font-bold text-slate-500 w-8')
                    
                    ui.space()
                    ui.label(f"{len(singletons)} unresolved | {len(masters)} master profiles").classes('text-xs font-bold text-slate-400')
                
                # Dual-pane column layout
                with ui.row().classes('w-full flex-1 min-h-0 gap-4 mt-2 items-stretch flex-nowrap'):
                    # 1. Left Panel: Standalone Singletons (Zap Queue)
                    with ui.column().classes('w-[380px] shrink-0 border rounded-xl p-4 bg-slate-50 gap-3 overflow-hidden flex flex-col h-full'):
                        ui.label('Unresolved Queue').classes('text-xs font-extrabold text-slate-600 uppercase tracking-wider')
                        
                        visible_singletons = []
                        if self.active_master_id is not None and self.magnet_filter_enabled:
                            active_master = next((m for m in masters if m["id"] == self.active_master_id), None)
                            if active_master:
                                for sing in singletons:
                                    if is_loose_match(sing["name"], active_master["name"], active_master["aliases"], self.fuzz_threshold):
                                        visible_singletons.append(sing)
                            else:
                                visible_singletons = singletons
                        else:
                            visible_singletons = singletons
                            
                        # Scrolling sub-list of unresolved elements
                        with ui.column().classes('w-full flex-1 overflow-y-auto gap-2 pr-1'):
                            if not visible_singletons:
                                ui.label('No singletons fit active magnet filter.' if self.active_master_id else 'All singletons resolved!').classes('text-xs text-slate-400 italic m-auto text-center')
                            else:
                                for sing in visible_singletons:
                                    # Instantiate card container referencing element row for localized DOM deletion on click
                                    with ui.row().classes('w-full items-center justify-between bg-white hover:bg-blue-50/20 p-2 rounded-lg border border-slate-200 shadow-sm gap-2 flex-nowrap') as row_el:
                                        
                                        def make_zap_cb(cid=sing["id"], el=row_el, name=sing["name"]):
                                            def on_zap():
                                                if self.active_master_id is None:
                                                    ui.notify("Select an active target profile (Green) on the right first!", type="warning")
                                                    return
                                                
                                                zap_singleton_into_master(self.project.id, cid, self.active_master_id)
                                                # Delete left-pane row element programmatically (maintains scrolling position perfectly!)
                                                el.delete()
                                                # Re-render active master alias chips in-place
                                                self.refresh_active_chips_only()
                                                
                                            return on_zap
                                            
                                        ui.button(icon='chevron_right', on_click=make_zap_cb()).props('dense unelevated').classes('bg-blue-600 hover:bg-blue-700 text-white w-7 h-7 rounded-md shrink-0 flex items-center justify-center')
                                        ui.label(sing["name"]).classes('text-xs font-bold text-slate-700 flex-1 truncate')
                                        
                                        def make_promote_cb(cid=sing["id"]):
                                            def on_promote():
                                                promote_singleton_to_master(self.project.id, cid)
                                                self.draw_view()
                                            return on_promote
                                            
                                        ui.button(icon='star_border', on_click=make_promote_cb()).props('dense flat round').classes('text-slate-400 hover:text-blue-600 shrink-0').tooltip('Promote to Master')

                    # 2. Right Panel: Master Targets Cards
                    with ui.column().classes('flex-1 border rounded-xl p-4 bg-slate-50 gap-3 overflow-hidden flex flex-col h-full'):
                        ui.label('Master Profiles').classes('text-xs font-extrabold text-slate-600 uppercase tracking-wider')
                        
                        visible_masters = []
                        for m in masters:
                            if m["id"] == self.active_master_id:
                                visible_masters.append(m)
                            elif self.hide_populated_enabled and m["is_worked_on"]:
                                continue
                            else:
                                visible_masters.append(m)
                                
                        with ui.column().classes('w-full flex-1 overflow-y-auto gap-2.5 pr-1'):
                            if not visible_masters:
                                ui.label('No master profiles to display.').classes('text-xs text-slate-400 italic m-auto')
                            else:
                                for m in visible_masters:
                                    if m["id"] == self.active_master_id:
                                        # Active target card: Green, expanded, displays all custom chips with delete/rollbacks
                                        with ui.card().classes('w-full border-2 border-green-500 rounded-xl p-4 bg-green-50/40 shadow-sm gap-3 shrink-0'):
                                            with ui.row().classes('w-full justify-between items-center flex-nowrap'):
                                                with ui.row().classes('items-center gap-2 flex-1 min-w-0'):
                                                    ui.icon('flash_on', color='green-600', size='sm')
                                                    ui.label(m["name"]).classes('text-sm font-extrabold text-green-950 truncate')
                                                    if m["is_seeded"]:
                                                        with ui.badge('', color='green-100').classes('text-green-800 border border-green-300 text-[9px] font-bold shrink-0'):
                                                            ui.label('Goodreads Seed')
                                                ui.button(icon='close', on_click=lambda: self.set_active_master(None)).props('dense flat round').classes('text-green-600 hover:text-green-800 shrink-0')
                                                
                                            # We save reference to the active chips container for lightweight localized redrawing
                                            self.active_chips_container = ui.row().classes('w-full flex-wrap gap-2 bg-white/70 p-3 rounded-lg border border-green-200 min-h-[50px]')
                                            
                                            # Draw its initial contents
                                            self.refresh_active_chips_only()
                                    else:
                                        # Inactive target card: compact style, white/light blue depending on populated state
                                        if m["is_worked_on"]:
                                            card_style = 'w-full border border-blue-200 rounded-xl p-3 bg-blue-50/40 hover:bg-blue-50/80 cursor-pointer shadow-sm shrink-0'
                                            text_style = 'text-xs font-bold text-blue-900'
                                        else:
                                            card_style = 'w-full border border-slate-200 rounded-xl p-3 bg-white hover:bg-slate-50 cursor-pointer shadow-sm shrink-0'
                                            text_style = 'text-xs font-bold text-slate-800'
                                            
                                        with ui.card().classes(card_style).on('click', lambda m_id=m["id"]: self.set_active_master(m_id)):
                                            with ui.row().classes('w-full justify-between items-center flex-nowrap'):
                                                with ui.row().classes('items-center gap-2 flex-1 min-w-0'):
                                                    if m["is_seeded"]:
                                                        ui.icon('verified', color='green-500', size='xs')
                                                        ui.label(m["name"]).classes('text-xs font-bold text-green-700 truncate')
                                                    else:
                                                        ui.icon('person', color='slate-400', size='xs')
                                                        ui.label(m["name"]).classes(f"{text_style} truncate")
                                                        
                                                # Alias count badge
                                                alias_count = len(m["aliases"])
                                                if alias_count > 1:
                                                    with ui.badge('', color='blue-100').classes('text-blue-800 border border-blue-200 text-[10px] font-bold shrink-0'):
                                                        ui.label(f"{alias_count} aliases")

    def handle_paste(self, e):
        raw_text = e.value.strip() if e.value else ""
        if not raw_text:
            return
        
        ui.timer(0.1, lambda: self.gobbler_input.set_value(''), once=True)
        
        result = parse_goodreads_dump(raw_text)
        if not result["success"]:
            ui.notify("No valid Goodreads character list or metadata detected in paste.", type="warning")
            return
            
        with Session(engine) as session:
            matched_book = find_matching_project_book(self.project.id, result["title"], session)
            
        if matched_book:
            self.associate_and_stage(result, matched_book.id)
            ui.notify(f"Successfully gobbled {len(result['characters'])} characters for book: '{matched_book.name}'", type="positive")
        else:
            self.show_book_selector_dialog(result)

    def associate_and_stage(self, result: Dict[str, Any], book_id: Optional[int]):
        if book_id is not None:
            if book_id not in self.staged_book_metadata:
                self.staged_book_metadata[book_id] = {}
            if result["title"]:
                self.staged_book_metadata[book_id]["title"] = result["title"]
            if result["author"]:
                self.staged_book_metadata[book_id]["author"] = result["author"]
                
        for char in result["characters"]:
            if char not in self.staged_characters:
                self.staged_characters[char] = {"book_ids": set(), "count": 0}
            self.staged_characters[char]["count"] += 1
            if book_id is not None:
                self.staged_characters[char]["book_ids"].add(book_id)
                
        self.draw_chipboard()
        if self.current_view == 'gr' and self.staged_characters:
            self.action_btn.set_text('Import Staged Seeds')

    def show_book_selector_dialog(self, result: Dict[str, Any]):
        selector = ui.dialog()
        with selector, ui.card().classes('w-[450px] p-5 gap-3 rounded-lg'):
            ui.label("Map Goodreads Metadata").classes("font-bold text-sm text-slate-800")
            parsed_title_lbl = result["title"] or "Unrecognized Book Title"
            ui.markdown(
                f"We parsed the pasted text as belonging to **{parsed_title_lbl}**, "
                "but couldn't find a matching book in this project. Which book does this content belong to?"
            ).classes("text-xs text-slate-500 leading-relaxed")
            
            options = {b.id: b.name for b in self.books}
            options[None] = "Project-wide / No specific book"
            
            default_val = self.books[0].id if self.books else None
            radio = ui.radio(options=options, value=default_val).classes("text-xs gap-1")
            
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=selector.close).props("flat dense").classes("text-xs text-slate-500")
                
                def on_confirm():
                    selected_book_id = radio.value
                    self.associate_and_stage(result, selected_book_id)
                    selector.close()
                    ui.notify(f"Linked {len(result['characters'])} characters successfully.", type="positive")
                    
                ui.button("Associate", on_click=on_confirm).classes("bg-blue-600 text-white font-bold text-xs px-3 py-1 rounded")
        selector.open()

    def draw_chipboard(self):
        if not hasattr(self, 'chip_board') or self.chip_board is None:
            return
            
        self.chip_board.clear()
        
        if not self.staged_characters:
            with self.chip_board:
                ui.label('No characters staged yet. Paste Goodreads content to seed.').classes('text-xs text-slate-400 italic m-auto')
            return
            
        with self.chip_board:
            for char_name in sorted(self.staged_characters.keys()):
                char_data = self.staged_characters[char_name]
                count_lbl = f" ({char_data['count']})" if char_data['count'] > 1 else ""
                
                with ui.row().classes('items-center bg-blue-50 text-blue-800 border border-blue-200 px-2 py-0.5 rounded-full text-xs gap-1 font-semibold shadow-sm'):
                    ui.label(f"{char_name}{count_lbl}")
                    
                    def remove_char(target_name=char_name):
                        self.staged_characters.pop(target_name, None)
                        self.draw_chipboard()
                        if not self.staged_characters and self.current_view == 'gr':
                            self.action_btn.set_text('Next Step')
                            
                    ui.button(icon='close', on_click=remove_char).props('flat round dense').classes('text-blue-400 hover:text-blue-600 text-[10px] w-4 h-4 p-0 min-h-0')

    def execute_import(self):
        if not self.staged_characters:
            ui.notify("No characters are currently staged to import.", type="warning")
            return
            
        flat_staged_names = {name: data["book_ids"] for name, data in self.staged_characters.items()}
            
        for book_id, meta in self.staged_book_metadata.items():
            update_book_metadata(book_id, meta.get("title"), meta.get("author"))
            
        commit_seeded_characters(self.project.id, flat_staged_names)
        
        ui.notify(f"Successfully created {len(self.staged_characters)} seeded characters!", type="positive")
        self.staged_characters.clear()
        self.staged_book_metadata.clear()
        self.draw_chipboard()
        self.refresh_callback()
        
        self.switch_view('scan')

    def next_step(self):
        if self.current_view == 'gr':
            if self.staged_characters:
                self.execute_import()
            else:
                self.switch_view('scan')
        elif self.current_view == 'scan':
            extract_characters_from_prompts(self.project.id)
            ui.notify("Scanned prompts for new character tags.", type="info")
            self.switch_view('merge')
        elif self.current_view == 'merge':
            # Run final calculations and cleanups upon exiting
            ui.notify("Recalculating character hit counts...", type="info")
            recalculate_project_character_hits(self.project.id)
            
            ui.notify("Pruning unused characters...", type="info")
            prune_unused_seeded_characters(self.project.id)
            
            ui.notify("Consolidation finalized!", type="positive")
            
            self.active_master_id = None
            self.refresh_callback()
            self.close()

    def reset_steps(self):
        self.staged_characters.clear()
        self.staged_book_metadata.clear()
        self.active_master_id = None
        self.draw_chipboard()
        ui.notify("Organizer workflow reset.", type="warning")
        self.switch_view('gr')