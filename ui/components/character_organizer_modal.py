# ui/components/character_organizer_modal.py

from typing import List, Callable, Dict, Set, Any, Optional
import inspect
import asyncio
from nicegui import ui
from sqlmodel import Session, select
from database.connection import engine
from database.models import Project, Book, Character, CharacterAlias, CharacterBookLink

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
    demote_master_to_singleton,
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
        
        # Workspace Selection / Toggle State
        self.active_master_id: Optional[int] = None
        self.active_singleton_id: Optional[int] = None
        self.magnet_filter_enabled = True
        self.hide_populated_enabled = False
        self.show_masters_in_queue = False
        self.fuzz_threshold = 0.60
        
        # Panel sorting states (False = Alphabetical, True = Hits Descending)
        self.unresolved_sort_by_hits = False
        self.masters_sort_by_hits = False
        
        # Live Search Filters
        self.unresolved_filter_text = ""
        self.masters_filter_text = ""
        
        # Programmatic local references
        self.active_chips_container = None
        self.singletons_container = None
        self.masters_list_container = None
        self.unresolved_filter_input = None
        self.masters_filter_input = None
        self.rendered_singleton_ids: Set[int] = set()
        
        self.current_view = 'gr'
        
        with self, ui.card().classes('w-[1020px] max-w-[98vw] h-[820px] max-h-[94vh] p-6 rounded-xl flex flex-col overflow-hidden'):
            # Consolidated Compact Header Block
            with ui.row().classes('w-full justify-between items-center border-b pb-2.5 shrink-0'):
                ui.label('Character Utility Workspace').classes('text-sm font-extrabold text-slate-800')
                ui.button(icon='close', on_click=self.close).props('flat dense').classes('text-slate-400')

            # Main Body Layout
            with ui.row().classes('w-full flex-1 min-h-0 gap-4 mt-3 items-stretch flex-nowrap'):
                # Consolidated Left Navigation Bar
                with ui.column().classes('w-52 shrink-0 border-r pr-4 gap-2 flex flex-col h-full justify-between pb-1'):
                    with ui.column().classes('w-full gap-2 shrink-0'):
                        self.tab_gr = ui.button(
                            'Character Seeder', 
                            icon='list_alt', 
                            on_click=lambda: self.switch_view('gr')
                        ).props('flat align=left').classes('w-full text-xs font-semibold justify-start text-slate-700 bg-slate-50')
                        
                        self.tab_scan = ui.button(
                            'Prompt Tag Scanner', 
                            icon='tag', 
                            on_click=lambda: self.switch_view('scan')
                        ).props('flat align=left').classes('w-full text-xs font-semibold justify-start text-slate-700')
                        
                        self.tab_merge = ui.button(
                            'Merge Workspace', 
                            icon='call_merge', 
                            on_click=lambda: self.switch_view('merge')
                        ).props('flat align=left').classes('w-full text-xs font-semibold justify-start text-slate-700')
                    
                    with ui.column().classes('w-full gap-2 mt-auto shrink-0'):
                        ui.button(
                            'Close Workspace', 
                            icon='check', 
                            on_click=self.close
                        ).classes('w-full bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs py-2 rounded-lg shadow-sm')

                # Right Workspace Container
                with ui.column().classes('flex-1 min-h-0 gap-4 flex flex-col overflow-hidden') as self.workspace:
                    self.draw_view()

    def safe_invoke_callback(self):
        """Safely executes the user refresh callback whether it is synchronous or an async coroutine."""
        if not self.refresh_callback:
            return
        if inspect.iscoroutinefunction(self.refresh_callback):
            asyncio.create_task(self.refresh_callback())
        else:
            self.refresh_callback()

    def switch_view(self, view_name: str):
        with self.client:
            self.current_view = view_name
            
            self.tab_gr.classes(add='bg-slate-50' if view_name == 'gr' else '', remove='bg-slate-50' if view_name != 'gr' else '')
            self.tab_scan.classes(add='bg-slate-50' if view_name == 'scan' else '', remove='bg-slate-50' if view_name != 'scan' else '')
            self.tab_merge.classes(add='bg-slate-50' if view_name == 'merge' else '', remove='bg-slate-50' if view_name != 'merge' else '')
            
            if view_name == 'merge':
                # Run hit recalculation dynamically to refresh zeroed states on entries
                recalculate_project_character_hits(self.project.id)
                
            self.draw_view()

    def set_active_master(self, master_id: Optional[int]):
        with self.client:
            self.active_master_id = master_id
            if master_id is not None:
                self.active_singleton_id = None # Clear active singleton when selecting a master
            self.refresh_masters_list_only()
            self.refresh_unresolved_list_only()

    def set_active_singleton(self, singleton_id: Optional[int]):
        with self.client:
            self.active_singleton_id = singleton_id
            if singleton_id is not None:
                self.active_master_id = None # Clear active master when selecting a singleton
            self.refresh_masters_list_only()
            self.refresh_unresolved_list_only()

    def toggle_magnet(self, e):
        with self.client:
            self.magnet_filter_enabled = e.value
            self.refresh_unresolved_list_only()
            self.refresh_masters_list_only()

    def toggle_hide_populated(self, e):
        with self.client:
            self.hide_populated_enabled = e.value
            self.refresh_masters_list_only()

    def toggle_show_masters(self, e):
        with self.client:
            self.show_masters_in_queue = e.value
            self.refresh_unresolved_list_only()

    def toggle_unresolved_sort(self):
        with self.client:
            self.unresolved_sort_by_hits = not self.unresolved_sort_by_hits
            self.refresh_unresolved_list_only()

    def toggle_masters_sort(self):
        with self.client:
            self.masters_sort_by_hits = not self.masters_sort_by_hits
            self.refresh_masters_list_only()

    def handle_fuzz_change(self, e):
        with self.client:
            new_val = e.args if hasattr(e, 'args') and e.args is not None else getattr(e, 'value', self.fuzz_threshold)
            if isinstance(new_val, list) and len(new_val) > 0:
                new_val = new_val[0]
                
            try:
                self.fuzz_threshold = float(new_val)
            except (ValueError, TypeError):
                return
                
            self.refresh_unresolved_list_only()
            self.refresh_masters_list_only()

    def handle_unresolved_search(self, e):
        with self.client:
            self.unresolved_filter_text = e.value.strip().lower() if e.value else ""
            self.refresh_unresolved_list_only()

    def handle_masters_search(self, e):
        with self.client:
            self.masters_filter_text = e.value.strip().lower() if e.value else ""
            self.refresh_masters_list_only()

    def populate_active_chips_inline(self, char_id: int, tgt_name: str):
        with Session(engine) as session:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char_id)).all()
            alias_list = [a.alias for a in aliases]
        
        other_aliases = [al for al in alias_list if al.lower() != tgt_name.lower()]
        if not other_aliases:
            ui.label('No custom aliases yet. Click any left-pane [->] buttons to zap them here!').classes('text-xs text-green-600/70 italic m-auto text-center')
        else:
            for alias in other_aliases:
                with ui.row().classes('items-center bg-green-100 border border-green-300 text-green-800 px-2 py-0.5 rounded-full text-xs gap-1 font-semibold shadow-sm'):
                    ui.label(alias)
                    
                    def make_rollback_cb(al=alias, tid=char_id):
                        def rollback_alias():
                            with self.client:
                                remove_alias_from_master_by_text(self.project.id, tid, al)
                                self.draw_view()
                        return rollback_alias
                        
                    ui.button(icon='close', on_click=make_rollback_cb()).props('flat round dense').classes('text-green-400 hover:text-green-600 text-[9px] w-4 h-4 p-0 min-h-0')

    def refresh_active_chips_only(self):
        """Re-draws active target alias chips in-place during dynamic updates."""
        if not self.active_master_id or self.active_chips_container is None:
            return
            
        with Session(engine) as session:
            tgt_char = session.get(Character, self.active_master_id)
            if not tgt_char:
                return
            tgt_name = tgt_char.name
            
        self.active_chips_container.clear()
        with self.active_chips_container:
            self.populate_active_chips_inline(self.active_master_id, tgt_name)

    def render_singleton_row(self, item):
        is_selected = (item["id"] == self.active_singleton_id)
        border_style = 'border-2 border-blue-500 bg-blue-50/50 shadow-md' if is_selected else 'border-slate-200 bg-white hover:bg-blue-50/20'
        
        with ui.row().classes(f'w-full items-center justify-between p-2 rounded-lg border shadow-sm gap-2 flex-nowrap cursor-pointer {border_style}') as row_el:
            row_el.on('click', lambda: self.set_active_singleton(None if is_selected else item["id"]))
            
            def make_zap_cb(cid=item["id"], el=row_el, name=item["name"]):
                def on_zap():
                    with self.client:
                        if self.active_master_id is None:
                            ui.notify("Select an active target profile (Green) on the right first!", type="warning")
                            return
                        
                        zap_singleton_into_master(self.project.id, cid, self.active_master_id)
                        el.delete()
                        self.rendered_singleton_ids.discard(cid)
                        
                        self.refresh_active_chips_only()
                        self.refresh_unresolved_list_only()
                        self.refresh_masters_list_only()
                return on_zap
                
            ui.button(icon='chevron_right').on('click.stop', make_zap_cb()).props('dense unelevated').classes('bg-blue-600 hover:bg-blue-700 text-white w-7 h-7 rounded-md shrink-0 flex items-center justify-center')
            
            if item["is_master"]:
                ui.label(item["name"]).classes('text-xs font-bold text-blue-700 flex-1 truncate').tooltip('Promoted Master Profile')
            else:
                ui.label(item["name"]).classes('text-xs font-bold text-slate-700 flex-1 truncate')
            
            hits = item.get("hit_count", 0)
            with ui.badge('', color='amber-100' if hits > 0 else 'slate-100').classes('text-amber-800 border border-amber-200 text-[10px] font-bold shrink-0' if hits > 0 else 'text-slate-400 border border-slate-200 text-[10px] font-medium shrink-0'):
                ui.label(f"{hits} hits")
            
            if not item["is_master"]:
                def make_promote_cb(cid=item["id"], name=item["name"]):
                    def on_promote():
                        with self.client:
                            promote_singleton_to_master(self.project.id, cid)
                            self.set_active_master(cid)
                            ui.notify(f"Promoted '{name}' to Master Profile!", type="success")
                    return on_promote
                    
                ui.button(icon='star_border').on('click.stop', make_promote_cb()).props('dense flat round').classes('text-slate-400 hover:text-blue-600 shrink-0').tooltip('Promote to Master')
            else:
                ui.icon('star', color='amber-500', size='xs').classes('shrink-0 mr-1').tooltip('Promoted Master Indicator')

        self.rendered_singleton_ids.add(item["id"])

    def check_and_append_new_fuzzy_matches(self):
        """Appends newly qualified queue items matching the master's expanded tag boundaries."""
        if not self.active_master_id or not self.magnet_filter_enabled:
            return
            
        singletons = get_unresolved_singletons(self.project.id)
        masters = get_master_profiles(self.project.id)
        active_master = next((m for m in masters if m["id"] == self.active_master_id), None)
        if not active_master:
            return
            
        queue_items = []
        for s in singletons:
            queue_items.append({
                "id": s["id"],
                "name": s["name"],
                "hit_count": s["hit_count"],
                "is_master": False
            })
            
        if self.show_masters_in_queue:
            for m in masters:
                if m["id"] == self.active_master_id:
                    continue
                queue_items.append({
                    "id": m["id"],
                    "name": f"★ {m['name']}",
                    "hit_count": m["hit_count"],
                    "is_master": True
                })
                
        newly_matched = []
        for item in queue_items:
            if item["id"] in self.rendered_singleton_ids:
                continue
            clean_name = item["name"].replace("★ ", "").strip()
            item_aliases = []
            if item["is_master"]:
                matched_m = next((m for m in masters if m["id"] == item["id"]), None)
                if matched_m:
                    item_aliases = matched_m["aliases"]
            if is_loose_match(clean_name, active_master["name"], active_master["aliases"], self.fuzz_threshold) or \
               any(is_loose_match(al, active_master["name"], active_master["aliases"], self.fuzz_threshold) for al in item_aliases):
                if self.unresolved_filter_text and self.unresolved_filter_text not in item["name"].lower():
                    continue
                newly_matched.append(item)
                
        if newly_matched:
            with self.singletons_container:
                for item in newly_matched:
                    self.render_singleton_row(item)

    def auto_merge_easy(self):
        """
        Clustered Auto-Merge. Runs two consolidation passes:
        Pass 1: Merges singletons matching exactly one master profile.
        Pass 2: Groups remaining singletons with each other, promoting the one with the highest hit frequency.
        """
        with self.client:
            singletons = get_unresolved_singletons(self.project.id)
            masters = get_master_profiles(self.project.id)
            
            if not singletons:
                ui.notify("No unresolved singletons to auto-merge.", type="info")
                return
                
            merged_count = 0
            promoted_count = 0
            processed_ids = set()
            
            # Pass 1: Resolve Singleton-to-Master mapping (unambiguous matches)
            for sing in singletons:
                matched_masters = []
                for m in masters:
                    if is_loose_match(sing["name"], m["name"], m["aliases"], self.fuzz_threshold):
                        matched_masters.append(m)
                        
                if len(matched_masters) == 1:
                    target_master = matched_masters[0]
                    zap_singleton_into_master(self.project.id, sing["id"], target_master["id"])
                    processed_ids.add(sing["id"])
                    merged_count += 1
                    target_master["aliases"].append(sing["name"])
                    
            # Pass 2: Resolve Singleton-to-Singleton matches (unseeded environments)
            remaining_singletons = [s for s in singletons if s["id"] not in processed_ids]
            
            for sing in remaining_singletons:
                if sing["id"] in processed_ids:
                    continue
                    
                cluster = [sing]
                for other in remaining_singletons:
                    if other["id"] == sing["id"] or other["id"] in processed_ids:
                        continue
                    if is_loose_match(other["name"], sing["name"], [], self.fuzz_threshold):
                        cluster.append(other)
                        
                if len(cluster) > 1:
                    # Sort cluster descending by hit frequency to find the optimal canonical master
                    cluster.sort(key=lambda x: x["hit_count"], reverse=True)
                    parent = cluster[0]
                    
                    promote_singleton_to_master(self.project.id, parent["id"])
                    promoted_count += 1
                    processed_ids.add(parent["id"])
                    
                    for child in cluster[1:]:
                        zap_singleton_into_master(self.project.id, child["id"], parent["id"])
                        processed_ids.add(child["id"])
                        merged_count += 1
                        
            if merged_count > 0 or promoted_count > 0:
                ui.notify(f"Auto-merged {merged_count} characters and promoted {promoted_count} new master profiles!", type="positive")
                self.draw_view()
            else:
                ui.notify("No unambiguous fuzzy matches found at current threshold.", type="info")

    def run_explicit_pruning(self):
        """Silently deletes any unlocked character with hit_count == 0."""
        with self.client:
            prune_unused_seeded_characters(self.project.id)
            ui.notify("Pruned unused seeded characters with 0 hits.", type="positive")
            self.safe_invoke_callback()
            self.draw_view()

    def render_master_row(self, m):
        if m["id"] == self.active_master_id:
            # Active target card: Green, expanded, displays all custom chips with delete/rollbacks
            with ui.card().classes('w-full border-2 border-green-500 rounded-xl p-4 bg-green-50/40 shadow-sm gap-3 shrink-0') as active_card:
                with ui.row().classes('w-full justify-between items-center flex-nowrap'):
                    with ui.row().classes('items-center gap-2 flex-1 min-w-0'):
                        ui.icon('flash_on', color='green-600', size='sm')
                        ui.label(m["name"]).classes('text-sm font-extrabold text-green-950 truncate')
                        if m["is_seeded"]:
                            with ui.badge('', color='green-100').classes('text-green-800 border border-green-300 text-[9px] font-bold shrink-0'):
                                ui.label('Goodreads Seed')
                        
                        hits = m.get("hit_count", 0)
                        with ui.badge('', color='amber-100' if hits > 0 else 'slate-100').classes('text-amber-800 border border-amber-200 text-[10px] font-bold shrink-0' if hits > 0 else 'text-slate-400 border border-slate-200 text-[10px] font-medium shrink-0'):
                            ui.label(f"{hits} hits")
                            
                    ui.button(icon='close', on_click=lambda: self.set_active_master(None)).props('flat round dense').classes('text-green-600 hover:text-green-800 shrink-0')
                    
                self.active_chips_container = ui.row().classes('w-full flex-wrap gap-2 bg-white/70 p-3 rounded-lg border border-green-200 min-h-[50px]')
                with self.active_chips_container:
                    self.populate_active_chips_inline(m["id"], m["name"])
                    
            ui.timer(0.05, lambda: ui.run_javascript(f'document.getElementById("c{active_card.id}").scrollIntoView({{behavior: "smooth", block: "nearest"}})' if active_card else ''), once=True)
        else:
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
                            
                            if not m["locked"]:
                                def make_demote_cb(mid=m["id"]):
                                    def on_demote():
                                        with self.client:
                                            demote_master_to_singleton(self.project.id, mid)
                                            ui.notify("Character demoted back to Unresolved Queue.", type="info")
                                            self.draw_view()
                                    return on_demote
                                ui.button(icon='undo').on('click.stop', make_demote_cb).props('dense flat round size=xs').classes('text-slate-400 hover:text-red-500 shrink-0 ml-1').tooltip('Demote to Unresolved Singleton')
                                
                    with ui.row().classes('items-center gap-1 shrink-0'):
                        hits = m.get("hit_count", 0)
                        with ui.badge('', color='amber-100' if hits > 0 else 'slate-100').classes('text-amber-800 border border-amber-200 text-[10px] font-bold' if hits > 0 else 'text-slate-400 border border-slate-200 text-[10px] font-medium'):
                            ui.label(f"{hits} hits")
                        
                        alias_count = len(m["aliases"])
                        if alias_count > 1:
                            with ui.badge('', color='blue-100').classes('text-blue-800 border border-blue-200 text-[10px] font-bold shrink-0'):
                                ui.label(f"{alias_count} aliases")

    def refresh_unresolved_list_only(self):
        """Clears and rebuilds only the unresolved list container, protecting search field input focus."""
        if not self.singletons_container:
            return
            
        self.singletons_container.clear()
        self.rendered_singleton_ids.clear()
        
        singletons = get_unresolved_singletons(self.project.id)
        masters = get_master_profiles(self.project.id)
        
        queue_items = []
        for s in singletons:
            queue_items.append({
                "id": s["id"],
                "name": s["name"],
                "hit_count": s["hit_count"],
                "is_master": False
            })
            
        if self.show_masters_in_queue:
            for m in masters:
                if self.active_master_id is not None and m["id"] == self.active_master_id:
                    continue
                queue_items.append({
                    "id": m["id"],
                    "name": f"★ {m['name']}",
                    "hit_count": m["hit_count"],
                    "is_master": True
                })
        
        visible_items = []
        if self.active_master_id is not None and self.magnet_filter_enabled:
            active_master = next((m for m in masters if m["id"] == self.active_master_id), None)
            if active_master:
                for item in queue_items:
                    clean_name = item["name"].replace("★ ", "").strip()
                    item_aliases = []
                    if item["is_master"]:
                        matched_m = next((m for m in masters if m["id"] == item["id"]), None)
                        if matched_m:
                            item_aliases = matched_m["aliases"]
                    if is_loose_match(clean_name, active_master["name"], active_master["aliases"], self.fuzz_threshold) or \
                       any(is_loose_match(al, active_master["name"], active_master["aliases"], self.fuzz_threshold) for al in item_aliases):
                        visible_items.append(item)
            else:
                visible_items = queue_items
        else:
            visible_items = queue_items
            
        if self.unresolved_filter_text:
            visible_items = [
                item for item in visible_items 
                if self.unresolved_filter_text in item["name"].lower()
            ]
            
        # Apply sorting
        if self.unresolved_sort_by_hits:
            visible_items.sort(key=lambda x: x["hit_count"], reverse=True)
        else:
            visible_items.sort(key=lambda x: x["name"].lower())
            
        with self.singletons_container:
            if not visible_items:
                ui.label('No matches found.').classes('text-xs text-slate-400 italic m-auto text-center')
            else:
                for item in visible_items:
                    self.render_singleton_row(item)

    def refresh_masters_list_only(self):
        """Clears and rebuilds only the master profiles list container, protecting search field input focus."""
        if not self.masters_list_container:
            return
            
        self.masters_list_container.clear()
        masters = get_master_profiles(self.project.id)
        
        active_sing = None
        if self.active_singleton_id is not None and self.magnet_filter_enabled:
            singletons = get_unresolved_singletons(self.project.id)
            active_sing = next((s for s in singletons if s["id"] == self.active_singleton_id), None)
            
        visible_masters = []
        for m in masters:
            if self.hide_populated_enabled and m["is_worked_on"] and m["id"] != self.active_master_id:
                continue
            
            if active_sing:
                if not is_loose_match(active_sing["name"], m["name"], m["aliases"], self.fuzz_threshold):
                    continue
                    
            visible_masters.append(m)
            
        if self.masters_filter_text:
            visible_masters = [
                m for m in visible_masters 
                if self.masters_filter_text in m["name"].lower() or any(self.masters_filter_text in al.lower() for al in m["aliases"])
            ]
            
        # Apply sorting
        if self.masters_sort_by_hits:
            visible_masters.sort(key=lambda x: x["hit_count"], reverse=True)
        else:
            visible_masters.sort(key=lambda x: x["name"].lower())
            
        with self.masters_list_container:
            if not visible_masters:
                ui.label('No matches found.').classes('text-xs text-slate-400 italic m-auto')
            else:
                for m in visible_masters:
                    self.render_master_row(m)

    def run_tag_scan(self):
        """Executes the prompt tag scanning process."""
        with self.client:
            ui.notify("Scanning prompts for bracketed character tags...", type="info")
            discovered = extract_characters_from_prompts(self.project.id)
            ui.notify(f"Scan complete! Cataloged {len(discovered)} tags.", type="positive")
            self.safe_invoke_callback()
            self.switch_view('merge')

    def draw_view(self):
        self.workspace.clear()
        with self.workspace:
            if self.current_view == 'gr':
                with ui.column().classes('w-full flex-1 overflow-y-auto gap-4 pr-1'):
                    ui.label('Character Seeder').classes('text-sm font-bold text-slate-800')
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
                        with ui.row().classes('w-full justify-between items-center'):
                            ui.label('Staged Characters for Import').classes('text-xs font-semibold text-slate-700')
                            self.import_staged_btn = ui.button(
                                'Import & Commit Staged Seeds', 
                                icon='cloud_upload', 
                                on_click=self.execute_import
                            ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-[10px] px-3 py-1 rounded-lg shadow-sm hidden')
                            
                        self.chip_board = ui.element('div').classes('w-full flex flex-wrap gap-2 p-3 bg-slate-50 rounded-lg min-h-[100px] max-h-[250px] overflow-y-auto border border-dashed')
                        self.draw_chipboard()
                
            elif self.current_view == 'scan':
                with ui.column().classes('w-full flex-1 overflow-y-auto gap-4 pr-1'):
                    ui.label('Prompt Tag Scanner').classes('text-sm font-bold text-slate-800')
                    ui.markdown(
                        "This tool scans your project's `prompts.csv` files for bracketed character tags like `[Dino]`. "
                        "Any newly found tags will be cross-referenced and indexed in the database."
                    ).classes('text-xs text-slate-500 leading-normal')
                    
                    with ui.row().classes('items-center gap-2 bg-slate-50 p-4 border rounded-xl w-full'):
                        ui.icon('info', color='blue', size='sm')
                        ui.label("This will search all books belonging to the active project.").classes('text-xs text-slate-600')
                        
                    ui.button(
                        'Run Prompt Scan Now', 
                        icon='play_arrow', 
                        on_click=self.run_tag_scan
                    ).classes('w-48 bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs py-2 rounded-lg shadow-sm mt-4')
                    
            elif self.current_view == 'merge':
                singletons = get_unresolved_singletons(self.project.id)
                masters = get_master_profiles(self.project.id)
                
                with ui.row().classes('w-full items-center bg-slate-50 border px-3 py-1.5 rounded-xl gap-3 flex-nowrap shrink-0'):
                    ui.checkbox('Auto-Filter Magnet', value=self.magnet_filter_enabled, on_change=self.toggle_magnet).props('dense').classes('text-[11px] font-bold text-slate-700')
                    ui.checkbox('Hide Populated', value=self.hide_populated_enabled, on_change=self.toggle_hide_populated).props('dense').classes('text-[11px] font-bold text-slate-700')
                    
                    ui.label('Fuzz:').classes('text-[11px] font-extrabold text-slate-500 ml-1')
                    ui.slider(min=0.3, max=0.9, step=0.05, value=self.fuzz_threshold).classes('w-20').props('dense').on('change', self.handle_fuzz_change)
                    ui.label(f"{int(self.fuzz_threshold*100)}%").classes('text-[11px] font-bold text-slate-500 w-8')
                    
                    ui.button('Auto-Merge Easy', icon='auto_awesome', on_click=self.auto_merge_easy).classes('bg-purple-600 hover:bg-purple-700 text-white font-bold text-[10px] px-2.5 py-1 rounded-lg shadow-sm')
                    ui.button('Prune Unused (0 Hits)', icon='delete_sweep', on_click=self.run_explicit_pruning).classes('bg-red-500 hover:bg-red-600 text-white font-bold text-[10px] px-2.5 py-1 rounded-lg shadow-sm').tooltip('Silently deletes any unlocked character with hit_count == 0.')
                    
                    ui.space()
                    ui.label(f"{len(singletons)} unresolved | {len(masters)} masters").classes('text-[11px] font-bold text-slate-400')
                
                with ui.row().classes('w-full flex-1 min-h-0 gap-4 mt-2 items-stretch flex-nowrap'):
                    # 1. Left Panel: Standalone Singletons (Zap Queue)
                    with ui.column().classes('w-[380px] shrink-0 border rounded-xl p-4 bg-slate-50 gap-3 overflow-hidden flex flex-col h-full'):
                        with ui.row().classes('w-full justify-between items-center'):
                            ui.label('Unresolved Queue').classes('text-xs font-extrabold text-slate-600 uppercase tracking-wider')
                            with ui.row().classes('items-center gap-1'):
                                ui.checkbox('Show Masters', value=self.show_masters_in_queue, on_change=self.toggle_show_masters).props('dense').classes('text-[10px] font-bold text-slate-500 mr-1')
                                ui.button(
                                    icon='bar_chart' if self.unresolved_sort_by_hits else 'sort_by_alpha', 
                                    on_click=self.toggle_unresolved_sort
                                ).props('flat dense size=sm').classes('text-slate-500').tooltip('Sorted by Hits' if self.unresolved_sort_by_hits else 'Sorted Alphabetically')
                        
                        self.unresolved_filter_input = ui.input(
                            placeholder='Search unresolved...',
                            value=self.unresolved_filter_text,
                            on_change=self.handle_unresolved_search
                        ).props('dense outlined clearable').classes('w-full text-xs bg-white shrink-0')
                        
                        self.singletons_container = ui.column().classes('w-full flex-1 overflow-y-auto gap-2 pr-1')
                        self.refresh_unresolved_list_only()

                    # 2. Right Panel: Master Targets Cards
                    with ui.column().classes('flex-1 border rounded-xl p-4 bg-slate-50 gap-3 overflow-hidden flex flex-col h-full'):
                        with ui.row().classes('w-full justify-between items-center'):
                            ui.label('Master Profiles').classes('text-xs font-extrabold text-slate-600 uppercase tracking-wider')
                            ui.button(
                                icon='bar_chart' if self.masters_sort_by_hits else 'sort_by_alpha', 
                                on_click=self.toggle_masters_sort
                            ).props('flat dense size=sm').classes('text-slate-500').tooltip('Sorted by Hits' if self.masters_sort_by_hits else 'Sorted Alphabetically')
                        
                        self.masters_filter_input = ui.input(
                            placeholder='Search master names or aliases...',
                            value=self.masters_filter_text,
                            on_change=self.handle_masters_search
                        ).props('dense outlined clearable').classes('w-full text-xs bg-white shrink-0')
                        
                        self.masters_list_container = ui.column().classes('w-full flex-1 overflow-y-auto gap-2.5 pr-1')
                        self.refresh_masters_list_only()

    def handle_paste(self, e):
        with self.client:
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
                    with self.client:
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
            if hasattr(self, 'import_staged_btn'):
                self.import_staged_btn.classes('hidden')
            with self.chip_board:
                ui.label('No characters staged yet. Paste Goodreads content to seed.').classes('text-xs text-slate-400 italic m-auto')
            return
            
        if hasattr(self, 'import_staged_btn'):
            self.import_staged_btn.classes(remove='hidden')
            
        with self.chip_board:
            for char_name in sorted(self.staged_characters.keys()):
                char_data = self.staged_characters[char_name]
                count_lbl = f" ({char_data['count']})" if char_data['count'] > 1 else ""
                
                with ui.row().classes('items-center bg-blue-50 text-blue-800 border border-blue-200 px-2 py-0.5 rounded-full text-xs gap-1 font-semibold shadow-sm'):
                    ui.label(f"{char_name}{count_lbl}")
                    
                    def remove_char(target_name=char_name):
                        with self.client:
                            self.staged_characters.pop(target_name, None)
                            self.draw_chipboard()
                            
                    ui.button(icon='close', on_click=remove_char).props('flat round dense').classes('text-blue-400 hover:text-blue-600 text-[10px] w-4 h-4 p-0 min-h-0')

    def execute_import(self):
        with self.client:
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
            
            self.safe_invoke_callback()
            self.switch_view('scan')

    def reset_steps(self):
        """Legacy compatibility reset, clears memory stage arrays."""
        with self.client:
            self.staged_characters.clear()
            self.staged_book_metadata.clear()
            self.active_master_id = None
            self.active_singleton_id = None
            self.draw_chipboard()
            ui.notify("Workspace state reset.", type="warning")
            self.switch_view('gr')