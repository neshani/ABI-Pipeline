# ui/pages/project/packager_playground.py

import time
import asyncio
from pathlib import Path
from nicegui import ui
from sqlmodel import Session, select
from typing import Any, Dict

from database.connection import engine
from database.models import Project, Book, Setting
from services.packager import build_illumination_pack, find_audiobook_file, get_book_total_duration, get_audio_file_metadata
from services.project_settings import get_project_settings_path
from services.sync_engine import get_book_stats_cached
import json

class PackagerPlayground:
    def __init__(self, project_id: int, project_name: str):
        self.project_id = project_id
        self.project_name = project_name
        
        # Initial configurations loaded dynamically
        self.pack_author = ""
        self.author_website = ""
        self.pack_version = "1.0.0"
        self.rating = "teen"
        self.tags_str = ""
        self.curation_type = "light-curation"
        self.art_type = "ai-generated"
        self.use_cover = True
        self.max_dimension = 1024
        self.webp_quality = 85
        self.global_pack_description = ""
        
        self.progress_value = 0.0
        self.progress_label = "Ready to package"
        self.logs = []
        self.is_running = False
        
        # Non-blocking telemetry state caches
        self.telemetry_loaded = False
        self.book_durations = {}          # {book_id: float}
        self.book_audio_files = {}        # {book_id: Optional[str]}
        self.selected_books = {}          # {book_id: bool}
        self.book_stats = {}              # {book_id: dict}
        self.book_metadata_overrides = {}  # {book_id: dict}
        
        self.load_packager_settings()

    def load_packager_settings(self):
        # 1. Fetch Global Baseline defaults from settings table to avoid hardcoding personal details
        with Session(engine) as session:
            db_setting = session.exec(select(Setting).where(Setting.key == "packager_defaults")).first()
            if db_setting:
                try:
                    defaults = json.loads(db_setting.value)
                    self.pack_author = defaults.get("pack_author", "")
                    self.author_website = defaults.get("author_website", "")
                    self.rating = defaults.get("rating", "teen")
                    self.curation_type = defaults.get("curation_type", "light-curation")
                    self.art_type = defaults.get("art_type", "ai-generated")
                    self.tags_str = defaults.get("tags_str", "")
                    self.global_pack_description = defaults.get("global_pack_description", "")
                except Exception:
                    pass

        # 2. Extract Project Level Overrides
        settings_path = get_project_settings_path(self.project_name)
        if settings_path.exists():
            try:
                with open(settings_path, "r") as f:
                    data = json.load(f)
                    self.pack_author = data.get("pack_author", self.pack_author)
                    self.author_website = data.get("author_website", self.author_website)
                    self.pack_version = data.get("pack_version", self.pack_version)
                    self.rating = data.get("rating", self.rating)
                    self.tags_str = data.get("tags_str", self.tags_str)
                    self.curation_type = data.get("curation_type", self.curation_type)
                    self.art_type = data.get("art_type", self.art_type)
                    self.use_cover = data.get("use_cover", self.use_cover)
                    self.max_dimension = data.get("max_dimension", self.max_dimension)
                    self.webp_quality = data.get("webp_quality", self.webp_quality)
                    self.global_pack_description = data.get("global_pack_description", self.global_pack_description)
            except Exception:
                pass

    def save_packager_settings(self):
        # 1. Update Global database default settings for subsequent volumes
        defaults = {
            "pack_author": self.pack_author,
            "author_website": self.author_website,
            "rating": self.rating,
            "curation_type": self.curation_type,
            "art_type": self.art_type,
            "tags_str": self.tags_str,
            "global_pack_description": self.global_pack_description
        }
        with Session(engine) as session:
            db_setting = session.exec(select(Setting).where(Setting.key == "packager_defaults")).first()
            if not db_setting:
                db_setting = Setting(key="packager_defaults", value=json.dumps(defaults))
            else:
                db_setting.value = json.dumps(defaults)
            session.add(db_setting)
            session.commit()

        # 2. Update Project Level Configuration Settings
        settings_path = get_project_settings_path(self.project_name)
        data = {}
        if settings_path.exists():
            try:
                with open(settings_path, "r") as f:
                    data = json.load(f)
            except Exception:
                pass
        
        data.update({
            "pack_author": self.pack_author,
            "author_website": self.author_website,
            "pack_version": self.pack_version,
            "rating": self.rating,
            "tags_str": self.tags_str,
            "curation_type": self.curation_type,
            "art_type": self.art_type,
            "use_cover": self.use_cover,
            "max_dimension": self.max_dimension,
            "webp_quality": self.webp_quality,
            "global_pack_description": self.global_pack_description
        })
        
        try:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(settings_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def add_log(self, msg: str):
        self.logs.append(msg)
        if len(self.logs) > 100:
            self.logs.pop(0)
        if self.log_area:
            self.log_area.push(msg)

    def set_all_selection(self, books: list, selection_state: bool):
        """Helper to modify all qualified books selection flags and refresh the panel."""
        for b in books:
            stats = self.book_stats.get(b.id, {})
            rendered_count = stats.get("generated_images", 0)
            has_file = self.book_audio_files.get(b.id)
            
            if rendered_count > 0 and has_file:
                self.selected_books[b.id] = selection_state
            else:
                self.selected_books[b.id] = False
                
        self.volumes_list_panel.refresh()

    async def async_load_telemetry(self, project: Project, books: list):
        """Asynchronously loads timing, ID3 tags and caches stats from disk to keep UI instant."""
        from database.connection import get_setting
        
        with Session(engine) as session:
            base_output_dir = Path(get_setting("output_dir", "./output", session)).resolve()

        for book in books:
            duration_sec = await asyncio.to_thread(get_book_total_duration, project.name, book.name)
            self.book_durations[book.id] = duration_sec
            
            audio_file = await asyncio.to_thread(find_audiobook_file, project.name, book.name)
            self.book_audio_files[book.id] = audio_file.name if audio_file else None
            
            # Non-blocking fetch of book cached statistics
            stats = get_book_stats_cached(project.name, book.name)
            self.book_stats[book.id] = stats
            
            # Check for localized JSON overrides to avoid loss of metadata adjustments
            book_dir = base_output_dir / project.name / book.name
            metadata_file = book_dir / "packager_metadata.json"
            
            loaded_override = None
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        loaded_override = json.load(f)
                except Exception:
                    pass
            
            if loaded_override:
                self.book_metadata_overrides[book.id] = loaded_override
            else:
                # Extract ID3 Track Tags
                audio_meta = {"title": "", "artist": ""}
                if audio_file:
                    audio_meta = await asyncio.to_thread(get_audio_file_metadata, audio_file)
                    
                book_title = audio_meta.get("title") or book.name
                book_author = audio_meta.get("artist") or (project.name.split("-")[0].strip() if "-" in project.name else "Author Unknown")
                
                # Setup book overrides structure if not already compiled (using standard Mobile Mode defaults)
                self.book_metadata_overrides[book.id] = {
                    "book_title": book_title,
                    "book_author": book_author,
                    "pack_title": f"{book_title} Illuminations",
                    "pack_description": self.global_pack_description or f"Custom illumination pack for {book_title}.",
                    "pack_version": "1.0.0",
                    "default_variant_name": "Mobile Mode",
                    "default_variant_desc": "Portrait mode",
                    "include_desktop": True,
                    "desktop_variant_name": "Desktop Mode",
                    "desktop_variant_desc": "Landscape mode.",
                    "include_static": True,
                    "static_variant_name": "Static Mode",
                    "static_variant_desc": "No animations.",
                    "exclude_unapproved": False
                }
            
            rendered_count = stats.get("generated_images", 0)
            if rendered_count == 0 or not audio_file:
                self.selected_books[book.id] = False
            elif book.id not in self.selected_books:
                self.selected_books[book.id] = True
            
        self.telemetry_loaded = True
        self.volumes_list_panel.refresh()

    def update_override(self, book_id: int, key: str, value: Any, book_name: str):
        """Updates overrides and flushes instantly to disk to guarantee persistence."""
        if book_id in self.book_metadata_overrides:
            self.book_metadata_overrides[book_id][key] = value
            self.save_single_book_override_file(book_id, book_name)

    def save_single_book_override_file(self, book_id: int, book_name: str):
        """Saves current override states to packager_metadata.json inside target output directory."""
        from database.connection import get_setting
        base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
        book_dir = base_output_dir / self.project_name / book_name
        book_dir.mkdir(parents=True, exist_ok=True)
        
        override = self.book_metadata_overrides.get(book_id, {})
        metadata_file = book_dir / "packager_metadata.json"
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(override, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def apply_global_description_to_all_volumes(self, books: list):
        """Copies the global metadata description to each initialized volume's metadata file."""
        for book in books:
            if book.id in self.book_metadata_overrides:
                self.book_metadata_overrides[book.id]["pack_description"] = self.global_pack_description
                self.save_single_book_override_file(book.id, book.name)
        ui.notify("Global description propagated successfully!", type="positive")
        self.volumes_list_panel.refresh()

    def apply_author_to_all_volumes(self, books: list, source_book_id: int):
        """Propagates the author from a specific book volume to all other volumes in the active project."""
        if source_book_id not in self.book_metadata_overrides:
            return
            
        source_author = self.book_metadata_overrides[source_book_id].get("book_author", "")
        for book in books:
            if book.id in self.book_metadata_overrides:
                self.book_metadata_overrides[book.id]["book_author"] = source_author
                self.save_single_book_override_file(book.id, book.name)
                
        ui.notify(f"Author '{source_author}' applied to all project volumes!", type="positive")
        self.volumes_list_panel.refresh()

    async def execute_packager(self, project: Project, books: list):
        if self.is_running:
            return
            
        client = ui.context.client

        # Gather selected book IDs directly from the local UI selection state
        selected_book_ids = [b_id for b_id, selected in self.selected_books.items() if selected]
        if not selected_book_ids:
            with client:
                ui.notify("Please select at least one valid, compiled volume to package.", type="warning")
            return
            
        self.is_running = True
        self.logs = []
        self.progress_value = 0.0
        self.progress_label = "Initializing packaging sequencing..."
        
        with client:
            self.action_buttons.refresh()
            self.progress_panel.refresh()

        self.save_packager_settings()

        parsed_tags = [t.strip() for t in self.tags_str.split(",") if t.strip()]
        metadata_config = {
            "pack_author": self.pack_author,
            "author_website": self.author_website,
            "art_type": self.art_type,
            "curation_type": self.curation_type,
            "content_rating": self.rating,
            "tags": parsed_tags,
            "use_cover": self.use_cover,
            "max_dimension": int(self.max_dimension),
            "webp_quality": int(self.webp_quality),
            "pack_creation_date": time.strftime("%Y-%m-%d")
        }

        # Query fresh, session-bound data blocks to guarantee no DetachedInstanceError raises
        books_data = []
        with Session(engine) as session:
            db_books = session.exec(select(Book).where(Book.id.in_(selected_book_ids))).all()
            for b in db_books:
                books_data.append({
                    "id": b.id,
                    "name": b.name,
                    "path": b.path
                })

        loop = asyncio.get_running_loop()

        def progress_tracker(p: float, msg: str):
            def update():
                with client:
                    self.progress_value = p
                    self.progress_label = msg
                    self.add_log(f"-> {msg}")
            loop.call_soon_threadsafe(update)

        try:
            for idx, b_data in enumerate(books_data):
                book_id = b_data["id"]
                book_name = b_data["name"]
                
                with client:
                    self.add_log(f"=== Starting Packager for Volume: '{book_name}' ({idx + 1}/{len(books_data)}) ===")
                
                overrides = self.book_metadata_overrides.get(book_id, {})
                book_metadata = metadata_config.copy()
                book_metadata.update(overrides)
                
                zip_result = await asyncio.to_thread(
                    build_illumination_pack,
                    project.name,
                    book_name,
                    book_metadata,
                    progress_tracker
                )
                
                with client:
                    self.add_log("SUCCESS: Package successfully generated!")
                    self.add_log(f"Saved directly to: {zip_result}\n")
                
                # Persist final status to SQLite via fresh local session
                with Session(engine) as session:
                    db_book = session.get(Book, book_id)
                    if db_book:
                        db_book.status = "Finished"
                        session.add(db_book)
                        session.commit()
                
            with client:
                self.progress_value = 1.0
                self.progress_label = "Batch packaging sequence finished successfully!"
                ui.notify("Illumination packaging finished successfully!", type="positive")
            
        except Exception as e:
            with client:
                self.add_log(f"FATAL ERROR DURING PACKAGING: {str(e)}")
                ui.notify(f"Packaging failed: {str(e)}", type="negative")
            
        finally:
            self.is_running = False
            with client:
                self.action_buttons.refresh()
                self.progress_panel.refresh()

    @ui.refreshable
    def progress_panel(self):
        with ui.column().classes('w-full bg-slate-50 border p-4 rounded-xl gap-3'):
            with ui.row().classes('w-full justify-between items-center text-xs font-bold text-slate-500'):
                ui.label(self.progress_label).classes('text-blue-600')
                ui.label(f"{int(self.progress_value * 100)}%")
            ui.linear_progress(value=self.progress_value).classes('w-full h-2 rounded-full')

    @ui.refreshable
    def action_buttons(self, project=None, books=None):
        with ui.row().classes('w-full justify-end mt-4'):
            ui.button(
                'Build Illumination Packs',
                icon='inventory_2',
                on_click=lambda: self.execute_packager(project, books)
            ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold').props(f'loading={self.is_running}')

    @ui.refreshable
    def volumes_list_panel(self, project: Project, books: list):
        with ui.column().classes('w-full bg-white border rounded-xl p-5 gap-3 shadow-sm'):
            ui.label('Project Volumes Telemetry').classes('text-xs font-black uppercase tracking-wider text-slate-400')
            
            if not self.telemetry_loaded:
                with ui.row().classes('w-full justify-center items-center p-6 gap-2'):
                    ui.spinner(size='sm', color='blue')
                    ui.label('Reading timing duration metadata & ID3 tags...').classes('text-xs text-slate-500 animate-pulse')
                return

            # Selection Helpers
            with ui.row().classes('gap-2 mb-1'):
                ui.button('Select All', on_click=lambda: self.set_all_selection(books, True)).props('flat dense').classes('text-[10px] text-blue-600 font-bold')
                ui.button('Deselect All', on_click=lambda: self.set_all_selection(books, False)).props('flat dense').classes('text-[10px] text-slate-500 font-bold')

            for idx, book in enumerate(books):
                if book.id not in self.selected_books:
                    self.selected_books[book.id] = False

                stats = self.book_stats.get(book.id, {})
                rendered_count = stats.get("generated_images", 0)
                total_scenes = stats.get("total_prompts", 0) or stats.get("estimated_scenes", 1)
                
                has_file = self.book_audio_files.get(book.id)
                duration_sec = self.book_durations.get(book.id) or 0.0
                
                # Dynamic Pre-flight Validations
                can_package = rendered_count > 0
                is_complete = rendered_count >= total_scenes if total_scenes > 0 else False
                
                if not has_file:
                    duration_text = f"Audiobook path missing on disk: {book.path}"
                    duration_color = "border-red-200 bg-red-50/50"
                    duration_icon = "error"
                    can_package = False
                    self.selected_books[book.id] = False
                elif rendered_count == 0:
                    duration_text = "Pre-flight Blocked: No images rendered. Render frames first before packaging."
                    duration_color = "border-slate-200 bg-slate-100 text-slate-400"
                    duration_icon = "block"
                    self.selected_books[book.id] = False
                elif not is_complete:
                    h = int(duration_sec // 3600)
                    m = int((duration_sec % 3600) // 60)
                    duration_text = f"Pre-flight Warning: Incomplete ({rendered_count}/{total_scenes} rendered) • {h:02d}h {m:02d}m total duration"
                    duration_color = "border-amber-200 bg-amber-50/50 text-amber-800"
                    duration_icon = "warning"
                else:
                    h = int(duration_sec // 3600)
                    m = int((duration_sec % 3600) // 60)
                    duration_text = f"Verified: Complete ({rendered_count}/{total_scenes} rendered) • {h:02d}h {m:02d}m total duration"
                    duration_color = "border-emerald-200 bg-emerald-50/50 text-emerald-800"
                    duration_icon = "check_circle"

                # Expandable Book detail card
                with ui.expansion().classes(f'w-full border rounded-lg p-1 transition-all {duration_color}') as book_exp:
                    with book_exp.add_slot('header'):
                        with ui.row().classes('items-center justify-between w-full pr-4'):
                            with ui.row().classes('items-center gap-3 flex-1 min-w-0'):
                                ui.checkbox(
                                    value=self.selected_books[book.id],
                                    on_change=lambda e, b_id=book.id: self.selected_books.__setitem__(b_id, e.value)
                                ).props(f'disable={not can_package}').classes('mr-1')
                                
                                ui.icon(duration_icon, size='18px')
                                with ui.column().classes('gap-0.5 flex-1 min-w-0'):
                                    ui.label(book.name).classes('text-xs font-bold leading-none text-slate-800 truncate')
                                    ui.label(duration_text).classes('text-[10px] font-medium leading-none truncate')
                    
                    if book.id in self.book_metadata_overrides:
                        override = self.book_metadata_overrides[book.id]
                        
                        with ui.column().classes('w-full p-4 bg-white border-t gap-4'):
                            ui.label('Book & Pack Identification Overrides').classes('text-[10px] font-bold text-slate-400 tracking-wider uppercase')
                            
                            with ui.grid(columns=2).classes('w-full gap-4'):
                                ui.input(
                                    'Book Title', 
                                    value=override["book_title"], 
                                    on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "book_title", e.value, b_name)
                                ).classes('w-full').props('dense outlined')
                                
                                with ui.row().classes('w-full items-center gap-2'):
                                    ui.input(
                                        'Book Author', 
                                        value=override["book_author"], 
                                        on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "book_author", e.value, b_name)
                                    ).classes('flex-1').props('dense outlined')
                                    with ui.button(
                                        icon='reply_all',
                                        on_click=lambda _, b_id=book.id: self.apply_author_to_all_volumes(books, b_id)
                                    ).props('flat dense').classes('text-blue-600'):
                                        ui.tooltip('Apply this author to all volumes')
                            
                            with ui.grid(columns=2).classes('w-full gap-4'):
                                ui.input(
                                    'Pack Title', 
                                    value=override["pack_title"], 
                                    on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "pack_title", e.value, b_name)
                                ).classes('w-full').props('dense outlined')
                                
                                ui.input(
                                    'Pack Version', 
                                    value=override["pack_version"], 
                                    on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "pack_version", e.value, b_name)
                                ).classes('w-full').props('dense outlined')

                            ui.textarea(
                                'Pack Description Override', 
                                value=override["pack_description"], 
                                on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "pack_description", e.value, b_name)
                            ).classes('w-full').props('dense outlined rows=2')

                            ui.separator()

                            ui.label('Primary/Default Mobile Variant').classes('text-[10px] font-bold text-slate-400 tracking-wider uppercase')
                            with ui.grid(columns=2).classes('w-full gap-4'):
                                ui.input(
                                    'Display Name', 
                                    value=override["default_variant_name"], 
                                    on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "default_variant_name", e.value, b_name)
                                ).classes('w-full').props('dense outlined')
                                
                                ui.input(
                                    'Description', 
                                    value=override["default_variant_desc"], 
                                    on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "default_variant_desc", e.value, b_name)
                                ).classes('w-full').props('dense outlined')

                            ui.separator()

                            ui.label('Alternate Viewing Experience Variants').classes('text-[10px] font-bold text-slate-400 tracking-wider uppercase')
                            
                            with ui.row().classes('w-full gap-4 items-start'):
                                with ui.column().classes('flex-1 border p-3 rounded-lg bg-slate-50 gap-2'):
                                    desktop_cb = ui.checkbox(
                                        "Include Desktop (Landscape) Variant", 
                                        value=override["include_desktop"],
                                        on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "include_desktop", e.value, b_name)
                                    ).classes('text-xs font-bold text-slate-700')
                                    
                                    desktop_name_inp = ui.input(
                                        'Variant Name', 
                                        value=override["desktop_variant_name"], 
                                        on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "desktop_variant_name", e.value, b_name)
                                    ).classes('w-full').props('dense outlined')
                                    desktop_name_inp.bind_visibility_from(desktop_cb, 'value')
                                    
                                    desktop_desc_inp = ui.textarea(
                                        'Variant Description', 
                                        value=override["desktop_variant_desc"], 
                                        on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "desktop_variant_desc", e.value, b_name)
                                    ).classes('w-full').props('dense outlined rows=1')
                                    desktop_desc_inp.bind_visibility_from(desktop_cb, 'value')

                                with ui.column().classes('flex-1 border p-3 rounded-lg bg-slate-50 gap-2'):
                                    static_cb = ui.checkbox(
                                        "Include Static (No Animation) Variant", 
                                        value=override["include_static"],
                                        on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "include_static", e.value, b_name)
                                    ).classes('text-xs font-bold text-slate-700')
                                    
                                    static_name_inp = ui.input(
                                        'Variant Name', 
                                        value=override["static_variant_name"], 
                                        on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "static_variant_name", e.value, b_name)
                                    ).classes('w-full').props('dense outlined')
                                    static_name_inp.bind_visibility_from(static_cb, 'value')
                                    
                                    static_desc_inp = ui.textarea(
                                        'Variant Description', 
                                        value=override["static_variant_desc"], 
                                        on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "static_variant_desc", e.value, b_name)
                                    ).classes('w-full').props('dense outlined rows=1')
                                    static_desc_inp.bind_visibility_from(static_cb, 'value')

                            ui.separator()
                            
                            ui.label('Image Curation Filters').classes('text-[10px] font-bold text-slate-400 tracking-wider uppercase')
                            with ui.row().classes('w-full gap-4'):
                                ui.checkbox(
                                    'Only include "Approved" images (exclude pending/unapproved scenes)', 
                                    value=override.get("exclude_unapproved", False),
                                    on_change=lambda e, b_id=book.id, b_name=book.name: self.update_override(b_id, "exclude_unapproved", e.value, b_name)
                                ).classes('text-xs font-medium text-slate-700')

    def render(self, project: Project, books: list):
        if not self.telemetry_loaded and not self.is_running:
            asyncio.create_task(self.async_load_telemetry(project, books))

        with ui.column().classes('w-full gap-6'):
            with ui.row().classes('w-full items-center justify-between'):
                with ui.column().classes('gap-1'):
                    ui.label('Direct OIS Packager Studio').classes('text-lg font-bold text-slate-800')
                    ui.label('Configure, optimize and package synchronized visual OIS 1.3 Zip archives directly next to original audiobook formats.').classes('text-xs text-slate-500')

            with ui.grid(columns='1fr 1fr').classes('w-full gap-6 items-start'):
                with ui.column().classes('bg-white border rounded-xl p-5 gap-4 shadow-sm w-full'):
                    ui.label('OIS Manifest Global Metadata').classes('text-xs font-black uppercase tracking-wider text-slate-400 mb-2')
                    
                    with ui.grid(columns=2).classes('w-full gap-4'):
                        ui.input(
                            'Pack Author', 
                            value=self.pack_author, 
                            placeholder='Your name or alias',
                            on_change=lambda e: setattr(self, 'pack_author', e.value)
                        ).classes('w-full').props('dense outlined')
                        
                        ui.input(
                            'Pack Version', 
                            value=self.pack_version, 
                            on_change=lambda e: setattr(self, 'pack_version', e.value)
                        ).classes('w-full').props('dense outlined')
                    
                    ui.input(
                        'Author Website URL', 
                        value=self.author_website, 
                        placeholder='Portfolio, social media, etc.',
                        on_change=lambda e: setattr(self, 'author_website', e.value)
                    ).classes('w-full').props('dense outlined')

                    ui.input(
                        'Tags (comma-separated)', 
                        value=self.tags_str, 
                        placeholder='Add tags (e.g., cinematic, dark)',
                        on_change=lambda e: setattr(self, 'tags_str', e.value)
                    ).classes('w-full').props('dense outlined')

                    with ui.grid(columns=3).classes('w-full gap-4'):
                        ui.select(
                            options={
                                'all-ages': 'All Ages', 
                                'teen': 'Teen', 
                                'mature': 'Mature'
                            }, 
                            value=self.rating, 
                            label='Content Rating',
                            on_change=lambda e: setattr(self, 'rating', e.value)
                        ).classes('w-full').props('dense outlined')
                        
                        ui.select(
                            options={
                                'random-order': 'Random Order',
                                'bulk-import': 'Bulk Import',
                                'bulk-import-by-chapter': 'Bulk Import by Chapter',
                                'light-curation': 'Light Curation',
                                'full-curation': 'Full Curation'
                            }, 
                            value=self.curation_type, 
                            label='Curation Level',
                            on_change=lambda e: setattr(self, 'curation_type', e.value)
                        ).classes('w-full').props('dense outlined')

                        ui.select(
                            options={
                                'ai-generated': 'AI Generated',
                                'fan-art': 'Fan Art',
                                'official-art': 'Official Art',
                                'screen-cap': 'Screen Caps',
                                'mixed-ai': 'Mixed Including AI',
                                'mixed-no-ai': 'Mixed No AI'
                            }, 
                            value=self.art_type, 
                            label='Art Type',
                            on_change=lambda e: setattr(self, 'art_type', e.value)
                        ).classes('w-full').props('dense outlined')

                    # Global Series Pack Description with inline propagation
                    with ui.column().classes('w-full gap-2 mt-2'):
                        ui.label('Global Pack Description (Series Baseline)').classes('text-xs font-semibold text-slate-700')
                        ui.textarea(
                            value=self.global_pack_description,
                            placeholder='Baseline description applicable across all books or the whole series...',
                            on_change=lambda e: setattr(self, 'global_pack_description', e.value)
                        ).classes('w-full').props('dense outlined rows=2')
                        
                        ui.button(
                            'Apply Description to All Volumes',
                            icon='arrow_downward',
                            on_click=lambda: self.apply_global_description_to_all_volumes(books)
                        ).classes('w-full bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs py-1.5')

                with ui.column().classes('bg-white border rounded-xl p-5 gap-4 shadow-sm w-full'):
                    ui.label('Asset Downscaling & Custom Overrides').classes('text-xs font-black uppercase tracking-wider text-slate-400 mb-2')
                    
                    with ui.column().classes('w-full gap-2'):
                        ui.checkbox(
                            'Inject audiobook cover art as first index keyframe (00:00:00.00)', 
                            value=self.use_cover,
                            on_change=lambda e: setattr(self, 'use_cover', e.value)
                        ).classes('text-xs font-medium text-slate-700')
                        
                        ui.label('If enabled, the cover image found next to your project volume is compressed as 0000_cover.webp and injected dynamically with gentle zoom transitions.').classes('text-[10px] text-slate-400 pl-8 leading-snug')

                    ui.separator()

                    with ui.row().classes('w-full items-center gap-4'):
                        ui.number(
                            'Max Width/Height (px)', 
                            value=self.max_dimension, 
                            on_change=lambda e: setattr(self, 'max_dimension', e.value)
                        ).classes('w-28').props('dense outlined suffix="px"')
                        
                        ui.slider(
                            min=50, max=100, step=5, 
                            value=self.webp_quality,
                            on_change=lambda e: setattr(self, 'webp_quality', e.value)
                        ).classes('flex-1')
                        ui.label('').bind_text_from(self, 'webp_quality', backward=lambda q: f"WebP Quality: {q}%").classes('text-xs text-slate-500 font-bold')

            # Expandable volumes list panel
            self.volumes_list_panel(project, books)

            self.progress_panel()

            with ui.column().classes('w-full bg-slate-900 border rounded-xl p-4 gap-2'):
                ui.label('Packaging Studio Terminal Output').classes('text-[10px] font-bold text-slate-400 tracking-wider uppercase')
                self.log_area = ui.log(max_lines=100).classes(
                    'w-full font-mono text-[10px] text-emerald-400 bg-slate-900 border-none h-48'
                ).style('color: #34d399 !important; background-color: #0f172a !important;')
                
                if self.logs:
                    for log in self.logs:
                        self.log_area.push(log)

            self.action_buttons(project, books)