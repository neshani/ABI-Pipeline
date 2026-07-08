import asyncio
from typing import Optional, Callable, List, Any
from nicegui import ui
from sqlmodel import Session, select
from database.connection import engine
from database.models import Character, CharacterAlias, Project, Book

class PromptAutocompleteManager:
    def __init__(
        self, 
        textarea: ui.textarea, 
        project_id: int, 
        on_change_callback: Optional[Callable[[str], Any]] = None
    ):
        self.textarea = textarea
        self.project_id = project_id
        self.on_change_callback = on_change_callback
        
        # Suggestions state
        self.active_matches: List[dict] = []
        self.selected_index = 0
        self.last_bracket = -1
        self.cursor_pos = -1
        
        # Load database characters and aliases cache
        self.characters = self._load_characters()
        
        # Ensure target element behaves as a relative reference container for the overlay
        self.textarea.classes('relative')
        
        # Absolute-positioned floating suggestion card within the element
        with self.textarea:
            # Hide it explicitly via style so it doesn't shift parent layout on mount
            self.popup = ui.card().classes(
                'absolute z-[200] bg-white border border-slate-200 shadow-xl max-h-48 overflow-y-auto p-1.5 rounded-lg w-full bottom-full mb-1 left-0'
            ).style('min-width: 250px; display: none;')
            with self.popup:
                self.results_container = ui.column().classes('w-full gap-0.5')

        # Capture-phase native keydown interceptor directly on the native <textarea>
        # This prevents Quasar's keydown handler from seeing Tab/Enter before we can prevent defaults
        ui.run_javascript(f'''
            (() => {{
                const setupCapture = () => {{
                    const el = getElement({self.textarea.id});
                    if (el) {{
                        const textarea = el.$refs.qRef.getNativeElement();
                        if (textarea) {{
                            textarea.addEventListener('keydown', (e) => {{
                                const popup = document.getElementById("c{self.popup.id}");
                                const popupVisible = popup && popup.style.display !== "none";
                                if (popupVisible && ["ArrowUp", "ArrowDown", "Enter", "Tab", "Escape"].includes(e.key)) {{
                                    e.preventDefault();
                                }}
                            }}, true); // true targets the capture phase
                            return;
                        }}
                    }}
                    setTimeout(setupCapture, 50);
                }};
                setupCapture();
            }})();
        ''')

        # Register keyup/click trackers to detect if the cursor enters brackets
        self.textarea.on(
            'keyup', 
            self._handle_input_change, 
            js_handler='(e) => emit(e.target.selectionStart, e.target.value)'
        )
        self.textarea.on(
            'click', 
            self._handle_input_change, 
            js_handler='(e) => emit(e.target.selectionStart, e.target.value)'
        )
        
        # Close on blur safely
        self.textarea.on('blur', self._handle_blur)

        # Intercept bubbling keyboard events to trigger Python matching logic
        self.textarea.on(
            'keydown', 
            self._handle_keydown, 
            js_handler=f'''(e) => {{
                const popup = document.getElementById("c{self.popup.id}");
                const popupVisible = popup && popup.style.display !== "none";
                if (popupVisible && ["ArrowUp", "ArrowDown", "Enter", "Tab", "Escape"].includes(e.key)) {{
                    e.preventDefault();
                    e.stopPropagation();
                    emit(e.key);
                }}
            }}'''
        )

    def refresh_characters(self):
        """Re-reads character data from the database."""
        self.characters = self._load_characters()

    def _handle_input_change(self, e):
        if not e.args or len(e.args) < 2:
            self._hide_popup()
            return
            
        cursor_pos = e.args[0]
        val = e.args[1] or ""
        
        # Trigger live, instant character chip drawing on every single manual keystroke
        if self.on_change_callback:
            if asyncio.iscoroutinefunction(self.on_change_callback):
                asyncio.create_task(self.on_change_callback(val))
            else:
                self.on_change_callback(val)
        
        text_before_cursor = val[:cursor_pos]
        last_bracket = text_before_cursor.rfind('[')
        last_close = text_before_cursor.rfind(']')
        
        if last_bracket != -1 and last_bracket > last_close:
            query = text_before_cursor[last_bracket + 1:].strip().lower()
            
            self.last_bracket = last_bracket
            self.cursor_pos = cursor_pos
            
            matches = []
            for char in self.characters:
                if not query or any(query in a for a in char["aliases"]):
                    matches.append(char)
                    
            # Smart Priority Sorting Key:
            # 1. Matches starting with the prefix go first.
            # 2. Sort descending by project popularity ("hits").
            # 3. Sort alphabetically as a tie-breaker.
            def sort_key(char):
                if not query:
                    return (-char.get("hits", 0), char["name"].lower())
                
                starts_with = any(a.startswith(query) for a in char["aliases"])
                starts_with_val = -1 if starts_with else 0
                hits_val = -char.get("hits", 0)
                return (starts_with_val, hits_val, char["name"].lower())

            matches.sort(key=sort_key)
                    
            self.active_matches = matches[:5]
            if self.active_matches:
                if self.selected_index >= len(self.active_matches):
                    self.selected_index = 0
                self._render_suggestions()
                self.popup.style('display: block;')
            else:
                self._hide_popup()
        else:
            self._hide_popup()

    def _render_suggestions(self):
        self.results_container.clear()
        with self.results_container:
            ui.label('Suggestions (Arrows / Enter / Tab)').classes(
                'text-[8px] font-black text-slate-400 tracking-wider mb-1 px-2 py-0.5 select-none uppercase border-b pb-1 border-slate-100'
            )
            for idx, match in enumerate(self.active_matches):
                is_active = idx == self.selected_index
                bg_class = 'bg-blue-50 text-blue-800 font-bold shadow-2xs' if is_active else 'hover:bg-slate-50 text-slate-600'
                
                btn = ui.button(
                    f"👤 {match['name']}", 
                    on_click=lambda _, m=match: self.select_match(m)
                ).props('flat dense align=left').classes(
                    f'text-xs w-full py-1.5 px-2.5 rounded justify-start transition-all {bg_class}'
                )
                # Prevent focus theft on mousedown to stop caret from vanishing
                btn.on('mousedown', js_handler='(e) => e.preventDefault()')

    def _handle_keydown(self, e):
        key = e.args[0] if isinstance(e.args, list) and e.args else e.args
        if not self.active_matches:
            return
            
        if key == 'ArrowDown':
            self.selected_index = (self.selected_index + 1) % len(self.active_matches)
            self._render_suggestions()
        elif key == 'ArrowUp':
            self.selected_index = (self.selected_index - 1 + len(self.active_matches)) % len(self.active_matches)
            self._render_suggestions()
        elif key in ('Enter', 'Tab'):
            if 0 <= self.selected_index < len(self.active_matches):
                self.select_match(self.active_matches[self.selected_index])
        elif key == 'Escape':
            self._hide_popup()

    async def _handle_blur(self):
        # Delay hiding to allow active list selection click events to register first
        await asyncio.sleep(0.2)
        self._hide_popup()

    def _hide_popup(self):
        self.popup.style('display: none;')
        self.active_matches.clear()

    def select_match(self, match: dict):
        val = self.textarea.value or ""
        before_bracket = val[:self.last_bracket]
        after_cursor = val[self.cursor_pos:]
        
        close_bracket_idx = after_cursor.find(']')
        next_open = after_cursor.find('[')
        if close_bracket_idx != -1 and (next_open == -1 or close_bracket_idx < next_open):
            after_cursor = after_cursor[close_bracket_idx + 1:].lstrip()
            
        completed_text = before_bracket + f"[{match['name']}]" + after_cursor
        new_cursor_pos = self.last_bracket + len(match['name']) + 2
        
        self._hide_popup()
        
        # 1. Update the Python/NiceGUI textarea value natively (Quasar component updates natively)
        self.textarea.value = completed_text
        
        # 2. Cascading cursor position enforcer using NiceGUI's native element retriever
        ui.run_javascript(f'''
            (() => {{
                const setCursor = () => {{
                    const el = getElement({self.textarea.id});
                    if (el) {{
                        const textarea = el.$refs.qRef.getNativeElement();
                        if (textarea) {{
                            if (document.activeElement !== textarea) {{
                                textarea.focus();
                            }}
                            textarea.setSelectionRange({new_cursor_pos}, {new_cursor_pos});
                        }}
                    }}
                }};
                
                // Repeatedly enforce position over successive frames to capture Vue microtasks
                setCursor();
                setTimeout(setCursor, 10);
                setTimeout(setCursor, 30);
                setTimeout(setCursor, 60);
                setTimeout(setCursor, 120);
                setTimeout(setCursor, 250);
            }})();
        ''')
        
        # 3. Fire Python-side scene chips callback instantly with completed text
        if self.on_change_callback:
            if asyncio.iscoroutinefunction(self.on_change_callback):
                asyncio.create_task(self.on_change_callback(completed_text))
            else:
                self.on_change_callback(completed_text)

    def _load_characters(self) -> List[dict]:
        with Session(engine) as session:
            project = session.get(Project, self.project_id)
            if not project:
                return []
                
            # Fetch all books in the project to calculate global frequencies
            books = session.exec(select(Book).where(Book.project_id == self.project_id)).all()
            
            frequencies = {}
            try:
                from ui.pages.project.characters_tab import get_character_frequency_map
                frequencies = get_character_frequency_map(project.name, books)
            except Exception as e:
                print(f"[Autocomplete] Could not load character frequency map: {e}")
                
            chars = session.exec(select(Character).where(Character.project_id == self.project_id)).all()
            data = []
            for c in chars:
                aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == c.id)).all()
                alias_list = [a.alias.lower() for a in aliases]
                if c.name.lower() not in alias_list:
                    alias_list.append(c.name.lower())
                
                # Sum the total hits for all alias mentions in this project [3.5]
                total_hits = sum(frequencies.get(alias, 0) for alias in alias_list)
                
                data.append({
                    "id": c.id,
                    "name": c.name,
                    "aliases": alias_list,
                    "hits": total_hits
                })
            return data