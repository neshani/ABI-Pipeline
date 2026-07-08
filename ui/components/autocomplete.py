import asyncio
from typing import Optional, Callable, List, Any
from nicegui import ui
from sqlmodel import Session, select
from database.connection import engine
from database.models import Character, CharacterAlias

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

        # Conditionally intercept keyboard events ONLY when suggestions are visible
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
                
                ui.button(
                    f"👤 {match['name']}", 
                    on_click=lambda _, m=match: self.select_match(m)
                ).props('flat dense align=left').classes(
                    f'text-xs w-full py-1.5 px-2.5 rounded justify-start transition-all {bg_class}'
                )

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
        
        # 1. Update the Python/NiceGUI textarea value natively
        self.textarea.value = completed_text
        
        # Escape string for JavaScript literal block injection
        js_text = completed_text.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('\r', '\\r')
        
        # 2. Focus and align cursor by polling the DOM until Vue completes its render
        ui.run_javascript(f'''
            (() => {{
                const expectedText = '{js_text}';
                let attempts = 0;
                
                const checkAndSetCursor = () => {{
                    const textarea = document.getElementById("c{self.textarea.id}").querySelector("textarea");
                    if (textarea) {{
                        if (textarea.value === expectedText) {{
                            textarea.focus();
                            textarea.setSelectionRange({new_cursor_pos}, {new_cursor_pos});
                        }} else if (attempts < 50) {{
                            attempts++;
                            setTimeout(checkAndSetCursor, 10);
                        }}
                    }}
                }};
                checkAndSetCursor();
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
            chars = session.exec(select(Character).where(Character.project_id == self.project_id)).all()
            data = []
            for c in chars:
                aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == c.id)).all()
                alias_list = [a.alias.lower() for a in aliases]
                if c.name.lower() not in alias_list:
                    alias_list.append(c.name.lower())
                data.append({
                    "id": c.id,
                    "name": c.name,
                    "aliases": alias_list
                })
            return data