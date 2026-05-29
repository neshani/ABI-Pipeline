from typing import Optional, Dict, Any, List

# Dynamic Workspace State
active_project_id: Optional[int] = None
active_book_id: Optional[int] = None

# Tab selections (bound to persist state during navigation)
active_project_tab: str = 'Dashboard'
active_book_tab: str = 'Dashboard'

# Directory filters
search_query: str = ""
selected_project_type: str = "All"

# Scan context
current_scan_result: Optional[Dict[str, Any]] = None
scan_error: str = ""
custom_project_name_value: str = ""

# --- Stable UI Binding Stores (Dictionaried mapped by ID) ---
project_status: str = "Imported"
books_progress: Dict[int, float] = {}  # {book_id: float}
books_status: Dict[int, str] = {}      # {book_id: status}
books_subtitle: Dict[int, str] = {}    # {book_id: "status • percentage%"}

# --- Stable Live Logger Tracker ---
console_logs: List[str] = [
    "[ABI-Pipeline] System initialized.",
    "[ABI-Pipeline] Listening for background pipeline orchestration events..."
]
active_log_widget: Optional[Any] = None
logs_pushed_index: int = 0

def add_console_log(message: str):
    console_logs.append(message)
    if len(console_logs) > 500:
        console_logs.pop(0)

# --- Playgrounds & Prompt Settings bindings ---
playground_template: str = ""
playground_selected_template: str = "default"
playground_book_selection: Optional[str] = None
playground_chunk_count: int = 1
playground_start_index: int = 30           # Defaulting to 30 to skip intros
playground_seed: int = 42                 # Seeded random default
playground_selection_mode: str = "Seeded Random"
playground_loading: bool = False
playground_results: List[Dict[str, Any]] = []  # [{"chunk": "", "quote": "", "prompt": "", "status": ""}]