from typing import Optional, Dict, Any, List

# Dynamic Workspace State
active_project_id: Optional[int] = None
active_book_id: Optional[int] = None
active_tool: Optional[str] = None  # Tracks global tools like 'lora_contact_sheet'
lora_tool_selected_workflow: str = "None"
lora_library: List[Dict[str, Any]] = []  # Holds the parsed loras.csv
lora_tool_active_lora_id: Optional[str] = None
lora_tool_generating: bool = False
lora_tool_cancel_flag: bool = False
lora_tool_progress: Dict[str, Any] = {}  # Tracks {"lora_id": str, "current": int, "total": int}

# Tab selections (bound to persist state during navigation)
active_project_tab: str = 'Dashboard'
active_book_tab: str = 'Dashboard'

# Directory filters / sorting
search_query: str = ""
selected_sort: str = "Most Recent"
expanded_projects: set[int] = set()

# Scan context
current_scan_result: Optional[Dict[str, Any]] = None
scan_error: str = ""
custom_project_name_value: str = ""

# Multi-format import variables
selected_txt_files: List[str] = []
selected_epub_files: List[str] = []
import_project_name: str = ""

# --- Stable UI Binding Stores (Dictionaried mapped by ID) ---
project_status: str = "Imported"
project_progress: float = 0.0
project_progress_label: str = "Batch Progress (0%)"
books_progress: Dict[int, float] = {}  # {book_id: float}
books_status: Dict[int, str] = {}      # {book_id: status}
books_subtitle: Dict[int, str] = {}    # {book_id: "status â€¢ percentage%"}

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
playground_chunk_size: int = 350           # Defines how many words make up a scene chunk
playground_start_index: int = 30           # Defaulting to 30 to skip intros
playground_seed: int = 42                 # Seeded random default
playground_selection_mode: str = "Seeded Random"
playground_loading: bool = False
playground_results: List[Dict[str, Any]] = []  # [{"chunk": "", "quote": "", "prompt": "", "status": ""}]

# --- Dynamic Prompt Generation Task States ---
recent_prompts: List[Dict[str, Any]] = []  # Stores last 5 generated prompts: [{"book": "", "chapter": 1, "scene": 1, "prompt": "", "quote": "", "status": ""}]
prompt_gen_active: bool = False
cancel_prompt_gen_flag: bool = False

# --- Dynamic Image Generation Task States ---
image_gen_active: bool = False
cancel_image_gen_flag: bool = False

# --- Style Playground & Workflow Analyzer Bindings ---
style_selected_preset: str = "default"
style_selected_workflow: str = ""
style_prompt_prefix: str = "ArsMJStyle, 1890s Victorian illustration, detailed pen and ink with soft watercolor wash, Sidney Paget style. "
style_prompt_suffix: str = ""
style_negative_prompt: str = "blurry, bad quality, text, watermark, photorealistic, photography"
style_test_prompts: List[Dict[str, Any]] = []
style_test_images: List[Optional[str]] = []  # List of Base64-encoded strings or None
style_test_seeds: List[int] = []
style_lock_samples: bool = False
style_playground_loading: bool = False
style_discovered_params: Dict[str, Any] = {}  # Dynamic structures found by introspection
style_workflow_overrides: Dict[str, Any] = {}  # {node_id: {field_name: value}}

# Seeds and controls for Prompt Playground matching
style_prompt_seed: int = 42
style_image_seed: int = 42
style_use_random_image_seed: bool = True
style_chunk_count: int = 4

# --- Live Rendered Images Feed States ---
recent_rendered_images: List[Dict[str, Any]] = []  # [{"filename": "", "base64": "", "chapter": 1, "scene": 1, "quote": "", "prompt": ""}]
recent_images_refresh: Optional[Any] = None

# --- Persistent Image Preview Dialog Bindings ---
preview_image_src: str = ""
preview_image_title: str = ""
global_preview_dialog: Optional[Any] = None

# --- Cache-Busted On-Disk Volume Statistics Engine ---
_stats_cache: Dict[str, Any] = {}
stats_refresh_callback: Optional[Any] = None

# --- Real-Time Batch Process Telemetry ---
batch_start_time: Optional[float] = None
batch_elapsed_sec: float = 0.0
batch_eta_label: str = "ETA: Estimating..."

