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
needs_restart: bool = False  # Set to True when native binaries are installed on-the-fly

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
was_manually_cancelled: bool = False  # Track user-triggered process stops
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
    print(message, flush=True)  # Mirror directly to standard Python system console
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
comfy_online: bool = False
style_selected_preset: str = "retro_graphic_novel"
style_preset_save_name: str = ""
style_selected_workflow: str = ""
style_prompt_prefix: str = "space opera adventure graphic novel illustration, sharp ink sketch, crisp outlines, retro-futuristic sci-fi aesthetic, cosmic wonder, detailed, "
style_prompt_suffix: str = ", high-contrast shadows, selective color accents, bold ink-wash shading"
style_negative_prompt: str = "blurry, bad quality, text, watermark, photorealistic, photography, dystopian, gritty, grimy, decay, cyberpunk"
style_test_prompts: List[Dict[str, Any]] = []
style_test_images: List[Optional[str]] = []  # List of Base64-encoded strings or None
style_test_seeds: List[int] = []
style_lock_samples: bool = False
style_playground_loading: bool = False
style_discovered_params: Dict[str, Any] = {}  # Dynamic structures found by introspection
style_workflow_overrides: Dict[str, Any] = {}  # {node_id: {field_name: value}}
style_contact_sheet_overlay: str = ""  # Custom text banner drawn bottom-center of the sheet

# Seeds and controls for Prompt Playground matching
style_prompt_seed: int = 42
style_image_seed: int = 42
style_use_random_image_seed: bool = True
style_chunk_count: int = 4
style_has_source_material: bool = False

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

# --- GPU / VRAM NVML Telemetry State ---
gpu_telemetry_supported: bool = False
gpu_name: str = ""
gpu_utilization: int = 0
gpu_vram_used: float = 0.0      # GB used
gpu_vram_total: float = 0.0     # GB total
gpu_vram_pct: float = 0.0       # 0.0 to 1.0
gpu_temp: int = 0               # °C
gpu_power_used: float = 0.0     # Watts
gpu_power_limit: float = 0.0    # Watts

# --- Active Workspace Event Subscriptions (Prevention of Timer/Keyboard Leaks) ---
book_scroll_timer: Optional[Any] = None
book_update_timer: Optional[Any] = None
book_keyboard: Optional[Any] = None