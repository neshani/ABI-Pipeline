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
selected_book_paths: set[str] = set()

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

# Character prompt-replacement tuning. The old playground modal that exposed
# these was removed; the compiler reads its own defaults (all traits on,
# sentence structure on), so the getattr fallbacks at the call sites hold.

# --- Dynamic Prompt Generation Task States ---
recent_prompts: List[Dict[str, Any]] = []  # Stores last 5 generated prompts: [{"book": "", "chapter": 1, "shot": 1, "prompt": "", "quote": "", "status": ""}]
prompt_gen_active: bool = False
analysis_active: bool = False
cancel_prompt_gen_flag: bool = False

# --- Dashboard run model (2026-10) -------------------------------------------
# One task runs map -> cast -> prompts for a chosen set of volumes, so there is
# one Stop and one progress line. ``cancel_pipeline_flag`` is the stop signal the
# stages poll between units; it is separate from ``cancel_prompt_gen_flag``
# because the prompt phase already owns that one.
pipeline_active: bool = False
cancel_pipeline_flag: bool = False
# Whether this project builds character descriptions at all. Per project, saved
# in project_settings.json; off means the cast stages are skipped, not deleted -
# prompts are unaffected either way because tags compile at render time.
cast_enabled: bool = True
# How the `carried` trait reaches rendered prompts: "scene" (only where the text
# recorded it), "off" (never rendered, still stored) or "always" (carried
# forward, the old behaviour). Per project, saved in project_settings.json.
carried_mode: str = "scene"
# Bulk description sources, in the order they run. Empty means the defaults from
# services.description_runner.STEPS (reference, wiki, mcp on; transcript off).
description_steps: List[Dict[str, Any]] = []
# How many known traits a character may have and still be picked up by a bulk
# description run. 0 means "only the ones with nothing yet", which is the point:
# chasing the last missing field for 300 characters buys nothing.
bulk_threshold: int = 0
# The reference document and the one wiki site this project imports from. Per
# project: they are series data, not app configuration.
reference_path: str = ""
wiki_url: str = ""
wiki_term: str = ""
# How the series is named when asking the MCP server ("In the book series X,
# describe Y"). The folder name is often wrong for that, so it is its own field.
mcp_series_name: str = ""
# Volume ids ticked in the dashboard table. Ticked volumes narrow the run; an
# empty list means the whole project.
pipeline_selected_books: List[int] = []
# Run bar state. Both survive a refresh (the table re-renders every 3 s), which a
# widget's own value does not.
pipeline_run_force: bool = False
pipeline_run_stages: List[str] = ["map", "cast", "prompts"]
# Cached per-book pipeline snapshots, keyed by output dir (see
# services/pipeline_status.py). Entries are (signature, snapshot).
_pipeline_cache: Dict[str, Any] = {}

# --- Transcript cleanup options ---
# When set, "Clean Up Transcripts" also repairs volumes that have no EPUB
# source, using the spelling corrections learned from the volumes that do.
spellfix_from_corrections: bool = True

# --- Transcript cleanup progress (survives panel re-renders / tab switches) ---
# The cleanup run writes here; whichever Source Text panel is currently rendered
# exposes its refresh via ``cleanup_refresh`` so the run can update it safely.
cleanup_progress: Dict[str, Any] = {"phase": "confirm", "current": "", "rows": [], "summary": ""}
cleanup_refresh: Optional[Any] = None

# --- Dynamic Image Generation Task States ---
image_gen_active: bool = False
cancel_image_gen_flag: bool = False
stop_after_current_book: bool = False  # Graceful circuit-breaker to halt batch loops after active book completion

# --- Style Playground & Workflow Analyzer Bindings ---
# A style set (styles/<name>.json) is the Character base plus per-role deltas.
# ``style_base_style`` holds the Character values; ``style_role_overrides`` holds
# the deltas. The three style_prompt_* globals and ``style_workflow_overrides``
# are the *view*: the effective values for ``style_active_role``, rewritten by
# the Style & Workflow panel whenever the role changes or a field is edited.
comfy_online: bool = False
style_selected_preset: str = "retro_graphic_novel"
style_preset_save_name: str = ""
style_selected_workflow: str = ""
style_active_role: str = "character"  # role shown in the Definition panel
style_base_style: Dict[str, Any] = {
    "prompt_prefix": "space opera adventure graphic novel illustration, sharp ink sketch, crisp outlines, retro-futuristic sci-fi aesthetic, cosmic wonder, detailed, ",
    "prompt_suffix": ", high-contrast shadows, selective color accents, bold ink-wash shading",
    "negative_prompt": "blurry, bad quality, text, watermark, photorealistic, photography, dystopian, gritty, grimy, decay, cyberpunk",
    "overrides": {},
}
style_role_overrides: Dict[str, Dict[str, Any]] = {}
style_prompt_prefix: str = "space opera adventure graphic novel illustration, sharp ink sketch, crisp outlines, retro-futuristic sci-fi aesthetic, cosmic wonder, detailed, "
style_prompt_suffix: str = ", high-contrast shadows, selective color accents, bold ink-wash shading"
style_negative_prompt: str = "blurry, bad quality, text, watermark, photorealistic, photography, dystopian, gritty, grimy, decay, cyberpunk"
style_test_prompts: List[Dict[str, Any]] = []
style_test_images: List[Optional[str]] = []  # List of Base64-encoded strings or None
style_test_seeds: List[int] = []
style_test_role_mode: str = "current"  # current | variety | a role name
style_lock_samples: bool = False
style_playground_loading: bool = False

# --- Freeform ("Custom Prompt") Playground Mode ---
# Lets the user render hand-written prompts through the active workflow, LoRA
# selections, and node overrides without sourcing text from a book's prompts.csv.
# When active, the sample-sync engine must not overwrite style_test_prompts, and
# character-tag replacement is skipped because freeform text has no #tags# syntax.
style_freeform_active: bool = False
style_freeform_prompt: str = ""
style_freeform_use_style_modifiers: bool = False  # Whether to apply Style Prefix/Suffix
style_freeform_batch_size: int = 1  # 1-4; each entry gets a fresh random seed
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
recent_rendered_images: List[Dict[str, Any]] = []  # [{"filename": "", "base64": "", "chapter": 1, "shot": 1, "quote": "", "prompt": ""}]
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
# Live inference rate reported by the ABI-Subtitles server (audio-seconds per
# wall-second) for the job it is currently decoding/inferring. Sticky across
# the server's per-file decode gap; 0.0 means "use the pipeline's own estimate".
transcription_speedup: float = 0.0

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