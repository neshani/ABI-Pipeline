import json
import random
from pathlib import Path
from typing import Any, List, Optional, Dict
from nicegui import ui
from sqlmodel import Session
from database.models import Project
from ui import state
from services.prompt_engine import (
    list_stored_templates,
    load_template_by_name,
    save_template_by_name,
    fetch_test_chunks,
    get_llm_response,
    parse_llm_response,
    ensure_templates_directory
)
from database.connection import get_setting

# Unified Pipeline Steps
STAGES = ["Imported", "Transcription", "Prompt Gen", "Image Gen", "Proofreading", "Finished"]

def get_active_stage_idx(status: str) -> int:
    mapping = {
        "Imported": 0,
        "Transcribing": 1,
        "Transcribed": 2,          # Advances stepper to "Prompt Gen" once transcribed
        "Generating Prompts": 2,
        "Prompts Created": 3,      # Advances stepper to "Image Gen" once prompts are ready
        "Rendering Images": 3,
        "Images Created": 4,       # Advances stepper to "Proofreading" once images are ready
        "Proofreading": 4,
        "Finished": 5
    }
    return mapping.get(status, 0)


@ui.refreshable
def render_stepper(status: str):
    """Isolated stepper layout container that refreshes extremely fast with zero DOM flashing."""
    current_stage_idx = get_active_stage_idx(status)
    
    with ui.row().classes('w-full justify-between items-center bg-white border rounded-xl p-4 shadow-sm mb-2'):
        for idx, stage_name in enumerate(STAGES):
            is_completed = idx < current_stage_idx
            is_active = idx == current_stage_idx
            
            with ui.row().classes('items-center gap-1.5'):
                if is_completed:
                    ui.icon('check_circle', color='emerald-500', size='18px')
                    ui.label(stage_name).classes('text-xs font-bold text-emerald-600')
                elif is_active:
                    ui.icon('radio_button_checked', color='blue-600', size='18px').classes('animate-pulse')
                    ui.label(stage_name).classes('text-xs font-black text-blue-700')
                else:
                    ui.icon('radio_button_unchecked', color='slate-300', size='18px')
                    ui.label(stage_name).classes('text-xs font-semibold text-slate-400')
                    
            if idx < len(STAGES) - 1:
                ui.icon('chevron_right', color='slate-300', size='xs')

def open_large_image(img_base64: str, title: str):
    """Opens a modal popup dialog displaying the full-size rendered image."""
    with ui.dialog() as d, ui.card().classes('w-full max-w-3xl p-4 items-center bg-white rounded-xl shadow-lg'):
        ui.label(title).classes('text-sm font-bold text-slate-800 mb-3 uppercase tracking-wider')
        ui.image(img_base64).classes('w-full rounded-lg max-h-[75vh] object-contain cursor-zoom-out').on('click', d.close)
        with ui.row().classes('w-full justify-end mt-3'):
            ui.button('Close', on_click=d.close).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs')
    d.open()

def copy_results_to_clipboard():
    """Formats prompt configurations and outputs as markdown, copying to host clipboard safely."""
    if not state.playground_results:
        ui.notify("No results to copy yet.", type="warning")
        return

    markdown_lines = []
    markdown_lines.append("### Active Prompt Template:")
    markdown_lines.append(f"```text\n{state.playground_template}\n```\n")
    
    markdown_lines.append("### Pipeline Run Parameters:")
    markdown_lines.append(f"- **Volume:** {state.playground_book_selection}")
    markdown_lines.append(f"- **Mode:** {state.playground_selection_mode}")
    if state.playground_selection_mode == "Static Segment":
        markdown_lines.append(f"- **Start Index:** {state.playground_start_index}")
    else:
        markdown_lines.append(f"- **Seed:** {state.playground_seed}")
    markdown_lines.append(f"- **Chunk Count Tested:** {state.playground_chunk_count}\n")

    markdown_lines.append("### Segment Evaluation Output:")
    for idx, res in enumerate(state.playground_results):
        status_label = "Refusal Skipped" if res.get("status") == "refusal" else "Extraction Match"
        markdown_lines.append(f"#### Segment Chunk {idx + 1} ({status_label})")
        markdown_lines.append(f"**Source Text Passage:**\n> \"{res['chunk']}\"\n")
        markdown_lines.append(f"**Extracted Verbatim Quote:**\n> \"{res['quote']}\"\n")
        markdown_lines.append(f"**Generated Visual Prompt:**\n> {res['prompt']}\n")
        markdown_lines.append("-" * 30 + "\n")

    formatted_markdown = "\n".join(markdown_lines)

    js_code = f"""
    (function() {{
        const text = {json.dumps(formatted_markdown)};
        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(() => {{
                console.log('Copied safely via native API.');
            }}).catch(err => {{
                console.error('Native copy failed, attempting fallback: ', err);
            }});
        }} else {{
            const textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed";
            textArea.style.opacity = "0";
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {{
                document.execCommand('copy');
                console.log('Copied safely via legacy fallback.');
            }} catch (err) {{
                console.error('Legacy fallback copy failed: ', err);
            }}
            document.body.removeChild(textArea);
        }}
    }})();
    """
    ui.run_javascript(js_code)
    ui.notify("Markdown results copied to clipboard!", type="positive", icon="assignment_turned_in")


def copy_condensed_results_to_clipboard():
    """Formats only prompt/quote extracts to save LLM chat tokens, completely omitting the prompt template."""
    if not state.playground_results:
        ui.notify("No results to copy yet.", type="warning")
        return

    markdown_lines = []
    markdown_lines.append("### Pipeline Run Parameters (Condensed):")
    markdown_lines.append(f"- **Volume:** {state.playground_book_selection}")
    markdown_lines.append(f"- **Mode:** {state.playground_selection_mode}")
    if state.playground_selection_mode == "Static Segment":
        markdown_lines.append(f"- **Start Index:** {state.playground_start_index}")
    else:
        markdown_lines.append(f"- **Seed:** {state.playground_seed}")
    markdown_lines.append(f"- **Chunk Count Tested:** {state.playground_chunk_count}\n")

    markdown_lines.append("### Segment Evaluation (Prompt & Quote Only):")
    for idx, res in enumerate(state.playground_results):
        status_label = "Refusal Skipped" if res.get("status") == "refusal" else "Extraction Match"
        markdown_lines.append(f"#### Segment Chunk {idx + 1} ({status_label})")
        markdown_lines.append(f"- **Extracted Verbatim Quote:** \"{res['quote']}\"")
        markdown_lines.append(f"- **Generated Visual Prompt:** {res['prompt']}\n")

    formatted_markdown = "\n".join(markdown_lines)

    js_code = f"""
    (function() {{
        const text = {json.dumps(formatted_markdown)};
        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(() => {{
                console.log('Copied condensed results safely via native API.');
            }}).catch(err => {{
                console.error('Native copy failed, attempting fallback: ', err);
            }});
        }} else {{
            const textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed";
            textArea.style.opacity = "0";
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {{
                document.execCommand('copy');
                console.log('Copied condensed results safely via legacy fallback.');
            }} catch (err) {{
                console.error('Legacy fallback copy failed: ', err);
            }}
            document.body.removeChild(textArea);
        }}
    }})();
    """
    ui.run_javascript(js_code)
    ui.notify("Condensed results copied to clipboard!", type="positive", icon="assignment_turned_in")


@ui.refreshable
def render_playground_results_container():
    """Isolated results window showing generated parsed prompts and input source texts."""
    if state.playground_loading:
        with ui.column().classes('w-full items-center justify-center p-12 bg-slate-50 border rounded-xl border-dashed'):
            ui.spinner(size='lg', color='blue')
            ui.label("Dispatching test requests to local LLM...").classes('text-sm text-slate-500 mt-2 font-medium')
        return

    if not state.playground_results:
        with ui.column().classes('w-full items-center justify-center p-12 text-slate-400 border border-dashed rounded-xl bg-slate-50'):
            ui.icon('science', size='lg', color='slate-300')
            ui.label("Testing output is currently empty. Define configurations and click 'Test Prompt Template' to run.").classes('text-xs text-center max-w-sm')
        return

    with ui.column().classes('w-full gap-4'):
        # Copy to Clipboard Toolbar Row
        with ui.row().classes('w-full justify-between items-center bg-slate-100 p-3 rounded-lg border'):
            with ui.column().classes('gap-0'):
                ui.label("Evaluation Iteration Ready").classes('text-xs font-bold text-slate-700')
                ui.label("Format optimized for sharing with diagnostic AIs").classes('text-[10px] text-slate-500')
            with ui.row().classes('gap-2'):
                ui.button(
                    "Copy Full", 
                    icon="content_copy", 
                    on_click=copy_results_to_clipboard
                ).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs font-semibold px-4')
                ui.button(
                    "Copy Condensed (Saves Tokens)", 
                    icon="compress", 
                    on_click=copy_condensed_results_to_clipboard
                ).classes('bg-blue-700 hover:bg-blue-800 text-white text-xs font-semibold px-4')

        for idx, res in enumerate(state.playground_results):
            is_refusal = res.get("status") == "refusal"
            border_color = "border-red-200 bg-red-50/20" if is_refusal else "border-slate-200 bg-white"
            badge_color = "bg-red-100 text-red-800" if is_refusal else "bg-emerald-100 text-emerald-800"
            badge_label = "Refusal Skipped" if is_refusal else "Extraction Match"

            with ui.card().classes(f'w-full border p-4 rounded-xl shadow-xs gap-3 {border_color}'):
                with ui.row().classes('w-full justify-between items-center pb-2 border-b border-dashed'):
                    ui.label(f"Segment Chunk {idx + 1}").classes('text-xs font-bold text-slate-600 uppercase')
                    ui.badge(badge_label).classes(f'px-2 py-0.5 rounded text-[10px] font-bold {badge_color}')

                with ui.grid(columns='1fr 1fr').classes('w-full gap-4'):
                    # LEFT: Original text chunk
                    with ui.column().classes('gap-1 bg-blue-50/30 p-3 rounded-lg border border-blue-50/50'):
                        ui.label("Source Text Passage:").classes('text-[10px] font-black text-slate-400 uppercase')
                        ui.label(f'"{res["chunk"][:320]}..."').classes('text-xs text-slate-600 italic leading-relaxed')

                    # RIGHT: Parsed Prompt and Quote
                    with ui.column().classes('gap-2 p-3 bg-emerald-50/20 rounded-lg border border-emerald-50/50'):
                        with ui.column().classes('gap-0.5'):
                            ui.label("Extracted Verbatim Quote:").classes('text-[10px] font-black text-slate-400 uppercase')
                            ui.label(f'"{res["quote"]}"').classes('text-xs font-semibold text-slate-700')
                        with ui.column().classes('gap-0.5 mt-2'):
                            ui.label("Generated Visual Prompt:").classes('text-[10px] font-black text-slate-400 uppercase')
                            ui.label(res["prompt"]).classes('text-xs font-semibold text-blue-700 leading-relaxed')


async def execute_playground_test(project_name: str):
    """Gathers settings, reads segments, calls local LLM, and populates UI results list."""
    if not state.playground_book_selection:
        ui.notify("Please select a target book to test.", type="warning")
        return

    state.playground_loading = True
    state.playground_results.clear()
    render_playground_results_container.refresh()

    chunks = fetch_test_chunks(
        project_name=project_name,
        book_name=state.playground_book_selection,
        count=state.playground_chunk_count,
        mode=state.playground_selection_mode,
        start_index=state.playground_start_index,
        seed=state.playground_seed
    )

    if not chunks:
        ui.notify("No transcript texts available for this volume. Please run Transcription first.", type="negative")
        state.playground_loading = False
        render_playground_results_container.refresh()
        return

    llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
    model_name = get_setting("llm_model", "local-model")

    results = []
    for chunk in chunks:
        final_prompt = state.playground_template.replace("<text>", chunk)
        raw_resp = await get_llm_response(final_prompt, llm_url, model_name)
        parsed = parse_llm_response(raw_resp)
        results.append({
            "chunk": chunk,
            "quote": parsed["quote"],
            "prompt": parsed["prompt"],
            "status": parsed["status"]
        })

    state.playground_results = results
    state.playground_loading = False
    render_playground_results_container.refresh()
    ui.notify("Playground test iteration complete!", type="positive")

def handle_dashboard_template_change(val: str):
    """Updates the active template state and pre-loads its text contents globally."""
    if not val:
        return
    state.playground_selected_template = val
    loaded = load_template_by_name(val)
    if loaded:
        state.playground_template = loaded
        ui.notify(f"Active Prompt Template changed to: {val}", type="info")

def handle_template_dropdown_selection(val: str, prompt_editor_widget):
    """Loads a named prompt template and updates the text editor binding."""
    if not val:
        return
    state.playground_selected_template = val
    loaded = load_template_by_name(val)
    if loaded:
        state.playground_template = loaded
        prompt_editor_widget.set_value(loaded)
        ui.notify(f"Loaded template: {val}", type="info")


def handle_save_custom_template(custom_name: str, template_dropdown):
    """Saves editor contents into a named txt template and refreshes dropdown options."""
    name_clean = custom_name.strip().replace(" ", "_")
    if not name_clean:
        ui.notify("Please enter a valid template name.", type="negative")
        return
    save_template_by_name(name_clean, state.playground_template)
    ui.notify(f"Template '{name_clean}' saved successfully!", type="positive")
    
    # Reload dropdown items, update active value, and trigger UI update
    template_dropdown.options = list_stored_templates()
    template_dropdown.value = name_clean
    template_dropdown.update()
    state.playground_selected_template = name_clean


def handle_delete_template(template_dropdown, prompt_editor_widget):
    """Deletes the active template and resets the dropdown to 'default'."""
    target_name = state.playground_selected_template
    if not target_name:
        ui.notify("No template selected for deletion.", type="warning")
        return
    if target_name == "default":
        ui.notify("The default template cannot be deleted.", type="negative")
        return

    from pathlib import Path
    templates_dir = Path("./prompt_templates")
    target_file = templates_dir / f"{target_name}.txt"
    if target_file.exists():
        try:
            target_file.unlink()
            ui.notify(f"Deleted template: {target_name}", type="positive")
        except Exception as ex:
            ui.notify(f"Error deleting template file: {str(ex)}", type="negative")
            return
    else:
        ui.notify(f"Template file '{target_name}.txt' not found.", type="warning")

    # Reload dropdown options, reset value, and trigger UI update
    all_templates = list_stored_templates()
    template_dropdown.options = all_templates
    template_dropdown.value = "default"
    template_dropdown.update()
    
    # Reset internal playground state variables
    state.playground_selected_template = "default"
    loaded_default = load_template_by_name("default")
    state.playground_template = loaded_default
    prompt_editor_widget.set_value(loaded_default)

@ui.refreshable
def render_recent_prompts_feed():
    """Renders the last 5 generated prompts dynamically in-place during pipeline execution."""
    with ui.card().classes('w-full border p-5 shadow-sm bg-white mt-4 gap-3'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('auto_awesome', size='sm', color='blue-500')
            ui.label('Live Prompt Generation Feed (Most Recent)').classes('text-sm font-bold text-slate-800')
            
        # Conditionally show descriptive placeholder states based on stage progress
        if state.project_status in ("Imported", "Transcribing"):
            with ui.column().classes('w-full items-center justify-center p-6 bg-slate-50 border border-dashed rounded-lg text-slate-400'):
                ui.icon('lock', size='md', color='slate-300')
                ui.label('Feed is currently locked. Complete transcription to unlock prompt generation.').classes('text-xs text-center')
        elif not state.recent_prompts:
            with ui.column().classes('w-full items-center justify-center p-6 bg-slate-50 border border-dashed rounded-lg text-slate-400'):
                ui.icon('science', size='md', color='slate-300')
                ui.label('Feed is currently empty. Start prompt generation to stream live prompts here...').classes('text-xs text-center')
        else:
            with ui.column().classes('w-full gap-2'):
                for item in reversed(state.recent_prompts):
                    is_refusal = item["status"] == "refusal" or item["prompt"] == "REFUSAL"
                    bg_color = "bg-red-50/40 border-red-100" if is_refusal else "bg-slate-50/50 border-slate-100"
                    badge_label = "Refused" if is_refusal else "Success"
                    badge_color = "rose" if is_refusal else "emerald"
                    
                    with ui.card().classes(f'w-full border p-3 rounded-lg shadow-xs {bg_color}'):
                        with ui.row().classes('w-full justify-between items-center'):
                            ui.label(f"{item['book']} — Ch {item['chapter']}, Scene {item['scene']}").classes('text-[10px] font-bold text-slate-500 uppercase')
                            ui.badge(badge_label, color=badge_color).classes('text-[9px]')
                        ui.label(f'Quote: "{item["quote"][:220]}..."' if len(item["quote"]) > 220 else f'Quote: "{item["quote"]}"').classes('text-xs italic text-slate-600 leading-normal')
                        ui.label(f'Prompt: {item["prompt"]}').classes('text-xs font-semibold text-blue-700 leading-normal')

    # Register refresh callback globally so background updates bind successfully
    state.recent_prompts_refresh = render_recent_prompts_feed.refresh


# --- DYNAMIC STYLE PRESETS & WORKFLOW ANALYZER UTILITIES ---

def list_available_workflows() -> list:
    """Discovers .json workflows inside local './workflows' directory."""
    workflows_dir = Path("./workflows")
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Search also in extra legacy folders to maintain consistency
    legacy_dir = Path("./Comfy_Workflows")
    
    found = [f.name for f in workflows_dir.glob("*.json")]
    if legacy_dir.exists():
        found.extend([f.name for f in legacy_dir.glob("*.json")])
        
    if not found:
        # Autogenerate a dummy local api file for testing purposes
        dummy_workflow = {
            "3": {
                "inputs": {
                    "seed": 0, "steps": 7, "cfg": 1, "sampler_name": "euler_ancestral", "scheduler": "beta",
                    "model": ["20", 0], "positive": ["19", 0], "negative": ["7", 0], "latent_image": ["13", 0]
                },
                "class_type": "KSampler"
            },
            "6": { "inputs": { "text": "<prompt>" }, "class_type": "CLIPTextEncode" },
            "7": { "inputs": { "text": "<negPrompt>" }, "class_type": "CLIPTextEncode" },
            "13": { "inputs": { "width": 1024, "height": 1024 }, "class_type": "EmptySD3LatentImage" }
        }
        try:
            with open(workflows_dir / "default_comfy_api.json", "w") as f:
                json.dump(dummy_workflow, f, indent=2)
            found.append("default_comfy_api.json")
        except Exception:
            pass
            
    return sorted(list(set(found)))


def load_style_presets() -> list:
    """Discovers .json files inside local styles directory."""
    styles_dir = Path("./styles")
    styles_dir.mkdir(parents=True, exist_ok=True)
    presets = [f.stem for f in styles_dir.glob("*.json")]
    if "default" not in presets:
        presets.append("default")
    return sorted(presets)


def load_style_preset_by_name(name: str):
    """Parses style json presets, seeding prompt/negative prompt UI textboxes."""
    if name == "default":
        state.style_prompt_prefix = "ArsMJStyle, 1890s Victorian illustration, detailed pen and ink with soft watercolor wash, Sidney Paget style. "
        state.style_negative_prompt = "blurry, bad quality, text, watermark, photorealistic, photography"
        state.style_workflow_overrides.clear()
        return
        
    styles_dir = Path("./styles")
    file_path = styles_dir / f"{name}.json"
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                state.style_prompt_prefix = data.get("prompt_prefix", "")
                state.style_negative_prompt = data.get("negative_prompt", "")
                state.style_workflow_overrides = data.get("overrides", {})
        except Exception:
            pass
            
    render_workflow_overrides_ui.refresh()


def save_style_preset_by_name(name: str):
    """Saves style prefix & negative prompt variables into an on-disk style preset JSON file."""
    name_clean = name.strip().replace(" ", "_")
    if not name_clean:
        ui.notify("Please enter a valid Style name.", type="negative")
        return
    styles_dir = Path("./styles")
    styles_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        "name": name,
        "prompt_prefix": state.style_prompt_prefix,
        "negative_prompt": state.style_negative_prompt,
        "overrides": state.style_workflow_overrides
    }
    try:
        with open(styles_dir / f"{name_clean}.json", "w") as f:
            json.dump(data, f, indent=2)
        ui.notify(f"Style preset '{name_clean}' saved successfully!", type="positive")
    except Exception as e:
        ui.notify(f"Failed to save preset: {str(e)}", type="negative")


def update_override_state(node_id: str, key: str, value: Any):
    """Triggers on manual slider adjustments, storing changes targeting specific Node IDs."""
    if node_id not in state.style_workflow_overrides:
        state.style_workflow_overrides[node_id] = {}
    state.style_workflow_overrides[node_id][key] = value


def handle_style_workflow_change(val: str):
    """Reads a ComfyUI JSON, introspects active sampler/latents, and rebuilds dynamic UI sliders."""
    if not val:
        return
    state.style_selected_workflow = val
    
    wf_path = Path("./workflows") / val
    if not wf_path.exists():
        wf_path = Path("./Comfy_Workflows") / val
        
    if wf_path.exists():
        try:
            with open(wf_path, "r") as f:
                wf_json = json.load(f)
            from services.comfy_client import ComfyClient
            client = ComfyClient("127.0.0.1:8188")  # Mock address for analyzer pass
            state.style_discovered_params = client.analyze_workflow(wf_json)
            state.style_workflow_overrides.clear()  # Clear overrides upon swapping workflows
            ui.notify(f"Analyzed workflow '{val}'. Discovered {len(state.style_discovered_params)} overrides.", type="info")
        except Exception as e:
            ui.notify(f"Failed to analyze workflow: {str(e)}", type="warning")
            state.style_discovered_params.clear()
            
    render_workflow_overrides_ui.refresh()


def fetch_real_prompts(project_name: str, book_name: str, count: int = 4, prompt_seed: int = 42) -> List[Dict[str, Any]]:
    """Tries to read extracted prompts & quotes from project output directory, tracking full scene metadata [2]."""
    import pandas as pd
    
    csv_paths = [
        Path(f"./output/{project_name}/{book_name}/prompts.csv"),  # Exact nested directory match
        Path(f"./output/{project_name}/{book_name}_prompts.csv"),
        Path(f"./output/{project_name}/{book_name}/{book_name}_prompts.csv"),
        Path(f"./output/{project_name}/{project_name}_prompts.csv"),
        Path(f"./output/{project_name}_prompts.csv")
    ]
    
    items = []
    for path in csv_paths:
        if path.exists():
            try:
                df = pd.read_csv(path, sep='|')
                # Exclude missing or unfilled prompts
                valid_df = df.dropna(subset=['prompt'])
                valid_df = valid_df[valid_df['prompt'].str.strip().str.lower() != 'none']
                valid_df = valid_df[valid_df['prompt'].str.strip() != '']
                
                if not valid_df.empty:
                    sample_size = min(count, len(valid_df))
                    # Sample consistently using the user-provided prompt seed
                    sampled_df = valid_df.sample(n=sample_size, random_state=prompt_seed)
                    
                    for _, row in sampled_df.iterrows():
                        items.append({
                            "book": book_name,
                            "chapter": int(row.get('chapter', 1)),
                            "scene": int(row.get('scene', 1)),
                            "prompt": str(row['prompt']).strip(),
                            "quote": str(row.get('quote', '')).strip()
                        })
                    break
            except Exception:
                pass

    # Fallback to standard raw text chunks if no CSV exists
    if not items:
        chunks = fetch_test_chunks(project_name, book_name, count=count, mode="Seeded Random")
        if chunks:
            for idx, chunk in enumerate(chunks):
                items.append({
                    "book": book_name,
                    "chapter": 1,
                    "scene": idx + 1,
                    "prompt": f"Generating from raw passage fallback: {chunk[:80]}...",
                    "quote": chunk
                })
                
    return items


def draw_style_test_sample(project_name: str, book_name: str):
    """Pulls randomized visual prompt scenes using the active state.style_prompt_seed for consistency."""
    state.style_test_prompts = fetch_real_prompts(
        project_name=project_name,
        book_name=book_name,
        count=state.style_chunk_count,
        prompt_seed=state.style_prompt_seed
    )
    
    # Pre-populate static seeds
    if state.style_use_random_image_seed:
        state.style_test_seeds = [random.randint(100000, 999999) for _ in range(len(state.style_test_prompts))]
    else:
        state.style_test_seeds = [state.style_image_seed] * len(state.style_test_prompts)
        
    state.style_test_images = [None] * len(state.style_test_prompts)
    render_style_playground_cards.refresh()


async def execute_style_playground_batch(project_name: str):
    """Executes the test batch against ComfyUI, passing the correct dynamic prompt key."""
    if not state.style_selected_workflow:
        ui.notify("Please select a workflow first.", type="warning")
        return
        
    comfy_url = get_setting("comfy_url", "127.0.0.1:8188")
    if "http" in comfy_url:
        comfy_url = comfy_url.replace("http://", "").replace("https://", "").strip("/")
        
    wf_path = Path("./workflows") / state.style_selected_workflow
    if not wf_path.exists():
        wf_path = Path("./Comfy_Workflows") / state.style_selected_workflow
        
    if not wf_path.exists():
        ui.notify(f"Workflow file '{state.style_selected_workflow}' not found.", type="negative")
        return
        
    try:
        with open(wf_path, "r") as f:
            workflow_json = json.load(f)
    except Exception as e:
        ui.notify(f"Failed to load workflow JSON: {str(e)}", type="negative")
        return

    state.style_playground_loading = True
    state.style_test_images = [None] * len(state.style_test_prompts)
    render_style_playground_cards.refresh()

    from services.comfy_client import ComfyClient
    client = ComfyClient(comfy_url)

    import asyncio
    for idx, item in enumerate(state.style_test_prompts):
        # Decide whether to use randomized noise seeds or a single locked image seed
        if state.style_use_random_image_seed:
            seed = state.style_test_seeds[idx]
        else:
            seed = state.style_image_seed
            state.style_test_seeds[idx] = seed
            
        prompt_text = item["prompt"] if isinstance(item, dict) else item
        
        def run_image():
            return client.generate_image_sync(
                workflow_json=workflow_json,
                prompt_text=prompt_text,
                neg_prompt_text=state.style_negative_prompt,
                seed=seed,
                overrides=state.style_workflow_overrides,
                prefix=state.style_prompt_prefix
            )
            
        try:
            img_bytes, logs = await asyncio.to_thread(run_image)
            if img_bytes:
                import base64
                encoded = base64.b64encode(img_bytes).decode("utf-8")
                state.style_test_images[idx] = f"data:image/png;base64,{encoded}"
            
            for line in logs.split("\n"):
                state.add_console_log(line)
        except Exception as e:
            state.add_console_log(f"[Style-Playground] Exception generating sample card {idx+1}: {str(e)}")
            
        render_style_playground_cards.refresh()

    state.style_playground_loading = False
    render_style_playground_cards.refresh()
    ui.notify("Visual test batch processing complete!", type="positive")


@ui.refreshable
def render_workflow_overrides_ui():
    """Renders the self-introspecting overrides form dynamically."""
    if not state.style_discovered_params:
        ui.label("No customizable nodes discovered in this workflow.").classes('text-xs text-slate-400 italic')
        return

    with ui.column().classes('w-full gap-3 bg-slate-50 p-3 rounded-lg border mt-2'):
        ui.label("Workflow Parameters (Auto-Discovered)").classes('text-xs font-bold text-slate-700')
        
        for node_id, data in state.style_discovered_params.items():
            node_title = data["title"]
            node_type = data["type"]
            params = data["params"]
            
            with ui.expansion(f"{node_title} (ID: {node_id})").classes('w-full border rounded bg-white text-xs'):
                with ui.column().classes('w-full p-3 gap-3'):
                    if node_type == "sampler":
                        ui.number(
                            label="Steps",
                            value=params["steps"],
                            min=1, max=150, step=1,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "steps", e.value)
                        ).classes('w-full')
                        ui.number(
                            label="CFG Scale",
                            value=params["cfg"],
                            min=0.0, max=30.0, step=0.1,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "cfg", e.value)
                        ).classes('w-full')
                    elif node_type == "resolution":
                        ui.number(
                            label="Width",
                            value=params["width"],
                            min=128, max=4096, step=64,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "width", e.value)
                        ).classes('w-full')
                        ui.number(
                            label="Height",
                            value=params["height"],
                            min=128, max=4096, step=64,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "height", e.value)
                        ).classes('w-full')
                    elif node_type == "lora_loader":
                        ui.input(
                            label="LoRA Filename",
                            value=params["lora_name"],
                            on_change=lambda e, nid=node_id: update_override_state(nid, "lora_name", e.value)
                        ).classes('w-full')
                        ui.number(
                            label="Strength",
                            value=params["strength_model"],
                            min=0.0, max=2.0, step=0.1,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "strength_model", e.value)
                        ).classes('w-full')


@ui.refreshable
def render_style_playground_cards():
    """Renders test scene cards showing detailed book identifiers, visual prompt, and click-to-expand image modal."""
    if not state.style_test_prompts:
        with ui.column().classes('w-full items-center justify-center p-12 text-slate-400 border border-dashed rounded-xl bg-slate-50'):
            ui.icon('brush', size='lg', color='slate-300')
            ui.label("Visual Playground is empty. Use the settings panel above to draw or customize test scenes.").classes('text-xs text-center max-w-sm')
        return

    with ui.grid(columns='1fr 1fr').classes('w-full gap-4'):
        for idx, item in enumerate(state.style_test_prompts):
            img_data = state.style_test_images[idx] if idx < len(state.style_test_images) else None
            
            is_dict = isinstance(item, dict)
            chapter = item.get("chapter", 1) if is_dict else 1
            scene_num = item.get("scene", idx + 1) if is_dict else idx + 1
            book_title = item.get("book", "Novel") if is_dict else "Novel"
            prompt_str = item.get("prompt", "") if is_dict else item

            # Fetch active image seed cleanly
            seed = state.style_test_seeds[idx] if idx < len(state.style_test_seeds) else state.style_image_seed

            card_title = f"Ch {chapter}, Scene {scene_num}"
            full_title_header = f"{card_title} • {book_title}"

            with ui.card().classes('w-full border p-4 rounded-xl shadow-xs gap-3 bg-white'):
                with ui.row().classes('w-full justify-between items-center pb-1 border-b border-dashed'):
                    with ui.column().classes('gap-0'):
                        ui.label(card_title).classes('text-xs font-black text-slate-700 uppercase')
                        ui.label(book_title).classes('text-[9px] text-slate-400 truncate max-w-[150px]')
                    ui.badge(f"Seed: {seed}", color="slate").classes('text-[9px] font-bold')

                if img_data:
                    ui.image(img_data).classes('w-full h-48 rounded-lg object-cover border shadow-sm cursor-zoom-in hover:brightness-95 transition-all') \
                        .on('click', lambda _, img=img_data, title=full_title_header: open_large_image(img, title))
                elif state.style_playground_loading:
                    with ui.column().classes('w-full h-48 items-center justify-center bg-slate-50 rounded-lg border border-dashed'):
                        ui.spinner(size='md', color='blue')
                        ui.label("Rendering...").classes('text-[9px] text-slate-400 mt-1')
                else:
                    with ui.column().classes('w-full h-48 items-center justify-center bg-slate-50 rounded-lg border border-dashed text-slate-400'):
                        ui.icon('photo_library', size='md', color='slate-300')
                        ui.label("Awaiting Style Generation").classes('text-[10px]')

                with ui.column().classes('gap-1 bg-emerald-50/20 p-2.5 rounded border border-emerald-50/50 w-full'):
                    ui.label("Extracted Image Prompt:").classes('text-[8px] font-black text-slate-400 uppercase tracking-wider')
                    ui.label(prompt_str[:160] + "..." if len(prompt_str) > 160 else prompt_str).classes('text-xs font-semibold text-slate-700 leading-normal')


def render_project_tabs(
    project: Project, 
    books: list, 
    start_transcribe_cb, 
    stop_transcribe_cb,
    start_prompt_gen_cb=None,
    start_image_gen_cb=None
):
    # Prepare directory configurations
    ensure_templates_directory()
    available_templates = list_stored_templates()
    
    # Safeguard template values on page rendering
    if not state.playground_selected_template or state.playground_selected_template not in available_templates:
        state.playground_selected_template = "default"
    
    if not state.playground_template:
        state.playground_template = load_template_by_name(state.playground_selected_template)

    # Safeguard selected book value on project workspace change to avoid NiceGUI ValueError
    book_names = [b.name for b in books]
    if books and (not state.playground_book_selection or state.playground_book_selection not in book_names):
        state.playground_book_selection = books[0].name

    # Render Dynamic Stepper inside its container
    render_stepper(state.project_status)

    # Header Row
    ui.label(f'Project Settings: {project.name}').classes('text-lg font-bold text-slate-800 mt-2')
    
    with ui.tabs().classes('w-full border-b') as project_tabs:
        tab_dash = ui.tab('Dashboard', icon='dashboard')
        tab_style = ui.tab('Style & Workflows', icon='brush')
        tab_play = ui.tab('Prompt-Gen Playground', icon='science')
        
    project_tabs.bind_value(state, 'active_project_tab')
        
    with ui.tab_panels(project_tabs, value=state.active_project_tab).classes('w-full bg-transparent p-0'):
        with ui.tab_panel(tab_dash):
            # Dynamic stats panels utilizing stable reactive text bindings
            with ui.row().classes('w-full gap-4'):
                with ui.card().classes('flex-1 border p-4 shadow-sm bg-white'):
                    ui.label('Project Status').classes('text-xs font-semibold text-slate-500')
                    ui.label('').classes('text-xl font-bold text-slate-800').bind_text_from(state, 'project_status')
                with ui.card().classes('flex-1 border p-4 shadow-sm bg-white'):
                    ui.label('Discovered Books').classes('text-xs font-semibold text-slate-500')
                    ui.label(str(len(books))).classes('text-xl font-bold text-slate-800')
                    
            # Process Control Card
            with ui.card().classes('w-full border p-5 shadow-sm bg-white mt-4'):
                ui.label('Sequential Process Orchestration').classes('text-sm font-bold text-slate-800 mb-2')
                ui.label('Configure styling rules and workflows in the settings tab prior to launching processing.').classes('text-xs text-slate-400 mb-4')
                
                # Active Prompt Template Selection Row
                with ui.row().classes('w-full items-center gap-3 mb-4 bg-slate-50 p-3 rounded-lg border'):
                    ui.icon('psychology', size='sm', color='slate-500')
                    with ui.column().classes('gap-0 flex-1'):
                        ui.label('Active Prompt Instructions Template').classes('text-xs font-bold text-slate-700')
                        ui.label('The set of guidelines that the LLM will follow during generation.').classes('text-[10px] text-slate-500')
                    
                    ui.select(
                        options=available_templates,
                        value=state.playground_selected_template,
                        on_change=lambda e: handle_dashboard_template_change(e.value)
                    ).classes('w-56 bg-white').props('outlined dense')

                # Active Style Preset Selection Row
                with ui.row().classes('w-full items-center gap-3 mb-4 bg-slate-50 p-3 rounded-lg border'):
                    ui.icon('brush', size='sm', color='slate-500')
                    with ui.column().classes('gap-0 flex-1'):
                        ui.label('Active Visual Style Preset').classes('text-xs font-bold text-slate-700')
                        ui.label('The preset used to decorate and render image prompts in ComfyUI.').classes('text-[10px] text-slate-500')
                    
                    ui.select(
                        options=load_style_presets(),
                        value=state.style_selected_preset,
                        on_change=lambda e: (setattr(state, 'style_selected_preset', e.value), load_style_preset_by_name(e.value))
                    ).classes('w-56 bg-white').props('outlined dense')

                # Render context-aware action buttons inside their own container
                @ui.refreshable
                def action_buttons():
                    with ui.row().classes('items-center gap-3'):
                        status = state.project_status
                        
                        if status in ("Transcribing", "Generating Prompts"):
                            ui.spinner(size='md', color='blue')
                            ui.button(
                                'Stop Execution', 
                                icon='stop', 
                                color='red', 
                                on_click=lambda: stop_transcribe_cb(project.id)
                            ).classes('px-4 font-semibold')
                        elif status == "Imported":
                            ui.button(
                                'Start Transcription', 
                                icon='play_arrow', 
                                color='green', 
                                on_click=lambda: start_transcribe_cb(project.id)
                            ).classes('px-4 font-semibold')
                        elif status == "Transcribed":
                            ui.button(
                                'Generate Prompts', 
                                icon='psychology', 
                                color='purple', 
                                on_click=lambda: start_prompt_gen_cb(project.id) if start_prompt_gen_cb else None
                            ).classes('px-4 font-semibold text-white')
                        elif status == "Prompts Created":
                            ui.button(
                                'Render Images', 
                                icon='image', 
                                color='amber', 
                                on_click=lambda: start_image_gen_cb(project.id) if start_image_gen_cb else None
                            ).classes('px-4 font-semibold text-white')
                        elif status in ("Images Created", "Proofreading"):
                            ui.button(
                                'Open Proofreader Grid', 
                                icon='edit_note', 
                                color='blue', 
                                on_click=lambda: ui.notify("Proofreader and editor workspace selected.", type="info")
                            ).classes('px-4 font-semibold text-white')
                        else:
                            ui.button(
                                'Start Processing', 
                                icon='play_arrow', 
                                color='green', 
                                on_click=lambda: start_transcribe_cb(project.id)
                            ).classes('px-4 font-semibold')
                
                action_buttons()
                state.action_buttons_refresh = action_buttons.refresh
            
            # Stable Live Console Log Output Widget (Created ONCE)
            with ui.card().classes('w-full border p-5 shadow-sm bg-white mt-4 gap-3'):
                with ui.row().classes('w-full justify-between items-center'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('terminal', size='sm', color='slate-700')
                        ui.label('Process Control Console').classes('text-sm font-bold text-slate-800')
                    ui.button('Clear Logs', on_click=lambda: (state.console_logs.clear(), log_widget.clear())).props('flat dense').classes('text-xs text-slate-500')
                
                log_widget = ui.log(max_lines=300).classes('w-full h-64 bg-slate-900 text-slate-100 font-mono text-xs p-3 rounded-lg leading-relaxed')
                for line in state.console_logs:
                    log_widget.push(line)
                
                state.active_log_widget = log_widget
                state.logs_pushed_index = len(state.console_logs)
            
            # Live Feed of generated prompt cards
            render_recent_prompts_feed()
                        
        with ui.tab_panel(tab_style):
            with ui.grid(columns='420px 1fr').classes('w-full gap-6 items-start'):
                # LEFT CONFIG PANEL
                with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-4'):
                    ui.label('Style Preset & Workflow Config').classes('text-sm font-bold text-slate-800')
                    
                    # Workflow selection
                    available_workflows = list_available_workflows()
                    if not state.style_selected_workflow or state.style_selected_workflow not in available_workflows:
                        if available_workflows:
                            state.style_selected_workflow = available_workflows[0]
                            handle_style_workflow_change(available_workflows[0])
                            
                    ui.select(
                        options=available_workflows,
                        label="ComfyUI Base Workflow (.json)",
                        value=state.style_selected_workflow,
                        on_change=lambda e: handle_style_workflow_change(e.value)
                    ).classes('w-full')
                    
                    # Style presets selection dropdown
                    available_styles = load_style_presets()
                    preset_dropdown = ui.select(
                        options=available_styles,
                        label="Saved Style Preset",
                        value=state.style_selected_preset,
                        on_change=lambda e: (setattr(state, 'style_selected_preset', e.value), load_style_preset_by_name(e.value))
                    ).classes('w-full')
                    
                    # Quick save row
                    with ui.row().classes('w-full items-end gap-2'):
                        custom_style_name = ui.input(placeholder="Preset Name", label="Save Style Preset").classes('flex-1')
                        ui.button(
                            icon="save",
                            on_click=lambda: (
                                save_style_preset_by_name(custom_style_name.value),
                                setattr(preset_dropdown, 'options', load_style_presets()),
                                preset_dropdown.update()
                            )
                        ).props('outline').classes('h-10 text-blue-600')

                    ui.separator()
                    
                    # Active prompt modification text areas
                    ui.textarea(
                        label="Style Prompt Prefix",
                        value=state.style_prompt_prefix,
                        on_change=lambda e: setattr(state, 'style_prompt_prefix', e.value)
                    ).classes('w-full h-24 text-xs').props('outlined')
                    
                    ui.textarea(
                        label="Style Negative Prompt",
                        value=state.style_negative_prompt,
                        on_change=lambda e: setattr(state, 'style_negative_prompt', e.value)
                    ).classes('w-full h-24 text-xs').props('outlined')
                    
                    # Discovered parameters expansion grid container
                    render_workflow_overrides_ui()
                    
                # RIGHT: Visual Style Playground Grid
                with ui.column().classes('w-full gap-4'):
                    with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-4'):
                        with ui.row().classes('w-full items-center gap-2'):
                            ui.icon('brush', size='sm', color='blue-500')
                            ui.label('Style Visual Playground Settings').classes('text-sm font-bold text-slate-800')
                            
                        with ui.row().classes('items-end justify-between gap-4 w-full bg-slate-50 p-4 rounded-lg border'):
                            # Dynamic image count
                            ui.number(
                                label="Num Images",
                                value=state.style_chunk_count,
                                min=1, max=8, step=1,
                                on_change=lambda e: (setattr(state, 'style_chunk_count', int(e.value)) if e.value is not None else None, draw_style_test_sample(project.name, state.playground_book_selection))
                            ).classes('w-20')

                            # Prompt selection seed (determines which scenes are drawn)
                            with ui.row().classes('items-end gap-1'):
                                prompt_seed_input = ui.number(
                                    label="Prompt Seed",
                                    value=state.style_prompt_seed,
                                    precision=0,
                                    on_change=lambda e: (setattr(state, 'style_prompt_seed', int(e.value)) if e.value is not None else None, draw_style_test_sample(project.name, state.playground_book_selection))
                                ).classes('w-24')
                                
                                ui.button(
                                    icon="casino",
                                    on_click=lambda: (
                                        setattr(state, 'style_prompt_seed', random.randint(100000, 999999)),
                                        prompt_seed_input.set_value(state.style_prompt_seed)
                                    )
                                ).props('outline dense').classes('h-10 text-slate-500')

                            # Generation/Noise Seed toggles
                            ui.switch("Random Image Seeds").bind_value(state, 'style_use_random_image_seed').classes('text-xs mb-2')
                            
                            ui.number(
                                label="Image Seed",
                                precision=0
                            ).bind_value(state, 'style_image_seed').classes('w-28').bind_visibility_from(
                                state, 'style_use_random_image_seed', value=False
                            )

                            # Launch Batch Execution
                            ui.button(
                                'Test Style Preset',
                                icon='bolt',
                                on_click=lambda: execute_style_playground_batch(project.name)
                            ).classes('bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs px-5 h-10')
                    
                    # Cards grid
                    render_style_playground_cards()
                
        with ui.tab_panel(tab_play):
            with ui.grid(columns='380px 1fr').classes('w-full gap-6 items-start'):
                # LEFT: Prompt Template Configurator and Parameters
                with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-4'):
                    ui.label('Testing Configuration').classes('text-sm font-bold text-slate-800')
                    
                    # Target Book Dropdown
                    ui.select(
                        options=[b.name for b in books],
                        label="Select Book Volume",
                        value=state.playground_book_selection
                    ).bind_value(state, 'playground_book_selection').classes('w-full')

                    # Prompt template selection, delete button, & save row
                    with ui.row().classes('w-full items-center gap-2'):
                        template_dropdown = ui.select(
                            options=available_templates,
                            label="Saved Template",
                            value=state.playground_selected_template,
                            on_change=lambda e: handle_template_dropdown_selection(e.value, prompt_editor)
                        ).classes('flex-1')
                        
                        # Flat red delete button only visible when a custom template is chosen
                        ui.button(
                            icon="delete",
                            on_click=lambda: handle_delete_template(template_dropdown, prompt_editor)
                        ).props('flat color=red').classes('h-10').bind_visibility_from(
                            state, 'playground_selected_template', backward=lambda val: val not in ('default', '')
                        )
                    
                    # Custom Template Name saver row
                    with ui.row().classes('w-full items-end gap-2'):
                        custom_name_input = ui.input(placeholder="Template Name", label="Save Custom Name").classes('flex-1')
                        ui.button(
                            icon="save", 
                            on_click=lambda: handle_save_custom_template(custom_name_input.value, template_dropdown)
                        ).props('outline').classes('h-10 text-blue-600')

                    # Editor textbox
                    prompt_editor = ui.textarea(
                        label="Prompt Instructions (contains <text>)",
                        value=state.playground_template,
                        on_change=lambda e: setattr(state, 'playground_template', e.value)
                    ).classes('w-full h-64 font-mono text-xs leading-relaxed').props('outlined')

                    # Count & Sampling Mode controllers
                    with ui.row().classes('w-full gap-3 justify-between items-end'):
                        ui.number(
                            label="Chunk Count", 
                            value=state.playground_chunk_count,
                            min=1, 
                            max=5
                        ).bind_value_to(state, 'playground_chunk_count').classes('w-20')

                        ui.select(
                            options=["Seeded Random", "Static Segment"],
                            label="Selection Mode",
                            value=state.playground_selection_mode
                        ).bind_value_to(state, 'playground_selection_mode').classes('flex-1')

                    # Start Index (Only visible when Static Segment is selected)
                    ui.number(
                        label="Start Chunk Index",
                        value=state.playground_start_index,
                        min=0,
                        precision=0
                    ).bind_value_to(state, 'playground_start_index').classes('w-full').bind_visibility_from(
                        state, 'playground_selection_mode', value='Static Segment'
                    )

                    # Random Seed (Only visible when Seeded Random is selected)
                    ui.number(
                        label="Random Seed",
                        value=state.playground_seed,
                        precision=0
                    ).bind_value_to(state, 'playground_seed').classes('w-full').bind_visibility_from(
                        state, 'playground_selection_mode', value='Seeded Random'
                    )

                    # Launch Testing Button
                    ui.button(
                        'Test Prompt Template', 
                        icon='bolt', 
                        on_click=lambda: execute_playground_test(project.name)
                    ).classes('w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold')

                # RIGHT: Live Render Output Panel
                with ui.column().classes('w-full gap-4'):
                    render_playground_results_container()

# Parent layout register callback
main_layout_ref = None
def register_main_layout(layout):
    global main_layout_ref
    main_layout_ref = layout