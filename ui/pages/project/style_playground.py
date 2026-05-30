import json
import random
from pathlib import Path
from typing import Any, List, Dict
from nicegui import ui
from database.connection import get_setting
from ui import state
from ui.components.style_chooser_modal import StyleChooserModal

def list_available_workflows() -> list:
    """Discovers .json workflows inside local './workflows' directory."""
    workflows_dir = Path("./workflows")
    workflows_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = Path("./Comfy_Workflows")
    
    found = [f.name for f in workflows_dir.glob("*.json")]
    if legacy_dir.exists():
        found.extend([f.name for f in legacy_dir.glob("*.json")])
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
    # Ensure default.json is generated on disk
    from ui.components.style_chooser_modal import ensure_default_style_exists
    ensure_default_style_exists()
    
    styles_dir = Path("./styles")
    file_path = styles_dir / f"{name}.json"
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                
            associated_wf = data.get("workflow")
            if associated_wf:
                handle_style_workflow_change(associated_wf, clear_overrides=False)
                
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
        "workflow": state.style_selected_workflow,
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


def handle_style_workflow_change(val: str, clear_overrides: bool = True):
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
            client = ComfyClient("127.0.0.1:8188")
            state.style_discovered_params = client.analyze_workflow(wf_json)
            if clear_overrides:
                state.style_workflow_overrides.clear()
            ui.notify(f"Analyzed workflow '{val}'. Discovered {len(state.style_discovered_params)} overrides.", type="info")
        except Exception as e:
            ui.notify(f"Failed to analyze workflow: {str(e)}", type="warning")
            state.style_discovered_params.clear()
            
    render_workflow_overrides_ui.refresh()


def fetch_real_prompts(project_name: str, book_name: str, count: int = 4, prompt_seed: int = 42) -> List[Dict[str, Any]]:
    """Tries to read extracted prompts & quotes from project output directory, tracking full scene metadata."""
    import pandas as pd
    
    csv_paths = [
        Path(f"./output/{project_name}/{book_name}/prompts.csv"),
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
                valid_df = df.dropna(subset=['prompt'])
                valid_df = valid_df[valid_df['prompt'].str.strip().str.lower() != 'none']
                valid_df = valid_df[valid_df['prompt'].str.strip() != '']
                
                if not valid_df.empty:
                    sample_size = min(count, len(valid_df))
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

    if not items:
        from services.prompt_engine import fetch_test_chunks
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


# Global function to access the Style Chooser Modal across all files
style_chooser_instance = None

def open_style_chooser_modal_globally():
    """Triggers the StyleChooserModal instantly from any page view."""
    global style_chooser_instance
    
    def apply_style_preset_cb(selected_name: str):
        state.style_selected_preset = selected_name
        load_style_preset_by_name(selected_name)
        ui.notify(f"Applied visual style: {selected_name}", type="positive")
        # Trigger parent layout update to refresh text fields
        if hasattr(state, 'action_buttons_refresh') and state.action_buttons_refresh:
            state.action_buttons_refresh()
            
    if style_chooser_instance is None:
        style_chooser_instance = StyleChooserModal(on_apply=apply_style_preset_cb)
    
    style_chooser_instance.open(current_selection=state.style_selected_preset)


@ui.refreshable
def render_workflow_overrides_ui():
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
                        current_steps = state.style_workflow_overrides.get(node_id, {}).get("steps", params["steps"])
                        current_cfg = state.style_workflow_overrides.get(node_id, {}).get("cfg", params["cfg"])
                        
                        ui.number(
                            label="Steps",
                            value=current_steps,
                            min=1, max=150, step=1,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "steps", e.value)
                        ).classes('w-full')
                        ui.number(
                            label="CFG Scale",
                            value=current_cfg,
                            min=0.0, max=30.0, step=0.1,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "cfg", e.value)
                        ).classes('w-full')
                    elif node_type == "resolution":
                        current_width = state.style_workflow_overrides.get(node_id, {}).get("width", params["width"])
                        current_height = state.style_workflow_overrides.get(node_id, {}).get("height", params["height"])
                        
                        ui.number(
                            label="Width",
                            value=current_width,
                            min=128, max=4096, step=64,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "width", e.value)
                        ).classes('w-full')
                        ui.number(
                            label="Height",
                            value=current_height,
                            min=128, max=4096, step=64,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "height", e.value)
                        ).classes('w-full')
                    elif node_type == "lora_loader":
                        current_lora_name = state.style_workflow_overrides.get(node_id, {}).get("lora_name", params["lora_name"])
                        current_strength_model = state.style_workflow_overrides.get(node_id, {}).get("strength_model", params["strength_model"])
                        
                        ui.input(
                            label="LoRA Filename",
                            value=current_lora_name,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "lora_name", e.value)
                        ).classes('w-full')
                        ui.number(
                            label="Strength",
                            value=current_strength_model,
                            min=0.0, max=2.0, step=0.1,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "strength_model", e.value)
                        ).classes('w-full')


@ui.refreshable
def render_style_playground_cards():
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
            seed = state.style_test_seeds[idx] if idx < len(state.style_test_seeds) else state.style_image_seed

            card_title = f"Ch {chapter}, Scene {scene_num}"
            full_title_header = f"{card_title} • {book_title}"

            with ui.card().classes('w-full border p-4 rounded-xl shadow-xs gap-3 bg-white'):
                with ui.row().classes('w-full justify-between items-center pb-1 border-b border-dashed'):
                    with ui.column().classes('gap-0'):
                        ui.label(card_title).classes('text-xs font-black text-slate-700 uppercase')
                        ui.label(book_title).classes('text-[9px] text-slate-400 truncate max-w-[150px]')
                    ui.badge(f"Seed: {seed}", color="slate").classes('text-[9px] font-bold')

                from ui.pages.project.dashboard import open_large_image
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


def render_style_playground_tab(project, save_project_settings_cb=None):
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
                on_change=lambda e: handle_style_workflow_change(e.value)
            ).classes('w-full').bind_value(state, 'style_selected_workflow')
            
            # Integrated Style Preset Library Modal button! No ugly dropdowns!
            with ui.row().classes('w-full items-center justify-between p-3 bg-slate-50 rounded border border-slate-200 mt-1'):
                with ui.column().classes('gap-0 flex-1'):
                    ui.label('Active Preset').classes('text-[10px] font-bold text-slate-400 uppercase')
                    ui.label(state.style_selected_preset).classes('text-xs font-bold text-slate-800')
                ui.button(
                    'Browse Library', 
                    icon='photo_library', 
                    on_click=lambda: open_style_chooser_modal_globally()
                ).classes('bg-blue-600 text-white text-xs h-9 font-semibold')
            
            # Quick save row
            with ui.row().classes('w-full items-end gap-2'):
                custom_style_name = ui.input(placeholder="Preset Name", label="Save Current Preset").classes('flex-1')
                ui.button(
                    icon="save",
                    on_click=lambda: (
                        save_style_preset_by_name(custom_style_name.value)
                    )
                ).props('outline').classes('h-10 text-blue-600')

            ui.button(
                'Save Project Settings',
                icon='settings_backup_restore',
                on_click=lambda: save_project_settings_cb(project.id) if save_project_settings_cb else None
            ).classes('w-full bg-slate-100 text-slate-700 hover:bg-slate-200 text-xs font-semibold py-1.5 h-10 border')

            ui.separator()
            
            ui.textarea(
                label="Style Prompt Prefix"
            ).classes('w-full h-24 text-xs').props('outlined').bind_value(state, 'style_prompt_prefix')
            
            ui.textarea(
                label="Style Negative Prompt"
            ).classes('w-full h-24 text-xs').props('outlined').bind_value(state, 'style_negative_prompt')
            
            render_workflow_overrides_ui()
            
        # RIGHT: Visual Style Playground Grid
        with ui.column().classes('w-full gap-4'):
            with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-4'):
                with ui.row().classes('w-full items-center gap-2'):
                    ui.icon('brush', size='sm', color='blue-500')
                    ui.label('Style Visual Playground Settings').classes('text-sm font-bold text-slate-800')
                    
                with ui.row().classes('items-end justify-between gap-4 w-full bg-slate-50 p-4 rounded-lg border'):
                    ui.number(
                        label="Num Images",
                        value=state.style_chunk_count,
                        min=1, max=8, step=1,
                        on_change=lambda e: (setattr(state, 'style_chunk_count', int(e.value)) if e.value is not None else None, draw_style_test_sample(project.name, state.playground_book_selection))
                    ).classes('w-20')

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

                    ui.switch("Random Image Seeds").bind_value(state, 'style_use_random_image_seed').classes('text-xs mb-2')
                    
                    ui.number(
                        label="Image Seed",
                        precision=0
                    ).bind_value(state, 'style_image_seed').classes('w-28').bind_visibility_from(
                        state, 'style_use_random_image_seed', value=False
                    )

                    ui.button(
                        'Test Style Preset',
                        icon='bolt',
                        on_click=lambda: execute_style_playground_batch(project.name)
                    ).classes('bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs px-5 h-10')
            
            render_style_playground_cards()