import json
import random
import io
import base64
import asyncio
from pathlib import Path
from typing import Any, List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
from nicegui import ui
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Book
from ui import state
from ui.components.style_chooser_modal import StyleChooserModal

# Module-level dialog and state holders
contact_sheet_dialog_ref = None
contact_sheet_base64 = ""


def create_contact_sheet(base64_list: list) -> Optional[bytes]:
    """Stitches completed base64 image strings into a single PNG grid stamped with 1A, 1B coordinate labels."""
    images = []
    for img_str in base64_list:
        if not img_str:
            continue
        try:
            if "," in img_str:
                img_str = img_str.split(",", 1)[1]
            img_data = base64.b64decode(img_str)
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        except Exception:
            pass

    if not images:
        return None

    num_imgs = len(images)
    cols = min(4, num_imgs)
    rows = (num_imgs + cols - 1) // cols

    tile_w, tile_h = images[0].size
    grid_img = Image.new("RGB", (cols * tile_w, rows * tile_h), (255, 255, 255))

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols

        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)

        grid_img.paste(img, (c * tile_w, r * tile_h))

        row_num = r + 1
        col_letter = chr(65 + c)
        label_text = f"{row_num}{col_letter}"

        draw = ImageDraw.Draw(grid_img)
        box_w = int(tile_w * 0.1) if tile_w > 400 else 50
        box_h = int(tile_h * 0.08) if tile_h > 400 else 40
        bx0 = c * tile_w
        by0 = r * tile_h
        bx1 = bx0 + box_w
        by1 = by0 + box_h
        
        draw.rectangle([bx0, by0, bx1, by1], fill=(15, 15, 15))

        font = None
        try:
            font = ImageFont.truetype("arial.ttf", size=int(box_h * 0.55))
        except Exception:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size=int(box_h * 0.55))
            except Exception:
                font = ImageFont.load_default()

        draw.text((bx0 + int(box_w * 0.22), by0 + int(box_h * 0.18)), label_text, fill=(255, 255, 255), font=font)

    out_buf = io.BytesIO()
    grid_img.save(out_buf, format="PNG")
    return out_buf.getvalue()


def copy_style_settings_to_clipboard():
    """Formats active style selections, ComfyUI overrides, and test prompts into structured Markdown for an LLM."""
    lines = [
        "### ABI-Pipeline Style Playground Report",
        f"**Active Preset**: {state.style_selected_preset}",
        f"**Base Workflow**: {state.style_selected_workflow}",
        "",
        "#### Style Templates",
        f"**Style Prompt Prefix**:\n```\n{state.style_prompt_prefix}\n```",
        f"**Style Negative Prompt**:\n```\n{state.style_negative_prompt}\n```",
        "",
        "#### Active Parameter Overrides:",
    ]

    if state.style_workflow_overrides:
        for node_id, params in state.style_workflow_overrides.items():
            lines.append(f"- **Node {node_id}**:")
            for k, v in params.items():
                lines.append(f"  - `{k}`: {v}")
    else:
        lines.append("*No overrides applied (Default workflow settings)*")

    lines.append("")
    lines.append("#### Generated Scenes Context Map:")
    
    for idx, item in enumerate(state.style_test_prompts):
        is_dict = isinstance(item, dict)
        chap = item.get("chapter", 1) if is_dict else 1
        sec = item.get("scene", idx + 1) if is_dict else idx + 1
        p_text = item.get("prompt", "") if is_dict else item
        quote = item.get("quote", "") if is_dict else ""
        seed = state.style_test_seeds[idx] if idx < len(state.style_test_seeds) else state.style_image_seed

        row_num = (idx // 4) + 1
        col_letter = chr(65 + (idx % 4))
        label = f"{row_num}{col_letter}"

        lines.append(f"- **[{label}] Ch {chap}, Scene {sec}** (Seed: {seed})")
        if quote:
            lines.append(f"  - *Source Quote*: \"{quote}\"")
        lines.append(f"  - *Extracted Prompt*: {p_text}")

    full_text = "\n".join(lines)
    ui.clipboard.write(full_text)
    ui.notify("Style configuration & prompt context copied to clipboard!", type="positive")


def download_contact_sheet():
    """Generates and triggers download of the stitched contact sheet PNG file."""
    completed_images = [img for img in state.style_test_images if img is not None]
    if not completed_images:
        ui.notify("Generate test images first.", type="warning")
        return
    sheet_bytes = create_contact_sheet(completed_images)
    if sheet_bytes:
        ui.download(sheet_bytes, filename=f"style_grid_{state.style_selected_preset}.png")
    else:
        ui.notify("Failed to assemble download file.", type="negative")


def open_contact_sheet_modal():
    """Compiles completed renderings and opens the contact sheet modal overlay."""
    global contact_sheet_base64, contact_sheet_dialog_ref
    
    completed_images = [img for img in state.style_test_images if img is not None]
    if not completed_images:
        ui.notify("No completed images in current visual feed. Run style test first!", type="warning")
        return

    sheet_bytes = create_contact_sheet(completed_images)
    if not sheet_bytes:
        ui.notify("Failed to stitch contact sheet images.", type="negative")
        return

    encoded = base64.b64encode(sheet_bytes).decode("utf-8")
    contact_sheet_base64 = f"data:image/png;base64,{encoded}"

    if contact_sheet_dialog_ref:
        contact_sheet_dialog_ref.open()
        render_contact_sheet_preview.refresh(contact_sheet_base64)


@ui.refreshable
def render_contact_sheet_preview(base64_data: str):
    """Renders the compiled, high-resolution contact sheet preview inside the modal."""
    if not base64_data:
        ui.label("Stitching preview, please wait...").classes('text-xs text-slate-400')
        return
    ui.image(base64_data).classes('w-full max-h-[70vh] rounded-lg object-contain border shadow-sm')


def reroll_test_scenes(project_name: str, book_name: str):
    """Pulls a completely new set of random test scenes from the active book volume."""
    setattr(state, 'style_prompt_seed', random.randint(100000, 999999))
    draw_style_test_sample(project_name, book_name)
    ui.notify("Pulled new random test scenes!", type="info")


def reroll_image_seeds():
    """Keeps the exact same prompts but randomizes all of their generation seeds."""
    if not state.style_test_prompts:
        ui.notify("No test scenes loaded to reroll.", type="warning")
        return
    state.style_test_seeds = [random.randint(100000, 999999) for _ in range(len(state.style_test_prompts))]
    state.style_test_images = [None] * len(state.style_test_prompts)
    render_style_playground_cards.refresh()
    ui.notify("Randomized generation seeds (prompts kept). Ready to re-test!", type="info")


async def regenerate_single_card(project_name: str, idx: int):
    """Randomizes the seed for a single scene card and immediately triggers independent rendering."""
    if idx >= len(state.style_test_prompts):
        return

    # Randomize only this scene's seed
    state.style_test_seeds[idx] = random.randint(100000, 999999)
    state.style_test_images[idx] = "LOADING"
    render_style_playground_cards.refresh()

    comfy_url = get_setting("comfy_url", "127.0.0.1:8188")
    if "http" in comfy_url:
        comfy_url = comfy_url.replace("http://", "").replace("https://", "").strip("/")

    wf_path = Path("./workflows") / state.style_selected_workflow
    if not wf_path.exists():
        wf_path = Path("./Comfy_Workflows") / state.style_selected_workflow

    if not wf_path.exists():
        state.style_test_images[idx] = None
        render_style_playground_cards.refresh()
        ui.notify(f"Workflow '{state.style_selected_workflow}' not found.", type="negative")
        return

    try:
        with open(wf_path, "r") as f:
            workflow_json = json.load(f)
    except Exception as e:
        state.style_test_images[idx] = None
        render_style_playground_cards.refresh()
        ui.notify(f"Failed to load workflow: {str(e)}", type="negative")
        return

    from services.comfy_client import ComfyClient
    client = ComfyClient(comfy_url)

    item = state.style_test_prompts[idx]
    prompt_text = item["prompt"] if isinstance(item, dict) else item
    seed = state.style_test_seeds[idx]

    def run_single():
        return client.generate_image_sync(
            workflow_json=workflow_json,
            prompt_text=prompt_text,
            neg_prompt_text=state.style_negative_prompt,
            seed=seed,
            overrides=state.style_workflow_overrides,
            prefix=state.style_prompt_prefix
        )

    try:
        img_bytes, logs = await asyncio.to_thread(run_single)
        if img_bytes:
            encoded = base64.b64encode(img_bytes).decode("utf-8")
            state.style_test_images[idx] = f"data:image/png;base64,{encoded}"
        else:
            state.style_test_images[idx] = None
            ui.notify("Single render produced no output.", type="warning")

        for line in logs.split("\n"):
            if line.strip():
                state.add_console_log(line)
    except Exception as e:
        state.style_test_images[idx] = None
        state.add_console_log(f"[Style-Playground] Single card render failed: {str(e)}")
        ui.notify(f"Render failed: {str(e)}", type="negative")
    finally:
        render_style_playground_cards.refresh()


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


style_chooser_instance = None

def open_style_chooser_modal_globally():
    """Triggers the StyleChooserModal instantly from any page view."""
    global style_chooser_instance
    
    def apply_style_preset_cb(selected_name: str):
        state.style_selected_preset = selected_name
        load_style_preset_by_name(selected_name)
        ui.notify(f"Applied visual style: {selected_name}", type="positive")
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
                        current_sampler = state.style_workflow_overrides.get(node_id, {}).get("sampler_name", params.get("sampler_name", "euler"))
                        current_scheduler = state.style_workflow_overrides.get(node_id, {}).get("scheduler", params.get("scheduler", "normal"))
                        
                        ui.number(
                            label="Steps",
                            value=current_steps,
                            min=1, max=150, step=1,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "steps", int(e.value) if e.value is not None else None)
                        ).classes('w-full')
                        ui.number(
                            label="CFG Scale",
                            value=current_cfg,
                            min=0.0, max=30.0, step=0.1,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "cfg", e.value)
                        ).classes('w-full')
                        ui.input(
                            label="Sampler Name",
                            value=current_sampler,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "sampler_name", e.value)
                        ).classes('w-full')
                        ui.input(
                            label="Scheduler",
                            value=current_scheduler,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "scheduler", e.value)
                        ).classes('w-full')
                        
                    elif node_type == "resolution":
                        current_width = state.style_workflow_overrides.get(node_id, {}).get("width", params["width"])
                        current_height = state.style_workflow_overrides.get(node_id, {}).get("height", params["height"])
                        
                        ui.number(
                            label="Width",
                            value=current_width,
                            min=128, max=4096, step=64,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "width", int(e.value) if e.value is not None else None)
                        ).classes('w-full')
                        ui.number(
                            label="Height",
                            value=current_height,
                            min=128, max=4096, step=64,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "height", int(e.value) if e.value is not None else None)
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

                    elif node_type == "model_loader":
                        param_key = params.get("model_param_key", "ckpt_name")
                        current_model_name = state.style_workflow_overrides.get(node_id, {}).get(param_key, params.get(param_key, ""))
                        
                        ui.input(
                            label=f"Model Filename ({param_key})",
                            value=current_model_name,
                            on_change=lambda e, nid=node_id, pk=param_key: update_override_state(nid, pk, e.value)
                        ).classes('w-full')

                    elif node_type == "clip_loader":
                        param_key = params.get("clip_param_key", "clip_name")
                        current_clip_name = state.style_workflow_overrides.get(node_id, {}).get(param_key, params.get(param_key, ""))
                        
                        ui.input(
                            label=f"CLIP Filename ({param_key})",
                            value=current_clip_name,
                            on_change=lambda e, nid=node_id, pk=param_key: update_override_state(nid, pk, e.value)
                        ).classes('w-full')

                    elif node_type == "vae_loader":
                        current_vae_name = state.style_workflow_overrides.get(node_id, {}).get("vae_name", params.get("vae_name", ""))
                        
                        ui.input(
                            label="VAE Filename",
                            value=current_vae_name,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "vae_name", e.value)
                        ).classes('w-full')


@ui.refreshable
def render_style_playground_cards(project_name: str = ""):
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
                    
                    # Clickable Interactive Seed Badge
                    ui.badge(f"Seed: {seed}", color="slate") \
                        .classes('text-[10px] font-bold cursor-pointer hover:bg-slate-700 transition-all') \
                        .on('click', lambda _, i=idx: regenerate_single_card(project_name, i)) \
                        .tooltip('Click to randomize seed and regenerate just this scene!')

                from ui.pages.project.dashboard import open_large_image
                
                if img_data == "LOADING":
                    with ui.column().classes('w-full h-48 items-center justify-center bg-slate-50 rounded-lg border border-dashed'):
                        ui.spinner(size='md', color='blue')
                        ui.label("Rendering single card...").classes('text-[9px] text-slate-400 mt-1')
                elif img_data:
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
    global contact_sheet_dialog_ref
    
    # Query current volume listings inside database index
    with Session(engine) as session:
        books = session.exec(select(Book).where(Book.project_id == project.id)).all()
    book_names = [b.name for b in books]
    
    # Sync default selection
    if books and (not state.playground_book_selection or state.playground_book_selection not in book_names):
        state.playground_book_selection = books[0].name
        
    # Auto-initialize visual test prompts so workspace is never left empty
    if not state.style_test_prompts and state.playground_book_selection:
        draw_style_test_sample(project.name, state.playground_book_selection)

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
            
            # Integrated Style Preset Library Modal button!
            with ui.row().classes('w-full items-center justify-between p-3 bg-slate-50 rounded border border-slate-200 mt-1'):
                with ui.column().classes('gap-0 flex-1'):
                    ui.label('Active Preset').classes('text-[10px] font-bold text-slate-400 uppercase')
                    ui.label().classes('text-xs font-bold text-slate-800').bind_text_from(state, 'style_selected_preset')
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
            ).classes('w-full text-xs').props('outlined autogrow').bind_value(state, 'style_prompt_prefix')
            
            ui.textarea(
                label="Style Negative Prompt"
            ).classes('w-full text-xs').props('outlined autogrow').bind_value(state, 'style_negative_prompt')
            
            render_workflow_overrides_ui()
            
            
        # RIGHT: Visual Style Playground Grid
        with ui.column().classes('w-full gap-4'):
            with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-4'):
                with ui.row().classes('w-full items-center gap-2'):
                    ui.icon('tune', size='sm', color='blue-500')
                    ui.label('Style Visual Playground Settings').classes('text-sm font-bold text-slate-800')
                    
                with ui.column().classes('w-full gap-4 bg-slate-50 p-4 rounded-lg border'):
                    # Step 1: Volume Select & Number of scenes slider
                    with ui.row().classes('w-full items-center justify-between gap-4'):
                        ui.select(
                            options=book_names,
                            label="Book/Volume Source",
                            on_change=lambda e: (setattr(state, 'playground_book_selection', e.value), draw_style_test_sample(project.name, e.value))
                        ).classes('w-48 bg-white').bind_value(state, 'playground_book_selection')
                        
                        ui.slider(
                            min=1, max=8, step=1,
                            on_change=lambda e: (
                                setattr(state, 'style_chunk_count', int(e.value)) if e.value is not None else None, 
                                draw_style_test_sample(project.name, state.playground_book_selection)
                            )
                        ).classes('flex-1 mx-2').props('label-always').bind_value(state, 'style_chunk_count')
                        ui.label('Test Count').classes('text-xs font-bold text-slate-400')

                    # Step 2: Reroll controllers
                    with ui.row().classes('w-full gap-2 justify-end mt-1'):
                        ui.button(
                            'Reroll Scenes',
                            icon='casino',
                            on_click=lambda: reroll_test_scenes(project.name, state.playground_book_selection)
                        ).classes('bg-slate-700 text-white text-xs font-semibold h-9') \
                         .tooltip('Pulls a completely new set of random visual scenes from the selected volume')
                        
                        ui.button(
                            'Reroll Seeds',
                            icon='refresh',
                            on_click=reroll_image_seeds
                        ).classes('bg-slate-700 text-white text-xs font-semibold h-9') \
                         .tooltip('Keeps current text prompts but randomizes all of their image generation seeds')

                    ui.separator()

                    # Step 3: Seed locks & Main execution action
                    with ui.row().classes('w-full items-center justify-between gap-4'):
                        with ui.row().classes('items-center gap-4'):
                            ui.switch("Random Seeds").bind_value(state, 'style_use_random_image_seed').classes('text-xs')
                            
                            ui.number(
                                label="Manual Image Seed",
                                precision=0
                            ).bind_value(state, 'style_image_seed').classes('w-32 bg-white').props('outlined dense').bind_visibility_from(
                                state, 'style_use_random_image_seed', value=False
                            )

                        ui.button(
                            'Test Style Preset',
                            icon='bolt',
                            on_click=lambda: execute_style_playground_batch(project.name)
                        ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-5 h-10') \
                         .tooltip('Submits all active prompts to ComfyUI for rendering')

                # --- AI Integration Toolkit Panel ---
                with ui.row().classes('w-full items-center justify-between bg-blue-50/50 p-3 rounded-lg border border-blue-100/50 mt-1'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('smart_toy', color='blue', size='sm')
                        ui.label('AI Integration Toolkit').classes('text-xs font-black text-slate-700 uppercase tracking-wide')
                    
                    with ui.row().classes('gap-2'):
                        ui.button(
                            'Copy Prompt Pack',
                            icon='content_copy',
                            on_click=copy_style_settings_to_clipboard
                        ).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs font-semibold h-9')
                        
                        ui.button(
                            'Generate Contact Sheet',
                            icon='grid_view',
                            on_click=open_contact_sheet_modal
                        ).classes('bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold h-9')
            
            render_style_playground_cards(project.name)

    # Declare Dialog Overlay inside workspace hierarchy
    with ui.dialog() as contact_sheet_dialog:
        with ui.card().classes('w-full max-w-4xl p-6 rounded-xl gap-4 bg-white'):
            with ui.row().classes('w-full justify-between items-center pb-2 border-b'):
                with ui.column().classes('gap-0'):
                    ui.label('Generated AI Contact Sheet').classes('text-base font-bold text-slate-800')
                    ui.label('Grid coordinate labels (1A, 1B...) are stamped on the image. Perfect for sending to LLMs.').classes('text-xs text-slate-500')
                ui.button(
                    'Download Contact Sheet',
                    icon='download',
                    on_click=download_contact_sheet
                ).classes('bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold h-9')
            
            render_contact_sheet_preview(contact_sheet_base64)
            
            with ui.row().classes('w-full justify-between items-center pt-2 border-t text-[10px] text-slate-400'):
                ui.label('Tip: Right-click the image to copy it directly, or click Download.')
                ui.button('Close', on_click=contact_sheet_dialog.close).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs')

    contact_sheet_dialog_ref = contact_sheet_dialog