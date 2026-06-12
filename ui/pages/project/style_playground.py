import json
import random
import io
import base64
import asyncio
import httpx
from pathlib import Path
from typing import Any, List, Dict, Optional, Callable
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

# Cache holders for ComfyUI API selections and local Benchmarked LoRAs
comfy_options_cache: Dict[str, List[str]] = {}
associated_loras_list: List[Dict[str, Any]] = []

lora_chooser_dialog_ref = None
chooser_active_node_id = None
chooser_selected_lora_id = None

# Track expansion state of auto-discovered parameter workflow nodes
expansion_states: Dict[str, bool] = {}

# Keep track of the active workflow that was last successfully analyzed
_last_analyzed_workflow: Optional[str] = None


def get_node_class_type(node_id: str) -> Optional[str]:
    """Resolves the raw class_type of a node inside the active ComfyUI workflow JSON."""
    val = state.style_selected_workflow
    if not val:
        return None
    wf_path = Path("./workflows") / val
    if not wf_path.exists():
        wf_path = Path("./Comfy_Workflows") / val
    if wf_path.exists():
        try:
            with open(wf_path, "r") as f:
                wf_json = json.load(f)
            node_data = wf_json.get(node_id, {})
            return node_data.get("class_type")
        except Exception:
            pass
    return None


def load_associated_loras():
    """Filters, sorts, and loads LoRAs associated with the current workflow from output/_lora_library/loras.csv."""
    global associated_loras_list
    associated_loras_list.clear()
    
    lora_csv = Path("./output/_lora_library/loras.csv")
    if not lora_csv.exists():
        return
        
    try:
        import csv
        with open(lora_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            rows = list(reader)
            
            # Filter for current selected workflow
            filtered = [
                row for row in rows 
                if row.get("workflow") == state.style_selected_workflow
            ]
            
            # Sort: favorites at the top (favorite == "True"), then alphabetically by filename
            def sort_key(item):
                is_fav = item.get("favorite") == "True"
                filename = Path(item.get("lora_path", "")).name.lower()
                return (not is_fav, filename)
                
            associated_loras_list = sorted(filtered, key=sort_key)
    except Exception as e:
        print(f"[Style-Playground] Error reading loras.csv: {str(e)}")


async def async_load_comfy_and_lora_choices():
    """Fetches model and sampler selection options from ComfyUI API object_info in a background task."""
    global comfy_options_cache
    comfy_options_cache.clear()
    
    # Refresh LoRA options from local loras.csv
    load_associated_loras()
    
    comfy_url = get_setting("comfy_url", "127.0.0.1:8188")
    if "http" in comfy_url:
        comfy_url = comfy_url.replace("http://", "").replace("https://", "").strip("/")
        
    for node_id, data in list(state.style_discovered_params.items()):
        node_type = data["type"]
        params = data["params"]
        
        class_type = get_node_class_type(node_id)
        if not class_type:
            continue
            
        param_keys = []
        if node_type == "model_loader":
            param_keys = [params.get("model_param_key", "ckpt_name")]
        elif node_type == "lora_loader":
            param_keys = ["lora_name"]
        elif node_type == "clip_loader":
            param_keys = [params.get("clip_param_key", "clip_name")]
        elif node_type == "vae_loader":
            param_keys = ["vae_name"]
        elif node_type == "sampler":
            # Automatically request lists of both samplers and schedulers from node class
            param_keys = ["sampler_name", "scheduler"]
            
        for param_key in param_keys:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"http://{comfy_url}/object_info/{class_type}", timeout=2.0)
                    if resp.status_code == 200:
                        res_json = resp.json()
                        choices = res_json.get(class_type, {}).get("input", {}).get("required", {}).get(param_key, [[]])[0]
                        if isinstance(choices, list) and choices:
                            comfy_options_cache[f"{node_id}:{param_key}"] = sorted(choices)
            except Exception:
                pass
                
    render_workflow_overrides_ui.refresh()


def open_lora_chooser_modal(node_id: str):
    """Pre-selects the active LoRA and opens the Visual LoRA Chooser overlay."""
    global chooser_active_node_id, chooser_selected_lora_id
    chooser_active_node_id = node_id
    
    current_override_lora = state.style_workflow_overrides.get(node_id, {}).get("lora_name")
    chooser_selected_lora_id = None
    if current_override_lora:
        for item in associated_loras_list:
            if item["lora_path"] == current_override_lora:
                chooser_selected_lora_id = item["id"]
                break
                
    if lora_chooser_dialog_ref:
        lora_chooser_dialog_ref.open()
        render_lora_chooser_content.refresh()


@ui.refreshable
def render_lora_chooser_content():
    """Visual selection matrix showcasing trigger words and 2x2 grids of completed sample images."""
    global chooser_selected_lora_id, chooser_active_node_id
    
    if not associated_loras_list:
        with ui.column().classes('w-full items-center justify-center p-12 text-slate-400'):
            ui.icon('grid_off', size='lg')
            ui.label("No benchmarked LoRAs found for this workflow.").classes('text-xs text-center mt-2')
            ui.label("Run benchmarks in the LoRA Contact Sheets tool to see them here.").classes('text-[10px] text-slate-400 text-center')
        return

    if not chooser_selected_lora_id and associated_loras_list:
        chooser_selected_lora_id = associated_loras_list[0]["id"]

    selected_lora = next((l for l in associated_loras_list if l["id"] == chooser_selected_lora_id), None)

    with ui.grid(columns='240px 1fr').classes('w-full h-[500px] gap-4'):
        # LEFT SIDEBAR: Benchmarked LoRA list
        with ui.column().classes('border-r pr-2 gap-1 overflow-y-auto h-full'):
            for lora in associated_loras_list:
                is_selected = lora["id"] == chooser_selected_lora_id
                bg_color = "bg-blue-50 border-blue-200 text-blue-700 font-bold" if is_selected else "hover:bg-slate-50 border-transparent text-slate-700"
                
                def select_lora_item(lid=lora["id"]):
                    global chooser_selected_lora_id
                    chooser_selected_lora_id = lid
                    render_lora_chooser_content.refresh()
                    
                with ui.row().classes(f'w-full p-2 rounded-lg border cursor-pointer transition-colors items-center justify-between {bg_color}') \
                        .on('click', select_lora_item):
                    with ui.row().classes('items-center gap-1 truncate flex-1'):
                        if lora.get("favorite") == "True":
                            ui.icon('star', color='amber-500', size='14px').classes('flex-shrink-0')
                        ui.label(Path(lora["lora_path"]).name).classes('text-xs truncate')
                    ui.label(f"str: {float(lora['strength']):.1f}").classes('text-[9px] text-slate-400 flex-shrink-0')

        # RIGHT PREVIEW PANEL: Sample Grid & Selection Metadata
        with ui.column().classes('h-full flex flex-col gap-3 justify-between'):
            if selected_lora:
                with ui.column().classes('w-full gap-2 flex-1 min-h-0'):
                    ui.label(Path(selected_lora["lora_path"]).name).classes('text-base font-bold text-slate-800')
                    
                    triggers = selected_lora.get("triggers", "").strip()
                    if triggers and triggers != ".":
                        with ui.row().classes('w-full p-2 bg-blue-50 rounded border border-blue-100 text-xs items-center gap-1'):
                            ui.icon('bolt', color='blue', size='xs')
                            ui.label("Triggers:").classes('font-bold text-blue-800')
                            ui.label(triggers).classes('font-mono font-semibold text-blue-700')
                    else:
                        ui.label("(No trigger words configured for this LoRA)").classes('text-xs text-slate-400 italic')
                        
                    # Retrieve first 4 samples for styling demonstration
                    lora_dir = Path("./output/_lora_library") / selected_lora["id"]
                    sample_images = []
                    if lora_dir.exists():
                        sample_images = sorted(list(lora_dir.glob("*.png")))[:4]
                        
                    if sample_images:
                        with ui.grid(columns=2).classes('w-full gap-3 mt-1 overflow-y-auto flex-1 p-1'):
                            for img_path in sample_images:
                                ui.image(str(img_path)).props('fit=contain').classes('w-full rounded border shadow-sm')
                    else:
                        with ui.column().classes('w-full flex-1 items-center justify-center border border-dashed rounded bg-slate-50 text-slate-400'):
                            ui.icon('photo_library', size='lg')
                            ui.label("No benchmark images rendered yet for this LoRA.").classes('text-xs text-center')

                # Dialog action panel
                with ui.row().classes('w-full justify-end gap-3 border-t pt-2 flex-shrink-0'):
                    ui.button('Cancel', on_click=lambda: lora_chooser_dialog_ref.close() if lora_chooser_dialog_ref else None).props('flat color=slate')
                    
                    def apply_selection():
                        if selected_lora and chooser_active_node_id:
                            update_override_state(chooser_active_node_id, "lora_name", selected_lora["lora_path"])
                            update_override_state(chooser_active_node_id, "strength_model", float(selected_lora["strength"]))
                            
                            triggers_to_add = selected_lora.get("triggers", "").strip()
                            if triggers_to_add and triggers_to_add != ".":
                                if triggers_to_add not in state.style_prompt_prefix:
                                    cleaned_prefix = state.style_prompt_prefix.strip()
                                    if cleaned_prefix and not cleaned_prefix.endswith(","):
                                        cleaned_prefix += ","
                                    state.style_prompt_prefix = f"{cleaned_prefix} {triggers_to_add}, ".strip().replace("  ", " ")
                                    ui.notify(f"Selected LoRA and appended trigger words: '{triggers_to_add}'", type="positive")
                                else:
                                    ui.notify("Selected LoRA successfully!", type="positive")
                            else:
                                ui.notify("Selected LoRA successfully!", type="positive")
                                
                            lora_chooser_dialog_ref.close()
                            render_workflow_overrides_ui.refresh()
                            
                    ui.button('Accept & Apply', on_click=apply_selection).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs h-9')

                    

def create_contact_sheet(base64_list: list, overlay_text: str = "") -> Optional[bytes]:
    """Stitches completed base64 image strings into a single PNG grid stamped with sequential digits (1, 2, 3...)
    and an optional custom bottom-center evaluation text banner."""
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

        # Render simple sequential numbers (1, 2, 3...) instead of row/col coordinates
        label_text = str(idx + 1)

        draw = ImageDraw.Draw(grid_img)
        box_w = int(tile_w * 0.08) if tile_w > 400 else 40
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

        # Adjust text layout horizontally for sequential digits
        draw.text((bx0 + int(box_w * 0.35), by0 + int(box_h * 0.18)), label_text, fill=(255, 255, 255), font=font)

    # Apply global custom evaluation text overlay bottom-center
    clean_overlay = overlay_text.strip()
    if clean_overlay:
        draw = ImageDraw.Draw(grid_img)
        overlay_font_size = max(16, int(grid_img.height * 0.035))
        
        overlay_font = None
        try:
            overlay_font = ImageFont.truetype("arial.ttf", size=overlay_font_size)
        except Exception:
            try:
                overlay_font = ImageFont.truetype("DejaVuSans.ttf", size=overlay_font_size)
            except Exception:
                overlay_font = ImageFont.load_default()

        # Measure text boundaries safely
        try:
            left, top, right, bottom = draw.textbbox((0, 0), clean_overlay, font=overlay_font)
            text_w = right - left
            text_h = bottom - top
        except AttributeError:
            # Fallback for older PIL installations
            text_w = len(clean_overlay) * (overlay_font_size * 0.6)
            text_h = overlay_font_size

        padding_x = 24
        padding_y = 12
        rect_w = text_w + padding_x * 2
        rect_h = text_h + padding_y * 2

        # Center horizontally, stick to the lower edge
        rx0 = (grid_img.width - rect_w) // 2
        ry0 = grid_img.height - rect_h - 24
        rx1 = rx0 + rect_w
        ry1 = ry0 + rect_h

        # Dark background banner and white text overlay
        draw.rectangle([rx0, ry0, rx1, ry1], fill=(15, 15, 15))
        draw.text((rx0 + padding_x, ry0 + padding_y), clean_overlay, fill=(255, 255, 255), font=overlay_font)

    out_buf = io.BytesIO()
    grid_img.save(out_buf, format="PNG")
    return out_buf.getvalue()


def _get_resolved_workflow_settings() -> List[str]:
    """Helper to compile currently adjustable settings, merging workflow defaults and overrides."""
    lines = []
    if not state.style_discovered_params:
        return ["*No adjustable parameters detected or loaded for this workflow.*"]
    
    for node_id, data in state.style_discovered_params.items():
        node_title = data["title"]
        node_type = data["type"]
        params = data["params"]
        overrides = state.style_workflow_overrides.get(node_id, {})
        
        lines.append(f"- **{node_title} (ID: {node_id})**:")
        if node_type == "sampler":
            steps = overrides.get("steps", params.get("steps"))
            cfg = overrides.get("cfg", params.get("cfg"))
            sampler = overrides.get("sampler_name", params.get("sampler_name", "euler"))
            scheduler = overrides.get("scheduler", params.get("scheduler", "normal"))
            lines.append(f"  - `steps`: {steps}")
            lines.append(f"  - `cfg`: {cfg}")
            lines.append(f"  - `sampler_name`: {sampler}")
            lines.append(f"  - `scheduler`: {scheduler}")
        elif node_type == "resolution":
            w = overrides.get("width", params.get("width"))
            h = overrides.get("height", params.get("height"))
            lines.append(f"  - `width`: {w}")
            lines.append(f"  - `height`: {h}")
        elif node_type == "lora_loader":
            lora_name = overrides.get("lora_name", params.get("lora_name"))
            strength = overrides.get("strength_model", params.get("strength_model"))
            lines.append(f"  - `lora_name`: {lora_name}")
            lines.append(f"  - `strength_model`: {strength}")
        elif node_type == "model_loader":
            pk = params.get("model_param_key", "ckpt_name")
            model = overrides.get(pk, params.get(pk, ""))
            lines.append(f"  - `{pk}`: {model}")
        elif node_type == "clip_loader":
            pk = params.get("clip_param_key", "clip_name")
            clip = overrides.get(pk, params.get(pk, ""))
            lines.append(f"  - `{pk}`: {clip}")
        elif node_type == "vae_loader":
            vae = overrides.get("vae_name", params.get("vae_name", ""))
            lines.append(f"  - `vae_name`: {vae}")
            
    return lines


def copy_style_settings_to_clipboard(project_name: str = "Active Project"):
    """Formats active style selections, ComfyUI parameters, and test prompts into structured Markdown."""
    resolved_settings = "\n".join(_get_resolved_workflow_settings())
    
    lines = [
        "### ABI-Pipeline Style Playground Report",
        f"**Project**: {project_name}",
        f"**Active Preset**: {state.style_selected_preset}",
        f"**Base Workflow**: {state.style_selected_workflow}",
        "",
        "#### Style Templates",
        f"**Style Prompt Prefix**:\n```\n{state.style_prompt_prefix}\n```",
        f"**Style Prompt Suffix**:\n```\n{state.style_prompt_suffix}\n```",
        f"**Style Negative Prompt**:\n```\n{state.style_negative_prompt}\n```",
        "",
        "#### Adjustable Workflow Settings:",
        resolved_settings,
        "",
        "#### Generated Scenes Context Map:",
    ]

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


def copy_style_primer_to_clipboard(project_name: str):
    """Formats a diagnostic visual style framework primer and copies it to clipboard."""
    resolved_settings = "\n".join(_get_resolved_workflow_settings())
    
    scenes_lines = []
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

        scenes_lines.append(f"- **[{label}] Ch {chap}, Scene {sec}** (Seed: {seed})")
        if quote:
            scenes_lines.append(f"  - *Source Quote*: \"{quote}\"")
        scenes_lines.append(f"  - *Visual Prompt*: {p_text}")
    
    scenes_str = "\n".join(scenes_lines) if scenes_lines else "*No test scenes generated yet.*"

    primer_text = (
        f"I am using a local AI book illustration pipeline called ABI-Pipeline for my project: '{project_name}'. "
        "I need your expert help tuning and debugging my active visual style preset, ComfyUI settings, and descriptive prompts.\n\n"
        "Below is the current visual orchestration configuration and the resulting visual prompts extracted from our transcript:\n\n"
        "### ARTISTIC STYLE TEMPLATES:\n"
        f"- **Style Prompt Prefix (Appended at front)**:\n  ```text\n  {state.style_prompt_prefix}\n  ```\n"
        f"- **Style Prompt Suffix (Appended at end)**:\n  ```text\n  {state.style_prompt_suffix}\n  ```\n"
        f"- **Style Negative Prompt (Avoid list)**:\n  ```text\n  {state.style_negative_prompt}\n  ```\n\n"
        "--------------------------------------------------------------------------------\n\n"
        "### ADJUSTABLE WORKFLOW SETTINGS:\n"
        f"**Base Workflow**: {state.style_selected_workflow}\n"
        f"{resolved_settings}\n\n"
        "--------------------------------------------------------------------------------\n\n"
        "### CURRENT RUN DATA (PROMPTS & METADATA):\n"
        f"**Volume Selected**: {state.playground_book_selection}\n"
        f"{scenes_str}\n\n"
        "--------------------------------------------------------------------------------\n\n"
        "### MY SPECIFIC ADJUSTMENT REQUEST:\n"
        "[Insert your request here, e.g. 'My rendered images look way too realistic, and the lighting is harsh. "
        "Looking at my Style Prefix/Suffix and active adjustable parameters (like CFG and sampler choice above), what changes "
        "do you recommend to achieve a softer, hand-drawn look? Please suggest updated style templates or parameter values!']"
    )
    
    ui.clipboard.write(primer_text)
    ui.notify("Style AI Primer copied! Paste this first, then upload or paste your results.", type="positive", icon="psychology")


def download_contact_sheet():
    """Generates and triggers download of the stitched contact sheet PNG file."""
    completed_images = [img for img in state.style_test_images if img is not None]
    if not completed_images:
        ui.notify("Generate test images first.", type="warning")
        return
    sheet_bytes = create_contact_sheet(completed_images, state.style_contact_sheet_overlay)
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

    sheet_bytes = create_contact_sheet(completed_images, state.style_contact_sheet_overlay)
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
    ui.image(base64_data).props('fit=contain').classes('w-full max-h-[70vh] rounded-lg border shadow-sm bg-slate-50/20')


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
            prefix=state.style_prompt_prefix,
            suffix=state.style_prompt_suffix
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
    
    # Pre-populate the save preset input box with the active style's name
    state.style_preset_save_name = name
    
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
            state.style_prompt_suffix = data.get("prompt_suffix", "")
            state.style_negative_prompt = data.get("negative_prompt", "")
            state.style_workflow_overrides = data.get("overrides", {})
        except Exception:
            pass
            
    render_workflow_overrides_ui.refresh()


def save_style_preset_by_name(name: str):
    """Saves style prefix, suffix & negative prompt variables into an on-disk style preset JSON file."""
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
        "prompt_suffix": state.style_prompt_suffix,
        "negative_prompt": state.style_negative_prompt,
        "overrides": state.style_workflow_overrides
    }
    try:
        with open(styles_dir / f"{name_clean}.json", "w") as f:
            json.dump(data, f, indent=2)
        state.style_selected_preset = name_clean
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
    global _last_analyzed_workflow
    if not val:
        return
        
    # Guard: If this workflow is already analyzed and matches the requested workflow,
    # skip to avoid clearing overrides or performing redundant file reads/analysis.
    if _last_analyzed_workflow == val and state.style_discovered_params:
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
            
            comfy_url = get_setting("comfy_url", "127.0.0.1:8188")
            if "http" in comfy_url:
                comfy_url = comfy_url.replace("http://", "").replace("https://", "").strip("/")
            client = ComfyClient(comfy_url)
            
            state.style_discovered_params = client.analyze_workflow(wf_json)
            if clear_overrides:
                state.style_workflow_overrides.clear()
            
            # Commit the successfully analyzed workflow name
            _last_analyzed_workflow = val
            ui.notify(f"Analyzed workflow '{val}'. Discovered {len(state.style_discovered_params)} overrides.", type="info")
            
            # Fire-and-forget back-end loading of drop-down options from ComfyUI API and CSV
            asyncio.create_task(async_load_comfy_and_lora_choices())
            
        except Exception as e:
            ui.notify(f"Failed to analyze workflow: {str(e)}", type="warning")
            state.style_discovered_params.clear()
            _last_analyzed_workflow = None
            
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
                prefix=state.style_prompt_prefix,
                suffix=state.style_prompt_suffix
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
            
            # Ensure we track the expanded state across UI refreshes
            if node_id not in expansion_states:
                expansion_states[node_id] = False
                
            with ui.expansion(f"{node_title} (ID: {node_id})").classes('w-full border rounded bg-white text-xs') as exp:
                exp.bind_value(expansion_states, node_id)
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
                        
                        # Sampler Dropdown with API lookup
                        comfy_sampler_key = f"{node_id}:sampler_name"
                        if comfy_sampler_key in comfy_options_cache:
                            ui.select(
                                options=comfy_options_cache[comfy_sampler_key],
                                value=current_sampler if current_sampler in comfy_options_cache[comfy_sampler_key] else None,
                                label="Sampler Name",
                                on_change=lambda e, nid=node_id: (update_override_state(nid, "sampler_name", e.value), render_workflow_overrides_ui.refresh())
                            ).classes('w-full')
                        else:
                            ui.input(
                                label="Sampler Name",
                                value=current_sampler,
                                on_change=lambda e, nid=node_id: update_override_state(nid, "sampler_name", e.value)
                            ).classes('w-full')
                            
                        # Scheduler Dropdown with API lookup
                        comfy_scheduler_key = f"{node_id}:scheduler"
                        if comfy_scheduler_key in comfy_options_cache:
                            ui.select(
                                options=comfy_options_cache[comfy_scheduler_key],
                                value=current_scheduler if current_scheduler in comfy_options_cache[comfy_scheduler_key] else None,
                                label="Scheduler",
                                on_change=lambda e, nid=node_id: (update_override_state(nid, "scheduler", e.value), render_workflow_overrides_ui.refresh())
                            ).classes('w-full')
                        else:
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
                        
                        # Render ComfyUI dropdown alongside the visual palette chooser button
                        comfy_lora_key = f"{node_id}:lora_name"
                        if comfy_lora_key in comfy_options_cache:
                            with ui.row().classes('w-full items-end gap-2'):
                                ui.select(
                                    options=comfy_options_cache[comfy_lora_key],
                                    value=current_lora_name if current_lora_name in comfy_options_cache[comfy_lora_key] else None,
                                    label="Select LoRA",
                                    on_change=lambda e, nid=node_id: (update_override_state(nid, "lora_name", e.value), render_workflow_overrides_ui.refresh())
                                ).classes('flex-1')
                                
                                ui.button(
                                    icon="palette",
                                    on_click=lambda nid=node_id: open_lora_chooser_modal(nid)
                                ).props('flat round size=md').classes('text-blue-600 mb-1').tooltip("Open Visual LoRA Chooser")
                        else:
                            with ui.row().classes('w-full items-end gap-2'):
                                ui.input(
                                    label="LoRA Filename",
                                    value=current_lora_name,
                                    on_change=lambda e, nid=node_id: update_override_state(nid, "lora_name", e.value)
                                ).classes('flex-1')
                                
                                ui.button(
                                    icon="palette",
                                    on_click=lambda nid=node_id: open_lora_chooser_modal(nid)
                                ).props('flat round size=md').classes('text-blue-600 mb-1').tooltip("Open Visual LoRA Chooser")
                        
                        # Render triggers section if any benchmarked lora is selected
                        active_bench = None
                        for item in associated_loras_list:
                            if item["lora_path"] == current_lora_name:
                                active_bench = item
                                break
                                
                        if active_bench and active_bench.get("triggers"):
                            triggers_text = active_bench["triggers"]
                            with ui.row().classes('w-full items-center justify-between p-2 bg-blue-50/50 rounded border border-blue-100 text-xs mt-1'):
                                with ui.column().classes('gap-0 flex-1 min-w-0'):
                                    ui.label("Trigger Words:").classes('text-[9px] font-bold text-blue-400 uppercase')
                                    ui.label(triggers_text).classes('font-mono font-semibold text-blue-700 truncate w-full')
                                
                                with ui.row().classes('gap-1 flex-shrink-0'):
                                    ui.button(
                                        icon="content_copy",
                                        on_click=lambda t=triggers_text: (ui.clipboard.write(t), ui.notify("Trigger words copied to clipboard!", type="positive"))
                                    ).props('flat dense').classes('text-blue-600').tooltip("Copy triggers to clipboard")
                                    
                                    def add_to_prefix_val(t=triggers_text):
                                        if t not in state.style_prompt_prefix:
                                            cleaned_prefix = state.style_prompt_prefix.strip()
                                            if cleaned_prefix and not cleaned_prefix.endswith(","):
                                                cleaned_prefix += ","
                                            state.style_prompt_prefix = f"{cleaned_prefix} {t}, ".strip().replace("  ", " ")
                                            ui.notify("Added triggers to Style Prompt Prefix!", type="positive")
                                            
                                    ui.button(
                                        icon="playlist_add",
                                        on_click=lambda: add_to_prefix_val()
                                    ).props('flat dense').classes('text-blue-600').tooltip("Append triggers to Style Prompt Prefix")
                                    
                        ui.number(
                            label="Strength",
                            value=current_strength_model,
                            min=0.0, max=2.0, step=0.1,
                            on_change=lambda e, nid=node_id: update_override_state(nid, "strength_model", e.value)
                        ).classes('w-full')

                    elif node_type == "model_loader":
                        param_key = params.get("model_param_key", "ckpt_name")
                        current_model_name = state.style_workflow_overrides.get(node_id, {}).get(param_key, params.get(param_key, ""))
                        
                        comfy_key = f"{node_id}:{param_key}"
                        if comfy_key in comfy_options_cache:
                            # Dropdown only if ComfyUI online and data loaded successfully
                            ui.select(
                                options=comfy_options_cache[comfy_key],
                                value=current_model_name if current_model_name in comfy_options_cache[comfy_key] else None,
                                label=f"Select Model ({param_key})",
                                on_change=lambda e, nid=node_id, pk=param_key: (update_override_state(nid, pk, e.value), render_workflow_overrides_ui.refresh())
                            ).classes('w-full')
                        else:
                            # Self-healing text input fallback if ComfyUI is offline
                            ui.input(
                                label=f"Model Filename ({param_key})",
                                value=current_model_name,
                                on_change=lambda e, nid=node_id, pk=param_key: update_override_state(nid, pk, e.value)
                            ).classes('w-full')

                    elif node_type == "clip_loader":
                        param_key = params.get("clip_param_key", "clip_name")
                        current_clip_name = state.style_workflow_overrides.get(node_id, {}).get(param_key, params.get(param_key, ""))
                        
                        comfy_key = f"{node_id}:{param_key}"
                        if comfy_key in comfy_options_cache:
                            ui.select(
                                options=comfy_options_cache[comfy_key],
                                value=current_clip_name if current_clip_name in comfy_options_cache[comfy_key] else None,
                                label=f"Select CLIP Model ({param_key})",
                                on_change=lambda e, nid=node_id, pk=param_key: (update_override_state(nid, pk, e.value), render_workflow_overrides_ui.refresh())
                            ).classes('w-full')
                        else:
                            ui.input(
                                label=f"CLIP Filename ({param_key})",
                                value=current_clip_name,
                                on_change=lambda e, nid=node_id, pk=param_key: update_override_state(nid, pk, e.value)
                            ).classes('w-full')

                    elif node_type == "vae_loader":
                        current_vae_name = state.style_workflow_overrides.get(node_id, {}).get("vae_name", params.get("vae_name", ""))
                        
                        comfy_key = f"{node_id}:vae_name"
                        if comfy_key in comfy_options_cache:
                            ui.select(
                                options=comfy_options_cache[comfy_key],
                                value=current_vae_name if current_vae_name in comfy_options_cache[comfy_key] else None,
                                label="Select VAE Model",
                                on_change=lambda e, nid=node_id: (update_override_state(nid, "vae_name", e.value), render_workflow_overrides_ui.refresh())
                            ).classes('w-full')
                        else:
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
                    ui.image(img_data).props('fit=contain').classes('w-full h-48 bg-slate-50 rounded-lg border shadow-sm cursor-zoom-in hover:brightness-95 transition-all') \
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
    global contact_sheet_dialog_ref, lora_chooser_dialog_ref
    
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

    # Make sure loras list is hydrated and Comfy choices start loading asynchronously
    load_associated_loras()
    if state.style_selected_workflow:
        asyncio.create_task(async_load_comfy_and_lora_choices())

    with ui.grid(columns='420px 1fr').classes('w-full gap-6 items-start'):
        # LEFT CONFIG PANEL
        with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-4'):
            ui.label('Style Preset & Workflow Config').classes('text-sm font-bold text-slate-800')
            
            # 1. Style Preset Library Loader (The Parent Concept)
            with ui.row().classes('w-full items-center justify-between p-3 bg-slate-50 rounded border border-slate-200 mt-1'):
                with ui.column().classes('gap-0 flex-1'):
                    ui.label('Active Preset').classes('text-[10px] font-bold text-slate-400 uppercase')
                    ui.label().classes('text-xs font-bold text-slate-800').bind_text_from(state, 'style_selected_preset')
                ui.button(
                    'Browse Library', 
                    icon='photo_library', 
                    on_click=lambda: open_style_chooser_modal_globally()
                ).classes('bg-blue-600 text-white text-xs h-9 font-semibold')
            
            # 2. Preset Save Field (Grouped with Preset Management)
            with ui.row().classes('w-full items-end gap-2'):
                custom_style_name = ui.input(placeholder="Preset Name", label="Save Current Preset") \
                    .classes('flex-1') \
                    .bind_value(state, 'style_preset_save_name') \
                    .tooltip("Saves current prompts and overrides as a global, reusable preset in your styles library")
                ui.button(
                    icon="save",
                    on_click=lambda: (
                        save_style_preset_by_name(custom_style_name.value)
                    )
                ).props('outline').classes('h-10 text-blue-600').tooltip("Save or overwrite this preset in the global styles library")

            ui.separator()

            # 3. Workflow Selector (Child Engine of the Selected Style)
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
            
            ui.separator()
            
            ui.textarea(
                label="Style Prompt Prefix"
            ).classes('w-full text-xs').props('outlined autogrow').bind_value(state, 'style_prompt_prefix')

            ui.textarea(
                label="Style Prompt Suffix"
            ).classes('w-full text-xs').props('outlined autogrow').bind_value(state, 'style_prompt_suffix')
            
            ui.textarea(
                label="Style Negative Prompt"
            ).classes('w-full text-xs').props('outlined autogrow').bind_value(state, 'style_negative_prompt')
            
            render_workflow_overrides_ui()
            
            
        # RIGHT: Visual Style Playground Grid
        with ui.column().classes('w-full gap-4'):
            with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-4'):
                with ui.row().classes('w-full items-center justify-between'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('tune', size='sm', color='blue-500')
                        ui.label('Style Visual Playground Settings').classes('text-sm font-bold text-slate-800')
                    
                    # Relocated project settings save button with disk icon and tooltip helper
                    ui.button(
                        icon='save',
                        on_click=lambda: save_project_settings_cb(project.id) if save_project_settings_cb else None
                    ).props('flat round size=md').classes('text-blue-600') \
                     .tooltip("Saves active preset selection and manual seed choices specifically for this project")
                    
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
                with ui.column().classes('w-full bg-blue-50/50 p-3 rounded-lg border border-blue-100/50 mt-1 gap-2.5'):
                    # Row 1: Header Line on its own row
                    with ui.row().classes('items-center gap-1.5 w-full'):
                        ui.icon('smart_toy', color='blue', size='sm')
                        ui.label('AI Toolkit').classes('text-xs font-black text-slate-700 uppercase tracking-wide')
                    
                    # Row 2: Controls grouped and wrapping logically
                    with ui.row().classes('w-full items-center justify-between gap-3 flex-wrap'):
                        # Left grouping: Copy utility buttons
                        with ui.row().classes('items-center gap-2'):
                            ui.button(
                                'Copy Primer',
                                icon='psychology',
                                on_click=lambda: copy_style_primer_to_clipboard(project.name)
                            ).classes('bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-semibold h-9') \
                             .tooltip('Formats diagnostic visual parameters and prompt runs specifically for debugging chats with external LLMs')

                            ui.button(
                                'Copy Prompt Pack',
                                icon='content_copy',
                                on_click=lambda: copy_style_settings_to_clipboard(project.name)
                            ).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs font-semibold h-9')
                        
                        # Right grouping: Contact Sheet config (guaranteed on a single line)
                        with ui.row().classes('items-center gap-2 flex-nowrap'):
                            ui.input(
                                placeholder='Overlay text...'
                            ).classes('w-44 text-xs bg-white').props('outlined dense').bind_value(state, 'style_contact_sheet_overlay') \
                             .tooltip('Text written onto bottom-center of generated contact sheets')

                            ui.button(
                                'Contact Sheet',
                                icon='grid_view',
                                on_click=open_contact_sheet_modal
                            ).classes('bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold h-9 flex-shrink-0')
            
            render_style_playground_cards(project.name)

    # Declare Dialog Overlay inside workspace hierarchy
    with ui.dialog() as contact_sheet_dialog:
        with ui.card().classes('w-full max-w-4xl p-6 rounded-xl gap-4 bg-white'):
            with ui.row().classes('w-full justify-between items-center pb-2 border-b'):
                with ui.column().classes('gap-0'):
                    ui.label('Generated AI Contact Sheet').classes('text-base font-bold text-slate-800')
                    ui.label('Grid coordinate labels (1, 2...) are stamped on the image. Perfect for sending to LLMs.').classes('text-xs text-slate-500')
                ui.button(
                    'Download Contact Sheet',
                    icon='download',
                    on_click=download_contact_sheet
                ).classes('bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold h-9')
            
            render_contact_sheet_preview(contact_sheet_base64)
            
            with ui.row().classes('w-full justify-between items-center pt-2 border-t text-[10px] text-slate-400'):
                ui.label('Tip: Right-click the image to copy it directly, or click Download.')
                ui.button('Close', on_click=contact_sheet_dialog.close).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs')

    # Declare Lora Chooser Dialog Overlay inside workspace hierarchy
    with ui.dialog() as lora_chooser_dialog:
        with ui.card().classes('w-full max-w-4xl p-5 rounded-xl gap-4 bg-white'):
            with ui.row().classes('w-full justify-between items-center pb-1 border-b'):
                with ui.column().classes('gap-0'):
                    ui.label('Visual LoRA Chooser').classes('text-base font-bold text-slate-800')
                    ui.label('Select benchmarked LoRAs to preview styling and automatically configure strength and trigger words.').classes('text-xs text-slate-500')
                ui.button(icon='close', on_click=lora_chooser_dialog.close).props('flat round size=sm').classes('text-slate-400')
                
            render_lora_chooser_content()

    contact_sheet_dialog_ref = contact_sheet_dialog
    lora_chooser_dialog_ref = lora_chooser_dialog