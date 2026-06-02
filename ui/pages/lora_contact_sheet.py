# ui/pages/lora_contact_sheet.py
import os
import csv
import uuid
import time
import shutil
import asyncio
from pathlib import Path
from nicegui import ui
from ui import state
from typing import Callable, List, Dict, Any, Optional
import json

from services.comfy_client import ComfyClient
from database.connection import get_setting

LORA_LIB_DIR = Path("./output/_lora_library")
LORA_CSV_PATH = LORA_LIB_DIR / "loras.csv"
LORA_PROMPTS_PATH = LORA_LIB_DIR / "prompts.txt"

DEFAULT_PROMPTS = [
    "1890s Victorian magazine illustration, detailed pen and ink with soft watercolor wash, in the style of Sidney Paget, fine cross-hatching, antiquarian storybook aesthetic. A young boy with scraped skin in his infantile features, standing near an agitated cat amidst the chaos of a bustling Cairo train station, dusty sunlight filtering through overhead structures.",
    "1870s frontier realism, vintage oil on coarse canvas, in the style of Frederic Remington and Charles M. Russell, rough impasto texture. A lone figure walks down a dusty street under a starlit sky, where every bush and fence post casts a sharp, crisp shadow against the clear night air.",
    "Watercolor wash by Arthur Rackham and Alan Lee, fluid bleeding edges, muted earth tones punctuated by bright accents, historical fantasy, Westeros, Medieval. A massive seven-foot-tall man in battered plate armor and a knight’s helm stands beside a small bald boy, both amid banners and armored horses under bright daylight on a grassy meadow.",
    "Warhammer 40k concept art, Inquisitorial agent, grimdark sci-fi noir, gothic cyberpunk, atmospheric lighting, trench coats, cybernetics, dark urban setting, highly detailed, matte painting, masterpiece, A woman with a mechanical eye and a hand equipped with gleaming surgical tools stands in a dim, industrial morgue, surrounded by rows of lifeless bodies under cold, harsh lighting.",
    "1800s lithograph, vintage hand-colored engraving, Napoleonic Wars era, rough texture. Gritty atmosphere, coarse canvas and wool clothing, heavy line weight, muted palette. A sleek, well-kept naval brig, early 19th century, sailing with crisp rigging and polished wooden decks, bristling with cannons and Royal Navy markings, caught in the golden light of a clear sea horizon, its sails taut as if carved from the wind.",
    "1800s lithograph, vintage hand-colored engraving, Napoleonic Wars era, rough texture. Gritty atmosphere, coarse canvas and wool clothing, heavy line weight, muted palette. A young woman in a midshipman's jacket uniform, early 19th century, standing on the deck of a ship with a sword in hand, slashing downward as she cuts the waist cord of a man’s pants, causing them to fall to his ankles while he drops his own sword and looks up in terror as the blade glints near his neck, surrounded by crew members laughing, with a golden dragon pennant fluttering above them in the wind.",
    "sci-fi, future, Three astronauts in a futuristic spacecraft cockpit, floating in a vast, star-filled void, surrounded by glowing digital interfaces displaying thousands of software files and ancient books, with a holographic interface showing a Minesweeper game and a Sanskrit-to-English dictionary in the background, lighting softly illuminating their faces and the instruments.",
    "A 90s sci-fi movie still, shot on 35mm film, directed by Paul Verhoeven, cinematic lighting, heavy atmosphere, practical effects, detailed textures, realistic skin and metal, A group of diverse individuals sitting around a large, glowing screen in a dimly lit room, each holding a tablet or notebook, their faces illuminated by the flickering data columns showing star names and numerical readings, the scene bathed in soft blue and white light with subtle reflections on the screen.",
    "A historical oil painting from 1812, rendered with soft transitions, muted period-accurate pigments, visible canvas grain, gentle chiaroscuro, in the style of Sir Joshua Reynolds and Sir Thomas Lawrence, classical composition. A naval officer standing in a dimly lit London office, addressing a group of stern-faced officials at a large wooden table with maps and charts scattered across it, the light casting sharp shadows and emphasizing the weight of the conversation.",
    "A historical oil painting from 1812, rendered with soft transitions, muted period-accurate pigments, visible canvas grain, gentle chiaroscuro, in the style of Sir Joshua Reynolds and Sir Thomas Lawrence, classical composition. A woman in a smart, sun-bleached traveling dress, standing by a grand, colonial-era seaside house with ivy-covered walls, holding a large, ornate trunk and gazing out at a turquoise sea with a distant post-chaise in the distance.",
    "Color Charcoal rendering in the style of Gustave Doré with influence from Ashley Wood, crosshatching, stark contrast, A police car, red lights, moving slowly, on a four-lane street at night, in Tbilisi.",
    "Vintage 1960s pulp paperback cover illustration, painted in gouache and acrylic, style of Robert McGinnis, visible brushstrokes, matte texture, dramatic composition, A man in a worn wool coat sits in a grimy armchair, facing a narrow window, eyes fixed on the dim reflected glow of city traffic beneath, shadows stretching across the floor in a dimly lit room.",
    "Vintage 1960s pulp paperback cover illustration, painted in gouache and acrylic, style of Robert McGinnis, visible brushstrokes, matte texture, dramatic composition, A wide shot of London Avenue at night, sodium streetlights casting harsh, yellow glows over rain-slicked pavement, storefronts visible but shadowed above, the buildings' upper floors swallowed in deep shadows, industrial concrete beneath a cold, wind-blown sky.",
    "Color Charcoal rendering in the style of Gustave Doré with influence from Ashley Wood, crosshatching, stark contrast, A Roman man gazing into a dark river at twilight, standing on a stone pier along a narrow alleyway in ancient Rome.",
    "Digital Impasto in the style of Stanley Artgerm Lau, high-contrast, A slender, leggy woman with delicate features and long limbs standing in a moonlit field, her silhouette framed by tall grass swaying in the breeze, glowing faintly in the background.",
    "Illustration in the style of a modern graphic novel, Sean Gordon Murphy art style, A lone rider on a sleek, underpowered scooter, cutting diagonally across a busy urban street, approaching a stationary sedan head-on, with the scooter's engine whining and the sedan's driver window reflecting the glare of city lights.",
    "Illustration in the style of a modern graphic novel, Sean Gordon Murphy art style, A man in a dark suit, standing in a dimly lit, cluttered office with a vintage telephone and a framed photograph on the wall, looks directly at the camera with a quiet, sincere expression, his eyes glistening with emotion.",
    "1940s, WWII, World War 2 era style, A warm, sun-drenched 1940s French countryside garden during World War II, with families gathered at a rustic table under a canopy of trees, children in auburn hair and honey-colored eyes playfully chatting, a map of the surrounding area drawn on a notebook lies open, and a man in a wool coat quietly observes the scene with a thoughtful expression, the air filled with the quiet tension of wartime resilience and familial warmth.",
    "A simple glass of cold milk sitting on a rustic wooden kitchen table, natural morning light filtering through a window.",
    "A large diverse crowd of people cheering in the bleachers during a sunny daytime baseball game, wide angle shot."
]

# Persistent Global Dialog Element References
image_preview_dialog = None
preview_img = None
preview_caption = None
prev_btn = None
next_btn = None

preview_images_list = []
preview_current_idx = -1

edit_dialog = None
edit_triggers = None
edit_strength = None
active_edit_lora = None

delete_confirm_dialog = None
active_delete_lora = None

lora_contact_sheet_dialog = None
lora_contact_sheet_base64 = ""

prompts_editor_dialog = None
prompts_textarea_ref = None

def get_prompts_count() -> int:
    """Helper to retrieve the number of active lines inside prompts.txt dynamically."""
    if LORA_PROMPTS_PATH.exists():
        try:
            with open(LORA_PROMPTS_PATH, "r", encoding="utf-8") as f:
                return len([line.strip() for line in f if line.strip()])
        except Exception:
            pass
    return len(DEFAULT_PROMPTS)

def init_lora_library():
    LORA_LIB_DIR.mkdir(parents=True, exist_ok=True)
    if not LORA_PROMPTS_PATH.exists():
        with open(LORA_PROMPTS_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(DEFAULT_PROMPTS))
            
    if not LORA_CSV_PATH.exists():
        with open(LORA_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(["id", "workflow", "lora_path", "triggers", "strength", "avg_render_time", "status", "favorite"])
        state.lora_library = []
        return

    rows = []
    has_stuck_generations = False
    needs_migration = False
    with open(LORA_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            # Self-healing: Automatically recover stuck runs back to Pending
            if row.get("status") == "Generating":
                row["status"] = "Pending"
                has_stuck_generations = True
            
            # Migration: Ensure the favorite field exists on every old row
            if "favorite" not in row:
                row["favorite"] = "False"
                needs_migration = True
                
            rows.append(row)
            
    state.lora_library = rows
    
    # If we recovered stuck LoRAs or migrated schema columns, rewrite the CSV now
    if has_stuck_generations or needs_migration:
        save_lora_library_full()


def save_lora_library_full():
    with open(LORA_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "workflow", "lora_path", "triggers", "strength", "avg_render_time", "status", "favorite"], delimiter='|')
        writer.writeheader()
        for l in state.lora_library:
            writer.writerow(l)


def save_lora_to_library(lora_data: List[Dict[str, Any]]):
    for l in lora_data:
        state.lora_library.append(l)
    save_lora_library_full()
    render_lora_sidebar.refresh()


def parse_and_add_loras(raw_text: str, default_strength: float, dialog: Any):
    if not raw_text.strip():
        ui.notify("No LoRA data provided.", type="warning")
        return
    workflow = state.lora_tool_selected_workflow
    if workflow == "None":
        ui.notify("Error: No base workflow selected.", type="negative")
        return

    new_entries = []
    lines = raw_text.strip().split('\n')
    
    for line in lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split('|')]
        path = parts[0]
        triggers = parts[1] if len(parts) > 1 else ""
        try:
            strength = float(parts[2]) if len(parts) > 2 else default_strength
        except ValueError:
            strength = default_strength

        duplicate = any(ex["workflow"] == workflow and ex["lora_path"] == path and float(ex["strength"]) == strength for ex in state.lora_library)
        
        if not duplicate:
            new_entries.append({
                "id": str(uuid.uuid4())[:8],
                "workflow": workflow,
                "lora_path": path,
                "triggers": triggers,
                "strength": strength,
                "avg_render_time": "0.0",
                "status": "Pending",
                "favorite": "False"
            })

    if new_entries:
        save_lora_to_library(new_entries)
        ui.notify(f"Added {len(new_entries)} new LoRA configurations!", type="positive")
        dialog.close()
    else:
        ui.notify("No new configurations added. (Duplicates ignored)", type="info")


def get_lora_workflows() -> list[str]:
    client = ComfyClient("")
    valid_workflows = []
    for folder in ["./workflows", "./Comfy_Workflows"]:
        path = Path(folder)
        if path.exists():
            for file_path in path.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        wf_json = json.load(f)
                    discovered = client.analyze_workflow(wf_json)
                    if any(node.get("type") == "lora_loader" for node in discovered.values()):
                        valid_workflows.append(file_path.name)
                except Exception:
                    pass
    return sorted(list(set(valid_workflows)))


# --- ASYNC EXECUTION ENGINE ---
async def run_benchmark_task(lora_ids: List[str]):
    state.lora_tool_generating = True
    state.lora_tool_cancel_flag = False
    
    try:
        comfy_url = get_setting("comfy_url", "127.0.0.1:8188")
        if "http" in comfy_url:
            comfy_url = comfy_url.replace("http://", "").replace("https://", "").strip("/")
        client = ComfyClient(comfy_url)
        
        with open(LORA_PROMPTS_PATH, "r", encoding="utf-8") as f:
            prompts = [p.strip() for p in f.readlines() if p.strip()]
            
        for lora_id in lora_ids:
            if state.lora_tool_cancel_flag:
                break
                
            lora = next((l for l in state.lora_library if l["id"] == lora_id), None)
            if not lora:
                continue
                
            # Only set focus to the generating LoRA if the user hasn't selected anything else
            if state.lora_tool_active_lora_id is None:
                state.lora_tool_active_lora_id = lora_id
            
            wf_path = Path("./workflows") / lora["workflow"]
            if not wf_path.exists():
                wf_path = Path("./Comfy_Workflows") / lora["workflow"]
                
            try:
                with open(wf_path, "r", encoding="utf-8") as f:
                    wf_json = json.load(f)
            except Exception as e:
                ui.notify(f"Could not load workflow: {str(e)}", type="negative")
                lora["status"] = "Pending"
                continue
                
            discovered = client.analyze_workflow(wf_json)
            lora_node_id = next((n_id for n_id, data in discovered.items() if data["type"] == "lora_loader"), None)
            
            if not lora_node_id:
                ui.notify(f"No LoRA loader found in {lora['workflow']}!", type="negative")
                lora["status"] = "Pending"
                continue
                
            out_dir = LORA_LIB_DIR / lora_id
            out_dir.mkdir(parents=True, exist_ok=True)
            
            lora["status"] = "Generating"
            state.lora_tool_progress = {"lora_id": lora_id, "current": 0, "total": len(prompts)}
            save_lora_library_full()
            render_lora_sidebar.refresh()
            render_lora_workspace.refresh()
            
            start_time = time.time()
            cancelled_this_lora = False
            
            for idx, prompt_text in enumerate(prompts):
                if state.lora_tool_cancel_flag:
                    cancelled_this_lora = True
                    break
                    
                state.lora_tool_progress = {
                    "lora_id": lora_id,
                    "current": idx + 1,
                    "total": len(prompts)
                }
                
                overrides = {
                    lora_node_id: {
                        "lora_name": lora["lora_path"],
                        "strength_model": float(lora["strength"]),
                        "strength_clip": float(lora["strength"])
                    }
                }
                
                triggers = lora["triggers"].strip()
                if triggers == ".":
                    triggers = ""
                
                seed = 1000 + idx
                prefix_val = f"{triggers}, " if triggers else ""
                
                def render_block():
                    return client.generate_image_sync(
                        workflow_json=wf_json,
                        prompt_text=prompt_text,
                        neg_prompt_text="blurry, bad quality, watermark, text",
                        seed=seed,
                        overrides=overrides,
                        prefix=prefix_val
                    )
                    
                img_bytes, logs = await asyncio.to_thread(render_block)
                
                if img_bytes:
                    img_path = out_dir / f"{idx+1:02d}.png"
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                        
            if cancelled_this_lora:
                lora["status"] = "Pending"
                # Wipe the folder cleanly so we don't end up with fragmented contact sheets
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                save_lora_library_full()
            elif not state.lora_tool_cancel_flag:
                end_time = time.time()
                avg_time = (end_time - start_time) / len(prompts)
                lora["avg_render_time"] = f"{avg_time:.2f}"
                lora["status"] = "Completed"
                save_lora_library_full()

    finally:
        # Fallback Cleanup: Revert any lingering "Generating" state anywhere in the library to Pending
        for l in state.lora_library:
            if l["status"] == "Generating":
                l["status"] = "Pending"
                p_dir = LORA_LIB_DIR / l["id"]
                if p_dir.exists():
                    shutil.rmtree(p_dir)
                    
        save_lora_library_full()
        state.lora_tool_generating = False
        state.lora_tool_progress = {}
        render_lora_sidebar.refresh()
        render_lora_workspace.refresh()


def cancel_generation():
    state.lora_tool_cancel_flag = True
    ui.notify("Stopping after current image finishes...", type="warning")


def update_preview_content():
    global preview_images_list, preview_current_idx, preview_img, preview_caption
    global prev_btn, next_btn
    
    if not preview_images_list or preview_current_idx < 0 or preview_current_idx >= len(preview_images_list):
        return

    img_path = str(preview_images_list[preview_current_idx])
    preview_img.set_source(img_path)

    current_prompts = DEFAULT_PROMPTS
    if LORA_PROMPTS_PATH.exists():
        with open(LORA_PROMPTS_PATH, "r", encoding="utf-8") as f:
            current_prompts = [p.strip() for p in f.readlines() if p.strip()]

    caption_text = f"[{preview_current_idx + 1}/{len(preview_images_list)}] " + (
        current_prompts[preview_current_idx] if preview_current_idx < len(current_prompts) else "Prompt unknown"
    )
    preview_caption.set_text(caption_text)

    # Disable navigation buttons at boundaries to keep experience cohesive
    if prev_btn:
        if preview_current_idx > 0:
            prev_btn.enable()
            prev_btn.classes('opacity-100', remove='opacity-20')
        else:
            prev_btn.disable()
            prev_btn.classes('opacity-20', remove='opacity-100')

    if next_btn:
        if preview_current_idx < len(preview_images_list) - 1:
            next_btn.enable()
            next_btn.classes('opacity-100', remove='opacity-20')
        else:
            next_btn.disable()
            next_btn.classes('opacity-20', remove='opacity-100')


def navigate_preview(direction: int):
    global preview_images_list, preview_current_idx
    if not preview_images_list:
        return
    new_idx = preview_current_idx + direction
    if 0 <= new_idx < len(preview_images_list):
        preview_current_idx = new_idx
        update_preview_content()


def handle_keyboard(e):
    # Only intercept if the image preview modal is active
    if not image_preview_dialog or not image_preview_dialog.value:
        return

    key = e.key.name.lower() if e.key and e.key.name else ""
    if key in ["arrowleft", "s"]:
        navigate_preview(-1)
    elif key in ["arrowright", "f"]:
        navigate_preview(1)
    elif key == "escape":
        image_preview_dialog.close()


def open_preview(img_path: str, idx: int, images_list: list):
    global preview_images_list, preview_current_idx
    preview_images_list = images_list
    preview_current_idx = idx
    update_preview_content()
    image_preview_dialog.open()


def open_edit(lora: Dict[str, Any]):
    global active_edit_lora
    active_edit_lora = lora
    edit_triggers.set_value(lora["triggers"])
    edit_strength.set_value(float(lora["strength"]))
    edit_dialog.open()


def open_delete(lora: Dict[str, Any]):
    global active_delete_lora
    active_delete_lora = lora
    delete_confirm_dialog.open()


def confirm_delete_lora():
    global active_delete_lora
    if not active_delete_lora:
        return

    # Delete folder hierarchy containing generated png outputs
    out_dir = LORA_LIB_DIR / active_delete_lora["id"]
    if out_dir.exists():
        try:
            shutil.rmtree(out_dir)
        except Exception as e:
            state.add_console_log(f"[LoRA-Library] Error cleaning directory: {str(e)}")

    # Remove active selection element from registry and flush
    state.lora_library = [l for l in state.lora_library if l["id"] != active_delete_lora["id"]]
    save_lora_library_full()

    if state.lora_tool_active_lora_id == active_delete_lora["id"]:
        state.lora_tool_active_lora_id = None

    active_delete_lora = None
    delete_confirm_dialog.close()
    
    ui.notify("LoRA and associated benchmark files successfully deleted.", type="positive")
    render_lora_sidebar.refresh()
    render_lora_workspace.refresh()

def create_lora_contact_sheet(lora_id: str) -> Optional[bytes]:
    """Stitches completed disk benchmark images into a numbered, header-labeled PNG grid."""
    from PIL import Image, ImageDraw, ImageFont
    import io

    lora = next((l for l in state.lora_library if l["id"] == lora_id), None)
    if not lora:
        return None
        
    out_dir = LORA_LIB_DIR / lora_id
    if not out_dir.exists():
        return None
        
    images = sorted([f for f in out_dir.glob("*.png")])
    if not images:
        return None
        
    opened_images = []
    for img_path in images:
        try:
            opened_images.append(Image.open(img_path))
        except Exception:
            pass
            
    if not opened_images:
        return None
        
    num_imgs = len(opened_images)
    cols = 5  # Standard grid columns for benchmarking
    rows = (num_imgs + cols - 1) // cols
    
    tile_w, tile_h = opened_images[0].size
    
    # Calculate responsive header height
    header_h = int(tile_h * 0.15) if tile_h > 400 else 60
    grid_w = cols * tile_w
    grid_h = rows * tile_h + header_h
    
    # Dark gray theme background canvas
    grid_img = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = ImageDraw.Draw(grid_img)
    
    # Set text fonts
    font_title = None
    font_label = None
    try:
        font_title = ImageFont.truetype("arial.ttf", size=int(header_h * 0.4))
        font_label = ImageFont.truetype("arial.ttf", size=int(tile_h * 0.08))
    except Exception:
        try:
            font_title = ImageFont.truetype("DejaVuSans.ttf", size=int(header_h * 0.4))
            font_label = ImageFont.truetype("DejaVuSans.ttf", size=int(tile_h * 0.08))
        except Exception:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
            
    short_name = Path(lora["lora_path"]).name
    header_text = f"LoRA: {short_name}  |  Strength: {float(lora['strength']):.2f}  |  Workflow: {lora['workflow']}"
    
    # Center header text on canvas
    try:
        text_w = draw.textlength(header_text, font=font_title)
    except AttributeError:
        text_w = font_title.getsize(header_text)[0] if hasattr(font_title, 'getsize') else len(header_text) * 8
        
    x_pos = max(20, (grid_w - text_w) // 2)
    y_pos = int((header_h - (header_h * 0.4)) // 2)
    draw.text((x_pos, y_pos), header_text, fill=(255, 255, 255), font=font_title)
    
    # Paste and stamp sequential numbers onto individual tiles
    for idx, img in enumerate(opened_images):
        r = idx // cols
        c = idx % cols
        
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            
        x_offset = c * tile_w
        y_offset = r * tile_h + header_h
        grid_img.paste(img, (x_offset, y_offset))
        
        label_text = str(idx + 1)
        box_w = int(tile_w * 0.12) if tile_w > 400 else 40
        box_h = int(tile_h * 0.10) if tile_h > 400 else 32
        
        bx0 = x_offset
        by0 = y_offset
        bx1 = bx0 + box_w
        by1 = by0 + box_h
        
        # Stamp background rectangle
        draw.rectangle([bx0, by0, bx1, by1], fill=(15, 15, 15))
        
        # Center coordinates
        try:
            lbl_w = draw.textlength(label_text, font=font_label)
        except AttributeError:
            lbl_w = font_label.getsize(label_text)[0] if hasattr(font_label, 'getsize') else len(label_text) * 8
            
        lx = bx0 + (box_w - lbl_w) // 2
        ly = by0 + (box_h - int(tile_h * 0.08)) // 2 if tile_h > 400 else by0 + 2
        draw.text((lx, ly), label_text, fill=(255, 255, 255), font=font_label)
        
    out_buf = io.BytesIO()
    grid_img.save(out_buf, format="PNG")
    return out_buf.getvalue()


def download_lora_contact_sheet(lora_id: str):
    """Triggers direct browser download of the compiled benchmark image grid."""
    lora = next((l for l in state.lora_library if l["id"] == lora_id), None)
    if not lora:
        return
    sheet_bytes = create_lora_contact_sheet(lora_id)
    if sheet_bytes:
        short_name = Path(lora["lora_path"]).stem
        ui.download(sheet_bytes, filename=f"lora_grid_{short_name}_{lora['strength']}.png")
    else:
        ui.notify("Failed to assemble download file.", type="negative")

def open_prompts_editor_modal():
    """Reads the current prompts file from disk, populates the textarea, and opens the editor modal."""
    global prompts_editor_dialog, prompts_textarea_ref
    if not prompts_editor_dialog or not prompts_textarea_ref:
        return
    
    initial_prompts_txt = ""
    if LORA_PROMPTS_PATH.exists():
        try:
            initial_prompts_txt = LORA_PROMPTS_PATH.read_text(encoding="utf-8")
        except Exception:
            initial_prompts_txt = "\n".join(DEFAULT_PROMPTS)
    else:
        initial_prompts_txt = "\n".join(DEFAULT_PROMPTS)
    
    prompts_textarea_ref.set_value(initial_prompts_txt)
    prompts_editor_dialog.open()


def open_lora_contact_sheet_modal(lora_id: str):
    """Encodes stitched image, loads dialog base, and opens the preview overlay."""
    global lora_contact_sheet_base64, lora_contact_sheet_dialog
    sheet_bytes = create_lora_contact_sheet(lora_id)
    if not sheet_bytes:
        ui.notify("Failed to generate contact sheet preview.", type="negative")
        return
        
    import base64
    encoded = base64.b64encode(sheet_bytes).decode("utf-8")
    lora_contact_sheet_base64 = f"data:image/png;base64,{encoded}"
    
    if lora_contact_sheet_dialog:
        lora_contact_sheet_dialog.open()
        render_lora_contact_sheet_preview.refresh(lora_contact_sheet_base64)


def copy_numbered_prompts_to_clipboard():
    """Formats only active evaluation prompts as sequential lines and saves to keyboard cache."""
    current_prompts = DEFAULT_PROMPTS
    if LORA_PROMPTS_PATH.exists():
        with open(LORA_PROMPTS_PATH, "r", encoding="utf-8") as f:
            current_prompts = [p.strip() for p in f.readlines() if p.strip()]
            
    lines = []
    for idx, prompt_text in enumerate(current_prompts):
        lines.append(f"{idx + 1}. {prompt_text}")
        
    full_text = "\n".join(lines)
    ui.clipboard.write(full_text)
    ui.notify("Numbered evaluation prompts copied to clipboard!", type="positive")


@ui.refreshable
def render_lora_contact_sheet_preview(base64_data: str):
    """Renders the stitched high-res contact sheet within the modal dialog viewport."""
    if not base64_data:
        ui.label("Stitching preview, please wait...").classes('text-xs text-slate-400')
        return
    ui.image(base64_data).props('fit=contain').classes('w-full max-h-[70vh] rounded-lg border shadow-sm bg-slate-50/20')

@ui.refreshable
def render_lora_sidebar():
    workflows = ["None"] + get_lora_workflows()
    if state.lora_tool_selected_workflow not in workflows:
        state.lora_tool_selected_workflow = "None"
        state.lora_tool_active_lora_id = None

    ui.label('Base Workflow').classes('text-xs font-bold text-slate-400 uppercase tracking-wider')
    
    def on_workflow_change(e):
        if state.lora_tool_generating:
            ui.notify("Cannot change workflow while generating.", type="warning")
            e.sender.value = state.lora_tool_selected_workflow
            return
        state.lora_tool_selected_workflow = e.value
        state.lora_tool_active_lora_id = None
        render_lora_sidebar.refresh()
        render_lora_workspace.refresh()

    ui.select(options=workflows, value=state.lora_tool_selected_workflow, on_change=on_workflow_change).classes('w-full mb-2')
    
    with ui.dialog() as add_lora_dialog, ui.card().classes('w-full max-w-2xl p-6 rounded-xl'):
        ui.label('Add LoRAs to Benchmark').classes('text-xl font-bold text-slate-800 mb-2')
        ui.label('Paste your LoRA paths and triggers. Format: path | trigger_words | [optional_strength]').classes('text-sm text-slate-500 mb-4')
        raw_text = ui.textarea(placeholder='SDXL\\zyd232.safetensors | zydink, ink sketch\nSDXL\\50sNoir.safetensors | 50s Noir, detective | 0.8').classes('w-full h-48')
        with ui.row().classes('w-full items-center gap-4 mt-4 bg-slate-50 p-4 rounded-lg border'):
            ui.icon('tune', size='sm').classes('text-slate-400')
            with ui.column().classes('gap-0 flex-1'):
                ui.label('Default Strength').classes('text-sm font-bold text-slate-700')
                ui.label('Applied if strength is omitted from the piped text.').classes('text-xs text-slate-500')
            default_strength = ui.slider(min=0.1, max=2.0, step=0.05, value=1.0).classes('w-48')
            ui.label().bind_text_from(default_strength, 'value', backward=lambda v: f'{v:.2f}').classes('font-mono font-bold text-blue-600 w-12 text-right')
            
        with ui.row().classes('w-full justify-end gap-3 mt-4'):
            ui.button('Cancel', on_click=add_lora_dialog.close).props('flat color=slate')
            ui.button('Parse & Add', on_click=lambda: parse_and_add_loras(raw_text.value, default_strength.value, add_lora_dialog)).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold px-6')

    if state.lora_tool_selected_workflow != "None" and not state.lora_tool_generating:
        ui.button('Add LoRAs', icon='add', on_click=add_lora_dialog.open).classes('w-full bg-blue-600 hover:bg-blue-700 text-white font-bold mb-2')
        
        # Collapsible prompts flat-file configuration drawer
        ui.button(
            'Edit Prompts', 
            icon='edit_note', 
            on_click=open_prompts_editor_modal
        ).classes('w-full bg-slate-700 hover:bg-slate-800 text-white font-bold mb-2') \
         .tooltip('Open full-size benchmark prompts editor modal.')
        
    ui.separator().classes('mb-4 mt-2')
    
    active_loras = [l for l in state.lora_library if l.get("workflow") == state.lora_tool_selected_workflow]

    # Bubble favorites up to the top, and sort alphabetically underneath
    active_loras = sorted(active_loras, key=lambda l: (l.get("favorite") != "True", Path(l["lora_path"]).name.lower()))

    with ui.column().classes('w-full gap-2'):
        if state.lora_tool_selected_workflow == "None":
            ui.label('Select a workflow above to see its LoRAs.').classes('text-sm text-slate-400 text-center w-full mt-2')
        elif not active_loras:
            ui.label(f'No LoRAs added for {state.lora_tool_selected_workflow} yet.').classes('text-sm text-slate-400 text-center w-full mt-2')
        else:
            pending_loras = [l["id"] for l in active_loras if l["status"] == "Pending"]
            
            with ui.row().classes('w-full justify-between items-center mb-1'):
                ui.label(f'Library ({len(active_loras)})').classes('text-xs font-bold text-slate-400 uppercase tracking-wider')
                
                if state.lora_tool_generating:
                    ui.button('Stop Generator', icon='stop', on_click=cancel_generation).props('flat dense').classes('text-xs text-rose-500 font-bold bg-rose-50 px-2 rounded')
                elif pending_loras:
                    ui.button(f'Run {len(pending_loras)} Pending', icon='play_arrow', on_click=lambda: asyncio.create_task(run_benchmark_task(pending_loras))).props('flat dense').classes('text-xs text-blue-600 font-bold bg-blue-50 px-2 rounded')

            for lora in active_loras:
                short_path = Path(lora["lora_path"]).name
                is_active = state.lora_tool_active_lora_id == lora["id"]
                
                card_bg = 'bg-blue-50 border-blue-300 shadow-sm' if is_active else 'border-slate-200 hover:bg-slate-50'
                if lora["status"] == "Generating":
                    status_color, status_text = 'blue-200', 'blue-900'
                else:
                    status_color = 'emerald-100' if lora["status"] == "Completed" else 'slate-100'
                    status_text = 'emerald-800' if lora["status"] == "Completed" else 'slate-600'
                
                def select_lora(lora_id=lora["id"]):
                    state.lora_tool_active_lora_id = lora_id
                    render_lora_sidebar.refresh()
                    render_lora_workspace.refresh()

                with ui.card().classes(f'w-full p-3 border cursor-pointer transition-all gap-1 {card_bg}').on('click', select_lora):
                    with ui.row().classes('w-full items-center justify-between gap-1'):
                        ui.label(short_path).classes('text-xs font-bold text-slate-800 truncate flex-1')
                        if lora.get("favorite") == "True":
                            ui.icon('star', color='amber-500', size='14px')
                            
                    with ui.row().classes('w-full justify-between items-center mt-1'):
                        ui.badge(f'str: {float(lora["strength"]):.2f}', color='blue-100').classes('text-blue-800 text-[10px] px-1 py-0 rounded')
                        ui.badge(lora["status"], color=status_color).classes(f'text-{status_text} text-[10px] px-1 py-0 rounded font-bold')


@ui.refreshable
def render_lora_workspace():
    if not state.lora_tool_active_lora_id:
        with ui.column().classes('w-full h-full bg-slate-50 border border-dashed rounded-xl p-8 items-center justify-center'):
            ui.icon('grid_on', size='64px').classes('text-slate-300 mb-4')
            ui.label('Select a LoRA from the sidebar to view its contact sheet.').classes('text-lg font-medium text-slate-500')
            ui.label('Or select a workflow and click "Add LoRAs" to start benchmarking a new batch.').classes('text-sm text-slate-400')
        return

    active_lora = next((l for l in state.lora_library if l["id"] == state.lora_tool_active_lora_id), None)
    if not active_lora:
        return

    short_path = Path(active_lora["lora_path"]).name
    clean_triggers = active_lora["triggers"].strip()
    if clean_triggers == "." or not clean_triggers:
        clean_triggers = "(No trigger words)"

    with ui.column().classes('w-full gap-4'):
        with ui.card().classes('w-full bg-white border rounded-xl p-4 shadow-sm gap-2'):
            # Beautiful flex layout utilizing flex-nowrap to prevent the right action cluster wrapping below
            with ui.row().classes('w-full justify-between items-start gap-4 flex-nowrap'):
                with ui.column().classes('gap-0 flex-1 min-w-0'):
                    ui.label(short_path).classes('text-xl font-bold text-slate-800 truncate w-full')
                    with ui.row().classes('items-center gap-2 text-sm text-slate-500 flex-wrap'):
                        ui.icon('bolt', size='sm').classes('flex-shrink-0')
                        ui.label(f'Triggers: {clean_triggers}').classes('truncate max-w-md')
                        ui.label('•')
                        ui.label(f'Strength: {float(active_lora["strength"]):.2f}').classes('flex-shrink-0')
                
                with ui.row().classes('items-center gap-2 flex-shrink-0'):
                    if not state.lora_tool_generating:
                        # Star toggle button
                        is_fav = active_lora.get("favorite") == "True"
                        star_icon = "star" if is_fav else "star_border"
                        star_color = "text-amber-500" if is_fav else "text-slate-400"
                        
                        def toggle_favorite():
                            active_lora["favorite"] = "False" if active_lora.get("favorite") == "True" else "True"
                            save_lora_library_full()
                            render_lora_sidebar.refresh()
                            render_lora_workspace.refresh()
                            
                        ui.button(
                            icon=star_icon,
                            on_click=toggle_favorite
                        ).props('flat dense').classes(f'{star_color} hover:text-amber-500 text-xs font-bold').tooltip('Toggle Favorite')

                        # Edit config settings button
                        ui.button('Edit Settings', icon='edit', on_click=lambda: open_edit(active_lora)).props('flat dense').classes('text-slate-400 hover:text-blue-500 text-xs font-bold')
                        
                        # Permanent deletion trigger
                        ui.button('Delete', icon='delete', on_click=lambda: open_delete(active_lora)).props('flat dense').classes('text-slate-400 hover:text-rose-500 text-xs font-bold')
                        
                    if active_lora["status"] == "Completed":
                        with ui.column().classes('items-end gap-0 bg-slate-50 p-2 rounded border'):
                            ui.label('Avg Render Time').classes('text-[10px] font-bold text-slate-400 uppercase tracking-wider')
                            ui.label(f'{float(active_lora["avg_render_time"]):.1f}s / image').classes('text-lg font-black text-blue-600')

        if active_lora["status"] == "Pending":
            p_count = get_prompts_count()
            with ui.column().classes('w-full h-64 bg-slate-50 border border-dashed rounded-xl p-8 items-center justify-center gap-4'):
                ui.icon('speed', size='64px').classes('text-blue-200')
                ui.label('Ready to Benchmark').classes('text-lg font-bold text-slate-700')
                ui.label(f'This will generate {p_count} images using your default prompts to evaluate the style and render speed.').classes('text-sm text-slate-500 text-center max-w-md')
                if not state.lora_tool_generating:
                    ui.button(f'Start {p_count}-Image Benchmark', icon='play_arrow', on_click=lambda: asyncio.create_task(run_benchmark_task([active_lora["id"]]))).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold px-6 py-2 mt-2 shadow-sm')

        elif active_lora["status"] == "Generating":
            p_count = get_prompts_count()
            with ui.column().classes('w-full h-64 bg-blue-50 border border-blue-200 border-dashed rounded-xl p-8 items-center justify-center gap-4'):
                ui.spinner(size='xl', color='blue-500')
                
                progress_label = ui.label().classes('text-lg font-bold text-blue-800')
                progress_label.bind_text_from(
                    state, 
                    'lora_tool_progress', 
                    backward=lambda p, pc=p_count: f"Rendering Image {p.get('current', 0)} of {p.get('total', pc)}"
                )
                
                progress_bar = ui.linear_progress(show_value=False).classes('w-64 h-2 rounded-full')
                progress_bar.bind_value_from(
                    state, 
                    'lora_tool_progress', 
                    backward=lambda p, pc=p_count: p.get("current", 0) / max(1, p.get("total", pc))
                )
                
                ui.button('Cancel Generation', icon='stop', on_click=cancel_generation).props('flat').classes('text-rose-500 mt-2')

        elif active_lora["status"] == "Completed":
            out_dir = LORA_LIB_DIR / active_lora["id"]
            
            if not state.lora_tool_generating:
                # Premium Toolkit row housing evaluations, grids, and restarts
                with ui.row().classes('w-full justify-between items-center bg-slate-50 p-3 rounded-lg border mb-2'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('assistant', color='blue', size='sm')
                        ui.label('AI Benchmark Evaluation').classes('text-xs font-bold text-slate-700 uppercase tracking-wide')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.button(
                            'Copy Prompts',
                            icon='content_copy',
                            on_click=copy_numbered_prompts_to_clipboard
                        ).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs font-semibold h-9') \
                         .tooltip('Copy only numbered prompts list to send to LLM context.')
                        
                        ui.button(
                            'Generate Contact Sheet',
                            icon='grid_view',
                            on_click=lambda lid=active_lora["id"]: open_lora_contact_sheet_modal(lid)
                        ).classes('bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold h-9') \
                         .tooltip('Generates a numbered grid image of all 20 frames.')
                        
                        ui.button(
                            'Regenerate All',
                            icon='refresh',
                            on_click=lambda lid=active_lora["id"]: asyncio.create_task(run_benchmark_task([lid]))
                        ).props('flat').classes('text-slate-500 hover:text-blue-600 text-xs font-bold h-9')
            
            with ui.grid(columns=5).classes('w-full gap-4'):
                if out_dir.exists():
                    images = sorted([f for f in out_dir.glob("*.png")])
                    for idx, img in enumerate(images):
                        with ui.card().classes('p-0 overflow-hidden border shadow-sm hover:shadow-md transition-shadow'):
                            # Capture and pass the current array sequence to the navigation cycle in open_preview
                            ui.image(str(img)).classes('w-full h-48 object-cover cursor-pointer').on('click', lambda p=img, i=idx, img_list=images: open_preview(str(p), i, img_list))


def render_lora_contact_sheet(exit_tool_cb: Callable):
    init_lora_library()

    global image_preview_dialog, preview_img, preview_caption
    global prev_btn, next_btn
    global edit_dialog, edit_triggers, edit_strength, active_edit_lora
    global delete_confirm_dialog, active_delete_lora
    global lora_contact_sheet_dialog, lora_contact_sheet_base64
    global prompts_editor_dialog, prompts_textarea_ref

    # Global keyboard event handlers registered cleanly onto this window view scope
    ui.keyboard(on_key=handle_keyboard)

    # Declared outside of refreshable containers so they never flash or close on refresh
    with ui.dialog() as image_preview_dialog:
        # Side-by-side row container allowing large floating arrows next to the media card
        with ui.row().classes('items-center justify-center w-full max-w-6xl bg-transparent gap-4 no-wrap'):
            prev_btn = ui.button(icon='chevron_left', on_click=lambda: navigate_preview(-1)) \
                .props('round flat size=xl color=white') \
                .classes('bg-black/40 hover:bg-black/60 shadow flex-shrink-0 opacity-100 transition-all')
            
            with ui.column().classes('p-0 bg-transparent shadow-none items-center justify-center flex-1 max-w-4xl'):
                # Scale-down maintains original aspect ratio cleanly
                preview_img = ui.image().props('no-spinner fit=scale-down').classes('w-full max-h-[75vh] rounded-lg shadow-lg bg-black/50')
                preview_caption = ui.label().classes('w-full text-center text-white bg-slate-900/80 p-3 rounded-b-lg text-sm mt-[-4px] shadow-lg')
                
            next_btn = ui.button(icon='chevron_right', on_click=lambda: navigate_preview(1)) \
                .props('round flat size=xl color=white') \
                .classes('bg-black/40 hover:bg-black/60 shadow flex-shrink-0 opacity-100 transition-all')

    with ui.dialog() as edit_dialog, ui.card().classes('p-6 rounded-xl w-96'):
        ui.label('Edit LoRA Settings').classes('text-lg font-bold text-slate-800')
        ui.label('Changing these settings will reset the LoRA and delete the current contact sheet.').classes('text-xs text-slate-500 mb-4')
        edit_triggers = ui.input('Trigger Words')
        edit_strength = ui.number('Strength', format='%.2f', step=0.05)
        
        def save_edits():
            if active_edit_lora:
                active_edit_lora["triggers"] = edit_triggers.value
                active_edit_lora["strength"] = edit_strength.value
                active_edit_lora["status"] = "Pending"
                active_edit_lora["avg_render_time"] = "0.0"
                
                out_dir = LORA_LIB_DIR / active_edit_lora["id"]
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                    
                save_lora_library_full()
                edit_dialog.close()
                render_lora_sidebar.refresh()
                render_lora_workspace.refresh()
                
        with ui.row().classes('w-full justify-end gap-3 mt-6'):
            ui.button('Cancel', on_click=edit_dialog.close).props('flat color=slate')
            ui.button('Save & Reset', on_click=save_edits).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold')

    with ui.dialog() as delete_confirm_dialog, ui.card().classes('p-6 rounded-xl w-96'):
        ui.label('Delete LoRA?').classes('text-lg font-bold text-slate-800')
        ui.label('This will permanently delete this LoRA configuration and all generated contact sheet benchmark images. This action cannot be undone.').classes('text-xs text-slate-500 mb-6')
        with ui.row().classes('w-full justify-end gap-3'):
            ui.button('Cancel', on_click=delete_confirm_dialog.close).props('flat color=slate')
            ui.button('Delete Permanently', on_click=confirm_delete_lora).classes('bg-rose-600 hover:bg-rose-700 text-white font-bold')

    # Header-focused Prompts Editor Modal with optimized vertical heights
    with ui.dialog() as prompts_editor_dialog:
        with ui.card().classes('w-full max-w-5xl p-6 rounded-xl gap-4 bg-white max-h-[85vh] flex flex-col'):
            # Combined Header Row: Title & Counter on Left, Save/Cancel Controls on Right
            with ui.row().classes('w-full justify-between items-center pb-2 border-b flex-shrink-0 flex-nowrap'):
                with ui.column().classes('gap-0'):
                    ui.label('Configure Benchmark Prompts').classes('text-xl font-bold text-slate-800')
                    # Dynamic counter sits cleanly right beneath the title
                    count_label = ui.label().classes('text-xs font-bold text-blue-600')
                
                with ui.row().classes('items-center gap-3 flex-shrink-0'):
                    ui.button('Cancel', on_click=prompts_editor_dialog.close).props('flat color=slate')
                    
                    def save_custom_prompts():
                        try:
                            with open(LORA_PROMPTS_PATH, "w", encoding="utf-8") as f:
                                f.write(prompts_textarea_ref.value)
                            ui.notify("Benchmark prompts updated successfully!", type="positive")
                            prompts_editor_dialog.close()
                            render_lora_sidebar.refresh()
                            render_lora_workspace.refresh()
                        except Exception as err:
                            ui.notify(f"Failed to save prompts: {str(err)}", type="negative")
                            
                    ui.button('Save Prompts', icon='save', on_click=save_custom_prompts).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold px-4')

            # Secondary instruction row sitting directly under the header line
            ui.label('Modify the default list of prompts used for benchmarking your LoRAs. Enter one prompt per line. Length adapts dynamically.').classes('text-xs text-slate-500 flex-shrink-0 mt-[-4px]')
            
            # Optimized height prevents the outer card container from overflowing its viewport limits
            prompts_textarea_ref = ui.textarea(
                label="Active Prompts List (Line-by-Line)"
            ).classes('w-full h-[380px] bg-white text-xs') \
             .props('outlined input-style="height: 310px"')
            
            # Count listener hook that triggers automatically when values are loaded or changed
            def update_count_label():
                lines = [line.strip() for line in prompts_textarea_ref.value.split('\n') if line.strip()]
                count_label.set_text(f"Total: {len(lines)} active benchmark prompts")
            
            prompts_textarea_ref.on('update:value', update_count_label)

    # Stamped Stitched Contact Sheet Dialog Overlay
    with ui.dialog() as lora_contact_sheet_dialog:
        with ui.card().classes('w-full max-w-4xl p-6 rounded-xl gap-4 bg-white'):
            with ui.row().classes('w-full justify-between items-center pb-2 border-b'):
                with ui.column().classes('gap-0'):
                    ui.label('Generated LoRA Contact Sheet').classes('text-base font-bold text-slate-800')
                    ui.label('Numbered grid coordinates are stamped on the image. Perfect for sending directly to LLMs.').classes('text-xs text-slate-500')
                ui.button(
                    'Download Contact Sheet',
                    icon='download',
                    on_click=lambda: download_lora_contact_sheet(state.lora_tool_active_lora_id)
                ).classes('bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold h-9')
            
            render_lora_contact_sheet_preview(lora_contact_sheet_base64)
            
            with ui.row().classes('w-full justify-between items-center pt-2 border-t text-[10px] text-slate-400'):
                ui.label('Tip: Right-click the image to copy it directly, or click Download.')
                ui.button('Close', on_click=lora_contact_sheet_dialog.close).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs')

    with ui.row().classes('w-full justify-between items-center mb-4'):
        with ui.column().classes('gap-0'):
            ui.label('LoRA Contact Sheets').classes('text-2xl font-bold text-slate-800')
            ui.label('Generate, benchmark, and compare visual samples for your LoRA library.').classes('text-sm text-slate-500')
        
        ui.button('Exit Tool', icon='close', on_click=exit_tool_cb).props('flat dense').classes('text-slate-600')

    with ui.grid(columns='300px 1fr').classes('w-full gap-6 items-start'):
        with ui.column().classes('bg-white border rounded-xl p-4 shadow-sm h-[calc(100vh-160px)] sticky top-24 overflow-y-auto'):
            render_lora_sidebar()

        with ui.column().classes('w-full h-full'):
            render_lora_workspace()