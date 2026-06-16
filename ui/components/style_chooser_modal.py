import base64
import json
import asyncio
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from nicegui import ui
from database.connection import get_setting
from ui import state

# Benchmark Prompts for uniform styling comparisons
BENCHMARK_PROMPTS = {
    "portrait": "A detailed close-up portrait of an old mariner with a weathered face, looking into the distance, neutral expression.",
    "landscape": "An expansive cinematic landscape of a rocky coast with stormy waves crashing against cliffs, dramatic sky.",
    "architecture": "A majestic stone cathedral interior, high gothic arches, dust motes dancing in shafts of light from stained glass windows."
}

# The 7 Default Built-in Style Presets
DEFAULT_PRESETS = {
    "retro_graphic_novel": {
        "name": "Retro Graphic Novel",
        "workflow": "",
        "prompt_prefix": "space opera adventure graphic novel illustration, sharp ink sketch, crisp outlines, retro-futuristic sci-fi aesthetic, cosmic wonder, detailed, ",
        "prompt_suffix": ", high-contrast shadows, selective color accents, bold ink-wash shading",
        "negative_prompt": "blurry, bad quality, text, watermark, photorealistic, photography, dystopian, gritty, grimy, decay, cyberpunk",
        "overrides": {}
    },
    "fantasy_adventure": {
        "name": "Fantasy & Adventure",
        "workflow": "",
        "prompt_prefix": "whimsical storybook illustration, watercolor and ink wash, vibrant fantasy scene, detailed ink outlines, magical atmosphere, ",
        "prompt_suffix": ", soft magical glow, highly detailed, storybook aesthetic, charming, clean composition",
        "negative_prompt": "photorealistic, photography, modern, futuristic, neon, 3d render, plastic, low quality, bad anatomy, text, watermark",
        "overrides": {}
    },
    "sci_fi": {
        "name": "Sci-Fi & Cyberpunk",
        "workflow": "",
        "prompt_prefix": "cinematic digital concept art, futuristic sci-fi aesthetic, dramatic volumetric lighting, crisp focus, matte painting style, ",
        "prompt_suffix": ", rich sci-fi details, ultra-high-definition, atmospheric depth, epic composition",
        "negative_prompt": "canvas texture, watercolor, hand-drawn, cozy, medieval, low quality, text, watermark, bad anatomy, paint strokes",
        "overrides": {}
    },
    "historical_classic": {
        "name": "Historical Fiction & Drama",
        "workflow": "",
        "prompt_prefix": "classical oil painting, rich academic art style, dramatic chiaroscuro lighting, textured canvas, elegant visible brushstrokes, ",
        "prompt_suffix": ", timeless masterpiece, warm dramatic tones, fine art gallery quality",
        "negative_prompt": "anime, neon, digital art, line art, modern, futuristic, low-poly, vector, comic, cartoon, text, watermark, photorealistic",
        "overrides": {}
    },
    "anime": {
        "name": "Anime & Manga",
        "workflow": "",
        "prompt_prefix": "vibrant anime visual novel key art, clean digital line work, colorful cel shading, beautiful detailed anime background, expressive dynamic lighting, ",
        "prompt_suffix": ", high-quality anime illustration, crisp colors, modern anime aesthetic",
        "negative_prompt": "photorealistic, oil painting, watercolor, rough sketch, grimy, textured canvas, 3d render, real-world texture, text, watermark, bad proportions",
        "overrides": {}
    },
    "cozy_mystery": {
        "name": "Cozy Mystery & Slice of Life",
        "workflow": "",
        "prompt_prefix": "stylized cozy mystery book cover art, modern flat vector illustration with subtle grain texture, warm and inviting atmosphere, clean lines, charming color palette, ",
        "prompt_suffix": ", whimsical layout, minimalist details, cozy slice-of-life aesthetic",
        "negative_prompt": "gritty, violent, terrifying, dark fantasy, heavy metallic, photorealistic, photography, messy sketches, text, watermark, messy lines",
        "overrides": {}
    },
    "childrens_fiction": {
        "name": "Children's Fiction",
        "workflow": "",
        "prompt_prefix": "soft gouache and colored pencil illustration, whimsical children's book style, gentle textures, cozy and warm atmosphere, pastel color palette, ",
        "prompt_suffix": ", adorable and friendly aesthetic, clean storytelling layout, cozy magic",
        "negative_prompt": "dark, gritty, scary, photorealistic, neon, high contrast shadows, cyberpunk, complex digital painting, text, watermark, violent, gloomy",
        "overrides": {}
    }
}


def get_image_base64_or_placeholder(image_path: Path) -> Optional[str]:
    """Reads a local file and encodes it as a base64 string for direct rendering."""
    if image_path.exists():
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            encoded = base64.b64encode(data).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
        except Exception:
            pass
    return None


def get_available_workflows() -> List[str]:
    """Scans workflow directories and returns a unique list of available json files on disk."""
    workflows = []
    for d in [Path("./workflows"), Path("./Comfy_Workflows")]:
        if d.exists():
            workflows.extend([f.name for f in d.glob("*.json")])
    return sorted(list(set(workflows)))


def heal_style_workflow(preset_name: str, preset_data: dict) -> dict:
    """Checks if the assigned workflow is missing or blank, then resolves the best match dynamically."""
    current_wf = preset_data.get("workflow", "")
    
    # 1. Check if the currently bound workflow already exists on disk
    wf_valid = False
    if current_wf:
        for d in [Path("./workflows"), Path("./Comfy_Workflows")]:
            if (d / current_wf).exists():
                wf_valid = True
                break
                
    if wf_valid:
        return preset_data

    # 2. Since it is missing or blank, load available workflows
    workflows = get_available_workflows()
    if not workflows:
        preset_data["workflow"] = "default_comfy_api.json"
        return preset_data

    # 3. Apply keyword matching strategies
    resolved_wf = ""
    name_lower = preset_name.lower()
    
    if "anime" in name_lower:
        for wf in workflows:
            wf_lower = wf.lower()
            if "anima" in wf_lower or "anime" in wf_lower:
                resolved_wf = wf
                break
        if not resolved_wf:
            for wf in workflows:
                if "turbo" in wf.lower() or "lightning" in wf.lower():
                    resolved_wf = wf
                    break
    elif "sci_fi" in name_lower:
        for wf in workflows:
            wf_lower = wf.lower()
            if "scifi" in wf_lower or "sci-fi" in wf_lower or "flux" in wf_lower or "sdxl" in wf_lower:
                resolved_wf = wf
                break

    # 4. Fallback to the first available alphabetical choice
    if not resolved_wf:
        resolved_wf = workflows[0]

    preset_data["workflow"] = resolved_wf

    # 5. Save healed preset back to disk so the decision is persisted
    file_path = Path("./styles") / f"{preset_name}.json"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(preset_data, f, indent=2)
    except Exception:
        pass

    return preset_data


def ensure_default_presets_exist():
    """Seeds the styles directory with core genre presets only if the directory contains zero JSON files."""
    styles_dir = Path("./styles")
    styles_dir.mkdir(parents=True, exist_ok=True)
    
    existing_presets = list(styles_dir.glob("*.json"))
    if not existing_presets:
        for preset_key, content in DEFAULT_PRESETS.items():
            file_path = styles_dir / f"{preset_key}.json"
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=2)
            except Exception:
                pass


def ensure_default_style_exists():
    """Backward-compatible wrapper to maintain visual dashboard entry points during refactoring imports."""
    ensure_default_presets_exist()


class StyleChooserModal:
    def __init__(self, on_apply: Callable[[str], None]):
        self.on_apply = on_apply
        self.dialog = None
        self.preview_dialog = None
        self.preview_title = None
        self.preview_image_widget = None
        self.search_query = ""
        self.selected_style = "retro_graphic_novel"
        
        # Loaded metadata details
        self.style_prefix = ""
        self.style_suffix = ""
        self.style_negative = ""
        self.style_workflow = ""
        
        # Image base64 variables
        self.portrait_img = None
        self.landscape_img = None
        self.architecture_img = None
        self.generating_samples = False

        # Component References
        self.sidebar_container = None
        self.preview_pane_container = None

    def list_styles(self) -> List[str]:
        """Discovers .json files inside local styles directory."""
        ensure_default_presets_exist()
        styles_dir = Path("./styles")
        styles_dir.mkdir(parents=True, exist_ok=True)
        presets = [f.stem for f in styles_dir.glob("*.json")]
        return sorted(presets)

    def load_preset_details(self, name: str):
        """Parses style json presets, seeding prompt/negative prompt variables with dynamic healer fallback."""
        ensure_default_presets_exist()
        self.selected_style = name
        
        styles_dir = Path("./styles")
        file_path = styles_dir / f"{name}.json"
        
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Heal and persist missing/broken workflow definitions dynamically
                data = heal_style_workflow(name, data)
                
                self.style_prefix = data.get("prompt_prefix", "")
                self.style_suffix = data.get("prompt_suffix", "")
                self.style_negative = data.get("negative_prompt", "")
                self.style_workflow = data.get("workflow", "")
            except Exception:
                self.style_prefix = ""
                self.style_suffix = ""
                self.style_negative = ""
                self.style_workflow = ""
        else:
            self.style_prefix = ""
            self.style_suffix = ""
            self.style_negative = ""
            self.style_workflow = ""
                    
        # Load local image base64 cache
        samples_dir = Path("./styles/samples") / name
        self.portrait_img = get_image_base64_or_placeholder(samples_dir / "portrait.png")
        self.landscape_img = get_image_base64_or_placeholder(samples_dir / "landscape.png")
        self.architecture_img = get_image_base64_or_placeholder(samples_dir / "architecture.png")

    async def generate_samples(self):
        """Dispatches sample generation directly in ComfyUI for standard benchmark comparisons."""
        self.generating_samples = True
        self.refresh_preview()

        try:
            # Connect to ComfyUI
            comfy_url = get_setting("comfy_url", "127.0.0.1:8188")
            if "http" in comfy_url:
                comfy_url = comfy_url.replace("http://", "").replace("https://", "").strip("/")

            wf_name = self.style_workflow
            if not wf_name:
                raise ValueError("No base workflow is configured for this style.")

            wf_path = Path("./workflows") / wf_name
            if not wf_path.exists():
                wf_path = Path("./Comfy_Workflows") / wf_name

            if not wf_path.exists():
                raise FileNotFoundError(f"Base workflow JSON '{wf_name}' was not found. Please verify placement.")

            with open(wf_path, "r") as f:
                wf_json = json.load(f)

            from services.comfy_client import ComfyClient
            client = ComfyClient(comfy_url)

            # Build output folder
            samples_dir = Path("./styles/samples") / self.selected_style
            samples_dir.mkdir(parents=True, exist_ok=True)

            for category, prompt_text in BENCHMARK_PROMPTS.items():
                seed = 42
                state.add_console_log(f"[Style-Chooser] Rendering benchmark '{category}' for '{self.selected_style}'...")
                
                overrides = {}
                styles_dir = Path("./styles")
                file_path = styles_dir / f"{self.selected_style}.json"
                if file_path.exists():
                    try:
                        with open(file_path, "r") as f:
                            overrides = json.load(f).get("overrides", {})
                    except Exception:
                        pass

                def render_block():
                    return client.generate_image_sync(
                        workflow_json=wf_json,
                        prompt_text=prompt_text,
                        neg_prompt_text=self.style_negative,
                        seed=seed,
                        overrides=overrides,
                        prefix=self.style_prefix,
                        suffix=self.style_suffix
                    )

                try:
                    img_bytes, logs = await asyncio.to_thread(render_block)
                    if img_bytes:
                        target_file = samples_dir / f"{category}.png"
                        with open(target_file, "wb") as f:
                            f.write(img_bytes)
                    for line in logs.split("\n"):
                        if line.strip():
                            state.add_console_log(line)
                except Exception as e:
                    state.add_console_log(f"[Style-Chooser] Generation failed for category '{category}': {str(e)}")

            # Reload updated local caches
            self.portrait_img = get_image_base64_or_placeholder(samples_dir / "portrait.png")
            self.landscape_img = get_image_base64_or_placeholder(samples_dir / "landscape.png")
            self.architecture_img = get_image_base64_or_placeholder(samples_dir / "architecture.png")

            try:
                with self.dialog:
                    ui.notify(f"Style samples generated successfully for '{self.selected_style}'!", type="positive")
            except Exception:
                pass

        except Exception as e:
            state.add_console_log(f"[Style-Chooser] General error during sample generation: {str(e)}")
            try:
                with self.dialog:
                    ui.notify(f"Failed to generate samples: {str(e)}", type="negative")
            except Exception:
                pass
        finally:
            self.generating_samples = False
            self.refresh_preview()

    def refresh_sidebar(self):
        """Clears and rebuilds the left sidebar list containing style items."""
        if self.sidebar_container:
            self.sidebar_container.clear()
            with self.sidebar_container:
                styles = self.list_styles()
                filtered = [s for s in styles if self.search_query.lower() in s.lower()]
                if not filtered:
                    ui.label("No styles found.").classes("text-xs text-slate-400 italic p-3")
                    return

                for style in filtered:
                    is_active = style == self.selected_style
                    bg_color = "bg-blue-50 text-blue-700 font-bold" if is_active else "hover:bg-slate-50 text-slate-700"
                    with ui.row().classes(f"w-full p-2.5 rounded-lg cursor-pointer items-center justify-between transition-all {bg_color}") \
                            .on('click', lambda _, s=style: self.select_style(s)):
                        ui.label(style).classes("text-xs truncate")
                        if is_active:
                            ui.icon("check", color="blue", size="xs")

    def show_large_preview(self, category_title: str, image_base64: str):
        """Populates and displays the nested secondary preview dialog with the selected image."""
        if hasattr(self, 'preview_dialog') and self.preview_dialog:
            self.preview_title.set_text(f"{self.selected_style} — {category_title}")
            self.preview_image_widget.set_source(image_base64)
            self.preview_dialog.open()

    def refresh_preview(self):
        """Refreshes the right-hand metadata details and image cards."""
        if self.preview_pane_container:
            self.preview_pane_container.clear()
            with self.preview_pane_container:
                # Header Details
                with ui.row().classes("w-full justify-between items-center pb-2 border-b"):
                    with ui.column().classes("gap-0"):
                        ui.label(f"Style Preset: {self.selected_style}").classes("text-base font-bold text-slate-800")
                        ui.label(f"Workflow: {self.style_workflow or 'default_comfy_api.json'}").classes("text-[10px] text-slate-400")
                    
                    if self.generating_samples:
                        ui.button("Generating...", icon="hourglass_empty").props("disabled").classes("text-xs")
                    else:
                        ui.button(
                            "Generate Samples", 
                            icon="sync", 
                            on_click=self.generate_samples
                        ).classes("bg-blue-600 text-white text-xs font-semibold px-3 py-1.5 rounded h-8 shadow-xs")

                # Prefix Text Block Preview
                with ui.column().classes("gap-1 bg-slate-50 p-2 rounded border w-full text-[10px] min-w-0"):
                    ui.label("PROMPT PREFIX:").classes("font-bold text-slate-400")
                    ui.label(self.style_prefix or "None").classes("text-slate-600 leading-normal break-words whitespace-normal")

                # Suffix Text Block Preview
                with ui.column().classes("gap-1 bg-slate-50 p-2 rounded border w-full text-[10px] min-w-0"):
                    ui.label("PROMPT SUFFIX:").classes("font-bold text-slate-400")
                    ui.label(self.style_suffix or "None").classes("text-slate-600 leading-normal break-words whitespace-normal")

                # Visual Benchmarks Row
                with ui.grid(columns=3).classes("w-full gap-3 mt-1"):
                    # Portrait Card
                    with ui.card().classes("p-2 border rounded shadow-xs items-center gap-2 bg-white"):
                        ui.label("Portrait").classes("text-[10px] font-bold text-slate-400")
                        if self.portrait_img:
                            ui.image(self.portrait_img).props("fit=contain").classes("w-full rounded border cursor-zoom-in hover:opacity-90 transition-opacity") \
                                .on('click', lambda: self.show_large_preview("Portrait", self.portrait_img))
                        else:
                            with ui.column().classes("w-full h-24 items-center justify-center bg-slate-100 rounded border border-dashed text-slate-400"):
                                ui.icon("face", size="sm")
                                ui.label("No Sample").classes("text-[8px]")

                    # Landscape Card
                    with ui.card().classes("p-2 border rounded shadow-xs items-center gap-2 bg-white"):
                        ui.label("Landscape").classes("text-[10px] font-bold text-slate-400")
                        if self.landscape_img:
                            ui.image(self.landscape_img).props("fit=contain").classes("w-full rounded border cursor-zoom-in hover:opacity-90 transition-opacity") \
                                .on('click', lambda: self.show_large_preview("Landscape", self.landscape_img))
                        else:
                            with ui.column().classes("w-full h-24 items-center justify-center bg-slate-100 rounded border border-dashed text-slate-400"):
                                ui.icon("landscape", size="sm")
                                ui.label("No Sample").classes("text-[8px]")

                    # Architecture Card
                    with ui.card().classes("p-2 border rounded shadow-xs items-center gap-2 bg-white"):
                        ui.label("Architecture").classes("text-[10px] font-bold text-slate-400")
                        if self.architecture_img:
                            ui.image(self.architecture_img).props("fit=contain").classes("w-full rounded border cursor-zoom-in hover:opacity-90 transition-opacity") \
                                .on('click', lambda: self.show_large_preview("Architecture", self.architecture_img))
                        else:
                            with ui.column().classes("w-full h-24 items-center justify-center bg-slate-100 rounded border border-dashed text-slate-400"):
                                ui.icon("account_balance", size="sm")
                                ui.label("No Sample").classes("text-[8px]")

    def select_style(self, name: str):
        """Swaps the active preview selection and repopulates visual containers."""
        self.load_preset_details(name)
        self.refresh_sidebar()
        self.refresh_preview()

    def apply_selection(self):
        """Fires the client callback applying this style selection, then closes the dialog."""
        # Write applied metadata selections directly back to the global state bindings
        state.style_selected_preset = self.selected_style
        state.style_prompt_prefix = self.style_prefix
        state.style_prompt_suffix = self.style_suffix
        state.style_negative_prompt = self.style_negative
        state.style_selected_workflow = self.style_workflow
        
        self.on_apply(self.selected_style)
        if self.dialog:
            self.dialog.close()

    def open(self, current_selection: str = "retro_graphic_novel"):
        """Instantiates the modal in memory and opens the dialog."""
        self.selected_style = current_selection
        self.load_preset_details(current_selection)

        with ui.context.client:
            # Nested Large Image Preview Dialog
            with ui.dialog() as self.preview_dialog:
                with ui.card().classes("w-full max-w-3xl p-4 items-center bg-white rounded-xl shadow-lg gap-2"):
                    self.preview_title = ui.label().classes("text-sm font-bold text-slate-800 uppercase tracking-wider mb-2")
                    self.preview_image_widget = ui.image().props("fit=contain").classes("w-full rounded-lg max-h-[70vh] cursor-zoom-out bg-slate-50") \
                        .on('click', lambda: self.preview_dialog.close())
                    with ui.row().classes("w-full justify-end mt-2"):
                        ui.button("Close", on_click=self.preview_dialog.close).classes("bg-slate-700 hover:bg-slate-800 text-white text-xs")

            # Main Style Preset Selection Dialog
            with ui.dialog() as self.dialog:
                with ui.card().classes("w-full max-w-4xl p-6 rounded-xl gap-4 bg-white"):
                    with ui.row().classes("w-full justify-between items-center pb-2 border-b"):
                        ui.label("Select Visual Style Preset").classes("text-lg font-bold text-slate-800")
                        ui.input(
                            placeholder="Search styles...", 
                            on_change=lambda e: (setattr(self, 'search_query', e.value), self.refresh_sidebar())
                        ).props("outlined dense").classes("w-48")

                    with ui.grid(columns="240px 1fr").classes("w-full gap-4 items-start"):
                        self.sidebar_container = ui.column().classes("w-full border rounded-lg p-2 max-h-[50vh] overflow-y-auto bg-slate-50 min-w-0")
                        self.preview_pane_container = ui.column().classes("w-full border rounded-lg p-4 gap-3 bg-slate-50 min-w-0")

                    self.refresh_sidebar()
                    self.refresh_preview()

                    with ui.row().classes("w-full justify-end gap-3 mt-2"):
                        ui.button("Cancel", on_click=self.dialog.close).props("flat color=slate").classes("text-xs font-semibold")
                        ui.button(
                            "Apply Style Preset", 
                            on_click=self.apply_selection
                        ).classes("bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold px-5 h-9 rounded")

        self.dialog.open()