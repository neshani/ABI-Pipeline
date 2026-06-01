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


def ensure_default_style_exists():
    """Ensures a default.json style preset file exists inside ./styles."""
    styles_dir = Path("./styles")
    styles_dir.mkdir(parents=True, exist_ok=True)
    default_file = styles_dir / "default.json"
    if not default_file.exists():
        # Look for first available workflow on disk
        workflows = []
        for d in [Path("./workflows"), Path("./Comfy_Workflows")]:
            if d.exists():
                workflows.extend([f.name for f in d.glob("*.json")])
        default_wf = workflows[0] if workflows else "default_comfy_api.json"
        
        default_content = {
            "name": "default",
            "workflow": default_wf,
            "prompt_prefix": "ArsMJStyle, 1890s Victorian illustration, detailed pen and ink with soft watercolor wash, Sidney Paget style. ",
            "negative_prompt": "blurry, bad quality, text, watermark, photorealistic, photography",
            "overrides": {}
        }
        try:
            with open(default_file, "w", encoding="utf-8") as f:
                json.dump(default_content, f, indent=2)
        except Exception:
            pass


class StyleChooserModal:
    def __init__(self, on_apply: Callable[[str], None]):
        self.on_apply = on_apply
        self.dialog = None
        self.preview_dialog = None
        self.preview_title = None
        self.preview_image_widget = None
        self.search_query = ""
        self.selected_style = "default"
        
        # Loaded metadata details
        self.style_prefix = ""
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
        ensure_default_style_exists()
        styles_dir = Path("./styles")
        styles_dir.mkdir(parents=True, exist_ok=True)
        presets = [f.stem for f in styles_dir.glob("*.json")]
        return sorted(presets)

    def load_preset_details(self, name: str):
        """Parses style json presets, seeding prompt/negative prompt variables."""
        ensure_default_style_exists()
        self.selected_style = name
        
        styles_dir = Path("./styles")
        file_path = styles_dir / f"{name}.json"
        
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                self.style_prefix = data.get("prompt_prefix", "")
                self.style_negative = data.get("negative_prompt", "")
                self.style_workflow = data.get("workflow", "")
            except Exception:
                self.style_prefix = ""
                self.style_negative = ""
                self.style_workflow = ""
        else:
            self.style_prefix = ""
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
                        prefix=self.style_prefix
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
                with ui.column().classes("gap-1 bg-slate-50 p-2 rounded border w-full text-[10px]"):
                    ui.label("PROMPT PREFIX:").classes("font-bold text-slate-400")
                    ui.label(self.style_prefix or "None").classes("text-slate-600 leading-normal truncate")

                # Visual Benchmarks Row
                with ui.grid(columns=3).classes("w-full gap-3 mt-1"):
                    # Portrait Card
                    with ui.card().classes("p-2 border rounded shadow-xs items-center gap-2 bg-white"):
                        ui.label("Portrait").classes("text-[10px] font-bold text-slate-400")
                        if self.portrait_img:
                            ui.image(self.portrait_img).classes("w-full h-24 object-cover rounded border cursor-zoom-in hover:opacity-90 transition-opacity") \
                                .on('click', lambda: self.show_large_preview("Portrait", self.portrait_img))
                        else:
                            with ui.column().classes("w-full h-24 items-center justify-center bg-slate-100 rounded border border-dashed text-slate-400"):
                                ui.icon("face", size="sm")
                                ui.label("No Sample").classes("text-[8px]")

                    # Landscape Card
                    with ui.card().classes("p-2 border rounded shadow-xs items-center gap-2 bg-white"):
                        ui.label("Landscape").classes("text-[10px] font-bold text-slate-400")
                        if self.landscape_img:
                            ui.image(self.landscape_img).classes("w-full h-24 object-cover rounded border cursor-zoom-in hover:opacity-90 transition-opacity") \
                                .on('click', lambda: self.show_large_preview("Landscape", self.landscape_img))
                        else:
                            with ui.column().classes("w-full h-24 items-center justify-center bg-slate-100 rounded border border-dashed text-slate-400"):
                                ui.icon("landscape", size="sm")
                                ui.label("No Sample").classes("text-[8px]")

                    # Architecture Card
                    with ui.card().classes("p-2 border rounded shadow-xs items-center gap-2 bg-white"):
                        ui.label("Architecture").classes("text-[10px] font-bold text-slate-400")
                        if self.architecture_img:
                            ui.image(self.architecture_img).classes("w-full h-24 object-cover rounded border cursor-zoom-in hover:opacity-90 transition-opacity") \
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
        state.style_negative_prompt = self.style_negative
        state.style_selected_workflow = self.style_workflow
        
        self.on_apply(self.selected_style)
        if self.dialog:
            self.dialog.close()

    def open(self, current_selection: str = "default"):
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
                        self.sidebar_container = ui.column().classes("w-full border rounded-lg p-2 max-h-[50vh] overflow-y-auto bg-slate-50")
                        self.preview_pane_container = ui.column().classes("w-full border rounded-lg p-4 gap-3 bg-slate-50")

                    self.refresh_sidebar()
                    self.refresh_preview()

                    with ui.row().classes("w-full justify-end gap-3 mt-2"):
                        ui.button("Cancel", on_click=self.dialog.close).props("flat color=slate").classes("text-xs font-semibold")
                        ui.button(
                            "Apply Style Preset", 
                            on_click=self.apply_selection
                        ).classes("bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold px-5 h-9 rounded")

        self.dialog.open()