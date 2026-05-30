import json
import random
import datetime
import asyncio
import os
from pathlib import Path
from typing import Any, List, Optional, Dict
from nicegui import ui
from sqlmodel import Session, select
from database.models import Project, Book
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
from database.connection import get_setting, engine

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


# --- DATABASE AND DISK ROLLBACK HELPERS ---

import datetime
from sqlmodel import select

def backup_and_cleanup_files(project_id: int, target_status: str):
    """Safely renames directories and csv/txt files to prevent loss and reset pipeline states on disk."""
    from sqlmodel import Session
    from database.connection import engine
    from database.models import Project, Book
    
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            state.add_console_log("[Rollback-Engine] Error: Project not found.")
            return
        project_name = project.name
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(get_setting("output_dir", "./output")).resolve()
    
    state.add_console_log(f"[Rollback-Engine] Initiating file archiver for project: {project_name}")
    
    for book in books:
        book_dir = output_base / project_name / book.name
        if not book_dir.exists():
            state.add_console_log(f"[Rollback-Engine] Directory not found for book: {book.name}")
            continue
            
        state.add_console_log(f"[Rollback-Engine] Archiving files in book directory: {book.name}")
            
        if target_status == "Imported":
            # Transcription rollback: Move transcript.txt, prompts.csv, images/
            transcript_path = book_dir / "transcript.txt"
            if transcript_path.exists():
                new_transcript = book_dir / f"transcript_backup_{timestamp}.txt"
                try:
                    transcript_path.rename(new_transcript)
                    state.add_console_log(f"[Rollback-Engine] Archived transcript to: {new_transcript.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving transcript.txt: {str(e)}")
                    
            prompts_path = book_dir / "prompts.csv"
            if prompts_path.exists():
                new_prompts = book_dir / f"prompts_backup_{timestamp}.csv"
                try:
                    prompts_path.rename(new_prompts)
                    state.add_console_log(f"[Rollback-Engine] Archived prompts to: {new_prompts.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving prompts.csv: {str(e)}")
                    
            images_dir = book_dir / "images"
            if images_dir.exists() and images_dir.is_dir():
                new_images = book_dir / f"images_backup_{timestamp}"
                try:
                    images_dir.rename(new_images)
                    state.add_console_log(f"[Rollback-Engine] Archived images folder to: {new_images.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving images folder: {str(e)}")
                    
        elif target_status == "Transcribed":
            # Prompt gen rollback: Keep transcript.txt, but archive prompts.csv and images/
            prompts_path = book_dir / "prompts.csv"
            if prompts_path.exists():
                new_prompts = book_dir / f"prompts_backup_{timestamp}.csv"
                try:
                    prompts_path.rename(new_prompts)
                    state.add_console_log(f"[Rollback-Engine] Archived prompts to: {new_prompts.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving prompts.csv: {str(e)}")
                    
            images_dir = book_dir / "images"
            if images_dir.exists() and images_dir.is_dir():
                new_images = book_dir / f"images_backup_{timestamp}"
                try:
                    images_dir.rename(new_images)
                    state.add_console_log(f"[Rollback-Engine] Archived images folder to: {new_images.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving images folder: {str(e)}")
                    
        elif target_status == "Prompts Created":
            # Image Gen rollback: Keep transcript.txt & prompts.csv, but archive images/
            images_dir = book_dir / "images"
            if images_dir.exists() and images_dir.is_dir():
                new_images = book_dir / f"images_backup_{timestamp}"
                try:
                    images_dir.rename(new_images)
                    state.add_console_log(f"[Rollback-Engine] Archived images folder to: {new_images.name}")
                except Exception as e:
                    state.add_console_log(f"[Rollback-Engine] Error archiving images folder: {str(e)}")
                    
    state.add_console_log("[Rollback-Engine] State rollback file archival step complete.")


def rollback_project_status(project_id: int, target_status: str, refresh_callback) -> None:
    """Safely shifts project and book statuses backwards to allow pipeline step re-runs."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return
        project.status = target_status
        session.add(project)
        
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        for b in books:
            b.status = target_status
            if target_status == "Imported":
                b.progress = 0.0
            elif target_status == "Transcribed":
                b.progress = 0.0
            session.add(b)
            
        session.commit()
        
    state.project_status = target_status
    ui.notify(f"Project rolled back to: {target_status}", type="warning")
    refresh_callback()
    
    # Refresh global topbar progress stepper
    if hasattr(state, 'active_header_refresh') and state.active_header_refresh:
        state.active_header_refresh()


# --- STEP-AWARE COMPACT DASHBOARD VIEWS ---

def render_transcription_step_view(project, books, start_transcribe_cb, stop_transcribe_cb):
    with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-3'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('record_voice_over', size='sm', color='blue-500')
            ui.label('Phase 1: Speech-to-Text Transcription').classes('text-sm font-bold text-slate-800')
            
        ui.label('The local STT pipeline scans and transcribes audiobook track files into chapter-level transcript.txt files.').classes('text-xs text-slate-500')
        
        with ui.row().classes('items-center gap-4 bg-blue-50/50 p-3 rounded-lg border border-blue-100/50 w-full mt-1'):
            ui.icon('info', color='blue', size='sm')
            ui.label(f'Active STT Config: {get_setting("stt_engine", "Parakeet ONNX")} | Device: {get_setting("stt_device", "GPU/CUDA")}').classes('text-xs font-semibold text-slate-700')
            
        with ui.row().classes('w-full justify-between items-center mt-2 pt-2 border-t border-slate-100'):
            if state.project_status == "Transcribing":
                with ui.row().classes('items-center gap-2'):
                    ui.spinner(size='sm', color='blue')
                    ui.label('Transcribing audio files in background...').classes('text-xs font-semibold text-blue-700 animate-pulse')
                ui.button(
                    'Stop Transcription', 
                    icon='stop', 
                    color='red', 
                    on_click=lambda: stop_transcribe_cb(project.id)
                ).classes('px-4 font-semibold text-xs')
            else:
                ui.label('Awaiting transcription pipeline initiation.').classes('text-xs font-semibold text-slate-400')
                ui.button(
                    'Start Transcription', 
                    icon='play_arrow', 
                    on_click=lambda: start_transcribe_cb(project.id)
                ).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs px-5')


def render_prompt_gen_step_view(project, books, start_prompt_gen_cb, stop_transcribe_cb, trigger_rollback_cb):
    available_templates = list_stored_templates()
    
    with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-3'):
        with ui.row().classes('w-full justify-between items-center'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('psychology', size='sm', color='purple-500')
                ui.label('Phase 2: LLM Prompt Generation & Extraction').classes('text-sm font-bold text-slate-800')
            
            ui.button(
                'Rollback to Transcription',
                icon='history',
                on_click=lambda: trigger_rollback_cb("Imported")
            ).classes('text-[10px] text-slate-500 hover:text-red-600').props('flat dense')
            
        ui.label('Transcription passages are processed through a local LLM to extract key quotes and generate visual rendering prompts.').classes('text-xs text-slate-500')
        
        with ui.row().classes('w-full items-center gap-3 bg-slate-50 p-3 rounded-lg border mt-1'):
            ui.icon('description', size='sm', color='slate-500')
            with ui.column().classes('gap-0 flex-1'):
                ui.label('Active Guidelines Prompt Template').classes('text-xs font-bold text-slate-700')
                ui.label('Select template rules. Customize rules or add templates inside the Playground tab.').classes('text-[9px] text-slate-500')
            
            ui.select(
                options=available_templates,
                value=state.playground_selected_template,
                on_change=lambda e: handle_dashboard_template_change(e.value)
            ).classes('w-48 bg-white').props('outlined dense')

        with ui.row().classes('w-full justify-between items-center mt-2 pt-2 border-t border-slate-100'):
            if state.project_status == "Generating Prompts":
                with ui.row().classes('items-center gap-2'):
                    ui.spinner(size='sm', color='purple')
                    ui.label('Generating prompts in background...').classes('text-xs font-semibold text-purple-700 animate-pulse')
                ui.button(
                    'Stop Prompt Gen', 
                    icon='stop', 
                    color='red', 
                    on_click=lambda: stop_transcribe_cb(project.id)
                ).classes('px-4 font-semibold text-xs')
            else:
                ui.label('Guidelines configured. Ready to run local prompt generation.').classes('text-xs font-semibold text-slate-400')
                ui.button(
                    'Generate Prompts', 
                    icon='bolt', 
                    on_click=lambda: start_prompt_gen_cb(project.id) if start_prompt_gen_cb else None
                ).classes('bg-purple-600 hover:bg-purple-700 text-white font-bold text-xs px-5')


def render_image_gen_step_view(project, books, start_image_gen_cb, stop_transcribe_cb, trigger_rollback_cb):
    with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-3'):
        with ui.row().classes('w-full justify-between items-center'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('image', size='sm', color='amber-500')
                ui.label('Phase 3: ComfyUI Image Generation').classes('text-sm font-bold text-slate-800')
            
            ui.button(
                'Rollback to Prompt Gen',
                icon='history',
                on_click=lambda: trigger_rollback_cb("Transcribed")
            ).classes('text-[10px] text-slate-500 hover:text-red-600').props('flat dense')
            
        ui.label('Render image batches via ComfyUI. Visual style settings and base workflows can be adjusted inside the Style & Workflows tab.').classes('text-xs text-slate-500')
        
        with ui.row().classes('w-full items-center gap-3 bg-slate-50 p-3 rounded-lg border mt-1'):
            ui.icon('brush', size='sm', color='slate-500')
            with ui.column().classes('gap-0 flex-1'):
                ui.label('Active Visual Style Preset').classes('text-xs font-bold text-slate-700')
                ui.label('Prefixes and details applied globally onto active image prompts.').classes('text-[9px] text-slate-500')
            
            ui.select(
                options=load_style_presets(),
                value=state.style_selected_preset,
                on_change=lambda e: (setattr(state, 'style_selected_preset', e.value), load_style_preset_by_name(e.value))
            ).classes('w-48 bg-white').props('outlined dense')

        with ui.row().classes('w-full justify-between items-center mt-2 pt-2 border-t border-slate-100'):
            if state.project_status == "Rendering Images":
                with ui.row().classes('items-center gap-2'):
                    ui.spinner(size='sm', color='amber')
                    ui.label('Rendering images in background...').classes('text-xs font-semibold text-amber-700 animate-pulse')
                ui.button(
                    'Stop Rendering', 
                    icon='stop', 
                    color='red', 
                    on_click=lambda: stop_transcribe_cb(project.id)
                ).classes('px-4 font-semibold text-xs')
            else:
                ui.label('Batch prompts ready. Initiates automated image renderings.').classes('text-xs font-semibold text-slate-400')
                ui.button(
                    'Render Images', 
                    icon='play_circle_filled', 
                    on_click=lambda: start_image_gen_cb(project.id) if start_image_gen_cb else None
                ).classes('bg-amber-600 hover:bg-amber-700 text-white font-bold text-xs px-5')


def render_completed_step_view(project, books, trigger_rollback_cb):
    with ui.card().classes('w-full border p-5 shadow-sm bg-emerald-50/10 gap-3'):
        with ui.row().classes('w-full justify-between items-center'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('task_alt', size='sm', color='emerald-500')
                ui.label('Automatic Execution Phases Completed!').classes('text-sm font-bold text-slate-800')
            
            ui.button(
                'Rollback to Image Gen',
                icon='history',
                on_click=lambda: trigger_rollback_cb("Prompts Created")
            ).classes('text-[10px] text-slate-500 hover:text-red-600').props('flat dense')
            
        ui.label('All automatic generation steps have completed. You can now proofread chapters, tune image prompts, and bake description metadata.').classes('text-xs text-slate-500')
        
        with ui.row().classes('w-full gap-4 items-center bg-white p-4 rounded-lg border mt-2'):
            ui.icon('info', color='emerald')
            with ui.column().classes('gap-0 flex-1'):
                ui.label('Next Step: Workspace Proofreader').classes('text-xs font-bold text-slate-700')
                ui.label('Select a specific book in the left sidebar to open the Proofreader & Interactive Editor Grid.').classes('text-[10px] text-slate-500')

# --- PREVIEWS & CLIPBOARD ---

def open_large_image(img_base64: str, title: str):
    """Opens a modal popup dialog displaying the full-size rendered image using stable global references."""
    state.preview_image_title = title
    state.preview_image_src = img_base64
    if hasattr(state, 'global_preview_dialog') and state.global_preview_dialog:
        state.global_preview_dialog.open()


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
def render_recent_images_feed():
    """Renders a real-time horizontal strip of the 5 most recent generated images."""
    with ui.card().classes('w-full border p-5 shadow-sm bg-white mt-4 gap-3'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('photo_library', size='sm', color='blue-500')
            ui.label('Live Rendered Images Feed (Most Recent)').classes('text-sm font-bold text-slate-800')
            
        if not state.recent_rendered_images:
            with ui.column().classes('w-full items-center justify-center p-6 bg-slate-50 border border-dashed rounded-lg text-slate-400'):
                ui.icon('image', size='md', color='slate-300')
                ui.label('Images will stream here in real-time as they are completed by ComfyUI...').classes('text-xs text-center')
        else:
            with ui.row().classes('w-full gap-4 items-start overflow-x-auto pb-2 flex-nowrap'):
                for item in reversed(state.recent_rendered_images):
                    with ui.card().classes('w-44 border p-2 rounded-lg shadow-xs bg-slate-50 flex-shrink-0 cursor-pointer') \
                            .on('click', lambda _, img=item['base64'], title=f"Ch {item['chapter']}, Scene {item['scene']}": open_large_image(img, title)):
                        ui.image(item['base64']).classes('w-full h-28 rounded object-cover border')
                        with ui.column().classes('gap-0 mt-1'):
                            ui.label(f"Ch {item['chapter']}, Scene {item['scene']}").classes('text-[10px] font-bold text-slate-700')
                            ui.label(item['prompt'][:40] + "...").classes('text-[8px] text-slate-500 leading-tight')

    # Register refresh callback globally so background updates bind successfully
    state.recent_images_refresh = render_recent_images_feed.refresh


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
        render_workflow_overrides_ui.refresh()
        return
        
    styles_dir = Path("./styles")
    file_path = styles_dir / f"{name}.json"
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                
                # Associated workflow loading logic
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
            client = ComfyClient("127.0.0.1:8188")  # Mock address for analyzer pass
            state.style_discovered_params = client.analyze_workflow(wf_json)
            if clear_overrides:
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
    start_image_gen_cb=None,
    save_project_settings_cb=None
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

    # Instantiate a stable page-level dialog once
    with ui.dialog() as global_preview_dialog:
        with ui.card().classes('w-full max-w-3xl p-4 items-center bg-white rounded-xl shadow-lg'):
            ui.label().classes('text-sm font-bold text-slate-800 mb-3 uppercase tracking-wider').bind_text_from(state, 'preview_image_title')
            ui.image().classes('w-full rounded-lg max-h-[75vh] object-contain cursor-zoom-out').bind_source_from(state, 'preview_image_src').on('click', global_preview_dialog.close)
            with ui.row().classes('w-full justify-end mt-3'):
                ui.button('Close', on_click=global_preview_dialog.close).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs')
                
    state.global_preview_dialog = global_preview_dialog

    # Reusable rollback confirmation dialog
    with ui.dialog() as rollback_dialog, ui.card().classes('w-full max-w-md p-6 rounded-xl gap-4'):
        ui.label('Confirm State Rollback').classes('text-lg font-bold text-slate-800')
        
        warning_msg = ui.label().classes('text-xs text-slate-600 leading-relaxed')
        backup_details = ui.markdown().classes('text-[10px] text-slate-500 bg-slate-50 p-2.5 rounded border border-slate-200 font-mono leading-normal w-full')
        
        rollback_target = {"status": "Imported"}
        
        async def confirm_action():
            target = rollback_target["status"]
            ui.notify("Archiving and renaming on-disk folders...", type="info")
            await asyncio.to_thread(backup_and_cleanup_files, project.id, target)
            rollback_project_status(project.id, target, render_dynamic_step_dashboard.refresh)
            rollback_dialog.close()
            
        with ui.row().classes('w-full justify-end gap-3 mt-2'):
            ui.button('Cancel', on_click=rollback_dialog.close).props('flat color=slate').classes('text-xs font-semibold')
            ui.button('Confirm & Rollback', on_click=confirm_action, color='red').classes('text-xs font-bold text-white')

    def trigger_rollback_prompt(target_status: str):
        rollback_target["status"] = target_status
        if target_status == "Imported":
            warning_msg.set_text("You are rolling back to the Transcription phase. This will archive active transcripts, prompt files, and rendered images so transcription can be executed fresh.")
            backup_details.set_content(
                "**Action items:**\n"
                "- Rename `transcript.txt` ➔ `transcript_backup_*.txt`\n"
                "- Rename `prompts.csv` ➔ `prompts_backup_*.csv`\n"
                "- Rename `images/` ➔ `images_backup_*`"
            )
        elif target_status == "Transcribed":
            warning_msg.set_text("You are rolling back to the Prompt Generation phase. This will keep transcripts intact, but archive your active prompt list and rendered images so scene prompts can be generated fresh.")
            backup_details.set_content(
                "**Action items:**\n"
                "- Rename `prompts.csv` ➔ `prompts_backup_*.csv`\n"
                "- Rename `images/` ➔ `images_backup_*`"
            )
        elif target_status == "Prompts Created":
            warning_msg.set_text("You are rolling back to the Image Generation phase. This will keep transcripts and prompt files intact, but archive your rendered images folder so you can clean render the entire visual list.")
            backup_details.set_content(
                "**Action items:**\n"
                "- Rename `images/` ➔ `images_backup_*`"
            )
        rollback_dialog.open()

    # Workspace Navigation Layout with Folder Shortcut
    with ui.row().classes('w-full justify-between items-center mb-1'):
        with ui.row().classes('items-center gap-3'):
            with ui.column().classes('gap-0'):
                ui.label('Project Workspace Controls').classes('text-base font-bold text-slate-800')
                ui.label('Configure orchestration guidelines and render dynamic style models.').classes('text-xs text-slate-500')
            
            def open_project_folder():
                import platform
                import subprocess
                base_dir = Path(get_setting("output_dir", "./output")).resolve()
                proj_dir = base_dir / project.name
                proj_dir.mkdir(parents=True, exist_ok=True)
                try:
                    if platform.system() == "Windows":
                        os.startfile(proj_dir)
                    elif platform.system() == "Darwin":
                        subprocess.Popen(["open", str(proj_dir)])
                    else:
                        subprocess.Popen(["xdg-open", str(proj_dir)])
                except Exception as ex:
                    ui.notify(f"Failed to open project folder: {str(ex)}", type="negative")

            ui.button(
                'Open Folder', 
                icon='folder_open', 
                on_click=open_project_folder
            ).props('flat dense').classes('text-xs text-slate-600')
    
    with ui.tabs().classes('w-full border-b') as project_tabs:
        tab_dash = ui.tab('Dashboard', icon='dashboard')
        tab_style = ui.tab('Style & Workflows', icon='brush')
        tab_play = ui.tab('Prompt-Gen Playground', icon='science')
        
    project_tabs.bind_value(state, 'active_project_tab')
        
    with ui.tab_panels(project_tabs, value=state.active_project_tab).classes('w-full bg-transparent p-0'):
        with ui.tab_panel(tab_dash):
            # --- Dynamic Step-Aware Dashboard ---
            with ui.column().classes('w-full gap-4'):
                @ui.refreshable
                def render_dynamic_step_dashboard():
                    status = state.project_status
                    
                    if status in ("Imported", "Transcribing"):
                        render_transcription_step_view(project, books, start_transcribe_cb, stop_transcribe_cb)
                    elif status in ("Transcribed", "Generating Prompts"):
                        render_prompt_gen_step_view(project, books, start_prompt_gen_cb, stop_transcribe_cb, trigger_rollback_prompt)
                    elif status in ("Prompts Created", "Rendering Images"):
                        render_image_gen_step_view(project, books, start_image_gen_cb, stop_transcribe_cb, trigger_rollback_prompt)
                    else:
                        render_completed_step_view(project, books, trigger_rollback_prompt)
                        
                render_dynamic_step_dashboard()

            # Dynamic visibility container for conditional dashboard feeds
            @ui.refreshable
            def render_conditional_feeds():
                status = state.project_status
                # Live Render Feed - visible only from image gen stage onwards
                if status not in ("Imported", "Transcribing", "Transcribed", "Generating Prompts"):
                    render_recent_images_feed()
                    
            render_conditional_feeds()
            
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

            # Dynamic visibility container for generated prompt feed
            @ui.refreshable
            def render_conditional_prompt_feed():
                status = state.project_status
                # Live Prompt Feed - visible only from prompt gen stage onwards
                if status not in ("Imported", "Transcribing"):
                    render_recent_prompts_feed()
                    
            render_conditional_prompt_feed()

            # Sync the action_buttons_refresh callback to update all dynamically changing parts
            state.action_buttons_refresh = lambda: (
                render_dynamic_step_dashboard.refresh(), 
                render_conditional_feeds.refresh(), 
                render_conditional_prompt_feed.refresh()
            )
                        
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
                        on_change=lambda e: handle_style_workflow_change(e.value)
                    ).classes('w-full').bind_value(state, 'style_selected_workflow')
                    
                    # Style presets selection dropdown
                    available_styles = load_style_presets()
                    preset_dropdown = ui.select(
                        options=available_styles,
                        label="Saved Style Preset",
                        on_change=lambda e: load_style_preset_by_name(e.value)
                    ).classes('w-full').bind_value(state, 'style_selected_preset')
                    
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
                
        with ui.tab_panel(tab_play):
            with ui.grid(columns='380px 1fr').classes('w-full gap-6 items-start'):
                # LEFT: Prompt Template Configurator and Parameters
                with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-4'):
                    ui.label('Testing Configuration').classes('text-sm font-bold text-slate-800')
                    
                    ui.select(
                        options=[b.name for b in books],
                        label="Select Book Volume",
                        value=state.playground_book_selection
                    ).bind_value(state, 'playground_book_selection').classes('w-full')

                    with ui.row().classes('w-full items-center gap-2'):
                        template_dropdown = ui.select(
                            options=available_templates,
                            label="Saved Template",
                            value=state.playground_selected_template,
                            on_change=lambda e: handle_template_dropdown_selection(e.value, prompt_editor)
                        ).classes('flex-1')
                        
                        ui.button(
                            icon="delete",
                            on_click=lambda: handle_delete_template(template_dropdown, prompt_editor)
                        ).props('flat color=red').classes('h-10').bind_visibility_from(
                            state, 'playground_selected_template', backward=lambda val: val not in ('default', '')
                        )
                    
                    with ui.row().classes('w-full items-end gap-2'):
                        custom_name_input = ui.input(placeholder="Template Name", label="Save Custom Name").classes('flex-1')
                        ui.button(
                            icon="save", 
                            on_click=lambda: handle_save_custom_template(custom_name_input.value, template_dropdown)
                        ).props('outline').classes('h-10 text-blue-600')

                    prompt_editor = ui.textarea(
                        label="Prompt Instructions (contains <text>)",
                        value=state.playground_template,
                        on_change=lambda e: setattr(state, 'playground_template', e.value)
                    ).classes('w-full h-64 font-mono text-xs leading-relaxed').props('outlined')

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

                    ui.number(
                        label="Start Chunk Index",
                        value=state.playground_start_index,
                        min=0,
                        precision=0
                    ).bind_value_to(state, 'playground_start_index').classes('w-full').bind_visibility_from(
                        state, 'playground_selection_mode', value='Static Segment'
                    )

                    ui.number(
                        label="Random Seed",
                        value=state.playground_seed,
                        precision=0
                    ).bind_value_to(state, 'playground_seed').classes('w-full').bind_visibility_from(
                        state, 'playground_selection_mode', value='Seeded Random'
                    )

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