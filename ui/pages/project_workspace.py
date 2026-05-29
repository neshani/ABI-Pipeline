import json
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
            with ui.card().classes('w-full border p-5 shadow-sm bg-white'):
                ui.label('Visual Style & Workflow Configurations').classes('text-sm font-bold text-slate-800')
                ui.label('Placeholder scaffolding for style selections and ComfyUI definitions. (Phase 2)').classes('text-xs text-slate-500')
                
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