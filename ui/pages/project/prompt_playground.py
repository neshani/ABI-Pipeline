import json
import asyncio
from pathlib import Path
from nicegui import ui
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

# Module-level references
source_text_dialog_ref = None
source_text_modal_title_label = None
source_text_modal_body_label = None
active_project_name = ""


def help_icon(title: str, description: str, additional_details: list = None):
    """Renders a styled custom help icon with a clean, dark-themed structured tooltip."""
    icon_el = ui.icon('help_outline').classes('text-slate-400 hover:text-blue-500 cursor-help ml-1.5 text-sm transition-colors')
    with icon_el:
        with ui.tooltip().classes('bg-slate-950 text-slate-200 p-4 rounded-xl border border-slate-800 max-w-sm shadow-2xl flex flex-col gap-1.5'):
            ui.label(title).classes('text-blue-400 text-[11px] font-black uppercase tracking-wider')
            ui.separator().classes('bg-slate-800/80 my-0.5')
            ui.label(description).classes('text-[11px] leading-relaxed text-slate-300 font-medium')
            if additional_details:
                ui.separator().classes('bg-slate-800/80 my-0.5')
                with ui.column().classes('gap-1 w-full'):
                    for line in additional_details:
                        ui.label(line).classes('text-[10px] text-slate-400 font-medium leading-normal')


def get_chunk_percentage(project_name: str, book_name: str, chunk: str) -> int:
    """Calculates approximate percentage depth of a chunk inside the book transcript file."""
    if not chunk or not project_name or not book_name:
        return 0
    possible_paths = [
        Path(f"./output/{project_name}/{book_name}/transcript.txt"),
        Path(f"./output/{project_name}/{book_name}_transcript.txt"),
        Path(f"./output/{project_name}/transcript.txt"),
        Path(f"./{book_name}_transcript.txt"),
        Path(f"./transcript.txt")
    ]
    for path in possible_paths:
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                pos = content.find(chunk)
                if pos != -1:
                    return int((pos / len(content)) * 100)
            except Exception:
                pass
    return 0


def open_source_text_modal(idx: int, chunk: str):
    """Binds text values to the persistent overlay structure and opens the modal."""
    global source_text_dialog_ref, source_text_modal_title_label, source_text_modal_body_label
    if not source_text_dialog_ref:
        return
    if source_text_modal_title_label:
        source_text_modal_title_label.set_text(f"Source Text Passage (Segment Chunk {idx + 1})")
    if source_text_modal_body_label:
        source_text_modal_body_label.set_text(f'"{chunk}"')
    source_text_dialog_ref.open()


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

    templates_dir = Path("./prompt_templates")
    target_file = templates_dir / f"{target_name}.txt"
    if target_file.exists():
        try:
            target_file.unlink()
            ui.notify(f"Deleted template: {target_name}", type="positive")
        except Exception as ex:
            ui.notify(f"Error deleting template file: {str(ex)}", type="negative")
            return

    all_templates = list_stored_templates()
    template_dropdown.options = all_templates
    template_dropdown.value = "default"
    template_dropdown.update()
    
    state.playground_selected_template = "default"
    loaded_default = load_template_by_name("default")
    state.playground_template = loaded_default
    prompt_editor_widget.set_value(loaded_default)


def copy_diagnostic_primer_to_clipboard():
    """Formats a diagnostic framework primer with active run results for external larger AIs and copies it to clipboard."""
    if not state.playground_results:
        ui.notify("Please run a prompt test first to populate current run data.", type="warning")
        return

    run_data_lines = []
    run_data_lines.append("### Active Prompt Template:")
    run_data_lines.append(f"```text\n{state.playground_template}\n```\n")
    
    run_data_lines.append("### Pipeline Run Parameters:")
    run_data_lines.append(f"- **Volume:** {state.playground_book_selection}")
    run_data_lines.append(f"- **Mode:** Seeded Random")
    run_data_lines.append(f"- **Seed:** {state.playground_seed}")
    run_data_lines.append(f"- **Chunk Count Tested:** {state.playground_chunk_count}\n")

    run_data_lines.append("### Segment Evaluation Output:")
    for idx, res in enumerate(state.playground_results):
        status_label = "Refusal Skipped" if res.get("status") == "refusal" else "Extraction Match"
        run_data_lines.append(f"#### Segment Chunk {idx + 1} ({status_label})")
        run_data_lines.append(f"**Source Text Passage:**\n> \"{res['chunk']}\"\n")
        run_data_lines.append(f"**Extracted Verbatim Quote:**\n> \"{res['quote']}\"\n")
        run_data_lines.append(f"**Generated Visual Prompt:**\n> {res['prompt']}\n")
        run_data_lines.append("-" * 30 + "\n")

    formatted_run_data = "\n".join(run_data_lines)

    primer_text = (
        "I am using a local AI book illustration tool called ABI-Pipeline. It extracts verbatim "
        "quotes from audiobook text transcripts and generates style-free image prompts for ComfyUI.\n\n"
        "I need your help debugging and fine-tuning my active prompt template. Below are the structural "
        "limitations and rules we MUST respect:\n\n"
        "### PIPELINE CONSTRAINTS & RULES:\n"
        "1. REGEX-PARSED OUTPUT: The small local LLM must output exactly in this format:\n"
        "   QUOTE: [verbatim quote from the passage]\n"
        "   PROMPT: [single-sentence descriptive prompt]\n"
        "   Any conversational chatter, introductory words, or departures from this format will break our regex parser.\n\n"
        "2. NO EXAMPLE CONTAMINATION: Do not include literal example QUOTES or PROMPTS in the system instructions. "
        "The local 4B model has extremely weak generalization; if we show it any concrete examples in quotes, it "
        "will overfit and try to recycle parts of those examples (or their specific subjects/structures) in every subsequent output.\n\n"
        "3. STYLE-FREE PROMPTS: The generated prompt must ONLY describe neutral, observable visual details (subject, setting, actions). "
        "Do NOT include camera terminology, medium styles (e.g., 'watercolor', 'illustration'), or lighting jargon. "
        "Artistic style is controlled downstream in ComfyUI via a separate prompt prefix/suffix system.\n\n"
        "4. STATELESS RUNS: The prompt generator runs on individual chunks of text in isolation. There is no chat history, "
        "memory, or context from previous chapters.\n\n"
        "--------------------------------------------------------------------------------\n\n"
        "### CURRENT RUN DATA:\n"
        f"{formatted_run_data}\n"
        "--------------------------------------------------------------------------------\n\n"
        "### MY SPECIFIC ADJUSTMENT REQUEST:\n"
        "[Insert your request here, e.g. 'Since we are running Harry Potter, we can tell the LLM it is a Harry Potter scene "
        "and allow it to use character names like Harry, Ron, and Hermione directly instead of describing them neutrally. "
        "Please provide an updated system prompt reflecting this change!']"
    )
    _copy_via_javascript(primer_text)
    ui.notify("Diagnostic AI Primer with run data copied!", type="positive", icon="psychology")


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
    markdown_lines.append(f"- **Mode:** Seeded Random")
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
    _copy_via_javascript(formatted_markdown)
    ui.notify("Markdown results copied to clipboard!", type="positive", icon="assignment_turned_in")


def copy_condensed_results_to_clipboard():
    """Formats only prompt/quote extracts to save LLM chat tokens, completely omitting the prompt template."""
    if not state.playground_results:
        ui.notify("No results to copy yet.", type="warning")
        return

    markdown_lines = []
    markdown_lines.append("### Pipeline Run Parameters (Condensed):")
    markdown_lines.append(f"- **Volume:** {state.playground_book_selection}")
    markdown_lines.append(f"- **Seed:** {state.playground_seed}")
    markdown_lines.append(f"- **Chunk Count Tested:** {state.playground_chunk_count}\n")

    markdown_lines.append("### Segment Evaluation (Prompt & Quote Only):")
    for idx, res in enumerate(state.playground_results):
        markdown_lines.append(f"#### Segment Chunk {idx + 1}")
        markdown_lines.append(f"- **Extracted Verbatim Quote:** \"{res['quote']}\"")
        markdown_lines.append(f"- **Generated Visual Prompt:** {res['prompt']}\n")

    formatted_markdown = "\n".join(markdown_lines)
    _copy_via_javascript(formatted_markdown)
    ui.notify("Condensed results copied to clipboard!", type="positive", icon="assignment_turned_in")


def _copy_via_javascript(text: str):
    """Utility helper triggering direct clipboard copy injection."""
    js_code = f"""
    (function() {{
        const text = {json.dumps(text)};
        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text);
        }} else {{
            const textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed";
            textArea.style.opacity = "0";
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
        }}
    }})();
    """
    ui.run_javascript(js_code)


@ui.refreshable
def render_playground_results_container():
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
        for idx, res in enumerate(state.playground_results):
            is_refusal = res.get("status") == "refusal"
            border_color = "border-red-200 bg-red-50/20" if is_refusal else "border-slate-200 bg-white"
            badge_color = "bg-red-50 text-red-700 border border-red-200" if is_refusal else "bg-blue-50 text-blue-700 border border-blue-200"
            badge_label = "Refusal Skipped" if is_refusal else "Extraction Match"

            with ui.card().classes(f'w-full border p-4 rounded-xl shadow-xs gap-3 cursor-pointer hover:bg-slate-50/60 transition-colors {border_color}') \
                    .on('click', lambda _, i=idx, chunk=res["chunk"]: open_source_text_modal(i, chunk)) \
                    .tooltip("Click to view full Source Text Passage"):
                
                with ui.row().classes('w-full justify-between items-center pb-2 border-b border-dashed'):
                    pct = get_chunk_percentage(active_project_name, state.playground_book_selection, res["chunk"])
                    ui.label(f"Segment Chunk {idx + 1} • Excerpt at {pct}%").classes('text-xs font-bold text-slate-600 uppercase')
                    ui.label(badge_label).classes(f'px-2 py-0.5 rounded text-[10px] font-bold {badge_color}')

                with ui.column().classes('w-full gap-3'):
                    with ui.column().classes('gap-0.5 w-full'):
                        ui.label("Extracted Verbatim Quote:").classes('text-[9px] font-black text-slate-400 uppercase tracking-wide')
                        ui.label(f'"{res["quote"]}"').classes('text-xs font-semibold text-slate-700 italic leading-relaxed')
                    
                    with ui.column().classes('gap-0.5 w-full'):
                        ui.label("Generated Visual Prompt:").classes('text-[9px] font-black text-slate-400 uppercase tracking-wide')
                        ui.label(res["prompt"]).classes('text-xs font-semibold text-blue-700 leading-relaxed')

async def execute_playground_test(project_name: str):
    global active_project_name
    active_project_name = project_name

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
        mode="Seeded Random",
        start_index=0,
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


def render_prompt_playground_tab(project, books):
    global source_text_dialog_ref, source_text_modal_title_label, source_text_modal_body_label, active_project_name
    active_project_name = project.name
    
    available_templates = list_stored_templates()
    
    with ui.grid(columns='1.2fr 1fr').classes('w-full gap-6 items-start'):
        # LEFT COLUMN (Prompt Architecture & Parameters)
        with ui.column().classes('w-full gap-4'):
            
            # CARD 1: RUN PARAMETERS (Top Aligned)
            with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-4'):
                with ui.row().classes('w-full items-center justify-between'):
                    with ui.row().classes('items-center gap-1'):
                        ui.label('Run Parameters').classes('text-sm font-bold text-slate-800')
                        help_icon(
                            title="Iterative Testing Workflow",
                            description="ABI-Pipeline is built for rapid, real-time prompt engineering directly in your browser.",
                            additional_details=[
                                "• Sandbox Mode: You can type and modify prompt text and test them immediately without saving first.",
                                "• Interactive Tuning: Tweak instructions, run a test, share results with an LLM chat via the AI Toolkit, and refine.",
                                "• Save to Commit: Once the output is optimized, type a Custom Name below and click Save to lock in your template before switching tabs!"
                            ]
                        )

                ui.select(
                    options=[b.name for b in books],
                    label="Select Book Volume",
                    value=state.playground_book_selection
                ).bind_value(state, 'playground_book_selection').classes('w-full')

                with ui.row().classes('w-full gap-4'):
                    ui.number(
                        label="Chunk Count", 
                        value=state.playground_chunk_count,
                        min=1, max=10
                    ).bind_value_to(state, 'playground_chunk_count').classes('flex-1')

                    ui.number(
                        label="Random Seed",
                        value=state.playground_seed,
                        precision=0
                    ).bind_value_to(state, 'playground_seed').classes('flex-1')

                ui.button(
                    'Test Prompt Template', 
                    icon='bolt', 
                    color='positive',
                    on_click=lambda: execute_playground_test(project.name)
                ).classes('w-full text-white font-bold text-xs h-10 shadow-sm')

            # CARD 2: PROMPT ARCHITECTURE
            with ui.card().classes('w-full border p-5 shadow-sm bg-white gap-4'):
                ui.label('Prompt Architecture').classes('text-sm font-bold text-slate-800')
                
                # Dropdown Selector
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
                
                # Custom Save Name
                with ui.row().classes('w-full items-end gap-2'):
                    custom_name_input = ui.input(placeholder="Template Name", label="Save Custom Name").classes('flex-1')
                    ui.button(
                        icon="save", 
                        on_click=lambda: handle_save_custom_template(custom_name_input.value, template_dropdown)
                    ).props('outline').classes('h-10 text-blue-600')

                # Flat, borderless autogrow text editor wrapper
                with ui.column().classes('w-full gap-1'):
                    ui.label("Prompt Instructions (<text> tag is required and will be replaced by book chunks)")\
                      .classes('text-[10px] font-bold text-slate-500 tracking-wide')
                    
                    prompt_editor = ui.textarea(
                        value=state.playground_template,
                        on_change=lambda e: setattr(state, 'playground_template', e.value)
                    ).classes('w-full font-mono text-xs leading-relaxed bg-slate-50 rounded-lg p-3 border border-slate-200')\
                     .props('autogrow borderless shadow-none')

        # RIGHT COLUMN (AI Toolkit & Results)
        with ui.column().classes('w-full gap-4'):
            # CARD 3: AI TOOLKIT
            with ui.card().classes('w-full border p-4 shadow-sm bg-white gap-3'):
                with ui.row().classes('w-full items-center justify-between'):
                    with ui.row().classes('items-center gap-1.5'):
                        ui.icon('smart_toy', color='blue', size='xs')
                        ui.label('AI Toolkit').classes('text-[10px] font-black text-slate-500 uppercase tracking-wide')
                    help_icon(
                        title="AI Companion Toolkit",
                        description="Bridge active visual metadata to external conversational AI models.",
                        additional_details=[
                            "• Copy AI Primer: Generates a complete system prompt with active scene data and parser safety rules for ChatGPT/Claude/Ollama.",
                            "• Copy Full: Copies active prompt guidelines alongside complete evaluation outputs.",
                            "• Copy Condensed: Excludes prompt instructions, copying only quote and prompt tuples."
                        ]
                    )
                
                with ui.column().classes('w-full gap-2'):
                    ui.button(
                        "Copy AI Primer", 
                        icon="psychology", 
                        on_click=copy_diagnostic_primer_to_clipboard
                    ).classes('w-full bg-slate-700 hover:bg-slate-800 text-white text-xs font-semibold h-9 shadow-sm')
                    
                    ui.button(
                        "Copy Full", 
                        icon="content_copy", 
                        on_click=copy_results_to_clipboard
                    ).classes('w-full bg-slate-700 hover:bg-slate-800 text-white text-xs font-semibold h-9 shadow-sm')
                    
                    ui.button(
                        "Copy Condensed", 
                        icon="compress", 
                        on_click=copy_condensed_results_to_clipboard
                    ).classes('w-full bg-blue-700 hover:bg-blue-800 text-white text-xs font-semibold h-9 shadow-sm')

            # Rendered results output feed
            render_playground_results_container()

    # Declare Dialog Overlays inside workspace hierarchy
    with ui.dialog() as source_text_dialog:
        with ui.card().classes('w-full max-w-2xl p-5 rounded-xl gap-4 bg-white'):
            with ui.row().classes('w-full justify-between items-center border-b pb-2'):
                source_text_modal_title_label = ui.label("Source Text Passage").classes('text-base font-bold text-slate-800')
                ui.button(icon='close', on_click=source_text_dialog.close).props('flat round dense').classes('text-slate-400')
            
            with ui.column().classes('w-full bg-slate-50 p-4 rounded-lg border text-xs leading-relaxed max-h-[50vh] overflow-y-auto'):
                source_text_modal_body_label = ui.label("").classes('italic text-slate-700 font-medium')
            
            with ui.row().classes('w-full justify-end border-t pt-2'):
                ui.button('Close', on_click=source_text_dialog.close).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs')

    # Bind elements to global scope reference variables
    source_text_dialog_ref = source_text_dialog