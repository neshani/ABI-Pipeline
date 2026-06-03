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

    all_templates = list_stored_templates()
    template_dropdown.options = all_templates
    template_dropdown.value = "default"
    template_dropdown.update()
    
    state.playground_selected_template = "default"
    loaded_default = load_template_by_name("default")
    state.playground_template = loaded_default
    prompt_editor_widget.set_value(loaded_default)

def copy_diagnostic_primer_to_clipboard():
    """Formats a diagnostic framework primer for external larger AIs and copies it to clipboard."""
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
        "[PASTE YOUR COPIED RUN RESULTS HERE]\n\n"
        "--------------------------------------------------------------------------------\n\n"
        "### MY SPECIFIC ADJUSTMENT REQUEST:\n"
        "[Insert your request here, e.g. 'Since we are running Harry Potter, we can tell the LLM it is a Harry Potter scene "
        "and allow it to use character names like Harry, Ron, and Hermione directly instead of describing them neutrally. "
        "Please provide an updated system prompt reflecting this change!']"
    )
    _copy_via_javascript(primer_text)
    ui.notify("Diagnostic AI Primer copied! Paste this first, then your results.", type="positive", icon="psychology")


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
        with ui.row().classes('w-full justify-between items-center bg-slate-100 p-3 rounded-lg border'):
            with ui.column().classes('gap-0'):
                ui.label("Evaluation Iteration Ready").classes('text-xs font-bold text-slate-700')
                ui.label("Format optimized for sharing with diagnostic AIs").classes('text-[10px] text-slate-500')
            with ui.row().classes('gap-2'):
                ui.button("Copy AI Primer", icon="psychology", on_click=copy_diagnostic_primer_to_clipboard).classes('bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-semibold px-3')
                ui.button("Copy Full", icon="content_copy", on_click=copy_results_to_clipboard).classes('bg-slate-700 hover:bg-slate-800 text-white text-xs font-semibold px-3')
                ui.button("Copy Condensed", icon="compress", on_click=copy_condensed_results_to_clipboard).classes('bg-blue-700 hover:bg-blue-800 text-white text-xs font-semibold px-3')

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
                    with ui.column().classes('gap-1 bg-blue-50/30 p-3 rounded-lg border border-blue-50/50'):
                        ui.label("Source Text Passage:").classes('text-[10px] font-black text-slate-400 uppercase')
                        ui.label(f'"{res["chunk"][:320]}..."').classes('text-xs text-slate-600 italic leading-relaxed')

                    with ui.column().classes('gap-2 p-3 bg-emerald-50/20 rounded-lg border border-emerald-50/50'):
                        with ui.column().classes('gap-0.5'):
                            ui.label("Extracted Verbatim Quote:").classes('text-[10px] font-black text-slate-400 uppercase')
                            ui.label(f'"{res["quote"]}"').classes('text-xs font-semibold text-slate-700')
                        with ui.column().classes('gap-0.5 mt-2'):
                            ui.label("Generated Visual Prompt:").classes('text-[10px] font-black text-slate-400 uppercase')
                            ui.label(res["prompt"]).classes('text-xs font-semibold text-blue-700 leading-relaxed')


async def execute_playground_test(project_name: str):
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


def render_prompt_playground_tab(project, books):
    available_templates = list_stored_templates()
    
    with ui.grid(columns='380px 1fr').classes('w-full gap-6 items-start'):
        # LEFT CONFIG PANEL
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
                    min=1, max=5
                ).bind_value_to(state, 'playground_chunk_count').classes('w-20')

                ui.select(
                    options=["Seeded Random", "Static Segment"],
                    label="Selection Mode",
                    value=state.playground_selection_mode
                ).bind_value_to(state, 'playground_selection_mode').classes('flex-1')

            ui.number(
                label="Start Chunk Index",
                value=state.playground_start_index,
                min=0, precision=0
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