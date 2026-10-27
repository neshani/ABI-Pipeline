import csv
import httpx
import re
import os
import random
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from database.connection import get_setting
from services import vram_manager
from services import compat
from services import scene_map
# The linear chunker and the sentence machinery live in ``scene_map`` - the
# module kept its name when the map stopped being a pipeline stage. They are
# re-exported here because prompt generation, the shot planner and the character
# manager have always imported them from this module.
from services.scene_map import (
    PARA_SLACK,
    SENT_SLACK,
    StaleSceneMap,
    book_chunks,
    smart_chunk_text,
    transcript_digest,
)
from services import pipeline_manifest
from ui import state

REFUSAL_KEYWORDS = [
    "i cannot", "i can't", "i apologize", "i'm sorry", "im sorry", 
    "unable to generate", "policy", "forbidden", "restricted", 
    "cannot fulfill", "as an ai", "language model"
]

# Answers that mean "no image" instead of describing one. The INSERT fragment is
# allowed to decline when a passage holds nothing worth showing close, and a model
# that ignores the OUTPUT FORMAT echoes that refusal into the PROMPT line. The
# sentinel has to be recognised in one place: the writer, the renderer, the
# progress counters and the workspace all have to agree, or a declined row is
# stored as if it were a prompt and then dropped silently further down - which is
# how a book renders with a hole in it and still reports 100%.
UNUSABLE_PROMPTS = {"", "none", "refusal", "n/a", "na", "nil", "-", "--", "..."}

# Refusal words are only meaningful in a short reply. A real prompt can legitimately
# contain "restricted" or "forbidden" ("a forbidden shrine"), and it is 20+ words long.
_REFUSAL_MAX_WORDS = 12


def is_unusable_prompt(text: Optional[str]) -> bool:
    """True when ``text`` is empty, a ``NONE``/``REFUSAL`` sentinel, or a refusal echo.

    The single definition of "this row cannot be rendered". Callers that decide
    whether a prompt is worth storing or drawing must use it rather than comparing
    against their own list of sentinels.
    """
    cleaned = (text or "").strip().strip('"\'*_`').strip().lower()
    cleaned = cleaned.strip(".:;!?()[] ")
    if not cleaned or cleaned in UNUSABLE_PROMPTS:
        return True
    if len(cleaned.split()) <= _REFUSAL_MAX_WORDS:
        return any(keyword in cleaned for keyword in REFUSAL_KEYWORDS)
    return False

def ensure_templates_directory() -> Path:
    """Ensures prompt_templates folder exists and creates a default.txt file if empty."""
    templates_dir = Path("./prompt_templates")
    templates_dir.mkdir(exist_ok=True)
    default_file = templates_dir / "default.txt"
    if not default_file.exists():
        default_content = (
            "You are a specialized AI function that performs two tasks:\n"
            "1.  Select a visually rich, verbatim quote from a passage of text.\n"
            "2.  Create a single-sentence image prompt inspired by that quote for a text-to-image AI like SDXL.\n\n"
            "### INSTRUCTIONS ###\n"
            "1.  **Quote Selection:** From the PASSAGE, find the single most visually descriptive sentence or short phrase. "
            "This quote will be used to inspire the image. It MUST be an exact quote.\n"
            "2.  **Image Prompt Generation:**\n"
            "    *   The image prompt MUST be directly inspired by the quote you selected.\n"
            "    *   **Character Descriptions:** Instead of using a character's proper name, describe their visual appearance or "
            "role in the scene. Focus on neutral, observable details that a camera would see.\n"
            "    *   The prompt MUST be a single, descriptive sentence.\n"
            "    *   Structure the sentence like this: [Subject], [Setting], [Visual Details].\n"
            "3.  **Output Format:** Your entire response MUST follow this exact format. Do NOT add conversational text.\n\n"
            "QUOTE: [The verbatim quote you selected from the passage]\n"
            "PROMPT: [The single-sentence image prompt you generated]\n\n"
            "### TASK ###\n"
            "PASSAGE:\n"
            "<text>\n\n"
            "REMEMBER: YOUR ENTIRE RESPONSE MUST FOLLOW THE EXACT `QUOTE: ... PROMPT: ...` FORMAT."
        )
        default_file.write_text(default_content, encoding="utf-8")
    return templates_dir


def list_stored_templates() -> List[str]:
    """Lists all available prompt template files (.txt) in the templates directory."""
    templates_dir = ensure_templates_directory()
    return sorted([f.stem for f in templates_dir.glob("*.txt")])


def load_template_by_name(name: str) -> str:
    """Loads prompt template text by filename stem."""
    templates_dir = ensure_templates_directory()
    target = templates_dir / f"{name}.txt"
    if target.exists():
        return target.read_text(encoding="utf-8")
    return ""


def save_template_by_name(name: str, content: str) -> None:
    """Saves/stores a custom named template into prompt_templates."""
    templates_dir = ensure_templates_directory()
    target = templates_dir / f"{name}.txt"
    target.write_text(content, encoding="utf-8")


async def get_llm_response(
    prompt: str,
    llm_url: str,
    model_name: str,
    temperature: float = 0.7,
    extra_payload: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    with_meta: bool = False,
    with_message: bool = False,
    messages: Optional[List[Dict[str, Any]]] = None,
):
    """Dispatches asynchronous request to local Ollama or LM Studio OpenAI-compatible endpoints.

    ``extra_payload`` merges into the request body, e.g. to disable a model's
    thinking mode: ``{"chat_template_kwargs": {"enable_thinking": False}}``.
    ``timeout`` is per request: scene segmentation reads a whole chapter and can
    take minutes, while a one-shot prompt is expected in under a minute.
    ``with_meta`` returns ``(content, finish_reason)`` instead of the bare string.
    ``with_message`` returns ``(message, finish_reason)`` - the whole message dict,
    which is the only way to see ``tool_calls`` in a tool-calling loop.
    ``messages`` replaces the single-turn ``prompt`` with a whole conversation,
    which a tool loop needs; ``prompt`` is then ignored.
    ``finish_reason == "length"`` is the only signal that the model was cut off at
    ``max_tokens``; callers that parse structured output need it to tell "nothing
    to report" from "the rest of the reply was never written".
    """
    # Loading the LLM will evict ComfyUI if it is the one holding the GPU.
    await vram_manager.ensure_llm_async()

    endpoint = llm_url.rstrip("/")
    if "/v1" not in endpoint and "/api" not in endpoint:
        endpoint = f"{endpoint}/v1/chat/completions"
    elif endpoint.endswith("/v1"):
        endpoint = f"{endpoint}/chat/completions"

    # Retrieve and apply API authorization header if available
    api_key = get_setting("llm_api_key", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": messages or [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    if extra_payload:
        payload.update(extra_payload)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(endpoint, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if with_message:
                return choice["message"], choice.get("finish_reason") or ""
            if with_meta:
                return content, choice.get("finish_reason") or ""
            return content
        except Exception as e:
            # Fallback to direct Ollama generate endpoint if connection fails and endpoint targets port 11434
            if "11434" in endpoint or "ollama" in endpoint.lower():
                try:
                    direct_endpoint = f"{llm_url.rstrip('/')}/api/generate"
                    direct_payload = {
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature}
                    }
                    resp = await client.post(direct_endpoint, json=direct_payload, timeout=timeout)
                    resp.raise_for_status()
                    text = resp.json().get("response", "")
                    if with_message:
                        return {"role": "assistant", "content": text}, ""
                    return (text, "") if with_meta else text
                except Exception as inner_e:
                    msg = f"API Connection Failed: {e} (Direct API Fallback failed: {inner_e})"
                    if with_message:
                        return {"role": "assistant", "content": msg}, "error"
                    return (msg, "error") if with_meta else msg
            msg = f"API Connection Failed: {e}"
            if with_message:
                return {"role": "assistant", "content": msg}, "error"
            return (msg, "error") if with_meta else msg


def parse_llm_response(raw_response: str) -> Dict[str, str]:
    """Strips and extracts structured output matching QUOTE: ... PROMPT: ... with fallbacks."""
    parsed_prompt = ""
    parsed_quote = ""
    current_capture = None

    clean_raw = raw_response.replace('**', '').replace('__', '')

    for line in clean_raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        
        lower_line = line.lower()
        
        if lower_line.startswith("quote:") or lower_line.startswith("- quote:"):
            current_capture = "quote"
            parts = line.split(":", 1)
            if len(parts) > 1:
                parsed_quote += parts[1].strip() + " "
        elif lower_line.startswith("prompt:") or lower_line.startswith("- prompt:"):
            current_capture = "prompt"
            parts = line.split(":", 1)
            if len(parts) > 1:
                parsed_prompt += parts[1].strip() + " "
        elif current_capture == "quote":
            parsed_quote += line + " "
        elif current_capture == "prompt":
            parsed_prompt += line + " "

    parsed_quote = parsed_quote.strip().strip('"')
    parsed_prompt = parsed_prompt.strip().strip('"')

    # Apply extreme fallbacks
    if not parsed_prompt and not parsed_quote:
        clean_txt = clean_raw.strip()
        clean_txt = re.sub(r'^```[\w]*\n', '', clean_txt)
        clean_txt = re.sub(r'\n```$', '', clean_txt).strip('"').strip()
        parsed_prompt = clean_txt
        parsed_quote = " "
    elif not parsed_prompt:
        parsed_prompt = parsed_quote
    elif not parsed_quote:
        parsed_quote = " "

    raw_lower = raw_response.lower()
    has_refusal_words = any(keyword in raw_lower for keyword in REFUSAL_KEYWORDS)

    # A refusal is the stronger evidence: a short reply that says "I can't" is a
    # refusal whether or not it lands in the PROMPT line. A decline is the lawful
    # ``QUOTE: NONE`` / ``PROMPT: NONE`` answer the INSERT fragment is allowed to
    # give. Either way the row is unusable - the split only decides what the
    # console and ``refusals.log`` call it.
    short_reply = len(parsed_prompt.split()) <= _REFUSAL_MAX_WORDS
    if has_refusal_words and short_reply:
        status = "refusal"
    elif is_unusable_prompt(parsed_prompt):
        status = "declined"
    else:
        status = "success"

    return {
        "raw": raw_response,
        "quote": parsed_quote,
        "prompt": parsed_prompt,
        "status": status
    }


def _log_unusable(res: Dict[str, Any], chunk: str, log_path: Path) -> None:
    """Replace a declined or refused result with the ``REFUSAL`` sentinel.

    ``NONE`` is meaningful to the prompt model and meaningless to everything
    downstream, so it is normalised to the sentinel the renderer, the progress
    counters and the workspace already recognise. The raw reply is kept in
    ``refusals.log``. The quote is the passage itself, because that is what the
    row is *about* - it still places on the timeline even though it draws nothing.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tried = " -> ".join([*(res.get("declined_roles") or []), "nothing"])
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"Chapter {res['chapter']}, Scene {res['shot']} "
                     f"({tried}):\n{res.get('raw', '')}\n\n{'-' * 40}\n\n")
    res["prompt"] = "REFUSAL"
    res["quote"] = chunk


def fetch_test_chunks(
    project_name: str, 
    book_name: str, 
    count: int, 
    mode: str, 
    start_index: int = 15, 
    seed: int = 42
) -> List[str]:
    """Reads transcript.txt from the specified book folder and returns selected test chunks based on mode, start index, and seed."""
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    transcript_path = base_output_dir / project_name / book_name / "transcript.txt"
    if not transcript_path.exists():
        return []

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunk_size = getattr(state, "playground_chunk_size", 350)
    if not chunk_size or chunk_size <= 0:
        chunk_size = 350

    # The real chunker, on the real text: a style sample should be a passage the
    # pipeline would actually turn into one image, at the same size.
    all_chunks = [chunk.text for chunk in scene_map.linear_chunks(content, chunk_size)]
    if not all_chunks:
        return []

    # Safeguard NiceGUI bound floats to standard integers
    count = int(count)
    if count <= 0:
        count = 1

    if mode == "Static Segment":
        start_idx = max(0, int(start_index))
        # Clamp to avoid going out of bounds
        if start_idx >= len(all_chunks):
            start_idx = max(0, len(all_chunks) - count)
        return all_chunks[start_idx : start_idx + count]
    else:
        # Seeded Random selection for deterministic, reusable random samples
        r = random.Random(int(seed))
        return r.sample(all_chunks, min(count, len(all_chunks)))


async def generate_prompt_for_chunk_async(
    task: dict, 
    llm_url: str, 
    model_name: str, 
    semaphore: asyncio.Semaphore,
    extra_payload: Optional[Dict[str, Any]] = None,
    simulate_decline: bool = False,
) -> dict:
    """Generates prompt for a single chunk using a concurrency semaphore.

    ``task["prompt"]`` is already fully composed (shared block plus the
    shot-role fragment chosen for this scene) by the caller.

    A role may lawfully decline: the INSERT fragment answers ``NONE`` when the
    passage holds nothing worth showing close. Declining is not an error, it is
    the planner having picked a shot type the passage cannot fill, so the same
    passage is re-asked under the next role in ``prompt_library.FALLBACK_ROLES``
    (INSERT -> WIDE -> SINGLE) before anything is written. The role that actually
    produced the text comes back as ``shot_role`` so the caller records it.

    ``extra_payload`` is forwarded to the request body, so a caller that wants a
    model's thinking mode off (the Prompt Lab does) still gets the same fallback
    walk the pipeline uses. ``simulate_decline`` forces the first role's answer to
    be the ``NONE`` sentinel; it exists so the Prompt Lab can demonstrate the walk
    on demand, and nothing in the pipeline passes it.
    """
    async with semaphore:
        # Local import: character_manager imports this module.
        from services.character_manager import auto_bracket_character_names
        from services import prompt_library

        original_role = task.get("shot_role") or "single"
        chain = [original_role, *prompt_library.fallback_roles(original_role)]
        role = original_role
        parsed = None
        declined: List[str] = []
        for index, role in enumerate(chain):
            if index == 0:
                final_prompt = task["prompt"]
            else:
                # Rebuild from the same passage and the same plan, with a role the
                # passage can satisfy. The shared block is identical, so this is one
                # extra call, not a re-plan.
                final_prompt = prompt_library.render_with_selection(
                    role,
                    task.get("chunk", ""),
                    task.get("selection") or {},
                    focus=task.get("focus", ""),
                    place=task.get("place", ""),
                )
            if simulate_decline and index == 0:
                # Prompt Lab only: a decline is not reliably reproducible on demand
                # - the INSERT fragment refuses by judgement, not by rule - so the
                # test bench can force the first answer to be the sentinel and watch
                # the real fallback walk from there.
                raw_response = "QUOTE: NONE\nPROMPT: NONE"
            else:
                raw_response = await get_llm_response(
                    final_prompt, llm_url, model_name, extra_payload=extra_payload)
            parsed = parse_llm_response(raw_response)
            if parsed["status"] == "success":
                break
            declined.append(role)
            nxt = chain[index + 1] if index + 1 < len(chain) else None
            state.add_console_log(
                f"[Prompt-Gen] Ch {task['chapter_num']} shot {task['shot_num']}: "
                f"{role} {parsed['status']} the passage -> "
                + (f"retrying as {nxt}" if nxt else "no role left to try")
            )

        used_role = role if parsed and parsed["status"] == "success" else original_role
        return {
            "chapter": task["chapter_num"],
            "shot": task["shot_num"],
            "quote": parsed["quote"],
            "prompt": auto_bracket_character_names(
                parsed["prompt"], task.get("project_id") or 0),
            "status": parsed["status"],
            "shot_role": used_role,
            # Every role that came back unusable, in order. On a fallback that
            # succeeded this is the roles it stepped over; on total failure it is
            # the whole chain, so ``refusals.log`` names what was tried.
            "declined_roles": declined,
            "raw": parsed["raw"]
        }


def cancel_prompt_generation(project_id: int):
    """Flags active prompt generation to cancel."""
    state.cancel_prompt_gen_flag = True
    state.add_console_log(f"[Prompt-Gen] Cancellation signal sent to Project ID {project_id}...")


def prompt_gen_stopped() -> bool:
    """True when the running prompt stage has been asked to stop.

    Both latches count: ``cancel_prompt_gen_flag`` is this stage's own and
    ``cancel_pipeline_flag`` is the dashboard run's, which is set for the whole
    run - shot planning included. Reading only the prompt flag is what let a
    stopped run resume: the stage that owns the flag resets it when it starts,
    so a stop raised during shot planning (or just before the phase began) was
    cleared by the next entry into the prompt stage.
    """
    return bool(getattr(state, "cancel_prompt_gen_flag", False)
                or getattr(state, "cancel_pipeline_flag", False))


def planner_progress_log(book_name: str):
    """The planner's console log, plus the Run bar's one-line summary.

    The Shot Planner already says where it is, per chapter and per window, so
    the summary is read off those same lines instead of threading a second
    callback through three functions. Two string comparisons per line is the
    whole cost.
    """
    chapter = ""

    def log(message: str) -> None:
        nonlocal chapter
        state.add_console_log(message)
        if message.startswith("[Shot-Plan] Chapter "):
            # "[Shot-Plan] Chapter 3: 12 scenes"
            chapter = message.split("Chapter ", 1)[1].split(":", 1)[0].strip()
            state.pipeline_progress_line = (
                f"Shot planning: Chapter {chapter} - {book_name}")
        elif message.startswith("[Shot-Plan] classifying scenes "):
            span = message.split("classifying scenes ", 1)[1].strip()
            where = f"Chapter {chapter}, " if chapter else ""
            state.pipeline_progress_line = (
                f"Shot planning: {where}scenes {span} - {book_name}")

    return log


def ensure_shot_role_column(csv_path: Path) -> None:
    """Guarantee ``shot_role`` exists and sits directly after ``quote``.

    Books whose prompts.csv predates shot roles get the column added, blank on
    the old rows - which resolve to the Character base style, i.e. exactly how
    they were originally rendered.

    The position matters: the append paths write
    ``chapter, shot, prompt, quote, shot_role`` positionally, so a header that
    carries shot_role anywhere else (``sync_book_timing`` moves unknown columns
    to the end, for instance) would misalign every subsequently appended row.
    This rewrites the file so the column always lands after ``quote``.
    """
    if not csv_path.exists():
        return
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f, delimiter="|"))
    except Exception:
        return
    if not rows:
        return

    header = compat.normalize_fieldnames(rows[0])
    current = header.index("shot_role") if "shot_role" in header else None

    if current is None:
        target = (header.index("quote") + 1) if "quote" in header else len(header)
    else:
        without = [h for i, h in enumerate(header) if i != current]
        target = (without.index("quote") + 1) if "quote" in without else len(without)
        if current == target:
            # Already in place: nothing to do.
            return

    width = len(header)
    without_header = [h for i, h in enumerate(header) if i != current]
    new_header = without_header[:target] + ["shot_role"] + without_header[target:]

    out_rows = [new_header]
    for row in rows[1:]:
        padded = list(row) + [""] * (width - len(row))
        role = padded[current] if current is not None else ""
        without = [v for i, v in enumerate(padded) if i != current]
        out_rows.append(without[:target] + [role] + without[target:])

    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f, delimiter="|").writerows(out_rows)
    except Exception:
        pass


def clear_prompt_approvals(csv_path: Path) -> int:
    """Set ``approved`` to False on every row. Returns how many were cleared.

    Used when the images those approvals referred to have been archived: a
    regenerated image has not been looked at, so it must not arrive already
    ticked. That is the same rule the workspace applies when one image is
    deleted - the difference is that a rollback deletes all of them at once.

    The file is rewritten in place, keeping its own header order: the append
    paths downstream write rows positionally, so the column layout must not move.
    """
    if not csv_path.exists():
        return 0
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f, delimiter="|"))
    except Exception:
        return 0
    if not rows:
        return 0

    header = compat.normalize_fieldnames(rows[0])
    if "approved" not in header:
        return 0  # legacy file with no such column: nothing was ever approved

    column = header.index("approved")
    cleared = 0
    out_rows = [rows[0]]
    for row in rows[1:]:
        row = list(row)
        if len(row) > column and (row[column] or "").strip().lower() == "true":
            row[column] = "False"
            cleared += 1
        out_rows.append(row)

    if cleared:
        try:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f, delimiter="|").writerows(out_rows)
        except Exception:
            return 0
    return cleared


def sort_csv_by_chapter_scene(csv_path: Path):
    """Sorts the final prompts.csv file numerically by chapter and shot."""
    if not csv_path.exists():
        return
    try:
        import csv
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            header = next(reader)
            for row in reader:
                if len(row) >= 2:
                    rows.append(row)
        
        # Sort rows numerically by chapter (index 0) and shot (index 1)
        rows.sort(key=lambda r: (int(r[0]), int(r[1])))
        
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(header)
            writer.writerows(rows)
    except Exception as ex:
        print(f"Error sorting prompts.csv: {str(ex)}")


def prepare_book_chunks(book_dir, chunk_words: int, full_text: str,
                        completed_scenes: set):
    """``(chunks, source, problem)`` for one book, enforcing the coordinate guard.

    The book is chunked by the linear packer, at the version it is bound to: v2
    for a book with no prompt rows yet, v1 for one that was numbered by the older
    greedy packer. A ``prompts.csv`` built under a different chunking or packer
    comes back as ``problem`` instead of raising or silently switching coordinate
    systems; the caller logs it and skips the book. On success the pipeline
    manifest records which chunking and packer own the book from now on.
    """
    prompts_exist = bool(completed_scenes) or scene_map.prompts_have_rows(book_dir)
    version = scene_map.chunker_version(book_dir, prompts_exist=prompts_exist)
    try:
        chunks, source = book_chunks(book_dir, chunk_words, text=full_text,
                                     source="linear", version=version)
    except StaleSceneMap as ex:
        return None, "", str(ex)

    sha = transcript_digest(full_text)
    allowed, reason = pipeline_manifest.check(
        book_dir, source, chunk_words, transcript_sha=sha,
        prompts_exist=prompts_exist, chunker_version=version)
    if not allowed:
        return None, source, reason
    pipeline_manifest.record_chunking(
        book_dir, source, chunk_words, transcript_sha=sha,
        chunker_version=version)
    return chunks, source, ""


async def start_project_prompt_gen(project_id: int,
                                   book_ids: Optional[List[int]] = None):
    """
    Orchestrates sequential resumable prompt generation for all books in a project.

    ``book_ids`` narrows the run to those volumes; ``None`` means the whole
    project. The dashboard's per-volume Run needs this, and so does a single
    stale volume in an otherwise finished series: the alternative was re-running
    fifteen books to regenerate one.

    Processes chapters and chunks in parallel using asyncio Semaphores.
    """
    import csv
    import asyncio
    from sqlmodel import Session, select
    from database.models import Project, Book
    from database.connection import engine, get_setting
    from services import checkpoints

    if state.prompt_gen_active:
        state.add_console_log("[Prompt-Gen] Warning: A prompt generation task is already active.")
        return

    # The stop latches are cleared by whoever *starts* the run (the dashboard's
    # Run bar, or the prompt-gen button), never here: a Stop pressed while the
    # previous stage was still unwinding used to be wiped by this line, and the
    # run carried on into prompt generation as if it had never been asked to stop.
    if prompt_gen_stopped():
        state.add_console_log(
            "[Prompt-Gen] A stop is already pending - the run will not start.")
        return

    state.prompt_gen_active = True
    state.recent_prompts.clear()
    skipped_books = 0
    completed_books = 0
    
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            state.add_console_log("[Prompt-Gen] Error: Project not found.")
            state.prompt_gen_active = False
            return
            
        project.status = "Generating Prompts"
        session.add(project)
        
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        if book_ids is not None:
            wanted = set(book_ids)
            books = [b for b in books if b.id in wanted]
        
        # Extract properties to standard types before commit to avoid DetachedInstanceError
        project_name = project.name
        books_data = []
        for b in books:
            b.status = "Generating Prompts"
            session.add(b)
            books_data.append({
                "id": b.id,
                "name": b.name
            })
        session.commit()
        
    state.add_console_log(f"[Prompt-Gen] Starting generation for Project: {project_name}")

    try:
        # Load LLM connection configurations
        llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
        from services import llm_config
        model_name = llm_config.stage_model("prompt")
        
        # Load dynamic chunk size setting
        chunk_size_words = getattr(state, "playground_chunk_size", 350)
        if not chunk_size_words or chunk_size_words <= 0:
            chunk_size_words = 350

        base_output_dir = Path(get_setting("output_dir", "./output")).resolve()

        for book_dict in books_data:
            book_id = book_dict["id"]
            book_name = book_dict["name"]

            if prompt_gen_stopped():
                state.add_console_log("[Prompt-Gen] Process interrupted by user.")
                break

            state.add_console_log(f"[Prompt-Gen] Processing Book: {book_name}")
            
            transcript_path = base_output_dir / project_name / book_name / "transcript.txt"
            output_csv_path = base_output_dir / project_name / book_name / "prompts.csv"

            if not transcript_path.exists():
                state.add_console_log(f"[Prompt-Gen] Missing transcript for {book_name}. Skipping.")
                skipped_books += 1
                continue

            # 1. Load Existing Progress for Resuming
            completed_scenes = set()
            if output_csv_path.exists():
                try:
                    with open(output_csv_path, "r", encoding="utf-8") as csv_f:
                        reader = csv.DictReader(csv_f, delimiter="|")
                        reader.fieldnames = compat.normalize_fieldnames(reader.fieldnames)
                        for row in reader:
                            if "chapter" in row and "shot" in row:
                                completed_scenes.add((int(row["chapter"]), int(row["shot"])))
                except Exception as ex:
                    state.add_console_log(f"[Prompt-Gen] Warning reading existing prompts.csv: {str(ex)}")

            if completed_scenes:
                state.add_console_log(f"[Prompt-Gen] Found {len(completed_scenes)} existing prompts. Resuming where left off...")

            with open(transcript_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            # 2. Chunk the book. A scene map owns the coordinates when the book has
            # one; otherwise the linear chunker reproduces the legacy numbering.
            book_dir = output_csv_path.parent
            chunks, chunk_source, problem = prepare_book_chunks(
                book_dir, chunk_size_words, full_text, completed_scenes)
            if problem:
                state.add_console_log(
                    f"[Prompt-Gen] {book_name}: {problem} Skipping this book.")
                skipped_books += 1
                # Do not leave a skipped book stuck in "Generating Prompts" for the
                # finally block to downgrade: rows that exist are still renderable.
                with Session(engine) as session:
                    db_book = session.get(Book, book_id)
                    if db_book:
                        db_book.status = ("Prompts Created" if completed_scenes
                                          else "Transcribed")
                        session.add(db_book)
                        session.commit()
                continue
            source_label = "linear chunker"
            state.add_console_log(
                f"[Prompt-Gen] Chunking '{book_name}' with the {source_label}: "
                f"{len(chunks)} chunks")
            if scene_map.scene_map_path(book_dir).exists():
                state.add_console_log(
                    f"[Prompt-Gen] Note: '{book_name}' has a scene map on disk, but "
                    f"maps are no longer part of the pipeline; its shots follow the "
                    f"linear chunker.")
            current_keys = {(chunk.chapter, chunk.shot) for chunk in chunks}
            orphans = completed_scenes - current_keys
            if orphans:
                state.add_console_log(
                    f"[Prompt-Gen] Warning: {len(orphans)} existing row(s) for "
                    f"'{book_name}' do not match any current chunk and are left in "
                    f"place; the transcript may have changed since they were generated.")

            # Enumerate every scene once. Planning is skipped when nothing is
            # pending, so re-running on a finished book costs no LLM time.
            all_scenes = [(chunk.chapter, chunk.shot, chunk.text) for chunk in chunks]
            pending = [(c, s, t) for (c, s, t) in all_scenes
                       if (c, s) not in completed_scenes]

            # Shot mode for this book, plus a snapshot of the active prompt
            # variants taken once per book so a mid-run change cannot split it.
            from services import prompt_library as prompt_lib
            from services import shot_planner
            shot_mode = prompt_lib.get_shot_mode(project_name, book_name)
            selection = prompt_lib.active_selection()
            shot_map = {}
            focus_map = {}
            place_map = {}
            if pending:
                state.pipeline_progress_line = (
                    f"Shot planning: {book_name} - checking the stored plan")
                shot_map = await shot_planner.ensure_book_plan(
                    project_name, book_name, llm_url,
                    llm_config.stage_model("plan"),
                    chunk_words=chunk_size_words, mode=shot_mode,
                    log=planner_progress_log(book_name), selection=selection,
                    should_cancel=prompt_gen_stopped,
                )
                # The focus the planner chose for each shot, read back from the stored plan.
                # Without it the shared block's "the director has chosen a focus" line has
                # nothing to point at, and the prompt model has to infer the subject.
                focus_map = shot_planner.load_focus_map(project_name, book_name)
                # ...and the place it read for that shot. A passage that is pure
                # dialogue names no setting, so without this line the prompt model
                # fills the background in itself - measured at 29 invented rooms in
                # one book's 328 outdoor scenes.
                place_map = shot_planner.load_place_map(project_name, book_name)
            state.add_console_log(
                f"[Prompt-Gen] Shot mode for '{book_name}': {shot_mode}"
                + f" - planner {selection.get('shot_direction', '?')}"
                + (f" - {len(shot_map)} planned scenes" if shot_map else "")
                + ("" if pending else " - nothing pending, skipping")
            )

            tasks = []
            for chapter_num, shot_num, chunk_text in pending:
                # NONE is a valid plan outcome, but pacing is one image per
                # chunk, so those scenes fall back to the single prompt.
                role = prompt_lib.resolve_role(
                    shot_map.get((chapter_num, shot_num)), shot_mode
                ) or "single"
                focus_for_shot = focus_map.get((chapter_num, shot_num), "")
                place_for_shot = place_map.get((chapter_num, shot_num), "")
                tasks.append({
                    "chunk": chunk_text,
                    "chapter_num": chapter_num,
                    "shot_num": shot_num,
                    "shot_role": role,
                    "project_id": project_id,
                    # Carried so a declined role can be re-asked from the same
                    # passage and the same plan without re-planning.
                    "selection": selection,
                    "focus": focus_for_shot,
                    "place": place_for_shot,
                    "prompt": prompt_lib.render_with_selection(
                        role, chunk_text, selection,
                        focus=focus_for_shot,
                        place=place_for_shot),
                })

            total_chunks = len(completed_scenes) + len(tasks)
            state.add_console_log(f"[Prompt-Gen] Total chunks/scenes: {total_chunks} ({len(tasks)} pending)")

            # The Run bar's counter. Incremented where a row is saved, so it can
            # never claim more prompts than prompts.csv holds.
            progress = {"done": len(completed_scenes), "total": total_chunks}

            def report_prompt_progress() -> None:
                state.pipeline_progress_line = (
                    f"Prompt-gen: {progress['done']}/{progress['total']} - {book_name}")

            # Save total count in SQLite for metrics tracking
            with Session(engine) as session:
                db_book = session.get(Book, book_id)
                if db_book:
                    db_book.total_images = total_chunks
                    db_book.completed_images = len(completed_scenes)
                    db_book.progress = len(completed_scenes) / total_chunks if total_chunks > 0 else 0.0
                    session.add(db_book)
                    session.commit()
            report_prompt_progress()

            if not tasks:
                state.add_console_log(f"[Prompt-Gen] All scenes for {book_name} are already generated.")
                state.pipeline_progress_line = (
                    f"Prompt-gen: {book_name} - already complete")
                completed_books += 1
                with Session(engine) as session:
                    db_book = session.get(Book, book_id)
                    if db_book:
                        db_book.status = "Prompts Created"
                        session.add(db_book)
                        session.commit()
                continue

            # Ensure CSV file header exists
            if not output_csv_path.exists():
                output_csv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_csv_path, "w", encoding="utf-8", newline="") as csv_f:
                    writer = csv.writer(csv_f, delimiter="|")
                    writer.writerow(["chapter", "shot", "prompt", "quote", "shot_role"])

            # Migrate legacy files that predate the shot_role column so appended
            # rows line up with the header.
            ensure_shot_role_column(output_csv_path)

            # 3. Warm-Up (First Prompt) Sequential Loading Phase
            first_task = tasks[0]
            state.add_console_log(f"[Prompt-Gen] Sending sequential warm-up request to load the model into VRAM (Ch {first_task['chapter_num']}, Scene {first_task['shot_num']})...")
            
            warm_up_success = False
            warm_up_res = None
            warm_up_sem = asyncio.Semaphore(1)
            
            # Allow up to 3 warm-up attempts to handle GPU allocation spikes
            for attempt in range(3):
                if prompt_gen_stopped():
                    break
                try:
                    res = await checkpoints.await_cancellable(
                        generate_prompt_for_chunk_async(
                            first_task, llm_url, model_name, warm_up_sem),
                        prompt_gen_stopped)
                    if res and res["status"] != "error":
                        warm_up_success = True
                        warm_up_res = res
                        state.add_console_log("[Prompt-Gen] Warm-up successful. Model is loaded and active.")
                        break
                    else:
                        state.add_console_log(f"[Prompt-Gen] Warm-up attempt {attempt+1} returned API error. Retrying...")
                except checkpoints.Cancelled:
                    # A stop is not a failed warm-up: re-raising keeps it out of
                    # the retry loop, which would otherwise re-ask three times.
                    raise
                except Exception as e:
                    state.add_console_log(f"[Prompt-Gen] Warm-up attempt {attempt+1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(2.5)

            if prompt_gen_stopped():
                state.add_console_log("[Prompt-Gen] Process interrupted by user during warm-up.")
                break

            if not warm_up_success:
                state.add_console_log("[Prompt-Gen] Error: Model warm-up failed. Suspending generation to prevent concurrent load errors.")
                break

            # Persist sequential Warm-up result
            if warm_up_res:
                if warm_up_res["status"] in ("refusal", "declined"):
                    _log_unusable(
                        warm_up_res, first_task["chunk"],
                        base_output_dir / project_name / book_name / "refusals.log")

                with open(output_csv_path, "a", encoding="utf-8", newline="") as csv_f:
                    writer = csv.writer(csv_f, delimiter="|")
                    writer.writerow([warm_up_res["chapter"], warm_up_res["shot"],
                                     warm_up_res["prompt"], warm_up_res["quote"],
                                     warm_up_res.get("shot_role")
                                     or first_task.get("shot_role", "character")])

                with Session(engine) as session:
                    db_book = session.get(Book, book_id)
                    if db_book:
                        db_book.completed_images = (db_book.completed_images or 0) + 1
                        db_book.progress = db_book.completed_images / db_book.total_images if db_book.total_images > 0 else 0.0
                        session.add(db_book)
                        session.commit()

                state.recent_prompts.append({
                    "book": book_name,
                    "chapter": warm_up_res["chapter"],
                    "shot": warm_up_res["shot"],
                    "prompt": warm_up_res["prompt"],
                    "quote": warm_up_res["quote"],
                    "status": warm_up_res["status"]
                })
                progress["done"] += 1
                report_prompt_progress()

            # Exclude the first task and process the rest of the queue concurrently
            remaining_tasks = tasks[1:]
            if not remaining_tasks:
                # If there were no other tasks, sort the completed CSV
                sort_csv_by_chapter_scene(output_csv_path)
                completed_books += 1
                with Session(engine) as session:
                    db_book = session.get(Book, book_id)
                    if db_book:
                        db_book.status = "Prompts Created"
                        session.add(db_book)
                        session.commit()
                continue

            # 4. Dynamic Parallel Processing via Semaphores & Cancellable Tasks
            semaphore = asyncio.Semaphore(4)  # Limit concurrent API requests to 4
            
            async def run_and_save_task(task_data):
                if prompt_gen_stopped():
                    return None

                try:
                    res = await generate_prompt_for_chunk_async(task_data, llm_url, model_name, semaphore)
                    
                    if prompt_gen_stopped():
                        return None

                    # Treat refusals and failures gracefully
                    if res["status"] in ("refusal", "declined"):
                        _log_unusable(
                            res, task_data["chunk"],
                            base_output_dir / project_name / book_name / "refusals.log")

                    # Append results to prompts.csv immediately
                    with open(output_csv_path, "a", encoding="utf-8", newline="") as csv_f:
                        writer = csv.writer(csv_f, delimiter="|")
                        writer.writerow([res["chapter"], res["shot"], res["prompt"],
                                         res["quote"], res.get("shot_role")
                                         or task_data.get("shot_role", "character")])

                    # Update database progress in-place
                    with Session(engine) as session:
                        db_book = session.get(Book, book_id)
                        if db_book:
                            db_book.completed_images = (db_book.completed_images or 0) + 1
                            db_book.progress = db_book.completed_images / db_book.total_images if db_book.total_images > 0 else 0.0
                            session.add(db_book)
                            session.commit()

                    # Push metadata details to our live UI feed
                    ui_item = {
                        "book": book_name,
                        "chapter": res["chapter"],
                        "shot": res["shot"],
                        "prompt": res["prompt"],
                        "quote": res["quote"],
                        "status": res["status"]
                    }
                    state.recent_prompts.append(ui_item)
                    if len(state.recent_prompts) > 5:
                        state.recent_prompts.pop(0)

                    progress["done"] += 1
                    report_prompt_progress()

                    return res
                except asyncio.CancelledError:
                    raise
                except Exception as ex:
                    state.add_console_log(f"[Prompt-Gen] Chunk S{task_data['shot_num']} processing error: {str(ex)}")
                    return None

            # Wrap tasks as explicit task handles inside the event loop
            active_tasks = [asyncio.create_task(run_and_save_task(t)) for t in remaining_tasks]
            
            # Active Loop Monitor
            while not all(t.done() for t in active_tasks):
                if prompt_gen_stopped():
                    state.add_console_log("[Prompt-Gen] Cancelling active background LLM requests...")
                    for task in active_tasks:
                        if not task.done():
                            task.cancel()
                    break
                await asyncio.sleep(0.5)

            # Force synchronization and await active tasks to let cancellation unwind cleanly
            await asyncio.gather(*active_tasks, return_exceptions=True)

            # Sort prompts.csv, run timing synchronization, and update book status to final
            if not prompt_gen_stopped():
                sort_csv_by_chapter_scene(output_csv_path)
                
                # Trigger Phase C Timing Sync Pipeline automatically
                state.add_console_log(f"[Prompt-Gen] Initiating fuzzy timing alignment for {book_name}...")
                from services.timing_sync import sync_book_timing
                await asyncio.to_thread(sync_book_timing, book_id, project_name, book_name, state.add_console_log)
                
                with Session(engine) as session:
                    db_book = session.get(Book, book_id)
                    if db_book:
                        db_book.status = "Prompts Created"
                        session.add(db_book)
                        session.commit()
                completed_books += 1

        # Update Project Status upon completing loop. A project with skipped books
        # stays in the prompt phase so the analysis card is still available for
        # the rest of the series.
        if not prompt_gen_stopped():
            with Session(engine) as session:
                db_project = session.get(Project, project_id)
                if db_project:
                    db_project.status = ("Prompts Created" if skipped_books == 0
                                         else "Transcribed")
                    session.add(db_project)
                    session.commit()
            if skipped_books:
                state.add_console_log(
                    f"[Prompt-Gen] {completed_books} book(s) complete, "
                    f"{skipped_books} skipped; the project stays in the prompt phase.")
            else:
                state.add_console_log("[Prompt-Gen] Success! All prompts generated and synced with timestamps.")
        else:
            state.add_console_log("[Prompt-Gen] Prompt generation suspended successfully.")

    except checkpoints.Cancelled:
        # A stop raised out of shot planning or a request: not a failure. The
        # book's plan is written whole or not at all, so the run resumes cleanly.
        state.add_console_log(
            "[Prompt-Gen] Stopped by user. Rows already written are kept; Run "
            "resumes from them.")
    except Exception as ex:
        state.add_console_log(f"[Prompt-Gen] Critical Error: {str(ex)}")
    finally:
        state.prompt_gen_active = False
        
        # Bulletproof fallback to ensure statuses revert on any interrupt, cancellation, or exception
        with Session(engine) as session:
            db_project = session.get(Project, project_id)
            if db_project and db_project.status == "Generating Prompts":
                db_project.status = "Transcribed"
                session.add(db_project)
                
            for b_data in books_data:
                b_id = b_data["id"]
                db_book = session.get(Book, b_id)
                if db_book and db_book.status == "Generating Prompts":
                    db_book.status = "Transcribed"
                    session.add(db_book)
            session.commit()