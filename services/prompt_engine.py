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
# The paragraph chunker is the fast path of the scene-map module and lives
# there now. It is re-exported here because prompt generation, the shot planner
# and the character manager have always imported it from this module.
from services.scene_map import (
    PARA_SLACK,
    SENT_SLACK,
    SEG_VERSION,
    StaleSceneMap,
    book_chunks,
    scene_map_signature,
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
):
    """Dispatches asynchronous request to local Ollama or LM Studio OpenAI-compatible endpoints.

    ``extra_payload`` merges into the request body, e.g. to disable a model's
    thinking mode: ``{"chat_template_kwargs": {"enable_thinking": False}}``.
    ``timeout`` is per request: scene segmentation reads a whole chapter and can
    take minutes, while a one-shot prompt is expected in under a minute.
    ``with_meta`` returns ``(content, finish_reason)`` instead of the bare string.
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
        "messages": [{"role": "user", "content": prompt}],
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
                    return (text, "") if with_meta else text
                except Exception as inner_e:
                    msg = f"API Connection Failed: {e} (Direct API Fallback failed: {inner_e})"
                    return (msg, "error") if with_meta else msg
            msg = f"API Connection Failed: {e}"
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
    is_refusal = has_refusal_words and (not parsed_prompt or parsed_prompt == parsed_quote)

    return {
        "raw": raw_response,
        "quote": parsed_quote,
        "prompt": parsed_prompt,
        "status": "refusal" if is_refusal else "success"
    }


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

    cleaned_text = content.replace("==CHAPTER==", " ").strip()
    chunk_size = getattr(state, "playground_chunk_size", 350)
    if not chunk_size or chunk_size <= 0:
        chunk_size = 350

    all_chunks = smart_chunk_text(cleaned_text, chunk_size)
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
    semaphore: asyncio.Semaphore
) -> dict:
    """Generates prompt for a single chunk using a concurrency semaphore.

    ``task["prompt"]`` is already fully composed (shared block plus the
    shot-role fragment chosen for this scene) by the caller.
    """
    async with semaphore:
        final_prompt = task["prompt"]
        raw_response = await get_llm_response(final_prompt, llm_url, model_name)
        parsed = parse_llm_response(raw_response)
        # Local import: character_manager imports this module.
        from services.character_manager import auto_bracket_character_names
        return {
            "chapter": task["chapter_num"],
            "shot": task["shot_num"],
            "quote": parsed["quote"],
            "prompt": auto_bracket_character_names(
                parsed["prompt"], task.get("project_id") or 0),
            "status": parsed["status"],
            "raw": parsed["raw"]
        }


def cancel_prompt_generation(project_id: int):
    """Flags active prompt generation to cancel."""
    state.cancel_prompt_gen_flag = True
    state.add_console_log(f"[Prompt-Gen] Cancellation signal sent to Project ID {project_id}...")


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

    The chunks come from the chunking the book is bound to: its manifest, else
    the legacy linear rule when it already has prompt rows, else its scene map
    when it has one. A stale map, or a ``prompts.csv`` built under a different
    chunking, comes back as ``problem`` instead of raising or silently switching
    coordinate systems; the caller logs it and skips the book. On success the
    pipeline manifest records which chunking owns the book from now on.
    """
    prompts_exist = bool(completed_scenes) or scene_map.prompts_have_rows(book_dir)
    # In Full/Scenes mode a book without a scene map has not been analysed yet.
    # Generating it linearly would bind it to the old chunking and lock the new
    # pipeline out, so skip it with an actionable message instead. Fast mode is
    # the explicit request for linear, and a legacy book with rows keeps its
    # existing coordinates.
    try:
        mode = str(get_setting("pipeline_mode", "full") or "full")
    except Exception:
        mode = "full"
    if (mode in ("scenes", "full") and not prompts_exist
            and not scene_map.scene_map_path(book_dir).exists()):
        return None, "", (
            f"no scene map yet; the project is in '{mode}' mode. Run character "
            f"analysis for this book first, or switch the project to Fast.")
    try:
        source = scene_map.book_source(book_dir, prompts_exist=prompts_exist)
        chunks, source = book_chunks(book_dir, chunk_words, text=full_text,
                                     source=source)
    except StaleSceneMap as ex:
        return None, "", str(ex)

    sha = transcript_digest(full_text)
    map_sha = scene_map_signature(book_dir) if source == "scene_map" else ""

    # The characters' event positions are the first chunk of their group at the
    # chunk size analysis ran with. A different size would put every description
    # a few shots off, so refuse rather than write subtly wrong prompts.
    recorded_analysis_words = pipeline_manifest.read(book_dir).get("analysis_chunk_words")
    if source == "scene_map" and recorded_analysis_words is not None:
        try:
            if int(recorded_analysis_words) != int(chunk_words):
                return None, source, (
                    f"character analysis was run at {recorded_analysis_words} words "
                    f"per shot, but the current setting is {chunk_words}; restore "
                    f"that setting or re-run analysis for this book.")
        except (TypeError, ValueError):
            pass

    allowed, reason = pipeline_manifest.check(
        book_dir, source, chunk_words, transcript_sha=sha, scene_map_sha=map_sha,
        prompts_exist=prompts_exist)
    if not allowed:
        return None, source, reason
    pipeline_manifest.record_chunking(
        book_dir, source, chunk_words, seg_version=SEG_VERSION,
        transcript_sha=sha, scene_map_sha=map_sha)
    return chunks, source, ""


async def start_project_prompt_gen(project_id: int):
    """
    Orchestrates sequential resumable prompt generation for all books in a project.
    Processes chapters and chunks in parallel using asyncio Semaphores.
    """
    import csv
    import asyncio
    from sqlmodel import Session, select
    from database.models import Project, Book
    from database.connection import engine, get_setting

    if state.prompt_gen_active:
        state.add_console_log("[Prompt-Gen] Warning: A prompt generation task is already active.")
        return

    state.prompt_gen_active = True
    state.cancel_prompt_gen_flag = False
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

            if state.cancel_prompt_gen_flag:
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
            source_label = "scene map" if chunk_source == "scene_map" else "linear chunker"
            state.add_console_log(
                f"[Prompt-Gen] Chunking '{book_name}' with the {source_label}: "
                f"{len(chunks)} chunks")
            if chunk_source == "linear" and scene_map.scene_map_path(book_dir).exists():
                state.add_console_log(
                    f"[Prompt-Gen] Note: '{book_name}' has a scene map, but its "
                    f"prompts were generated linearly; rebuild the book to use the map.")
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
            if pending:
                shot_map = await shot_planner.ensure_book_plan(
                    project_name, book_name, llm_url,
                    llm_config.stage_model("plan"),
                    chunk_words=chunk_size_words, mode=shot_mode,
                    log=state.add_console_log, selection=selection,
                )
                # The focus the planner chose for each shot, read back from the stored plan.
                # Without it the shared block's "the director has chosen a focus" line has
                # nothing to point at, and the prompt model has to infer the subject.
                focus_map = shot_planner.load_focus_map(project_name, book_name)
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
                tasks.append({
                    "chunk": chunk_text,
                    "chapter_num": chapter_num,
                    "shot_num": shot_num,
                    "shot_role": role,
                    "project_id": project_id,
                    "prompt": prompt_lib.render_with_selection(
                        role, chunk_text, selection,
                        focus=focus_map.get((chapter_num, shot_num), "")),
                })

            total_chunks = len(completed_scenes) + len(tasks)
            state.add_console_log(f"[Prompt-Gen] Total chunks/scenes: {total_chunks} ({len(tasks)} pending)")

            # Save total count in SQLite for metrics tracking
            with Session(engine) as session:
                db_book = session.get(Book, book_id)
                if db_book:
                    db_book.total_images = total_chunks
                    db_book.completed_images = len(completed_scenes)
                    db_book.progress = len(completed_scenes) / total_chunks if total_chunks > 0 else 0.0
                    session.add(db_book)
                    session.commit()

            if not tasks:
                state.add_console_log(f"[Prompt-Gen] All scenes for {book_name} are already generated.")
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
                if state.cancel_prompt_gen_flag:
                    break
                try:
                    res = await generate_prompt_for_chunk_async(first_task, llm_url, model_name, warm_up_sem)
                    if res and res["status"] != "error":
                        warm_up_success = True
                        warm_up_res = res
                        state.add_console_log("[Prompt-Gen] Warm-up successful. Model is loaded and active.")
                        break
                    else:
                        state.add_console_log(f"[Prompt-Gen] Warm-up attempt {attempt+1} returned API error. Retrying...")
                except Exception as e:
                    state.add_console_log(f"[Prompt-Gen] Warm-up attempt {attempt+1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(2.5)

            if state.cancel_prompt_gen_flag:
                state.add_console_log("[Prompt-Gen] Process interrupted by user during warm-up.")
                break

            if not warm_up_success:
                state.add_console_log("[Prompt-Gen] Error: Model warm-up failed. Suspending generation to prevent concurrent load errors.")
                state.cancel_prompt_gen_flag = True
                break

            # Persist sequential Warm-up result
            if warm_up_res:
                if warm_up_res["status"] == "refusal":
                    ref_log_path = base_output_dir / project_name / book_name / "refusals.log"
                    ref_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(ref_log_path, "a", encoding="utf-8") as ref_f:
                        ref_f.write(f"Chapter {warm_up_res['chapter']}, Scene {warm_up_res['shot']}:\n{warm_up_res['raw']}\n\n{'-'*40}\n\n")
                    warm_up_res["prompt"] = "REFUSAL"
                    warm_up_res["quote"] = first_task["chunk"]

                with open(output_csv_path, "a", encoding="utf-8", newline="") as csv_f:
                    writer = csv.writer(csv_f, delimiter="|")
                    writer.writerow([warm_up_res["chapter"], warm_up_res["shot"], warm_up_res["prompt"], warm_up_res["quote"], first_task.get("shot_role", "character")])

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
                if state.cancel_prompt_gen_flag:
                    return None

                try:
                    res = await generate_prompt_for_chunk_async(task_data, llm_url, model_name, semaphore)
                    
                    if state.cancel_prompt_gen_flag:
                        return None

                    # Treat refusals and failures gracefully
                    if res["status"] == "refusal":
                        ref_log_path = base_output_dir / project_name / book_name / "refusals.log"
                        ref_log_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(ref_log_path, "a", encoding="utf-8") as ref_f:
                            ref_f.write(f"Chapter {res['chapter']}, Scene {res['shot']}:\n{res['raw']}\n\n{'-'*40}\n\n")
                        
                        # Store refusal in CSV with a placeholder for grid highlighting in Phase 5
                        res["prompt"] = "REFUSAL"
                        res["quote"] = task_data["chunk"]

                    # Append results to prompts.csv immediately
                    with open(output_csv_path, "a", encoding="utf-8", newline="") as csv_f:
                        writer = csv.writer(csv_f, delimiter="|")
                        writer.writerow([res["chapter"], res["shot"], res["prompt"], res["quote"], task_data.get("shot_role", "character")])

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
                if state.cancel_prompt_gen_flag:
                    state.add_console_log("[Prompt-Gen] Cancelling active background LLM requests...")
                    for task in active_tasks:
                        if not task.done():
                            task.cancel()
                    break
                await asyncio.sleep(0.5)

            # Force synchronization and await active tasks to let cancellation unwind cleanly
            await asyncio.gather(*active_tasks, return_exceptions=True)

            # Sort prompts.csv, run timing synchronization, and update book status to final
            if not state.cancel_prompt_gen_flag:
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
        if not state.cancel_prompt_gen_flag:
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