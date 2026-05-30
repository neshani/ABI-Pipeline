import httpx
import re
import os
import random
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from database.connection import get_setting
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


def smart_chunk_text(text: str, max_chunk_words: int) -> List[str]:
    """Ported robust text chunking algorithm."""
    if not text.strip() or max_chunk_words <= 0:
        return []

    def split_into_sentences(paragraph_text: str) -> List[str]:
        if not paragraph_text:
            return []
        return re.findall(r'[^.!?]+[.!?]?', paragraph_text)

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk_words = []

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        if not paragraph_words:
            continue

        if len(current_chunk_words) + len(paragraph_words) <= max_chunk_words:
            current_chunk_words.extend(paragraph_words)
        else:
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []

            sentences = split_into_sentences(paragraph)
            for sentence in sentences:
                sentence_words = sentence.strip().split()
                if not sentence_words:
                    continue

                if len(current_chunk_words) + len(sentence_words) > max_chunk_words and current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))
                    current_chunk_words = []
                
                current_chunk_words.extend(sentence_words)

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


async def get_llm_response(prompt: str, llm_url: str, model_name: str) -> str:
    """Dispatches asynchronous request to local Ollama or LM Studio OpenAI-compatible endpoints."""
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
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(endpoint, json=payload, headers=headers, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            # Fallback to direct Ollama generate endpoint if connection fails and endpoint targets port 11434
            if "11434" in endpoint or "ollama" in endpoint.lower():
                try:
                    direct_endpoint = f"{llm_url.rstrip('/')}/api/generate"
                    direct_payload = {
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7}
                    }
                    resp = await client.post(direct_endpoint, json=direct_payload, timeout=60.0)
                    resp.raise_for_status()
                    return resp.json().get("response", "")
                except Exception as inner_e:
                    return f"API Connection Failed: {e} (Direct API Fallback failed: {inner_e})"
            return f"API Connection Failed: {e}"


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
    chunk_size = int(get_setting("batch_size", 30)) * 10
    if chunk_size <= 0:
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
    template: str, 
    semaphore: asyncio.Semaphore
) -> dict:
    """Generates prompt for a single chunk using a concurrency semaphore."""
    async with semaphore:
        final_prompt = template.replace("<text>", task["chunk"])
        raw_response = await get_llm_response(final_prompt, llm_url, model_name)
        parsed = parse_llm_response(raw_response)
        return {
            "chapter": task["chapter_num"],
            "scene": task["scene_num"],
            "quote": parsed["quote"],
            "prompt": parsed["prompt"],
            "status": parsed["status"],
            "raw": parsed["raw"]
        }


def cancel_prompt_generation(project_id: int):
    """Flags active prompt generation to cancel."""
    state.cancel_prompt_gen_flag = True
    state.add_console_log(f"[Prompt-Gen] Cancellation signal sent to Project ID {project_id}...")


def sort_csv_by_chapter_scene(csv_path: Path):
    """Sorts the final prompts.csv file numerically by chapter and scene."""
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
        
        # Sort rows numerically by chapter (index 0) and scene (index 1)
        rows.sort(key=lambda r: (int(r[0]), int(r[1])))
        
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(header)
            writer.writerows(rows)
    except Exception as ex:
        print(f"Error sorting prompts.csv: {str(ex)}")


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
        model_name = get_setting("llm_model", "local-model")
        chunk_size_words = int(get_setting("batch_size", 30)) * 10
        if chunk_size_words <= 0:
            chunk_size_words = 300

        # Load Prompt Template
        template_name = state.playground_selected_template or "default"
        template_text = load_template_by_name(template_name)
        if not template_text:
            template_text = load_template_by_name("default")

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
                continue

            # 1. Load Existing Progress for Resuming
            completed_scenes = set()
            if output_csv_path.exists():
                try:
                    with open(output_csv_path, "r", encoding="utf-8") as csv_f:
                        reader = csv.DictReader(csv_f, delimiter="|")
                        for row in reader:
                            if "chapter" in row and "scene" in row:
                                completed_scenes.add((int(row["chapter"]), int(row["scene"])))
                except Exception as ex:
                    state.add_console_log(f"[Prompt-Gen] Warning reading existing prompts.csv: {str(ex)}")

            if completed_scenes:
                state.add_console_log(f"[Prompt-Gen] Found {len(completed_scenes)} existing prompts. Resuming where left off...")

            # 2. Segment transcript into chapters & chunks
            with open(transcript_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            chapters = [ch for ch in full_text.split("==CHAPTER==") if ch.strip()]
            
            tasks = []
            for i, chapter_text in enumerate(chapters):
                chapter_num = i + 1
                chunks = smart_chunk_text(chapter_text, chunk_size_words)
                for j, chunk_text in enumerate(chunks):
                    scene_num = j + 1
                    if (chapter_num, scene_num) not in completed_scenes:
                        tasks.append({
                            "chunk": chunk_text,
                            "chapter_num": chapter_num,
                            "scene_num": scene_num
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
                    writer.writerow(["chapter", "scene", "prompt", "quote"])

            # 3. Warm-Up (First Prompt) Sequential Loading Phase
            first_task = tasks[0]
            state.add_console_log(f"[Prompt-Gen] Sending sequential warm-up request to load the model into VRAM (Ch {first_task['chapter_num']}, Scene {first_task['scene_num']})...")
            
            warm_up_success = False
            warm_up_res = None
            warm_up_sem = asyncio.Semaphore(1)
            
            # Allow up to 3 warm-up attempts to handle GPU allocation spikes
            for attempt in range(3):
                if state.cancel_prompt_gen_flag:
                    break
                try:
                    res = await generate_prompt_for_chunk_async(first_task, llm_url, model_name, template_text, warm_up_sem)
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
                        ref_f.write(f"Chapter {warm_up_res['chapter']}, Scene {warm_up_res['scene']}:\n{warm_up_res['raw']}\n\n{'-'*40}\n\n")
                    warm_up_res["prompt"] = "REFUSAL"
                    warm_up_res["quote"] = first_task["chunk"]

                with open(output_csv_path, "a", encoding="utf-8", newline="") as csv_f:
                    writer = csv.writer(csv_f, delimiter="|")
                    writer.writerow([warm_up_res["chapter"], warm_up_res["scene"], warm_up_res["prompt"], warm_up_res["quote"]])

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
                    "scene": warm_up_res["scene"],
                    "prompt": warm_up_res["prompt"],
                    "quote": warm_up_res["quote"],
                    "status": warm_up_res["status"]
                })

            # Exclude the first task and process the rest of the queue concurrently
            remaining_tasks = tasks[1:]
            if not remaining_tasks:
                # If there were no other tasks, sort the completed CSV
                sort_csv_by_chapter_scene(output_csv_path)
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
                    res = await generate_prompt_for_chunk_async(task_data, llm_url, model_name, template_text, semaphore)
                    
                    if state.cancel_prompt_gen_flag:
                        return None

                    # Treat refusals and failures gracefully
                    if res["status"] == "refusal":
                        ref_log_path = base_output_dir / project_name / book_name / "refusals.log"
                        ref_log_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(ref_log_path, "a", encoding="utf-8") as ref_f:
                            ref_f.write(f"Chapter {res['chapter']}, Scene {res['scene']}:\n{res['raw']}\n\n{'-'*40}\n\n")
                        
                        # Store refusal in CSV with a placeholder for grid highlighting in Phase 5
                        res["prompt"] = "REFUSAL"
                        res["quote"] = task_data["chunk"]

                    # Append results to prompts.csv immediately
                    with open(output_csv_path, "a", encoding="utf-8", newline="") as csv_f:
                        writer = csv.writer(csv_f, delimiter="|")
                        writer.writerow([res["chapter"], res["scene"], res["prompt"], res["quote"]])

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
                        "scene": res["scene"],
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
                    state.add_console_log(f"[Prompt-Gen] Chunk S{task_data['scene_num']} processing error: {str(ex)}")
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

        # Update Project Status upon completing loop
        if not state.cancel_prompt_gen_flag:
            with Session(engine) as session:
                db_project = session.get(Project, project_id)
                if db_project:
                    db_project.status = "Prompts Created"
                    session.add(db_project)
                    session.commit()
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