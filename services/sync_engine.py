# services/sync_engine.py
import os
import json
import csv
from pathlib import Path
from sqlmodel import Session, select, delete
from database.connection import engine, get_setting
from database.models import Project, Book, Chapter, ScenePrompt

def sync_project_status(project_id: int, session: Session) -> None:
    """
    Dynamically calculates and updates the parent Project status
    based on the status of all of its child Books. Evaluates the
    child statuses from the earliest bottleneck state to the latest.
    """
    project = session.get(Project, project_id)
    if not project:
        return

    books = session.exec(select(Book).where(Book.project_id == project_id)).all()
    if not books:
        return

    book_statuses = [b.status for b in books]

    # 1. Active pipeline execution states take priority immediately to show running animations
    if "Rendering Images" in book_statuses:
        project.status = "Rendering Images"
    elif "Transcribing" in book_statuses:
        project.status = "Transcribing"
    elif "Generating Prompts" in book_statuses:
        project.status = "Generating Prompts"
    # 2. Base static states default to the earliest incomplete bottleneck state
    elif "Imported" in book_statuses:
        project.status = "Imported"
    elif "Transcribed" in book_statuses:
        project.status = "Transcribed"
    elif "Prompts Created" in book_statuses:
        project.status = "Prompts Created"
    else:
        project.status = "Images Created"

    session.add(project)
    session.flush()


def sync_prompts_csv_to_db_cache(book_id: int, session: Session) -> None:
    """
    Parses prompts.csv and loads rows into SQLite ScenePrompt cache
    only if prompts.csv modification time has changed.
    """
    book = session.get(Book, book_id)
    if not book:
        return

    project = session.get(Project, book.project_id) if book.project_id else None
    project_name = project.name if project else "Default_Project"

    base_output_dir = Path(get_setting("output_dir", "./output", session)).resolve()
    book_output_dir = base_output_dir / project_name / book.name
    prompts_file = book_output_dir / "prompts.csv"

    if not prompts_file.exists():
        session.exec(delete(ScenePrompt).where(ScenePrompt.book_id == book_id))
        book.prompts_mtime = None
        session.add(book)
        session.flush()
        return

    mtime = prompts_file.stat().st_mtime
    if book.prompts_mtime == mtime:
        return  # Cache is already fully up to date!

    try:
        with open(prompts_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            if not reader.fieldnames:
                return
            
            # Normalize column headers to lowercase
            reader.fieldnames = [name.strip().lower() if name else "" for name in reader.fieldnames]
            
            # Clear existing scene cache for this book
            session.exec(delete(ScenePrompt).where(ScenePrompt.book_id == book_id))
            session.flush()

            for row in reader:
                cleaned_row = {k.strip().lower(): v.strip() if v else "" for k, v in row.items() if k}
                try:
                    ch = int(float(cleaned_row.get("chapter", "1")))
                    sc = int(float(cleaned_row.get("scene", "1")))
                except (ValueError, TypeError):
                    ch, sc = 1, 1

                is_approved = cleaned_row.get("approved", "false").strip().lower() == "true"
                timestamp = cleaned_row.get("timestamp", "00:00:00")

                scene_prompt = ScenePrompt(
                    book_id=book_id,
                    chapter_num=ch,
                    scene_num=sc,
                    prompt=cleaned_row.get("prompt", ""),
                    quote=cleaned_row.get("quote", ""),
                    approved=is_approved,
                    timestamp=timestamp
                )
                session.add(scene_prompt)

            book.prompts_mtime = mtime
            session.add(book)
            session.flush()
    except Exception as e:
        print(f"[Sync-Engine] Error caching prompts.csv to SQLite: {e}")



def sync_book_from_disk(book_id: int, session: Session) -> None:
    """
    Parses compiled transcript.txt and prompts.csv to update 
    the SQLite database index metrics (word counts, total/completed images)
    and status (Imported, Transcribed, Prompts Created, Images Created)
    for a book and its chapters on demand. Calculates logical phase-aware progress.
    """
    book = session.get(Book, book_id)
    if not book:
        return

    # First, guarantee the SQLite database ScenePrompt cache table is fully synchronized with disk CSV state
    sync_prompts_csv_to_db_cache(book_id, session)

    project = session.get(Project, book.project_id) if book.project_id else None
    project_name = project.name if project else "Default_Project"

    # Base output directories
    base_output_dir = Path(get_setting("output_dir", "./output", session)).resolve()
    book_output_dir = base_output_dir / project_name / book.name
    transcript_file = book_output_dir / "transcript.txt"
    prompts_file = book_output_dir / "prompts.csv"

    has_transcript = transcript_file.exists()
    has_prompts = prompts_file.exists()

    # 1. Update Word Count Metrics from transcript.txt
    if has_transcript:
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Split chapters on delimiter
            sections = content.split("==CHAPTER==")
            cleaned_sections = [s.strip() for s in sections if s.strip()]

            # Calculate global book word count
            book.word_count = len(content.replace("==CHAPTER==", "").strip().split())

            # Map word counts to individual database Chapters sequentially
            chapters = session.exec(
                select(Chapter).where(Chapter.book_id == book_id).order_by(Chapter.chapter_num)
            ).all()

            for idx, ch in enumerate(chapters):
                ch.status = "Completed"  # Mark transcription complete
                if idx < len(cleaned_sections):
                    ch.word_count = len(cleaned_sections[idx].split())
                session.add(ch)
            
            session.add(book)
        except Exception as e:
            print(f"Error parsing word count for book '{book.name}': {e}")
    else:
        # Reset word counts and statuses when transcript does not exist
        book.word_count = 0
        chapters = session.exec(
            select(Chapter).where(Chapter.book_id == book_id).order_by(Chapter.chapter_num)
        ).all()
        for ch in chapters:
            ch.word_count = 0
            ch.status = "Pending"
            session.add(ch)
        session.add(book)

    # 2. Update Image Counters from prompts.csv and generated images on disk
    global_completed = 0
    global_total = 0

    if has_prompts:
        try:
            # High-speed list lookup: find all PNGs in book's output structure
            images_dir = book_output_dir / "images"
            all_existing_images = set()
            if images_dir.exists():
                all_existing_images.update([f.name.lower() for f in images_dir.iterdir() if f.is_file()])
            if book_output_dir.exists():
                all_existing_images.update([f.name.lower() for f in book_output_dir.iterdir() if f.is_file()])

            chapter_totals = {}     # chapter_num -> expected prompts
            chapter_completed = {}  # chapter_num -> completed images

            with open(prompts_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="|")
                rows = list(reader)

            if rows:
                header = rows[0]
                header_clean = [h.strip().lower() for h in header]
                
                # Dynamically resolve columns by name or index fallback
                if "chapter" in header_clean or "prompt" in header_clean:
                    data_rows = rows[1:]
                    try:
                        ch_idx = header_clean.index("chapter")
                    except ValueError:
                        ch_idx = 0
                    try:
                        sc_idx = header_clean.index("scene")
                    except ValueError:
                        sc_idx = 1
                    try:
                        pr_idx = header_clean.index("prompt")
                    except ValueError:
                        pr_idx = 3
                else:
                    data_rows = rows
                    ch_idx = 0
                    sc_idx = 1
                    pr_idx = 3

                for row in data_rows:
                    if not row or len(row) <= max(ch_idx, sc_idx):
                        continue
                    
                    prompt_text = row[pr_idx].strip() if len(row) > pr_idx else ""
                    # Ignore unpopulated, NONE, or skipped refusal prompts in rendering counts
                    if not prompt_text or prompt_text.lower() == "none" or prompt_text.lower() == "refusal":
                        continue

                    try:
                        chapter_num = int(float(row[ch_idx].strip()))
                    except (ValueError, TypeError):
                        chapter_num = 1
                    try:
                        scene_num = int(float(row[sc_idx].strip()))
                    except (ValueError, TypeError):
                        scene_num = 1

                    scene_prefix = f"{chapter_num:02d}_{scene_num:02d}"

                    # Scan the high-speed set to see if this scene prefix is rendered
                    image_found = False
                    for img_name in all_existing_images:
                        if img_name.startswith(scene_prefix.lower()) and img_name.endswith(".png"):
                            image_found = True
                            break

                    global_total += 1
                    chapter_totals[chapter_num] = chapter_totals.get(chapter_num, 0) + 1
                    if image_found:
                        global_completed += 1
                        chapter_completed[chapter_num] = chapter_completed.get(chapter_num, 0) + 1

            # Update book-level exact tallies
            book.total_images = global_total
            book.completed_images = global_completed
            session.add(book)

            # Update chapter-level exact tallies
            chapters = session.exec(select(Chapter).where(Chapter.book_id == book_id)).all()
            for ch in chapters:
                ch.total_images = chapter_totals.get(ch.chapter_num, 0)
                ch.completed_images = chapter_completed.get(ch.chapter_num, 0)
                session.add(ch)

        except Exception as e:
            print(f"Error counting images in prompts.csv for '{book.name}': {e}")
    else:
        # If prompts don't exist, reset image metrics on books and chapters
        book.total_images = 0
        book.completed_images = 0
        session.add(book)
        
        chapters = session.exec(select(Chapter).where(Chapter.book_id == book_id)).all()
        for ch in chapters:
            ch.total_images = 0
            ch.completed_images = 0
            session.add(ch)

    # 3. Dynamic stage status recovery based on file existence and completion values
    if has_transcript:
        if has_prompts:
            if global_total > 0 and global_completed == global_total:
                book.status = "Images Created"
            else:
                book.status = "Prompts Created"
        else:
            book.status = "Transcribed"
    else:
        book.status = "Imported"

    # 4. Phase-Aware Logical Book Progress Bar Calculation
    if book.status in ("Imported", "Transcribing"):
        chapters = session.exec(select(Chapter).where(Chapter.book_id == book_id)).all()
        completed_chapters = len([c for c in chapters if c.status == "Completed"])
        total_chapters = len(chapters)
        book.progress = completed_chapters / total_chapters if total_chapters > 0 else 0.0
    elif book.status in ("Transcribed", "Generating Prompts"):
        book.progress = 1.0  # Transcription stage is fully complete
    else:  # "Prompts Created", "Rendering Images", "Images Created"
        book.progress = global_completed / global_total if global_total > 0 else 0.0

    session.add(book)
    session.flush()

    # Cascade the state update to recalculate the project's overall stage status
    if book.project_id:
        sync_project_status(book.project_id, session)


def recover_from_temp_workspaces(session: Session) -> None:
    """
    Scans both workspace_temp/ and output/ folders for transcription tracking metadata.
    Reconstructs the complete database index on database wipe.
    """
    detected_projects = {}  # project_path -> project_name
    meta_items = []

    # 1. Gather any incomplete/active workspace recovery configurations
    temp_dir = Path("./workspace_temp")
    if temp_dir.exists() and temp_dir.is_dir():
        for working_dir in temp_dir.iterdir():
            if working_dir.is_dir() and working_dir.name.startswith("book_"):
                state_file = working_dir / "transcription_state.json"
                if state_file.exists():
                    try:
                        with open(state_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            data["source_type"] = "temp"
                            data["working_dir"] = str(working_dir)
                            meta_items.append(data)
                    except Exception as e:
                        print(f"Error reading temp state file {state_file}: {e}")

    # 2. Gather any fully completed output configurations
    output_dir = Path(get_setting("output_dir", "./output", session)).resolve()
    if output_dir.exists() and output_dir.is_dir():
        for meta_file in output_dir.glob("*/*/metadata.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["source_type"] = "output"
                    data["book_output_dir"] = str(meta_file.parent)
                    meta_items.append(data)
            except Exception as e:
                print(f"Error reading output metadata file {meta_file}: {e}")

    if not meta_items:
        return

    print(f"[Sync-Engine] Found {len(meta_items)} project tracking records to synchronize.")

    # 3. Reconstruct parent Projects
    for item in meta_items:
        project_name = item.get("project_name")
        project_path = item.get("project_path")
        if project_name and project_path:
            detected_projects[project_path] = project_name

    # Ensure parent projects are scanned and ingested in full
    for proj_path, proj_name in detected_projects.items():
        project_exists = session.exec(
            select(Project).where(Project.name == proj_name).where(Project.path == proj_path)
        ).first()

        if not project_exists:
            print(f"[Sync-Engine] Re-scanning parent project path: {proj_path}")
            try:
                from services.scanner import scan_directory, ingest_project
                scan_res = scan_directory(proj_path)
                if scan_res["type"] != "none":
                    # Pass the active transaction session directly down to parent scanner ingest
                    ingest_project(scan_res, proj_name, session)
            except Exception as e:
                print(f"[Sync-Engine] Could not re-ingest parent project '{proj_name}': {e}")

    session.commit()

    # 4. Map and restore status indicators for each book
    for item in meta_items:
        book_name = item.get("book_name")
        book_path = item.get("book_path")
        source_type = item.get("source_type")

        if not book_name or not book_path:
            continue

        # Fetch the restored book database record
        book = session.exec(
            select(Book).where(Book.name == book_name).where(Book.path == book_path)
        ).first()

        if not book:
            continue

        if source_type == "output":
            # Book has a completed transcript on disk, perform audit to resolve correct step
            sync_book_from_disk(book.id, session)

        elif source_type == "temp":
            # Book transcription was in-progress, restore temp segments
            chapters = session.exec(
                select(Chapter).where(Chapter.book_id == book.id)
            ).all()
            working_dir = Path(item["working_dir"])
            for ch in chapters:
                ch_txt = working_dir / f"chapter_{ch.chapter_num}.txt"
                if ch_txt.exists():
                    ch.status = "Completed"
                    session.add(ch)
                else:
                    if ch.status in ("Transcribing", "Completed"):
                        ch.status = "Pending"
                        session.add(ch)

            completed_count = len([c for c in chapters if c.status == "Completed"])
            total_chapters = len(chapters)
            if total_chapters > 0:
                book.progress = completed_count / total_chapters
                
                # --- CORRECTION: Verify the merged transcript.txt actually exists on disk before declaring "Transcribed" ---
                project = session.get(Project, book.project_id) if book.project_id else None
                project_name = project.name if project else "Default_Project"
                base_output_dir = Path(get_setting("output_dir", "./output", session)).resolve()
                transcript_file = base_output_dir / project_name / book.name / "transcript.txt"

                if completed_count == total_chapters and transcript_file.exists():
                    book.status = "Transcribed"
                else:
                    book.status = "Imported"
                    
            session.add(book)
            session.commit()
            
            # Recalculate project overall status
            if book.project_id:
                sync_project_status(book.project_id, session)

    print("[Sync-Engine] Database state recovery sequence complete.")


def get_book_stats(project_name: str, book_name: str) -> dict:
    """Computes fast on-disk statistics for a book volume without database queries."""
    from ui import state
    stats = {
        "has_transcript": False,
        "char_count": 0,
        "word_count": 0,
        "total_prompts": 0,
        "approved_prompts": 0,
        "generated_images": 0,
        "estimated_scenes": 0
    }
    
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    book_dir = base_output_dir / project_name / book_name
    transcript_path = book_dir / "transcript.txt"
    prompts_path = book_dir / "prompts.csv"
    images_dir = book_dir / "images"
    
    # 1. Transcript Stats
    if transcript_path.exists():
        stats["has_transcript"] = True
        try:
            txt = transcript_path.read_text(encoding="utf-8", errors="ignore")
            stats["char_count"] = len(txt)
            stats["word_count"] = len(txt.split())
            
            # Apply dynamic custom chunk size setting
            chunk_size = getattr(state, "playground_chunk_size", 350)
            if not chunk_size or chunk_size <= 0:
                chunk_size = 350
            stats["estimated_scenes"] = max(1, stats["word_count"] // chunk_size)
        except Exception:
            pass
            
    # 2. Prompts CSV Stats
    if prompts_path.exists():
        try:
            with open(prompts_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='|')
                rows = list(reader)
                stats["total_prompts"] = len(rows)
                approved_count = 0
                for r in rows:
                    app_val = r.get("approved") or r.get("Approved") or "False"
                    if app_val.strip().lower() == "true":
                        approved_count += 1
                stats["approved_prompts"] = approved_count
        except Exception:
            pass
            
    # 3. Generated Images
    for d in [images_dir, book_dir]:
        if d.exists() and d.is_dir():
            try:
                png_count = len([f for f in os.listdir(d) if f.lower().endswith('.png')])
                if png_count > 0:
                    stats["generated_images"] = png_count
                    break
            except Exception:
                pass
                
    return stats


def get_book_stats_cached(project_name: str, book_name: str) -> dict:
    """Checks timestamps on disk before parsing, preventing I/O overhead on polling ticks."""
    from ui import state
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    book_dir = base_output_dir / project_name / book_name
    transcript_path = book_dir / "transcript.txt"
    prompts_path = book_dir / "prompts.csv"
    images_dir = book_dir / "images"
    
    # Build signature based on file modified times
    sig = ""
    for p in [transcript_path, prompts_path, images_dir]:
        if p.exists():
            sig += f"{p.name}:{p.stat().st_mtime}|"
            
    cache_key = f"{project_name}:{book_name}"
    if cache_key in state._stats_cache:
        cached_sig, cached_data = state._stats_cache[cache_key]
        if cached_sig == sig:
            return cached_data
            
    # Parse fresh and cache
    stats = get_book_stats(project_name, book_name)
    state._stats_cache[cache_key] = (sig, stats)
    return stats