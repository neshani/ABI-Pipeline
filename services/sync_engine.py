# services/sync_engine.py
import os
import json
import csv
from pathlib import Path
from typing import Optional
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


def prune_stale_database_records(session: Session) -> None:
    """
    Cleans up the database by removing Projects and Books that have been physically
    deleted from the output folder, keeping the index completely aligned with disk.
    """
    output_dir = Path(get_setting("output_dir", "./output", session)).resolve()
    if not output_dir.exists() or not output_dir.is_dir():
        return

    # 1. Fetch all projects in database
    db_projects = session.exec(select(Project)).all()
    for proj in db_projects:
        # Ignore special namespaces
        if proj.name.startswith('_') or proj.name.startswith('.'):
            continue

        proj_output_dir = output_dir / proj.name
        
        # If the project output directory does not exist on disk, prune it from DB
        if not proj_output_dir.exists():
            print(f"[Sync-Engine] Pruning stale project from database (deleted on disk): '{proj.name}'")
            
            # Delete child books, chapters, and prompts
            books = session.exec(select(Book).where(Book.project_id == proj.id)).all()
            for b in books:
                session.exec(delete(Chapter).where(Chapter.book_id == b.id))
                session.exec(delete(ScenePrompt).where(ScenePrompt.book_id == b.id))
                session.delete(b)
                
            session.delete(proj)
            session.flush()


def ensure_book_chapters_populated(book: Book, book_dir: Path, source_path_str: Optional[str], session: Session) -> None:
    """
    Ensures that Chapter records for a given Book are populated and match the current physical audio sources.
    If physical audio formats or counts change on disk (e.g. converted from split MP3s to a single M4B),
    automatically rebuilds the chapter plan in SQLite to stay aligned.
    """
    from services.scanner import find_audio_sources, create_chapter_plan_for_book
    from sqlmodel import delete

    audio_type = 'none'
    audio_files = []

    # 1. Attempt scanning original source directory first (most accurate timestamps)
    if source_path_str:
        src_path = Path(source_path_str)
        if src_path.exists():
            audio_type, audio_files = find_audio_sources(src_path)

    # 2. Fallback to scanning compiled output folder directly
    if audio_type == 'none' and book_dir.exists():
        audio_type, audio_files = find_audio_sources(book_dir)

    # 3. Query existing database chapter status
    existing_chapters = session.exec(
        select(Chapter).where(Chapter.book_id == book.id).order_by(Chapter.chapter_num)
    ).all()
    existing_ch_count = len(existing_chapters)

    # Self-healing: if the physical files on disk have changed structure, clear DB chapters to force rebuild
    needs_rebuild = False
    if existing_ch_count > 0:
        db_types = {ch.type for ch in existing_chapters}
        if audio_type == 'single_file' and 'file' in db_types:
            needs_rebuild = True
        elif audio_type == 'multi_file' and 'segment' in db_types:
            needs_rebuild = True
        elif audio_type == 'multi_file' and existing_ch_count != len(audio_files):
            needs_rebuild = True

    if existing_ch_count == 0 or needs_rebuild:
        if existing_ch_count > 0:
            print(f"[Sync-Engine] Audio files changed on disk for '{book.name}'. Wiping database chapters to rebuild cache...")
            session.exec(delete(Chapter).where(Chapter.book_id == book.id))
            session.flush()

        if audio_type != 'none' and audio_files:
            try:
                create_chapter_plan_for_book(
                    book_id=book.id,
                    audio_type=audio_type,
                    files=[str(f) for f in audio_files],
                    session=session
                )
                session.flush()
                print(f"[Sync-Engine] Successfully parsed chapters from audio files for '{book.name}'")
                return
            except Exception as e:
                print(f"[Sync-Engine] Error probing audio tracks during recovery: {e}")

    # 4. Last fallback: Split raw transcript.txt if present (e.g. text or EPUB books)
    if existing_ch_count == 0:
        transcript_file = book_dir / "transcript.txt"
        if transcript_file.exists():
            try:
                with open(transcript_file, "r", encoding="utf-8") as f:
                    content = f.read()
                sections = content.split("==CHAPTER==")
                cleaned_sections = [s.strip() for s in sections if s.strip()]

                for idx in range(len(cleaned_sections)):
                    ch_num = idx + 1
                    new_ch = Chapter(
                        book_id=book.id,
                        chapter_num=ch_num,
                        title=f"Chapter {ch_num}",
                        status="Completed",
                        word_count=len(cleaned_sections[idx].split()),
                        total_images=0,
                        completed_images=0
                    )
                    session.add(new_ch)
                session.flush()
                print(f"[Sync-Engine] Reconstructed chapters for '{book.name}' by splitting transcript.txt")
            except Exception as e:
                print(f"[Sync-Engine] Error reconstructing chapters from transcript: {e}")


def heal_project_book_orders(project_id: int, session: Session) -> None:
    """
    Ensures all books within a project have a non-None, unique, sequential book_order.
    Preserves any existing orders, filling in missing or duplicate values alphabetically.
    """
    books = session.exec(
        select(Book).where(Book.project_id == project_id)
    ).all()
    
    # Sort first by whether they have an order, then by their order, then alphabetically by name.
    # This preserves existing custom orders and appends unordered ones to the end alphabetically.
    sorted_books = sorted(
        books, 
        key=lambda b: (b.book_order is None, b.book_order if b.book_order is not None else 0, b.name.lower())
    )
    
    for idx, b in enumerate(sorted_books):
        b.book_order = idx
        session.add(b)
    session.flush()


def recover_from_temp_workspaces(session: Session) -> None:
    """
    Scans both workspace_temp/ and output/ folders for transcription tracking metadata.
    Reconstructs the complete database index on database wipe. Includes self-healing 
    cleanup for deleted projects and automatic metadata.json healing to align with disk split.
    """
    # Run a safe database cleanup first to synchronize DB records with disk deletions
    prune_stale_database_records(session)

    meta_items = []

    # 1. Gather any incomplete/active workspace recovery configurations from workspace_temp
    temp_dir = Path("./workspace_temp")
    if temp_dir.exists() and temp_dir.is_dir():
        for working_dir in temp_dir.iterdir():
            if working_dir.is_dir() and working_dir.name.startswith("book_"):
                state_file = working_dir / "transcription_state.json"
                if state_file.exists():
                    try:
                        with open(state_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            
                        proj_name = data.get("project_name")
                        if proj_name:
                            # Skip and delete immediately if the project name points to an internal/special folder
                            if proj_name.startswith('_') or proj_name.startswith('.'):
                                import shutil
                                shutil.rmtree(working_dir, ignore_errors=True)
                                print(f"[Sync-Engine] Cleaned up internal/invalid temp workspace: {working_dir.name}")
                                continue

                            db_proj = session.exec(
                                select(Project).where(Project.name == proj_name)
                            ).first()
                            
                            output_dir = Path(get_setting("output_dir", "./output", session)).resolve()
                            proj_output_dir = output_dir / proj_name
                            
                            if not db_proj and not proj_output_dir.exists():
                                import shutil
                                shutil.rmtree(working_dir, ignore_errors=True)
                                print(f"[Sync-Engine] Cleaned up stale temp workspace for deleted project '{proj_name}': {working_dir.name}")
                                continue

                        data["source_type"] = "temp"
                        data["working_dir"] = str(working_dir)
                        meta_items.append(data)
                    except Exception as e:
                        print(f"Error reading temp state file {state_file}: {e}")

    # 2. Gather any fully completed output configurations from output/ and heal metadata.json
    output_dir = Path(get_setting("output_dir", "./output", session)).resolve()
    if output_dir.exists() and output_dir.is_dir():
        for meta_file in output_dir.glob("*/*/metadata.json"):
            # Skip if any parent folder starts with '_' or '.' (like _lora_library)
            parts = meta_file.relative_to(output_dir).parts
            if any(p.startswith('_') or p.startswith('.') for p in parts):
                continue

            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # --- METADATA FILE HEALER ---
                # Detect if the project was split or renamed on disk (parent folder name differs from metadata)
                physical_proj_name = meta_file.parent.parent.name
                metadata_proj_name = data.get("project_name", "")
                
                if metadata_proj_name != physical_proj_name:
                    data["project_name"] = physical_proj_name
                    # Save the healed metadata.json back to disk
                    with open(meta_file, "w", encoding="utf-8") as out_f:
                        json.dump(data, out_f, indent=4)
                    print(f"[Sync-Engine] Healed stale project_name in '{meta_file.parent.name}' metadata: '{metadata_proj_name}' -> '{physical_proj_name}'")
                
                # Double-check namespace check on the healed project name
                proj_name = data.get("project_name")
                if proj_name and (proj_name.startswith('_') or proj_name.startswith('.')):
                    continue

                data["source_type"] = "output"
                data["book_output_dir"] = str(meta_file.parent)
                meta_items.append(data)
            except Exception as e:
                print(f"Error reading output metadata file {meta_file}: {e}")

    if not meta_items:
        return

    print(f"[Sync-Engine] Found {len(meta_items)} project tracking records to synchronize.")

    # 3. Direct Reconstruction of parent Projects and Books (No deep source folder re-scanning)
    for item in meta_items:
        proj_name = item.get("project_name")
        proj_path = item.get("project_path")
        book_name = item.get("book_name")
        book_path = item.get("book_path")
        source_type = item.get("source_type")

        if not proj_name or not book_name:
            continue

        # Find or create Project directly in database
        project = session.exec(
            select(Project).where(Project.name == proj_name)
        ).first()

        if not project:
            p_path = proj_path if proj_path else str(output_dir / proj_name)
            project = Project(
                name=proj_name,
                path=p_path,
                is_batch=True,
                status="Imported"
            )
            session.add(project)
            session.flush()
            print(f"[Sync-Engine] Reconstructed database Project record: '{proj_name}'")

        # Find or create Book directly in database
        book = session.exec(
            select(Book).where(Book.name == book_name).where(Book.project_id == project.id)
        ).first()

        if not book:
            # Detect cover image dynamically using scanner find_cover_art fallback pipeline
            from services.scanner import find_cover_art
            book_output_dir = Path(item["book_output_dir"]) if source_type == "output" else Path(item["working_dir"])
            cover_path = None
            if book_path:
                cover_path = find_cover_art(Path(book_path))
            if not cover_path:
                cover_path = find_cover_art(book_output_dir)

            b_path = book_path if book_path else str(book_output_dir)
            book = Book(
                project_id=project.id,
                name=book_name,
                path=b_path,
                status="Imported",
                progress=0.0
            )
            if cover_path:
                book.cover_path = cover_path
            session.add(book)
            session.flush()
            print(f"[Sync-Engine] Reconstructed database Book record: '{book_name}'")

    session.commit()

    # 4. Map and restore status indicators and chapters for each book
    for item in meta_items:
        proj_name = item.get("project_name")
        book_name = item.get("book_name")
        source_type = item.get("source_type")

        if not proj_name or not book_name:
            continue

        # Fetch the restored book database record
        project = session.exec(select(Project).where(Project.name == proj_name)).first()
        if not project:
            continue
            
        book = session.exec(
            select(Book).where(Book.name == book_name).where(Book.project_id == project.id)
        ).first()

        if not book:
            continue

        if source_type == "output":
            # Ensure Chapter records are fully populated (probes audio or parses transcript)
            ensure_book_chapters_populated(book, Path(item["book_output_dir"]), item.get("book_path"), session)

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
                
                # Verify the merged transcript.txt actually exists on disk before declaring "Transcribed"
                transcript_file = output_dir / proj_name / book.name / "transcript.txt"

                if completed_count == total_chapters and transcript_file.exists():
                    book.status = "Transcribed"
                else:
                    book.status = "Imported"
                    
            session.add(book)
            session.commit()
            
            # Recalculate project overall status
            if book.project_id:
                sync_project_status(book.project_id, session)

    # 1. Commit all recovered projects, books, and audio scans to disk safely first
    session.commit()

    # 2. Run self-healing pass to clean and re-index project book ordering sequence gaps/duplicates
    recovered_project_ids = set()
    for item in meta_items:
        proj_name = item.get("project_name")
        if proj_name:
            proj = session.exec(select(Project).where(Project.name == proj_name)).first()
            if proj:
                recovered_project_ids.add(proj.id)

    for p_id in recovered_project_ids:
        try:
            heal_project_book_orders(p_id, session)
        except Exception as oe:
            # Cleanly un-poison the outer session transaction if a specific project fails
            session.rollback()
            print(f"[Sync-Engine] Error healing book orders during recovery for project ID {p_id}: {oe}")

    # Commit all successfully healed book orders and release the SQLite write lock!
    session.commit()

    # 3. Re-scan and synchronize characters.json for each active project in a separate isolated transaction
    for p_id in recovered_project_ids:
        try:
            from services.character_manager import sync_project_characters_from_json
            # Calling with session=None allows the function to handle its own clean, non-deadlocking transaction
            sync_project_characters_from_json(p_id)
        except Exception as ce:
            print(f"[Sync-Engine] Error syncing characters during recovery for project ID {p_id}: {ce}")

    print("[Sync-Engine] Database state recovery sequence complete.")


def reconcile_database_with_output_folder(session: Session) -> dict:
    """
    Scans the local output directory to discover projects, books, and chapters 
    reorganized or split on disk. Reconstructs missing database records 
    and resynchronizes existing ones with the disk state. Ignores internal 
    and library directories starting with '.' or '_'.
    """
    stats = {
        "projects_discovered": 0,
        "books_discovered": 0,
        "new_projects_created": 0,
        "new_books_created": 0,
        "synced_books_count": 0
    }

    base_output_dir = Path(get_setting("output_dir", "./output", session)).resolve()
    if not base_output_dir.exists() or not base_output_dir.is_dir():
        return stats

    from services.scanner import find_cover_art

    reconciled_project_ids = []

    # Iterate through project subdirectories in the output folder
    for proj_dir in base_output_dir.iterdir():
        # Exclude directories starting with '_' (like _lora_library) or '.'
        if not proj_dir.is_dir() or proj_dir.name.startswith('.') or proj_dir.name.startswith('_'):
            continue

        stats["projects_discovered"] += 1

        # Check if the project already exists in the database
        project = session.exec(
            select(Project).where(Project.name == proj_dir.name)
        ).first()

        if not project:
            project = Project(
                name=proj_dir.name,
                path=str(proj_dir.resolve()),
                is_batch=True,
                status="Imported"
            )
            session.add(project)
            session.flush()
            stats["new_projects_created"] += 1

        reconciled_project_ids.append(project.id)

        # Collect, filter, and sort book subdirectories alphabetically to guarantee consistent order indexing
        subdirs = [d for d in proj_dir.iterdir() if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('_')]
        book_subdirs = [d for d in subdirs if d.name.lower() != "images"]
        book_subdirs.sort(key=lambda d: d.name.lower())

        for idx, book_dir in enumerate(book_subdirs):
            stats["books_discovered"] += 1

            # Resolve cover path and original book source path
            metadata_file = book_dir / "metadata.json"
            book_path_src = None
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Heal metadata project name on the fly if needed
                    if data.get("project_name") != proj_dir.name:
                        data["project_name"] = proj_dir.name
                        with open(metadata_file, "w", encoding="utf-8") as out_f:
                            json.dump(data, out_f, indent=4)
                        print(f"[Sync-Engine] Healed stale project_name in '{book_dir.name}' metadata.")
                    
                    book_path_src = data.get("book_path")
                except Exception as e:
                    print(f"Error reading metadata.json in {book_dir.name}: {e}")

            # Check if this book is registered in the database for the active project
            book = session.exec(
                select(Book).where(Book.name == book_dir.name).where(Book.project_id == project.id)
            ).first()

            # Dynamic Cover Resolve
            cover_path = None
            if book_path_src:
                cover_path = find_cover_art(Path(book_path_src))
            if not cover_path:
                cover_path = find_cover_art(book_dir)

            if not book:
                book = Book(
                    project_id=project.id,
                    name=book_dir.name,
                    path=book_path_src if book_path_src else str(book_dir.resolve()),
                    status="Imported",
                    progress=0.0
                )
                if cover_path:
                    book.cover_path = cover_path
                session.add(book)
                session.flush()
                stats["new_books_created"] += 1
            else:
                # Update missing details on existing books safely
                if cover_path and not book.cover_path:
                    book.cover_path = cover_path
                session.add(book)

            # Ensure Chapter records are fully populated (probes audio or parses transcript)
            ensure_book_chapters_populated(book, book_dir, book_path_src, session)

            # Re-index all properties, word counts, and rendering counts from files on disk
            try:
                sync_book_from_disk(book.id, session)
                stats["synced_books_count"] += 1
            except Exception as e:
                print(f"[Sync-Engine] Error syncing book '{book.name}' from disk: {e}")

    # 1. Commit all reconciled projects, books, and audio scans to disk safely first
    session.commit()

    # 2. Run self-healing pass on all reconciled projects to clean up sequential book orders
    for p_id in reconciled_project_ids:
        try:
            heal_project_book_orders(p_id, session)
        except Exception as oe:
            # Cleanly un-poison the outer session transaction if a specific project fails
            session.rollback()
            print(f"[Sync-Engine] Error healing book orders during reconciliation for project ID {p_id}: {oe}")

    # Commit all successfully healed book orders and release the SQLite write lock!
    session.commit()

    # 3. Run character syncing in fresh, isolated, non-overlapping transactions
    for p_id in reconciled_project_ids:
        try:
            from services.character_manager import sync_project_characters_from_json
            # Calling with session=None allows the function to handle its own clean, non-deadlocking transaction
            sync_project_characters_from_json(p_id)
        except Exception as ce:
            print(f"[Sync-Engine] Error syncing characters for project ID {p_id}: {ce}")

    return stats


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
    
    # Establish a safe local session to guarantee correct resolution of the output directory
    with Session(engine) as session:
        base_output_dir = Path(get_setting("output_dir", "./output", session)).resolve()
        
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
                # Count both png and webp assets
                img_count = len([
                    f for f in os.listdir(d) 
                    if f.lower().endswith('.png') or f.lower().endswith('.webp')
                ])
                if img_count > 0:
                    stats["generated_images"] = img_count
                    break
            except Exception:
                pass
                
    return stats


def get_book_stats_cached(project_name: str, book_name: str) -> dict:
    """Checks timestamps on disk before parsing, preventing I/O overhead on polling ticks."""
    from ui import state
    
    # Establish a safe local session to guarantee correct resolution of the output directory
    with Session(engine) as session:
        base_output_dir = Path(get_setting("output_dir", "./output", session)).resolve()
        
    book_dir = base_output_dir / project_name / book_name
    transcript_path = book_dir / "transcript.txt"
    prompts_path = book_dir / "prompts.csv"
    images_dir = book_dir / "images"
    
    # Build signature based on file modified times for text files
    sig = ""
    for p in [transcript_path, prompts_path]:
        if p.exists():
            sig += f"{p.name}:{p.stat().st_mtime}|"
            
    # Invalidate cache on directory file count changes instead of folder mtime (extremely reliable on Windows)
    for d in [images_dir, book_dir]:
        if d.exists() and d.is_dir():
            try:
                file_count = len(os.listdir(d))
                sig += f"{d.name}_count:{file_count}|"
            except Exception:
                pass
            
    cache_key = f"{project_name}:{book_name}"
    if cache_key in state._stats_cache:
        cached_sig, cached_data = state._stats_cache[cache_key]
        if cached_sig == sig:
            return cached_data
            
    # Parse fresh and cache
    stats = get_book_stats(project_name, book_name)
    state._stats_cache[cache_key] = (sig, stats)
    return stats