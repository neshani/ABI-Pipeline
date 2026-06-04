import os
import ffmpeg
from pathlib import Path
from typing import Optional, Dict, Any, List
from sqlmodel import Session
from database.connection import engine
from database.models import Project, Book, Chapter

# Supported audio extensions
AUDIO_EXTENSIONS = {'.mp3', '.m4b', '.m4a', '.wav', '.flac', '.ogg'}

# Supported image extensions for cover art
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def find_cover_art(book_dir: Path) -> Optional[str]:
    """
    Scans a directory for book cover art based on priority:
    1. Named exactly 'cover' (e.g. cover.jpg)
    2. Filename contains 'cover' or 'folder'
    3. Exactly one image file exists in the directory
    """
    if not book_dir.is_dir():
        return None

    image_files: List[Path] = []
    for file in book_dir.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file)

    if not image_files:
        return None

    # Priority 1: Named exactly "cover" (ignoring extension)
    for img in image_files:
        if img.stem.lower() == "cover":
            return str(img.resolve())

    # Priority 2: Contains "cover" or "folder" in the filename
    for img in image_files:
        stem_lower = img.stem.lower()
        if "cover" in stem_lower or "folder" in stem_lower:
            return str(img.resolve())

    # Priority 3: Exactly one image file in the directory
    if len(image_files) == 1:
        return str(image_files[0].resolve())

    return None


def find_audio_sources(book_dir: Path) -> tuple[str, List[Path]]:
    """
    Scans a directory for audiobook files safely across Windows and UNIX environments.
    """
    if not book_dir.is_dir():
        return 'none', []

    # Get a single flat list of files in the directory
    all_files = [f for f in book_dir.iterdir() if f.is_file()]

    # 1. Look for single-file chapter containers (.m4b, .m4a)
    m4_files = sorted([
        f for f in all_files if f.suffix.lower() in {'.m4b', '.m4a'}
    ])
    if m4_files:
        return 'single_file', [m4_files[0]]

    # 2. Look for multi-file MP3 tracks
    mp3_files = sorted([
        f for f in all_files if f.suffix.lower() == '.mp3'
    ])
    if len(mp3_files) > 1:
        return 'multi_file', mp3_files
    elif len(mp3_files) == 1:
        return 'single_file', [mp3_files[0]]

    # 3. Fallback to other raw audio file formats
    fallback_files = sorted([
        f for f in all_files if f.suffix.lower() in {'.wav', '.flac', '.ogg'}
    ])
    if fallback_files:
        if len(fallback_files) > 1:
            return 'multi_file', fallback_files
        return 'single_file', [fallback_files[0]]

    return 'none', []


def scan_directory(path_str: str) -> Dict[str, Any]:
    """
    Analyzes a directory to determine if it is a single audiobook or a batch.
    Does not save to database. Returns a metadata preview of discovered books.
    """
    cleaned_path_str = path_str.strip().strip('"')
    if not cleaned_path_str:
        return {"type": "none", "project_name": "", "path": "", "books": []}

    path = Path(cleaned_path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Directory path does not exist: {path_str}")

    # 1. Check if the directory itself has audio files directly (Single Audiobook)
    source_type, audio_files = find_audio_sources(path)
    if source_type != 'none':
        return {
            "type": "single",
            "project_name": path.name,
            "path": str(path),
            "books": [{
                "name": path.name,
                "path": str(path),
                "cover_path": find_cover_art(path),
                "audio_type": source_type,
                "files": [str(f) for f in audio_files]
            }]
        }

    # 2. Check first-level subdirectories (Batch of Audiobooks)
    subdirs = sorted([d for d in path.iterdir() if d.is_dir()])
    discovered_books = []
    for subdir in subdirs:
        sub_source_type, sub_audio_files = find_audio_sources(subdir)
        if sub_source_type != 'none':
            discovered_books.append({
                "name": subdir.name,
                "path": str(subdir),
                "cover_path": find_cover_art(subdir),
                "audio_type": sub_source_type,
                "files": [str(f) for f in sub_audio_files]
            })

    if discovered_books:
        return {
            "type": "batch",
            "project_name": path.name,
            "path": str(path),
            "books": discovered_books
        }

    # 3. No supported audiobook structures found
    return {
        "type": "none",
        "project_name": path.name,
        "path": str(path),
        "books": []
    }


def create_chapter_plan_for_book(book_id: int, audio_type: str, files: List[str], session: Session) -> None:
    """
    Probes audiobook files to generate and insert the individual Chapter models.
    """
    if audio_type == 'multi_file':
        # Every file is treated as a separate chapter
        for i, file_path in enumerate(files):
            chapter = Chapter(
                book_id=book_id,
                chapter_num=i + 1,
                title=Path(file_path).stem,
                input_file=file_path,
                type='file',
                status="Pending"
            )
            session.add(chapter)

    elif audio_type == 'single_file' and files:
        single_file = files[0]
        try:
            # Probe single file metadata for chapter timestamps
            metadata = ffmpeg.probe(single_file, show_chapters=None)
            chapters_metadata = metadata.get('chapters', [])

            if chapters_metadata:
                for i, ch in enumerate(chapters_metadata):
                    chapter = Chapter(
                        book_id=book_id,
                        chapter_num=i + 1,
                        title=ch.get('tags', {}).get('title', f"Chapter {i+1}"),
                        input_file=single_file,
                        type='segment',
                        start_time=float(ch['start_time']),
                        end_time=float(ch['end_time']),
                        status="Pending"
                    )
                    session.add(chapter)
            else:
                # No chapters metadata found; treat the entire file as a single chapter
                duration = float(metadata['format']['duration'])
                chapter = Chapter(
                    book_id=book_id,
                    chapter_num=1,
                    title=Path(single_file).stem,
                    input_file=single_file,
                    type='segment',
                    start_time=0.0,
                    end_time=duration,
                    status="Pending"
                )
                session.add(chapter)
        except Exception as e:
            # Fallback if ffprobe isn't installed or file parsing fails
            chapter = Chapter(
                book_id=book_id,
                chapter_num=1,
                title=Path(single_file).stem,
                input_file=single_file,
                type='segment',
                start_time=0.0,
                end_time=None,
                status="Pending"
            )
            session.add(chapter)


def ingest_project(scan_result: Dict[str, Any], custom_project_name: str, session: Optional[Session] = None) -> int:
    """
    Takes the scan metadata preview and inserts the Project, Books, and Chapter
    entities into the database within a single transaction.
    """
    # Use parent active session if provided, otherwise open a standalone session
    active_session = session if session is not None else Session(engine)
    try:
        # 1. Create and add Project record
        project = Project(
            name=custom_project_name,
            path=scan_result["path"],
            is_batch=(scan_result["type"] == "batch"),
            status="Imported"
        )
        active_session.add(project)
        active_session.flush()  # Populates project.id before committing

        # 2. Create Books and associated Chapter lists
        for book_data in scan_result["books"]:
            book = Book(
                project_id=project.id,
                name=book_data["name"],
                path=book_data["path"],
                cover_path=book_data["cover_path"],
                status="Imported",
                progress=0.0
            )
            active_session.add(book)
            active_session.flush()  # Populates book.id

            # 3. Create the chapters
            create_chapter_plan_for_book(
                book_id=book.id,
                audio_type=book_data["audio_type"],
                files=book_data["files"],
                session=active_session
            )

            # 4. Instant scan for any pre-existing output files on disk
            from services.sync_engine import sync_book_from_disk
            sync_book_from_disk(book.id, active_session)

        # Only commit here if we opened a standalone session locally
        if session is None:
            active_session.commit()
            
        return project.id
    except Exception as e:
        if session is None:
            active_session.rollback()
        raise e
    finally:
        if session is None:
            active_session.close()