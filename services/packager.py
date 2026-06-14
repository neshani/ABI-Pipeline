import os
import sys
import csv
import json
import shutil
import zipfile
import subprocess
from pathlib import Path
from PIL import Image
from typing import Optional, List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor

# --- Ken Burns Animation Presets ---
PORTRAIT_PRESETS = [
    # 21: MS Pan Left
    [{"position": 0.0, "scale": 1.85, "pan_x": 0.6986343885785224, "pan_y": 0.4998313226999163},
     {"position": 1.0, "scale": 1.85, "pan_x": 0.28978978978979003, "pan_y": 0.4939939939939939}],
    # 22: MS Pan Right
    [{"position": 0.0, "scale": 1.85, "pan_x": 0.2897897897897898, "pan_y": 0.506006006006006},
     {"position": 1.0, "scale": 1.85, "pan_x": 0.7102102102102098, "pan_y": 0.493993993993994}],
    # 23: MS Zoom Pan Down
    [{"position": 0.0, "scale": 1.3100000000000005, "pan_x": 0.43214588634435946, "pan_y": 0.5},
     {"position": 0.5, "scale": 1.9050000000000007, "pan_x": 0.30077277003136527, "pan_y": 0.4890182942638114},
     {"position": 1.0, "scale": 2.229999999999998, "pan_x": 0.7192326856003992, "pan_y": 0.5978204192303685}],
    # 24: MS Zoom Pan Up
    [{"position": 0.0, "scale": 1.16, "pan_x": 0.5, "pan_y": 0.5},
     {"position": 0.5, "scale": 2.119999999999999, "pan_x": 0.6939203354297697, "pan_y": 0.5419287211740037},
     {"position": 1.0, "scale": 2.179999999999999, "pan_x": 0.27573904179408765, "pan_y": 0.443934760448522}],
    # 25: MS Zoom Out
    [{"position": 0.0, "scale": 2.25, "pan_x": 0.5000000000000009, "pan_y": 0.4308641975308644},
     {"position": 1.0, "scale": 1.24, "pan_x": 0.5, "pan_y": 0.5}]
]

LANDSCAPE_PRESETS = [
    # 26: DS Pan Up
    [{"position": 0.0, "scale": 1.85, "pan_x": 0.4966216216216216, "pan_y": 0.7162162162162162},
     {"position": 1.0, "scale": 1.85, "pan_x": 0.5, "pan_y": 0.31081081081081047}],
    # 27: DS Pan Down
    [{"position": 0.0, "scale": 1.85, "pan_x": 0.4932432432432434, "pan_y": 0.28716216216216206},
     {"position": 1.0, "scale": 1.85, "pan_x": 0.4999999999999999, "pan_y": 0.7128378378378378}],
    # 28: DS Zoom In pan down
    [{"position": 0.0, "scale": 1.2, "pan_x": 0.5, "pan_y": 0.5},
     {"position": 0.5, "scale": 2.0, "pan_x": 0.5, "pan_y": 0.3187500000000005},
     {"position": 1.0, "scale": 1.8, "pan_x": 0.5, "pan_y": 0.7}],
    # 29: DS Zoom In Pan Up
    [{"position": 0.0, "scale": 1.2, "pan_x": 0.5, "pan_y": 0.5},
     {"position": 0.5, "scale": 2.0, "pan_x": 0.496875, "pan_y": 0.7062500000000003},
     {"position": 1.0, "scale": 1.8, "pan_x": 0.5, "pan_y": 0.31944444444444475}],
    # 30: DS Zoom Out
    [{"position": 0.0, "scale": 2.0, "pan_x": 0.5031250000000002, "pan_y": 0.30625000000000013},
     {"position": 1.0, "scale": 1.2, "pan_x": 0.5052083333333335, "pan_y": 0.5416666666666671}]
]


# --- Timing Formatting Utilities ---

def parse_timestamp_to_seconds(ts_str: str) -> float:
    """Converts 'HH:MM:SS' or 'HH:MM:SS.ss' into float seconds."""
    try:
        parts = ts_str.strip().split(':')
        if len(parts) < 3:
            return 0.0
        h = int(parts[0])
        m = int(parts[1])
        s_parts = parts[2].split('.')
        s = int(s_parts[0])
        ms = int(s_parts[1]) if len(s_parts) > 1 else 0
        frac = ms / (10 ** len(s_parts[1])) if len(s_parts) > 1 else 0.0
        return h * 3600 + m * 60 + s + frac
    except Exception:
        return 0.0


def format_seconds_to_timestamp(seconds: float) -> str:
    """Converts float seconds to compliant 'HH:MM:SS.ss' format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 100))
    if ms >= 100:
        ms = 99  # prevent layout overflow rounding up to next second
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:02d}"


# --- Audio Infrastructure Search Utilities ---

def find_audiobook_file(project_name: str, book_name: str) -> Optional[Path]:
    """Finds representative original audiobook file using SQLite database references."""
    from database.connection import engine
    from database.models import Project, Book
    from sqlmodel import Session, select
    
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name == project_name)).first()
        if not project:
            return None
        book = session.exec(select(Book).where(Book.project_id == project.id, Book.name == book_name)).first()
        if not book or not book.path:
            return None
            
        book_dir = Path(book.path)
        if not book_dir.exists():
            return None
            
        extensions = [".m4b", ".mp3", ".m4a", ".wav", ".aac"]
        for ext in extensions:
            for f in book_dir.glob(f"*{ext}"):
                if f.is_file():
                    return f
            for f in book_dir.glob(f"*{ext.upper()}"):
                if f.is_file():
                    return f
    return None


def get_audio_duration_seconds(audio_path: Path) -> float:
    """Uses FFprobe to query exact file duration or returns default estimation."""
    try:
        cmd = [
            "ffprobe", "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            str(audio_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 10800.0


def get_audio_file_metadata(audio_path: Path) -> Dict[str, str]:
    """Reads metadata tags (album/title, artist/author) from an audio file using FFprobe."""
    import json
    meta = {"title": "", "artist": ""}
    try:
        cmd = [
            "ffprobe", "-v", "quiet", 
            "-show_entries", "format_tags=album,title,artist,album_artist,author", 
            "-print_format", "json", 
            str(audio_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})
        
        lower_tags = {k.lower(): v for k, v in tags.items()}
        # Priority: 'album' tag first for book title on MP3/M4Bs, falling back to 'title'
        meta["title"] = lower_tags.get("album") or lower_tags.get("title") or ""
        meta["artist"] = lower_tags.get("artist") or lower_tags.get("album_artist") or lower_tags.get("author") or ""
    except Exception:
        pass
    return meta


def get_book_total_duration(project_name: str, book_name: str) -> float:
    """Calculates timing duration directly from database chapters, falling back to disk scans if needed."""
    from database.connection import engine
    from database.models import Project, Book, Chapter
    from sqlmodel import Session, select
    import re
    
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.name == project_name)).first()
        if not project:
            return 10800.0
        book = session.exec(select(Book).where(Book.project_id == project.id, Book.name == book_name)).first()
        if not book:
            return 10800.0
            
        # Priority 0: Instant DB Book Cache Check
        if getattr(book, "duration", None) is not None and book.duration > 0:
            return book.duration
            
        # Priority 1: Instant Database Lookup (0.1ms)
        chapters = session.exec(select(Chapter).where(Chapter.book_id == book.id)).all()
        if chapters:
            file_chapters = [c for c in chapters if c.type == 'file']
            if file_chapters:
                total_db_dur = 0.0
                for c in file_chapters:
                    if c.end_time is not None and c.start_time is not None:
                        total_db_dur += (c.end_time - c.start_time)
                    elif c.end_time is not None:
                        total_db_dur += c.end_time
                if total_db_dur > 0:
                    try:
                        book.duration = total_db_dur
                        session.add(book)
                        session.commit()
                    except Exception:
                        pass
                    return total_db_dur
            
            end_times = [c.end_time for c in chapters if c.end_time is not None]
            if end_times:
                max_end = max(end_times)
                if max_end > 0:
                    try:
                        book.duration = max_end
                        session.add(book)
                        session.commit()
                    except Exception:
                        pass
                    return max_end

        # Priority 2: Fallback Disk Scanner (Heavy Subprocess)
        if not book.path:
            return 10800.0
            
        book_dir = Path(book.path)
        if not book_dir.exists():
            return 10800.0
            
        extensions = [".m4b", ".mp3", ".m4a", ".wav", ".aac"]
        audio_files = []
        for ext in extensions:
            audio_files.extend(list(book_dir.glob(f"*{ext}")))
            audio_files.extend(list(book_dir.glob(f"*{ext.upper()}")))
            
        audio_files = list(set(audio_files))
        
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
            
        audio_files.sort(key=natural_sort_key)
        
        if not audio_files:
            return 10800.0
            
        total_dur = 0.0
        for f in audio_files:
            total_dur += get_audio_duration_seconds(f)
            
        # Cache calculated duration on the Book row to bypass disk scans on subsequent loads
        try:
            book.duration = total_dur
            session.add(book)
            session.commit()
        except Exception:
            pass
            
        return total_dur


# --- Multithreaded Image Processor ---

def process_and_compress_image(src_path: Path, dest_path: Path, max_dim: int, quality: int) -> bool:
    """Downscales and compresses a single PNG image into a highly-optimized WebP asset."""
    try:
        with Image.open(src_path) as img:
            # Convert to RGB mode if transparency exists to avoid compression failures
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            if max_dim > 0 and (img.width > max_dim or img.height > max_dim):
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest_path, "WEBP", quality=quality)
        return True
    except Exception:
        return False


# --- Local Timing CSV Utilities to prevent importing main.py ---

def find_prompts_csv(project_name: str, book_name: str) -> Optional[Path]:
    """Locates the prompts.csv file inside the output directory without calling main.py."""
    from database.connection import get_setting
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    csv_path = base_output_dir / project_name / book_name / "prompts.csv"
    if csv_path.exists():
        return csv_path
    return None


def read_prompts_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Reads and parses the pipe-delimited prompts.csv, normalizing header keys to lowercase."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        if not reader.fieldnames:
            return []
        reader.fieldnames = [name.strip().lower() if name else "" for name in reader.fieldnames]
        for row in reader:
            cleaned_row = {k: v.strip() if v else "" for k, v in row.items() if k}
            rows.append(cleaned_row)
    return rows


# --- Master Packaging Executor Engine ---

def archive_existing_pack(file_path: Path):
    """Checks if an illuminations.zip file already exists, and if so renames it with a timestamp."""
    import datetime
    if file_path.exists() and file_path.is_file():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive_name = f"Old_Illuminations_{timestamp}.zip"
        archive_path = file_path.parent / archive_name
        try:
            file_path.rename(archive_path)
        except Exception:
            pass


def build_illumination_pack(
    project_name: str,
    book_name: str,
    metadata: Dict[str, Any],
    on_progress: Optional[Callable[[float, str], None]] = None
) -> str:
    """Generates an OIS 1.3 compliant zip containing static, mobile and desktop variations."""
    def log(progress: float, msg: str):
        if on_progress:
            on_progress(progress, msg)

    log(0.05, "Locating audiobook timing CSV and workspace paths...")
    
    # Locate prompts CSV using the local helper logic
    csv_path = find_prompts_csv(project_name, book_name)
    if not csv_path:
        raise FileNotFoundError(f"Timing 'prompts.csv' not found for volume: {book_name}")

    # Read records from CSV
    try:
        all_rows = read_prompts_from_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to parse timing CSV data: {str(e)}")

    exclude_unapproved = metadata.get("exclude_unapproved", False)

    # Filter approved or fallback rows with valid timestamps
    scenes = []
    for row in all_rows:
        approved_val = str(row.get("approved", "")).strip().lower()
        timestamp_val = str(row.get("timestamp", "")).strip()
        
        if exclude_unapproved and approved_val not in ("true", "1", "yes", "approved"):
            continue
            
        # Build packaging scene targets if timestamps are valid
        if timestamp_val and timestamp_val != "0" and timestamp_val != "00:00:00":
            scenes.append(row)

    # Sort scenes chronologically by starting timestamp
    scenes.sort(key=lambda s: parse_timestamp_to_seconds(s.get("timestamp", "00:00:00")))

    if not scenes:
        raise ValueError("No valid timing entries discovered inside prompts.csv matching filter settings. Package cannot be generated.")

    # Locate database paths
    from database.connection import engine
    from database.models import Project, Book
    from sqlmodel import Session, select
    
    original_book_path = None
    db_cover_path = None
    
    with Session(engine) as session:
        project_db = session.exec(select(Project).where(Project.name == project_name)).first()
        if project_db:
            book_db = session.exec(select(Book).where(Book.project_id == project_db.id, Book.name == book_name)).first()
            if book_db:
                if book_db.path:
                    original_book_path = Path(book_db.path)
                if book_db.cover_path:
                    db_cover_path = Path(book_db.cover_path)

    # Get Natural Cumulative Duration
    duration = get_book_total_duration(project_name, book_name)
    log(0.15, f"Audiobook timing synchronized. Total book duration: {duration:.2f} seconds.")

    # Create safe workspace environments
    staging_dir = Path(f"./output/{project_name}/{book_name}_pack_staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    image_src_dir = Path(f"./output/{project_name}/{book_name}/images")
    if not image_src_dir.exists():
        image_src_dir = Path(f"./output/{project_name}/{book_name}")

    # Index existing rendered image assets
    log(0.20, "Pre-indexing rendered image frames...")
    rendered_images = {}
    if image_src_dir.exists():
        for f in image_src_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".png", ".webp"):
                stem = f.stem
                parts = stem.split('_')
                if len(parts) >= 2:
                    try:
                        ch = int(parts[0])
                        sc = int(parts[1])
                        rendered_images[(ch, sc)] = f
                    except ValueError:
                        pass

    # Cover injection handling
    cover_dest_filename = None
    first_scene_start_sec = parse_timestamp_to_seconds(scenes[0].get("timestamp", "00:00:00"))
    
    if metadata.get("use_cover", True) and first_scene_start_sec > 0:
        cover_candidates = []
        if db_cover_path:
            cover_candidates.append(db_cover_path)
        if original_book_path:
            cover_candidates.extend([
                original_book_path / "cover.jpg",
                original_book_path / "cover.png",
                original_book_path / "cover.webp",
                original_book_path / "cover_art.png"
            ])
        cover_candidates.extend([
            Path(f"./output/{project_name}/{book_name}/cover.png"),
            Path(f"./output/{project_name}/{book_name}/cover.jpg")
        ])

        found_cover = None
        for cp in cover_candidates:
            if cp.exists() and cp.is_file():
                found_cover = cp
                break
        
        if found_cover:
            log(0.25, f"Cover artwork detected: '{found_cover.name}'. Copying to zip root...")
            cover_dest_filename = "0000_cover.webp"
            process_and_compress_image(
                found_cover, 
                staging_dir / cover_dest_filename, 
                metadata.get("max_dimension", 1024), 
                metadata.get("webp_quality", 85)
            )
        else:
            log(0.25, "Cover injection requested but no source cover image was located on disk. Skipping.")

    # --- Compress scenes to WebP in a parallel process pool ---
    log(0.30, "Compressing, optimizing, and downscaling imagery in parallel processes...")
    compression_tasks = []
    scene_image_mappings = {}

    for idx, scene in enumerate(scenes):
        ch_str = scene.get("chapter", "1")
        sc_str = scene.get("scene", str(idx + 1))
        
        try:
            ch = int(float(ch_str))
        except ValueError:
            ch = 1
        try:
            sc = int(float(sc_str))
        except ValueError:
            sc = idx + 1

        src_img = rendered_images.get((ch, sc))
        if src_img and src_img.exists():
            # Build short sequential name to reduce zip layout footprint
            slug = src_img.stem.replace(f"{ch:02d}_{sc:02d}_", "")
            dest_filename = f"{ch:02d}_{sc:02d}_{slug}.webp"
            
            scene_image_mappings[idx] = dest_filename
            
            compression_tasks.append((
                src_img, 
                staging_dir / dest_filename, 
                metadata.get("max_dimension", 1024), 
                metadata.get("webp_quality", 85)
            ))
        else:
            scene_image_mappings[idx] = None

    # Run multi-process executor to bypass Python's GIL and use all CPU cores
    processed_count = 0
    total_tasks = len(compression_tasks)
    
    if total_tasks > 0:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = [executor.submit(process_and_compress_image, *t) for t in compression_tasks]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as ex:
                    log(0.30, f"Warning: Individual process failed: {str(ex)}")
                processed_count += 1
                progress_step = 0.30 + (0.40 * (processed_count / total_tasks))
                log(progress_step, f"Processing scene frames: {processed_count}/{total_tasks} processed...")

    log(0.70, "Building Open Illuminations manifest variations...")

    # --- Generate Keyframe Lists ---
    mobile_keyframes = []
    desktop_keyframes = []
    static_keyframes = []

    # Insert optional intro cover scenes
    if cover_dest_filename and first_scene_start_sec > 0:
        m_preset = PORTRAIT_PRESETS[0]
        for p in m_preset:
            p_time = p["position"] * first_scene_start_sec
            mobile_keyframes.append({
                "image": cover_dest_filename,
                "start": format_seconds_to_timestamp(p_time),
                "title": None,
                "notes": None,
                "quote": None,
                "view": {
                    "scale": p["scale"],
                    "pan_x": p["pan_x"],
                    "pan_y": p["pan_y"]
                }
            })

        d_preset = LANDSCAPE_PRESETS[0]
        for p in d_preset:
            p_time = p["position"] * first_scene_start_sec
            desktop_keyframes.append({
                "image": cover_dest_filename,
                "start": format_seconds_to_timestamp(p_time),
                "title": None,
                "notes": None,
                "quote": None,
                "view": {
                    "scale": p["scale"],
                    "pan_x": p["pan_x"],
                    "pan_y": p["pan_y"]
                }
            })

        static_keyframes.append({
            "image": cover_dest_filename,
            "start": "00:00:00.00",
            "title": None,
            "notes": None,
            "quote": None,
            "view": {"scale": 1.0, "pan_x": 0.5, "pan_y": 0.5}
        })

    # Filter out scenes with missing physical images
    valid_scenes = []
    for idx, scene in enumerate(scenes):
        img_name = scene_image_mappings[idx]
        if img_name:
            valid_scenes.append((scene, img_name))

    # Compile OIS Timeline Keyframes
    for idx, (scene, img_name) in enumerate(valid_scenes):
        start_sec = parse_timestamp_to_seconds(scene.get("timestamp", "00:00:00"))
        
        if idx < len(valid_scenes) - 1:
            next_scene, _ = valid_scenes[idx + 1]
            end_sec = parse_timestamp_to_seconds(next_scene.get("timestamp", "00:00:00"))
        else:
            end_sec = duration

        scene_duration = max(1.0, end_sec - start_sec)

        raw_quote = scene.get("quote", "").strip()
        quote_text = raw_quote if raw_quote else None

        # A. Compile PORTRAIT / MOBILE timeline variations
        m_preset = PORTRAIT_PRESETS[idx % len(PORTRAIT_PRESETS)]
        for kp_idx, pt in enumerate(m_preset):
            kp_time = start_sec + (pt["position"] * scene_duration)
            mobile_keyframes.append({
                "image": img_name,
                "start": format_seconds_to_timestamp(kp_time),
                "title": None,
                "notes": "",
                "quote": quote_text if kp_idx == 0 else ".",
                "view": {
                    "scale": pt["scale"],
                    "pan_x": pt["pan_x"],
                    "pan_y": pt["pan_y"]
                }
            })

        # B. Compile LANDSCAPE / DESKTOP timeline variations
        d_preset = LANDSCAPE_PRESETS[idx % len(LANDSCAPE_PRESETS)]
        for kp_idx, pt in enumerate(d_preset):
            kp_time = start_sec + (pt["position"] * scene_duration)
            desktop_keyframes.append({
                "image": img_name,
                "start": format_seconds_to_timestamp(kp_time),
                "title": None,
                "notes": "",
                "quote": quote_text if kp_idx == 0 else ".",
                "view": {
                    "scale": pt["scale"],
                    "pan_x": pt["pan_x"],
                    "pan_y": pt["pan_y"]
                }
            })

        # C. Compile STATIC variations
        static_keyframes.append({
            "image": img_name,
            "start": format_seconds_to_timestamp(start_sec),
            "title": None,
            "notes": "",
            "quote": quote_text,
            "view": {"scale": 1.0, "pan_x": 0.5, "pan_y": 0.5}
        })

    # Normalize URLs explicitly to validate successfully against strict OIS schema rules
    author_url = str(metadata.get("author_website", "")).strip()
    if author_url:
        if not (author_url.startswith("http://") or author_url.startswith("https://")):
            author_url = f"https://{author_url}"
    else:
        author_url = ""

    # --- Write Variant JSON Manifests ---
    def build_base_manifest(slug: str, title: str, kfs: List[Dict[str, Any]]) -> Dict[str, Any]:
        variants_config = []
        variants_config.append({
            "slug": "default",
            "name": metadata.get("default_variant_name", "Mobile Mode"),
            "description": metadata.get("default_variant_desc", "Portrait mode")
        })
        if metadata.get("include_desktop", True):
            variants_config.append({
                "slug": "desktop",
                "name": metadata.get("desktop_variant_name", "Desktop Mode"),
                "description": metadata.get("desktop_variant_desc", "Landscape mode")
            })
        if metadata.get("include_static", True):
            variants_config.append({
                "slug": "static",
                "name": metadata.get("static_variant_name", "Static Mode"),
                "description": metadata.get("static_variant_desc", "No animations")
            })

        return {
            "manifest_version": "1.3",
            "book_title": metadata.get("book_title", book_name),
            "book_author": metadata.get("book_author", "Unknown Author"),
            "pack_title": metadata.get("pack_title", f"{book_name} Illuminations"),
            "pack_version": metadata.get("pack_version", "1.0.0"),
            "pack_author": metadata.get("pack_author", "Anonymous"),
            "author_website": author_url,
            "pack_description": metadata.get("pack_description", "Generated dynamically with ABI-Pipeline."),
            "art_type": metadata.get("art_type", "ai-generated"),
            "curation_type": metadata.get("curation_type", "light-curation"),
            "tags": metadata.get("tags", []),
            "content_rating": metadata.get("content_rating", "teen"),
            "orientation": "mixed",
            "image_count": len(valid_scenes) + (1 if cover_dest_filename else 0),
            "pack_creation_date": metadata.get("pack_creation_date", "2026-06-12"),
            "variants": variants_config,
            "authored_for_duration_seconds": duration,
            "keyframes": kfs
        }

    log(0.80, "Writing manifest.json descriptor files...")
    manifest_default = build_base_manifest("default", "Mobile Mode", mobile_keyframes)
    with open(staging_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_default, f, indent=2, ensure_ascii=False)

    if metadata.get("include_desktop", True):
        manifest_desktop = build_base_manifest("desktop", "Desktop Mode", desktop_keyframes)
        with open(staging_dir / "manifest.desktop.json", "w", encoding="utf-8") as f:
            json.dump(manifest_desktop, f, indent=2, ensure_ascii=False)

    if metadata.get("include_static", True):
        manifest_static = build_base_manifest("static", "Static Mode", static_keyframes)
        with open(staging_dir / "manifest.static.json", "w", encoding="utf-8") as f:
            json.dump(manifest_static, f, indent=2, ensure_ascii=False)

    # --- Zip Package Assembly ---
    log(0.85, "Archiving illuminated zip payload assets...")
    
    pack_name = "illuminations.zip"
    
    output_zip_dir = Path(f"./output/{project_name}/{book_name}")
    output_zip_dir.mkdir(parents=True, exist_ok=True)
    output_zip_path = output_zip_dir / pack_name

    # Roll over and archive any older version inside output folder
    archive_existing_pack(output_zip_path)

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in staging_dir.iterdir():
            if file_path.is_file():
                zipf.write(file_path, arcname=file_path.name)

    shutil.rmtree(staging_dir)

    # Copy zip inside book's original folder directly and handle rollover
    if original_book_path and original_book_path.exists():
        destination_copy = original_book_path / pack_name
        archive_existing_pack(destination_copy)
        try:
            shutil.copy2(output_zip_path, destination_copy)
            log(1.0, f"Successfully created pack inside audiobook folder! Path: '{destination_copy}'")
            return str(destination_copy)
        except Exception as ex:
            log(0.95, f"Warning: Failed to copy zip to original audiobook folder: {str(ex)}")

    log(1.0, f"Illumination pack generated successfully! Path: '{output_zip_path}'")
    return str(output_zip_path)