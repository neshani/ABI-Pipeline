# retime_bulk.py
import sys
import shutil
import re
from pathlib import Path
from sqlmodel import Session, select
from database.connection import engine
from database.models import Project, Book
from services.transcription import transcribe_book, get_onnx_model
from services.timing_sync import sync_book_timing


def ensure_prompts_csv(book_dir: Path) -> Path:
    """
    Ensures 'prompts.csv' exists in book_dir.
    If 'prompts.csv' does not exist, searches for any other .csv file (e.g., custom-named prompt CSVs)
    and copies it to 'prompts.csv'.
    """
    target_csv = book_dir / "prompts.csv"
    if target_csv.exists():
        return target_csv

    # Search for alternative CSV files in the book directory
    csv_files = [f for f in book_dir.glob("*.csv") if f.name.lower() != "prompts.csv"]
    if csv_files:
        source_csv = csv_files[0]
        print(f"[Retimer] 'prompts.csv' missing in '{book_dir.name}'. Found '{source_csv.name}'. Copying to 'prompts.csv'...")
        shutil.copy2(source_csv, target_csv)
        return target_csv

    return target_csv


def fix_image_filenames_in_path(target_path: Path) -> int:
    """
    Scans target_path (folder or file) for images named like '01-01_desc.png' 
    and renames them to standard '01_01_desc.png'.
    """
    if not target_path.exists():
        print(f"[Fix-Images] Error: Path '{target_path}' does not exist.")
        return 0

    files_to_check = []
    if target_path.is_file():
        files_to_check = [target_path]
    else:
        files_to_check = list(target_path.rglob("*.png")) + list(target_path.rglob("*.webp"))

    renamed_count = 0
    for img_file in files_to_check:
        m = re.match(r"^(\d+)-(\d+)(.*)", img_file.name)
        if m:
            ch_str, sc_str, rest = m.groups()
            new_name = f"{int(ch_str):02d}_{int(sc_str):02d}{rest}"
            new_path = img_file.parent / new_name
            if new_path != img_file:
                print(f"[Fix-Images] Renaming: '{img_file.name}' -> '{new_name}'")
                img_file.rename(new_path)
                renamed_count += 1

    return renamed_count


def list_projects():
    """Lists all projects from the database with their ID, Name, Type, and Status."""
    print("\n================================================================================")
    print("                           ABI-Pipeline Projects")
    print("================================================================================")
    with Session(engine) as session:
        statement = select(Project).order_by(Project.id)
        projects = session.exec(statement).all()
        
        if not projects:
            print(" No projects found in the database.")
            print("================================================================================")
            return

        print(f" {'ID':<6} | {'Project Name':<35} | {'Type':<8} | {'Status':<15}")
        print("-" * 75)
        for p in projects:
            p_type = "Batch" if getattr(p, "is_batch", False) else "Single"
            print(f" {p.id:<6} | {p.name:<35} | {p_type:<8} | {p.status:<15}")
        print("================================================================================")


def run_image_fix_cli(provided_path: str = None):
    """CLI tool to rename images in a project or folder from 01-01_... to 01_01_..."""
    print("\n=== ABI-Pipeline: Dash-to-Underscore Image Filename Fixer ===")
    if provided_path:
        path_str = provided_path
    else:
        path_str = input("Enter Project ID, Project Name, or full path to output folder:\n> ").strip()

    if not path_str:
        print("Error: No path or ID provided.")
        return

    # Normalize dragged paths
    if (path_str.startswith('"') and path_str.endswith('"')) or (path_str.startswith("'") and path_str.endswith("'")):
        path_str = path_str[1:-1]

    # Check if user entered a numeric Project ID
    if path_str.isdigit():
        pid = int(path_str)
        with Session(engine) as session:
            project = session.get(Project, pid)
            if not project:
                print(f"Error: Project ID {pid} not found in database.")
                return
            target_path = Path("output") / project.name
    else:
        target_path = Path(path_str).resolve()

    print(f"Scanning directory: {target_path}")
    renamed = fix_image_filenames_in_path(target_path)
    print(f"Done! Renamed {renamed} image(s) to standard '01_01_...' format.")


def bulk_retime(project_id: int, target_book: str = None, force: bool = False, only_prompts: bool = False):
    """
    Automates timing map alignment, then updates the existing prompts.csv with 
    high-accuracy timestamps. If only_prompts is True, skips speech-to-text transcription.
    """
    model = None
    if not only_prompts:
        print(f"[Bulk-Retime] Loading speech-to-text model...")
        model = get_onnx_model() 
    else:
        print(f"[Bulk-Retime] Skipping speech-to-text model loading (/prompt_fix mode activated)...")

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            print(f"[Bulk-Retime] Error: Project ID {project_id} not found.")
            return
        
        query = select(Book).where(Book.project_id == project_id)
        books = session.exec(query).all()
        
        if target_book:
            filtered_books = []
            for b in books:
                if str(b.id) == target_book or target_book.lower() in b.name.lower():
                    filtered_books.append(b)
            books = filtered_books
            
        print(f"[Bulk-Retime] Found {len(books)} book(s) to process in project '{project.name}'")
        if target_book and not books:
            print(f"[Bulk-Retime] Warning: No books matched search criteria: '{target_book}'")
            return
        
        for book in books:
            print(f"\n==========================================")
            print(f" Processing Book: {book.name} (ID: {book.id})")
            print(f"==========================================")
            
            book_dir = Path("output") / project.name / book.name
            if not book_dir.exists() and project.path:
                book_dir = Path(project.path) / book.name
            if not book_dir.exists() and project.path:
                book_dir = Path(project.path)
            
            if book_dir.exists():
                ensure_prompts_csv(book_dir)
                # Auto-fix legacy dash-formatted images if present
                fix_image_filenames_in_path(book_dir)

            if not only_prompts:
                book.status = "Imported"
                session.add(book)
                session.commit()
                
                print(f"[Bulk-Retime] Running transcription to generate timing maps (Force={force})...")
                transcribe_book(book.id, model, project.id, force_retranscribe=force)
            else:
                print(f"[Bulk-Retime] Only recalculating prompt timestamps from physical audio. Skipping transcription...")
            
            print(f"[Bulk-Retime] Updating timestamps in prompts.csv...")
            success = sync_book_timing(book.id, project.name, book.name, auto_approve=True)
            if success:
                print(f"[Bulk-Retime] Success! Timestamps updated for {book.name}.")
            else:
                print(f"[Bulk-Retime] Failed to sync timing for {book.name}.")


def run_interactive_fix(provided_path: str = None):
    """
    Interactive standalone tool that repairs a malformed or broken prompts CSV file.
    Automatically copies custom-named CSV files (e.g. VolumeName_prompts.csv) to prompts.csv.
    Instantly recalculates quotes mapping and automatically marks all rows as approved.
    """
    print("\n=== ABI-Pipeline: Standalone Timing & Approval Repair ===")
    if provided_path:
        csv_path_str = provided_path
    else:
        csv_path_str = input("Please enter the full path to the prompts CSV (or book folder):\n> ").strip()

    if not csv_path_str:
        print("Error: No path was provided.")
        return

    if (csv_path_str.startswith('"') and csv_path_str.endswith('"')) or (csv_path_str.startswith("'") and csv_path_str.endswith("'")):
        csv_path_str = csv_path_str[1:-1]

    input_path = Path(csv_path_str).resolve()
    if not input_path.exists():
        print(f"Error: Target path not found at '{input_path}'")
        return

    if input_path.is_file():
        book_dir = input_path.parent
        target_csv = book_dir / "prompts.csv"
        if input_path.name.lower() != "prompts.csv":
            print(f"[Fix-Tool] Copying custom CSV '{input_path.name}' to 'prompts.csv'...")
            shutil.copy2(input_path, target_csv)
    elif input_path.is_dir():
        book_dir = input_path
        target_csv = ensure_prompts_csv(book_dir)
        if not target_csv.exists():
            print(f"Error: No .csv files found in directory '{book_dir}'")
            return
    else:
        print(f"Error: Invalid path type for '{input_path}'")
        return

    # Check and fix any legacy image filenames in book directory
    fix_image_filenames_in_path(book_dir)

    print(f"\nScanning path topology: {book_dir}")
    book_name = book_dir.name
    project_name = book_dir.parent.name
    print(f"Inferred Project: '{project_name}'")
    print(f"Inferred Volume:  '{book_name}'")

    with Session(engine) as session:
        statement = select(Book).join(Project).where(Book.name == book_name, Project.name == project_name)
        book = session.exec(statement).first()
        
        if not book:
            print("[Fix-Tool] Exact project match missed in database. Searching book name globally...")
            statement = select(Book).where(Book.name == book_name)
            book = session.exec(statement).first()

        if not book:
            print(f"[Fix-Tool] Error: Volume '{book_name}' was not found in active database index.")
            print("Cannot calculate relative timestamps without active volume audio configurations.")
            return

        db_project = session.get(Project, book.project_id)
        if db_project:
            project_name = db_project.name
        
        book_id = book.id

    print(f"[Fix-Tool] Indexed matched Database Book ID {book_id} under project '{project_name}'.")
    print("[Fix-Tool] Executing timing sync calculations...")
    
    success = sync_book_timing(book_id, project_name, book.name, auto_approve=True)
    if success:
        print(f"\n[Fix-Tool] SUCCESS! Prompts.csv at '{book_dir}' updated and marked 'approved'!")
    else:
        print(f"\n[Fix-Tool] Timing alignment task encountered warnings. Check logs above.")


def print_help():
    """Prints a structured help document for the CLI tool."""
    print("""
================================================================================
                    ABI-Pipeline: retime_bulk CLI Tool
================================================================================
This utility automates transcription generation, timing map alignment, 
image filename standardization, and updates target 'prompts.csv' flat-files.

Usage:
  python retime_bulk.py /list
  python retime_bulk.py /fix_images [project_id_or_path]
  python retime_bulk.py <PROJECT_ID> [options]
  python retime_bulk.py <PROJECT_ID> /prompt_fix [options]
  python retime_bulk.py /fix [path_to_csv_or_folder]
  python retime_bulk.py /?

Core Commands:
  /list                   Lists all projects from the database with their ID and Name.
  /fix_images             Scans folder/project and renames image files from 
                          '01-01_desc.png' to standard '01_01_desc.png'.
  <PROJECT_ID>            Target a specific SQLModel project ID to re-transcribe 
                          and align timestamps.
  /prompt_fix             Skips slow speech-to-text transcription and instantly
                          recalculates prompt timestamps based on physical audio 
                          and existing timing maps. Perfect for fixing VBR/offset drift.
  /fix, /redo             Launches standalone repair mode. If a custom CSV is targeted
                          (e.g., BookTitle_prompts.csv), it will be copied to prompts.csv
                          automatically.

Options:
  --book <ID_or_Name>     Process only a specific volume. Accepts database Book ID 
                          or a partial string search of the book name.
                          Example: --book "Exit Strategy" or --book 4
  --force                 Bypasses transcription caches. Deletes the 'workspace_temp' 
                          directories, resets all chapters to 'Pending', and 
                          runs fresh model inference.
  /?, -h, --help          Show this CLI guide.
================================================================================
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
    
    arg_lower_list = [a.lower() for a in sys.argv]
    if any(h in arg_lower_list for h in ["/?", "-h", "--help", "help"]):
        print_help()
        sys.exit(0)

    # Check for /list command
    if any(l in arg_lower_list for l in ["/list", "-list", "--list", "list", "/ls"]):
        list_projects()
        sys.exit(0)

    # Check for /fix_images command
    if any(fi in arg_lower_list for fi in ["/fix_images", "/fiximages", "--fix_images", "--fix-images", "-fix_images"]):
        extra_path = sys.argv[2] if len(sys.argv) > 2 else None
        run_image_fix_cli(provided_path=extra_path)
        sys.exit(0)
        
    prompt_fix_flags = {"/prompt_fix", "/prompt", "--prompt_fix", "--prompt-fix", "-prompt_fix", "-prompt"}
    only_prompts_opt = any(p in arg_lower_list for p in prompt_fix_flags)
    
    cleaned_argv = [a for a in sys.argv if a.lower() not in prompt_fix_flags]
    
    if len(cleaned_argv) < 2:
        run_interactive_fix()
        sys.exit(0)
        
    arg = cleaned_argv[1].lower()
    if arg in ["/fix", "/redo", "-fix", "-redo", "--fix", "--redo"]:
        extra_path = cleaned_argv[2] if len(cleaned_argv) > 2 else None
        run_interactive_fix(provided_path=extra_path)
    else:
        try:
            pid = int(cleaned_argv[1])
            
            target_book_opt = None
            force_retranscribe_opt = False
            
            args = cleaned_argv[2:]
            i = 0
            while i < len(args):
                arg_clean = args[i].lower()
                if arg_clean in ["--book", "-book", "/book"] and i + 1 < len(args):
                    target_book_opt = args[i+1]
                    i += 2
                elif arg_clean in ["--force", "-force", "/force"]:
                    force_retranscribe_opt = True
                    i += 1
                else:
                    i += 1
                    
            bulk_retime(pid, target_book=target_book_opt, force=force_retranscribe_opt, only_prompts=only_prompts_opt)
        except ValueError:
            print(f"Error: Unknown argument or invalid project ID: '{cleaned_argv[1]}'. Use /? for usage details.")
            sys.exit(1)