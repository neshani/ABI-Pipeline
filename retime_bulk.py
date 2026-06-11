# retime_bulk.py
import sys
from pathlib import Path
from sqlmodel import Session, select
from database.connection import engine
from database.models import Project, Book
from services.transcription import transcribe_book, get_onnx_model
from services.timing_sync import sync_book_timing

def bulk_retime(project_id: int, target_book: str = None, force: bool = False):
    """
    Automates re-transcribing audio files to generate precise timing JSONs,
    then updates the existing prompts.csv with high-accuracy timestamps.
    """
    print(f"[Bulk-Retime] Loading speech-to-text model...")
    # This automatically uses the model configured in your settings (Parakeet ONNX or Faster-Whisper)
    model = get_onnx_model() 

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            print(f"[Bulk-Retime] Error: Project ID {project_id} not found.")
            return
        
        # Query books for this project
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
            
            # Reset book status temporarily so the transcriber doesn't skip it
            book.status = "Imported"
            session.add(book)
            session.commit()
            
            # Step 1: Re-transcribe to generate the timing JSON (does NOT overwrite prompts.csv)
            print(f"[Bulk-Retime] Running transcription to generate timing maps (Force={force})...")
            transcribe_book(book.id, model, project.id, force_retranscribe=force)
            
            # Step 2: Run timing synchronization (overwrites ONLY the timestamp column in prompts.csv)
            # Pass auto_approve=True to automatically approve these existing pre-approved books
            print(f"[Bulk-Retime] Updating timestamps in prompts.csv...")
            success = sync_book_timing(book.id, project.name, book.name, auto_approve=True)
            if success:
                print(f"[Bulk-Retime] Success! Timestamps updated for {book.name}.")
            else:
                print(f"[Bulk-Retime] Failed to sync timing for {book.name}.")


def run_interactive_fix():
    """
    Interactive standalone tool that repairs a malformed or broken prompts.csv file.
    Does not run slow speech-to-text transcription. Instantly recalculates quotes mapping
    and automatically marks all rows as approved.
    """
    print("\n=== ABI-Pipeline: Standalone Timing & Approval Repair ===")
    csv_path_str = input("Please enter the full path to the prompts.csv file:\n> ").strip()
    if not csv_path_str:
        print("Error: No path was provided.")
        return

    # Normalize dragged paths containing quotes
    if (csv_path_str.startswith('"') and csv_path_str.endswith('"')) or (csv_path_str.startswith("'") and csv_path_str.endswith("'")):
        csv_path_str = csv_path_str[1:-1]

    prompts_path = Path(csv_path_str).resolve()
    if not prompts_path.exists():
        print(f"Error: Target file not found at '{prompts_path}'")
        return

    print(f"\nScanning path topology: {prompts_path}")
    book_name = prompts_path.parent.name
    project_name = prompts_path.parent.parent.name
    print(f"Inferred Project: '{project_name}'")
    print(f"Inferred Volume:  '{book_name}'")

    # Match folders back to the index database
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
    
    # Pass auto_approve=True so that the touch-up tool automatically approves all items
    success = sync_book_timing(book_id, project_name, book.name, auto_approve=True)
    if success:
        print(f"\n[Fix-Tool] SUCCESS! Prompts.csv at '{prompts_path}' updated and marked 'approved'!")
    else:
        print(f"\n[Fix-Tool] Timing alignment task encountered warnings. Check logs above.")


def print_help():
    """Prints a structured help document for the CLI tool."""
    print("""
================================================================================
                    ABI-Pipeline: retime_bulk CLI Tool
================================================================================
This utility automates transcription generation, timing map alignment, 
and updates the timestamp column in target 'prompts.csv' flat-files.

Usage:
  python retime_bulk.py <PROJECT_ID> [options]
  python retime_bulk.py /fix
  python retime_bulk.py /redo
  python retime_bulk.py /?

Core Commands:
  <PROJECT_ID>            Target a specific SQLModel project ID to re-transcribe 
                          and align timestamps.
  /fix, /redo             Launches interactive standalone repair mode. Repair and 
                          approve an out-of-sync 'prompts.csv' file without 
                          running slow audio re-transcription.

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
    
    arg = sys.argv[1].lower()
    if arg in ["/?", "-h", "--help", "help"]:
        print_help()
        sys.exit(0)
        
    if arg in ["/fix", "/redo", "-fix", "-redo", "--fix", "--redo"]:
        run_interactive_fix()
    else:
        try:
            pid = int(sys.argv[1])
            
            # Parse additional optional arguments
            target_book_opt = None
            force_retranscribe_opt = False
            
            args = sys.argv[2:]
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
                    
            bulk_retime(pid, target_book=target_book_opt, force=force_retranscribe_opt)
        except ValueError:
            print(f"Error: Unknown argument or invalid project ID: '{sys.argv[1]}'. Use /? for usage details.")
            sys.exit(1)