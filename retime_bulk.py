# retime_bulk.py
import sys
from sqlmodel import Session, select
from database.connection import engine
from database.models import Project, Book
from services.transcription import transcribe_book, get_onnx_model
from services.timing_sync import sync_book_timing

def bulk_retime(project_id: int):
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
        
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        print(f"[Bulk-Retime] Found {len(books)} books in project '{project.name}'")
        
        for book in books:
            print(f"\n==========================================")
            print(f" Processing Book: {book.name}")
            print(f"==========================================")
            
            # Reset book status temporarily so the transcriber doesn't skip it
            book.status = "Imported"
            session.add(book)
            session.commit()
            
            # Step 1: Re-transcribe to generate the timing JSON (does NOT overwrite prompts.csv)
            print(f"[Bulk-Retime] Running transcription to generate timing maps...")
            transcribe_book(book.id, model, project.id)
            
            # Step 2: Run timing synchronization (overwrites ONLY the timestamp column in prompts.csv)
            print(f"[Bulk-Retime] Updating timestamps in prompts.csv...")
            success = sync_book_timing(book.id, project.name, book.name)
            if success:
                print(f"[Bulk-Retime] Success! Timestamps updated for {book.name}.")
            else:
                print(f"[Bulk-Retime] Failed to sync timing for {book.name}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python retime_bulk.py <PROJECT_ID>")
        sys.exit(1)
    
    try:
        pid = int(sys.argv[1])
        bulk_retime(pid)
    except ValueError:
        print("Error: Project ID must be an integer.")