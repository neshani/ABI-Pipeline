import os
import tkinter as tk
from tkinter import filedialog
from typing import Optional, List

def run_directory_picker(title: str = "Select Directory") -> Optional[str]:
    """
    Spawns a native OS folder selection dialog safely on a background thread.
    Hides the empty parent Tkinter window and forces the dialog window to the front.
    """
    try:
        root = tk.Tk()
        root.withdraw()  # Withdraw the root helper window so it is invisible
        
        # Topmost lift trick: forces the window to pop up in front of the active web browser
        root.wm_attributes('-topmost', 1)
        
        # Execute the native directory picker
        selected_dir = filedialog.askdirectory(title=title)
        
        root.destroy()
        return os.path.abspath(selected_dir) if selected_dir else None
    except Exception as e:
        print(f"[Native-Picker] Error launching native directory picker: {e}")
        return None


def run_file_picker(title: str = "Select Files", file_types: List[tuple[str, str]] = None) -> List[str]:
    """
    Spawns a native OS file selection dialog supporting multi-file selection safely.
    """
    if file_types is None:
        file_types = [("All Files", "*.*")]
        
    try:
        root = tk.Tk()
        root.withdraw()
        
        # Topmost lift trick
        root.wm_attributes('-topmost', 1)
        
        selected_files = filedialog.askopenfilenames(title=title, filetypes=file_types)
        
        root.destroy()
        return [os.path.abspath(f) for f in selected_files] if selected_files else []
    except Exception as e:
        print(f"[Native-Picker] Error launching native file picker: {e}")
        return []