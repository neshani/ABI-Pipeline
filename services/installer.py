import sys
import os
import asyncio
import importlib
import subprocess
import threading
import queue
from typing import Callable, Awaitable

# Map of transcription engines to their required Python import check
ENGINE_REQUIREMENTS = {
    "Parakeet ONNX": ["onnxruntime", "onnx_asr", "soundfile"],
    "Whisper": ["faster_whisper", "torch"]
}

# Where we will save download model weights locally
MODEL_STORAGE_DIR = os.path.abspath(".models")
os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)

# Correct Hugging Face Repository IDs
MODEL_REPOS = {
    "Parakeet ONNX": "istupakov/parakeet-tdt-0.6b-v3-onnx",
    "Whisper": "Systran/faster-whisper-small"
}

def check_dependencies(engine_name: str) -> dict:
    """
    Checks if all packages for a given engine are installed in the environment.
    Returns a dict with 'status' (bool) and 'missing' (list of str).
    """
    reqs = ENGINE_REQUIREMENTS.get(engine_name, [])
    missing = []
    for package in reqs:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
            
    return {
        "status": len(missing) == 0,
        "missing": missing
    }

def get_model_dir(engine_name: str) -> str:
    """Returns the local directory where the model is stored."""
    folder = "parakeet" if "ONNX" in engine_name else "whisper"
    return os.path.join(MODEL_STORAGE_DIR, folder)

def check_model_downloaded(engine_name: str) -> bool:
    """Checks if the crucial model weight files exist locally."""
    model_dir = get_model_dir(engine_name)
    if "ONNX" in engine_name:
        # Check for Parakeet's key encoder file
        return os.path.exists(os.path.join(model_dir, "encoder-model.onnx"))
    else:
        # Check for Whisper's key model weights file
        return os.path.exists(os.path.join(model_dir, "model.bin"))


async def run_pip_install(packages: list, log_callback: Callable[[str], None]) -> bool:
    """
    Runs pip install inside a background thread to bypass Windows event loop limitations,
    and pipes stdout asynchronously and thread-safely back to NiceGUI.
    """
    log_callback(f"Starting installation of: {', '.join(packages)}\n")
    
    q = queue.Queue()
    
    def pip_worker():
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ""):
                q.put(line)
                
            process.wait()
            q.put(process.returncode == 0)
        except Exception as e:
            q.put(f"\nCRITICAL PROCESS ERROR: {e}\n")
            q.put(False)

    # Start the worker thread
    thread = threading.Thread(target=pip_worker, daemon=True)
    thread.start()
    
    success = False
    while thread.is_alive() or not q.empty():
        try:
            item = q.get_nowait()
            if isinstance(item, bool):
                success = item
            else:
                log_callback(item)
        except queue.Empty:
            await asyncio.sleep(0.1)
            
    return success


async def download_model_weights(
    engine_name: str, 
    progress_callback: Callable[[float], None],
    log_callback: Callable[[str], None]
) -> bool:
    """
    Downloads the model files using huggingface-cli inside a background thread.
    Pipes download logs and progress indicators live directly into NiceGUI.
    """
    repo_id = MODEL_REPOS.get(engine_name)
    local_dir = get_model_dir(engine_name)
    
    if not repo_id:
        log_callback(f"Error: No model repository ID defined for {engine_name}.\n")
        return False
        
    log_callback(f"Connecting to Hugging Face CLI... Downloading repository: {repo_id}\n")
    progress_callback(0.0)
    
    q = queue.Queue()
    
    def download_worker():
        # Executes huggingface-cli download within our active environment path.
        # This handles resumes automatically and prints live parallel download progress bars.
        cmd = [
            "huggingface-cli", "download", 
            repo_id, 
            "--local-dir", local_dir
        ]
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ""):
                q.put(line)
                
            process.wait()
            q.put(process.returncode == 0)
        except Exception as e:
            q.put(f"\nCRITICAL DOWNLOAD PROCESS ERROR: {e}\n")
            q.put(False)

    # Start the download worker thread
    thread = threading.Thread(target=download_worker, daemon=True)
    thread.start()
    
    success = False
    while thread.is_alive() or not q.empty():
        try:
            item = q.get_nowait()
            if isinstance(item, bool):
                success = item
            else:
                log_callback(item)
        except queue.Empty:
            await asyncio.sleep(0.1)
            
    if success:
        progress_callback(1.0)
        log_callback("\nRepository successfully downloaded!\n")
    else:
        log_callback("\nDownload encountered an error. Please retry.\n")
        
    return success