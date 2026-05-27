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

def check_dependencies(engine_name: str, device: str = "CPU") -> dict:
    """
    Checks if all packages for a given engine are installed in the environment.
    Differentiates between CPU (onnxruntime) and GPU (onnxruntime-gpu) packages
    by inspecting package distribution metadata rather than top-level imports.
    """
    reqs = []
    runtime_pkg = None

    if engine_name == "Parakeet ONNX":
        reqs = ["onnx_asr", "soundfile"]
        runtime_pkg = "onnxruntime-gpu" if device == "GPU/CUDA" else "onnxruntime"
    else:
        reqs = ENGINE_REQUIREMENTS.get(engine_name, [])

    missing = []
    
    # 1. Check standard module imports
    for package in reqs:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
            
    # 2. Check the specific distribution package metadata to avoid namespace overlap bugs
    if runtime_pkg:
        from importlib.metadata import version, PackageNotFoundError
        try:
            version(runtime_pkg)
        except PackageNotFoundError:
            missing.append(runtime_pkg)
            
    return {
        "status": len(missing) == 0,
        "missing": missing
    }

def get_model_dir(engine_name: str) -> str:
    """Returns the local directory where the model is stored."""
    folder = "parakeet" if "ONNX" in engine_name else "whisper"
    return os.path.join(MODEL_STORAGE_DIR, folder)

def check_model_downloaded(engine_name: str, device: str = "CPU") -> bool:
    """Checks if all necessary model weight files exist locally based on target hardware."""
    model_dir = get_model_dir(engine_name)
    if "ONNX" in engine_name:
        if device == "GPU/CUDA":
            # GPU requires both config.json and the fp16 quantized encoder file
            return os.path.exists(os.path.join(model_dir, "config.json")) and \
                   os.path.exists(os.path.join(model_dir, "encoder-model.fp16.onnx"))
        else:
            # CPU only requires the standard fp32 encoder file
            return os.path.exists(os.path.join(model_dir, "encoder-model.onnx"))
    else:
        # Check for Whisper's key model weights file
        return os.path.exists(os.path.join(model_dir, "model.bin"))


async def run_pip_install(packages: list, log_callback: Callable[[str], None]) -> bool:
    """
    Runs pip install inside a background thread to bypass Windows event loop limitations.
    Uninstalls CPU/GPU conflicting packages on-the-fly before installation.
    """
    log_callback(f"Starting installation of: {', '.join(packages)}\n")
    
    q = queue.Queue()
    
    def pip_worker():
        # Pre-uninstall conflicting packages to avoid broken onnxruntime installs
        if "onnxruntime-gpu" in packages:
            log_callback("Detected GPU/CUDA request. Cleaning CPU dependencies to prevent conflict...\n")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        elif "onnxruntime" in packages:
            log_callback("Detected CPU request. Cleaning GPU/CUDA dependencies to prevent conflict...\n")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime-gpu"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )

        # Set environment variable to force subprocess to output using UTF-8
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        cmd = [sys.executable, "-m", "pip", "install"] + packages
        try:
            # Specifically using encoding="utf-8" and env=env to prevent Windows charmap decoder crashes
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                env=env,
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
    device: str,
    progress_callback: Callable[[float], None],
    log_callback: Callable[[str], None]
) -> bool:
    """
    Downloads model files using hf download inside a background thread.
    If GPU/CUDA is active, downloads both base FP32 configs and quantized FP16 models
    and merges them into your workspace to keep things 100% offline-compatible.
    """
    local_dir = get_model_dir(engine_name)
    
    # Map target repositories dynamically
    repos = []
    if engine_name == "Parakeet ONNX":
        # Base files (config, vocabulary)
        repos.append("istupakov/parakeet-tdt-0.6b-v3-onnx")
        if device == "GPU/CUDA":
            # FP16 quantized execution files
            repos.append("grikdotnet/parakeet-tdt-0.6b-fp16")
    else:
        repo_id = MODEL_REPOS.get(engine_name)
        if repo_id:
            repos.append(repo_id)

    if not repos:
        log_callback(f"Error: No model repositories defined for {engine_name}.\n")
        return False
        
    progress_callback(0.0)
    q = queue.Queue()
    
    def download_worker():
        # Force the spawned python-based "hf" executable to format its stdout streams as UTF-8
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        for i, repo_id in enumerate(repos):
            log_callback(f"\n[{i+1}/{len(repos)}] Connecting to Hugging Face... Downloading repository: {repo_id}\n")
            cmd = [
                "hf", "download", 
                repo_id, 
                "--local-dir", local_dir
            ]
            try:
                # Specifically using encoding="utf-8" and env=env to prevent Windows charmap decoder crashes
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    env=env,
                    bufsize=1
                )
                
                for line in iter(process.stdout.readline, ""):
                    q.put(line)
                    
                process.wait()
                if process.returncode != 0:
                    q.put(False)
                    return
            except Exception as e:
                q.put(f"\nCRITICAL DOWNLOAD PROCESS ERROR: {e}\n")
                q.put(False)
                return
        q.put(True)

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
        log_callback("\nAll necessary repositories successfully downloaded and merged locally!\n")
    else:
        log_callback("\nDownload encountered an error. Please retry.\n")
        
    return success