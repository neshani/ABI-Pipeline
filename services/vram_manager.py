"""Automatic VRAM arbitration between the image and language backends.

ComfyUI and the LLM host are both GPU-hungry and cannot both sit in VRAM
comfortably. The agent flips between the two during a run and (as of this
module) no longer has to remember to release one before loading the other.

``ensure_comfy`` / ``ensure_llm`` track which backend touched the GPU last and,
on a switch, dispatch the matching release command *before* returning:

    comfy -> llm : POST ``/free`` to ComfyUI
    llm -> comfy : unload every loaded model from llama-server / LM Studio

The tracker is process-global and starts empty, so a cold start is never
treated as a switch - nothing is released until one backend is known to have
run and the other is about to. This is deliberately not persisted: a restart
usually means the backends were restarted too, and a stale "last used" would
release the wrong side.

The functions are synchronous (``requests``) and guarded by one lock so the
many concurrent LLM calls of a prompt-generation run only ever trigger a single
switch. Async callers use the ``*_async`` wrappers, which hop to a thread so
the event loop is never blocked by a backend probe.
"""
from __future__ import annotations

import asyncio
import threading
import urllib.parse
from typing import Callable, Optional

import requests

from database.connection import get_setting

COMFY = "comfy"
LLM = "llm"

_last_used: Optional[str] = None
_lock = threading.Lock()

Log = Optional[Callable[[str], None]]


def last_used() -> Optional[str]:
    """Which backend last used the GPU, or ``None`` before either has."""
    return _last_used


def reset() -> None:
    """Forget the last backend (tests, and after a full manual clearance)."""
    global _last_used
    with _lock:
        _last_used = None


def enabled() -> bool:
    """Whether automatic switching is turned on in settings (default yes)."""
    raw = get_setting("vram_auto_switch", True)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in ("false", "0", "no", "off", "")


def _log(log: Log, message: str) -> None:
    if log is not None:
        log(message)
        return
    try:
        from ui import state
        state.add_console_log(message)
    except Exception:
        print(message, flush=True)


def _comfy_base() -> str:
    comfy_url = get_setting("comfy_url", "127.0.0.1:8188") or "127.0.0.1:8188"
    return comfy_url.replace("http://", "").replace("https://", "").strip("/")


def _llm_base() -> str:
    """Root host of the LLM endpoint, without any trailing API path."""
    llm_url = get_setting("llm_url", "http://127.0.0.1:11434") or "http://127.0.0.1:11434"
    try:
        parsed = urllib.parse.urlparse(llm_url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        pass
    return llm_url.rstrip("/")


def free_comfy(log: Log = None) -> bool:
    """Ask ComfyUI to unload its models and drop its cache."""
    try:
        resp = requests.post(
            f"http://{_comfy_base()}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=2.0,
        )
        if resp.status_code == 200:
            _log(log, "[VRAM] ComfyUI model unload and cache clear requested.")
            return True
        _log(log, f"[VRAM] ComfyUI free returned status {resp.status_code}.")
        return False
    except Exception as e:
        _log(log, f"[VRAM] ComfyUI free skipped (host offline?): {e}")
        return False


def _llama_loaded_models(base: str) -> list:
    """Model ids llama-server reports as currently loaded."""
    resp = requests.get(f"{base}/models", timeout=2.0)
    if resp.status_code != 200:
        return []
    data = resp.json()
    if isinstance(data, dict):
        models = data.get("data", data.get("models", []))
        if not models and "id" in data:
            models = [data]
    elif isinstance(data, list):
        models = data
    else:
        models = []
    loaded = []
    for item in models:
        if not isinstance(item, dict):
            continue
        status = item.get("status", {})
        value = status.get("value") if isinstance(status, dict) else None
        if value == "loaded" and item.get("id"):
            loaded.append(item["id"])
    return loaded


def unload_llm(log: Log = None) -> bool:
    """Ask the LLM host to release its loaded models.

    Covers both conventions the run may be pointed at: LM Studio's targeted
    unload, and llama-server, which is enumerated first so only resident models
    are told to unload.
    """
    base = _llm_base()
    model_name = get_setting("llm_model", "local-model") or "local-model"
    dispatched = False

    # LM Studio: targeted unload of the configured model instance.
    try:
        requests.post(
            f"{base}/api/v1/models/unload",
            json={"instance_id": model_name},
            timeout=1.5,
        )
        dispatched = True
    except Exception:
        pass

    # llama-server: unload whatever it reports as loaded.
    try:
        loaded = _llama_loaded_models(base)
        if not loaded:
            _log(log, "[VRAM] LLM host reports no resident models to unload.")
        for model_id in loaded:
            try:
                resp = requests.post(
                    f"{base}/models/unload",
                    json={"model": model_id},
                    timeout=2.0,
                )
                if resp.status_code == 200:
                    dispatched = True
                    _log(log, f"[VRAM] Unloaded LLM model: {model_id}")
                else:
                    _log(log, f"[VRAM] LLM unload returned {resp.status_code} for {model_id}.")
            except Exception as e:
                _log(log, f"[VRAM] Failed to unload LLM model {model_id}: {e}")
    except Exception as e:
        _log(log, f"[VRAM] LLM unload skipped (host offline?): {e}")

    return dispatched


def free_all(log: Log = None) -> None:
    """Release both backends and forget the tracker (manual "Free VRAM")."""
    free_comfy(log)
    unload_llm(log)
    reset()


def ensure_comfy(log: Log = None) -> None:
    """Mark ComfyUI as the active backend, releasing the LLM on a switch."""
    global _last_used
    if not enabled():
        return
    with _lock:
        if _last_used == LLM:
            _log(log, "[VRAM] Switching to ComfyUI - unloading the LLM first.")
            unload_llm(log)
        _last_used = COMFY


def ensure_llm(log: Log = None) -> None:
    """Mark the LLM as the active backend, freeing ComfyUI on a switch."""
    global _last_used
    if not enabled():
        return
    with _lock:
        if _last_used == COMFY:
            _log(log, "[VRAM] Switching to the LLM - freeing ComfyUI VRAM first.")
            free_comfy(log)
        _last_used = LLM


async def ensure_comfy_async(log: Log = None) -> None:
    await asyncio.to_thread(ensure_comfy, log)


async def ensure_llm_async(log: Log = None) -> None:
    await asyncio.to_thread(ensure_llm, log)
