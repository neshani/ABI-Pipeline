import os
import time
import httpx
from typing import Optional, Dict, Any, List
from database.connection import get_setting


class SubtitlesClient:
    """HTTP Client for communicating with the local ABI-Subtitles DirectML inference server."""

    def __init__(self, base_url: Optional[str] = None):
        if base_url is None:
            base_url = get_setting("subtitles_server_url", "http://127.0.0.1:5050")
        
        self.base_url = base_url.rstrip("/")

    def is_alive(self) -> bool:
        """Checks if the ABI-Subtitles server is running and reachable."""
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.base_url}/v1/openapi.json")
                return resp.status_code == 200
        except Exception:
            return False

    def enqueue_job(self, path: str, response_format: str = "clean_json", write_sidecars: bool = False) -> Optional[Dict[str, Any]]:
        """Enqueues a single audio file path for asynchronous transcription."""
        abs_path = os.path.abspath(path)
        payload = {
            "path": abs_path,
            "response_format": response_format,
            "write_sidecars": write_sidecars
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(f"{self.base_url}/v1/jobs", json=payload)
                if resp.status_code == 202:
                    data = resp.json()
                    return data.get("data")
                return None
        except Exception as e:
            print(f"[ABI-Subtitles] Failed to enqueue job for {path}: {e}")
            return None

    def enqueue_batch(self, paths: List[str], response_format: str = "clean_json", write_sidecars: bool = False) -> Optional[Dict[str, Any]]:
        """Enqueues a batch of audio file paths."""
        abs_paths = [os.path.abspath(p) for p in paths]
        payload = {
            "paths": abs_paths,
            "response_format": response_format,
            "write_sidecars": write_sidecars
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(f"{self.base_url}/v1/jobs/batch", json=payload)
                if resp.status_code == 202:
                    data = resp.json()
                    return data.get("data")
                return None
        except Exception as e:
            print(f"[ABI-Subtitles] Failed to enqueue batch: {e}")
            return None

    def get_job_summary(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Safely inspects job status without removing it from server memory."""
        try:
            url = f"{self.base_url}/v1/jobs/{job_id}"
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("data")
                return None
        except Exception:
            return None

    def get_queue_status(self) -> Optional[Dict[str, Any]]:
        """Retrieves active queue health, speedup multiplier, and ETA."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/v1/jobs/status")
                if resp.status_code == 200:
                    return resp.json()
                return None
        except Exception:
            return None

    def get_job_result(self, job_id: str, delete_on_retrieval: bool = True) -> tuple[int, Optional[Any]]:
        """
        Retrieves the completed job result payload.
        Only call this when get_job_summary reports status == 'completed'.
        """
        try:
            url = f"{self.base_url}/v1/jobs/{job_id}/result"
            params = {"delete": "true" if delete_on_retrieval else "false"}
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 200:
                    try:
                        return 200, resp.json()
                    except Exception:
                        return 200, resp.text
                elif resp.status_code == 202:
                    return 202, None
                elif resp.status_code == 422:
                    return 422, resp.text
                else:
                    return resp.status_code, None
        except Exception as e:
            return 500, str(e)

    def cancel_jobs(self, job_id: Optional[str] = None) -> bool:
        """Cancels an active job or the entire queue."""
        try:
            payload = {"job_id": job_id} if job_id else {}
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(f"{self.base_url}/v1/jobs/cancel", json=payload)
                return resp.status_code == 200
        except Exception as e:
            print(f"[ABI-Subtitles] Failed to send cancel command: {e}")
            return False

    def clear_completed(self) -> bool:
        """Clears completed/failed job records from server memory."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(f"{self.base_url}/v1/jobs/clear")
                return resp.status_code == 200
        except Exception:
            return False