import os
import json
import uuid
import copy
import requests
import websocket
import random
from typing import Dict, Any, List, Optional, Tuple

class ComfyClient:
    def __init__(self, api_address: str):
        self.api_address = api_address
        self.client_id = str(uuid.uuid4())

    def check_connection(self) -> bool:
        """
        Performs a lightweight heartbeat request to ComfyUI to verify it is online.
        """
        try:
            resp = requests.get(f"http://{self.api_address}/system_stats", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def analyze_workflow(self, workflow_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dynamically analyzes any ComfyUI API JSON workflow to see what overrides can be applied.
        Scans for standard KSamplers, custom samplers, schedulers, checkpoint loaders, etc.
        """
        discovered = {}
        for node_id, node_data in workflow_json.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            title = node_data.get("_meta", {}).get("title", class_type)

            if class_type in ["KSampler", "KSamplerAdvanced"]:
                steps = inputs.get("steps", 20)
                cfg = inputs.get("cfg", 8.0)
                sampler_name = inputs.get("sampler_name", "euler")
                scheduler = inputs.get("scheduler", "normal")

                if not isinstance(steps, (int, float)): steps = 20
                if not isinstance(cfg, (int, float)): cfg = 8.0
                if not isinstance(sampler_name, str): sampler_name = "euler"
                if not isinstance(scheduler, str): scheduler = "normal"

                discovered[node_id] = {
                    "type": "sampler",
                    "title": title,
                    "class_type": class_type,
                    "params": {
                        "steps": int(steps),
                        "cfg": float(cfg),
                        "sampler_name": sampler_name,
                        "scheduler": scheduler
                    }
                }
            elif class_type == "BasicScheduler":
                steps = inputs.get("steps", 10)
                scheduler = inputs.get("scheduler", "normal")
                denoise = inputs.get("denoise", 1.0)

                if not isinstance(steps, (int, float)): steps = 10
                if not isinstance(scheduler, str): scheduler = "normal"
                if not isinstance(denoise, (int, float)): denoise = 1.0

                discovered[node_id] = {
                    "type": "basic_scheduler",
                    "title": title,
                    "class_type": class_type,
                    "params": {
                        "steps": int(steps),
                        "scheduler": scheduler,
                        "denoise": float(denoise)
                    }
                }
            elif class_type == "KSamplerSelect":
                sampler_name = inputs.get("sampler_name", "euler")
                if not isinstance(sampler_name, str): sampler_name = "euler"

                discovered[node_id] = {
                    "type": "sampler_select",
                    "title": title,
                    "class_type": class_type,
                    "params": {
                        "sampler_name": sampler_name
                    }
                }
            elif class_type == "SamplerCustom":
                cfg = inputs.get("cfg", 8.0)
                if not isinstance(cfg, (int, float)): cfg = 8.0

                discovered[node_id] = {
                    "type": "custom_sampler",
                    "title": title,
                    "class_type": class_type,
                    "params": {
                        "cfg": float(cfg)
                    }
                }
            elif class_type in ["EmptyLatentImage", "EmptySD3LatentImage", "FluxEmptyLatentImage"]:
                width = inputs.get("width", 512)
                height = inputs.get("height", 512)

                if not isinstance(width, (int, float)): width = 512
                if not isinstance(height, (int, float)): height = 512

                discovered[node_id] = {
                    "type": "resolution",
                    "title": title,
                    "class_type": class_type,
                    "params": {
                        "width": int(width),
                        "height": int(height)
                    }
                }
            elif class_type == "ResolutionSelector":
                aspect_ratio = inputs.get("aspect_ratio", "1:1 (Square)")
                megapixels = inputs.get("megapixels", 1.0)
                multiple = inputs.get("multiple", 8)

                if not isinstance(aspect_ratio, str): aspect_ratio = "1:1 (Square)"
                if not isinstance(megapixels, (int, float)): megapixels = 1.0
                if not isinstance(multiple, (int, float)): multiple = 8

                discovered[node_id] = {
                    "type": "resolution_selector",
                    "title": title,
                    "class_type": class_type,
                    "params": {
                        "aspect_ratio": aspect_ratio,
                        "megapixels": float(megapixels),
                        "multiple": int(multiple)
                    }
                }
            elif class_type in ["CheckpointLoaderSimple", "UNETLoader"]:
                model_param = "ckpt_name" if "ckpt_name" in inputs else "unet_name"
                model_val = inputs.get(model_param, "")
                if not isinstance(model_val, str): model_val = ""

                discovered[node_id] = {
                    "type": "model_loader",
                    "title": title,
                    "class_type": class_type,
                    "params": {
                        "model_param_key": model_param,
                        model_param: model_val
                    }
                }
            elif class_type in ["LoraLoader", "LoraLoaderModelOnly"]:
                lora_name = inputs.get("lora_name", "")
                strength_model = inputs.get("strength_model", 1.0)
                strength_clip = inputs.get("strength_clip", 1.0)

                if not isinstance(lora_name, str): lora_name = ""
                if not isinstance(strength_model, (int, float)): strength_model = 1.0
                if not isinstance(strength_clip, (int, float)): strength_clip = 1.0

                discovered[node_id] = {
                    "type": "lora_loader",
                    "title": title,
                    "class_type": class_type,
                    "params": {
                        "lora_name": lora_name,
                        "strength_model": float(strength_model),
                        "strength_clip": float(strength_clip)
                    }
                }
            elif class_type in ["CLIPLoader", "DualCLIPLoader"]:
                clip_param1 = "clip_name1" if "clip_name1" in inputs else "clip_name"
                clip_param2 = "clip_name2" if "clip_name2" in inputs else None
                
                clip_val1 = inputs.get(clip_param1, "")
                if not isinstance(clip_val1, str): clip_val1 = ""

                discovered_params = {
                    "clip_param_key": clip_param1,
                    clip_param1: clip_val1
                }
                if clip_param2:
                    clip_val2 = inputs.get(clip_param2, "")
                    if not isinstance(clip_val2, str): clip_val2 = ""
                    discovered_params["clip_param_key2"] = clip_param2
                    discovered_params[clip_param2] = clip_val2
                    
                discovered[node_id] = {
                    "type": "clip_loader",
                    "title": title,
                    "class_type": class_type,
                    "params": discovered_params
                }
            elif class_type == "VAELoader":
                vae_name = inputs.get("vae_name", "")
                if not isinstance(vae_name, str): vae_name = ""

                discovered[node_id] = {
                    "type": "vae_loader",
                    "title": title,
                    "class_type": class_type,
                    "params": {
                        "vae_name": vae_name
                    }
                }
            elif class_type == "PrimitiveBoolean":
                value = inputs.get("value", False)
                if not isinstance(value, bool):
                    value = False

                discovered[node_id] = {
                    "type": "primitive_boolean",
                    "title": title,
                    "class_type": class_type,
                    "params": {
                        "value": value
                    }
                }
        return discovered

    def generate_image_sync(
        self,
        workflow_json: Dict[str, Any],
        prompt_text: str,
        neg_prompt_text: str,
        seed: Optional[int] = None,
        overrides: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        suffix: str = ""
    ) -> Tuple[Optional[bytes], str]:
        """
        Processes and submits the workflow, monitors execution via WebSockets,
        and returns (image_bytes, logs_as_text). Zero hardcoding: modifies any 
        nodes with string placeholders or seed settings dynamically.
        """
        workflow = copy.deepcopy(workflow_json)
        logs = []

        # 1. Substitute prompt and negative placeholders recursively
        full_positive_prompt = f"{prefix}{prompt_text}{suffix}".strip()
        target_seed = seed if seed is not None else random.randint(1, 4294967294)
        logs.append(f"[Comfy-API] Injecting Seed: {target_seed}")

        def process_node(node):
            if "inputs" not in node:
                return
            for k, v in list(node["inputs"].items()):
                # Prompt substitution
                if isinstance(v, str):
                    if v == "<prompt>":
                        node["inputs"][k] = full_positive_prompt
                    elif v == "<negPrompt>" or v == "<negprompt>":
                        node["inputs"][k] = neg_prompt_text
                # Seed auto-replacement
                if k in ["seed", "noise_seed"] and isinstance(v, (int, float)):
                    node["inputs"][k] = target_seed

        for node_id, node in workflow.items():
            process_node(node)
            
            # 2. Apply Custom UI Overrides (Targeted dynamically by node_id and parameter key)
            if overrides and node_id in overrides:
                for param_key, param_val in overrides[node_id].items():
                    if param_key in node.get("inputs", {}):
                        node["inputs"][param_key] = param_val
                        logs.append(f"[Comfy-API] Node {node_id} override applied: {param_key} -> {param_val}")

            # 3. Dynamic Prefixing of SaveImage Nodes to prevent duplicate overrides
            if node.get("class_type") == "SaveImage":
                node["inputs"]["filename_prefix"] = f"abi_{str(uuid.uuid4())[:8]}"

        # Submit prompt payload
        try:
            p = {"prompt": workflow, "client_id": self.client_id}
            resp = requests.post(f"http://{self.api_address}/prompt", json=p, timeout=10.0)
            resp.raise_for_status()
            res_json = resp.json()
            prompt_id = res_json["prompt_id"]
            logs.append(f"[Comfy-API] Queued job prompt_id: {prompt_id}")
        except Exception as e:
            logs.append(f"[Comfy-API] Failed to connect or post payload: {str(e)}")
            return None, "\n".join(logs)

        # Connect WebSocket to track ComfyUI execution
        try:
            ws_url = f"ws://{self.api_address}/ws?clientId={self.client_id}"
            ws = websocket.create_connection(ws_url, timeout=120)
            logs.append("[Comfy-API] WebSocket connected. Monitoring run progress...")
        except Exception as e:
            logs.append(f"[Comfy-API] Failed to connect WebSocket: {str(e)}")
            return None, "\n".join(logs)

        output_image_info = None
        try:
            while True:
                msg = ws.recv()
                if not msg:
                    break
                if isinstance(msg, bytes):
                    # Binary preview image frames - skip
                    continue
                
                event = json.loads(msg)
                event_type = event.get("type")
                event_data = event.get("data", {})

                if event_type == "executing":
                    node_executing = event_data.get("node")
                    p_id = event_data.get("prompt_id")
                    if p_id == prompt_id:
                        if node_executing is None:
                            logs.append("[Comfy-API] Workflow execution completed.")
                            break
                        else:
                            logs.append(f"[Comfy-API] Running Node ID: {node_executing}")
                
                elif event_type == "executed":
                    p_id = event_data.get("prompt_id")
                    if p_id == prompt_id and "output" in event_data:
                        output = event_data["output"]
                        if "images" in output:
                            output_image_info = output["images"][0]
                            logs.append(f"[Comfy-API] Node {event_data.get('node')} produced output image: {output_image_info['filename']}")
        except Exception as e:
            logs.append(f"[Comfy-API] Error during execution polling: {str(e)}")
        finally:
            try:
                ws.close()
            except Exception:
                pass

        if not output_image_info:
            logs.append("[Comfy-API] Completed without discovering any output images.")
            return None, "\n".join(logs)

        # Fetch completed image bytes directly over ComfyUI's native /view API
        try:
            view_params = {
                "filename": output_image_info["filename"],
                "subfolder": output_image_info.get("subfolder", ""),
                "type": output_image_info.get("type", "output")
            }
            view_url = f"http://{self.api_address}/view"
            img_resp = requests.get(view_url, params=view_params, timeout=15.0)
            img_resp.raise_for_status()
            logs.append("[Comfy-API] Downloaded output image bytes successfully.")
            return img_resp.content, "\n".join(logs)
        except Exception as e:
            logs.append(f"[Comfy-API] Failed to retrieve output image bytes: {str(e)}")
            return None, "\n".join(logs)