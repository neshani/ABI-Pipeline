from nicegui import ui
import asyncio
import httpx
from pathlib import Path
import json
from database.connection import set_setting, get_setting
from services.installer import (
    check_dependencies,
    check_model_downloaded,
    run_pip_install,
    download_model_weights
)

class OnboardingWizard:
    def __init__(self, app_settings: dict, on_complete_callback=None, launch_comfy_callback=None) -> None:
        self.settings = app_settings
        self.on_complete = on_complete_callback
        self.launch_comfy = launch_comfy_callback
        self.step = 1
        
        # Scanner states
        self.scanning_llm = False
        self.scan_results = []
        self.detected_checkpoints = []
        self.suggested_workflow = "None"
        self.models_list = []
        self.scanning_models = False
        self.scanning_comfy = False
        self.comfy_online = False
        
        # STT installation states inside wizard
        self.dep_status = {"status": False, "missing": []}
        self.model_status = False
        self.installing = False
        self.download_progress = 0.0
        
        # Buffer for wizard changes (written to DB only on completion)
        self.temp_settings = {
            "stt_engine": self.settings.get("stt_engine", "Parakeet ONNX"),
            "stt_device": self.settings.get("stt_device", "GPU/CUDA"),
            "llm_url": self.settings.get("llm_url", "http://127.0.0.1:11434"),
            "llm_model": self.settings.get("llm_model", "local-model"),
            "comfy_path": self.settings.get("comfy_path", ""),
            "comfy_url": self.settings.get("comfy_url", "http://127.0.0.1:8188"),
            "selected_starters": ["sdxl_base", "sdxl_lora", "z_image_turbo", "z_image_turbo_lora", "anima_lora"],
            "wizard_completed": False
        }

        # Run an initial check of package states on initialization
        self.update_installation_statuses()

        # Build persistent dialogue overlay
        with ui.dialog().props('persistent') as self.dialog, ui.card().classes('w-full max-w-2xl p-6 rounded-2xl shadow-xl bg-white'):
            self.wizard_ui()

    def open(self) -> None:
        self.step = 1
        self.update_installation_statuses()
        self.wizard_ui.refresh()
        self.dialog.open()
        # Trigger an LLM scan automatically on start
        asyncio.create_task(self.scan_llm_endpoints())

    def close(self) -> None:
        self.dialog.close()

    def update_installation_statuses(self) -> None:
        """Checks if selected STT engine packages and models are fully downloaded."""
        engine = self.temp_settings.get("stt_engine", "Parakeet ONNX")
        device = self.temp_settings.get("stt_device", "GPU/CUDA")
        if engine == "Text Only":
            self.dep_status = {"status": True, "missing": []}
            self.model_status = True
        else:
            self.dep_status = check_dependencies(engine, device)
            self.model_status = check_model_downloaded(engine, device)

    async def execute_installation_pipeline(self, client) -> None:
        """
        Performs Python package installs and downloads model weights sequentially inside the wizard.
        Delegates completely to the centralized services/installer.py module to ensure 
        locked version constraints are enforced during package deployments.
        """
        self.installing = True
        
        with client:
            self.wizard_ui.refresh()
            
            if hasattr(self, 'terminal_log') and self.terminal_log:
                self.terminal_log.clear()
            if hasattr(self, 'progress_bar') and self.progress_bar:
                self.progress_bar.set_value(0.0)
                
            engine = self.temp_settings.get("stt_engine", "Parakeet ONNX")
            device = self.temp_settings.get("stt_device", "GPU/CUDA")
            
            # Step 1: Install Python Libraries (if missing)
            # Centralized version constraints inside services/installer.py automatically translate 
            # these raw dependency strings into deployment-ready, frozen packages.
            if not self.dep_status["status"]:
                success = await run_pip_install(self.dep_status["missing"], self.write_to_terminal)
                if not success:
                    ui.notify("Installation failed during library deployment.", type="negative")
                    self.installing = False
                    self.update_installation_statuses()
                    self.wizard_ui.refresh()
                    return
                
            # Step 2: Download Model Weights (if missing)
            if not self.model_status:
                success = await download_model_weights(
                    engine, 
                    device,
                    self.update_download_progress, 
                    self.write_to_terminal
                )
                if not success:
                    ui.notify("Download failed. Check your internet connection.", type="negative")
                    self.installing = False
                    self.update_installation_statuses()
                    self.wizard_ui.refresh()
                    return

            ui.notify("Engine setup successfully completed!", type="positive")
            self.installing = False
            self.update_installation_statuses()
            self.wizard_ui.refresh()
            
            if len(self.dep_status["missing"]) > 0:
                ui.notify("Libraries containing binaries installed. Restart recommended.", type="warning", timeout=10)

    def write_to_terminal(self, text: str) -> None:
        """Pipes subprocess outputs into the terminal widget inside the active step."""
        if hasattr(self, 'terminal_log') and self.terminal_log:
            self.terminal_log.push(text)

    def update_download_progress(self, val: float) -> None:
        """Updates linear download progress bar state."""
        self.download_progress = val
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.set_value(val)

    def ensure_default_workflows(self) -> None:
        """Pre-packages default ComfyUI JSON workflows to ./workflows/ directory based on user selections."""
        wf_dir = Path("./workflows")
        wf_dir.mkdir(exist_ok=True)
        
        # Pull selected starters list to determine what gets saved
        starters = self.temp_settings.get("selected_starters", [])
        
        default_flows = {}

        if "sdxl_base" in starters:
            default_flows["sdxl_base.json"] = {
              "5": {
                "inputs": { "width": 1024, "height": 1024, "batch_size": 1 },
                "class_type": "EmptyLatentImage",
                "_meta": { "title": "Empty Latent Image" }
              },
              "6": {
                "inputs": { "text": "<prompt>", "clip": ["20", 1] },
                "class_type": "CLIPTextEncode",
                "_meta": { "title": "CLIP Text Encode (Prompt)" }
              },
              "7": {
                "inputs": { "text": "<negprompt>", "clip": ["20", 1] },
                "class_type": "CLIPTextEncode",
                "_meta": { "title": "CLIP Text Encode (Prompt)" }
              },
              "8": {
                "inputs": { "samples": ["13", 0], "vae": ["20", 2] },
                "class_type": "VAEDecode",
                "_meta": { "title": "VAE Decode" }
              },
              "13": {
                "inputs": {
                  "add_noise": True,
                  "noise_seed": 345756756756,
                  "cfg": 1,
                  "model": ["20", 0],
                  "positive": ["6", 0],
                  "negative": ["7", 0],
                  "sampler": ["34", 0],
                  "sigmas": ["33", 0],
                  "latent_image": ["5", 0]
                },
                "class_type": "SamplerCustom",
                "_meta": { "title": "SamplerCustom" }
              },
              "20": {
                "inputs": { "ckpt_name": "juggernautXL_juggXILightningByRD.safetensors" },
                "class_type": "CheckpointLoaderSimple",
                "_meta": { "title": "Load Checkpoint" }
              },
              "27": {
                "inputs": { "filename_prefix": "ComfyUI", "images": ["8", 0] },
                "class_type": "SaveImage",
                "_meta": { "title": "Save Image" }
              },
              "33": {
                "inputs": { "scheduler": "simple", "steps": 10, "denoise": 1, "model": ["20", 0] },
                "class_type": "BasicScheduler",
                "_meta": { "title": "BasicScheduler" }
              },
              "34": {
                "inputs": { "sampler_name": "euler_ancestral_cfg_pp" },
                "class_type": "KSamplerSelect",
                "_meta": { "title": "KSamplerSelect" }
              }
            }

        if "sdxl_lora" in starters:
            default_flows["sdxl_lora.json"] = {
              "5": {
                "inputs": { "width": 1024, "height": 1024, "batch_size": 1 },
                "class_type": "EmptyLatentImage",
                "_meta": { "title": "Empty Latent Image" }
              },
              "6": {
                "inputs": { "text": "<prompt>", "clip": ["20", 1] },
                "class_type": "CLIPTextEncode",
                "_meta": { "title": "CLIP Text Encode (Prompt)" }
              },
              "7": {
                "inputs": { "text": "<negprompt>", "clip": ["20", 1] },
                "class_type": "CLIPTextEncode",
                "_meta": { "title": "CLIP Text Encode (Prompt)" }
              },
              "8": {
                "inputs": { "samples": ["13", 0], "vae": ["20", 2] },
                "class_type": "VAEDecode",
                "_meta": { "title": "VAE Decode" }
              },
              "13": {
                "inputs": {
                  "add_noise": True,
                  "noise_seed": 345756756756,
                  "cfg": 1,
                  "model": ["29", 0],
                  "positive": ["6", 0],
                  "negative": ["7", 0],
                  "sampler": ["34", 0],
                  "sigmas": ["33", 0],
                  "latent_image": ["5", 0]
                },
                "class_type": "SamplerCustom",
                "_meta": { "title": "SamplerCustom" }
              },
              "20": {
                "inputs": { "ckpt_name": "juggernautXL_juggXILightningByRD.safetensors" },
                "class_type": "CheckpointLoaderSimple",
                "_meta": { "title": "Load Checkpoint" }
              },
              "27": {
                "inputs": { "filename_prefix": "ComfyUI", "images": ["8", 0] },
                "class_type": "SaveImage",
                "_meta": { "title": "Save Image" }
              },
              "29": {
                "inputs": {
                  "lora_name": "",
                  "strength_model": 1,
                  "model": ["20", 0]
                },
                "class_type": "LoraLoaderModelOnly",
                "_meta": { "title": "Load LoRA" }
              },
              "33": {
                "inputs": { "scheduler": "simple", "steps": 10, "denoise": 1, "model": ["20", 0] },
                "class_type": "BasicScheduler",
                "_meta": { "title": "BasicScheduler" }
              },
              "34": {
                "inputs": { "sampler_name": "euler_ancestral_cfg_pp" },
                "class_type": "KSamplerSelect",
                "_meta": { "title": "KSamplerSelect" }
              }
            }

        if "z_image_turbo" in starters:
            default_flows["z_image_turbo.json"] = {
              "3": {
                "inputs": {
                  "seed": 0, "steps": 7, "cfg": 1,
                  "sampler_name": "euler_ancestral", "scheduler": "simple", "denoise": 1,
                  "model": ["11", 0], "positive": ["19", 0], "negative": ["7", 0],
                  "latent_image": ["13", 0]
                },
                "class_type": "KSampler",
                "_meta": { "title": "KSampler" }
              },
              "6": {
                "inputs": { "text": "<prompt>", "clip": ["18", 0] },
                "class_type": "CLIPTextEncode",
                "_meta": { "title": "CLIP Text Encode (Positive Prompt)" }
              },
              "7": {
                "inputs": { "text": "<negprompt>", "clip": ["18", 0] },
                "class_type": "CLIPTextEncode",
                "_meta": { "title": "CLIP Text Encode (Negative Prompt)" }
              },
              "8": {
                "inputs": { "samples": ["3", 0], "vae": ["17", 0] },
                "class_type": "VAEDecode",
                "_meta": { "title": "VAE Decode" }
              },
              "9": {
                "inputs": { "filename_prefix": "ComfyUI", "images": ["8", 0] },
                "class_type": "SaveImage",
                "_meta": { "title": "Save Image" }
              },
              "11": {
                "inputs": { "shift": 7, "model": ["16", 0] },
                "class_type": "ModelSamplingAuraFlow",
                "_meta": { "title": "ModelSamplingAuraFlow" }
              },
              "13": {
                "inputs": { "width": 1024, "height": 1024, "batch_size": 1 },
                "class_type": "EmptySD3LatentImage",
                "_meta": { "title": "EmptySD3LatentImage" }
              },
              "16": {
                "inputs": { "unet_name": "z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors", "weight_dtype": "default" },
                "class_type": "UNETLoader",
                "_meta": { "title": "Load Diffusion Model" }
              },
              "17": {
                "inputs": { "vae_name": "ae.safetensors" },
                "class_type": "VAELoader",
                "_meta": { "title": "Load VAE" }
              },
              "18": {
                "inputs": { "clip_name": "qwen_3_4b.safetensors", "type": "qwen_image", "device": "default" },
                "class_type": "CLIPLoader",
                "_meta": { "title": "Load CLIP" }
              },
              "19": {
                "inputs": {
                  "randomize_percent": 50, "strength": 20, "noise_insert": "noise on beginning steps",
                  "steps_switchover_percent": 40, "seed": 0, "conditioning": ["6", 0]
                },
                "class_type": "SeedVarianceEnhancer",
                "_meta": { "title": "SeedVarianceEnhancer" }
              }
            }

        if "z_image_turbo_lora" in starters:
            default_flows["z_image_turbo_lora.json"] = {
              "3": {
                "inputs": {
                  "seed": 0, "steps": 7, "cfg": 1,
                  "sampler_name": "euler_ancestral", "scheduler": "simple", "denoise": 1,
                  "model": ["20", 0], "positive": ["19", 0], "negative": ["7", 0],
                  "latent_image": ["13", 0]
                },
                "class_type": "KSampler",
                "_meta": { "title": "KSampler" }
              },
              "6": {
                "inputs": { "text": "<prompt>", "clip": ["20", 1] },
                "class_type": "CLIPTextEncode",
                "_meta": { "title": "CLIP Text Encode (Positive Prompt)" }
              },
              "7": {
                "inputs": { "text": "<negprompt>", "clip": ["20", 1] },
                "class_type": "CLIPTextEncode",
                "_meta": { "title": "CLIP Text Encode (Negative Prompt)" }
              },
              "8": {
                "inputs": { "samples": ["3", 0], "vae": ["17", 0] },
                "class_type": "VAEDecode",
                "_meta": { "title": "VAE Decode" }
              },
              "9": {
                "inputs": { "filename_prefix": "ComfyUI", "images": ["8", 0] },
                "class_type": "SaveImage",
                "_meta": { "title": "Save Image" }
              },
              "11": {
                "inputs": { "shift": 7, "model": ["16", 0] },
                "class_type": "ModelSamplingAuraFlow",
                "_meta": { "title": "ModelSamplingAuraFlow" }
              },
              "13": {
                "inputs": { "width": 1024, "height": 1024, "batch_size": 1 },
                "class_type": "EmptySD3LatentImage",
                "_meta": { "title": "EmptySD3LatentImage" }
              },
              "16": {
                "inputs": { "unet_name": "z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors", "weight_dtype": "default" },
                "class_type": "UNETLoader",
                "_meta": { "title": "Load Diffusion Model" }
              },
              "17": {
                "inputs": { "vae_name": "ae.safetensors" },
                "class_type": "VAELoader",
                "_meta": { "title": "Load VAE" }
              },
              "18": {
                "inputs": { "clip_name": "qwen_3_4b.safetensors", "type": "qwen_image", "device": "default" },
                "class_type": "CLIPLoader",
                "_meta": { "title": "Load CLIP" }
              },
              "19": {
                "inputs": {
                  "randomize_percent": 50, "strength": 20, "noise_insert": "noise on beginning steps",
                  "steps_switchover_percent": 40, "seed": 0, "conditioning": ["6", 0]
                },
                "class_type": "SeedVarianceEnhancer",
                "_meta": { "title": "SeedVarianceEnhancer" }
              },
              "20": {
                "inputs": {
                  "lora_name": "",
                  "strength_model": 0.5, "strength_clip": 1,
                  "model": ["11", 0], "clip": ["18", 0]
                },
                "class_type": "LoraLoader",
                "_meta": { "title": "Load LoRA" }
              }
            }

        if "anima_lora" in starters:
            default_flows["anima_lora.json"] = {
              "46": {
                "inputs": { "filename_prefix": "Anima", "images": ["60:8", 0] },
                "class_type": "SaveImage",
                "_meta": { "title": "Save Image" }
              },
              "60:45": {
                "inputs": { "clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default" },
                "class_type": "CLIPLoader",
                "_meta": { "title": "Load CLIP" }
              },
              "60:15": {
                "inputs": { "vae_name": "qwen_image_vae.safetensors" },
                "class_type": "VAELoader",
                "_meta": { "title": "Load VAE" }
              },
              "60:8": {
                "inputs": { "samples": ["60:19", 0], "vae": ["60:15", 0] },
                "class_type": "VAEDecode",
                "_meta": { "title": "VAE Decode" }
              },
              "60:28": {
                "inputs": { "width": 1024, "height": 1024, "batch_size": 1 },
                "class_type": "EmptyLatentImage",
                "_meta": { "title": "Empty Latent Image" }
              },
              "60:12": {
                "inputs": {
                  "text": "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
                  "clip": ["60:45", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": { "title": "CLIP Text Encode (Negative Prompt)" }
              },
              "60:19": {
                "inputs": {
                  "seed": 0, "steps": 12, "cfg": 1, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1,
                  "model": ["60:72", 0], "positive": ["60:11", 0], "negative": ["60:12", 0], "latent_image": ["60:28", 0]
                },
                "class_type": "KSampler",
                "_meta": { "title": "KSampler" }
              },
              "60:44": {
                "inputs": { "unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default" },
                "class_type": "UNETLoader",
                "_meta": { "title": "Load Diffusion Model" }
              },
              "60:11": {
                "inputs": { "text": "<prompt>", "clip": ["60:45", 0] },
                "class_type": "CLIPTextEncode",
                "_meta": { "title": "CLIP Text Encode (Positive Prompt)" }
              },
              "60:72": {
                "inputs": {
                  "lora_name": "",
                  "strength_model": 0.9,
                  "model": ["60:44", 0]
                },
                "class_type": "LoraLoaderModelOnly",
                "_meta": { "title": "Load LoRA" }
              }
            }
        
        for name, data in default_flows.items():
            file_path = wf_dir / name
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

    async def scan_llm_endpoints(self) -> None:
        """Pings potential local loopback endpoints in parallel to find active server wrappers."""
        self.scanning_llm = True
        self.scan_results = []
        self.wizard_ui.refresh()
        
        scans = [
            {"name": "LM Studio", "url": "http://localhost:1234", "test_path": "/v1/models"},
            {"name": "Ollama", "url": "http://localhost:11434", "test_path": "/api/tags"},
            {"name": "llama-server", "url": "http://localhost:8080", "test_path": "/v1/models"}
        ]
        
        async with httpx.AsyncClient() as client:
            for item in scans:
                try:
                    res = await client.get(f"{item['url']}{item['test_path']}", timeout=0.8)
                    if res.status_code == 200:
                        self.scan_results.append({"name": item["name"], "url": item["url"], "status": "Online"})
                    else:
                        self.scan_results.append({"name": item["name"], "url": item["url"], "status": "Offline"})
                except Exception:
                    self.scan_results.append({"name": item["name"], "url": item["url"], "status": "Offline"})
                    
        self.scanning_llm = False
        
        # Auto-populate the URL with the first discovered active service
        active = [r for r in self.scan_results if r["status"] == "Online"]
        if active:
            self.temp_settings["llm_url"] = active[0]["url"]
            asyncio.create_task(self.scan_llm_models())
            
        self.wizard_ui.refresh()

    async def scan_llm_models(self) -> None:
        """Queries the currently resolved LLM Base URL for its active model strings."""
        url = self.temp_settings.get("llm_url", "").strip().rstrip("/")
        if not url:
            return
            
        self.scanning_models = True
        self.models_list = []
        self.wizard_ui.refresh()
        
        headers = {}
        async with httpx.AsyncClient() as client:
            # 1. Try standard OpenAI compatible path
            try:
                res = await client.get(f"{url}/v1/models", headers=headers, timeout=2.0)
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, dict) and "data" in data:
                        self.models_list = [m["id"] for m in data["data"] if "id" in m]
            except Exception:
                pass
                
            # 2. Try Ollama direct path if models list is still empty
            if not self.models_list:
                try:
                    res = await client.get(f"{url}/api/tags", timeout=2.0)
                    if res.status_code == 200:
                        data = res.json()
                        if isinstance(data, dict) and "models" in data:
                            self.models_list = [m["name"] for m in data["models"] if "name" in m]
                except Exception:
                    pass
                    
        if self.models_list:
            self.models_list = sorted(self.models_list)
            if self.temp_settings["llm_model"] not in self.models_list:
                self.temp_settings["llm_model"] = self.models_list[0]
        else:
            self.models_list = ["local-model"]
            self.temp_settings["llm_model"] = "local-model"
            
        self.scanning_models = False
        self.wizard_ui.refresh()

    async def async_scan_comfy_api_models(self) -> None:
        """Connects directly to active ComfyUI backend port to retrieve active available models."""
        comfy_url = self.temp_settings.get("comfy_url", "http://127.0.0.1:8188").strip()
        if "http" not in comfy_url:
            comfy_url = f"http://{comfy_url}"
        comfy_url = comfy_url.replace("http://", "").replace("https://", "").strip("/")
        
        self.scanning_comfy = True
        self.comfy_online = False
        self.detected_checkpoints = []
        self.suggested_workflow = "None"
        self.wizard_ui.refresh()
        
        async with httpx.AsyncClient() as client:
            # 1. Verify if ComfyUI is online
            try:
                res = await client.get(f"http://{comfy_url}/system_stats", timeout=1.5)
                if res.status_code == 200:
                    self.comfy_online = True
            except Exception:
                self.comfy_online = False
                
            if self.comfy_online:
                found_models = []
                # Fetch standard checkpoint loader list
                try:
                    res = await client.get(f"http://{comfy_url}/object_info/CheckpointLoaderSimple", timeout=2.0)
                    if res.status_code == 200:
                        data = res.json()
                        choices = data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
                        if isinstance(choices, list):
                            found_models.extend(choices)
                except Exception:
                    pass
                    
                # Fetch UNET loaders (for ZIT & Anima integrations)
                try:
                    res = await client.get(f"http://{comfy_url}/object_info/UNETLoader", timeout=2.0)
                    if res.status_code == 200:
                        data = res.json()
                        choices = data.get("UNETLoader", {}).get("input", {}).get("required", {}).get("unet_name", [[]])[0]
                        if isinstance(choices, list):
                            found_models.extend(choices)
                except Exception:
                    pass
                    
                self.detected_checkpoints = sorted(list(set(found_models)))
                
                # Predict optimal default workflow template
                if self.detected_checkpoints:
                    first = self.detected_checkpoints[0].lower()
                    if "juggernaut" in first or "lightning" in first:
                        self.suggested_workflow = "sdxl_lora.json"
                    elif "z-image" in first or "zit" in first:
                        self.suggested_workflow = "z_image_turbo_lora.json"
                    elif "anima" in first:
                        self.suggested_workflow = "anima_lora.json"
                    else:
                        self.suggested_workflow = "sdxl_lora.json"

        self.scanning_comfy = False
        self.wizard_ui.refresh()

    def trigger_launch_comfy(self) -> None:
        """Persists custom path directory and runs background ComfyUI launcher callback."""
        set_setting("comfy_path", self.temp_settings["comfy_path"])
        self.settings["comfy_path"] = self.temp_settings["comfy_path"]
        
        if self.launch_comfy:
            self.launch_comfy()
            ui.notify("Startup signal dispatched to ComfyUI launcher process...", type="info")

    def save_and_complete(self) -> None:
        """Persists choices into database backend, unpacks chosen starter workflows, and transitions wizard flag."""
        self.temp_settings["wizard_completed"] = True
        
        # Perform physical file unpacking only for checked starter workflows
        self.ensure_default_workflows()
        
        # Apply all buffered configurations
        for k, v in self.temp_settings.items():
            if k == "selected_starters":
                continue  # Skip raw selection array (not an active configuration string)
            set_setting(k, v)
            self.settings[k] = v
            
        ui.notify("Onboarding setup finalized!", type="positive")
        self.close()
        if self.on_complete:
            self.on_complete()

    @ui.refreshable
    def wizard_ui(self) -> None:
        # Header banner
        with ui.row().classes('w-full items-center justify-between border-b pb-4 mb-4'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('auto_awesome', size='md', color='blue-500')
                ui.label('ABI-Pipeline Setup Wizard').classes('text-xl font-bold text-slate-800')
            ui.label(f"Step {self.step} of 4").classes('text-xs font-bold text-slate-400 uppercase bg-slate-100 px-2 py-1 rounded')

        # Steps body container with bounded height & native scroll avoids layout overlaps
        with ui.column().classes('w-full h-[410px] overflow-y-auto justify-start gap-4 pr-1'):
            if self.step == 1:
                self.render_step_1()
            elif self.step == 2:
                self.render_step_2()
            elif self.step == 3:
                self.render_step_3()
            elif self.step == 4:
                self.render_step_4()

        # Footer Actions
        with ui.row().classes('w-full justify-between items-center border-t pt-4 mt-4'):
            # Back Button
            if self.step > 1:
                ui.button('Back', on_click=self.prev_step).props('flat color=slate').classes('text-slate-600')
            else:
                ui.label('') # Spacer
                
            # Next / Finish Button
            if self.step < 4:
                ui.button('Next Step', on_click=self.next_step).classes('bg-blue-600 text-white font-semibold px-6')
            else:
                ui.button('Complete Setup', on_click=self.save_and_complete).classes('bg-emerald-600 text-white font-bold px-8')

    def next_step(self) -> None:
        self.step += 1
        if self.step == 2:
            asyncio.create_task(self.scan_llm_endpoints())
        elif self.step == 3:
            # Run background scanner for checkpoints, do not block step transition
            asyncio.create_task(self.async_scan_comfy_api_models())
        self.wizard_ui.refresh()

    def prev_step(self) -> None:
        self.step -= 1
        self.wizard_ui.refresh()

    # Step 1: Transcription
    def render_step_1(self) -> None:
        ui.label('Choose Speech-to-Text / Transcription Strategy').classes('text-md font-bold text-slate-700')
        ui.label('Select how you plan to ingest audiobook files. You can entirely skip audio dependency setup if your project only uses EPUB or raw text files.').classes('text-xs text-slate-500')
        
        with ui.column().classes('w-full gap-3 mt-2'):
            # Audio Engine Choices
            def select_stt(engine_name: str):
                self.temp_settings["stt_engine"] = engine_name
                self.update_installation_statuses()
                self.wizard_ui.refresh()

            engines = [
                {
                    "id": "Parakeet ONNX",
                    "title": "Parakeet ONNX (Recommended)",
                    "desc": "Ultra-fast parallel sequencing. Transcribes ~20 hours in ~5 minutes on GPU. Small footprint.",
                    "icon": "bolt"
                },
                {
                    "id": "Whisper",
                    "title": "Faster-Whisper (Backup Engine)",
                    "desc": "Highly detailed phrase-level timings. Processes sequentially. Larger disk space footprint.",
                    "icon": "hourglass_empty"
                },
                {
                    "id": "Text Only",
                    "title": "Skip Transcription / Text-Only Setup",
                    "desc": "Bypass audio library checks entirely. Perfect if you only import pre-existing text or EPUB novels.",
                    "icon": "edit_note"
                }
            ]

            for eng in engines:
                active = self.temp_settings["stt_engine"] == eng["id"]
                border_color = 'border-blue-500 bg-blue-50/40 text-blue-900' if active else 'border-slate-200 hover:bg-slate-50 text-slate-700'
                
                with ui.card().classes(f'w-full p-4 border rounded-xl shadow-xs cursor-pointer transition-all {border_color}') \
                        .on('click', lambda name=eng["id"]: select_stt(name)):
                    with ui.row().classes('items-center gap-3 w-full'):
                        ui.icon(eng["icon"], size='sm', color='blue-500' if active else 'slate-400')
                        with ui.column().classes('flex-1 gap-0.5'):
                            ui.label(eng["title"]).classes('text-sm font-bold')
                            ui.label(eng["desc"]).classes('text-[11px] text-slate-500 leading-tight')

            # Hardware Device target (Only visible if not text-only skip mode)
            if self.temp_settings["stt_engine"] != "Text Only":
                ui.separator().classes('my-2')
                ui.label('Select Hardware Acceleration Target').classes('text-xs font-bold text-slate-500 uppercase tracking-wider')
                
                def on_device_change(e):
                    self.temp_settings["stt_device"] = e.value
                    self.update_installation_statuses()
                    self.wizard_ui.refresh()

                ui.radio(
                    options={
                        'GPU/CUDA': 'GPU / CUDA (Recommended for Nvidia GPU systems)',
                        'CPU': 'CPU (Not recommended - Slow)'
                    },
                    value=self.temp_settings["stt_device"],
                    on_change=on_device_change
                ).classes('text-sm')

                # Real-Time Installation Status Indicators
                ui.separator().classes('my-1')
                with ui.column().classes('gap-1 mt-1'):
                    if self.dep_status["status"]:
                        ui.label("✓ Python dependencies installed.").classes('text-xs text-emerald-600 font-medium')
                    else:
                        missing_str = ", ".join(self.dep_status["missing"])
                        ui.label(f"✗ Missing packages: {missing_str}").classes('text-xs text-rose-500 font-medium')

                    if self.model_status:
                        ui.label("✓ Model weights downloaded locally.").classes('text-xs text-emerald-600 font-medium')
                    else:
                        ui.label("✗ Model weights missing.").classes('text-xs text-rose-500 font-medium')

                # Installation controls
                ready = self.dep_status["status"] and self.model_status
                btn_text = "Engine Ready" if ready else "Install & Download Engine"
                btn_color = "bg-emerald-600" if ready else "bg-blue-600"
                
                self.action_btn = ui.button(
                    btn_text, 
                    on_click=lambda: asyncio.create_task(self.execute_installation_pipeline(ui.context.client))
                ).classes(f'w-full mt-2 {btn_color} text-white rounded-lg text-sm')
                
                if ready or self.installing:
                    self.action_btn.disable()

                self.progress_bar = ui.linear_progress(value=self.download_progress).classes('w-full mt-1')

                # Installation Console Log Output
                ui.label('Installation Console Output').classes('text-[11px] font-bold text-slate-500 mt-2')
                self.terminal_log = ui.log().classes('h-28 w-full bg-slate-950 p-2 text-emerald-400 font-mono text-[10px] rounded-lg')

    # Step 2: LLM Configurer
    def render_step_2(self) -> None:
        ui.label('Verify Local LLM Server').classes('text-md font-bold text-slate-700')
        ui.label('Pings typical local AI host port configurations to automate system configurations. Ensure LM Studio, Ollama, or llama-server is currently active.').classes('text-xs text-slate-500')
        
        with ui.column().classes('w-full gap-3 mt-2'):
            # Scanner box
            with ui.card().classes('w-full bg-slate-50 border p-4 rounded-xl gap-2'):
                with ui.row().classes('w-full justify-between items-center mb-1'):
                    ui.label('Autodetected Engines').classes('text-xs font-bold text-slate-500 uppercase tracking-wider')
                    if self.scanning_llm:
                        ui.spinner(size='xs').classes('text-blue-500')
                    else:
                        ui.button('Scan Again', icon='refresh', on_click=self.scan_llm_endpoints).props('flat dense').classes('text-xs text-blue-600')

                # Render dynamic discovery rows
                if not self.scan_results:
                    ui.label('No scan ran yet or port mapping is empty.').classes('text-xs text-slate-400')
                else:
                    for res in self.scan_results:
                        is_online = res["status"] == "Online"
                        badge_bg = 'bg-emerald-100 text-emerald-800' if is_online else 'bg-slate-200 text-slate-600'
                        with ui.row().classes('w-full items-center justify-between text-xs py-1 border-b border-slate-100/50 last:border-0'):
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('circle', size='8px', color='emerald-500' if is_online else 'slate-300')
                                ui.label(res["name"]).classes('font-bold text-slate-700')
                                ui.label(res["url"]).classes('text-[11px] text-slate-400 font-mono')
                            ui.badge(res["status"]).classes(f'px-2 py-0.5 rounded-full font-bold text-[10px] uppercase {badge_bg}')

            # Direct input override and testing row
            with ui.row().classes('w-full items-end gap-3'):
                ui.input(
                    'LLM API Host URL',
                    placeholder='e.g., http://localhost:11434'
                ).bind_value(self.temp_settings, 'llm_url').classes('flex-1')
                
                ui.button(
                    'Test & Scan Models',
                    icon='network_ping',
                    on_click=lambda: asyncio.create_task(self.scan_llm_models())
                ).classes('bg-blue-600 text-white font-semibold')

            # Model Dropdown Selection
            with ui.row().classes('w-full items-center justify-between gap-3'):
                if self.scanning_models:
                    with ui.row().classes('items-center gap-2 flex-1'):
                        ui.spinner(size='xs')
                        ui.label('Scanning endpoint for models...').classes('text-xs text-slate-500')
                else:
                    ui.select(
                        options=self.models_list if self.models_list else ["local-model"],
                        label='Selected Target Model'
                    ).bind_value(self.temp_settings, 'llm_model').classes('flex-1')

    # Step 3: ComfyUI Configuration
    def render_step_3(self) -> None:
        ui.label('Connect ComfyUI & Unpack Workflows').classes('text-md font-bold text-slate-700')
        ui.label('Configure your ComfyUI setup. This tool lets you target your books with your favorite artistic render pipelines.').classes('text-xs text-slate-500')
        
        with ui.column().classes('w-full gap-3 mt-2'):
            # Educational Framed Instructions
            with ui.card().classes('w-full p-4 bg-blue-50/50 border border-blue-100 rounded-xl gap-2'):
                with ui.row().classes('items-center gap-2 text-blue-900 font-bold text-xs'):
                    ui.icon('info', size='18px', color='blue')
                    ui.label('Integrating Your Own Custom Workflows')
                
                ui.markdown(
                    "The core goal is that you can use your favorite ComfyUI workflows in ABI-Pipeline. "
                    "To do that, simply:\n\n"
                    "1. Configure your prompt text parameter inside ComfyUI to contain exactly `<prompt>` and negative prompt to `<negprompt>`.\n"
                    "2. Go to **File -> Export (API)** (may require enabling dev mode in settings).\n"
                    "3. Save that output `.json` file directly inside the `./workflows/` folder of this project.\n\n"
                    "Centralized services/installer.py is used to secure all python environments, and metadata baking is handled internally."
                ).classes('text-[11px] text-slate-600 leading-normal')

            # Location Path & API Target input rows
            with ui.row().classes('w-full gap-3'):
                ui.input(
                    'ComfyUI Base Directory Path',
                    placeholder='e.g., F:/AI/ComfyUI/ComfyUI'
                ).bind_value(self.temp_settings, 'comfy_path').classes('flex-1')
                
                ui.input(
                    'ComfyUI API Host Port URL',
                    placeholder='e.g., http://127.0.0.1:8188'
                ).bind_value(self.temp_settings, 'comfy_url').classes('w-56')

            # Starter Workflow Checklist Box
            ui.separator().classes('my-1')
            ui.label("Pre-packaged Starter Workflows").classes('text-xs font-bold text-slate-500 uppercase tracking-wider')
            ui.label("If you do not have your own workflows or want example setups that connect to ABI-Pipeline, check which starter workflows to unpack:").classes('text-[11px] text-slate-400 -mt-2')

            with ui.card().classes('w-full p-4 border rounded-xl bg-slate-50 gap-2.5'):
                def toggle_starter(slug: str, val: bool):
                    starters = self.temp_settings["selected_starters"]
                    if val and slug not in starters:
                        starters.append(slug)
                    elif not val and slug in starters:
                        starters.remove(slug)

                starters_config = [
                    ("sdxl_base", "SDXL Base", "Clean standard SDXL workflow without a LoRA node. Fast and straightforward."),
                    ("sdxl_lora", "SDXL + LoRA", "SDXL pipeline pre-configured with a LoRA loader and standard latent parameters."),
                    ("z_image_turbo", "Z-Image Turbo", "High-efficiency Z-Image / AuraFlow pipeline without a LoRA node."),
                    ("z_image_turbo_lora", "Z-Image Turbo + LoRA", "Z-Image / AuraFlow pipeline pre-configured with an active LoRA loader node."),
                    ("anima_lora", "Anima + LoRA", "Specialized Qwen/Anima-based generation pipeline utilizing an active LoRA loader.")
                ]

                for slug, title, desc in starters_config:
                    is_checked = slug in self.temp_settings["selected_starters"]
                    with ui.row().classes('w-full items-start gap-3 py-1 border-b border-slate-100 last:border-0'):
                        ui.checkbox(value=is_checked, on_change=lambda e, s=slug: toggle_starter(s, e.value)).classes('mt-1')
                        with ui.column().classes('gap-0.5 flex-1'):
                            ui.label(title).classes('text-xs font-bold text-slate-700')
                            ui.label(desc).classes('text-[10px] text-slate-500 leading-snug')

    # Step 4: Summary Confirmation
    def render_step_4(self) -> None:
        ui.label('Ready to Build Your Workspace').classes('text-md font-bold text-slate-700')
        ui.label('Your system settings have been configured successfully. Let’s double check your configurations:').classes('text-xs text-slate-500')
        
        with ui.column().classes('w-full gap-2 mt-2 bg-slate-50 p-4 border rounded-xl text-xs text-slate-700'):
            
            with ui.row().classes('w-full justify-between py-1.5 border-b border-slate-200/60'):
                ui.label('Transcription Engine:').classes('font-bold text-slate-500')
                ui.label(self.temp_settings["stt_engine"]).classes('font-bold text-slate-800')
                
            if self.temp_settings["stt_engine"] != "Text Only":
                with ui.row().classes('w-full justify-between py-1.5 border-b border-slate-200/60'):
                    ui.label('Hardware Target:').classes('font-bold text-slate-500')
                    ui.label(self.temp_settings["stt_device"]).classes('font-semibold')
                    
            with ui.row().classes('w-full justify-between py-1.5 border-b border-slate-200/60'):
                ui.label('LLM API Connection:').classes('font-bold text-slate-500')
                ui.label(self.temp_settings["llm_url"]).classes('font-semibold font-mono')
                
            with ui.row().classes('w-full justify-between py-1.5 border-b border-slate-200/60'):
                ui.label('Target LLM Model:').classes('font-bold text-slate-500')
                ui.label(self.temp_settings["llm_model"]).classes('font-semibold')
                
            with ui.row().classes('w-full justify-between py-1.5 border-b border-slate-200/60'):
                ui.label('ComfyUI Path:').classes('font-bold text-slate-500')
                ui.label(self.temp_settings["comfy_path"] if self.temp_settings["comfy_path"] else "Not configured").classes('font-semibold truncate max-w-[300px]')
                
            with ui.row().classes('w-full justify-between py-1.5 last:border-0'):
                ui.label('ComfyUI URL:').classes('font-bold text-slate-500')
                ui.label(self.temp_settings["comfy_url"]).classes('font-semibold font-mono')

        with ui.row().classes('w-full justify-start items-center gap-2 text-emerald-700 bg-emerald-50/50 p-3 rounded-lg border border-emerald-100/50 mt-3 text-xs'):
            ui.icon('check_circle', size='18px')
            ui.label("Everything looks great! Click below to complete setup and activate the pipeline roadmap.").classes('font-semibold')