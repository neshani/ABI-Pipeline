from nicegui import ui, app
import asyncio
import httpx
import sys
from database.connection import set_setting
from services.installer import (
    check_dependencies, 
    check_model_downloaded, 
    run_pip_install, 
    download_model_weights
)
from ui import state

async def fetch_llm_models(url: str, api_key: str = "") -> list[str]:
    """Queries standard local or remote endpoints for available model IDs."""
    base_url = url.rstrip("/")
    if base_url.endswith("/v1"):
        openai_models_url = f"{base_url}/models"
    else:
        openai_models_url = f"{base_url}/v1/models"
        
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    async with httpx.AsyncClient() as client:
        # 1. Attempt standard OpenAI-compatible API lookup
        try:
            response = await client.get(openai_models_url, headers=headers, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                    return sorted([m["id"] for m in data["data"] if "id" in m])
        except Exception:
            pass
        
        # 2. Fallback to direct Ollama engine endpoints
        ollama_tags_url = f"{base_url}/api/tags"
        try:
            response = await client.get(ollama_tags_url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
                    return sorted([m["name"] for m in data["models"] if "name" in m])
        except Exception:
            pass
            
    return []


class SettingsModal:
    def __init__(self, app_settings: dict, restart_callback) -> None:
        self.settings = app_settings
        self.restart_app = restart_callback
        
        # Live reactive states for the STT Installer
        self.dep_status = {"status": False, "missing": []}
        self.model_status = False
        self.installing = False
        self.download_progress = 0.0
        
        # Build the Dialog UI element
        with ui.dialog() as self.dialog, ui.card().classes('w-full max-w-xl p-6 rounded-xl'):
            self.build_ui()
            
        # Run a status check as soon as we initialize
        self.update_installation_statuses()

    def update_installation_statuses(self) -> None:
        """Query backend to check if chosen engine is fully installed and ready."""
        engine = self.settings.get("stt_engine", "Parakeet ONNX")
        device = self.settings.get("stt_device", "CPU")
        self.dep_status = check_dependencies(engine, device)
        self.model_status = check_model_downloaded(engine, device)
        self.update_ui_elements()

    def update_ui_elements(self) -> None:
        """Refresh individual UI elements depending on the status values."""
        # Dependency Indicator
        if self.dep_status["status"]:
            self.dep_label.set_text("✓ Python dependencies installed.")
            self.dep_label.classes('text-emerald-600', remove='text-rose-500 text-slate-500')
        else:
            missing_str = ", ".join(self.dep_status["missing"])
            self.dep_label.set_text(f"— Missing packages: {missing_str}")
            self.dep_label.classes('text-rose-500', remove='text-emerald-600 text-slate-500')
            
        # Model File Indicator
        if self.model_status:
            self.model_label.set_text("✓ Model weights downloaded locally.")
            self.model_label.classes('text-emerald-600', remove='text-rose-500 text-slate-500')
        else:
            self.model_label.set_text("— Model weights missing.")
            self.model_label.classes('text-rose-500', remove='text-emerald-600 text-slate-500')

        # Main Button Controller
        ready = self.dep_status["status"] and self.model_status
        if ready:
            self.action_btn.set_text("Engine Ready")
            self.action_btn.disable()
            self.action_btn.classes('bg-emerald-600', remove='bg-blue-600 bg-amber-600')
        else:
            self.action_btn.set_text("Install & Download Engine")
            self.action_btn.enable()
            self.action_btn.classes('bg-blue-600', remove='bg-emerald-600 bg-amber-600')

    async def execute_installation_pipeline(self) -> None:
        """Background process that installs dependencies and downloads models without freezing UI."""
        self.installing = True
        self.action_btn.disable()
        self.terminal_log.clear()
        self.progress_bar.set_value(0.0)
        
        engine = self.settings.get("stt_engine", "Parakeet ONNX")
        device = self.settings.get("stt_device", "CPU")
        
        # Step 1: Install Python Libraries (if missing)
        had_missing_deps = len(self.dep_status["missing"]) > 0
        if not self.dep_status["status"]:
            success = await run_pip_install(self.dep_status["missing"], self.write_to_terminal)
            if not success:
                ui.notify("Installation failed during library deployment.", type="negative")
                self.installing = False
                self.update_installation_statuses()
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
                return

        # Success!
        ui.notify("Engine set up successfully!", type="positive")
        self.installing = False
        self.update_installation_statuses()
        
        # Flag a clean restart requirement if new libraries containing DLLs/binaries were deployed
        if had_missing_deps:
            state.needs_restart = True
            ui.notify("Libraries containing binaries installed. Restart recommended.", type="warning", timeout=10)

    async def refresh_llm_models(self) -> None:
        """Queries current connection details and updates options in the model selector."""
        url = self.settings.get("llm_url", "")
        api_key = self.settings.get("llm_api_key", "")
        
        if not url.strip():
            ui.notify("Please enter a valid LLM API URL first.", type="warning")
            return
            
        ui.notify("Querying server for available models...", type="info")
        models = await fetch_llm_models(url, api_key)
        
        if models:
            self.model_select.options = models
            current = self.settings.get("llm_model", "")
            if current not in models:
                self.settings["llm_model"] = models[0]
                self.model_select.value = models[0]
            else:
                self.model_select.value = current
            self.model_select.update()
            ui.notify(f"Discovered {len(models)} model options!", type="positive")
        else:
            ui.notify("Could not retrieve models. Verify your URL, API Key, and server status.", type="warning")

    def write_to_terminal(self, text: str) -> None:
        """Pipes console logs into our terminal widget."""
        self.terminal_log.push(text)
        # Force auto-scroll to the bottom of all active NiceGUI log views
        ui.run_javascript('const logs = document.querySelectorAll(".nicegui-log"); logs.forEach(el => el.scrollTop = el.scrollHeight);')

    def update_download_progress(self, val: float) -> None:
        """Updates progress bar."""
        self.progress_bar.set_value(val)

    def save_and_close(self) -> None:
        """Saves values to DB and closes."""
        for k, v in self.settings.items():
            set_setting(k, v)
        ui.notify("Settings saved.", type="positive")
        self.dialog.close()

    def open(self) -> None:
        """Syncs settings with current database values and opens settings dialog."""
        # 1. Sync settings in memory with latest database entries to prevent out-of-sync states
        from database.connection import get_setting
        for k in list(self.settings.keys()):
            db_val = get_setting(k)
            if db_val is not None:
                if db_val == "True":
                    self.settings[k] = True
                elif db_val == "False":
                    self.settings[k] = False
                elif isinstance(db_val, str) and db_val.isdigit():
                    self.settings[k] = int(db_val)
                else:
                    self.settings[k] = db_val

        # 2. Prevent NiceGUI's blank dropdown bug by ensuring the saved value is in the options list
        saved_model = self.settings.get('llm_model', '')
        if not saved_model:
            saved_model = 'local-model'
            self.settings['llm_model'] = saved_model

        if saved_model not in self.model_select.options:
            self.model_select.options = [saved_model]
        
        self.model_select.value = saved_model
        self.model_select.update()

        self.update_installation_statuses()
        self.dialog.open()

    async def shutdown_application(self) -> None:
        """Performs a clean shutdown sequence, allowing launchers to exit cleanly."""
        ui.notify("Shutting down safely... Please restart from your active launcher.", type="warning", timeout=5)
        await asyncio.sleep(1.5)
        app.shutdown()

    def build_ui(self) -> None:
        """Draws the modular Settings Layout"""
        ui.label('Global Configuration').classes('text-xl font-bold text-slate-800 mb-2')
        
        with ui.column().classes('w-full gap-4'):
            # Dynamic warning card bound to restart state
            with ui.card().classes('w-full bg-amber-50 border border-amber-200 p-4 rounded-xl gap-2') \
                    .bind_visibility_from(state, 'needs_restart'):
                with ui.row().classes('items-center gap-2 text-amber-900 font-bold text-xs'):
                    ui.icon('warning', size='18px', color='amber')
                    ui.label('Restart Required to Apply Transcription Libraries')
                ui.label(
                    "You have successfully installed external Python packages containing native binaries (C++ DLLs). "
                    "These modules cannot be loaded into the current active memory workspace. "
                    "Please shut down and restart ABI-Pipeline to transcription-enable the app."
                ).classes('text-[11px] text-amber-800 leading-normal')
                with ui.row().classes('w-full justify-end mt-1'):
                    ui.button(
                        'Shutdown App Safely', 
                        icon='power_settings_new', 
                        on_click=self.shutdown_application
                    ).classes('bg-amber-600 text-white font-semibold text-xs')

            # 1. ComfyUI and LLM settings (Expanded by default)
            with ui.expansion('AI Server & Connections', icon='settings', value=True).classes('w-full border rounded-lg'):
                with ui.column().classes('w-full p-4 gap-3'):
                    ui.input('ComfyUI Base URL').bind_value(self.settings, 'comfy_url').classes('w-full')
                    ui.input('Local Comfy Directory Path').bind_value(self.settings, 'comfy_path').classes('w-full')
                    ui.input('ComfyUI Launch Arguments', placeholder="e.g., --windows-standalone-build").bind_value(self.settings, 'comfy_args').classes('w-full')
                    ui.input('LLM API Endpoint URL', placeholder="e.g., http://localhost:11434").bind_value(self.settings, 'llm_url').classes('w-full')
                    ui.input('LLM API Key (Optional)', password=True, password_toggle_button=True).bind_value(self.settings, 'llm_api_key').classes('w-full')
                    
                    # Local LLM Launcher Configuration
                    ui.input('Local LLM Host Executable Path', placeholder="e.g., E:/llama-cpp/llama-server.exe").bind_value(self.settings, 'llm_launch_path').classes('w-full')
                    ui.input('Local LLM Launch Arguments', placeholder="e.g., --models-preset models.ini --port 1357").bind_value(self.settings, 'llm_launch_args').classes('w-full')
                    
                    # Row with Model Dropdown & Dynamic Refresh Option
                    with ui.row().classes('w-full items-end gap-2'):
                        saved_model = self.settings.get('llm_model', '')
                        initial_options = [saved_model] if saved_model else ['local-model']
                        
                        self.model_select = ui.select(
                            options=initial_options,
                            label='Target LLM Model'
                        ).bind_value(self.settings, 'llm_model').classes('flex-1')
                        
                        ui.button(
                            icon='refresh', 
                            on_click=self.refresh_llm_models
                        ).props('flat dense').classes('h-10 text-blue-600').tooltip('Scan Connection for Models')

            # 2. UI & Notifications Accordion
            with ui.expansion('Notifications & Alerts', icon='notifications').classes('w-full border rounded-lg bg-slate-50/50'):
                with ui.column().classes('w-full p-4 gap-3 bg-white'):
                    
                    def on_notification_toggle(e):
                        if e.value:
                            ui.run_javascript('''
                                if (!("Notification" in window)) {
                                    alert("This browser does not support desktop notifications.");
                                } else if (Notification.permission !== "granted") {
                                    Notification.requestPermission().then(permission => {
                                        if (permission !== "granted") {
                                            alert("Notification permission was denied. Enable permission in browser settings to receive alerts.");
                                        }
                                    });
                                }
                            ''')
                            
                    ui.checkbox(
                        'Enable Desktop Notifications',
                        value=self.settings.get('enable_desktop_notifications', False)
                    ).bind_value(self.settings, 'enable_desktop_notifications').on_value_change(on_notification_toggle)
                    
                    ui.number(
                        'Alert Item Count Threshold',
                        value=int(self.settings.get('notification_threshold', 30)),
                        placeholder='e.g., 30'
                    ).bind_value(self.settings, 'notification_threshold').classes('w-full').tooltip('Only notifies if the batch contains at least this many processed items')

            # 3. STT Selection & Installation Status (Collapsed at the bottom)
            with ui.expansion('Transcription Setup (One-Time)', icon='construction').classes('w-full border rounded-lg bg-slate-50/50'):
                with ui.column().classes('w-full p-4 gap-4 bg-white'):
                    
                    # Engine Radio Selector
                    with ui.column().classes('w-full gap-1'):
                        ui.label('STT Engine').classes('text-xs font-bold text-slate-500')
                        self.engine_radio = ui.radio(
                            options={
                                'Parakeet ONNX': 'Parakeet ONNX (Recommended - ~160x Speed)',
                                'Whisper': 'Faster-Whisper (Backup - ~38x Speed)'
                            }
                        ).bind_value(self.settings, 'stt_engine').on_value_change(self.update_installation_statuses).classes('w-full text-sm')
                        
                        # Subtext with the newly uncovered file sizes and speed benchmarks
                        with ui.column().classes('bg-slate-50 p-3 rounded-lg border border-slate-100 mt-1 gap-1.5 text-[11px] text-slate-600 leading-normal'):
                            ui.label('• Parakeet ONNX: Ultra-fast parallel sequential batching. Transcribes a 20-hour audiobook in ~5 to 8 minutes on GPU. Footprint: ~300 MB packages + ~2.5 GB model weights.').classes('font-medium')
                            ui.label('• Faster-Whisper: Highly detailed phrase-level timing maps, but processes audio sequentially. Transcribes a 20-hour audiobook in ~35 minutes on GPU. Footprint: ~3.5 GB PyTorch packages + ~484 MB model weights.').classes('font-medium')

                    ui.separator()

                    # Hardware Target Radio Selector
                    with ui.column().classes('w-full gap-1'):
                        ui.label('STT Device / Hardware Target').classes('text-xs font-bold text-slate-500')
                        self.device_radio = ui.radio(
                            options={
                                'GPU/CUDA': 'GPU / CUDA (Recommended for Nvidia GPU setups)',
                                'CPU': 'CPU (Not Recommended - Slow)'
                            }
                        ).bind_value(self.settings, 'stt_device').on_value_change(self.update_installation_statuses).classes('w-full text-sm')
                    
                    # Real-Time Status Indicators
                    with ui.column().classes('gap-1 mt-1'):
                        self.dep_label = ui.label("Checking dependencies...").classes('text-xs text-slate-500 font-medium')
                        self.model_label = ui.label("Checking model files...").classes('text-xs text-slate-500 font-medium')
                    
                    # Action Button & Progress Bar
                    self.action_btn = ui.button('Install & Download Engine', on_click=self.execute_installation_pipeline).classes('w-full mt-2 bg-blue-600 text-white rounded-lg text-sm')
                    self.progress_bar = ui.linear_progress(value=0.0).classes('w-full')

                    # Installation Console Log Output
                    ui.label('Installation Console Output').classes('text-xs font-bold text-slate-500 mt-2')
                    self.terminal_log = ui.log().classes('h-36 w-full bg-slate-950 p-2 text-emerald-400 font-mono text-[10px] rounded-lg')

                    
            # Actions Bottom Bar
            with ui.row().classes('w-full justify-end gap-3 mt-2'):
                ui.button('Cancel', on_click=self.dialog.close).props('flat color=slate')
                ui.button('Save Configurations', on_click=self.save_and_close).classes('bg-blue-600 text-white')