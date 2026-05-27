from nicegui import ui
import asyncio
from database.connection import set_setting
from services.installer import (
    check_dependencies, 
    check_model_downloaded, 
    run_pip_install, 
    download_model_weights
)

class SettingsModal:
    def __init__(self, app_settings: dict, restart_callback) -> None:
        self.settings = app_settings
        self.restart_app = restart_callback
        
        # Live reactive states for the UI
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
        engine = self.settings.get("stt_engine", "Parakeet ONNX")
        
        # Dependency Indicator
        if self.dep_status["status"]:
            self.dep_label.set_text("✓ Python dependencies installed.")
            self.dep_label.classes('text-emerald-600', remove='text-rose-500 text-slate-500')
        else:
            missing_str = ", ".join(self.dep_status["missing"])
            self.dep_label.set_text(f"✗ Missing packages: {missing_str}")
            self.dep_label.classes('text-rose-500', remove='text-emerald-600 text-slate-500')
            
        # Model File Indicator
        if self.model_status:
            self.model_label.set_text("✓ Model weights downloaded locally.")
            self.model_label.classes('text-emerald-600', remove='text-rose-500 text-slate-500')
        else:
            self.model_label.set_text("✗ Model weights missing.")
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
        
        # If we had to install Python libraries containing DLLs, prompt for restart
        if len(self.dep_status["missing"]) > 0:
            ui.notify("Libraries containing binaries installed. Restart recommended.", type="warning", timeout=10)

    def write_to_terminal(self, text: str) -> None:
        """Pipes console logs into our terminal widget."""
        self.terminal_log.push(text)

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
        self.update_installation_statuses()
        self.dialog.open()

    def build_ui(self) -> None:
        """Draws the modular Settings Layout"""
        ui.label('Global Settings').classes('text-xl font-bold text-slate-800 mb-2')
        
        with ui.column().classes('w-full gap-4'):
            # STT Selection & Installation Status
            with ui.card().classes('w-full p-4 border bg-slate-50/50'):
                ui.label('Transcription Setup').classes('text-sm font-bold text-slate-700')
                
                # Engine Dropdown Selector
                ui.select(
                    options=['Parakeet ONNX', 'Whisper'], 
                    label='STT Engine'
                ).bind_value(self.settings, 'stt_engine').on_value_change(self.update_installation_statuses).classes('w-full')
                
                # Hardware Target Dropdown Selector
                ui.select(
                    options=['GPU/CUDA', 'CPU'], 
                    label='STT Device / Hardware'
                ).bind_value(self.settings, 'stt_device').on_value_change(self.update_installation_statuses).classes('w-full mt-2')
                
                # Real-Time Status Indicators
                with ui.column().classes('gap-1 mt-2'):
                    self.dep_label = ui.label("Checking dependencies...").classes('text-xs text-slate-500 font-medium')
                    self.model_label = ui.label("Checking model files...").classes('text-xs text-slate-500 font-medium')
                
                # Action Button & Progress Bar
                self.action_btn = ui.button('Install & Download Engine', on_click=self.execute_installation_pipeline).classes('w-full mt-3 bg-blue-600 text-white rounded-lg text-sm')
                self.progress_bar = ui.linear_progress(value=0.0).classes('w-full mt-2')

            # Installation Console Log Output
            with ui.expansion('Installation Terminal Console', icon='terminal').classes('w-full border rounded-lg bg-slate-900 text-slate-100 font-mono text-xs'):
                self.terminal_log = ui.log().classes('h-40 w-full bg-slate-950 p-2 text-emerald-400')

            # ComfyUI and LLM settings
            with ui.expansion('Advanced Server Connections', icon='settings').classes('w-full border rounded-lg'):
                ui.input('ComfyUI URL').bind_value(self.settings, 'comfy_url').classes('w-full p-2')
                ui.input('Local Comfy Path').bind_value(self.settings, 'comfy_path').classes('w-full p-2')
                ui.select(options=['Ollama', 'LM Studio'], label='LLM Provider').bind_value(self.settings, 'llm_provider').classes('w-full p-2')
                ui.input('LLM API URL').bind_value(self.settings, 'llm_url').classes('w-full p-2')

            # Actions Bottom Bar
            with ui.row().classes('w-full justify-end gap-3 mt-2'):
                ui.button('Cancel', on_click=self.dialog.close).props('flat color=slate')
                ui.button('Save Configs', on_click=self.save_and_close).classes('bg-blue-600 text-white')