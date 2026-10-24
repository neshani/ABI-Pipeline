from nicegui import ui, app
import asyncio
import httpx
import sys
from database.connection import set_setting
from services import llm_config
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
        self.subtitles_online = False
        
        # Build the Dialog UI element
        with ui.dialog() as self.dialog, ui.card().classes('w-full max-w-xl p-6 rounded-xl'):
            self.build_ui()
            
        # Run a status check as soon as we initialize
        self.update_installation_statuses()

    def update_installation_statuses(self) -> None:
        """Query backend to check if chosen engine is fully installed and ready."""
        engine = self.settings.get("stt_engine", "ABI-Subtitles")
        device = self.settings.get("stt_device", "GPU/CUDA")
        
        if engine == "ABI-Subtitles":
            from services.subtitles_client import SubtitlesClient
            client = SubtitlesClient(self.settings.get("subtitles_server_url", "http://127.0.0.1:5050"))
            self.subtitles_online = client.is_alive()
            self.dep_status = {"status": True, "missing": []}
            self.model_status = self.subtitles_online
        else:
            self.dep_status = check_dependencies(engine, device)
            self.model_status = check_model_downloaded(engine, device)
            
        self.update_ui_elements()

    def update_ui_elements(self) -> None:
        """Refresh individual UI elements depending on the status values."""
        engine = self.settings.get("stt_engine", "ABI-Subtitles")
        
        if engine == "ABI-Subtitles":
            if self.subtitles_online:
                self.dep_label.set_text("✓ ABI-Subtitles Server is ONLINE and reachable.")
                self.dep_label.classes('text-emerald-600', remove='text-rose-500 text-slate-500')
                self.model_label.set_text("✓ DirectML inference engine active (~180x realtime).")
                self.model_label.classes('text-emerald-600', remove='text-rose-500 text-slate-500')
                self.action_btn.set_text("Server Connected")
                self.action_btn.disable()
                self.action_btn.classes('bg-emerald-600', remove='bg-blue-600 bg-amber-600')
            else:
                url = self.settings.get("subtitles_server_url", "http://127.0.0.1:5050")
                self.dep_label.set_text(f"— ABI-Subtitles Server is OFFLINE (Cannot connect to {url})")
                self.dep_label.classes('text-rose-500', remove='text-emerald-600 text-slate-500')
                self.model_label.set_text("— Launch ABI-Subtitles app to enable high-speed transcription.")
                self.model_label.classes('text-rose-500', remove='text-emerald-600 text-slate-500')
                self.action_btn.set_text("Refresh Connection")
                self.action_btn.enable()
                self.action_btn.classes('bg-blue-600', remove='bg-emerald-600 bg-amber-600')
            return

        # Fallback for Python-managed engines (Parakeet ONNX / Faster-Whisper)
        if self.dep_status["status"]:
            self.dep_label.set_text("✓ Python dependencies installed.")
            self.dep_label.classes('text-emerald-600', remove='text-rose-500 text-slate-500')
        else:
            missing_str = ", ".join(self.dep_status["missing"])
            self.dep_label.set_text(f"— Missing packages: {missing_str}")
            self.dep_label.classes('text-rose-500', remove='text-emerald-600 text-slate-500')
            
        if self.model_status:
            self.model_label.set_text("✓ Model weights downloaded locally.")
            self.model_label.classes('text-emerald-600', remove='text-rose-500 text-slate-500')
        else:
            self.model_label.set_text("— Model weights missing.")
            self.model_label.classes('text-rose-500', remove='text-emerald-600 text-slate-500')

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
        """Background process that installs dependencies and downloads models or tests connection."""
        engine = self.settings.get("stt_engine", "ABI-Subtitles")
        
        if engine == "ABI-Subtitles":
            self.update_installation_statuses()
            if self.subtitles_online:
                ui.notify("Connected to ABI-Subtitles server!", type="positive")
            else:
                ui.notify("Could not connect to ABI-Subtitles server. Is the app running?", type="warning")
            return

        self.installing = True
        self.action_btn.disable()
        self.terminal_log.clear()
        self.progress_bar.set_value(0.0)
        device = self.settings.get("stt_device", "CPU")
        
        had_missing_deps = len(self.dep_status["missing"]) > 0
        if not self.dep_status["status"]:
            success = await run_pip_install(self.dep_status["missing"], self.write_to_terminal)
            if not success:
                ui.notify("Installation failed during library deployment.", type="negative")
                self.installing = False
                self.update_installation_statuses()
                return
            
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

        self.installing = False
        self.update_installation_statuses()
        
        if had_missing_deps:
            state.needs_restart = True
            ui.notify("Engine setup completed! Please quit the app and run start.bat again to apply libraries.", type="warning", timeout=0)
        else:
            ui.notify("Engine setup successfully completed!", type="positive")

    # --- MCP server (character lookup) -----------------------------------
    def _mcp_status(self, text: str, kind: str = "plain") -> None:
        colours = {"ok": "text-emerald-600", "error": "text-rose-500",
                   "plain": "text-slate-500"}
        self.mcp_status.set_text(text)
        self.mcp_status.classes(f'text-[10px] {colours.get(kind, colours["plain"])}',
                                remove=" ".join(f'text-[10px] {c}'
                                                for c in colours.values()))

    async def connect_mcp(self) -> None:
        """Handshake with the server, list its tools, remember them.

        The HTTP call runs off the UI loop: a slow or unreachable server must not
        freeze the modal, and the failure has to be visible rather than silent.
        """
        url = (self.settings.get("mcp_url") or "").strip()
        if not url:
            self._mcp_status("Enter a server URL first.", "error")
            return
        self._mcp_status("Connecting...")
        try:
            from services import mcp_client
            result = await asyncio.to_thread(
                mcp_client.list_tools, url,
                self.settings.get("mcp_auth_header") or "")
        except Exception as ex:
            self._mcp_status(f"Failed: {ex}", "error")
            ui.notify(f"MCP connection failed: {ex}", type="negative")
            return

        tools = result.get("tools") or []
        cache = [{"name": tool["name"],
                  "description": (tool.get("description") or "")[:400]}
                 for tool in tools]
        prefs = self.settings.get("mcp_tool_prefs")
        if not isinstance(prefs, dict):
            prefs = {}
        # A tool that is new to this list defaults to enabled; one that vanished
        # loses its entry so the panel cannot show a checkbox for nothing.
        self.settings["mcp_tool_cache"] = cache
        self.settings["mcp_tool_prefs"] = {
            tool["name"]: bool(prefs.get(tool["name"], True)) for tool in cache}
        self.redraw_mcp_tools()

        server = result.get("server") or {}
        label = server.get("name") or "server"
        version = server.get("version")
        self._mcp_status(
            f"{len(tools)} tool(s) from {label}"
            + (f" {version}" if version else ""), "ok")

    def forget_mcp(self) -> None:
        """Clear the server and the per-tool choices."""
        self.settings["mcp_url"] = ""
        self.settings["mcp_auth_header"] = ""
        self.settings["mcp_tool_prefs"] = {}
        self.settings["mcp_tool_cache"] = []
        self.redraw_mcp_tools()
        self._mcp_status("Server cleared. Save to apply.")

    def redraw_mcp_tools(self) -> None:
        """Render one checkbox per tool. Only ticked tools reach the model.

        The summary is updated in place rather than by rebuilding the list: a
        checkbox that clears the container it lives in while handling its own
        change event is a NiceGUI delete-during-event hazard.
        """
        box = getattr(self, "mcp_tools_box", None)
        if box is None:
            return
        box.clear()
        cache = self.settings.get("mcp_tool_cache")
        prefs = self.settings.get("mcp_tool_prefs")
        if not isinstance(cache, list):
            cache = []
        if not isinstance(prefs, dict):
            prefs = {}
        if not cache:
            self.mcp_tools_summary = None
            with box:
                ui.label("Connect to a server to list its tools.").classes(
                    'text-[10px] text-slate-400')
            return

        def summary_text() -> str:
            live = sum(1 for tool in cache if prefs.get(tool["name"], True))
            return (f"{live} of {len(cache)} tools offered to the model. "
                    f"Unticked tools are not described, which saves prompt tokens.")

        def make_handler(tool_name: str):
            def handler(event) -> None:
                current = self.settings.setdefault("mcp_tool_prefs", {})
                current[tool_name] = bool(event.value)
                if self.mcp_tools_summary is not None:
                    self.mcp_tools_summary.set_text(summary_text())
            return handler

        with box:
            self.mcp_tools_summary = ui.label(summary_text()).classes(
                'text-[10px] text-slate-500 leading-tight')
            for tool in cache:
                name = tool.get("name") or ""
                ui.checkbox(name, value=bool(prefs.get(name, True)),
                            on_change=make_handler(name)
                            ).classes('text-[11px] text-slate-700').props('dense') \
                    .tooltip(tool.get("description") or "No description provided.")

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
        if self.terminal_log.client.has_socket_connection:
            self.terminal_log.client.run_javascript('const logs = document.querySelectorAll(".nicegui-log"); logs.forEach(el => el.scrollTop = el.scrollHeight);')

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

        saved_model = self.settings.get('llm_model', '')
        if not saved_model:
            saved_model = 'local-model'
            self.settings['llm_model'] = saved_model

        if saved_model not in self.model_select.options:
            self.model_select.options = [saved_model]
        
        self.model_select.value = saved_model
        self.model_select.update()

        # The tool checkboxes are rebuilt from saved state: this modal is created
        # once, so the list would otherwise still show what it saw last time.
        self.redraw_mcp_tools()
        mcp_url = (self.settings.get("mcp_url") or "").strip()
        cached = self.settings.get("mcp_tool_cache") or []
        if mcp_url and cached:
            self._mcp_status(f"{len(cached)} saved tool(s). Reconnect to refresh.")
        else:
            self._mcp_status("")

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
            with ui.card().classes('w-full bg-amber-50 border border-amber-200 p-4 rounded-xl gap-2') \
                    .bind_visibility_from(state, 'needs_restart'):
                with ui.row().classes('items-center gap-2 text-amber-900 font-bold text-xs'):
                    ui.icon('warning', size='18px', color='amber')
                    ui.label('Restart Required to Apply Transcription Libraries')
                ui.label(
                    "You have successfully installed external Python packages containing native binaries (C++ DLLs). "
                    "These modules cannot be loaded into the current active memory workspace. "
                    "Please quit the app and run start.bat again to apply these changes."
                ).classes('text-[11px] text-amber-800 leading-normal')
                with ui.row().classes('w-full justify-end mt-1'):
                    with ui.column().classes('items-end gap-0'):
                        ui.button(
                            'Quit App Safely', 
                            icon='power_settings_new', 
                            on_click=self.shutdown_application
                        ).classes('bg-amber-600 text-white font-semibold text-xs')
                        ui.label('Manually launch start.bat afterwards').classes('text-[8px] text-amber-600 mt-1 font-semibold')

            # 1. ComfyUI and LLM settings
            with ui.expansion('AI Server & Connections', icon='settings', value=True).classes('w-full border rounded-lg'):
                with ui.column().classes('w-full p-4 gap-3'):
                    ui.input('ComfyUI Base URL').bind_value(self.settings, 'comfy_url').classes('w-full')
                    ui.input('Local Comfy Directory Path').bind_value(self.settings, 'comfy_path').classes('w-full')
                    ui.input('LLM API Endpoint URL', placeholder="e.g., http://localhost:11434").bind_value(self.settings, 'llm_url').classes('w-full')
                    ui.input('LLM API Key (Optional)', password=True, password_toggle_button=True).bind_value(self.settings, 'llm_api_key').classes('w-full')
                    
                    ui.input('Local LLM Host Executable Path', placeholder="e.g., E:/llama-cpp/llama-server.exe").bind_value(self.settings, 'llm_launch_path').classes('w-full')
                    ui.input('Local LLM Launch Arguments', placeholder="e.g., --models-preset models.ini --port 1357").bind_value(self.settings, 'llm_launch_args').classes('w-full')
                    
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

                    with ui.expansion('Per-stage model overrides', icon='tune')\
                            .classes('w-full border rounded-lg'):
                        with ui.column().classes('w-full p-2 gap-2'):
                            ui.label(
                                'Leave a stage blank to use the Target LLM Model above. '
                                'Point the expensive stages at a larger or remote model '
                                'and keep the rest local.'
                            ).classes('text-[11px] text-slate-500 leading-normal')
                            overrides = self.settings.setdefault('llm_stage_models', {})
                            if not isinstance(overrides, dict):
                                overrides = {}
                                self.settings['llm_stage_models'] = overrides
                            for stage, stage_label in llm_config.STAGES:
                                overrides.setdefault(stage, '')
                                ui.input(stage_label).bind_value(overrides, stage)\
                                    .classes('w-full')

                    # How much text one analysis call may read. The word budget is
                    # the measured value; the context is a hard cap, because users
                    # bring their own model and a 12,000-word window does not fit
                    # an 8k context.
                    with ui.row().classes('w-full items-end gap-2'):
                        ui.number(
                            'Model Context (tokens)',
                            value=int(self.settings.get('llm_context_tokens', 32768) or 32768),
                            min=2048, max=1048576, step=1024
                        ).bind_value(self.settings, 'llm_context_tokens').classes('flex-1').tooltip(
                            'The context size your LLM server was started with. '
                            'Used to cap how much text one analysis call may read.'
                        )
                        ui.number(
                            'Analysis Window (words)',
                            value=int(self.settings.get('analysis_window_words', 12000) or 12000),
                            min=500, max=100000, step=500
                        ).bind_value(self.settings, 'analysis_window_words').classes('flex-1').tooltip(
                            'Scene and character analysis reads this many words per call, '
                            'capped at 60% of the context above. Smaller means more calls.'
                        )
                    ui.label(
                        'Effective window: the smaller of the two values. '
                        '12,000 words is the measured default for a 32k-context model.'
                    ).classes('text-[9px] text-slate-500 leading-tight')

                    ui.checkbox(
                        'Auto-release VRAM when switching between ComfyUI and the LLM',
                        value=self.settings.get('vram_auto_switch', True)
                    ).bind_value(self.settings, 'vram_auto_switch').tooltip(
                        'Sends the ComfyUI cache-clear before an LLM request, and unloads the LLM before rendering.'
                    )

            # Character research through a user-supplied MCP server. The app is the
            # MCP client, so the search provider is whatever server is added here
            # (a SearXNG wrapper, Brave, Tavily, Firecrawl) and nothing has to be
            # configured on the LLM host. Only ticked tools are described to the
            # model, which keeps the prompt small.
            with ui.expansion('Character Lookup (MCP)', icon='travel_explore') \
                    .classes('w-full border rounded-lg'):
                with ui.column().classes('w-full p-4 gap-3'):
                    ui.label(
                        'Optional. A Streamable-HTTP MCP server used when researching '
                        'character descriptions. Any server that exposes search or '
                        'page-reading tools works; the LLM host needs no changes.'
                    ).classes('text-[11px] text-slate-500 leading-normal')
                    ui.input('MCP Server URL', placeholder='https://example.com/mcp') \
                        .bind_value(self.settings, 'mcp_url').classes('w-full')
                    ui.input('Authorization Header (Optional)', password=True,
                             password_toggle_button=True,
                             placeholder='a bare token, or "Header-Name: value"') \
                        .bind_value(self.settings, 'mcp_auth_header').classes('w-full') \
                        .tooltip('A bare token is sent as "Authorization: Bearer <token>". '
                                 'Use "Header-Name: value" for a service that wants a '
                                 'different header.')
                    with ui.row().classes('w-full items-center gap-2'):
                        ui.button('Connect & List Tools', icon='sync',
                                  on_click=self.connect_mcp).classes(
                            'bg-blue-600 hover:bg-blue-700 text-white font-bold '
                            'text-xs px-3 py-1 rounded')
                        ui.button('Forget Server', icon='delete_outline',
                                  on_click=self.forget_mcp).props('flat dense').classes(
                            'text-[11px] text-slate-500')
                        self.mcp_status = ui.label('').classes('text-[10px] text-slate-500')
                    self.mcp_tools_box = ui.column().classes('w-full gap-1')
                    self.redraw_mcp_tools()

            # 2. UI & Notifications Accordion
            with ui.expansion('Notifications & Export Options', icon='notifications').classes('w-full border rounded-lg bg-slate-50/50'):
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
                    
                    ui.checkbox(
                        'Export Word-Timestamped JSON (transcript.json) for Read-Along',
                        value=self.settings.get('save_word_timestamps', True)
                    ).bind_value(self.settings, 'save_word_timestamps').tooltip('Generates a word-level timing JSON file formatted for subtitle/read-along sync tools')

                    ui.number(
                        'Alert Item Count Threshold',
                        value=int(self.settings.get('notification_threshold', 30)),
                        placeholder='e.g., 30'
                    ).bind_value(self.settings, 'notification_threshold').classes('w-full').tooltip('Only notifies if the batch contains at least this many processed items')

            # 3. STT Selection & Configuration
            with ui.expansion('Transcription Setup', icon='construction', value=False).classes('w-full border rounded-lg bg-slate-50/50'):
                with ui.column().classes('w-full p-4 gap-4 bg-white'):
                    
                    with ui.column().classes('w-full gap-1'):
                        ui.label('STT Engine').classes('text-xs font-bold text-slate-500')
                        self.engine_radio = ui.radio(
                            options={
                                'ABI-Subtitles': 'ABI-Subtitles (Recommended - ~180x DirectML)',
                                'Parakeet ONNX': 'Parakeet ONNX (Python Local - ~65x)',
                                'Whisper': 'Faster-Whisper (Python Local - ~38x)'
                            }
                        ).bind_value(self.settings, 'stt_engine').on_value_change(self.update_installation_statuses).classes('w-full text-sm')
                        
                        with ui.column().classes('bg-slate-50 p-3 rounded-lg border border-slate-100 mt-1 gap-1.5 text-[11px] text-slate-600 leading-normal'):
                            ui.label('• ABI-Subtitles: External DirectML server with asynchronous queue. 160x–190x speed. Transcribes a 20-hour audiobook in ~6 minutes. Zero Python VRAM footprint.').classes('font-medium')
                            ui.label('• Parakeet ONNX: Python in-process ONNX runtime engine. Transcribes a 20-hour audiobook in ~18 minutes on GPU.').classes('font-medium')
                            ui.label('• Faster-Whisper: Sequential Whisper engine. Transcribes a 20-hour audiobook in ~35 minutes on GPU.').classes('font-medium')

                    # ABI-Subtitles specific server settings
                    with ui.column().classes('w-full gap-2 border-t pt-3') \
                            .bind_visibility_from(self.settings, 'stt_engine', backward=lambda e: e == 'ABI-Subtitles'):
                        with ui.row().classes('w-full justify-between items-center text-xs bg-blue-50/70 p-2.5 rounded-lg border border-blue-100'):
                            with ui.row().classes('items-center gap-1.5 text-blue-900'):
                                ui.icon('download', size='16px', color='blue-600')
                                ui.label('Need to install or update the standalone server?').classes('font-medium')
                            ui.link('Download ABI-Subtitles ↗', 'https://audiobooksilluminated.com/downloads/', new_tab=True).classes('text-blue-600 hover:text-blue-800 font-bold underline')
                        
                        ui.input('ABI-Subtitles Server URL', placeholder='http://127.0.0.1:5050').bind_value(self.settings, 'subtitles_server_url').classes('w-full')
                        ui.input('ABI-Subtitles Executable Path (Optional)', placeholder='e.g., E:/ABI-Subtitles/abi-subtitles.exe').bind_value(self.settings, 'subtitles_launch_path').classes('w-full')

                    # Hardware Target Selector (for in-process Python engines)
                    with ui.column().classes('w-full gap-1 border-t pt-3') \
                            .bind_visibility_from(self.settings, 'stt_engine', backward=lambda e: e != 'ABI-Subtitles'):
                        ui.label('STT Device / Hardware Target').classes('text-xs font-bold text-slate-500')
                        self.device_radio = ui.radio(
                            options={
                                'GPU/CUDA': 'GPU / CUDA (Recommended for Nvidia GPU setups)',
                                'CPU': 'CPU (Not Recommended - Slow)'
                            }
                        ).bind_value(self.settings, 'stt_device').on_value_change(self.update_installation_statuses).classes('w-full text-sm')
                    
                    with ui.column().classes('gap-1 mt-1'):
                        self.dep_label = ui.label("Checking status...").classes('text-xs text-slate-500 font-medium')
                        self.model_label = ui.label("Checking engine...").classes('text-xs text-slate-500 font-medium')
                    
                    self.action_btn = ui.button('Check / Install Engine', on_click=self.execute_installation_pipeline).classes('w-full mt-2 bg-blue-600 text-white rounded-lg text-sm')
                    self.progress_bar = ui.linear_progress(value=0.0).classes('w-full')

                    with ui.column().classes('w-full') \
                            .bind_visibility_from(self.settings, 'stt_engine', backward=lambda e: e != 'ABI-Subtitles'):
                        ui.label('Installation Console Output').classes('text-xs font-bold text-slate-500 mt-2')
                        self.terminal_log = ui.log().classes('h-36 w-full bg-slate-950 p-2 text-emerald-400 font-mono text-[10px] rounded-lg')
                              
            with ui.row().classes('w-full justify-end gap-3 mt-2'):
                ui.button('Cancel', on_click=self.dialog.close).props('flat color=slate')
                ui.button('Save Configurations', on_click=self.save_and_close).classes('bg-blue-600 text-white')