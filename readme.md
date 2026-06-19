
# ABI-Pipeline

ABI-Pipeline (Audiobooks Illuminated Pipeline) is a local, semi-automated book illustrator designed for audiobooks and text/EPUB files. It acts as an orchestrator that takes a book as an input and outputs thousands of highly organized, sequentially rendered images aligned with the narrative's progression.

### Core Purpose

The primary purpose of ABI-Pipeline is to automate the complex process of illustrating a book at scale. By coordinating local transcription, language models, and image generation interfaces, it systematically processes a text or audio source file to produce a comprehensive visual narrative.

The pipeline manages three main output assets:
* **Sequential Images:** Thousands of rendered PNG files, organized by chapter and scene, representing every distinct visual moment in the book.
* **Baked-in Metadata:** Individual target quotes and subtitle text embedded directly into the metadata chunk of each corresponding PNG image.
* **Timing and Text Alignment:** A structured map pairing every image and quote with its corresponding chapter, scene, or precise time stamp.

Because these assets are exported to clean, standard directories on your local drive, you have complete ownership of the output. You can use the generated files to construct illustrated EPUBs, build video slideshows, or integrate visual narratives into any custom media player.

#### Optional Player Integration
For users who wish to experience their illustrated books with synchronized audio, text, and graphics, ABI-Pipeline includes an optional packaging tool (the OIS Packager). This tool bundles the images and timing maps into a single file format optimized for the free **Audiobooks Illuminated** player ecosystem:
* **Desktop Player (Windows):** Available at [audiobooksilluminated.com](https://audiobooksilluminated.com).
* **Mobile Player (Android):** Available on the Google Play Store.

![Audiobooks Illuminated Mobile and Desktop Players](docs/screenshots/platforms.jpg)

---

### Prerequisites & Requirements

ABI-Pipeline acts as an orchestrator. It does not need to run on the same physical machine as the heavy AI model hosts, allowing you to distribute workload across your local network if necessary.

#### Software & Host Requirements
* **Operating System:** Windows (native `.bat` files provided) or Linux (shell scripts provided).
* **Python Version:** Python 3.10.
* **LLM Host (Local or Remote):** A running instance of Ollama, LM Studio, or any OpenAI-compatible API provider to handle text analysis and prompt generation.
* **Image Generator (Local or Remote):** A running ComfyUI instance accessible via its API port (default: 8188).

#### Hardware Considerations
* **Fully Local Run:** If running the pipeline interface, the transcription model, the LLM host, and ComfyUI all on a single machine, a high-VRAM NVIDIA GPU is recommended. The fully local, single-machine workflow has been primarily developed and tested on an NVIDIA RTX 3090 (24GB VRAM).
* **Distributed Run:** If ComfyUI or your LLM host runs on a separate machine on your network, the local system running the pipeline interface requires minimal resources.

---

### Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/neshani/ABI-Pipeline
   cd ABI-Pipeline
   ```

2. **Run the Environment Installer:**
   * **Windows:** Double-click `setup.bat` or run it from your terminal:
     ```cmd
     setup.bat
     ```
   * **Linux/macOS:** Run the shell setup script:
     ```bash
     chmod +x setup.sh
     ./setup.sh
     ```

3. **Launch the Application:**
   * **Windows:** Run `start.bat` to launch the NiceGUI interface.
   * **Linux/macOS:** Run `start.sh`.

Once launched, open your web browser and navigate to `http://127.0.0.1:8910` to access the pipeline.
