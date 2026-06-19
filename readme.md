
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

---


### Initial Configuration

On first launch, the pipeline will display a setup wizard. This interface walks you through configuring the external AI engines and importing starter workflows.

![ABI-Pipeline Setup Wizard](docs/screenshots/onboarding.png)

The wizard guides you through the following configurations:
1. **Transcription Engine (Optional):** Select your preferred transcription engine. Note that if you choose to install the local transcription backend, you must restart the application once setup completes before transcription tasks can run.
2. **LLM Configuration:** Specify the host port URL for your local or remote LLM (such as Ollama or LM Studio).
3. **ComfyUI Integration:** Set the path to your ComfyUI base directory along with the ComfyUI API host URL.
4. **Starter Workflows:** Choose from pre-packaged workflows (like SDXL Base) to unpack into your local `./workflows/` folder.

---

### Project Creation

To illustrate a book, start by creating a project. Click the **Add Project** button to open the import configuration menu.

![Import New Project](docs/screenshots/import.png)

The pipeline supports three distinct import structures:
* **Audiobook Folder:** Select a directory where each audiobook is contained within its own subfolder. The pipeline automatically scans the directory, maps out the subfolders as separate volumes, and attempts to find local cover art.
* **EPUB Novels:** Point the importer to a folder containing EPUB files. Multiple books can reside in the same import folder.
* **Text Transcripts:** Import plain text files. The importer divides the narrative into chapters based on the `==CHAPTER==` separator tag.

---

### Audiobook Transcription

For audiobook projects, the first phase of the pipeline is translating audio files into plain text transcripts.

![Project Workspace Dashboard - Transcription Phase](docs/screenshots/transcribing.png)

1. Select your project from the main portal to enter the workspace.
2. Verify that your STT config is active (e.g., Parakeet ONNX on GPU/CUDA).
3. Click **Start Transcription** to begin background processing. 

#### Monitoring Progress
Because processing high-volume text and audio takes time, the pipeline runs tasks in the background. The current step progress is displayed in real-time directly inside your browser's tab title, allowing you to monitor active runs while working in other tabs.

![Browser Tab Progress Indicator](docs/screenshots/tab.png)

---

### Transcript Approval and Formatting

Before you can generate prompts, you must approve the imported or transcribed text for each book. This serves as a quality-control step to ensure your narrative content is clean and structured.

![Phase 2 Review Warning](docs/screenshots/approveTranscripts.png)

#### Audiobook Transcripts
Transcriptions generated directly from audio files generally align well and do not require heavy editing. For these projects, you can safely use the **Approve All** button to unlock prompt generation for all volumes in the batch.

#### EPUB & Text Imports
When importing text directly from publisher files, the raw text often contains front matter, licensing information, tables of contents, or index lists that will waste rendering resources. 

![Transcript Review and Formatting Interface](docs/screenshots/epubTranscript.png)

Use the built-in text editor to review and edit each transcript:
1. Open the target volume in the workspace.
2. Delete publisher boilerplate, preambles, and table of contents.
3. Verify that your chapter divisions are clean.
4. Click **Save Changes** and then **Approve Transcript** to unlock the prompt generation phase.

---

### Prompt Generation Configuration

Prompt generation analyzes the narrative chunks of your approved transcripts and instructs your LLM to extract key quotes and compile detailed visual prompts.

#### The Decoupled Prompt Philosophy (Content vs. Style)
A core architectural feature of ABI-Pipeline is the absolute separation of **scene content** from **artistic style**:

* **The LLM's Job (Content Only):** The LLM template is strictly instructed to generate descriptive scene details (e.g., characters, poses, actions, physical settings, and objects). It must avoid generating any stylistic, lighting, or medium-related keywords (such as "digital art," "photorealistic," "oil painting," or "volumetric lighting").
* **The Pipeline's Job (Global Styling):** The artistic medium and visual aesthetic are managed globally in the subsequent rendering phase using a global **Style** which includes things like **Prompt Prefix** and **Prompt Suffix**.

This decoupled design is highly efficient. Because the prompts stored in your `prompts.csv` files contain only raw scene data, you can completely change the art style of a multi-book project (e.g., switching from watercolor sketch to high-detail oil painting) instantly without wasting hours regenerating prompts from your LLM.

#### The Prompt Playground & Template Tuning
Different genres and books may require distinct instruction setups to extract clear, style-free scene descriptions. Before running a batch prompt generation, navigate to the **Prompt-Gen Playground** tab to test and tune your parameters.

![Prompt-Gen Playground Interface](docs/screenshots/promptPlayground.png)

* **Local LLM Performance:** For optimal local generation speeds, fast models like Gemma 4 (4B or 12B) are recommended. If your local LLM is taking more than a second or two per prompt, ensure that "thinking" or "reasoning" modes are turned off in your LLM host settings.
* **Enterprise LLM Assistance:** If you need assistance tuning your system prompt template to ensure it strictly isolates content from style, the playground provides quick tools to coordinate with enterprise APIs like Claude or ChatGPT.
  * **Copy Primer:** Copies detailed prompt construction instructions, your raw text chunks, and the target output schema to send to an external model.
  * **Copy Full:** Copies your current transcript chunks, extracted quotes, and active prompt structure.
  * **Copy Condensed:** Copies only the output quotes and LLM prompts for rapid iteration.
* **Saving Templates:** Once your prompt instructions reliably extract vivid, style-free scene descriptions, give your custom template a name and save it.

#### Running Batch Generation
With your template finalized, return to the main Dashboard to run the process on your project.

1. Select your newly saved template.
2. Set the **Scene Chunk Size (Words per Image)**. This defines how much text is analyzed for each visual scene. A lower word count results in a higher density of generated images across the book (e.g., 350 words per scene is standard).
3. Click **Generate Prompts** to start the local generation run.

As the LLM processes your volumes, scrolling down the Dashboard will display a live feed of successfully generated quotes and visual prompts.

![Live Prompt Generation Feed](docs/screenshots/promptGen.png)

All active progress, quotes, and prompt parameters are persistently populated in real-time directly inside each book's local master record at `./output/<project name>/<book name>/prompts.csv`.