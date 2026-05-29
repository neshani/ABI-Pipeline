import httpx
import re
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from database.connection import get_setting

REFUSAL_KEYWORDS = [
    "i cannot", "i can't", "i apologize", "i'm sorry", "im sorry", 
    "unable to generate", "policy", "forbidden", "restricted", 
    "cannot fulfill", "as an ai", "language model"
]

def ensure_templates_directory() -> Path:
    """Ensures prompt_templates folder exists and creates a default.txt file if empty."""
    templates_dir = Path("./prompt_templates")
    templates_dir.mkdir(exist_ok=True)
    default_file = templates_dir / "default.txt"
    if not default_file.exists():
        default_content = (
            "You are a specialized AI function that performs two tasks:\n"
            "1.  Select a visually rich, verbatim quote from a passage of text.\n"
            "2.  Create a single-sentence image prompt inspired by that quote for a text-to-image AI like SDXL.\n\n"
            "### INSTRUCTIONS ###\n"
            "1.  **Quote Selection:** From the PASSAGE, find the single most visually descriptive sentence or short phrase. "
            "This quote will be used to inspire the image. It MUST be an exact quote.\n"
            "2.  **Image Prompt Generation:**\n"
            "    *   The image prompt MUST be directly inspired by the quote you selected.\n"
            "    *   **Character Descriptions:** Instead of using a character's proper name, describe their visual appearance or "
            "role in the scene. Focus on neutral, observable details that a camera would see.\n"
            "    *   The prompt MUST be a single, descriptive sentence.\n"
            "    *   Structure the sentence like this: [Subject], [Setting], [Visual Details].\n"
            "3.  **Output Format:** Your entire response MUST follow this exact format. Do NOT add conversational text.\n\n"
            "QUOTE: [The verbatim quote you selected from the passage]\n"
            "PROMPT: [The single-sentence image prompt you generated]\n\n"
            "### TASK ###\n"
            "PASSAGE:\n"
            "<text>\n\n"
            "REMEMBER: YOUR ENTIRE RESPONSE MUST FOLLOW THE EXACT `QUOTE: ... PROMPT: ...` FORMAT."
        )
        default_file.write_text(default_content, encoding="utf-8")
    return templates_dir


def list_stored_templates() -> List[str]:
    """Lists all available prompt template files (.txt) in the templates directory."""
    templates_dir = ensure_templates_directory()
    return sorted([f.stem for f in templates_dir.glob("*.txt")])


def load_template_by_name(name: str) -> str:
    """Loads prompt template text by filename stem."""
    templates_dir = ensure_templates_directory()
    target = templates_dir / f"{name}.txt"
    if target.exists():
        return target.read_text(encoding="utf-8")
    return ""


def save_template_by_name(name: str, content: str) -> None:
    """Saves/stores a custom named template into prompt_templates."""
    templates_dir = ensure_templates_directory()
    target = templates_dir / f"{name}.txt"
    target.write_text(content, encoding="utf-8")


def smart_chunk_text(text: str, max_chunk_words: int) -> List[str]:
    """Ported robust text chunking algorithm."""
    if not text.strip() or max_chunk_words <= 0:
        return []

    def split_into_sentences(paragraph_text: str) -> List[str]:
        if not paragraph_text:
            return []
        return re.findall(r'[^.!?]+[.!?]?', paragraph_text)

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk_words = []

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        if not paragraph_words:
            continue

        if len(current_chunk_words) + len(paragraph_words) <= max_chunk_words:
            current_chunk_words.extend(paragraph_words)
        else:
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []

            sentences = split_into_sentences(paragraph)
            for sentence in sentences:
                sentence_words = sentence.strip().split()
                if not sentence_words:
                    continue

                if len(current_chunk_words) + len(sentence_words) > max_chunk_words and current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))
                    current_chunk_words = []
                
                current_chunk_words.extend(sentence_words)

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


async def get_llm_response(prompt: str, llm_url: str, model_name: str) -> str:
    """Dispatches asynchronous request to local Ollama or LM Studio OpenAI-compatible endpoints."""
    endpoint = llm_url.rstrip("/")
    if "/v1" not in endpoint and "/api" not in endpoint:
        endpoint = f"{endpoint}/v1/chat/completions"
    elif endpoint.endswith("/v1"):
        endpoint = f"{endpoint}/chat/completions"

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(endpoint, json=payload, headers=headers, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            # Fallback to direct Ollama generate endpoint if connection fails and endpoint targets port 11434
            if "11434" in endpoint or "ollama" in endpoint.lower():
                try:
                    direct_endpoint = f"{llm_url.rstrip('/')}/api/generate"
                    direct_payload = {
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7}
                    }
                    resp = await client.post(direct_endpoint, json=direct_payload, timeout=60.0)
                    resp.raise_for_status()
                    return resp.json().get("response", "")
                except Exception as inner_e:
                    return f"API Connection Failed: {e} (Direct API Fallback failed: {inner_e})"
            return f"API Connection Failed: {e}"


def parse_llm_response(raw_response: str) -> Dict[str, str]:
    """Strips and extracts structured output matching QUOTE: ... PROMPT: ... with fallbacks."""
    parsed_prompt = ""
    parsed_quote = ""
    current_capture = None

    clean_raw = raw_response.replace('**', '').replace('__', '')

    for line in clean_raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        
        lower_line = line.lower()
        
        if lower_line.startswith("quote:") or lower_line.startswith("- quote:"):
            current_capture = "quote"
            parts = line.split(":", 1)
            if len(parts) > 1:
                parsed_quote += parts[1].strip() + " "
        elif lower_line.startswith("prompt:") or lower_line.startswith("- prompt:"):
            current_capture = "prompt"
            parts = line.split(":", 1)
            if len(parts) > 1:
                parsed_prompt += parts[1].strip() + " "
        elif current_capture == "quote":
            parsed_quote += line + " "
        elif current_capture == "prompt":
            parsed_prompt += line + " "

    parsed_quote = parsed_quote.strip().strip('"')
    parsed_prompt = parsed_prompt.strip().strip('"')

    # Apply extreme fallbacks
    if not parsed_prompt and not parsed_quote:
        clean_txt = clean_raw.strip()
        clean_txt = re.sub(r'^```[\w]*\n', '', clean_txt)
        clean_txt = re.sub(r'\n```$', '', clean_txt).strip('"').strip()
        parsed_prompt = clean_txt
        parsed_quote = " "
    elif not parsed_prompt:
        parsed_prompt = parsed_quote
    elif not parsed_quote:
        parsed_quote = " "

    raw_lower = raw_response.lower()
    has_refusal_words = any(keyword in raw_lower for keyword in REFUSAL_KEYWORDS)
    is_refusal = has_refusal_words and (not parsed_prompt or parsed_prompt == parsed_quote)

    return {
        "raw": raw_response,
        "quote": parsed_quote,
        "prompt": parsed_prompt,
        "status": "refusal" if is_refusal else "success"
    }


def fetch_test_chunks(
    project_name: str, 
    book_name: str, 
    count: int, 
    mode: str, 
    start_index: int = 15, 
    seed: int = 42
) -> List[str]:
    """Reads transcript.txt from the specified book folder and returns selected test chunks based on mode, start index, and seed."""
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    transcript_path = base_output_dir / project_name / book_name / "transcript.txt"
    if not transcript_path.exists():
        return []

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()

    cleaned_text = content.replace("==CHAPTER==", " ").strip()
    chunk_size = int(get_setting("batch_size", 30)) * 10
    if chunk_size <= 0:
        chunk_size = 350

    all_chunks = smart_chunk_text(cleaned_text, chunk_size)
    if not all_chunks:
        return []

    # Safeguard NiceGUI bound floats to standard integers
    count = int(count)
    if count <= 0:
        count = 1

    if mode == "Static Segment":
        start_idx = max(0, int(start_index))
        # Clamp to avoid going out of bounds
        if start_idx >= len(all_chunks):
            start_idx = max(0, len(all_chunks) - count)
        return all_chunks[start_idx : start_idx + count]
    else:
        # Seeded Random selection for deterministic, reusable random samples
        r = random.Random(int(seed))
        return r.sample(all_chunks, min(count, len(all_chunks)))