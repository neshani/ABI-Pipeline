import csv
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from nicegui import ui, app
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Book, Project, Character, CharacterAlias, CharacterTimelineEvent
from services.character_manager import (
    compile_character_visual_prompt,
    is_placeholder_description,
    replace_character_tags_in_prompt,
    save_project_characters_to_json,
    ensure_book_orders,
    normalize_character_key,
    clean_character_key,
)
from services import style_library
from services import compat
from ui.pages.project.characters_tab import get_character_frequency_map
from ui.components.prompt_editor_view import render_prompt_editor_view
from ui.components.autocomplete import PromptAutocompleteManager
from ui.components.quick_characters_modal import QuickCharactersModal
from ui import state

try:
    app.add_static_files('/output_media', './output')
except Exception:
    pass


# --- Robust CSV Loading & Saving Utilities ---

# Shot-role presentation for the workspace: canonical order, display labels and
# badge colors. ``single`` only appears on books rendered with the planner off.
SHOT_ROLE_ORDER = ("character", "wide", "insert", "single")
SHOT_ROLE_LABELS = {
    "character": "Character",
    "wide": "Wide",
    "insert": "Insert",
    "single": "Single",
}
SHOT_ROLE_BADGE_COLORS = {
    "character": "indigo",
    "wide": "teal",
    "insert": "purple",
    "single": "slate",
}
SHOT_ROLE_FROM_LABEL = {label: role for role, label in SHOT_ROLE_LABELS.items()}


def roles_present(prompts: List[Dict[str, Any]]) -> List[str]:
    """Distinct shot roles across a book's rows, in canonical order."""
    found = {style_library.normalize_role((p or {}).get("shot_role")) for p in prompts}
    return [r for r in SHOT_ROLE_ORDER if r in found]


def shot_role_matches(item: Dict[str, Any], shot_choice: str) -> bool:
    """True when ``item`` passes the shot-type filter (``All Roles`` always does)."""
    if not shot_choice or shot_choice == "All Roles":
        return True
    role = style_library.normalize_role((item or {}).get("shot_role"))
    return SHOT_ROLE_LABELS.get(role) == shot_choice


def load_prompts_csv(project_name: str, book_name: str) -> List[Dict[str, Any]]:
    """Discovers and parses prompts.csv from output folders, self-healing missing approved columns."""
    csv_paths = [
        Path(f"./output/{project_name}/{book_name}/prompts.csv"),
        Path(f"./output/{project_name}/{book_name}_prompts.csv"),
        Path(f"./output/{project_name}/{book_name}/{book_name}_prompts.csv"),
        Path(f"./output/{project_name}/{project_name}_prompts.csv"),
        Path(f"./output/{project_name}_prompts.csv")
    ]
    
    csv_path = None
    for p in csv_paths:
        if p.exists():
            csv_path = p
            break
            
    if not csv_path or not csv_path.exists():
        return []
        
    rows = []
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            reader.fieldnames = compat.normalize_fieldnames(reader.fieldnames)
            
            for row in reader:
                cleaned_row = {}
                for k, v in row.items():
                    if k:
                        cleaned_row[k.strip().lower()] = v.strip() if v else ""
                
                if "approved" not in cleaned_row:
                    cleaned_row["approved"] = "False"
                rows.append(cleaned_row)
    except Exception as e:
        print(f"[Proofreader-CSV] Error reading prompts: {e}")
        return []
        
    return rows


def _row_key(chapter, shot) -> str:
    """Stable (chapter, shot) key, tolerant of ints, floats and strings."""
    def norm(v):
        try:
            return str(int(float(v)))
        except (TypeError, ValueError):
            return str(v or "").strip()
    return f"{norm(chapter)}|{norm(shot)}"


def _existing_prompt_columns(target_path: Path) -> tuple:
    """The file's header order and its rows keyed by (chapter, shot).

    Used by :func:`save_prompts_csv` to carry over columns the caller's rows do
    not know about. Returns ``([], {})`` when there is nothing readable yet.
    """
    if not target_path.exists():
        return [], {}
    try:
        with open(target_path, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='|')
            reader.fieldnames = compat.normalize_fieldnames(reader.fieldnames)
            header = [h for h in (reader.fieldnames or []) if h]
            rows = {}
            for row in reader:
                cleaned = {k.strip().lower(): (v.strip() if v else "")
                           for k, v in row.items() if k}
                rows[_row_key(cleaned.get("chapter"), cleaned.get("shot"))] = cleaned
        return header, rows
    except Exception as e:
        print(f"[Proofreader-CSV] Error reading existing prompts for merge: {e}")
        return [], {}


def save_prompts_csv(project_name: str, book_name: str, rows: List[Dict[str, Any]]) -> None:
    """Serializes rows back to prompts.csv using pipes, keeping the approved column structured.

    ``prompts.csv`` is the master record, but the rows handed here are a partial
    view of it - they come from the ``ScenePrompt`` cache, whose schema has no
    ``shot_role``. So any column the file already carries that a row does not is
    copied back over from the matching ``(chapter, shot)`` row. Without that,
    every save silently deletes ``shot_role``, and image generation then renders
    the whole book with the base shot-role style.
    """
    target_path = Path(f"./output/{project_name}/{book_name}/prompts.csv")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not rows:
        return

    existing_header, existing_rows = _existing_prompt_columns(target_path)

    # Keep the file's own column order; fall back to the canonical seed for a new
    # file, then append any column the caller introduces.
    headers = list(existing_header) or ["chapter", "shot", "prompt", "quote", "approved"]
    for row in rows:
        for k in row.keys():
            if k and k.lower() not in headers:
                headers.append(k.lower())

    merged_rows = []
    for r in rows:
        row = {(k or "").strip().lower(): v for k, v in r.items() if k}
        prior = existing_rows.get(_row_key(row.get("chapter"), row.get("shot")), {})
        merged = {h: prior.get(h, "") for h in headers}
        for h in headers:
            if h in row and row[h] is not None:
                merged[h] = row[h]
        merged_rows.append(merged)

    try:
        with open(target_path, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter='|')
            writer.writeheader()
            for merged in merged_rows:
                writer.writerow(merged)

        from services.sync_engine import sync_prompts_csv_to_db_cache
        with Session(engine) as session:
            # Safely locate the unique book by matching both project name and book name
            project = session.exec(select(Project).where(Project.name == project_name)).first()
            if project:
                book = session.exec(
                    select(Book)
                    .where(Book.name == book_name)
                    .where(Book.project_id == project.id)
                ).first()
                if book:
                    sync_prompts_csv_to_db_cache(book.id, session)
                    session.commit()
    except Exception as e:
        print(f"[Proofreader-CSV] Error saving prompts: {e}")


# --- High Performance Directory Caching & Parsing ---

def get_book_images_cache(project_name: str, book_name: str) -> Dict[tuple, str]:
    """Scans directories once. Returns a fast lookup dict mapping {(chapter, scene): static_url}.

    The ``?t=`` version is the file's own mtime rather than the time the page was
    built, so an image re-rendered under the same filename still reloads while
    every untouched image keeps its URL - and therefore its browser cache entry -
    across page loads.
    """
    cache = {}
    out_dirs = [
        (Path(f"./output/{project_name}/{book_name}/images"), f"/output_media/{project_name}/{book_name}/images"),
        (Path(f"./output/{project_name}/{book_name}"), f"/output_media/{project_name}/{book_name}")
    ]

    custom_versions = getattr(state, 'custom_image_timestamps', {})

    for d, url_prefix in out_dirs:
        if not d.exists():
            continue
        try:
            with os.scandir(d) as entries:
                for entry in entries:
                    if not entry.name.lower().endswith('.png'):
                        continue
                    stem, _ = os.path.splitext(entry.name)
                    parts = stem.split('_')
                    if len(parts) < 2:
                        continue
                    try:
                        ch = int(parts[0])
                        sc = int(parts[1])
                    except ValueError:
                        continue
                    t_val = custom_versions.get((ch, sc))
                    if t_val is None:
                        try:
                            t_val = int(entry.stat().st_mtime * 1000)
                        except OSError:
                            t_val = 0
                    cache[(ch, sc)] = f"{url_prefix}/{entry.name}?t={t_val}"
        except Exception as ex:
            print(f"[Proofreader-Cache] Error indexing folder {d}: {ex}")

    return cache


def delete_scene_image_file(project_name: str, book_name: str, chapter_str: str, scene_str: str) -> bool:
    """Removes standard image patterns from disk, preparing the scene for ComfyUI regenerations."""
    try:
        chapter = int(float(chapter_str))
        scene = int(float(scene_str))
    except ValueError:
        return False
        
    out_dirs = [
        Path(f"./output/{project_name}/{book_name}/images"),
        Path(f"./output/{project_name}/{book_name}")
    ]
    
    deleted = False
    for d in out_dirs:
        if not d.exists():
            continue
        matches = list(d.glob(f"{chapter:02d}_{scene:02d}_*.png")) or list(d.glob(f"{chapter:02d}_{scene:02d}.png"))
        for m in matches:
            try:
                m.unlink()
                deleted = True
            except Exception as e:
                print(f"[Proofreader] Error deleting file {m.name}: {e}")
                
    if not hasattr(state, 'custom_image_timestamps'):
        state.custom_image_timestamps = {}
    state.custom_image_timestamps[(chapter, scene)] = int(time.time() * 1000)
                
    return deleted


# --- Stateful Context Finder with Pagination Support ---

# Quote marks that can arrive *with* the passage rather than being added for
# presentation. ``’`` is in the closing set because curly single quotes are both
# the closing mark and the apostrophe; a stray outer mark is the lesser evil.
_QUOTE_MARKS = ('"', "\u201c", "\u201d", "\u2018", "\u2019", "\u00ab", "\u00bb")


def quoted_for_display(quote: str) -> str:
    """Wrap a stored quote in quotation marks, unless it already carries its own.

    The ``quote`` column is the passage as the model copied it, so dialogue
    arrives already quoted (``“I’ll not let myself be stilled.”``) and wrapping it
    blindly rendered ``"“I’ll not let myself be stilled.”"``. When *either* end
    already has a mark the text is left alone: the other end is usually the
    transcript's own open-ended dialogue (``“Your task? Your task was to seize
    Almoth Plain!``) and closing it here would invent punctuation the book does
    not have.
    """
    text = (quote or "").strip()
    if not text:
        return ""
    if text[0] in _QUOTE_MARKS or text[-1] in _QUOTE_MARKS:
        return text
    return f'"{text}"'


# A quote character in a stored quote may face any of the transcript's spellings.
_LOOSE_CHARS = {
    "'": "['\u2019]", '"': '["\u201c\u201d]', "-": "[-\u2013\u2014]",
    "\u2018": "['\u2019]", "\u2019": "['\u2019]",
    "\u201c": '["\u201c\u201d]', "\u201d": '["\u201c\u201d]',
    "\u2013": "[-\u2013\u2014]", "\u2014": "[-\u2013\u2014]",
}


def _loose_fragment_pattern(fragment: str) -> str:
    """Regex for one fragment of a quote, blind to typography rather than to wording."""
    frag = re.sub(r"\[[^\]]*\]", " ", fragment or "")  # `[Name]` tags are markup, not text
    frag = " ".join(frag.split()).strip(' .,;:!?"\u2019\u2018\u201c\u201d')
    if len(frag) < 8:
        return ""
    parts = []
    for word in frag.split(" "):
        if parts:
            parts.append(r"\s+")
        parts.append("".join(_LOOSE_CHARS.get(c, re.escape(c)) for c in word))
    return "".join(parts)


def locate_quote_loosely(text: str, quote: str) -> Optional[tuple]:
    """``(start, end)`` span of ``quote`` in ``text``, tolerating typography.

    The stored quote is what a model copied out of the passage, so it carries
    marks the transcript may not repeat: a ``“`` the model reconstructed where the
    transcript simply continues a paragraph of dialogue, straight quotes against
    curly ones, an ellipsis joining two fragments, or a ``[Name]`` tag that
    bracket repair left inside the quote. An exact search then finds nothing and
    the theater reports "context match could not be found" beside a passage that
    is right there. Wording is still matched literally - this is a matching
    tolerance, not a fuzzy search.
    """
    spans = []
    for fragment in re.split(r"\.\.\.|\u2026", quote or ""):
        pattern = _loose_fragment_pattern(fragment)
        if not pattern:
            continue
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        spans.append((match.start(), match.end()))
    if not spans:
        return None
    return spans[0][0], spans[-1][1]


def find_quote_context(project_name: str, book_name: str, quote: str, char_window: int = 400) -> str:
    """Lazy parses transcript.txt, extracting surrounding paragraphs to highlight narration context."""
    res = find_quote_context_paginated(project_name, book_name, quote, window_before=char_window, window_after=char_window)
    return res["html"]


def find_quote_context_paginated(project_name: str, book_name: str, quote: str, window_before: int = 400, window_after: int = 400) -> dict:
    """
    Finds the quote context inside transcript.txt using distinct before and after boundaries,
    returning both the highlighted HTML snippet and pagination capability flags.
    """
    transcript_path = Path(f"./output/{project_name}/{book_name}/transcript.txt")
    if not transcript_path.exists():
        return {
            "html": "Transcript file not found on disk. Run Transcription to load context narration.", 
            "has_prev": False, 
            "has_next": False
        }
            
    try:
        text = transcript_path.read_text(encoding='utf-8')
    except Exception as e:
        return {"html": f"Error reading transcript: {str(e)}", "has_prev": False, "has_next": False}
            
    if not quote:
        return {"html": "Narrative quote is empty.", "has_prev": False, "has_next": False}
            
    text_lower = text.lower()
    quote_lower = quote.lower()
        
    idx = text_lower.find(quote_lower)
    match_len = len(quote)
        
    if idx == -1 and len(quote_lower) > 25:
        idx = text_lower.find(quote_lower[:25])
        match_len = 25

    if idx == -1:
        # Typography differs but the wording is there - see locate_quote_loosely.
        loose = locate_quote_loosely(text, quote)
        if loose:
            idx, match_len = loose[0], loose[1] - loose[0]
            
    if idx == -1:
        return {"html": "Narrative segment context match could not be found inside transcript.txt.", "has_prev": False, "has_next": False}
            
    start_idx = max(0, idx - window_before)
    end_idx = min(len(text), idx + match_len + window_after)
    snippet = text[start_idx:end_idx]

    # Highlight by offset, not by searching the snippet again: the 25-character
    # fallback and the loose match both know exactly where the passage is.
    hit_start = max(0, idx - start_idx)
    hit_end = min(len(snippet), hit_start + match_len)

    has_prev = start_idx > 0
    has_next = end_idx < len(text)

    highlighted = (
        snippet[:hit_start] +
        '<mark id="quote-target" class="bg-yellow-200 text-slate-900 font-bold px-1 rounded shadow-2xs">' +
        snippet[hit_start:hit_end] +
        '</mark>' +
        snippet[hit_end:]
    )
    return {"html": f"... {highlighted} ...", "has_prev": has_prev, "has_next": has_next}



# --- Main Proofing UI Component ---

def render_book_tabs(book_id: int):
    import platform
    import subprocess
    
    with Session(engine) as session:
        book = session.get(Book, book_id)
        if not book:
            ui.label("Book details not available.").classes('text-slate-400 text-sm')
            return
            
        from database.models import Project
        project = session.get(Project, book.project_id)
        if not project:
            ui.label("Project workspace details not available.").classes('text-slate-400 text-sm')
            return

    ensure_book_orders(project.id)

    project_name = project.name
    book_name = book.name
    
    loaded_scene_coords = [-1, -1]

    def extract_prompt_character_tags(prompt_text: str) -> List[str]:
        """Parses bracketed tags from the prompt text area."""
        if not prompt_text:
            return []
        bracket_regex = re.compile(r"\[(.*?)\]")
        return [t.strip() for t in bracket_regex.findall(prompt_text) if t.strip()]

    # Character lookup for the chip row under the prompt editor.
    #
    # This is on the keystroke path: the autocomplete manager calls it on every
    # keyup, so it has to be a dictionary lookup, not a database round trip. The
    # old version loaded every character and then queried its aliases one by one
    # (1,083 queries on a 1,082-character project) and recompiled a tooltip per
    # keypress - about half a second of work per character typed, which starved
    # the websocket until the client reported "connection lost".
    chip_cache: Dict[str, Any] = {
        "index": None,      # cleaned alias -> character id
        "names": {},        # character id -> Character
        "tooltips": {},     # (id, chapter, shot) -> compiled hover text
        "signature": None,  # what is currently drawn
    }

    def character_tag_index():
        """``{cleaned alias: [character ids]}`` and the characters, built once.

        A list per alias, not one id: an alias table can carry the same name on
        more than one character (a prompt-scan artefact), and the old per-keystroke
        loop showed a chip for every one of them. Keeping that would be an
        accident; keeping it *identical* is what makes this cache safe.
        """
        if chip_cache["index"] is None:
            index: Dict[str, List[int]] = {}
            names: Dict[int, Character] = {}
            with Session(engine) as session:
                for char in session.exec(
                        select(Character).where(Character.project_id == project.id)).all():
                    names[char.id] = char
                    index.setdefault(clean_character_key(char.name), []).append(char.id)
                # One join, not one query per character.
                for char_id, alias in session.exec(
                        select(CharacterAlias.character_id, CharacterAlias.alias)
                        .join(Character, Character.id == CharacterAlias.character_id)
                        .where(Character.project_id == project.id)).all():
                    ids = index.setdefault(clean_character_key(alias), [])
                    if char_id not in ids:
                        ids.append(char_id)
            chip_cache["index"] = index
            chip_cache["names"] = names
        return chip_cache["index"], chip_cache["names"]

    def forget_character_cache() -> None:
        """Called when characters or aliases change, so the chips re-read them."""
        chip_cache["index"] = None
        chip_cache["names"] = {}
        chip_cache["tooltips"] = {}
        chip_cache["signature"] = None

    def open_character_edit_dialog(char_id: int):
        """Spawns the quick characters manager modal pre-selected to a specific character."""
        def after_change():
            forget_character_cache()
            if modal_prompt_input:
                render_scene_character_chips(modal_prompt_input.value)

        modal = QuickCharactersModal(
            project_id=project.id,
            book_id=book_id,
            initial_char_id=char_id,
            on_change_callback=after_change
        )
        modal.open()

    def open_quick_characters_dialog():
        """Spawns an interactive split-screen project characters quick manager dialog."""
        def after_change():
            forget_character_cache()
            if modal_prompt_input:
                render_scene_character_chips(modal_prompt_input.value)

        modal = QuickCharactersModal(
            project_id=project.id, 
            book_id=book_id,
            on_change_callback=after_change
        )
        modal.open()

    def render_scene_character_chips(prompt_text: str):
        """Safely clears and populates matched character tags on the scene with descriptive hover tooltips."""
        tags = extract_prompt_character_tags(prompt_text)

        # The theater's cursor IS a (chapter, shot) position: ``book_active_scene``
        # is the row's ``shot``, not a narrative scene. Pass it through so every
        # description is resolved where the image will actually be rendered.
        target_ch = int(float(getattr(state, 'book_active_chapter', 1)))
        target_sc = int(float(getattr(state, 'book_active_scene', 1)))

        index, names = character_tag_index()
        wanted_ids: set = set()
        for tag in tags:
            for char_id in index.get(clean_character_key(tag), ()):
                wanted_ids.add(char_id)

        # Character order, so the row looks the same as it always did.
        matched_chars: List[Character] = [names[cid] for cid in names
                                          if cid in wanted_ids]

        # Same characters at the same position: nothing to redraw. This is what
        # keeps ordinary typing from clearing and rebuilding the row per key.
        signature = (tuple(char.id for char in matched_chars), target_ch, target_sc)
        if signature == chip_cache["signature"]:
            return
        chip_cache["signature"] = signature

        scene_chips_container.clear()
        if not matched_chars:
            return

        # The hover text is whatever the renderer would inject at this scene, so it
        # is produced by ``replace_character_tags_in_prompt`` rather than by
        # resolving an event here. That is what keeps the chip honest: pins, the
        # ``carried`` scope and the bracket rules all live in that path. The old
        # inline call applied none of them and compiled every trait.
        enabled_fields = getattr(state, "style_enabled_character_traits", {})
        use_sentence = getattr(state, "style_use_sentence_structure", True)

        def chip_tooltip(char: Character) -> str:
            key = (char.id, target_ch, target_sc)
            cached = chip_cache["tooltips"].get(key)
            if cached is not None:
                return cached
            try:
                text = replace_character_tags_in_prompt(
                    f"[{char.name}]",
                    project.id,
                    enabled_fields,
                    use_sentence,
                    book_id=book_id,
                    chapter_num=target_ch,
                    shot_num=target_sc,
                )
            except Exception as exc:
                print(f"[Proofreader-Chips] Could not compile '{char.name}': {exc}")
                return f"{char.name} (description unavailable)"
            text = (text or "").strip()
            # With no traits and no override the compiler names the character as
            # the fallback; there is no visual description to report yet.
            if (not text or text == char.name
                    or is_placeholder_description(text)):
                text = f"{char.name} (no description yet)"
            if len(chip_cache["tooltips"]) > 600:
                # Bounded: a long reviewing session touches thousands of
                # (character, position) pairs and none of them is precious.
                chip_cache["tooltips"] = {}
            chip_cache["tooltips"][key] = text
            return text

        if matched_chars:
            with scene_chips_container:
                with ui.row().classes('w-full items-center gap-1 bg-slate-50 p-2 rounded-lg border border-dashed mt-1 flex-wrap outline-none focus:outline-none'):
                    ui.label('Scene Characters:').classes('text-[9px] font-black text-slate-400 uppercase tracking-wider')
                    for char in matched_chars:
                        chip = ui.chip(
                            f"👤 {char.name}",
                            on_click=lambda _, cid=char.id: open_character_edit_dialog(cid)
                        ).classes('text-[11px] bg-white hover:bg-blue-50 hover:text-blue-700 cursor-pointer border py-0.5 px-2.5 rounded-md outline-none focus:outline-none')
                        with chip:
                            ui.tooltip(chip_tooltip(char)).classes('bg-slate-800 text-white text-[10px] p-2.5 rounded-md')

    if getattr(state, 'book_scroll_timer', None):
        try:
            state.book_scroll_timer.cancel()
            state.book_scroll_timer.delete()
        except Exception:
            pass
        state.book_scroll_timer = None

    if getattr(state, 'book_update_timer', None):
        try:
            state.book_update_timer.cancel()
            state.book_update_timer.delete()
        except Exception:
            pass
        state.book_update_timer = None

    if getattr(state, 'book_keyboard', None):
        try:
            state.book_keyboard.delete()
        except Exception:
            pass
        state.book_keyboard = None

    with Session(engine) as session:
        book = session.get(Book, book_id)
        if not book:
            ui.label("Book details not available.").classes('text-slate-400 text-sm')
            return
            
        from database.models import Project
        project = session.get(Project, book.project_id)
        if not project:
            ui.label("Project workspace details not available.").classes('text-slate-400 text-sm')
            return

    project_name = project.name
    book_name = book.name

    if not hasattr(state, 'key_approve'):
        state.key_approve = 'a'
    if not hasattr(state, 'key_delete'):
        state.key_delete = 'd'
    if not hasattr(state, 'key_next'):
        state.key_next = 'f'
    if not hasattr(state, 'key_prev'):
        state.key_prev = 's'
    if not hasattr(state, 'book_active_chapter'):
        state.book_active_chapter = 1
    if not hasattr(state, 'book_active_scene'):
        state.book_active_scene = 1
    if not hasattr(state, 'book_view_mode'):
        state.book_view_mode = 'Gallery'
    
    import time
    state.workspace_load_time = int(time.time())
    state.custom_image_timestamps = {}
    
    state.book_gallery_limit = 24

    from services.sync_engine import sync_prompts_csv_to_db_cache
    with Session(engine) as session:
        sync_prompts_csv_to_db_cache(book_id, session)
        session.commit()

    def get_all_prompts_as_dicts() -> List[Dict[str, Any]]:
        """Loads and converts ScenePrompts from SQLite to standard dicts compatible with the workspace."""
        from database.models import ScenePrompt
        with Session(engine) as session:
            query = select(ScenePrompt).where(ScenePrompt.book_id == book_id).order_by(ScenePrompt.chapter_num, ScenePrompt.shot_num)
            results = session.exec(query).all()
        
        return [
            {
                "chapter": str(r.chapter_num),
                "shot": str(r.shot_num),
                "prompt": r.prompt,
                "quote": r.quote,
                "approved": "True" if r.approved else "False",
                "timestamp": r.timestamp or "00:00:00"
            }
            for r in results
        ]

    # Pre-cache character names and aliases for robust tag search & resolution
    character_alias_map: Dict[int, List[str]] = {}
    with Session(engine) as session:
        book_characters = session.exec(
            select(Character).where(Character.project_id == project.id)
        ).all()
        for char in book_characters:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
            character_alias_map[char.id] = [clean_character_key(a.alias) for a in aliases] + [clean_character_key(char.name)]

    # Exact-alias -> character ids, inverted once per page render. The gallery
    # search used to walk every character's alias list for every tag of every
    # prompt *on every keystroke* (~1,000 prompts x ~1,000 characters on a long
    # book). That is a third of a second of blocking work per letter typed, which
    # starves the websocket until the client reconnects and rebuilds the page -
    # which is why the search box looked like it silently reset itself mid-word.
    # With the index the tag side of the search is a dictionary lookup.
    character_ids_by_alias: Dict[str, set] = {}
    for char_id, aliases in character_alias_map.items():
        for alias in aliases:
            character_ids_by_alias.setdefault(alias, set()).add(char_id)

    prompts = get_all_prompts_as_dicts()
    images_cache = get_book_images_cache(project_name, book_name)

    # Attach the shot role from prompts.csv. The ScenePrompt cache has no
    # shot_role column, so the role lives only in the CSV itself; image
    # generation reads it from there, and the workspace mirrors it here for the
    # grid badges and the shot-type filter. A legacy file with no role column
    # resolves every scene to Character, which is the base style it was
    # rendered with anyway.
    csv_role_by_key = {}
    for csv_row in load_prompts_csv(project_name, book_name):
        csv_role_by_key[_row_key(csv_row.get("chapter"), csv_row.get("shot"))] = csv_row.get("shot_role", "")
    for p in prompts:
        raw_role = csv_role_by_key.get(_row_key(p.get("chapter"), p.get("shot")), "")
        p["shot_role"] = style_library.normalize_role(raw_role)

    if not hasattr(state, 'book_active_scene_idx'):
        state.book_active_scene_idx = 0

    # Fast in-memory search container for keyboard queries
    search_filter_container = [None]

    # Typing debounce. Every keystroke used to rebuild the whole gallery, so a
    # burst of typing queued one full refresh per character and kept the event
    # loop blocked long enough for the client to time out and reload the page
    # (losing the typed query). Filter once the typing pauses instead.
    search_refresh_task: List[Optional[asyncio.Task]] = [None]

    def cancel_pending_search_refresh() -> None:
        task = search_refresh_task[0]
        if task is not None and not task.done():
            task.cancel()
        search_refresh_task[0] = None

    async def _refresh_gallery_after_pause() -> None:
        try:
            await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            return
        search_refresh_task[0] = None
        try:
            render_content.refresh()
        except (RuntimeError, AssertionError):
            # The page (or this client) went away while the timer was pending.
            pass

    def schedule_search_refresh() -> None:
        cancel_pending_search_refresh()
        search_refresh_task[0] = asyncio.create_task(_refresh_gallery_after_pause())

    def flush_search_refresh() -> None:
        """Apply the pending query now (Enter), instead of waiting out the pause."""
        cancel_pending_search_refresh()
        render_content.refresh()

    # --- 1. NESTED HANDLERS DEFINED FIRST ---

    def open_directory(path: Path):
        abs_path = path.resolve()
        if not abs_path.exists():
            abs_path.mkdir(parents=True, exist_ok=True)
        try:
            if platform.system() == "Windows":
                os.startfile(abs_path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(abs_path)])
            else:
                subprocess.Popen(["xdg-open", str(abs_path)])
        except Exception as e:
            ui.notify(f"Failed to open directory: {str(e)}", type="negative")

    # Per-prompt normalised text and tagged characters, rebuilt only when the
    # prompt text itself changes. Keyed by ``id(row)``: ``prompts`` holds the only
    # reference to each dict for the life of the page, so the id is stable.
    prompt_search_cache: Dict[int, tuple] = {}

    def prompt_search_facts(row: Dict[str, Any]) -> tuple:
        """(normalised prompt, normalised quote, tagged character ids) for a row."""
        raw_prompt = row.get("prompt") or ""
        cached = prompt_search_cache.get(id(row))
        if cached is not None and cached[0] == raw_prompt:
            return cached[1], cached[2], cached[3]

        norm_prompt = normalize_character_key(raw_prompt)
        tagged_ids: set = set()
        for tag in re.findall(r"\[(.*?)\]", norm_prompt):
            tagged_ids.update(character_ids_by_alias.get(clean_character_key(tag), ()))
        facts = (
            raw_prompt,
            norm_prompt,
            normalize_character_key(row.get("quote") or ""),
            tagged_ids,
        )
        prompt_search_cache[id(row)] = facts
        return facts[1], facts[2], facts[3]

    def get_filtered_prompts():
        query = ""
        if search_filter_container[0] and search_filter_container[0].value:
            query = normalize_character_key(search_filter_container[0].value)

        filtered = []
        
        # Parse search query into individual terms
        terms = []
        if query:
            brackets = re.findall(r'\[(.*?)\]', query)
            if brackets:
                terms = [normalize_character_key(b) for b in brackets if b.strip()]
            elif ',' in query:
                terms = [normalize_character_key(t) for t in query.split(',') if t.strip()]
            else:
                terms = [normalize_character_key(t) for t in query.split(' ') if t.strip()]
                if len(terms) > 1:
                    terms = [t for t in terms if t not in ('and', 'or', '&', 'with', 'in')]

        # Resolve each term to the characters it names once for the whole pass,
        # not once per prompt. Substring matching is deliberate: "mat" is meant to
        # also find "Mat Cauthon", not only an exact tag.
        term_character_ids = []
        for term in terms:
            term_matched_char_ids = set()
            for char_id, aliases in character_alias_map.items():
                if any(term in alias for alias in aliases):
                    term_matched_char_ids.add(char_id)
            term_character_ids.append((term, term_matched_char_ids))

        shot_choice = shot_filter.value if shot_filter is not None else "All Roles"
        mode = filter_mode.value

        for p in prompts:
            if term_character_ids:
                p_prompt, p_quote, scene_character_ids = prompt_search_facts(p)
                ch_str = f"ch {p.get('chapter', '')}"
                sc_str = f"sc {p.get('shot', '')}"

                matched_all_terms = True
                for term, term_matched_char_ids in term_character_ids:
                    if term_matched_char_ids:
                        if not (term_matched_char_ids & scene_character_ids):
                            matched_all_terms = False
                            break
                    else:
                        if term not in p_prompt and term not in p_quote and term not in ch_str and term not in sc_str:
                            matched_all_terms = False
                            break
                            
                if not matched_all_terms:
                    continue

            # Shot-type filter (roles resolved from prompts.csv).
            if not shot_role_matches(p, shot_choice):
                continue

            is_approved = p.get("approved", "False").strip().lower() == "true"
            try:
                ch = int(float(p.get("chapter", "1")))
                sc = int(float(p.get("shot", "1")))
            except ValueError:
                ch, sc = 1, 1
            has_image = (ch, sc) in images_cache
            
            if mode == "Unapproved Only":
                if has_image and not is_approved:
                    filtered.append(p)
            elif mode == "Missing Only":
                if not has_image:
                    filtered.append(p)
            else:
                filtered.append(p)
        return filtered

    def update_prompt_text(val: str, row_dict: dict):
        if not row_dict or val is None:
            return
        cleaned = val.strip()
        # Guard: never allow an accidental blank or UI clear to overwrite a valid prompt
        if not cleaned and row_dict.get("prompt"):
            return
        if cleaned == row_dict.get("prompt", ""):
            return
        row_dict["prompt"] = cleaned
        save_prompts_csv(project_name, book_name, prompts)
        try:
            ch = int(float(row_dict.get("chapter", "1")))
            sc = int(float(row_dict.get("shot", "1")))
            if (ch, sc) in grid_card_references:
                grid_card_references[(ch, sc)]["item"]["prompt"] = cleaned
        except ValueError:
            pass

    # --- Active Theater Review Queue & FIFO History State ---
    theater_queue: List[Dict[str, Any]] = []
    theater_cursor: int = 0
    history_stack: List[Dict[str, Any]] = []
    history_pointer: int = -1  # -1 indicates active in live queue; >= 0 indicates inspecting history

    def get_current_filtered_index(filtered_list: list) -> int:
        for idx, p in enumerate(filtered_list):
            try:
                ch = int(float(p.get("chapter", "1")))
                sc = int(float(p.get("shot", "1")))
            except ValueError:
                ch, sc = 1, 1
            if ch == state.book_active_chapter and sc == state.book_active_scene:
                return idx
        return 0

    def update_active_scene_ui():
        nonlocal theater_cursor, history_pointer
        current_scene = None
        is_history = False
        history_offset = 0

        if history_pointer != -1 and 0 <= history_pointer < len(history_stack):
            current_scene = history_stack[history_pointer]
            is_history = True
            history_offset = len(history_stack) - history_pointer
        elif theater_queue:
            theater_cursor = min(max(0, theater_cursor), len(theater_queue) - 1)
            current_scene = theater_queue[theater_cursor]

        if not current_scene:
            # Active filter review queue is empty
            active_row_ref[0] = None
            loaded_scene_coords[0] = -1
            loaded_scene_coords[1] = -1

            if modal_img_el:
                modal_img_el.visible = False
            if modal_placeholder:
                modal_placeholder.visible = True
            if modal_title_el:
                modal_title_el.set_text("Review Queue Complete")
            if modal_subtitle_el:
                modal_subtitle_el.set_text("All scenes matching the active filter have been reviewed.")
            if modal_badge_missing:
                modal_badge_missing.visible = False
            if modal_badge_approved:
                modal_badge_approved.visible = False
            if modal_badge_review:
                modal_badge_review.visible = False
            if modal_quote_el:
                modal_quote_el.set_text("")
            if modal_prompt_input:
                modal_prompt_input.set_value("")
            if modal_role_select:
                modal_role_select.disable()
            if modal_regen_btn[0]:
                modal_regen_btn[0].disable()
            return

        active_row_ref[0] = current_scene

        try:
            ch = int(float(current_scene.get("chapter", "1")))
            sc = int(float(current_scene.get("shot", "1")))
        except ValueError:
            ch, sc = 1, 1

        state.book_active_chapter = ch
        state.book_active_scene = sc

        img_url = images_cache.get((ch, sc))
        is_approved = current_scene.get("approved", "False").strip().lower() == "true"

        if modal_img_el:
            if img_url:
                if modal_img_el.source != img_url:
                    modal_img_el.set_source(img_url)
                modal_img_el.visible = True
            else:
                modal_img_el.visible = False

        if modal_placeholder:
            modal_placeholder.visible = not img_url

        if modal_title_el:
            modal_title_el.set_text(f"Chapter {current_scene.get('chapter')}, Scene {current_scene.get('shot')}")

        if modal_subtitle_el:
            if is_history:
                step_str = f"{history_offset} step back" if history_offset == 1 else f"{history_offset} steps back"
                modal_subtitle_el.set_text(f"Viewing History ({step_str}) • Press [F] to return to review queue")
            else:
                modal_subtitle_el.set_text(f"Review Scene: {theater_cursor + 1} of {len(theater_queue)} remaining")

        if modal_badge_missing:
            modal_badge_missing.visible = not img_url
        if modal_badge_approved:
            modal_badge_approved.visible = bool(img_url and is_approved)
        if modal_badge_review:
            modal_badge_review.visible = bool(img_url and not is_approved)

        # Only one card carries the selection ring, so remembering which one it is
        # avoids sending a class update for every card in the grid on each pass -
        # the in-place image updates now run on every new render, and the grid can
        # hold hundreds of cards.
        previous_ring = ringed_card[0]
        if previous_ring is not None:
            previous_ring.classes(remove="ring-4 ring-blue-500 ring-offset-2")
            ringed_card[0] = None

        if (ch, sc) in grid_card_references:
            target_ref = grid_card_references[(ch, sc)]
            target_ref["card"].classes(add="ring-4 ring-blue-500 ring-offset-2")
            ringed_card[0] = target_ref["card"]

        # Always update target narration quote text for the active scene
        if modal_quote_el:
            modal_quote_el.set_text(quoted_for_display(current_scene.get("quote", "")))

        if loaded_scene_coords[0] != ch or loaded_scene_coords[1] != sc:
            loaded_scene_coords[0] = ch
            loaded_scene_coords[1] = sc

            theater_context_state["window_before"] = 400
            theater_context_state["window_after"] = 400

            if modal_prompt_input:
                modal_prompt_input.set_value(current_scene.get("prompt", ""))

            if modal_context_html[0]:
                update_theater_context_view()

            # The chips follow the prompt, and the prompt belongs to the scene, so
            # they are rebuilt with it rather than on every in-place image update.
            if modal_prompt_input:
                render_scene_character_chips(modal_prompt_input.value)
        elif modal_prompt_input and not modal_prompt_input.value and current_scene.get("prompt"):
            # Self-heal prompt input if it was cleared by a previous empty state
            modal_prompt_input.set_value(current_scene.get("prompt", ""))
            render_scene_character_chips(modal_prompt_input.value)

        # Mirror the scene's shot role into the theater selector (guard the
        # change handler so programmatic sync never counts as a user edit).
        current_role = style_library.normalize_role(current_scene.get("shot_role"))
        if modal_role_select:
            modal_role_select.enable()
            if modal_role_select.value != SHOT_ROLE_LABELS[current_role]:
                modal_role_select.set_value(SHOT_ROLE_LABELS[current_role])
        if modal_regen_btn[0]:
            modal_regen_btn[0].enable()

    def next_scene():
        nonlocal theater_cursor, history_pointer
        if history_pointer != -1:
            # Walking forward through visited history back toward live queue
            if history_pointer < len(history_stack) - 1:
                history_pointer += 1
            else:
                history_pointer = -1
            update_active_scene_ui()
            return

        if not theater_queue:
            return

        # Skipping forward without approving: store in history so 'Prev' can return
        current_scene = theater_queue[theater_cursor]
        history_stack.append(current_scene)
        if len(history_stack) > 20:
            history_stack.pop(0)

        if theater_cursor < len(theater_queue) - 1:
            theater_cursor += 1
            update_active_scene_ui()
        else:
            ui.notify("End of review queue reached.", type="info", timeout=1.0)

    def prev_scene():
        nonlocal theater_cursor, history_pointer
        if history_pointer != -1:
            # Walking further back in history
            if history_pointer > 0:
                history_pointer -= 1
                update_active_scene_ui()
            else:
                ui.notify("Earliest history item reached.", type="info", timeout=1.0)
            return

        # Currently in active review queue: step into history stack if available
        if history_stack:
            history_pointer = len(history_stack) - 1
            update_active_scene_ui()
        elif theater_cursor > 0:
            theater_cursor -= 1
            update_active_scene_ui()
        else:
            ui.notify("Beginning of review queue reached.", type="info", timeout=1.0)

    def approve_current():
        nonlocal theater_cursor, history_pointer
        if history_pointer != -1 and 0 <= history_pointer < len(history_stack):
            # Approving while inspecting an item from history
            row = history_stack[history_pointer]
            row["approved"] = "True"
            save_prompts_csv(project_name, book_name, prompts)

            try:
                ch = int(float(row.get("chapter", "1")))
                sc = int(float(row.get("shot", "1")))
                if (ch, sc) in grid_card_references:
                    grid_card_references[(ch, sc)]["item"]["approved"] = "True"
            except ValueError:
                pass

            update_grid_views_in_place()
            ui.notify(f"Ch {row.get('chapter')}, Sc {row.get('shot')} Approved!", type="positive", timeout=1.0)
            next_scene()
            return

        if not theater_queue:
            return

        row = theater_queue[theater_cursor]
        row["approved"] = "True"
        save_prompts_csv(project_name, book_name, prompts)

        try:
            ch = int(float(row.get("chapter", "1")))
            sc = int(float(row.get("shot", "1")))
            if (ch, sc) in grid_card_references:
                grid_card_references[(ch, sc)]["item"]["approved"] = "True"
        except ValueError:
            pass

        update_grid_views_in_place()
        ui.notify(f"Ch {row.get('chapter')}, Sc {row.get('shot')} Approved!", type="positive", timeout=1.0)

        # Log into visited FIFO history
        history_stack.append(row)
        if len(history_stack) > 20:
            history_stack.pop(0)

        # In filtered review modes, remove the approved item immediately from queue
        if filter_mode.value in ("Unapproved Only", "Missing Only"):
            theater_queue.pop(theater_cursor)
            if theater_cursor >= len(theater_queue):
                theater_cursor = max(0, len(theater_queue) - 1)
        else:
            theater_cursor = min(theater_cursor + 1, len(theater_queue) - 1)

        update_active_scene_ui()

    def delete_current():
        nonlocal theater_cursor, history_pointer
        if history_pointer != -1 and 0 <= history_pointer < len(history_stack):
            # Deleting while inspecting an item from history
            row = history_stack[history_pointer]
            row["approved"] = "False"
            save_prompts_csv(project_name, book_name, prompts)

            was_deleted = delete_scene_image_file(project_name, book_name, row.get("chapter", "1"), row.get("shot", "1"))

            try:
                ch = int(float(row.get("chapter", "1")))
                sc = int(float(row.get("shot", "1")))
                images_cache.pop((ch, sc), None)
                if (ch, sc) in grid_card_references:
                    grid_card_references[(ch, sc)]["item"]["approved"] = "False"
            except ValueError:
                pass

            update_grid_views_in_place()
            if was_deleted:
                ui.notify(f"Deleted image for Ch {row.get('chapter')}, Sc {row.get('shot')}!", type="warning", timeout=1.0)
            else:
                ui.notify(f"Ch {row.get('chapter')}, Sc {row.get('shot')} file was already missing.", type="info", timeout=1.0)

            update_active_scene_ui()
            return

        if not theater_queue:
            return

        row = theater_queue[theater_cursor]
        row["approved"] = "False"
        save_prompts_csv(project_name, book_name, prompts)

        was_deleted = delete_scene_image_file(project_name, book_name, row.get("chapter", "1"), row.get("shot", "1"))

        try:
            ch = int(float(row.get("chapter", "1")))
            sc = int(float(row.get("shot", "1")))
            images_cache.pop((ch, sc), None)
            if (ch, sc) in grid_card_references:
                grid_card_references[(ch, sc)]["item"]["approved"] = "False"
        except ValueError:
            pass

        update_grid_views_in_place()

        if was_deleted:
            ui.notify(f"Deleted image for Ch {row.get('chapter')}, Sc {row.get('shot')}!", type="warning", timeout=1.0)
        else:
            ui.notify(f"Ch {row.get('chapter')}, Sc {row.get('shot')} file was already missing.", type="info", timeout=1.0)

        # Log deleted scene into history so user can inspect, edit prompt, or undo if needed
        history_stack.append(row)
        if len(history_stack) > 20:
            history_stack.pop(0)

        if filter_mode.value == "Unapproved Only":
            # Image is now missing (no longer unapproved-with-image), remove from queue
            theater_queue.pop(theater_cursor)
            if theater_queue:
                if theater_cursor >= len(theater_queue):
                    theater_cursor = max(0, len(theater_queue) - 1)
            else:
                # If that was the last scene in queue, smoothly remain on it in history mode
                history_pointer = len(history_stack) - 1
        else:
            theater_cursor = min(theater_cursor + 1, len(theater_queue) - 1)

        update_active_scene_ui()

    async def trigger_batch_restart():
        start_fn = getattr(state, 'start_image_generation_cb', None)
        if not start_fn:
            ui.notify("Pipeline process control callbacks are not fully registered in state.", type="negative")
            return

        if state.project_status == "Rendering Images" or state.image_gen_active:
            ui.notify("Waiting for current image to finish rendering before restarting...", type="info", timeout=2.0)
            state.cancel_image_gen_flag = True
            
            while state.image_gen_active:
                await asyncio.sleep(0.5)
                
        state.cancel_image_gen_flag = False
        state.image_gen_active = False
        start_fn(project.id)

    def trigger_stop_rendering():
        stop_fn = getattr(state, 'stop_image_generation_cb', None)
        if stop_fn:
            stop_fn(project.id)
            ui.notify("Stop signal dispatched. Halting after active image completes...", type="warning")
        else:
            ui.notify("Stop callback is not registered.", type="negative")

    def update_grid_views_in_place():
        """Updates rendering frames, cards, and dots in-place with zero layout shifts."""
        shot_choice = shot_filter.value if shot_filter is not None else "All Roles"
        for (ch, sc), ref in grid_card_references.items():
            img_url = images_cache.get((ch, sc))
            is_approved = ref["item"].get("approved", "False").strip().lower() == "true"
            has_image = bool(img_url)

            role_ok = shot_role_matches(ref["item"], shot_choice)

            should_be_visible = True
            if filter_mode.value == "Unapproved Only":
                should_be_visible = bool(has_image and not is_approved)
            elif filter_mode.value == "Missing Only":
                should_be_visible = not has_image

            should_be_visible = should_be_visible and role_ok
            ref["card"].visible = should_be_visible
            
            if not should_be_visible:
                continue
            
            if not img_url:
                border_style = "border-red-300 bg-red-50/10"
                dot_color = "bg-red-500"
            elif is_approved:
                border_style = "border-emerald-300 bg-emerald-50/10"
                dot_color = "bg-emerald-500"
            else:
                border_style = "border-amber-300 bg-amber-50/10"
                dot_color = "bg-amber-500"
                
            ref["card"].classes(replace=f"border rounded-lg shadow-sm p-2 cursor-pointer hover:shadow-md transition-all {border_style} outline-none focus:outline-none")
            ref["dot"].classes(replace=f"w-2 h-2 rounded-full {dot_color}")
            
            if img_url:
                if ref["image"].source != img_url:
                    ref["image"].set_source(img_url)
                ref["image"].visible = True
                ref["placeholder"].visible = False
            else:
                ref["image"].visible = False
                ref["placeholder"].visible = True

    def apply_shot_role(label: str):
        """Persist a shot-role change made from the theater selector.

        Changing the role alone re-styles the scene on the next render; the
        prompt *text* is regenerated only when the user asks for a new prompt.
        """
        row = active_row_ref[0]
        new_role = SHOT_ROLE_FROM_LABEL.get(label)
        if not row or not new_role:
            return
        if style_library.normalize_role(row.get("shot_role")) == new_role:
            return

        row["shot_role"] = new_role
        try:
            ch = int(float(row.get("chapter", "1")))
            sc = int(float(row.get("shot", "1")))
            if (ch, sc) in grid_card_references:
                grid_card_references[(ch, sc)]["item"]["shot_role"] = new_role
        except ValueError:
            pass

        save_prompts_csv(project_name, book_name, prompts)

        # The new role makes this book multi-role; surface it in the filter.
        new_present = roles_present(prompts)
        if shot_filter is not None:
            shot_filter.options = ["All Roles"] + [SHOT_ROLE_LABELS[r] for r in new_present]
            shot_filter.set_visibility(len(new_present) > 1)
            shot_filter.update()

        render_content.refresh()
        update_active_scene_ui()
        ui.notify(
            f"Shot type set to {SHOT_ROLE_LABELS[new_role]}. "
            f"Click 'New Prompt' to regenerate the text for it.",
            type="info", timeout=3.0
        )

    async def request_new_prompt():
        """Regenerate one scene's prompt text with its selected shot role.

        ``get_llm_response`` routes through the VRAM manager, so ComfyUI's cache
        is released before the LLM takes the GPU - the whole reason this action
        is safe to fire while ComfyUI is still resident.
        """
        row = active_row_ref[0]
        if not row:
            return
        try:
            ch = int(float(row.get("chapter", "1")))
            sc = int(float(row.get("shot", "1")))
        except ValueError:
            ch, sc = 1, 1

        role = style_library.normalize_role(row.get("shot_role"))

        from services import shot_planner, prompt_library
        from services.prompt_engine import (
            get_llm_response, parse_llm_response, REFUSAL_KEYWORDS, is_unusable_prompt,
        )

        try:
            chunk = shot_planner.scene_chunk(project_name, book_name, ch, sc)
        except Exception as ex:
            ui.notify(f"Could not re-derive this scene's passage: {ex}", type="negative")
            return
        if not chunk:
            ui.notify("Could not re-derive this scene's passage from transcript.txt.", type="negative")
            return

        request_text = prompt_library.render_with_selection(role, chunk, prompt_library.active_selection())
        llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
        model_name = get_setting("llm_model", "local-model")

        btn = modal_regen_btn[0]
        if btn:
            btn.props('loading')
        state.add_console_log(
            f"[Theater] Requesting a fresh {SHOT_ROLE_LABELS[role]} prompt for Ch {ch}, Scene {sc}."
        )

        try:
            raw = await get_llm_response(request_text, llm_url, model_name)
        except Exception as e:
            raw = f"API Connection Failed: {e}"
        finally:
            if btn:
                btn.props(remove='loading')

        raw_text = raw if isinstance(raw, str) else str(raw)
        parsed = parse_llm_response(raw_text)
        # ``parse_llm_response`` falls back to the whole reply as the prompt when
        # the model ignores the QUOTE/PROMPT format, which would let a plain
        # refusal overwrite a good prompt. Reject those here - and reject a
        # lawful decline too: the INSERT fragment answers NONE when the passage
        # holds nothing worth showing close, and writing "NONE" into the row only
        # parks the same hole in the file for the renderer to skip.
        looks_like_refusal = any(k in raw_text.lower() for k in REFUSAL_KEYWORDS)
        if is_unusable_prompt(parsed.get("prompt")) or looks_like_refusal:
            state.add_console_log(
                f"[Theater] No usable {SHOT_ROLE_LABELS[role]} prompt for Ch {ch}, Scene {sc}:\n"
                f"{parsed.get('raw', '')}"
            )
            alternatives = prompt_library.fallback_roles(role)
            if parsed.get("status") == "declined" and alternatives:
                hint = (f" This passage suits {SHOT_ROLE_LABELS.get(alternatives[0], alternatives[0])}"
                        f" better - switch the shot type and try again.")
            else:
                hint = " Prompt left unchanged."
            ui.notify(f"The {SHOT_ROLE_LABELS[role]} prompt declined this passage.{hint}",
                      type="warning", timeout=6.0)
            return

        row["prompt"] = parsed["prompt"]
        new_quote = (parsed.get("quote") or "").strip()
        if new_quote:
            row["quote"] = new_quote

        save_prompts_csv(project_name, book_name, prompts)

        if (ch, sc) in grid_card_references:
            grid_card_references[(ch, sc)]["item"]["prompt"] = row["prompt"]
            grid_card_references[(ch, sc)]["item"]["quote"] = row.get("quote", "")

        # Force the modal to reload the scene's text and narration context.
        loaded_scene_coords[0] = -1
        update_active_scene_ui()
        ui.notify(f"New {SHOT_ROLE_LABELS[role]} prompt generated for Ch {ch}, Scene {sc}.", type="positive")

    def launch_theater_from_coordinate(ch_val: int, sc_val: int):
        nonlocal theater_cursor, history_pointer
        state.book_active_chapter = ch_val
        state.book_active_scene = sc_val

        # Populate active queue from currently applied filter
        filtered = get_filtered_prompts()
        theater_queue.clear()
        theater_queue.extend(filtered)

        # Locate clicked item in queue
        found_idx = 0
        for idx, item in enumerate(theater_queue):
            try:
                ch = int(float(item.get("chapter", "1")))
                sc = int(float(item.get("shot", "1")))
                if ch == ch_val and sc == sc_val:
                    found_idx = idx
                    break
            except ValueError:
                pass

        theater_cursor = found_idx
        history_pointer = -1
        history_stack.clear()
        loaded_scene_coords[0] = -1
        loaded_scene_coords[1] = -1

        theater_dialog.open()
        update_active_scene_ui()

    def refresh_theater_queue():
        """Reconcile the open theater's queue with the active filter.

        The queue is a snapshot taken when the modal opened, but the two review
        filters describe *live* membership: a scene that was missing gains an
        image and leaves "Missing Only", a scene that gains its first image joins
        "Unapproved Only". So every scene the renderer finished while the modal
        was open was invisible - unreachable by Prev/Next, uncounted in "N of M
        remaining" - until the modal was closed and reopened.

        The rows themselves are shared with ``prompts``, so only the list is
        rebuilt: the cursor is re-found by (chapter, shot) and the history stack,
        which holds row references, stays valid.
        """
        nonlocal theater_cursor
        if filter_mode.value == "All":
            return  # membership is fixed; only the images themselves change

        fresh = get_filtered_prompts()
        if not fresh:
            return

        anchor = theater_queue[theater_cursor] if theater_queue else None
        theater_queue[:] = fresh

        if anchor is not None:
            anchor_key = _row_key(anchor.get("chapter"), anchor.get("shot"))
            for idx, row in enumerate(theater_queue):
                if _row_key(row.get("chapter"), row.get("shot")) == anchor_key:
                    theater_cursor = idx
                    return
        # The anchor has left the filter - it just got an image, or was approved.
        # The same ordinal now holds the next scene still matching.
        theater_cursor = min(theater_cursor, len(theater_queue) - 1)

    # --- 2. LAYOUT RENDERING COMPONENT DECLARATIONS ---

    if not prompts:
        transcript_path = Path(f"./output/{project_name}/{book_name}/transcript.txt")
        has_transcript = transcript_path.exists()
        
        if has_transcript:
            approved_marker_path = Path(f"./output/{project_name}/{book_name}/.transcript_approved")
            
            @ui.refreshable
            def render_transcript_editor():
                is_approved = approved_marker_path.exists()
                try:
                    text_content = transcript_path.read_text(encoding="utf-8")
                except Exception as e:
                    text_content = f"Error reading transcript: {str(e)}"
                    
                char_count = len(text_content)
                word_count = len(text_content.split())
                
                with ui.card().classes('w-full border p-6 shadow-sm bg-white gap-4 outline-none focus:outline-none'):
                    with ui.row().classes('items-center justify-between border-b pb-3 w-full'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('description', size='md', color='blue-500')
                            with ui.column().classes('gap-0'):
                                ui.label('Step 1.5: Transcript Review & Formatting').classes('text-base font-bold text-slate-800')
                                ui.label('Review text, prune publishing boilerplate, and mark as approved to unlock prompt generation.').classes('text-xs text-slate-500')
                        
                        if is_approved:
                            ui.badge('Approved & Ready', color='emerald').classes('px-3 py-1 text-xs font-semibold rounded-full')
                        else:
                            ui.badge('Awaiting Review', color='amber').classes('px-3 py-1 text-xs font-semibold rounded-full')
                    
                    with ui.row().classes('w-full justify-start gap-6 bg-slate-50 p-3 rounded-lg border border-dashed text-xs'):
                        with ui.column().classes('gap-0'):
                            ui.label('Character Count').classes('text-slate-400 font-medium')
                            ui.label(f"{char_count:,} characters").classes('font-bold text-slate-700')
                        with ui.column().classes('gap-0'):
                            ui.label('Word Count').classes('text-slate-400 font-medium')
                            ui.label(f"{word_count:,} words").classes('font-bold text-slate-700')
                        with ui.column().classes('gap-0'):
                            ui.label('Estimated Scenes').classes('text-slate-400 font-medium')
                            ui.label(f"~ {max(1, word_count // 350)} scenes").classes('font-bold text-slate-700')
                    
                    def save_transcript():
                        try:
                            transcript_path.write_text(editor.value, encoding="utf-8")
                            ui.notify("Transcript file saved successfully!", type="positive")
                            render_transcript_editor.refresh()
                        except Exception as e:
                            ui.notify(f"Failed to save: {str(e)}", type="negative")
                            
                    def open_in_system_editor():
                        import platform
                        import subprocess
                        abs_path = transcript_path.resolve()
                        try:
                            if platform.system() == "Windows":
                                os.startfile(abs_path)
                            elif platform.system() == "Darwin":
                                subprocess.Popen(["open", str(abs_path)])
                            else:
                                subprocess.Popen(["xdg-open", str(abs_path)])
                            ui.notify("Opening in native text editor...", type="info")
                        except Exception as e:
                            ui.notify(f"Could not open editor: {str(e)}", type="negative")
                            
                    def reload_from_disk():
                        render_transcript_editor.refresh()

                    def toggle_approval():
                        try:
                            transcript_path.write_text(editor.value, encoding="utf-8")
                        except Exception as e:
                            state.add_console_log(f"[Workspace] Failed to auto-save transcript on approval: {str(e)}")

                        if approved_marker_path.exists():
                            approved_marker_path.unlink()
                            ui.notify("Transcript approval revoked.", type="warning")
                        else:
                            approved_marker_path.touch()
                            ui.notify("Transcript approved! Ready for Phase 2 prompt generation.", type="positive")
                        
                        render_transcript_editor.refresh()
                        
                        if state.stats_refresh_callback:
                            asyncio.create_task(state.stats_refresh_callback())
                    
                    with ui.row().classes('w-full justify-between items-center border-b pb-3 mb-1'):
                        with ui.row().classes('gap-2'):
                            ui.button('Save Changes', icon='save', on_click=save_transcript).classes('bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold px-4')
                            ui.button('Reload from Disk', icon='refresh', on_click=reload_from_disk).classes('bg-slate-500 hover:bg-slate-600 text-white text-xs font-semibold px-4')
                            ui.button('Open in External Editor', icon='open_in_new', on_click=open_in_system_editor).props('flat').classes('text-xs text-slate-600')
                            
                        if is_approved:
                            ui.button('Revoke Approval', icon='cancel', on_click=toggle_approval).classes('bg-red-600 hover:bg-red-700 text-white text-xs font-bold px-5')
                        else:
                            ui.button('Approve Transcript', icon='check_circle', on_click=toggle_approval).classes('bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold px-5')

                    editor = ui.textarea(
                        label="Editable transcript.txt", 
                        value=text_content
                    ).classes('w-full text-xs') \
                     .props('outlined input-style="height: 500px; font-family: monospace; overflow-y: auto;"') \
                     .style('height: 520px;')

            render_transcript_editor()
            return

        else:
            with ui.card().classes('w-full border p-6 shadow-sm bg-white gap-4 outline-none focus:outline-none'):
                with ui.row().classes('items-center gap-2 border-b pb-3 w-full'):
                    ui.icon('pending_actions', size='md', color='amber-500')
                    with ui.column().classes('gap-0'):
                        ui.label('Awaiting Generation Pipeline').classes('text-base font-bold text-slate-800')
                        ui.label('Complete the initial setup phases to start image proofing.').classes('text-xs text-slate-500')
                
                with ui.grid(columns='1fr 1fr').classes('w-full gap-4'):
                    with ui.column().classes('gap-3 bg-slate-50 p-4 rounded-xl border border-dashed'):
                        ui.label('Volume Statistics').classes('text-xs font-bold text-slate-700 uppercase tracking-wide')
                        
                        with ui.row().classes('items-center justify-between w-full text-xs'):
                            ui.label('Transcript File:').classes('text-slate-500')
                            ui.badge('Missing', color='red').classes('px-2 py-0.5 text-[10px]')
                                
                        with ui.row().classes('items-center justify-between w-full text-xs'):
                            ui.label('Character Count:').classes('text-slate-500')
                            ui.label("0 characters").classes('font-bold text-slate-700')
                            
                        with ui.row().classes('items-center justify-between w-full text-xs'):
                            ui.label('Word Count:').classes('text-slate-500')
                            ui.label("0 words").classes('font-bold text-slate-700')

                        with ui.row().classes('items-center justify-between w-full text-xs'):
                            ui.label('Estimated Scenes:').classes('text-slate-500')
                            ui.label("~ 0 scenes").classes('font-bold text-slate-700')

                    with ui.column().classes('gap-3 bg-slate-50 p-4 rounded-xl border border-dashed'):
                        ui.label('Orchestration Guide').classes('text-xs font-bold text-slate-700 uppercase tracking-wide')
                        
                        with ui.row().classes('items-center gap-2 text-xs'):
                            ui.icon('radio_button_unchecked', color='slate', size='16px')
                            ui.label('Step 1: Transcription').classes('font-bold text-slate-700')
                            
                        with ui.row().classes('items-center gap-2 text-xs'):
                            ui.icon('radio_button_unchecked', color='slate', size='16px')
                            ui.label('Step 2: Generate Prompts').classes('font-bold text-slate-700')
                            
                        with ui.row().classes('items-center gap-2 text-xs'):
                            ui.icon('radio_button_unchecked', color='slate', size='16px')
                            ui.label('Step 3: Render Images').classes('font-bold text-slate-500')
                            
                with ui.row().classes('w-full justify-end mt-2 border-t pt-3'):
                    ui.label("Switch to the 'Dashboard' tab on the project workspace to run these steps.").classes('text-[11px] text-slate-500 italic')
            return

    filter_mode = ui.select(
        options=["All", "Unapproved Only", "Missing Only"], 
        value="All"
    ).classes('hidden')

    # Roles actually present in this book's prompts.csv, in canonical order.
    present_roles = roles_present(prompts)

    # --- Optimized Dense One-Line Toolbar Header ---
    # `sticky-under-header` (styles live in main.py's global head block) keeps
    # the toolbar pinned below the fixed app header while the gallery scrolls,
    # so Regen/Stop stay reachable without scrolling back up.
    with ui.row().classes('sticky-under-header w-full justify-between items-center bg-white py-1.5 px-3 border rounded-xl shadow-sm mb-3 flex-nowrap gap-2 outline-none focus:outline-none'):
        
        # Left-aligned toolbar group (Filter switcher, Search bar, Regen & Stop)
        with ui.row().classes('items-center gap-2 flex-nowrap flex-1'):
            
            # 1. Filter selectors (All, Unapproved, Missing)
            with ui.row().classes('items-center gap-0.5 bg-slate-100 p-0.5 rounded-lg border flex-nowrap'):
                btn_all = ui.button(icon='grid_view').props('flat dense').classes('px-2 py-1 rounded-md text-xs')
                btn_unapproved = ui.button(icon='rate_review').props('flat dense').classes('px-2 py-1 rounded-md text-xs')
                btn_missing = ui.button(icon='image_not_supported').props('flat dense').classes('px-2 py-1 rounded-md text-xs')
                
                with btn_all:
                    ui.tooltip('Show All')
                with btn_unapproved:
                    ui.tooltip('Unapproved Only')
                with btn_missing:
                    ui.tooltip('Missing Only')
                    
                def update_button_styles(mode: str):
                    for m, btn in [('All', btn_all), ('Unapproved Only', btn_unapproved), ('Missing Only', btn_missing)]:
                        if mode == m:
                            btn.classes(replace='px-2 py-1 rounded-md bg-white text-blue-600 shadow-sm font-bold')
                        else:
                            btn.classes(replace='px-2 py-1 rounded-md text-slate-500 hover:text-slate-700 hover:bg-slate-200/50')
                            
                def set_filter(mode: str):
                    filter_mode.value = mode
                    update_button_styles(mode)
                    render_content.refresh()
                    
                btn_all.on('click', lambda: set_filter('All'))
                btn_unapproved.on('click', lambda: set_filter('Unapproved Only'))
                btn_missing.on('click', lambda: set_filter('Missing Only'))
                
                update_button_styles(filter_mode.value)

            # 1b. Shot-type filter. Hidden for books whose scenes are all the
            # base Character role (legacy files, or the planner left off).
            shot_filter = ui.select(
                options=["All Roles"] + [SHOT_ROLE_LABELS[r] for r in present_roles],
                value="All Roles",
                on_change=lambda e: render_content.refresh(),
            ).classes('w-36 flex-shrink-0').props('outlined dense options-dense') \
             .tooltip('Filter the gallery by shot type')
            shot_filter.set_visibility(len(present_roles) > 1)

            # 2. Expanded text input filter matching the prompt editor search bar.
            # Debounced: the filter is cheap now, but one full gallery rebuild per
            # keystroke still queues up and fights the typing.
            search_filter_container[0] = ui.input(
                placeholder="Search chapters, quotes, prompts...",
                on_change=lambda e: schedule_search_refresh()
            ).classes('flex-1 bg-slate-50 rounded-lg').props('outlined dense clearable') \
             .on('keydown.enter', lambda: flush_search_refresh())

            # 3. Action Buttons (Regen / Stop / Stop After Book Toggle)
            ui.button(
                'Regen', 
                icon='refresh', 
                on_click=trigger_batch_restart
            ).classes('bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold px-3 py-1 h-8 rounded-lg')

            ui.button(
                'Stop',
                icon='stop',
                on_click=trigger_stop_rendering
            ).classes('bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold px-3 py-1 h-8 rounded-lg') \
             .bind_visibility_from(state, 'image_gen_active')

            # Conditional stop at the end of active book toggle
            def toggle_stop_after_book():
                state.stop_after_current_book = not getattr(state, 'stop_after_current_book', False)
                status_str = "ARMED" if state.stop_after_current_book else "DISABLED"
                ui.notify(f"Stop at end of book: {status_str}", type="info", timeout=1.5)
                update_stop_after_btn_style()

            btn_stop_after = ui.button(
                icon='auto_stories',
                on_click=toggle_stop_after_book,
                color=None
            ).props('unelevated dense').classes('h-8 px-2.5 rounded-lg transition-all duration-200')
            
            with btn_stop_after:
                btn_stop_tooltip = ui.tooltip('')

            def update_stop_after_btn_style():
                is_armed = getattr(state, 'stop_after_current_book', False)
                if is_armed:
                    # Inset / latched amber armed state with glowing border and inner shadow
                    btn_stop_after.classes(
                        replace='h-8 px-2.5 rounded-lg bg-amber-500 hover:bg-amber-600 text-white '
                                'shadow-inner ring-2 ring-amber-400 ring-offset-1 font-bold translate-y-[1px]'
                    )
                    btn_stop_tooltip.set_text('Stop at end of book: ARMED (Click to disarm)')
                else:
                    # Muted, unpressed flat state
                    btn_stop_after.classes(
                        replace='h-8 px-2.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-400 '
                                'hover:text-slate-600 border border-slate-200 shadow-none'
                    )
                    btn_stop_tooltip.set_text('Stop at end of book: OFF (Click to arm)')

            update_stop_after_btn_style()

        # Right-aligned toolbar group (Gallery/Prompt switcher)
        with ui.row().classes('items-center gap-0.5 bg-slate-100 p-0.5 rounded-lg border flex-nowrap'):
            btn_view_gallery = ui.button('Gallery', icon='photo_library').props('flat dense').classes('px-2.5 py-1 rounded-md text-xs font-semibold')
            btn_view_editor = ui.button('Prompt', icon='edit_note').props('flat dense').classes('px-2.5 py-1 rounded-md text-xs font-semibold')
            
            def update_view_mode_styles(mode: str):
                if mode == 'Gallery':
                    btn_view_gallery.classes(replace='px-2.5 py-1 rounded-md text-xs bg-white text-blue-600 shadow-sm font-black')
                    btn_view_editor.classes(replace='px-2.5 py-1 rounded-md text-xs text-slate-500 hover:text-slate-700 hover:bg-slate-200/50 font-medium')
                else:
                    btn_view_editor.classes(replace='px-2.5 py-1 rounded-md text-xs bg-white text-blue-600 shadow-sm font-black')
                    btn_view_gallery.classes(replace='px-2.5 py-1 rounded-md text-xs text-slate-500 hover:text-slate-700 hover:bg-slate-200/50 font-medium')

            def swap_view_mode(mode: str):
                state.book_view_mode = mode
                update_view_mode_styles(mode)
                
            btn_view_gallery.on('click', lambda: swap_view_mode('Gallery'))
            btn_view_editor.on('click', lambda: swap_view_mode('Editor'))
            update_view_mode_styles(state.book_view_mode)

    modal_img_el = None
    modal_placeholder = None
    modal_quote_el = None
    modal_prompt_input = None
    
    # Stateful reference containers to prevent initialization out-of-order NameErrors
    modal_context_html = [None]
    theater_prev_btn = [None]
    theater_next_btn = [None]
    theater_context_label = [None]
    
    theater_context_state = {"window_before": 400, "window_after": 400}

    def load_theater_context_adjusted(before_delta, after_delta):
        if before_delta is None and after_delta is None:
            theater_context_state["window_before"] = 400
            theater_context_state["window_after"] = 400
        else:
            if before_delta:
                theater_context_state["window_before"] = max(200, theater_context_state["window_before"] - before_delta)
            if after_delta:
                theater_context_state["window_after"] = max(200, theater_context_state["window_after"] + after_delta)
        update_theater_context_view()

    def update_theater_context_view():
        if not modal_context_html[0]:
            return
        current_scene = active_row_ref[0]
        if not current_scene:
            return
            
        wb = theater_context_state["window_before"]
        wa = theater_context_state["window_after"]
        
        res = find_quote_context_paginated(
            project_name, 
            book_name, 
            current_scene.get("quote", ""), 
            window_before=wb, 
            window_after=wa
        )
        
        html_content = res["html"]
        
        # Flatten pre-cached dictionary values to extract all lowercase names and aliases
        all_aliases = []
        for aliases in character_alias_map_modal.values():
            all_aliases.extend(aliases)
            
        # Deduplicate, prune empty values, and sort by length descending to match multi-word phrases first
        unique_aliases = sorted(list(set(a.strip() for a in all_aliases if a.strip())), key=len, reverse=True)
        
        if unique_aliases and html_content:
            # Build a single-pass alternation regex to prevent sequential replacement tag corruption (e.g. matching "mark")
            combined_pattern = r'\b(' + '|'.join(re.escape(alias) for alias in unique_aliases) + r')\b'
            combined_regex = re.compile(combined_pattern, flags=re.IGNORECASE)
            
            # Tokenize by HTML tags to isolate raw text and prevent highlights inside tag structures
            tokens = re.split(r'(<[^>]+>)', html_content)
            highlighted_tokens = []
            
            for token in tokens:
                if token.startswith('<') and token.endswith('>'):
                    # Keep existing HTML tags intact
                    highlighted_tokens.append(token)
                else:
                    text_segment = token
                    text_segment = combined_regex.sub(
                        lambda m: f'<mark class="bg-green-100 text-green-800 px-1 rounded font-semibold">{m.group(1)}</mark>', 
                        text_segment
                    )
                    highlighted_tokens.append(text_segment)
                    
            html_content = "".join(highlighted_tokens)

        modal_context_html[0].set_content(html_content)
        
        if theater_context_label[0]:
            theater_context_label[0].set_text(f"Context Span: showing {wb:,} chars before, {wa:,} chars after")
        if theater_prev_btn[0]:
            theater_prev_btn[0].set_visibility(res["has_prev"])
        if theater_next_btn[0]:
            theater_next_btn[0].set_visibility(res["has_next"])

    modal_title_el = None
    modal_subtitle_el = None
    
    modal_badge_missing = None
    modal_badge_approved = None
    modal_badge_review = None
    modal_role_select = None
    modal_regen_btn = [None]
    
    active_row_ref = [None]
    grid_card_references: Dict[tuple, Dict[str, Any]] = {}
    ringed_card = [None]  # the one grid card carrying the "this is open" ring

    # Pre-cache character vocabulary for autocomplete inside the book workspace tabs
    book_characters_modal: List[Character] = []
    character_alias_map_modal: Dict[int, List[str]] = {}
    with Session(engine) as session:
        book_characters_modal = session.exec(
            select(Character).where(Character.project_id == project.id)
        ).all()
        for char in book_characters_modal:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
            character_alias_map_modal[char.id] = [clean_character_key(a.alias) for a in aliases] + [clean_character_key(char.name)]


    # --- High Performance Theater Modal ---
    with ui.dialog() as theater_dialog:
        with ui.card().classes('w-full max-w-[95vw] lg:max-w-7xl h-[90vh] p-4 rounded-xl bg-white flex flex-col items-stretch overflow-hidden gap-0 outline-none focus:outline-none'):
            with ui.grid(columns='1fr 380px').classes('w-full h-full gap-4 items-stretch overflow-hidden min-h-0'):
                with ui.column().classes('w-full h-full justify-center min-h-0 relative'):
                    with ui.card().classes('w-full h-full border rounded-xl overflow-hidden shadow-sm flex items-center justify-center bg-slate-900 relative p-0 m-0 outline-none focus:outline-none'):
                        modal_placeholder = ui.column().classes('items-center justify-center text-slate-400 w-full h-full')
                        with modal_placeholder:
                            ui.icon('photo_library', size='lg').classes('mb-2 text-slate-500 animate-pulse')
                            ui.label("Awaiting ComfyUI Generation...").classes('text-xs font-semibold text-slate-400')
                            
                        modal_img_el = ui.image("").classes('w-full h-full bg-transparent').props('fit=contain')
                        
                        modal_badge_missing = ui.badge("Missing", color="red").classes('absolute top-4 left-4 font-bold text-xs')
                        modal_badge_approved = ui.badge("Approved", color="emerald").classes('absolute top-4 left-4 font-bold text-xs')
                        modal_badge_review = ui.badge("Needs Review", color="amber").classes('absolute top-4 left-4 font-bold text-xs')
                        
                with ui.column().classes('w-full h-full gap-4 overflow-y-auto min-h-0 flex-nowrap pr-1'):
                    with ui.row().classes('w-full items-center justify-between border-b pb-2 flex-shrink-0'):
                        with ui.column().classes('gap-0'):
                            modal_title_el = ui.label("").classes('text-base font-bold text-slate-800 leading-none')
                            modal_subtitle_el = ui.label("").classes('text-[11px] text-slate-400 mt-1')
                        with ui.row().classes('items-center gap-1'):
                            # Regen Images: the workspace toolbar's Regen, reachable
                            # without closing the modal. Deleted/unapproved scenes
                            # render in the background and pop into this view.
                            ui.button(
                                icon='refresh',
                                on_click=trigger_batch_restart
                            ).props('flat round dense').classes('text-slate-400 hover:text-blue-600').tooltip('Regen Images')

                            # Jump to Prompt Editor Button
                            ui.button(
                                icon='edit_note', 
                                on_click=lambda: [
                                    swap_view_mode('Editor'),
                                    getattr(state, 'select_prompt_editor_scene_fn', lambda ch, sc: None)(
                                        state.book_active_chapter, state.book_active_scene
                                    ),
                                    theater_dialog.close()
                                ]
                            ).props('flat round dense').classes('text-slate-400 hover:text-blue-600').tooltip('Edit Prompt in Workspace')

                            # Quick-Curate Characters Dialog Button
                            ui.button(
                                icon='people', 
                                on_click=open_quick_characters_dialog
                            ).props('flat round dense').classes('text-slate-400 hover:text-blue-600').tooltip('Project Characters Manager')

                            with ui.button(icon='help_outline').props('flat round dense').classes('text-slate-400'):
                                with ui.tooltip().classes('bg-slate-800 text-white text-xs p-3 rounded-lg gap-1 flex flex-col shadow-lg'):
                                    ui.label('Keyboard Shortcuts').classes('font-bold border-b pb-1 text-blue-400')
                                    ui.label(f'[{state.key_approve.upper()}] Approve Scene')
                                    ui.label(f'[{state.key_delete.upper()}] Delete Image')
                                    ui.label(f'[{state.key_next.upper()}] Next Scene')
                                    ui.label(f'[{state.key_prev.upper()}] Prev Scene')
                            ui.button(icon='close', on_click=theater_dialog.close).props('flat round dense').classes('text-slate-400')
                    
                    with ui.column().classes('w-full gap-2 bg-slate-100 p-3 rounded-lg border border-dashed flex-shrink-0'):
                        with ui.row().classes('w-full gap-2 items-center justify-between'):
                            ui.button('Prev', icon='chevron_left', on_click=prev_scene).props('flat dense').classes('text-xs font-bold text-slate-600 flex-1 py-1.5 bg-white border rounded outline-none focus:outline-none')
                            ui.button('Next', icon='chevron_right', on_click=next_scene).props('flat dense').classes('text-xs font-bold text-slate-600 flex-1 py-1.5 bg-white border rounded outline-none focus:outline-none')
                        with ui.row().classes('w-full gap-2'):
                            ui.button('Delete Image', icon='delete', on_click=delete_current).classes('bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold flex-1 py-2 outline-none focus:outline-none')
                            ui.button('Approve', icon='check', on_click=approve_current).classes('bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold flex-1 py-2 outline-none focus:outline-none')

                    # Shot-role switcher + one-shot prompt regeneration. The
                    # regeneration frees ComfyUI's VRAM before calling the LLM.
                    with ui.row().classes('w-full gap-2 items-center flex-shrink-0'):
                        ui.label('Shot').classes('text-[9px] font-black text-slate-400 uppercase tracking-wider')
                        modal_role_select = ui.select(
                            options=[SHOT_ROLE_LABELS[r] for r in SHOT_ROLE_ORDER],
                            value=SHOT_ROLE_LABELS["character"],
                            on_change=lambda e: apply_shot_role(e.value),
                        ).classes('flex-1').props('outlined dense options-dense').tooltip(
                            'Shot type recorded for this scene; changing it re-styles the next render.'
                        )
                        modal_regen_btn[0] = ui.button(
                            'New Prompt', icon='auto_awesome', on_click=request_new_prompt
                        ).classes('bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold px-3 py-2') \
                         .tooltip('Ask the LLM for a fresh prompt using the selected shot type')

                    scene_chips_container = ui.row().classes('w-full flex-shrink-0')

                    with ui.column().classes('w-full gap-2 bg-slate-50 p-3 rounded border border-dashed flex-shrink-0'):
                        ui.label("Target Narration Quote").classes('text-[9px] font-black text-slate-400 uppercase tracking-wider')
                        modal_quote_el = ui.label("").classes('text-xs italic text-slate-700 leading-relaxed font-serif')
                        
                    with ui.element('div').classes('relative w-full flex-shrink-0'):
                        modal_prompt_input = ui.textarea(
                            label="Style-Ready Visual Prompt"
                        ).classes('w-full h-32 text-xs leading-relaxed flex-shrink-0 theater-prompt-editor').props('outlined')
                        
                        modal_prompt_input.on('blur', lambda: update_prompt_text(modal_prompt_input.value, active_row_ref[0]))
                        
                        PromptAutocompleteManager(
                            textarea=modal_prompt_input,
                            project_id=project.id,
                            on_change_callback=lambda val: render_scene_character_chips(val)
                        )
                    
                    with ui.expansion('Narrative Context (transcript.txt)').classes('w-full border rounded bg-slate-50 text-xs flex-shrink-0') as context_expansion:
                        with ui.column().classes('w-full gap-1.5 p-2 bg-slate-100 border-b flex-nowrap'):
                            with ui.row().classes('w-full justify-between items-center gap-1 flex-nowrap'):
                                theater_prev_btn[0] = ui.button(
                                    '← Load Back (+1,000)',
                                    on_click=lambda: load_theater_context_adjusted(-1000, 0)
                                ).props('dense flat').classes('text-[10px] font-bold text-blue-600')
                                
                                theater_reset_btn = ui.button(
                                    'Reset',
                                    on_click=lambda: load_theater_context_adjusted(None, None)
                                ).props('dense flat').classes('text-[10px] text-slate-500 font-bold')
                                
                                theater_next_btn[0] = ui.button(
                                    'Load Ahead (+1,000) →',
                                    on_click=lambda: load_theater_context_adjusted(0, 1000)
                                ).props('dense flat').classes('text-[10px] font-bold text-blue-600')
                                
                            theater_context_label[0] = ui.label("").classes('text-[9px] text-slate-400 self-center font-semibold')
                            
                        modal_context_html[0] = ui.html("").classes('p-3 leading-relaxed text-slate-700 bg-white font-serif')

    # Keyboard shortcut listener
    def handle_key(e):
        if not theater_dialog.value:
            return
        if e.action.keydown and not e.action.repeat:
            k = e.key.name.lower()
            if k == state.key_approve:
                approve_current()
            elif k == state.key_delete:
                delete_current()
            elif k == state.key_next:
                next_scene()
            elif k == state.key_prev:
                prev_scene()

    state.book_keyboard = ui.keyboard(on_key=handle_key)

    # --- Gallery Grid View Rendering (Dynamic In-Place Appending) ---
    
    grid_el_ref = [None]
    load_more_spinner_ref = [None]
    rendered_count_ref = [0]

    def render_batch(batch_list: list):
        """Builds and mounts card elements dynamically into the active grid context container."""
        for item in batch_list:
            is_approved = item.get("approved", "False").strip().lower() == "true"
            
            try:
                ch = int(float(item.get("chapter", "1")))
                sc = int(float(item.get("shot", "1")))
            except ValueError:
                ch, sc = 1, 1
                
            img_url = images_cache.get((ch, sc))
            
            if not img_url:
                border_style = "border-red-300 bg-red-50/10"
                status_color = "bg-red-500"
            elif is_approved:
                border_style = "border-emerald-300 bg-emerald-50/10"
                status_color = "bg-emerald-500"
            else:
                border_style = "border-amber-300 bg-amber-50/10"
                status_color = "bg-amber-500"
                
            with ui.card().classes(f'border rounded-lg shadow-sm p-2 cursor-pointer hover:shadow-md transition-all {border_style} outline-none focus:outline-none') \
                    .on('click', lambda _, ch_val=ch, sc_val=sc: launch_theater_from_coordinate(ch_val, sc_val)) as card_el:
                
                img_el = ui.image(img_url or "").classes('w-full aspect-square rounded object-cover border')
                img_el.visible = bool(img_url)
                
                placeholder_el = ui.column().classes('w-full aspect-square items-center justify-center bg-slate-100 rounded border border-dashed text-slate-400')
                with placeholder_el:
                    ui.icon('photo_library', size='sm')
                    ui.label('Missing').classes('text-[9px]')
                placeholder_el.visible = not img_url

                tags = extract_prompt_character_tags(item.get("prompt", ""))
                seen_tags = set()
                deduped_tags = []
                for t in tags:
                    key = clean_character_key(t)
                    if key not in seen_tags:
                        seen_tags.add(key)
                        deduped_tags.append(t)

                role = style_library.normalize_role(item.get("shot_role"))
                with ui.row().classes('w-full gap-1 items-center mt-1.5 px-1 flex-wrap'):
                    role_badge = ui.badge(
                        SHOT_ROLE_LABELS[role], color=SHOT_ROLE_BADGE_COLORS[role]
                    ).classes('text-[8px] font-bold py-0 px-1 rounded-sm flex-shrink-0')
                    with role_badge:
                        ui.tooltip(f"Shot type: {SHOT_ROLE_LABELS[role]}")

                    for t in deduped_tags[:4]:
                        disp = t
                        if len(disp) > 8:
                            disp = disp[:7] + "…"
                        badge_el = ui.badge(disp, color='slate').classes('text-[8px] font-medium py-0 px-1 rounded-sm flex-shrink-0')
                        with badge_el:
                            ui.tooltip(t).classes('bg-slate-800 text-white text-[9px] p-1.5 rounded')
                    if len(deduped_tags) > 4:
                        ui.badge(f"+{len(deduped_tags)-4}", color='blue').classes('text-[8px] font-bold py-0 px-1 rounded-sm flex-shrink-0')
                        
                with ui.row().classes('w-full justify-between items-center mt-1 px-1'):
                    ui.label(f"Ch {item.get('chapter')}, Sc {item.get('shot')}").classes('text-[10px] font-bold text-slate-700')
                    dot_el = ui.element('div').classes(f'w-2 h-2 rounded-full {status_color}')
                    
            grid_card_references[(ch, sc)] = {
                "card": card_el,
                "image": img_el,
                "placeholder": placeholder_el,
                "dot": dot_el,
                "item": item
            }

    # --- Parent Permanent Workspace Containers ---
    editor_view_container = ui.column().classes('w-full')
    editor_view_container.bind_visibility_from(state, 'book_view_mode', value='Editor')

    @ui.refreshable
    def render_content():
        filtered = get_filtered_prompts()
        
        with ui.column().classes('w-full').bind_visibility_from(state, 'book_view_mode', value='Gallery'):
            if not filtered:
                with ui.column().classes('w-full items-center justify-center p-12 text-slate-400 border border-dashed rounded-xl bg-slate-50 outline-none focus:outline-none'):
                    ui.icon('info', size='lg', color='slate-300')
                    ui.label("No scenes match your active filter.").classes('text-sm text-center font-semibold')
                return
                
            grid_card_references.clear()
            ringed_card[0] = None  # the ringed card is about to be destroyed
            rendered_count_ref[0] = min(24, len(filtered))
            
            grid_el = ui.grid(columns='repeat(auto-fill, minmax(180px, 1fr))').classes('w-full gap-4')
            grid_el_ref[0] = grid_el
            
            with grid_el:
                render_batch(filtered[:rendered_count_ref[0]])
                
            load_more_spinner_ref[0] = ui.row().classes('w-full justify-center p-4')
            with load_more_spinner_ref[0]:
                ui.spinner(size='md', color='blue')
                ui.label('Scrolling to load more scenes...').classes('text-xs text-slate-400 font-medium')
            
            load_more_spinner_ref[0].visible = len(filtered) > rendered_count_ref[0]

    # Initial paint call
    render_content()

    with editor_view_container:
        render_prompt_editor_view(
            project_id=project.id,
            book_id=book_id,
            images_cache=images_cache,
            prompts_list=prompts,
            on_refresh_grid=lambda: [render_content.refresh()],
            open_character_edit_dialog_fn=open_character_edit_dialog,
            open_theater_modal_fn=launch_theater_from_coordinate,
            open_quick_characters_dialog_fn=open_quick_characters_dialog
        )

    # --- Viewport-Aware Scroll Listener ---
    async def check_scroll():
        if theater_dialog.value or state.book_view_mode != 'Gallery':
            return
        try:
            is_near_bottom = await ui.run_javascript(
                'window.pageYOffset >= document.body.offsetHeight - 1.5 * window.innerHeight'
            )
            if is_near_bottom:
                filtered = get_filtered_prompts()
                current_count = rendered_count_ref[0]
                
                if current_count < len(filtered):
                    next_count = min(current_count + 24, len(filtered))
                    next_batch = filtered[current_count:next_count]
                    rendered_count_ref[0] = next_count
                    
                    if grid_el_ref[0]:
                        with grid_el_ref[0]:
                            render_batch(next_batch)
                            
                        if load_more_spinner_ref[0]:
                            load_more_spinner_ref[0].visible = len(filtered) > rendered_count_ref[0]
        except Exception:
            pass

    state.book_scroll_timer = ui.timer(0.3, check_scroll)

    # --- Real-Time Background Image Pop-in Timer ---
    last_folder_signature = [None]
    gallery_stale = [False]

    def repaint_gallery():
        if filter_mode.value == "All":
            update_grid_views_in_place()
        else:
            render_content.refresh()

    async def check_for_image_updates():
        if state.active_book_id is None:
            return
            
        img_dir = Path(f"./output/{project_name}/{book_name}/images")
        parent_dir = Path(f"./output/{project_name}/{book_name}")

        def folder_signature():
            """(entry count, mtime) per folder: cheap, and a rename or a delete
            plus a create between two ticks still shows up."""
            signature = []
            for d in (img_dir, parent_dir):
                if not d.exists():
                    signature.append((0, 0))
                    continue
                try:
                    signature.append((len(os.listdir(d)), os.stat(d).st_mtime_ns))
                except Exception:
                    signature.append((0, 0))
            return tuple(signature)
            
        signature = await asyncio.to_thread(folder_signature)

        if signature != last_folder_signature[0]:
            last_folder_signature[0] = signature

            # Update in place: the prompt editor holds a reference to this dict.
            fresh = await asyncio.to_thread(get_book_images_cache, project_name, book_name)
            images_cache.clear()
            images_cache.update(fresh)

            if theater_dialog.value:
                # The gallery is behind a modal, so rebuilding it now is wasted
                # work - and it used to throw away every card the user had
                # scrolled open. Repaint it once, after the modal closes.
                refresh_theater_queue()
                update_active_scene_ui()
                gallery_stale[0] = True
                return

            repaint_gallery()

        elif gallery_stale[0] and not theater_dialog.value:
            gallery_stale[0] = False
            repaint_gallery()

    state.book_update_timer = ui.timer(3.0, check_for_image_updates)