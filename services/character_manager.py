import re
import json
import csv
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Callable
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Project, Book, Character, CharacterAlias, CharacterBookLink, CharacterStateModifier, CharacterTimelineEvent
from services.prompt_engine import smart_chunk_text, get_llm_response
from services import compat

# Typographic single quotes look identical to a reader but are different bytes
# to every exact-match query in this app. Canonicalise them at the boundary:
# names, aliases, prompt tags, and search all use the ASCII apostrophe.
_APOSTROPHE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u2032": "'",
    "\u02bc": "'", "\u02bb": "'", "\uff07": "'", "\u00b4": "'", "`": "'",
})


def normalize_character_text(text: str) -> str:
    """Canonical single quotes and whitespace for a name, alias, or tag."""
    if not text:
        return ""
    cleaned = str(text).translate(_APOSTROPHE_MAP).replace("\u00a0", " ")
    return " ".join(cleaned.split())


def normalize_character_key(text: str) -> str:
    """The comparison key for a name or alias: canonical quotes, lowercased."""
    return normalize_character_text(text).lower()


_WRAPPING_QUOTES = "\"'“”‘’"


def clean_character_tag(text: str) -> str:
    """Normalise a prompt tag or imported name, removing wrapping quote marks.

    ``["Ba'alzamon"]`` and ``'Ba’alzamon'`` both become ``Ba'alzamon``. Internal
    apostrophes (``al'Vere``) are preserved; only a matching pair of outer quotes
    is removed, repeatedly, so ``"'Name'"`` also cleans up.
    """
    cleaned = normalize_character_text(text)
    while (len(cleaned) >= 2 and cleaned[0] in _WRAPPING_QUOTES
           and cleaned[-1] in _WRAPPING_QUOTES):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def clean_character_key(text: str) -> str:
    return clean_character_tag(text).lower()

def get_characters_json_path(project_id: int, session: Optional[Session] = None) -> Optional[Path]:
    """Retrieves the file-as-source-of-truth characters.json target path."""
    should_close = False
    if session is None:
        session = Session(engine)
        should_close = True
    try:
        project = session.get(Project, project_id)
        if not project:
            return None
        # Crucial: Passed the active session into get_setting to prevent secondary connection deadlocks
        base_output_dir = Path(get_setting("output_dir", "./output", session)).resolve()
        return base_output_dir / project.name / "characters.json"
    finally:
        if should_close:
            session.close()


def ensure_book_orders(project_id: int):
    """Fill in a book_order for books that have none, and never renumber.

    Renumbering alphabetically used to run on every characters-tab open, which
    silently overwrote a manually chosen reading order. Order is the axis that
    drives cross-book description inheritance, so only a missing value is
    assigned here; existing values are the user's (or the importer's) decision.
    """
    with Session(engine) as session:
        books = session.exec(
            select(Book).where(Book.project_id == project_id).order_by(Book.name)
        ).all()

        known = [b.book_order for b in books if b.book_order is not None]
        next_order = (max(known) + 1) if known else 0
        changed = False
        for book in books:
            if book.book_order is None:
                book.book_order = next_order
                next_order += 1
                session.add(book)
                changed = True
        if changed:
            session.commit()


def compile_character_visual_prompt(obj) -> str:
    """Assembles the inner description (no character name) from an event's fields.

    Accepts either a CharacterTimelineEvent or a legacy Character object. A
    new-field event uses the extracted trait set; a legacy event keeps the old
    four-field comma list, so a project profiled by an older build produces the
    same text it always did.
    """
    if has_new_traits(obj):
        inner = compile_event_fields(obj, {}, True)
        if inner:
            return inner

    pieces = []
    
    if getattr(obj, "demographics", None) and str(obj.demographics).strip():
        pieces.append(obj.demographics.strip())
    if getattr(obj, "hair_and_face", None) and str(obj.hair_and_face).strip():
        pieces.append(obj.hair_and_face.strip())
    if getattr(obj, "physical_build", None) and str(obj.physical_build).strip():
        pieces.append(obj.physical_build.strip())
    if getattr(obj, "distinguishing_marks", None) and str(obj.distinguishing_marks).strip():
        pieces.append(obj.distinguishing_marks.strip())

    cleaned_pieces = []
    seen = set()
    for p in pieces:
        p_clean = p.strip()
        if p_clean and p_clean.lower() not in seen:
            cleaned_pieces.append(p_clean)
            seen.add(p_clean.lower())

    if not cleaned_pieces:
        name = getattr(obj, "name", "person")
        return f"a person named {name}"
        
    return ", ".join(cleaned_pieces)


def save_project_characters_to_json(project_id: int):
    """
    Serializes all project characters, aliases, book links, and timeline override events to characters.json.
    Ensures that manual edits, book links, and LLM descriptions are always safely preserved on disk.
    """
    json_path = get_characters_json_path(project_id)
    if not json_path:
        return

    with Session(engine) as session:
        characters = session.exec(select(Character).where(Character.project_id == project_id)).all()
        
        serialized_data = []
        for char in characters:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
            book_links = session.exec(select(CharacterBookLink).where(CharacterBookLink.character_id == char.id)).all()
            
            # Resolve book names from links
            linked_book_names = []
            for link in book_links:
                book = session.get(Book, link.book_id)
                if book and book.name not in linked_book_names:
                    linked_book_names.append(book.name)

            timeline_events = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == char.id)
                .order_by(CharacterTimelineEvent.id)
            ).all()
            
            timeline_serialized = []
            for ev in timeline_events:
                book_name = None
                if ev.book_id is not None:
                    b = session.get(Book, ev.book_id)
                    if b:
                        book_name = b.name
                
                timeline_serialized.append({
                    "book_name": book_name,
                    "chapter_num": ev.chapter_num,
                    "shot_num": ev.shot_num,
                    "label": ev.label,
                    "visual_description": ev.visual_description,
                    "source": ev.source,
                    # The extracted trait set. `profile` is the legacy quartet,
                    # kept so an older build can still read the file and so a
                    # database rebuilt from it loses nothing.
                    "traits": {field: getattr(ev, field, None)
                               for field in NEW_TRAIT_FIELDS},
                    "profile": {
                        "demographics": ev.demographics,
                        "physical_build": ev.physical_build,
                        "hair_and_face": ev.hair_and_face,
                        "distinguishing_marks": ev.distinguishing_marks
                    }
                })

            char_entry = {
                "name": char.name,
                "locked": char.locked,
                "origin": getattr(char, "origin", "Prompt Scan"),
                "merge_checked": getattr(char, "merge_checked", False),
                "hit_count": getattr(char, "hit_count", 0) or 0,
                "pinned_traits": getattr(char, "pinned_traits", None),
                "aliases": [alias.alias for alias in aliases],
                "books": linked_book_names,
                "timeline": timeline_serialized
            }
            serialized_data.append(char_entry)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=2, ensure_ascii=False)


def robust_json_load(json_str: str) -> Any:
    """
    Attempts to parse JSON, with fallbacks to repair trailing commas and unclosed brackets/braces.
    Useful for handling legacy files or copy-paste truncated outputs cleanly.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    cleaned = json_str.strip()
    
    # Remove trailing commas before closing brackets or braces
    cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
    
    if cleaned.endswith(','):
        cleaned = cleaned[:-1].strip()
        
    open_brackets = cleaned.count('[') - cleaned.count(']')
    open_braces = cleaned.count('{') - cleaned.count('}')
    
    if open_braces > 0:
        cleaned += '}' * open_braces
    if open_brackets > 0:
        cleaned += ']' * open_brackets
        
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-ditch: Scan and extract any complete flat JSON objects {...} in sequence
        objects = []
        brace_level = 0
        start_idx = -1
        for i, char in enumerate(cleaned):
            if char == '{':
                if brace_level == 0:
                    start_idx = i
                brace_level += 1
            elif char == '}':
                brace_level -= 1
                if brace_level == 0 and start_idx != -1:
                    obj_str = cleaned[start_idx:i+1]
                    try:
                        objects.append(json.loads(obj_str))
                    except Exception:
                        pass
        if objects:
            return objects
        raise


def sync_project_characters_from_json(project_id: int, session: Optional[Session] = None):
    """
    Rebuilds the SQLModel character entries, book links, and timeline overrides from characters.json if the database was wiped.
    Maintains our strict File-as-Source-of-Truth database indexing principles.
    """
    json_path = get_characters_json_path(project_id, session)
    if not json_path or not json_path.exists():
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
        data = robust_json_load(raw_content)
    except Exception as e:
        print(f"[Characters] Failed to parse {json_path}: {str(e)}")
        return

    should_close = False
    if session is None:
        session = Session(engine)
        should_close = True

    try:
        # Clear out existing SQLModel character caches and relations for this project to perform a clean sync
        old_chars = session.exec(select(Character).where(Character.project_id == project_id)).all()
        for oc in old_chars:
            aliases_to_del = session.exec(
                select(CharacterAlias).where(CharacterAlias.character_id == oc.id)
            ).all()
            for a in aliases_to_del:
                session.delete(a)
            
            links_to_del = session.exec(
                select(CharacterBookLink).where(CharacterBookLink.character_id == oc.id)
            ).all()
            for lk in links_to_del:
                session.delete(lk)

            evs_to_del = session.exec(
                select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id == oc.id)
            ).all()
            for ev in evs_to_del:
                session.delete(ev)

            session.delete(oc)
        session.flush()

        # Build book lookup cache for this project
        project_books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        book_name_to_id = {b.name: b.id for b in project_books}

        # Reconstruct tables from file mapping
        for char_data in data:
            new_char = Character(
                project_id=project_id,
                name=char_data["name"],
                locked=char_data.get("locked", False),
                origin=char_data.get("origin", "Prompt Scan"),
                merge_checked=char_data.get("merge_checked", False),
                hit_count=char_data.get("hit_count", 0) or 0,
                pinned_traits=char_data.get("pinned_traits"),
            )
            session.add(new_char)
            session.flush()  # flush to acquire character ID

            # Attach extracted aliases
            for alias_text in char_data.get("aliases", []):
                new_alias = CharacterAlias(character_id=new_char.id, alias=alias_text)
                session.add(new_alias)

            # Attach book links
            for b_name in char_data.get("books", []):
                b_id = book_name_to_id.get(b_name)
                if b_id:
                    new_link = CharacterBookLink(character_id=new_char.id, book_id=b_id)
                    session.add(new_link)

            # Rebuild Timeline Events
            timeline_list = char_data.get("timeline", [])
            if not timeline_list:
                # Backward compatibility fallback
                legacy_profile = char_data.get("profile") or {}
                base_traits = char_data.get("traits") or {}
                base_ev = CharacterTimelineEvent(
                    character_id=new_char.id,
                    book_id=None,
                    chapter_num=0,
                    shot_num=0,
                    label="Base State",
                    demographics=legacy_profile.get("demographics"),
                    physical_build=legacy_profile.get("physical_build"),
                    hair_and_face=legacy_profile.get("hair_and_face"),
                    distinguishing_marks=legacy_profile.get("distinguishing_marks"),
                    **{field: (base_traits.get(field)
                               if base_traits.get(field) is not None
                               else legacy_profile.get(field))
                       for field in NEW_TRAIT_FIELDS},
                    visual_description=char_data.get("visual_description")
                )
                session.add(base_ev)
            else:
                for ev_data in timeline_list:
                    b_name = ev_data.get("book_name")
                    b_id = book_name_to_id.get(b_name) if b_name else None
                    
                    prof = ev_data.get("profile", {})
                    traits = ev_data.get("traits", {}) or {}
                    new_ev = CharacterTimelineEvent(
                        character_id=new_char.id,
                        book_id=b_id,
                        chapter_num=ev_data.get("chapter_num", 0),
                        # characters.json is the declared source of truth and may
                        # predate the scene->shot rename, so accept both spellings.
                        # Reading only the new key would silently load every
                        # existing timeline event as shot 0.
                        shot_num=compat.get_shot(ev_data, 0),
                        label=ev_data.get("label", "Base State" if b_id is None else "Override State"),
                        demographics=prof.get("demographics"),
                        physical_build=prof.get("physical_build"),
                        hair_and_face=prof.get("hair_and_face"),
                        distinguishing_marks=prof.get("distinguishing_marks"),
                        **{field: (traits.get(field)
                                   if traits.get(field) is not None
                                   else prof.get(field))
                           for field in NEW_TRAIT_FIELDS},
                        visual_description=ev_data.get("visual_description"),
                        source=ev_data.get("source")
                    )
                    session.add(new_ev)
                    
        session.flush()
    finally:
        if should_close:
            session.commit()
            session.close()


def get_character_mention_chunks(
    project_id: int,
    character_id: int,
    book_id: Optional[int] = None,
    chunk_size_words: int = 150
) -> List[Dict[str, Any]]:
    """
    Retrieves consecutive, chronological snippet windows of transcript.txt centered on character aliases.
    Maintains strictly sequential order (earliest first) to capture introductions and development in order.
    """
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return []
            
        aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == character_id)).all()
        alias_texts = {a.alias.lower().strip() for a in aliases}
        if not alias_texts:
            return []

        if book_id:
            books = [session.get(Book, book_id)]
        else:
            book_mentions = get_character_book_mentions(project_id, character_id)
            all_books = session.exec(
                select(Book).where(Book.project_id == project_id).order_by(Book.id)
            ).all()
            
            if book_mentions:
                books = [b for b in all_books if b.name in book_mentions]
            else:
                books = all_books

    mention_chunks = []
    
    for book in books:
        if not book:
            continue
        transcript_path = base_output_dir / project.name / book.name / "transcript.txt"
        if not transcript_path.exists():
            continue

        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"[Profiler] Error reading transcript for {book.name}: {str(e)}")
            continue

        cleaned_text = content.replace("==CHAPTER==", "\n\n--- CHAPTER BREAK ---\n\n").strip()
        
        match_positions = []
        for alias in alias_texts:
            pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            for m in pattern.finditer(cleaned_text):
                match_positions.append(m.start())
                
        if not match_positions:
            continue

        sampled_char_offsets = sorted(list(set(match_positions)))[:50]

        words = re.findall(r'\S+', cleaned_text)
        if not words:
            continue

        half_window = chunk_size_words // 2
        for start_pos in sampled_char_offsets:
            words_before_count = len(re.findall(r'\S+', cleaned_text[:start_pos]))
            
            start_idx = max(0, words_before_count - half_window)
            end_idx = min(len(words), words_before_count + half_window)
            
            snippet_words = words[start_idx:end_idx]
            snippet_text = " ".join(snippet_words)
            
            snippet_lower = snippet_text.lower()
            mentions_count = sum(len(re.findall(re.escape(alias), snippet_lower)) for alias in alias_texts)

            mention_chunks.append({
                "book_id": book.id,
                "book_name": book.name,
                "chunk_index": start_idx,
                "text": snippet_text,
                "mentions_count": mentions_count
            })

    return mention_chunks


def get_character_book_mentions(project_id: int, character_id: int) -> Dict[str, int]:
    """
    Returns a mapping of Book Name -> Mention Count for all aliases belonging
    to the given character across the project using cached frequency maps.
    """
    from services.character_organizer import get_book_character_frequency_map
    book_mentions = {}
    
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return {}
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == character_id)).all()
        alias_texts = {a.alias.lower().strip() for a in aliases}
        char = session.get(Character, character_id)
        if char:
            alias_texts.add(char.name.lower().strip())

        if not alias_texts:
            return {}

        for book in books:
            freq_map = get_book_character_frequency_map(project.name, book.name, session)
            total = sum(freq_map.get(tag, 0) for tag in alias_texts)
            if total > 0:
                book_mentions[book.name] = total
                
    return book_mentions


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Bulletproof helper to extract and parse a valid JSON block out of raw LLM output,
    ignoring background commentary, descriptions, or markdown fence syntax.
    """
    markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        try:
            return json.loads(markdown_match.group(1))
        except json.JSONDecodeError:
            pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace+1])
        except json.JSONDecodeError:
            pass

    return {}

def get_speculative_character_template() -> str:
    """Returns a creative system prompt instructing the LLM to act as a casting director when physical details are missing."""
    return (
        "You are an expert creative casting director and character concept designer. The target character, "
        "{character_name} (aka: {aliases}), lacks complete physical descriptions in the text.\n\n"
        "### YOUR TASK ###\n"
        "Analyze the provided passage's context—their name, honorifics (Mr., Mrs., Miss, Dr., Sir), dialog tone, age cues, "
        "and social role—to deduce their gender, approximate age, and stylistic vibe. Then, cast them with a "
        "highly cohesive, fitting, and visually distinct physical appearance appropriate for their character role.\n\n"
        "Every character in this series must look highly distinct from one another. Do NOT use lazy filler clichés.\n\n"
        "### TARGET SENTENCE SCHEMA ###\n"
        "We inject your output into this exact template:\n"
        "\"{character_name} (a {{demographics}}, {{hair_and_face}}, who is {{physical_build}}, and {{distinguishing_marks}})\"\n\n"
        "Your JSON values must be short, lowercase grammatical fragments:\n"
        "- 'demographics': Noun phrase defining their age and gender (do NOT use articles). E.g., 'elderly woman', 'young man'.\n"
        "- 'hair_and_face': Physical descriptors of hair style, hair color, or facial structure starting with 'with'. Make it highly cohesive.\n"
        "- 'physical_build': Physical height, stature, and body build.\n"
        "- 'distinguishing_marks': Do NOT invent accessories, or jewelry.\n\n"
        "### CRITICAL RESTRICTIONS ###\n"
        "1. NO CLOTHING: Do not specify suits, jackets, raincoats, uniform details, or hats. Focus strictly on their body, face, and permanent physical features.\n"
        "2. NO GAZE/LOOK/EXPRESSION: Focus on concrete, paintable physical features only.\n"
        "3. Output MUST be a single, valid JSON block. No commentary.\n\n"
        "### CURRENT PROFILE STATE ###\n"
        "Currently recorded:\n"
        "{known_traits}\n"
        "Unknown (needs data):\n"
        "{unknown_traits}\n\n"
        "### JSON TARGET SCHEMA ###\n"
        "{{\n"
        "  \"demographics\": \"string\",\n"
        "  \"hair_and_face\": \"string\",\n"
        "  \"physical_build\": \"string\",\n"
        "  \"distinguishing_marks\": \"string\" | null\n"
        "}}\n"
    )


def get_default_character_template() -> str:
    """Returns a strict, objective prompt for extracting written physical details from text."""
    return (
        "You are a strict, objective AI character profiler. Extract physical features for {character_name} "
        "(aka: {aliases}) from the provided book passage.\n\n"
        "### TARGET SENTENCE SCHEMA ###\n"
        "We inject your output into this exact template:\n"
        "\"{character_name} (a {{demographics}}, {{hair_and_face}}, who is {{physical_build}}, and {{distinguishing_marks}})\"\n\n"
        "Your JSON values must be short, lowercase grammatical fragments:\n"
        "- 'demographics': Noun phrase of age, race, gender (NO articles).\n"
        "- 'hair_and_face': Prepositional phrase starting with 'with' describing hair, eyes, or facial features.\n"
        "- 'physical_build': Height, posture, and build.\n"
        "- 'distinguishing_marks': Permanent details only (tattoos, scars, glasses). Otherwise leave null.\n\n"
        "### CRITICAL RESTRICTIONS (STRICTLY ENFORCED) ###\n"
        "1. NO CLOTHING: Do not extract suits, jackets, raincoats, hats, or attire.\n"
        "2. NO TRANSIENT GESTURES/EXPRESSIONS: Ignore voice, sounds, smiles, frowns, glances, or momentary physical movements.\n"
        "3. ENTITY SHIELD: Only extract traits if they explicitly describe {character_name}.\n"
        "4. ONLY PAINTABLE VISUAL DETAILS: Your extractions must describe direct, concrete physical colors, textures, shapes, and tangible sizes. If an artist cannot physically paint it, it is strictly forbidden.\n"
        "5. NO BLUEPRINT CLICHÉS: Do NOT invent, assume, or default to generic descriptors.\n"
        "6. Output MUST be a single, valid JSON block. No commentary.\n\n"
        "### CURRENT PROFILE STATE ###\n"
        "Currently recorded:\n"
        "{known_traits}\n"
        "Unknown (needs data):\n"
        "{unknown_traits}\n\n"
        "### JSON TARGET SCHEMA ###\n"
        "{{\n"
        "  \"demographics\": \"string\" | null,\n"
        "  \"hair_and_face\": \"string\" | null,\n"
        "  \"physical_build\": \"string\" | null,\n"
        "  \"distinguishing_marks\": \"string\" | null\n"
        "}}\n"
    )


def is_valid_permanent_trait(key: str, new_val: str, old_val: Optional[str] = None) -> bool:
    """
    Determines if a newly extracted trait value is a valid permanent visual descriptor.
    """
    val = new_val.lower().strip()
    if not val or val == "null" or val == "none":
        return False
        
    banned_terms = [
        "voice", "sound", "accent", "tone", "shout", "whisper", "screamed", "spoken", "spoke", "screaming",
        "smile", "grin", "frown", "scowl", "pout", "smirk", "laugh", "giggle", "chuckle",
        "twitching", "winking", "blinking", "crying", "tears", "shivering", "shivered", "recoiled",
        "raised eyebrow", "raised eyebrows", "furrowed", "dropped jaw", "parted lip", "parted lips",
        "gritting teeth", "gnashing", "biting lip", "chewing lip",
        "clap", "clapped", "clapping", "slap", "slapped", "slapping", "grab", "grabbed", "grabbing", 
        "hold", "held", "holding", "press", "pressed", "pressing", "touch", "touched", "touching",
        "unknown", "not specified", "unspecified", "unmentioned", "not mentioned", "not described"
    ]
    
    for term in banned_terms:
        if re.search(rf"\b{term}", val):
            print(f"[Profiler Filter] Discarding transient/action/auditory term '{term}' in: '{new_val}'")
            return False
            
    generic_words = ["tall", "short", "thin", "fat", "man", "woman", "boy", "girl", "hair", "face"]
    if val in generic_words and old_val and len(old_val.strip()) > 15:
        print(f"[Profiler Filter] Discarding generic single-word update '{new_val}' over descriptive: '{old_val}'")
        return False
        
    return True


def score_chunk_visual_relevance(text: str) -> int:
    """Computes a heuristic score for how likely a text chunk is to contain physical descriptions."""
    text_lower = text.lower()
    
    visual_keywords = [
        "hair", "eyes", "tall", "short", "build", "face", "handsome", "pretty", "slender", 
        "stocky", "athletic", "physique", "glasses", "beard", "mustache", "scar", "tattoo", 
        "height", "weight", "slim", "skin", "complexion", "features", "jaw", "shoulders", 
        "looking", "looked", "blond", "blonde", "brunette", "brown", "black", "blue", "green", 
        "gray", "grey", "bald", "shaven", "he was a", "she was a", "years old",
        "trim", "slender", "beautiful", "gorgeous", "shapely", "naked", "shower", "back", 
        "buttocks", "chest", "waist", "figure", "attractive", "stature"
    ]
    
    score = 0
    for kw in visual_keywords:
        if kw in text_lower:
            score += 1
            
    return score


async def run_stateful_character_profiling(
    project_id: int, 
    character_id: int, 
    book_id: Optional[int] = None, 
    max_chunks_to_scan: int = 5,
    clear_existing: bool = True,
    early_stopping_traits: Optional[List[str]] = None,
    is_cancelled_fn: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
    speculate: bool = False,
    event_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes profiling directly on a target CharacterTimelineEvent (representing a chronological state).
    Defaults to the nullable 'Base State' event if no event_id is supplied.
    """
    llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
    model_name = get_setting("llm_model", "local-model")
    
    # Load user-configured factual template from flat file (or fallback)
    factual_template_raw = load_prompt_template_from_file("factual")

    with Session(engine) as session:
        project = session.get(Project, project_id)
        char = session.get(Character, character_id)
        
        if not project or not char:
            return {}
            
        if char.locked:
            print(f"[Profiler] Character {char.name} is locked. Skipping.")
            return {}

        # Resolve targeted event
        if event_id:
            db_event = session.get(CharacterTimelineEvent, event_id)
        else:
            db_event = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == character_id)
                .where(CharacterTimelineEvent.book_id == None)
            ).first()
            if not db_event:
                db_event = CharacterTimelineEvent(
                    character_id=character_id,
                    book_id=None,
                    chapter_num=0,
                    shot_num=0,
                    label="Base State"
                )
                session.add(db_event)
                session.commit()
                session.refresh(db_event)

        if clear_existing:
            db_event.demographics = None
            db_event.physical_build = None
            db_event.hair_and_face = None
            db_event.distinguishing_marks = None
            db_event.visual_description = None
            session.add(db_event)
            session.commit()
            
            state_checklist = {
                "demographics": None,
                "physical_build": None,
                "hair_and_face": None,
                "distinguishing_marks": None
            }
        else:
            state_checklist = {
                "demographics": db_event.demographics,
                "physical_build": db_event.physical_build,
                "hair_and_face": db_event.hair_and_face,
                "distinguishing_marks": db_event.distinguishing_marks
            }

        aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
        alias_list = [a.alias for a in aliases]

    all_chunks = get_character_mention_chunks(project_id, character_id, book_id, chunk_size_words=120)
    if not all_chunks:
        print(f"[Profiler] No mention chunks found for character: {char.name}")
        return state_checklist

    for chunk in all_chunks:
        chunk["visual_score"] = score_chunk_visual_relevance(chunk["text"])

    all_chunks.sort(key=lambda x: x.get("visual_score", 0), reverse=True)
    sampled_chunks = all_chunks[:max_chunks_to_scan]

    # --- PHASE 1: OBJECTIVE FACTUAL EXTRACTION PASS ---
    scanned_count = 0
    for chunk_data in sampled_chunks:
        if is_cancelled_fn and is_cancelled_fn():
            break

        if early_stopping_traits:
            has_all_required = True
            for trait in early_stopping_traits:
                val = state_checklist.get(trait)
                if not val or str(val).strip() == "" or str(val).lower() == "null":
                    has_all_required = False
                    break
            if has_all_required:
                break

        unknown_fields = [k for k, v in state_checklist.items() if v is None or str(v).strip() == ""]
        scanned_count += 1
        chunk_text = chunk_data["text"]

        print(f"[Profiler] Scanning {chunk_data['book_name']} Chunk {chunk_data['chunk_index']} factually...")

        known_display = "\n".join([f"- {k}: {v}" for k, v in state_checklist.items() if v]) or "None"
        unknown_display = "\n".join([f"- {k}" for k in unknown_fields]) or "None"

        try:
            system_instructions = factual_template_raw.format(
                character_name=char.name,
                aliases=", ".join(alias_list),
                known_traits=known_display,
                unknown_traits=unknown_display
            )
        except Exception:
            system_instructions = factual_template_raw\
                .replace("{character_name}", char.name)\
                .replace("{aliases}", ", ".join(alias_list))\
                .replace("{known_traits}", known_display)\
                .replace("{unknown_traits}", unknown_display)

        user_prompt = (
            f"### PASSAGE ###\n"
            f"\"\"\"\n{chunk_text}\n\"\"\"\n\n"
            f"Task: Extract any written physical characteristics for {char.name}. Output a single JSON block."
        )

        full_prompt = f"{system_instructions}\n\n{user_prompt}"

        try:
            raw_response = await get_llm_response(full_prompt, llm_url, model_name)
            extracted_json = extract_json_from_text(raw_response)

            if extracted_json:
                for key in state_checklist.keys():
                    new_val = extracted_json.get(key)
                    if new_val and str(new_val).strip() != "" and str(new_val).lower() != "null":
                        if is_valid_permanent_trait(key, str(new_val), state_checklist[key]):
                            state_checklist[key] = str(new_val).strip()

            if progress_callback:
                progress_callback(char.id, scanned_count, len(sampled_chunks), state_checklist)

        except Exception as e:
            print(f"[Profiler] Error during factual pass: {str(e)}")

        await asyncio.sleep(0.5)

    # --- PHASE 2: CREATIVE SPECULATIVE CASTING PASS ---
    if speculate and not (is_cancelled_fn and is_cancelled_fn()):
        core_fields = ["demographics", "physical_build", "hair_and_face"]
        missing_fields = [f for f in core_fields if not state_checklist.get(f) or str(state_checklist.get(f)).strip() == ""]
        
        if missing_fields:
            # Load user-configured speculative template from flat file (or fallback)
            speculative_template_raw = load_prompt_template_from_file("speculative")
            
            known_display = "\n".join([f"- {k}: {v}" for k, v in state_checklist.items() if v]) or "None"
            unknown_display = "\n".join([f"- {k}" for k in missing_fields]) or "None"

            try:
                system_instructions = speculative_template_raw.format(
                    character_name=char.name,
                    aliases=", ".join(alias_list),
                    known_traits=known_display,
                    unknown_traits=unknown_display
                )
            except Exception:
                system_instructions = speculative_template_raw\
                    .replace("{character_name}", char.name)\
                    .replace("{aliases}", ", ".join(alias_list))\
                    .replace("{known_traits}", known_display)\
                    .replace("{unknown_traits}", unknown_display)

            representative_chunk = sampled_chunks[0]["text"] if sampled_chunks else "No passage context available."
            
            user_prompt = (
                f"### PASSAGE CONTEXT ###\n"
                f"\"\"\"\n{representative_chunk}\n\"\"\"\n\n"
                f"Task: Fill in only the missing traits {missing_fields} for {char.name}. Output a single JSON block."
            )

            full_prompt = f"{system_instructions}\n\n{user_prompt}"

            try:
                raw_response = await get_llm_response(full_prompt, llm_url, model_name)
                extracted_json = extract_json_from_text(raw_response)

                if extracted_json:
                    for key in missing_fields:
                        new_val = extracted_json.get(key)
                        if new_val and str(new_val).strip() != "" and str(new_val).lower() != "null":
                            if is_valid_permanent_trait(key, str(new_val), state_checklist[key]):
                                state_checklist[key] = str(new_val).strip()

                if progress_callback:
                    progress_callback(char.id, len(sampled_chunks), len(sampled_chunks), state_checklist)

                # Update internal checklist changes for timeline assignment
                db_event.demographics = state_checklist["demographics"]
                db_event.physical_build = state_checklist["physical_build"]
                db_event.hair_and_face = state_checklist["hair_and_face"]
                db_event.distinguishing_marks = state_checklist["distinguishing_marks"]

            except Exception as e:
                print(f"[Profiler] Error during speculative casting call: {str(e)}")

    # Save finalized profiling results to timeline event record
    with Session(engine) as session:
        if event_id:
            db_event = session.get(CharacterTimelineEvent, event_id)
        else:
            db_event = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == character_id)
                .where(CharacterTimelineEvent.book_id == None)
            ).first()
            
        char = session.get(Character, character_id)

        if db_event:
            db_event.demographics = state_checklist["demographics"]
            db_event.physical_build = state_checklist["physical_build"]
            db_event.hair_and_face = state_checklist["hair_and_face"]
            db_event.distinguishing_marks = state_checklist["distinguishing_marks"]
            
            if char and not char.locked:
                db_event.visual_description = compile_character_visual_prompt(db_event)
                
            session.add(db_event)
            session.commit()

    save_project_characters_to_json(project_id)
    return state_checklist


def save_setting(key: str, value: str):
    """Saves or updates a string configuration setting in the database."""
    from database.models import Setting
    with Session(engine) as session:
        setting = session.get(Setting, key)
        if setting:
            setting.value = value
        else:
            setting = Setting(key=key, value=value)
        session.add(setting)
        session.commit()

def get_template_file_path(mode: str) -> Path:
    """Returns the flat file Path for the given template mode."""
    filename = "factual_template.txt" if mode == "factual" else "speculative_template.txt"
    return Path("prompt_templates") / "characters" / filename


def save_prompt_template_to_file(mode: str, value: str):
    """Saves the template to a flat file as the primary source of truth, creating directories if needed."""
    file_path = get_template_file_path(mode)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(value)
    except Exception as e:
        print(f"[Templates] Failed to save {mode} template to {file_path}: {e}")


def load_prompt_template_from_file(mode: str) -> str:
    """Loads the template from a flat file, falling back to database setting then system default."""
    file_path = get_template_file_path(mode)
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[Templates] Failed to read {mode} template from {file_path}: {e}")
            
    # Fallback to database setting (allows migrating legacy profiles)
    setting_key = "character_profiler_template" if mode == "factual" else "character_speculative_template"
    db_val = get_setting(setting_key, "")
    if db_val:
        # Self-heal: Save database setting back to flat file for future access
        save_prompt_template_to_file(mode, db_val)
        return db_val
        
    # Final fallback to hardcoded system default
    if mode == "factual":
        return get_default_character_template()
    else:
        return get_speculative_character_template()


# --- the extracted trait set ------------------------------------------------
#
# The event's fields are one question each. Compilation is the only place that
# turns them into the sentence an image model reads, and it is deliberately
# mechanical: a bare colour gains the head noun it is missing ("gray" -> "gray
# hair"), a build becomes "who is ...", attire becomes "wearing ...". Nothing
# here invents a value; an empty field adds nothing.

NEW_TRAIT_FIELDS = ("sex", "species", "age", "hair_color", "hair_style",
                    "eye_color", "face", "build", "attire", "carried", "notes")
RENDERED_TRAIT_FIELDS = tuple(f for f in NEW_TRAIT_FIELDS if f != "notes")
# Build order for the compiled phrase. Hair is not in this list: it is one
# clause composed from two fields, so it is added before the loop.
TRAIT_ORDER = ("eye_color", "face", "build", "attire", "carried")

_SEX_NOUN = {"male": "man", "female": "woman"}
_VALID_AGES = {"young", "adult", "middle-aged", "old"}
_BLANK_VALUES = {"", "null", "none", "unknown", "not stated", "n/a", "unstated",
                 "blank", "blanks", "empty", "no value"}
# An animal word that already carries a sex must not be given a second one
# ("a female mare"). General English, not book vocabulary.
_SEXED_ANIMAL = {"mare", "stallion", "gelding", "bull", "cow", "ewe", "ram",
                 "sow", "boar", "buck", "doe", "hen", "cock", "queen", "drone",
                 "filly", "colt"}
_DETERMINERS = ("a ", "an ", "the ", "one ", "his ", "her ", "their ")
_HUMAN_SPECIES = {"human", "person", "people", "man", "woman"}


def is_non_human(species: str) -> bool:
    """Whether a species value says "not an ordinary human".

    There is deliberately no vocabulary of allowed species: a project may invent
    any, and a whitelist would either be series-specific or reject them. The
    useful signal is the opposite one - an empty value means human by default, so
    anything present is a claim the user should see. That is also what makes junk
    visible: a status such as "ta'veren" filed as a species shows up here and can
    be deleted in one keystroke.
    """
    text = (species or "").strip().lower()
    return bool(text) and text not in _HUMAN_SPECIES


# --- pinned traits ----------------------------------------------------------
#
# A pin is deliberately narrower than Character.locked: it says "this field
# always uses the Base value", not "do not re-analyse this character". The
# timeline still shows the changes the extraction found; the resolver ignores
# them for the pinned fields only.

def _clone_event(event: CharacterTimelineEvent, **overrides) -> CharacterTimelineEvent:
    """A transient copy of a timeline event. Never added to the session."""
    data = {field: getattr(event, field, None) for field in NEW_TRAIT_FIELDS}
    data.update(overrides)
    return CharacterTimelineEvent(
        id=event.id,
        character_id=event.character_id,
        book_id=event.book_id,
        chapter_num=event.chapter_num,
        shot_num=event.shot_num,
        label=event.label,
        demographics=event.demographics,
        physical_build=event.physical_build,
        hair_and_face=event.hair_and_face,
        distinguishing_marks=event.distinguishing_marks,
        source=event.source,
        visual_description=event.visual_description,
        **data,
    )


def carried_scope_mode() -> str:
    """``scene`` (default) / ``off`` / ``always`` for the ``carried`` trait.

    ``carried`` is the one trait whose persistence is wrong more often than not:
    an author says a character picks something up, and never says they put it
    down, so a snapshot that carries it forward decorates every later image with
    a prop the text stopped mentioning. Attire does not have this problem -
    clothes are re-described and genuinely persist.

    The setting is per project and lives in ``project_settings.json``. ``off``
    keeps the data and only drops it from rendered prompts (like ``notes``);
    ``scene`` shows it where the text actually said it; ``always`` is the old
    carry-forward behaviour. A pinned ``carried`` field always wins, which is the
    per-character way to say "this one really does carry the same thing".
    """
    try:
        from ui import state
        mode = str(getattr(state, "carried_mode", "scene") or "scene")
    except Exception:
        return "scene"
    return mode if mode in ("scene", "off", "always") else "scene"


def apply_carried_scope(event: Optional[CharacterTimelineEvent],
                        character: Optional[Character],
                        book_id: Optional[int], chapter_num: Optional[int],
                        shot_num: Optional[int]
                        ) -> Optional[CharacterTimelineEvent]:
    """Drop ``carried`` from a resolved event unless the project wants it.

    Returns the event itself when nothing changes, otherwise a transient copy -
    the row in the session must never be mutated by a render.
    """
    if event is None:
        return event
    mode = carried_scope_mode()
    if mode == "always":
        return event
    if character is not None and "carried" in get_pinned_fields(character):
        return event
    if mode == "scene":
        # The trait is valid at the position the description was recorded at.
        # Groups are one or two chunks, so this is "this scene" to within a
        # shot - and it cannot leak into a later scene, which is the point.
        query = (book_id, int(chapter_num or 0), int(shot_num or 0))
        here = (event.book_id, int(event.chapter_num or 0), int(event.shot_num or 0))
        if query == here:
            return event
    if not (getattr(event, "carried", None) or "").strip():
        return event
    return _clone_event(event, carried=None)


def get_pinned_fields(character: Optional[Character]) -> List[str]:
    raw = getattr(character, "pinned_traits", None) if character else None
    if not raw:
        return []
    try:
        values = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        values = str(raw).split(",")
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    return [str(value).strip() for value in values
            if str(value).strip() in NEW_TRAIT_FIELDS]


def set_pinned_fields(session: Session, character: Character,
                      fields: List[str]) -> None:
    clean = sorted({field for field in fields if field in NEW_TRAIT_FIELDS})
    character.pinned_traits = json.dumps(clean) if clean else None
    session.add(character)


def _effective_event(event: Optional[CharacterTimelineEvent],
                     base: Optional[CharacterTimelineEvent],
                     pinned: List[str]) -> Optional[CharacterTimelineEvent]:
    """A transient copy with pinned fields replaced by the Base value.

    Never added to the session: resolution must not write. An empty transient
    event is returned when there is no row at all, which is the old code's
    fallback without the empty Base row it used to persist.
    """
    if event is None:
        event = CharacterTimelineEvent(
            character_id=getattr(base, "character_id", None),
            book_id=None, chapter_num=0, shot_num=0, label="Base State")
    if not pinned or base is None:
        return event
    overrides = {}
    for field in pinned:
        value = getattr(base, field, None)
        if value:
            overrides[field] = value
    if not overrides:
        return event
    data = {field: getattr(event, field, None) for field in NEW_TRAIT_FIELDS}
    data.update(overrides)
    return CharacterTimelineEvent(
        id=event.id,
        character_id=event.character_id,
        book_id=event.book_id,
        chapter_num=event.chapter_num,
        shot_num=event.shot_num,
        label=event.label,
        demographics=event.demographics,
        physical_build=event.physical_build,
        hair_and_face=event.hair_and_face,
        distinguishing_marks=event.distinguishing_marks,
        source=event.source,
        visual_description=event.visual_description,
        **data,
    )


def resolve_event_at(session: Session, character: Character,
                     book_id: Optional[int], chapter_num: Optional[int],
                     shot_num: Optional[int], *,
                     book_order_map: Optional[Dict[int, int]] = None,
                     side_stories: Optional[set] = None,
                     events: Optional[List[CharacterTimelineEvent]] = None
                     ) -> tuple:
    """The effective event and the Base event at one story position.

    ``(effective, base)``. The effective event is transient and has pinned
    fields replaced by the Base value. The same positional rule the render path
    has always used: latest event at or before the position, across earlier
    books; side stories are visible only inside themselves.
    """
    if book_order_map is None or side_stories is None:
        books = session.exec(
            select(Book).where(Book.project_id == character.project_id)).all()
        book_order_map = {b.id: (b.book_order or 0) for b in books}
        side_stories = {b.id for b in books
                        if getattr(b, "is_side_story", False)}
    if events is None:
        events = session.exec(
            select(CharacterTimelineEvent)
            .where(CharacterTimelineEvent.character_id == character.id)).all()

    base = None
    matched = []
    target_order = book_order_map.get(book_id, 0) if book_id else 0
    for event in events:
        if event.book_id is None:
            base = event
            continue
        if event.book_id in side_stories and event.book_id != book_id:
            continue
        order = book_order_map.get(event.book_id, 0)
        if order < target_order:
            matched.append((event, order))
        elif order == target_order and chapter_num is not None:
            if event.chapter_num < chapter_num:
                matched.append((event, order))
            elif (event.chapter_num == chapter_num
                  and shot_num is not None and event.shot_num <= shot_num):
                matched.append((event, order))

    resolved = base
    if matched:
        matched.sort(key=lambda pair: (pair[1], pair[0].chapter_num,
                                       pair[0].shot_num, pair[0].id or 0))
        resolved = matched[-1][0]
    return _effective_event(resolved, base, get_pinned_fields(character)), base


def project_known_counts(session: Session, project_id: int) -> Dict[int, int]:
    """``{character_id: number of rendered fields known anywhere}``.

    The character list asks "who has traits?" - that is the union of every
    non-blank field across the base and all books, not the Base event's current
    snapshot. One query for the whole project keeps the list cheap.
    """
    char_ids = list(session.exec(
        select(Character.id).where(Character.project_id == project_id)).all())
    known: Dict[int, set] = {char_id: set() for char_id in char_ids}
    if not char_ids:
        return {}
    events = session.exec(
        select(CharacterTimelineEvent)
        .where(CharacterTimelineEvent.character_id.in_(char_ids))).all()
    for event in events:
        fields = known.setdefault(event.character_id, set())
        for field in RENDERED_TRAIT_FIELDS:
            if (getattr(event, field, None) or "").strip():
                fields.add(field)
    return {char_id: len(fields) for char_id, fields in known.items()}


def changed_trait_fields(event: CharacterTimelineEvent,
                         previous: Optional[CharacterTimelineEvent] = None
                         ) -> List[str]:
    """Rendered fields that differ from ``previous`` (or are populated)."""
    current = {field: (getattr(event, field, None) or "").strip()
               for field in RENDERED_TRAIT_FIELDS}
    if previous is None:
        return [field for field in RENDERED_TRAIT_FIELDS if current[field]]
    before = {field: (getattr(previous, field, None) or "").strip()
              for field in RENDERED_TRAIT_FIELDS}
    return [field for field in RENDERED_TRAIT_FIELDS
            if current[field] != before[field]]


def propagate_trait_change(session: Session, character: Character, field: str,
                           new_value: Optional[str], old_value: Optional[str], *,
                           book_id: Optional[int], chapter_num: int = 0,
                           shot_num: int = 0, force: bool = False,
                           log: Optional[Callable[[str], None]] = None) -> int:
    """Carry an edited trait forward through the events that repeat the old value.

    Events hold complete snapshots, so "this character wears a cloak at shot 40"
    and "the extractor said cloak at shot 3 and nothing has changed since" are
    stored identically. A manual edit therefore cannot rely on the renderer to
    inherit it - every later snapshot keeps asserting the extracted value - which
    is why filling in a Base trait did nothing visible and why the only way to
    make an edit stick was to pin it.

    This makes the edit mean what the user means by it: *this value from here
    on, until the book says otherwise*. Every snapshot that still holds the old
    value (or holds nothing) is rewritten; the first snapshot that holds a
    different value is a real change in the book, and propagation stops there.

    Re-saving an unchanged Base value re-asserts it, which is how an edit made
    before this existed is pushed through: with no new value to compare against,
    the change points are inferred from the book's own chain - the first event
    whose value differs from the one before it ends the run.

    Clearing a field is not propagated: blanking a Base value should not erase
    what each volume actually observed.

    ``force`` is not reachable from the UI. It exists for one-off repairs, where
    a value has to be imposed on events that already disagree.

    Returns the number of events rewritten.
    """
    say = log or (lambda *_a, **_k: None)
    new_text = (new_value or "").strip()
    new_norm = new_text.lower()
    old_norm = (old_value or "").strip().lower()
    if field not in RENDERED_TRAIT_FIELDS or not new_text:
        return 0
    if not force and new_norm == old_norm:
        # Same value saved again: only meaningful on the Base, and only as a
        # re-assertion of the whole series.
        if book_id is not None:
            return 0
    reassert = (not force) and new_norm == old_norm

    books = session.exec(
        select(Book).where(Book.project_id == character.project_id)).all()
    order_map = {b.id: (b.book_order if b.book_order is not None else 10 ** 6)
                 for b in books}
    side_stories = {b.id for b in books if getattr(b, "is_side_story", False)}

    def sort_key(event: CharacterTimelineEvent):
        if event.book_id is None:
            return (0, 0, 0, event.chapter_num, event.shot_num, event.id or 0)
        return (1, order_map.get(event.book_id, 10 ** 6), 0,
                event.chapter_num, event.shot_num, event.id or 0)

    # Everything at or after the edited position, in reading order.
    edit_key = ((0, 0, 0, 0, 0, 0) if book_id is None
                else (1, order_map.get(book_id, 10 ** 6), 0,
                      chapter_num, shot_num, 10 ** 9))

    events = session.exec(
        select(CharacterTimelineEvent)
        .where(CharacterTimelineEvent.character_id == character.id)).all()
    changed = 0
    previous_norm = None
    for event in sorted(events, key=sort_key):
        if sort_key(event) < edit_key:
            continue
        if event.book_id == book_id and event.chapter_num == chapter_num \
                and event.shot_num == shot_num:
            continue  # the edited row itself was already written by the caller
        # A side story is invisible to later volumes, so it must not take part in
        # - or interrupt - a change that carries forward.
        if event.book_id in side_stories and event.book_id != book_id:
            continue
        current = (getattr(event, field, None) or "").strip()
        current_norm = current.lower()
        if not force:
            if reassert:
                # No recorded old value: a change point is where the book's own
                # chain moves on from what the previous event said.
                if previous_norm and current_norm and current_norm != previous_norm:
                    break
            elif current and current_norm != old_norm:
                break  # the book changes it here; from this point it is the book's
        if current_norm:
            previous_norm = current_norm
        if current_norm == new_norm:
            continue  # already correct
        setattr(event, field, new_text)
        if event.source == "extraction":
            event.source = "manual"  # a re-analysis must not undo it
        if event.visual_description:
            # A stored override would otherwise keep the old wording.
            event.visual_description = compile_character_visual_prompt(event)
        session.add(event)
        changed += 1
    if changed:
        say(f"[Characters] '{character.name}': {field} applied to {changed} "
            f"later event(s) until the book changes it.")
    return changed


def character_field_stats(session: Session, character: Character,
                          book_id: Optional[int] = None,
                          order_map: Optional[Dict[int, int]] = None) -> Dict[str, dict]:
    """Per-field "populated / stable / changes" statistics for one character.

    ``book_id=None`` is the whole project; a book id restricts the value spans
    to that book (the Base event is used only when the book has no events at
    all, so an earlier book's base cannot make a later book look unstable).

    Values are compared in reading order and blanks are skipped: a field that
    is discovered halfway through the book is populated, not "changing".
    A pinned field reports the Base value as the effective one and keeps the
    raw change count as ``ignored_changes``.
    """
    events = session.exec(
        select(CharacterTimelineEvent)
        .where(CharacterTimelineEvent.character_id == character.id)).all()
    base = next((event for event in events if event.book_id is None), None)
    if book_id is None:
        if order_map is None:
            books = session.exec(
                select(Book).where(Book.project_id == character.project_id)).all()
            order_map = {b.id: (b.book_order or 0) for b in books}
        ordered = sorted(events, key=lambda event: (
            0 if event.book_id is None else 1,
            order_map.get(event.book_id, 0),
            event.chapter_num, event.shot_num, event.id or 0))
    else:
        ordered = [event for event in events if event.book_id == book_id]
        ordered.sort(key=lambda event: (event.chapter_num, event.shot_num,
                                        event.id or 0))
        if not ordered and base is not None:
            ordered = [base]

    pinned = set(get_pinned_fields(character))
    stats: Dict[str, dict] = {}
    for field in RENDERED_TRAIT_FIELDS:
        spans = []
        coverage = 0
        for event in ordered:
            value = (getattr(event, field, None) or "").strip()
            if not value:
                continue
            coverage += 1
            if spans and spans[-1]["value"].lower() == value.lower():
                spans[-1]["end_chapter"] = event.chapter_num
                spans[-1]["end_shot"] = event.shot_num
                continue
            spans.append({
                "value": value,
                "book_id": event.book_id,
                "start_chapter": event.chapter_num,
                "start_shot": event.shot_num,
                "end_chapter": event.chapter_num,
                "end_shot": event.shot_num,
                "event_id": event.id,
                "source": event.source,
            })
        changes = max(0, len(spans) - 1)
        within = across = 0
        for previous, current in zip(spans, spans[1:]):
            # The Base is the first appearance, not a separate book: a change
            # from it into the first book is a within-book change.
            if (previous["book_id"] is None or current["book_id"] is None
                    or previous["book_id"] == current["book_id"]):
                within += 1
            else:
                across += 1
        stats[field] = {
            "populated": bool(spans),
            "stable": len(spans) <= 1,
            "changes": changes,
            "within_book_changes": within,
            "across_book_changes": across,
            "coverage": coverage,
            "spans": spans,
            "pinned": field in pinned,
            "ignored_changes": changes if field in pinned else 0,
        }
        if field in pinned:
            # The effective value is the Base value, so the field is stable by
            # policy; keep the raw change count as "ignored".
            stats[field]["stable"] = True
            stats[field]["changes"] = 0
            stats[field]["within_book_changes"] = 0
            stats[field]["across_book_changes"] = 0
    return stats



# Which old settings checkbox governed each new field. A settings dict written
# before the new fields existed still controls them through these groups.
_TRAIT_LEGACY_GROUP = {
    "sex": "demographics", "species": "demographics", "age": "demographics",
    "hair_color": "hair_and_face", "hair_style": "hair_and_face",
    "eye_color": "hair_and_face", "face": "hair_and_face",
    "build": "physical_build",
}


def _trait_value(obj, key: str) -> str:
    """The field's value, or "" when it is absent or a placeholder."""
    value = getattr(obj, key, None)
    if value is None:
        return ""
    text = str(value).strip().strip(",;")
    return "" if text.lower() in _BLANK_VALUES else text


def _trait_enabled(enabled_fields: Dict[str, bool], key: str,
                   legacy_key: Optional[str] = None) -> bool:
    """Is a field enabled for rendering?

    New field names win. When a settings dict predates them, the old group it
    belonged to is honoured instead, so the existing checkboxes keep working.
    """
    if key in enabled_fields:
        return bool(enabled_fields[key])
    if legacy_key and legacy_key in enabled_fields:
        return bool(enabled_fields[legacy_key])
    return True


def has_new_traits(obj) -> bool:
    """True when the event carries any renderable new-field value."""
    return any(_trait_value(obj, key) for key in RENDERED_TRAIT_FIELDS)


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _base_noun(event) -> str:
    """The noun phrase the sentence is built on, from sex/species/age.

    Returns "" when none of them is stated. That is deliberate: "a person" adds
    nothing to an image, and a missing base is visible in the tab as a field the
    user can fill. A non-human species takes the slot instead of man/woman.
    """
    sex = _trait_value(event, "sex").lower()
    species = _trait_value(event, "species")
    age = _trait_value(event, "age").lower()
    species_word = species.lower()
    nonhuman = species_word not in _HUMAN_SPECIES and species_word != ""
    species_tokens = set(re.findall(r"[a-z]+", species_word))
    sex_word = ""
    if sex in _SEX_NOUN and not (species_tokens & _SEXED_ANIMAL):
        sex_word = sex if nonhuman else _SEX_NOUN[sex]
    parts = []
    if age in _VALID_AGES:
        parts.append(age)
    if sex_word:
        parts.append(sex_word)
    if nonhuman:
        parts.append(species)
    # An age with no noun ("a young") is not a base. Give it one.
    if parts and not sex_word and not nonhuman:
        parts.append("person")
    return " ".join(p for p in parts if p)


# --- hair: two fields, one clause -------------------------------------------
#
# hair_color and hair_style are separate fields because prose mentions one at a
# time ("her braid", "his grey hair"); they are one clause because a comma
# cannot join them ("brown hair, cut short"). The join is decided by the shape
# of the STYLE value, not by substring guessing at render time:
#
#   head noun present  -> the colour goes immediately before that noun
#                         "long curls" + "blond"       -> "long blond curls"
#   bare adjective     -> the colour takes the hair noun
#                         "cut short" + "dark"         -> "dark hair cut short"
#   prepositional      -> the hair noun leads, the arrangement follows
#                         "in a long braid" + "dark"   -> "dark hair in a long braid"
#   no hair at all     -> the colour is dropped; it contradicts the style
#                         "shaven head" + "white"      -> "shaven head"
#   animal noun        -> the mane/coat/tail is the noun
#                         "shaggy mane" + "black"      -> "shaggy black mane"
#
# Anything unrecognised falls back to the two comma items this code produced
# before, so a value the vocabulary does not know is never rendered as
# nonsense. The vocabulary is general English; a book's own coinages belong in
# the project's data, not here.
#
# A facial-hair value in hair_style ("long beard", "bushy white brows") is the
# extractor picking the wrong slot. `character_analysis._canonical_field` files
# it under `face` instead; the composer never has to know about beards.

_HAIR_HEAD_NOUNS = {
    "hair", "braid", "braids", "plait", "plaits", "curl", "curls", "ringlet",
    "ringlets", "wave", "waves", "fringe", "pigtail", "pigtails", "topknot",
    "topknots", "ponytail", "ponytails", "bun", "buns", "queue", "queues",
    "bangs", "crest", "tuft", "tufts", "crop", "mop", "shock", "lock", "locks",
    "wisp", "wisps", "strand", "strands", "mane", "manes", "tail", "tails",
    "coat", "coats",
}
_HAIR_ANIMAL_HEADS = {"mane", "manes", "tail", "tails", "coat", "coats"}
# Adjectives that premodify the hair noun: "shaggy dark hair", "long white hair".
_HAIR_PURE_ADJECTIVES = {
    "long", "short", "shoulder-length", "waist-length", "hip-length",
    "chin-length", "cropped", "curly", "wavy", "straight", "shaggy", "thick",
    "thin", "fine", "wispy", "sparse", "tousled", "tangled", "unkempt",
    "matted", "knotty", "lank", "smooth", "silky", "glossy", "neat", "braided",
    "plaited", "parted", "greying", "graying", "silvered", "thinning", "frizzy",
    "kinky", "coily", "wiry", "soft", "balding", "loose",
}
_HAIR_PREPOSITIONS = {"in", "with", "over", "under", "by", "to", "on", "at",
                      "across", "down", "from", "around", "behind", "past",
                      "back"}
# A participle only reads as a description when something completable follows it:
# "cut short" / "hanging to her waist", but not "shaved sides" (a stray noun).
_HAIR_PARTICIPLES = {
    "cut", "cropped", "shaved", "shorn", "parted", "tied", "pulled", "combed",
    "drawn", "clipped", "trimmed", "styled", "bleached", "dyed", "curled",
    "waved", "brushed", "hanging", "hung", "falling", "fell", "touching",
    "touches", "curling", "spilling", "swept", "braided", "plaited", "clubbed",
    "pinned", "knotted", "twisted", "slicked", "windblown",
}
_HAIR_PARTICIPLE_COMPLEMENT = _HAIR_PURE_ADJECTIVES | {
    "close", "back", "down", "up", "aside", "open", "properly", "neatly",
    "at", "to", "in", "on", "over", "under", "with", "by", "around", "across",
}
# A finite verb becomes a participle so the clause does not turn into a sentence.
_HAIR_PARTICIPLE_OF = {"fell": "falling", "hung": "hanging",
                       "touches": "touching", "touched": "touching",
                       "ran": "running", "runs": "running",
                       "spilled": "spilling", "circled": "circling",
                       "curled": "curling", "danced": "dancing",
                       "lay": "lying"}
_HAIR_NO_STYLE = re.compile(r"\b(bald|shaven|shaved|shorn)\b", re.I)
_HAIR_PARTIAL_HEAD = re.compile(
    r"\b(sides?|front|topknot|crest|fringe|patch|spot|braid|plaits?|"
    r"pattern|part|stripe|except)\b", re.I)

# Species we know to be animals: these lose the *invented* word "hair". Kept
# deliberately small and general - anything invented (a daemon, an ogier, a
# nameless apparition) is assumed to be haired, because a spurious "hair" reads
# far better than "a daemon, long black". Shape/plural variants are listed
# explicitly rather than stemmed, so an ordinary word is never matched by
# accident.
_ANIMAL_SPECIES = {
    "horse", "horses", "mare", "mares", "stallion", "stallions", "pony",
    "ponies", "donkey", "donkeys", "mule", "mules", "steed", "steeds", "ox",
    "oxen", "bull", "bulls", "cow", "cows", "calf", "calves", "goat", "goats",
    "sheep", "ewe", "ewes", "ram", "rams", "pig", "pigs", "boar", "boars",
    "sow", "sows", "dog", "dogs", "hound", "hounds", "cat", "cats", "kitten",
    "kittens", "wolf", "wolves", "cub", "cubs", "bear", "bears", "fox",
    "foxes", "deer", "stag", "stags", "doe", "does", "hare", "hares",
    "rabbit", "rabbits", "rat", "rats", "mouse", "mice", "bird", "birds",
    "crow", "crows", "raven", "ravens", "hawk", "hawks", "eagle", "eagles",
    "owl", "owls", "goose", "geese", "duck", "ducks", "hen", "hens",
    "chicken", "chickens", "snake", "snakes", "serpent", "serpents",
    "lizard", "lizards", "dragon", "dragons", "fish", "fishes",
}


def _hair_clean_colour(value: str) -> str:
    """A bare colour: 'dark hair' -> 'dark', 'silky hair' -> ''."""
    v = " ".join(str(value or "").split()).strip(" ,;.")
    if not v:
        return ""
    v = re.sub(r"\s+(head of )?hair$", "", v, flags=re.I)
    v = re.sub(r"\s+colou?ring$", "", v, flags=re.I)
    v = re.sub(r"\s+colou?red$", "", v, flags=re.I)
    v = v.strip(" ,")
    # A "colour" carrying a hair head noun is a style by mistake. Drop it
    # rather than splice it in ("thick curls hair").
    if _hair_head(v)[0]:
        return ""
    return v


def _hair_clean_style(value: str) -> str:
    v = " ".join(str(value or "").split()).strip(" ,;.")
    if not v:
        return ""
    v = re.sub(r"\s+", " ", v)
    return re.sub(r"^(is|was)\s+", "", v, flags=re.I)


def _hair_is_animal(species: str) -> bool:
    """Is the species one we know to be an animal?

    General-English nouns only; a book's invented kinds are not listed. The
    default for anything else - including an invented species and an empty
    species - is "haired", because most invented species are humanoid and a
    spurious "hair" is a smaller error than a bare adjective ("a daemon, long
    black"). A species we recognise as an animal loses the invented noun only;
    a noun the value already contains is always kept.
    """
    tokens = set(re.findall(r"[a-z]+", str(species or "").lower()))
    return bool(tokens & _ANIMAL_SPECIES)


def _hair_head(style: str):
    """The first hair head noun in a style value, and its token index."""
    for index, token in enumerate(re.findall(r"[a-z][a-z-]*", style.lower())):
        if token in _HAIR_HEAD_NOUNS:
            return token, index
    return "", -1


def _hair_split_colour(colour: str):
    """'black with gray at the temples' -> ('black', 'with', 'gray at the temples')."""
    match = re.match(
        r"^(.+?)\s+(with|flecked with|streaked with|shot with|turning|"
        r"graying|greying)\b\s*(.+)$", colour, re.I)
    if match:
        return (match.group(1).strip(), match.group(2).lower(),
                match.group(3).strip())
    return colour, "", ""


def _hair_colour_noun(colour: str, noun: str = "hair") -> str:
    """The colour as a mass noun: 'dark hair', or 'dark' with no noun (animals)."""
    main, joiner, rest = _hair_split_colour(colour)
    if joiner:
        return f"{main} {noun} {joiner} {rest}" if noun else f"{main} {joiner} {rest}"
    return f"{main} {noun}".strip() if noun else main


def _hair_insert(colour: str, style: str, head: str) -> str:
    match = re.search(rf"\b{re.escape(head)}\b", style, re.I)
    if not match:
        return ""
    return f"{style[:match.start()]}{colour} {style[match.start():]}".strip()


def _hair_value_after_participle(value: str) -> str:
    first = re.findall(r"[A-Za-z][A-Za-z-]*", value)
    if not first:
        return value
    bridged = _HAIR_PARTICIPLE_OF.get(first[0].lower(), first[0])
    return bridged + value[len(first[0]):]


_FACIAL_HAIR = re.compile(
    r"\b(beards?|bearded|moustaches?|mustaches?|whiskers|sideburns|stubble|"
    r"goatee|muttonchops|eyebrows?|brows?|lashes|eyelashes)\b", re.I)


def is_facial_hair(value: str) -> bool:
    """Is this value about a beard, moustache or brow?

    Used by the probes and available to a validator. It is **not** a write-path
    rule: facial hair stays in `hair_style`, because for most characters it is
    the hairstyle, and the composer renders it from there. A value that names
    both ("long hair and a bushy beard") is not a facial-hair-only value.
    """
    text = str(value or "")
    if not _FACIAL_HAIR.search(text):
        return False
    return not _hair_head(text)[0]


def _hair_hyphenated(value: str) -> str:
    """'clean cut' -> 'clean-cut': a premodifier, not a clause."""
    match = re.match(
        r"^(clean|close|neat|straight|thick|thin|fine|long|short)\s+"
        r"(cut|cropped|shaved|trimmed|parted)$", value, re.I)
    return f"{match.group(1).lower()}-{match.group(2).lower()}" if match else ""


def _hair_bucket(style: str):
    """(bucket, head noun) for a hair_style value.

    Order matters: a prepositional or participial opening means the value is a
    clause about the hair, not a noun phrase with the hair as its head.
    """
    if not style:
        return "", ""
    low = style.lower()
    words = re.findall(r"[A-Za-z][A-Za-z-]*", style)
    first = words[0].lower() if words else ""
    head, _index = _hair_head(style)
    if _HAIR_NO_STYLE.search(low) and not _HAIR_PARTIAL_HEAD.search(low):
        return "no-hair", ""
    if first in _HAIR_PREPOSITIONS:
        return "prepositional", ""
    if first in _HAIR_PARTICIPLES:
        second = words[1].strip(",;").lower() if len(words) > 1 else ""
        return ("trailing", "") if second in _HAIR_PARTICIPLE_COMPLEMENT \
            else ("fallback", "")
    if head in _HAIR_ANIMAL_HEADS:
        return "animal", head
    if _hair_hyphenated(style):
        return "adjective", ""
    if head:
        return "head-noun", head
    tokens = re.findall(r"[a-z][a-z-]*", low)
    if tokens and all(token in _HAIR_PURE_ADJECTIVES for token in tokens):
        return "adjective", ""
    if (len(words) > 1 and first in _HAIR_PURE_ADJECTIVES
            and words[1].lower() in _HAIR_PREPOSITIONS):
        return "adjective-then-prep", ""
    return "fallback", ""


def _hair_style_only(style: str, bucket: str, head: str, noun: str = "hair"):
    if bucket == "prepositional":
        return [f"{noun} {style}".strip() if noun else style]
    if bucket == "trailing":
        return [f"{noun} {_hair_value_after_participle(style)}".strip()
                if noun else style]
    if bucket in ("adjective", "adjective-then-prep"):
        prefix = _hair_hyphenated(style) or style
        return [f"{prefix} {noun}".strip() if noun else prefix]
    return [style]


def compose_hair(colour: str, style: str, species: str = "") -> List[str]:
    """The hair as comma-list clauses: usually one, sometimes two, never none.

    A species we recognise as an animal never has the word "hair" *invented* for
    it: a horse has a coat, and an adjective with no head noun ("shaggy") still
    reads as a description of the base noun that precedes it ("a horse,
    shaggy"). Anything else - human, an invented species, an empty species - is
    assumed to be haired.

    A noun that is already there is never removed. If either field names hair -
    "long hair", "black hair", "hair in a pigtail" - the value wins for every
    species, because the user typed it. Only the invented noun is suppressed.
    """
    # Read this before cleaning: the cleaner strips a redundant head noun
    # ("black hair" -> "black") precisely because the composer will put it back.
    # For a non-human that re-add is skipped, so a user-typed noun must be
    # honoured up front or it is deleted.
    named_hair = bool(re.search(r"\bhair\b", f"{colour or ''} {style or ''}", re.I))
    colour = _hair_clean_colour(colour)
    style = _hair_clean_style(style)
    if not colour and not style:
        return []
    noun = "hair" if (named_hair or not _hair_is_animal(species)) else ""
    if not style:
        return [_hair_colour_noun(colour, noun)]
    bucket, head = _hair_bucket(style)
    if not colour:
        return _hair_style_only(style, bucket, head, noun)
    if bucket == "no-hair":
        # A hairless head contradicts a colour; the style wins.
        return [style]
    if bucket == "animal":
        return [_hair_insert(colour, style, head)] if head else [style]
    main, joiner, rest = _hair_split_colour(colour)
    if bucket == "head-noun":
        # A multi-part colour ("black with grey at the temples") cannot go in
        # front of the noun; when that noun is "hair", the style's own
        # premodifier stays in front and the colour becomes the head.
        if rest and head == "hair":
            match = re.search(r"\bhair\b", style, re.I)
            pre = style[:match.start()].strip()
            post = style[match.end():]
            phrase = f"{main} {noun} {joiner} {rest}".strip() if noun \
                else f"{main} {joiner} {rest}"
            return [(f"{pre} {phrase}{post}").strip()]
        if rest:
            return [_hair_colour_noun(colour, noun), style]
        merged = _hair_insert(colour, style, head)
        return [merged]
    if bucket == "prepositional":
        if not noun:
            return [style]
        return [f"{_hair_colour_noun(colour, noun)} {style}"]
    if bucket == "adjective":
        return [f"{_hair_hyphenated(style) or style} "
                f"{_hair_colour_noun(colour, noun)}"]
    if bucket == "adjective-then-prep":
        first = re.findall(r"[A-Za-z][A-Za-z-]*", style)[0]
        remainder = style[len(style.split()[0]):].strip()
        return [f"{first} {_hair_colour_noun(colour, noun)} {remainder}".strip()]
    if bucket == "trailing":
        if not noun:
            # "black cut short" has no subject to hang the participle on.
            return [style]
        if " and " in style or " or " in style:
            return [_hair_colour_noun(colour, noun), style]
        return [f"{_hair_colour_noun(colour, noun)} "
                f"{_hair_value_after_participle(style)}"]
    return [_hair_colour_noun(colour, noun), style]


def _render_trait(field: str, value: str) -> str:
    """Add the mechanical connective a bare value is missing.

    Hair is not handled here: two fields become one clause, so
    ``compose_hair`` does that before the loop runs.
    """
    low = value.lower()
    prepositional = low.startswith(("in ", "with ", "over ", "under ", "by ",
                                    "to ", "on ", "at ", "a ", "an ", "the "))
    if field == "eye_color" and "eye" not in low and not prepositional:
        return f"{value} eyes"
    if field == "build":
        match = re.match(
            r"^(.*?)\s+(chest|body|frame|physique|stature|build|girl|boy|man|"
            r"woman|person|people|child)$", low)
        if match:
            stem = value[:match.start(2)].strip()
            stem = re.sub(r"\s+\b(of|with|in|on|at|and|or|to|from|by)\b\s*$", "",
                          stem, flags=re.I).strip()
            if not stem:
                return ""
            if match.group(2) == "chest":
                return f"who is {stem}-chested"
            return f"who is {stem}"
    if field == "build" and not low.startswith(("who is ", "is ")):
        return f"who is {value}"
    if field == "attire" and not low.startswith(("wearing ", "dressed ", "in ")):
        return f"wearing {value}"
    if field == "carried" and not low.startswith(("carrying ", "holding ", "with ")):
        return f"carrying {value}"
    return value


def compile_event_fields(event, enabled_fields: Dict[str, bool],
                         use_sentence_structure: bool) -> str:
    """The parenthetical description of one event - no character name.

    Sentence structure: ``a man, reddish hair, gray eyes, who is
    broad-shouldered, wearing a cloak, and carrying a sword``. Flat structure is
    the comma list the older non-sentence mode used.
    """
    base = _base_noun(event)
    base_words = set(re.findall(r"[a-z]+", base.lower()))

    values: List[tuple] = []
    seen_values = set()

    def add(field: str, value: str) -> None:
        if not value:
            return
        low = value.lower()
        if low in seen_values:
            return
        words = set(re.findall(r"[a-z]+", low))
        if words and words <= base_words:
            return
        seen_values.add(low)
        values.append((field, value))

    # Hair is one clause built from two fields, so it goes in before the loop.
    # Whichever half is enabled contributes; a settings dict that still uses the
    # old `hair_and_face` checkbox turns both off together.
    colour_on = _trait_enabled(enabled_fields, "hair_color", "hair_and_face")
    style_on = _trait_enabled(enabled_fields, "hair_style", "hair_and_face")
    colour = _trait_value(event, "hair_color") if colour_on else ""
    style = _trait_value(event, "hair_style") if style_on else ""

    if colour or style:
        for part in compose_hair(colour, style, _trait_value(event, "species")):
            add("hair", part)

    for field in TRAIT_ORDER:
        if not _trait_enabled(enabled_fields, field,
                              _TRAIT_LEGACY_GROUP.get(field)):
            continue
        add(field, _trait_value(event, field))

    # With traits but no sex/species/age, a neutral noun beats a description that
    # starts with a bare clause ("Name (gray hair, ...)"). It is also the honest
    # one: the missing sex is a field the user can fill, not a guess to make.
    if not base and values:
        base = "person"
        base_words = {base}

    if not use_sentence_structure:
        pieces = [base] + [value for _field, value in values]
        seen, out = set(), []
        for piece in pieces:
            key = piece.lower()
            if piece and key not in seen:
                seen.add(key)
                out.append(piece)
        return ", ".join(out)

    head = base
    clauses = [_render_trait(field, value) for field, value in values]
    if not head and not clauses:
        return ""
    if not head:
        return clauses[0] if len(clauses) == 1 else (
            ", ".join(clauses[:-1]) + ", and " + clauses[-1])
    if head.lower().startswith(_DETERMINERS):
        head_text = head
    else:
        head_text = f"{_article(head)} {head}"
    parts = [head_text] + clauses
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        text = f"{parts[0]}, {parts[1]}"
    else:
        text = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return re.sub(r"\band\s+who is\b", "who is", text)


def compile_character_description_from_event(
    char: Character,
    event: CharacterTimelineEvent,
    enabled_fields: Dict[str, bool],
    use_sentence_structure: bool
) -> str:
    """
    Returns the compiled character description (visual_description) for a specific timeline event state.
    """
    if event.visual_description and str(event.visual_description).strip():
        desc = event.visual_description.strip()
        if use_sentence_structure:
            if desc.startswith(char.name) or "(" in desc:
                return desc
            return f"{char.name} ({desc})"
        return desc

    # The extracted trait set, when this event has one.
    if has_new_traits(event):
        inner = compile_event_fields(event, enabled_fields, use_sentence_structure)
        if inner:
            if use_sentence_structure:
                return f"{char.name} ({inner})"
            return inner

    # Legacy: an event written by the old profiler, or one whose new fields are
    # all disabled. Kept verbatim so every existing project renders as it did.
    demo = event.demographics if enabled_fields.get("demographics", True) else None
    build = event.physical_build if enabled_fields.get("physical_build", True) else None
    hair_face = event.hair_and_face if enabled_fields.get("hair_and_face", True) else None
    marks = event.distinguishing_marks if enabled_fields.get("distinguishing_marks", True) else None

    has_any_details = any(
        f is not None and str(f).strip() != ""
        for f in [demo, build, hair_face, marks]
    )
    if not has_any_details:
        return char.name

    if not use_sentence_structure:
        pieces = []
        if demo: pieces.append(demo.strip())
        if hair_face: pieces.append(hair_face.strip())
        if build: pieces.append(build.strip())
        if marks: pieces.append(marks.strip())

        cleaned_pieces = []
        seen = set()
        for p in pieces:
            p_clean = p.strip()
            if p_clean and p_clean.lower() not in seen:
                cleaned_pieces.append(p_clean)
                seen.add(p_clean.lower())

        if not cleaned_pieces:
            return char.name
            
        return ", ".join(cleaned_pieces)
    else:
        base_noun = demo.strip() if demo else "person"
        first_char = base_noun[0].lower() if base_noun else 'p'
        article = "an" if first_char in "aeiou" else "a"
        
        clauses = []
        seen_clauses = set()
        
        for raw_val, name in [(hair_face, "hair_face"), (build, "build"), (marks, "marks")]:
            if not raw_val:
                continue
            val_clean = raw_val.strip()
            val_lower = val_clean.lower()
            if val_lower in seen_clauses:
                continue
            seen_clauses.add(val_lower)
            
            if name == "build":
                if not val_clean.lower().startswith("who is ") and not val_clean.lower().startswith("is "):
                    clauses.append(f"who is {val_clean}")
                else:
                    clauses.append(val_clean)
            else:
                clauses.append(val_clean)
            
        if clauses:
            if len(clauses) > 1:
                main_clauses = ", ".join(clauses[:-1])
                final_clause = clauses[-1]
                if not final_clause.lower().startswith("and "):
                    final_clause = f"and {final_clause}"
                parenthetical = f"{article} {base_noun}, {main_clauses}, {final_clause}"
            else:
                parenthetical = f"{article} {base_noun}, {clauses[0]}"
            
            parenthetical = re.sub(r'\s*,\s*,', ',', parenthetical)
            parenthetical = re.sub(r'\band\s+and\b', 'and', parenthetical)
            parenthetical = re.sub(r'\bwith\s+with\b', 'with', parenthetical)
            parenthetical = re.sub(r'\s+', ' ', parenthetical).strip()
            
            return f"{char.name} ({parenthetical})"
        else:
            return f"{char.name} ({article} {base_noun})"


def compile_character_description(char: Character, enabled_fields: Dict[str, bool], use_sentence_structure: bool) -> str:
    """Fallback legacy helper compiling base description from nullable base event."""
    with Session(engine) as session:
        base_ev = session.exec(
            select(CharacterTimelineEvent)
            .where(CharacterTimelineEvent.character_id == char.id)
            .where(CharacterTimelineEvent.book_id == None)
        ).first()
        if not base_ev:
            base_ev = CharacterTimelineEvent(
                character_id=char.id,
                book_id=None,
                chapter_num=0,
                shot_num=0,
                label="Base State"
            )
            session.add(base_ev)
            session.commit()
            session.refresh(base_ev)
    return compile_character_description_from_event(char, base_ev, enabled_fields, use_sentence_structure)


# --- automatic bracket repair ------------------------------------------------
#
# The prompt model names characters unprompted, but does not reliably wrap them
# in the square brackets the renderer substitutes on - so a bare "Ewin" renders
# as nothing at all. This puts those brackets back.
#
# It runs before tag resolution, which means prompts.csv files written before
# this existed are repaired at render time without being regenerated.

_AUTO_BRACKET_SETTING = "auto_bracket_character_names"
_TAG_SPAN = re.compile(r"\[[^\[\]]*\]")
_LOWERCASE_WORD = re.compile(r"\b[a-z][a-z'\-]*\b")
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)

# project_id -> every word that appears in lowercase somewhere in that project's
# prompts. A name in this set is also an ordinary word ("Stone" vs "stone"), so a
# capital letter cannot be trusted to mark it as a name. Built from the project's
# own prompts, so nothing is hardcoded about any language or series.
_LOWERCASE_WORDS_CACHE: Dict[int, Set[str]] = {}


def auto_bracket_enabled(session: Optional[Session] = None) -> bool:
    """Global toggle for bracket repair; on unless the setting says otherwise."""
    value = str(get_setting(_AUTO_BRACKET_SETTING, "1", session)).strip().lower()
    return value not in ("0", "false", "off", "no", "")


def get_bracketable_names(project_id: int,
                          session: Optional[Session] = None) -> List[str]:
    """A project's character names and aliases, longest first."""
    should_close = False
    if session is None:
        session = Session(engine)
        should_close = True
    try:
        names: List[str] = []
        for c in session.exec(
            select(Character).where(Character.project_id == project_id)
        ).all():
            name = clean_character_tag(c.name)
            if name and len(name) >= 3:
                names.append(name)
        for a in session.exec(
            select(CharacterAlias)
            .join(Character)
            .where(Character.project_id == project_id)
        ).all():
            alias = clean_character_tag(a.alias)
            if alias and len(alias) >= 3:
                names.append(alias)
        return sorted(set(names), key=len, reverse=True)
    finally:
        if should_close:
            session.close()


def project_lowercase_words(project_id: int) -> Set[str]:
    """Words that appear lowercased in this project's prompts, cached.

    Used to tell a name apart from an ordinary word. Empty when a project has no
    prompts yet, in which case callers fall back to the conservative behaviour.
    """
    if project_id in _LOWERCASE_WORDS_CACHE:
        return _LOWERCASE_WORDS_CACHE[project_id]

    words: Set[str] = set()
    try:
        with Session(engine) as session:
            project = session.get(Project, project_id)
            base_dir = Path(get_setting("output_dir", "./output", session))
        if project:
            for csv_path in (base_dir / project.name).glob("*/prompts.csv"):
                try:
                    text = csv_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                words.update(_LOWERCASE_WORD.findall(text))
    except Exception:
        pass

    _LOWERCASE_WORDS_CACHE[project_id] = words
    return words


def _starts_sentence(text: str, pos: int) -> bool:
    """True when pos opens a sentence, where a capital letter proves nothing."""
    i = pos - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    return i < 0 or text[i] in ".!?\n"


# (project_id, count, hash) -> compiled (multi-word, single-word) matchers.
# A large project carries over a thousand names, and rebuilding the alternations
# for every prompt dominated the cost, so they are compiled once.
_PATTERN_CACHE: Dict[tuple, tuple] = {}


def _name_patterns(project_id: int, names: List[str]):
    """Compiled matchers for a project's names, longest alternative first."""
    key = (project_id, len(names), hash(tuple(names)))
    cached = _PATTERN_CACHE.get(key)
    if cached is not None:
        return cached

    multi: Set[str] = set()
    single: List[str] = []
    for name in names:
        if " " in name:
            multi.add(name)
            stripped = _LEADING_ARTICLE.sub("", name)
            # Only keep the article-stripped form if it is STILL a phrase. A
            # single word would land in the case-insensitive pattern and bypass
            # every guard ("The Captain" -> bare "captain").
            if stripped != name and " " in stripped:
                multi.add(stripped)
        elif name != name.lower():
            # An all-lowercase single word ("captain", "doctor") matches the
            # ordinary noun it is spelled like, everywhere, so it is never
            # bracketed automatically. Multi-word aliases are still allowed.
            single.append(name)

    def build(values, flags=0, extra=""):
        if not values:
            return None
        alts = sorted(values, key=len, reverse=True)
        return re.compile("|".join(
            rf"(?<!\w){re.escape(v)}(?!\w){extra}" for v in alts), flags)

    # A capitalised word immediately followed by another capitalised word is part
    # of a longer proper noun that is not the character ("White District",
    # "Mataro Sphinxes"), so it is left alone.
    result = (build(multi, re.IGNORECASE), build(single, 0, r"(?!\s+[A-Z])"))
    if len(_PATTERN_CACHE) > 4:
        _PATTERN_CACHE.clear()
    _PATTERN_CACHE[key] = result
    return result


def auto_bracket_character_names(prompt: str, project_id: int,
                                 names: Optional[List[str]] = None) -> str:
    """Wrap bare character names and aliases so the renderer can substitute them.

    A single-word name is matched case-sensitively. Where the capital proves
    nothing - the word is also an ordinary one, judged from the project's own
    prompts, and the hit opens a sentence - it is left alone, so "Stone walls"
    is never mistaken for the character Stone.

    Multi-word aliases (epithets such as "tall man") match case-insensitively.
    A phrase has too little false-positive surface to need that guard, and it is
    how an unnamed character becomes resolvable at all, since the model will
    write "a tall man" whether or not it can name him.

    Idempotent: anything already inside brackets is left untouched. Brackets
    that resolve to no character are stripped downstream, so a wrong guess here
    degrades to plain text rather than to a wrong description.
    """
    if not prompt or not prompt.strip():
        return prompt
    prompt = normalize_character_text(prompt)
    if not project_id or not auto_bracket_enabled():
        return prompt
    if names is None:
        names = get_bracketable_names(project_id)
    if not names:
        return prompt

    lowercase_words = project_lowercase_words(project_id) if project_id else set()
    spans: List[str] = []

    def stash(text: str) -> str:
        spans.append(text)
        return f"\x00{len(spans) - 1}\x00"

    # Pull existing tags out first, so repair can never nest brackets.
    work = _TAG_SPAN.sub(lambda m: stash(m.group(0)), prompt)

    multi_rx, single_rx = _name_patterns(project_id, names)
    prompt_words = set(_LOWERCASE_WORD.findall(work))

    if multi_rx is not None:
        # Epithets resolve through the alias table, which matches
        # case-insensitively, so the bracketed text is whatever was written.
        work = multi_rx.sub(lambda m: stash(f"[{m.group(0)}]"), work)

    if single_rx is not None:
        def repl(m):
            word = m.group(0)
            ambiguous = (word.lower() in lowercase_words
                         or word.lower() in prompt_words)
            if ambiguous and _starts_sentence(work, m.start()):
                return word
            return stash(f"[{word}]")

        work = single_rx.sub(repl, work)

    for i, span in enumerate(spans):
        work = work.replace(f"\x00{i}\x00", span)
    return work


def replace_character_tags_in_prompt(
    prompt: str, 
    project_id: int, 
    enabled_fields: Dict[str, bool], 
    use_sentence_structure: bool,
    book_id: Optional[int] = None,
    chapter_num: Optional[int] = None,
    shot_num: Optional[int] = None
) -> str:
    """
    Scans a prompt string for bracketed tags, resolves characters using book-aware linking and recency fallback,
    and returns a modified prompt string containing compiled physical descriptions.

    Bare character names are bracketed first (see ``auto_bracket_character_names``),
    so prompts generated before that existed still resolve.
    """
    prompt = normalize_character_text(prompt)
    prompt = auto_bracket_character_names(prompt, project_id)
    bracket_regex = re.compile(r"\[(.*?)\]")
    matches = bracket_regex.findall(prompt)
    if not matches:
        return prompt

    modified_prompt = prompt
    expanded_character_ids = set()
    # A description is inserted ONCE per prompt. Repeating it for every mention bloats the
    # request and gives the image model the same person three times - measured on 48 real
    # prompts, 11 of them carried a repeated name and 20 of 64 insertions were redundant.
    # The first mention carries the description; later mentions are the bare name.
    described_tags: set = set()
    
    with Session(engine) as session:
        session.expire_all()
        
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        book_order_map = {b.id: (b.book_order or 0) for b in books}
        target_book_order = book_order_map.get(book_id, 0) if book_id else 0
        # A side story inherits from earlier volumes but passes nothing forward.
        # When resolving for a DIFFERENT book, its events and links are invisible.
        side_stories = {b.id for b in books if getattr(b, "is_side_story", False)}

        for match in matches:
            raw_tag = match.strip()
            tag = clean_character_tag(raw_tag)
            tag_lower = tag.lower()

            # Already given its full description earlier in this prompt: keep the bare name.
            if tag_lower in described_tags:
                modified_prompt = modified_prompt.replace(f"[{raw_tag}]", tag, 1)
                continue

            # 1. Gather all candidate characters matching either alias or canonical name
            candidate_chars = []
            
            alias_matches = session.exec(
                select(CharacterAlias)
                .join(Character)
                .where(Character.project_id == project_id)
                .where(CharacterAlias.alias.ilike(tag))
            ).all()
            for am in alias_matches:
                c = session.get(Character, am.character_id)
                if c and c not in candidate_chars:
                    candidate_chars.append(c)
                    
            direct_name_matches = session.exec(
                select(Character)
                .where(Character.project_id == project_id)
                .where(Character.name.ilike(tag))
            ).all()
            for dnm in direct_name_matches:
                if dnm not in candidate_chars:
                    candidate_chars.append(dnm)

            if not candidate_chars:
                # No character entity found; strip brackets and keep raw tag text
                modified_prompt = modified_prompt.replace(f"[{raw_tag}]", tag, 1)
                continue

            # 2. Book-Aware Disambiguation
            char = None
            if len(candidate_chars) == 1:
                char = candidate_chars[0]
            else:
                # Multi-match collision: check which candidate is explicitly linked to current book_id
                if book_id:
                    linked_candidates = []
                    for cand in candidate_chars:
                        link = session.exec(
                            select(CharacterBookLink)
                            .where(CharacterBookLink.character_id == cand.id)
                            .where(CharacterBookLink.book_id == book_id)
                        ).first()
                        if link:
                            linked_candidates.append(cand)
                            
                    if len(linked_candidates) == 1:
                        char = linked_candidates[0]
                    elif len(linked_candidates) > 1:
                        # Multiple linked to same book: prioritize exact name match over alias match
                        exact_name = [c for c in linked_candidates if c.name.lower() == tag_lower]
                        char = exact_name[0] if exact_name else linked_candidates[0]

                # Fallback: find candidate linked to closest previous book (chronological recency)
                if not char and book_id:
                    scored_candidates = []
                    for cand in candidate_chars:
                        links = session.exec(
                            select(CharacterBookLink).where(CharacterBookLink.character_id == cand.id)
                        ).all()
                        cand_orders = [book_order_map.get(lk.book_id, 0) for lk in links
                                       if lk.book_id in book_order_map
                                       and not (lk.book_id in side_stories
                                                and lk.book_id != book_id)]
                        cand_orders_before = [order for order in cand_orders if order < target_book_order]
                        best_order = max(cand_orders_before) if cand_orders_before else -1
                        scored_candidates.append((cand, best_order))
                    
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)
                    if scored_candidates and scored_candidates[0][1] >= 0:
                        char = scored_candidates[0][0]

                # Final fallback: choose first candidate (or exact name match)
                if not char:
                    exact_name = [c for c in candidate_chars if c.name.lower() == tag_lower]
                    char = exact_name[0] if exact_name else candidate_chars[0]

            # 3. Build Description or Canonical Name
            if char.id in expanded_character_ids:
                replacement = char.name
            else:
                events = session.exec(
                    select(CharacterTimelineEvent)
                    .where(CharacterTimelineEvent.character_id == char.id)
                ).all()
                resolved_ev, _base_ev = resolve_event_at(
                    session, char, book_id, chapter_num, shot_num,
                    book_order_map=book_order_map, side_stories=side_stories,
                    events=events)
                resolved_ev = apply_carried_scope(resolved_ev, char, book_id,
                                                  chapter_num, shot_num)
                replacement = compile_character_description_from_event(
                    char, resolved_ev, enabled_fields, use_sentence_structure
                )
                expanded_character_ids.add(char.id)
            
            modified_prompt = modified_prompt.replace(f"[{raw_tag}]", replacement, 1)
            described_tags.add(tag_lower)

    return modified_prompt


def get_alias_occurrences(project_id: int, alias_text: str) -> List[Dict[str, Any]]:
    """
    Searches all transcript.txt files in the project for occurrences of alias_text.
    """
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    occurrences = []

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return []
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()

    for book in books:
        transcript_path = base_output_dir / project.name / book.name / "transcript.txt"
        if not transcript_path.exists():
            continue

        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                content = f.read()
            content = content.replace("\r\n", "\n").replace("\r", "\n")
        except Exception as e:
            print(f"[Profiler] Error reading transcript for context search: {str(e)}")
            continue

        cleaned_text = content.replace("==CHAPTER==", " ")
        pattern = re.compile(rf"(\b{re.escape(alias_text)}(?:'s|’s)?\b)", re.IGNORECASE)
        
        for match in pattern.finditer(cleaned_text):
            start, end = match.span()
            window_start = max(0, start - 500)
            window_end = min(len(cleaned_text), end + 500)
            
            fragment = cleaned_text[window_start:window_end]
            match_start_in_frag = start - window_start
            match_end_in_frag = end - window_start
            
            prefix = fragment[:match_start_in_frag]
            match_word = fragment[match_start_in_frag:match_end_in_frag]
            suffix = fragment[match_end_in_frag:]
            
            highlighted_html = (
                f"... {prefix}<mark class='bg-yellow-200 text-slate-900 px-1 rounded font-bold'>{match_word}</mark>{suffix} ..."
            ).replace("\n", " ")

            occurrences.append({
                "book_id": book.id,
                "book_name": book.name,
                "raw_context": fragment,
                "html_context": highlighted_html,
                "match_word": match_word
            })

    return occurrences


def get_character_import_matches(tgt_project_id: int, src_project_id: int) -> List[Dict[str, Any]]:
    """
    Finds pairings between target project unlocked characters and source project characters.
    Matches if canonical name matches or if aliases intersect.
    Filters out source characters with empty descriptions or fallback "a person named X" descriptions.
    Ensures each source character is matched with at most one target character (best fit).
    Sorts matches descending by target mentions count so important characters are at the top.
    """
    from services.character_organizer import get_character_frequency_map_db
    with Session(engine) as session:
        tgt_project = session.get(Project, tgt_project_id)
        if not tgt_project:
            return []
            
        frequencies = get_character_frequency_map_db(tgt_project.name, session)
        
        def get_mentions(char_obj, alias_texts):
            total = sum(frequencies.get(a.lower(), 0) for a in alias_texts)
            if not total:
                total = frequencies.get(char_obj.name.lower(), 0)
            return total

        # Grab unlocked target characters
        tgt_chars = session.exec(
            select(Character).where(Character.project_id == tgt_project_id).where(Character.locked == False)
        ).all()
        
        # Grab all source characters
        src_chars = session.exec(
            select(Character).where(Character.project_id == src_project_id)
        ).all()
        
        # Fetch details for Target Project characters
        tgt_books = session.exec(select(Book).where(Book.project_id == tgt_project_id)).all()
        tgt_book_order_map = {b.id: (b.book_order if b.book_order is not None else 0) for b in tgt_books}
        
        tgt_data = []
        for tc in tgt_chars:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == tc.id)).all()
            alias_list = [a.alias for a in aliases]
            alias_texts = {a.alias.lower().strip() for a in aliases}
            alias_texts.add(tc.name.lower().strip())
            
            # Resolve target latest timeline event to show in comparison
            tgt_evs = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == tc.id)
            ).all()
            
            tgt_latest_ev = None
            if tgt_evs:
                def tgt_sort_key(ev: CharacterTimelineEvent):
                    if ev.book_id is None:
                        return (-1, 0, 0, ev.id or 0)
                    return (tgt_book_order_map.get(ev.book_id, 0), ev.chapter_num, ev.shot_num, ev.id or 0)
                tgt_latest_ev = sorted(tgt_evs, key=tgt_sort_key)[-1]
            
            mentions = get_mentions(tc, alias_texts)
            
            tgt_data.append({
                "char": tc,
                "aliases": alias_list,
                "all_terms_lower": alias_texts,
                "desc": tgt_latest_ev.visual_description if tgt_latest_ev else "",
                "mentions": mentions
            })
            
        # Fetch details for Source Project characters
        src_books = session.exec(select(Book).where(Book.project_id == src_project_id)).all()
        src_book_order_map = {b.id: (b.book_order if b.book_order is not None else 0) for b in src_books}
        
        src_data = []
        for sc in src_chars:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == sc.id)).all()
            alias_texts = {a.alias.lower().strip() for a in aliases}
            alias_texts.add(sc.name.lower().strip())
            
            src_evs = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == sc.id)
            ).all()
            
            if not src_evs:
                continue
                
            # Chronologically sort timeline events to retrieve the latest state
            def src_sort_key(ev: CharacterTimelineEvent):
                if ev.book_id is None:
                    return (-1, 0, 0, ev.id or 0)
                return (src_book_order_map.get(ev.book_id, 0), ev.chapter_num, ev.shot_num, ev.id or 0)
            
            latest_src_ev = sorted(src_evs, key=src_sort_key)[-1]
            
            # Verify if there is actual physical descriptive data in the latest state
            has_physical_data = any(
                f and str(f).strip() and str(f).lower() != "none"
                for f in [latest_src_ev.demographics, latest_src_ev.physical_build, latest_src_ev.hair_and_face, latest_src_ev.distinguishing_marks]
            )
            
            # Check if there is a custom visual description that is not just the default compiled fallback
            has_custom_desc = False
            if latest_src_ev.visual_description and latest_src_ev.visual_description.strip():
                fallback_str = f"a person named {sc.name.lower().strip()}"
                if latest_src_ev.visual_description.lower().strip() != fallback_str:
                    has_custom_desc = True
                    
            if not (has_physical_data or has_custom_desc):
                continue
            
            src_data.append({
                "char": sc,
                "aliases": [a.alias for a in aliases],
                "all_terms_lower": alias_texts,
                "desc": latest_src_ev.visual_description or "",
                "latest_ev": latest_src_ev
            })
            
        # Collect all candidates matching the criteria
        candidates = []
        for td in tgt_data:
            tc = td["char"]
            for sd in src_data:
                sc = sd["char"]
                score = 0
                if sc.name.lower().strip() == tc.name.lower().strip():
                    score = 3
                elif sc.name.lower().strip() in td["all_terms_lower"] or tc.name.lower().strip() in sd["all_terms_lower"]:
                    score = 2
                elif td["all_terms_lower"].intersection(sd["all_terms_lower"]):
                    score = 1
                    
                if score > 0:
                    candidates.append({
                        "score": score,
                        "tgt_mentions": td["mentions"],
                        "sd": sd,
                        "td": td
                    })
                    
        # Sort candidates: high match score first, then target mentions descending
        candidates.sort(key=lambda x: (x["score"], x["tgt_mentions"]), reverse=True)
        
        assigned_sources = set()
        assigned_targets = set()
        final_matches = []
        
        for cand in candidates:
            src_id = cand["sd"]["char"].id
            tgt_id = cand["td"]["char"].id
            
            if src_id in assigned_sources or tgt_id in assigned_targets:
                continue
                
            assigned_sources.add(src_id)
            assigned_targets.add(tgt_id)
            
            final_matches.append({
                "source_char_id": src_id,
                "source_name": cand["sd"]["char"].name,
                "source_aliases": cand["sd"]["aliases"],
                "source_desc": cand["sd"]["desc"] or "No traits profiled.",
                "target_char_id": tgt_id,
                "target_name": cand["td"]["char"].name,
                "target_aliases": cand["td"]["aliases"],
                "target_desc": cand["td"]["desc"] or "No traits profiled.",
                "target_mentions": cand["tgt_mentions"]
            })
            
        # Final sort: major characters (highest target hit counts) appear at the top
        final_matches.sort(key=lambda x: x["target_mentions"], reverse=True)
        return final_matches


def get_matching_source_projects(project_id: int) -> List[Dict[str, Any]]:
    """
    Returns list of source projects that have valid character matches with matching descriptions.
    """
    matching_projects = []
    with Session(engine) as session:
        other_projects = session.exec(
            select(Project).where(Project.id != project_id)
        ).all()
        
        for proj in other_projects:
            matches = get_character_import_matches(project_id, proj.id)
            if matches:
                matching_projects.append({
                    "id": proj.id,
                    "name": proj.name
                })
    return matching_projects


def execute_character_import(
    tgt_project_id: int,
    pairings: List[Dict[str, Any]],
    lock_after_import: bool,
    import_merge_aliases: bool
):
    """
    Executes core profiling copies for selected pairing items.
    Applies overrides, merges alias maps, and handles structural name updates.
    Consolidates and deletes duplicate target characters with overlapping aliases.
    """
    def safe_add_alias(char_id: int, alias_text: str, db_session: Session):
        alias_clean = alias_text.strip()
        if not alias_clean:
            return
        dup = db_session.exec(
            select(CharacterAlias)
            .where(CharacterAlias.character_id == char_id)
            .where(CharacterAlias.alias == alias_clean)
        ).first()
        if not dup:
            new_alias = CharacterAlias(character_id=char_id, alias=alias_clean)
            db_session.add(new_alias)

    with Session(engine) as session:
        for pair in pairings:
            src_char_id = pair["source_char_id"]
            tgt_char_id = pair["target_char_id"]
            
            src_char = session.get(Character, src_char_id)
            tgt_char = session.get(Character, tgt_char_id)
            
            if not src_char or not tgt_char:
                continue
                
            # 1. Resolve chronological latest source timeline event to sync as the baseline state
            src_books = session.exec(select(Book).where(Book.project_id == src_char.project_id)).all()
            src_book_order_map = {b.id: (b.book_order if b.book_order is not None else 0) for b in src_books}
            
            src_evs = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == src_char_id)
            ).all()
            
            src_latest_ev = None
            if src_evs:
                def src_sort_key(ev: CharacterTimelineEvent):
                    if ev.book_id is None:
                        return (-1, 0, 0, ev.id or 0)
                    return (src_book_order_map.get(ev.book_id, 0), ev.chapter_num, ev.shot_num, ev.id or 0)
                src_latest_ev = sorted(src_evs, key=src_sort_key)[-1]
            
            tgt_base_ev = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == tgt_char_id)
                .where(CharacterTimelineEvent.book_id == None)
            ).first()
            
            if not tgt_base_ev:
                tgt_base_ev = CharacterTimelineEvent(
                    character_id=tgt_char_id,
                    book_id=None,
                    chapter_num=0,
                    shot_num=0,
                    label="Base State"
                )
                session.add(tgt_base_ev)
                session.flush()
                
            if src_latest_ev:
                tgt_base_ev.demographics = src_latest_ev.demographics
                tgt_base_ev.physical_build = src_latest_ev.physical_build
                tgt_base_ev.hair_and_face = src_latest_ev.hair_and_face
                tgt_base_ev.distinguishing_marks = src_latest_ev.distinguishing_marks
                tgt_base_ev.visual_description = src_latest_ev.visual_description
                session.add(tgt_base_ev)
            
            # 2. Rename Target, Merge/Consolidate Duplicates, and Import Aliases
            if import_merge_aliases:
                old_tgt_name = tgt_char.name
                tgt_char.name = src_char.name
                session.add(tgt_char)
                
                # Retrieve source character aliases
                src_aliases = session.exec(
                    select(CharacterAlias).where(CharacterAlias.character_id == src_char_id)
                ).all()
                
                # Compile target names and aliases to watch out for
                conflict_names = {src_char.name.lower().strip()}
                for sa in src_aliases:
                    conflict_names.add(sa.alias.lower().strip())
                conflict_names.add(old_tgt_name.lower().strip())
                
                # Preserve the old target name as an alias of the renamed canonical target character
                if old_tgt_name.lower().strip() != src_char.name.lower().strip():
                    safe_add_alias(tgt_char_id, old_tgt_name, session)
                
                # Scan and merge other characters in the target project that share any of these names/aliases
                other_chars = session.exec(
                    select(Character)
                    .where(Character.project_id == tgt_project_id)
                    .where(Character.id != tgt_char_id)
                ).all()
                
                for other_char in other_chars:
                    is_conflict = other_char.name.lower().strip() in conflict_names
                    
                    other_aliases = session.exec(
                        select(CharacterAlias).where(CharacterAlias.character_id == other_char.id)
                    ).all()
                    
                    if not is_conflict:
                        for oa in other_aliases:
                            if oa.alias.lower().strip() in conflict_names:
                                is_conflict = True
                                break
                                
                    if is_conflict:
                        # Merge the duplicate character's name and aliases into our main target character
                        safe_add_alias(tgt_char_id, other_char.name, session)
                        for oa in other_aliases:
                            safe_add_alias(tgt_char_id, oa.alias, session)
                            session.delete(oa)
                            
                        # Reparent custom timeline overrides (e.g. from specific book coordinates)
                        other_evs = session.exec(
                            select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id == other_char.id)
                        ).all()
                        for ev in other_evs:
                            if ev.book_id is None:
                                session.delete(ev)  # Baseline state is redundant
                            else:
                                ev.character_id = tgt_char_id  # Reparent chronological override event
                                session.add(ev)
                                
                        session.delete(other_char)
                
                # Finally, import remaining source aliases to our main target character
                for sa in src_aliases:
                    safe_add_alias(tgt_char_id, sa.alias, session)
            else:
                # Standard traits import without renaming or merging other characters
                src_aliases = session.exec(
                    select(CharacterAlias).where(CharacterAlias.character_id == src_char_id)
                ).all()
                for sa in src_aliases:
                    safe_add_alias(tgt_char_id, sa.alias, session)
                    
            # 3. Lock target character from automatic LLM profiling overrides
            if lock_after_import:
                tgt_char.locked = True
                session.add(tgt_char)
                
        session.commit()
    
    # Keep flat file characters.json as the ultimate source of truth on disk
    save_project_characters_to_json(tgt_project_id)


def merge_apostrophe_duplicates(project_id: int, log=None) -> Dict[str, int]:
    """Merge characters whose names differ only by typographic single quotes.

    A reader sees ``al’Vere`` and ``al'Vere`` as the same name; exact-match
    queries do not. Import and matching now normalize at the boundary, and this
    repairs data created before that. It also normalizes every remaining name
    and alias, and removes aliases that collapse onto the canonical name.
    """
    say = log or (lambda *_a, **_k: None)
    stats = {"groups": 0, "merged": 0, "names_normalized": 0,
             "aliases_normalized": 0, "aliases_removed": 0}

    with Session(engine) as session:
        chars = session.exec(
            select(Character).where(Character.project_id == project_id)).all()
        event_counts = {}
        for char in chars:
            event_counts[char.id] = len(session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == char.id)).all())
        groups: Dict[str, List[Character]] = {}
        for char in chars:
            groups.setdefault(clean_character_key(char.name), []).append(char)

        pairs = []
        for key, members in groups.items():
            if len(members) < 2:
                continue
            stats["groups"] += 1
            target = sorted(
                members,
                key=lambda char: (
                    0 if clean_character_tag(char.name) == char.name else 1,
                    -(char.hit_count or 0),
                    -event_counts.get(char.id, 0),
                    char.id or 0,
                ))[0]
            for source in members:
                if source.id != target.id:
                    pairs.append((source.id, target.id))

    for source_id, target_id in pairs:
        if merge_character_into_target(source_id, target_id):
            stats["merged"] += 1

    with Session(engine) as session:
        chars = session.exec(
            select(Character).where(Character.project_id == project_id)).all()
        for char in chars:
            clean = clean_character_tag(char.name)
            if clean and clean != char.name:
                char.name = clean
                session.add(char)
                stats["names_normalized"] += 1
        aliases = session.exec(
            select(CharacterAlias).join(Character)
            .where(Character.project_id == project_id)).all()
        seen: Dict[int, set] = {}
        for alias in aliases:
            clean = clean_character_tag(alias.alias)
            key = clean_character_key(clean)
            char_id = alias.character_id
            if not clean:
                session.delete(alias)
                stats["aliases_removed"] += 1
                continue
            if key in seen.setdefault(char_id, set()):
                session.delete(alias)
                stats["aliases_removed"] += 1
                continue
            seen[char_id].add(key)
            if clean != alias.alias:
                alias.alias = clean
                session.add(alias)
                stats["aliases_normalized"] += 1
        session.commit()

    save_project_characters_to_json(project_id)
    try:
        # Keep the cached hit_count consistent with the merged aliases.
        from services.character_organizer import recalculate_project_character_hits
        recalculate_project_character_hits(project_id)
    except Exception:
        pass
    say(f"[Characters] merged {stats['merged']} apostrophe duplicate(s); "
        f"normalized {stats['names_normalized']} names, "
        f"{stats['aliases_normalized']} aliases; removed "
        f"{stats['aliases_removed']} duplicate aliases")
    return stats


def reset_project_characters(project_id: int):
    """
    Completely wipes all characters, aliases, book links, and timeline events for a project.
    Resets the characters.json file to an empty state.
    """
    with Session(engine) as session:
        chars = session.exec(select(Character).where(Character.project_id == project_id)).all()
        char_ids = [c.id for c in chars]

        if char_ids:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id.in_(char_ids))).all()
            for a in aliases:
                session.delete(a)

            book_links = session.exec(select(CharacterBookLink).where(CharacterBookLink.character_id.in_(char_ids))).all()
            for bl in book_links:
                session.delete(bl)

            events = session.exec(select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id.in_(char_ids))).all()
            for e in events:
                session.delete(e)

            for c in chars:
                session.delete(c)

        session.commit()

    save_project_characters_to_json(project_id)

def link_character_to_book(character_id: int, book_id: int, session: Optional[Session] = None) -> bool:
    """
    Ensures a CharacterBookLink relation exists between the character and book.
    Returns True if a new link was created, False if it already existed.
    """
    should_close = False
    if session is None:
        session = Session(engine)
        should_close = True

    try:
        existing = session.exec(
            select(CharacterBookLink)
            .where(CharacterBookLink.character_id == character_id)
            .where(CharacterBookLink.book_id == book_id)
        ).first()

        if not existing:
            link = CharacterBookLink(character_id=character_id, book_id=book_id)
            session.add(link)
            session.commit()
            return True
        return False
    finally:
        if should_close:
            session.close()


def _merge_timeline_events(session: Session, src: Character, tgt: Character,
                           target_base: Optional[CharacterTimelineEvent],
                           source_base: Optional[CharacterTimelineEvent]) -> None:
    """Merge source book events into target with target-wins conflicts.

    The target's snapshot wins at a shared position; source fields fill only the
    gaps there. At positions the target does not describe, the source's *changes*
    relative to its own previous state are applied on top of the target's current
    state. That gives "additional traits and changes from the source, target wins
    conflicts" without letting the source's base identity overwrite the target's.
    """
    books = session.exec(
        select(Book).where(Book.project_id == tgt.project_id)).all()
    order_map = {book.id: (book.book_order or 0) for book in books}

    source_events = session.exec(
        select(CharacterTimelineEvent)
        .where(CharacterTimelineEvent.character_id == src.id)
        .where(CharacterTimelineEvent.book_id.is_not(None))).all()
    target_events = session.exec(
        select(CharacterTimelineEvent)
        .where(CharacterTimelineEvent.character_id == tgt.id)
        .where(CharacterTimelineEvent.book_id.is_not(None))).all()

    def snapshot(event):
        if event is None:
            return {field: None for field in NEW_TRAIT_FIELDS}
        return {field: getattr(event, field, None) for field in NEW_TRAIT_FIELDS}

    def position(event):
        return (event.book_id, event.chapter_num, event.shot_num)

    def sort_key(pos):
        return (order_map.get(pos[0], 0), pos[1], pos[2])

    source_by_pos = {}
    for event in source_events:
        source_by_pos[position(event)] = event
    target_by_pos = {}
    for event in target_events:
        target_by_pos[position(event)] = event

    positions = sorted(set(source_by_pos) | set(target_by_pos), key=sort_key)
    target_state = snapshot(target_base)
    source_state = snapshot(source_base)

    for pos in positions:
        source_event = source_by_pos.get(pos)
        target_event = target_by_pos.get(pos)

        delta = {}
        if source_event is not None:
            event_state = snapshot(source_event)
            for field in NEW_TRAIT_FIELDS:
                value = (event_state.get(field) or "").strip()
                previous = (source_state.get(field) or "").strip()
                if value and value != previous:
                    delta[field] = event_state[field]
            source_state = event_state

        if target_event is not None:
            # Both describe this position: the target wins every conflict.
            changed = False
            if source_event is not None:
                for field in NEW_TRAIT_FIELDS:
                    if (not (getattr(target_event, field, None) or "").strip()
                            and (getattr(source_event, field, None) or "").strip()):
                        setattr(target_event, field, getattr(source_event, field))
                        changed = True
            if changed:
                session.add(target_event)
            target_state = snapshot(target_event)
            continue

        if source_event is None:
            continue

        # The target has no event here: apply the source's changes, and fill
        # fields the target still knows nothing about.
        merged = dict(target_state)
        for field in NEW_TRAIT_FIELDS:
            if field in delta:
                merged[field] = delta[field]
            elif not (merged.get(field) or "").strip():
                value = getattr(source_event, field, None)
                if value:
                    merged[field] = value
        if merged == target_state:
            continue

        session.add(CharacterTimelineEvent(
            character_id=tgt.id,
            book_id=pos[0], chapter_num=pos[1], shot_num=pos[2],
            label=source_event.label,
            source=source_event.source,
            demographics=source_event.demographics,
            physical_build=source_event.physical_build,
            hair_and_face=source_event.hair_and_face,
            distinguishing_marks=source_event.distinguishing_marks,
            **{field: merged.get(field) for field in NEW_TRAIT_FIELDS}))
        target_state = merged

    for event in source_events:
        session.delete(event)


def character_trait_summary(session: Session, character: Character,
                            book_id: Optional[int] = None
                            ) -> Dict[str, Dict[str, Any]]:
    """Base -> latest values per rendered field, for merge comparison.

    ``base`` is the canonical first-appearance value; ``latest`` is the last
    non-blank value in the chosen scope. ``changes`` is the span count minus one.
    """
    stats = character_field_stats(session, character, book_id=book_id)
    base_event = session.exec(
        select(CharacterTimelineEvent)
        .where(CharacterTimelineEvent.character_id == character.id)
        .where(CharacterTimelineEvent.book_id == None)
    ).first()
    out: Dict[str, Dict[str, Any]] = {}
    for field in RENDERED_TRAIT_FIELDS:
        base_value = ((getattr(base_event, field, None) or "").strip()
                      if base_event else "")
        spans = stats.get(field, {}).get("spans", [])
        latest_value = spans[-1]["value"] if spans else ""
        out[field] = {
            "base": base_value,
            "latest": latest_value,
            "populated": bool(spans),
            "changes": stats.get(field, {}).get("changes", 0),
        }
    return out


def merge_character_into_target(
    source_char_id: int,
    target_char_id: int,
    book_id: Optional[int] = None
) -> bool:
    """
    Reverse-merges source character into target character ("Become Alias" pattern).
    - Source name and aliases become aliases of target.
    - Source book links are reparented/merged into target.
    - Custom timeline overrides and baseline traits from source are migrated.
    - Deletes source character and syncs characters.json.
    """
    if source_char_id == target_char_id:
        return False

    with Session(engine) as session:
        src = session.get(Character, source_char_id)
        tgt = session.get(Character, target_char_id)
        if not src or not tgt:
            return False

        project_id = tgt.project_id
        src_pins = set(get_pinned_fields(src))
        tgt_pins = set(get_pinned_fields(tgt))
        if src_pins - tgt_pins:
            set_pinned_fields(session, tgt, sorted(tgt_pins | src_pins))

        # 1. Transfer source canonical name as an alias on target
        existing_tgt_aliases = {
            clean_character_key(a.alias)
            for a in session.exec(select(CharacterAlias).where(CharacterAlias.character_id == tgt.id)).all()
        }
        existing_tgt_aliases.add(clean_character_key(tgt.name))

        if clean_character_key(src.name) not in existing_tgt_aliases:
            session.add(CharacterAlias(character_id=tgt.id,
                                       alias=clean_character_tag(src.name)))
            existing_tgt_aliases.add(clean_character_key(src.name))

        # 2. Transfer source aliases to target
        src_aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == src.id)).all()
        for sa in src_aliases:
            alias_text = clean_character_tag(sa.alias)
            if clean_character_key(alias_text) not in existing_tgt_aliases:
                session.add(CharacterAlias(character_id=tgt.id, alias=alias_text))
                existing_tgt_aliases.add(clean_character_key(alias_text))
            session.delete(sa)

        # 3. Transfer and consolidate book links
        existing_tgt_links = {
            lk.book_id
            for lk in session.exec(select(CharacterBookLink).where(CharacterBookLink.character_id == tgt.id)).all()
        }

        src_links = session.exec(select(CharacterBookLink).where(CharacterBookLink.character_id == src.id)).all()
        for sl in src_links:
            if sl.book_id not in existing_tgt_links:
                session.add(CharacterBookLink(character_id=tgt.id, book_id=sl.book_id))
                existing_tgt_links.add(sl.book_id)
            session.delete(sl)

        # Ensure active book_id is linked to target
        if book_id and book_id not in existing_tgt_links:
            session.add(CharacterBookLink(character_id=tgt.id, book_id=book_id))

        # 4. Merge the Base and the timeline. The target's Base wins conflicts;
        # the source only fills empty fields. Book events then merge position by
        # position (target wins at a shared position; source changes apply where
        # the target has no event).
        src_events = session.exec(
            select(CharacterTimelineEvent)
            .where(CharacterTimelineEvent.character_id == src.id)).all()
        source_base = next((ev for ev in src_events if ev.book_id is None), None)
        target_events = session.exec(
            select(CharacterTimelineEvent)
            .where(CharacterTimelineEvent.character_id == tgt.id)
            .where(CharacterTimelineEvent.book_id.is_not(None))).all()
        target_base = session.exec(
            select(CharacterTimelineEvent)
            .where(CharacterTimelineEvent.character_id == tgt.id)
            .where(CharacterTimelineEvent.book_id == None)
        ).first()

        if target_base is None and source_base is not None:
            # Reparent the source Base rather than deleting the only canonical
            # description.
            source_base.character_id = tgt.id
            session.add(source_base)
            target_base = source_base
        elif target_base is not None and source_base is not None:
            for field in NEW_TRAIT_FIELDS:
                if (not getattr(target_base, field, None)
                        and getattr(source_base, field, None)):
                    setattr(target_base, field, getattr(source_base, field))
            if not target_base.demographics and source_base.demographics:
                target_base.demographics = source_base.demographics
            if not target_base.physical_build and source_base.physical_build:
                target_base.physical_build = source_base.physical_build
            if not target_base.hair_and_face and source_base.hair_and_face:
                target_base.hair_and_face = source_base.hair_and_face
            if (not target_base.distinguishing_marks
                    and source_base.distinguishing_marks):
                target_base.distinguishing_marks = source_base.distinguishing_marks
            if (not target_base.visual_description
                    and source_base.visual_description):
                target_base.visual_description = source_base.visual_description
            session.add(target_base)
            session.delete(source_base)

        _merge_timeline_events(session, src, tgt, target_base, source_base)

        # 5. Delete source character
        session.delete(src)
        session.commit()

        if tgt and not tgt.locked:
            if target_base and not target_base.visual_description:
                target_base.visual_description = compile_character_visual_prompt(target_base)
                session.add(target_base)
                session.commit()

    from services.character_organizer import recalculate_project_character_hits
    recalculate_project_character_hits(project_id)
    save_project_characters_to_json(project_id)
    return True

def detach_alias_to_character(
    project_id: int,
    character_id: int,
    alias_id: int,
    current_book_id: Optional[int] = None
) -> Optional[int]:
    """
    Detaches an alias from a character and restores it as a standalone Character.
    - Removes the CharacterAlias from the source character.
    - Creates a new Character with the alias name (or reuses existing if present).
    - Creates a Base State CharacterTimelineEvent for the new character.
    - Creates a self-referential CharacterAlias for the new character.
    - Links the new character to books where the alias appears, the current book,
      or inherits book links from the parent character so it never disappears from view.
    - Recalculates project character hits and syncs characters.json.
    Returns the new character ID.
    """
    from services.character_organizer import (
        get_book_character_frequency_map,
        recalculate_project_character_hits
    )

    with Session(engine) as session:
        alias_record = session.get(CharacterAlias, alias_id)
        parent_char = session.get(Character, character_id)

        if not alias_record or not parent_char or alias_record.character_id != character_id:
            return None

        alias_name = alias_record.alias.strip()

        # Disallow detaching if it matches the character's canonical name
        if alias_name.lower() == parent_char.name.lower().strip():
            return None

        # 1. Remove the alias from the parent character
        session.delete(alias_record)
        session.flush()

        # 2. Check if a character with this name already exists in the project
        existing_char = session.exec(
            select(Character)
            .where(Character.project_id == project_id)
            .where(Character.name.ilike(alias_name))
        ).first()

        if existing_char:
            new_char = existing_char
        else:
            new_char = Character(
                project_id=project_id,
                name=alias_name,
                locked=False,
                origin="Alias Detached",
                merge_checked=False
            )
            session.add(new_char)
            session.flush()

            base_ev = CharacterTimelineEvent(
                character_id=new_char.id,
                book_id=None,
                chapter_num=0,
                shot_num=0,
                label="Base State"
            )
            session.add(base_ev)
            session.flush()
            base_ev.visual_description = compile_character_visual_prompt(base_ev)
            session.add(base_ev)

            self_alias = CharacterAlias(character_id=new_char.id, alias=alias_name)
            session.add(self_alias)
            session.flush()

        # 3. Determine and assign book links so it appears in the active book view
        project = session.get(Project, project_id)
        project_name = project.name if project else ""

        books_to_link: Set[int] = set()
        project_books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        for b in project_books:
            freq_map = get_book_character_frequency_map(project_name, b.name, session)
            if freq_map.get(alias_name.lower(), 0) > 0:
                books_to_link.add(b.id)

        # Ensure the currently viewed book is linked
        if current_book_id:
            books_to_link.add(current_book_id)

        # Fallback: Inherit parent character's links if transcript frequencies are empty
        if not books_to_link:
            parent_links = session.exec(
                select(CharacterBookLink).where(CharacterBookLink.character_id == character_id)
            ).all()
            for pl in parent_links:
                books_to_link.add(pl.book_id)

        existing_links = {
            lk.book_id for lk in session.exec(
                select(CharacterBookLink).where(CharacterBookLink.character_id == new_char.id)
            ).all()
        }
        for b_id in books_to_link:
            if b_id not in existing_links:
                session.add(CharacterBookLink(character_id=new_char.id, book_id=b_id))

        session.commit()
        new_char_id = new_char.id

    recalculate_project_character_hits(project_id)
    save_project_characters_to_json(project_id)
    return new_char_id