import re
import json
import csv
import asyncio
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Project, Book, Character, CharacterAlias, CharacterStateModifier, CharacterTimelineEvent
from services.prompt_engine import smart_chunk_text, get_llm_response

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
    """
    Self-heals book_order fields for books within a project,
    sorting alphabetically by name as the natural default order.
    """
    with Session(engine) as session:
        books = session.exec(
            select(Book).where(Book.project_id == project_id).order_by(Book.name)
        ).all()
        
        changed = False
        for idx, book in enumerate(books):
            if book.book_order is None or book.book_order != idx:
                book.book_order = idx
                session.add(book)
                changed = True
        if changed:
            session.commit()


def compile_character_visual_prompt(obj) -> str:
    """
    Assembles a descriptive, natural language physical prompt from structured traits.
    Optimized for single-stream text encoders. Accepts either CharacterTimelineEvent or a legacy Character object.
    """
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
    Serializes all project characters, aliases, and timeline override events to characters.json.
    Ensures that manual edits and LLM descriptions are always safely preserved on disk.
    """
    json_path = get_characters_json_path(project_id)
    if not json_path:
        return

    with Session(engine) as session:
        # Pull all characters belonging to the project
        characters = session.exec(select(Character).where(Character.project_id == project_id)).all()
        
        serialized_data = []
        for char in characters:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
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
                    "scene_num": ev.scene_num,
                    "label": ev.label,
                    "visual_description": ev.visual_description,
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
                "aliases": [alias.alias for alias in aliases],
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
    Rebuilds the SQLModel character entries and timeline overrides from characters.json if the database was wiped.
    Maintains our strict File-as-Source-of-Truth database indexing principles.
    """
    # Resolve the path using the shared session
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
        # Clear out existing SQLModel character caches for this project to perform a clean sync
        old_chars = session.exec(select(Character).where(Character.project_id == project_id)).all()
        for oc in old_chars:
            aliases_to_del = session.exec(
                select(CharacterAlias).where(CharacterAlias.character_id == oc.id)
            ).all()
            for a in aliases_to_del:
                session.delete(a)
            
            evs_to_del = session.exec(
                select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id == oc.id)
            ).all()
            for ev in evs_to_del:
                session.delete(ev)

            session.delete(oc)
        session.flush()

        # Reconstruct tables from file mapping
        for char_data in data:
            new_char = Character(
                project_id=project_id,
                name=char_data["name"],
                locked=char_data.get("locked", False)
            )
            session.add(new_char)
            session.flush()  # flush to acquire character ID for relational attachments

            # Attach extracted aliases
            for alias_text in char_data.get("aliases", []):
                new_alias = CharacterAlias(character_id=new_char.id, alias=alias_text)
                session.add(new_alias)

            # Rebuild Timeline Events
            timeline_list = char_data.get("timeline", [])
            if not timeline_list:
                # Backward compatibility fallback
                legacy_profile = char_data.get("profile") or {}
                base_ev = CharacterTimelineEvent(
                    character_id=new_char.id,
                    book_id=None,
                    chapter_num=0,
                    scene_num=0,
                    label="Base State",
                    demographics=legacy_profile.get("demographics"),
                    physical_build=legacy_profile.get("physical_build"),
                    hair_and_face=legacy_profile.get("hair_and_face"),
                    distinguishing_marks=legacy_profile.get("distinguishing_marks"),
                    visual_description=char_data.get("visual_description")
                )
                session.add(base_ev)
            else:
                for ev_data in timeline_list:
                    book_name = ev_data.get("book_name")
                    b_id = None
                    if book_name:
                        b = session.exec(
                            select(Book)
                            .where(Book.project_id == project_id)
                            .where(Book.name == book_name)
                        ).first()
                        if b:
                            b_id = b.id
                    
                    prof = ev_data.get("profile", {})
                    new_ev = CharacterTimelineEvent(
                        character_id=new_char.id,
                        book_id=b_id,
                        chapter_num=ev_data.get("chapter_num", 0),
                        scene_num=ev_data.get("scene_num", 0),
                        label=ev_data.get("label", "Base State" if b_id is None else "Override State"),
                        demographics=prof.get("demographics"),
                        physical_build=prof.get("physical_build"),
                        hair_and_face=prof.get("hair_and_face"),
                        distinguishing_marks=prof.get("distinguishing_marks"),
                        visual_description=ev_data.get("visual_description")
                    )
                    session.add(new_ev)
                    
        session.flush()
    finally:
        if should_close:
            session.commit()
            session.close()


def extract_characters_from_prompts(project_id: int) -> Set[str]:
    """
    Scans the prompts.csv file of every book in the project, looking for bracketed names [Dino].
    Automatically indexes them in the database and saves them to characters.json.
    """
    discovered_tags: Set[str] = set()
    bracket_regex = re.compile(r"\[(.*?)\]")

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return discovered_tags

        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        base_output_dir = Path(get_setting("output_dir", "./output")).resolve()

        for book in books:
            csv_path = base_output_dir / project.name / book.name / "prompts.csv"
            if not csv_path.exists():
                continue

            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="|")
                    for row in reader:
                        prompt_text = row.get("prompt", "")
                        for match in bracket_regex.findall(prompt_text):
                            clean_tag = match.strip()
                            if clean_tag:
                                discovered_tags.add(clean_tag)
            except Exception as e:
                print(f"[Characters] Error reading prompt CSV for {book.name}: {str(e)}")

        if not discovered_tags:
            return discovered_tags

        for tag in discovered_tags:
            existing_alias = session.exec(
                select(CharacterAlias)
                .join(Character)
                .where(CharacterAlias.alias == tag)
                .where(Character.project_id == project_id)
            ).first()
            if existing_alias:
                continue

            existing_char = session.exec(
                select(Character).where(Character.project_id == project_id).where(Character.name == tag)
            ).first()
            if existing_char:
                new_alias = CharacterAlias(character_id=existing_char.id, alias=tag)
                session.add(new_alias)
                continue

            new_char = Character(project_id=project_id, name=tag)
            session.add(new_char)
            session.commit()

            # Ensure Base State timeline event is created automatically!
            base_ev = CharacterTimelineEvent(
                character_id=new_char.id,
                book_id=None,
                chapter_num=0,
                scene_num=0,
                label="Base State"
            )
            session.add(base_ev)
            session.commit()

            new_alias = CharacterAlias(character_id=new_char.id, alias=tag)
            session.add(new_alias)
            session.commit()

    save_project_characters_to_json(project_id)
    return discovered_tags


def merge_character_aliases(project_id: int, target_character_id: int, source_alias_ids: List[int]):
    """
    Merges multiple aliases into a single canonical target Character.
    Moves chronological timeline events to target character, cleans up empty source characters.
    """
    with Session(engine) as session:
        target_char = session.get(Character, target_character_id)
        if not target_char:
            return

        for alias_id in source_alias_ids:
            alias = session.get(CharacterAlias, alias_id)
            if not alias:
                continue

            old_char_id = alias.character_id
            
            alias.character_id = target_character_id
            session.add(alias)
            session.commit()

            remaining_aliases = session.exec(
                select(CharacterAlias).where(CharacterAlias.character_id == old_char_id)
            ).all()
            if not remaining_aliases:
                old_char = session.get(Character, old_char_id)
                if old_char and old_char.id != target_character_id:
                    # Merge timeline override events
                    cand_evs = session.exec(
                        select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id == old_char_id)
                    ).all()
                    for ev in cand_evs:
                        if ev.book_id is None:
                            # Recombine legacy traits into target's Base State if blank, then discard
                            target_base = session.exec(
                                select(CharacterTimelineEvent)
                                .where(CharacterTimelineEvent.character_id == target_character_id)
                                .where(CharacterTimelineEvent.book_id == None)
                            ).first()
                            if target_base:
                                if not target_base.demographics: target_base.demographics = ev.demographics
                                if not target_base.physical_build: target_base.physical_build = ev.physical_build
                                if not target_base.hair_and_face: target_base.hair_and_face = ev.hair_and_face
                                if not target_base.distinguishing_marks: target_base.distinguishing_marks = ev.distinguishing_marks
                                session.add(target_base)
                            session.delete(ev)
                        else:
                            ev.character_id = target_character_id
                            session.add(ev)
                    
                    session.delete(old_char)
                    session.commit()

        # Update target base's visual description if unlocked
        if target_char and not target_char.locked:
            base_ev = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == target_character_id)
                .where(CharacterTimelineEvent.book_id == None)
            ).first()
            if base_ev:
                base_ev.visual_description = compile_character_visual_prompt(base_ev)
                session.add(base_ev)
                session.commit()

    save_project_characters_to_json(project_id)


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
    Scans prompts.csv files dynamically to return a mapping of Book Name -> Mention Count
    for all aliases belonging to the given character across the project.
    """
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    book_mentions = {}
    
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return {}
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == character_id)).all()
        alias_texts = {a.alias.lower() for a in aliases}

    if not alias_texts:
        return {}

    bracket_regex = re.compile(r"\[(.*?)\]")
    for book in books:
        csv_path = base_output_dir / project.name / book.name / "prompts.csv"
        if not csv_path.exists():
            continue
        count = 0
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="|")
                for row in reader:
                    prompt_text = row.get("prompt", "")
                    for match in bracket_regex.findall(prompt_text):
                        if match.strip().lower() in alias_texts:
                            count += 1
        except Exception:
            pass
        if count > 0:
            book_mentions[book.name] = count
            
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
    
    factual_template_raw = get_default_character_template()

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
                    scene_num=0,
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
            speculative_template_raw = get_speculative_character_template()
            
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


def auto_merge_project_characters(project_id: int, similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    Scans all characters in a project, and automatically merges sub-characters
    into their most prominent canonical counterpart based on titles and possessives.
    """
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return []
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        base_output_dir = Path(get_setting("output_dir", "./output")).resolve()

    frequencies = {}
    bracket_regex = re.compile(r"\[(.*?)\]")
    for b in books:
        csv_path = base_output_dir / project.name / b.name / "prompts.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="|")
                for row in reader:
                    prompt_text = row.get("prompt", "")
                    for match in bracket_regex.findall(prompt_text):
                        clean_tag = match.strip().lower()
                        frequencies[clean_tag] = frequencies.get(clean_tag, 0) + 1
        except Exception:
            pass

    with Session(engine) as session:
        characters = session.exec(select(Character).where(Character.project_id == project_id)).all()
        if not characters:
            return []

        char_aliases = {}
        for char in characters:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
            char_aliases[char.id] = [a.alias for a in aliases]

        def get_char_freq(char_id):
            return sum(frequencies.get(a.lower(), 0) for a in char_aliases.get(char_id, []))

        sorted_chars = sorted(characters, key=lambda c: get_char_freq(c.id), reverse=True)

        merged_log = []
        merged_ids = set()

        titles = [
            "detective", "officer", "agent", "captain", "mr", "mrs", "ms", "dr", 
            "doctor", "professor", "lieutenant", "sergeant", "colonel", "general", 
            "sheriff", "deputy", "chief", "father", "aunt", "uncle", "miss"
        ]
        
        def normalize(name_str: str) -> str:
            val = name_str.lower().strip()
            if val.endswith("'s") or val.endswith("’s"):
                val = val[:-2].strip()
            for t in titles:
                if val.startswith(t + " ") or val.startswith(t + "."):
                    val = val[len(t) + 1:].strip()
            return val

        def is_word_contained(short_name: str, long_name: str) -> bool:
            # Word-boundary check prevents substring matches inside whole words (e.g. "arrington" in "barrington")
            pattern = rf"\b{re.escape(short_name)}\b"
            return bool(re.search(pattern, long_name))

        for i, target_char in enumerate(sorted_chars):
            if target_char.id in merged_ids:
                continue

            target_aliases = char_aliases.get(target_char.id, [])
            all_target_texts = set(target_aliases + [target_char.name])
            normalized_target_texts = {normalize(t) for t in all_target_texts if t}

            for j in range(i + 1, len(sorted_chars)):
                candidate_char = sorted_chars[j]
                if candidate_char.id in merged_ids or candidate_char.id == target_char.id:
                    continue

                candidate_aliases = char_aliases.get(candidate_char.id, [])
                all_candidate_texts = set(candidate_aliases + [candidate_char.name])
                normalized_candidate_texts = {normalize(c) for c in all_candidate_texts if c}

                is_match = False
                match_reason = ""

                for target_norm in normalized_target_texts:
                    if not target_norm:
                        continue
                    for cand_norm in normalized_candidate_texts:
                        if not cand_norm:
                            continue

                        # 1. Exact match after normalization
                        if target_norm == cand_norm:
                            is_match = True
                            match_reason = "Title/Possessive Normalization"
                            break

                        # Determine smaller and larger profiles
                        short_n, long_n = (target_norm, cand_norm) if len(target_norm) < len(cand_norm) else (cand_norm, target_norm)

                        # 2. Smart Word/Token Containment (requires length >= 3)
                        if len(short_n) >= 3:
                            if is_word_contained(short_n, long_n):
                                is_match = True
                                match_reason = f"Word containment ('{short_n}' in '{long_n}')"
                                break

                        # 3. Fuzzy Similarity Check
                        if len(target_norm) >= 4 and len(cand_norm) >= 4:
                            target_words = target_norm.split()
                            cand_words = cand_norm.split()
                            
                            # Filter out short single-word differences that start with different letters (Molly/Polly, Lance/Vance)
                            if len(target_words) == 1 and len(cand_words) == 1:
                                if len(target_norm) <= 6 and len(cand_norm) <= 6:
                                    if target_norm[0] != cand_norm[0]:
                                        continue
                                        
                            ratio = difflib.SequenceMatcher(None, target_norm, cand_norm).ratio()
                            if ratio >= similarity_threshold:
                                is_match = True
                                match_reason = f"Fuzzy similarity ({int(ratio*100)}%)"
                                break
                    if is_match:
                        break

                if is_match:
                    cand_aliases_db = session.exec(
                        select(CharacterAlias).where(CharacterAlias.character_id == candidate_char.id)
                    ).all()
                    
                    merged_log.append({
                        "target_name": target_char.name,
                        "merged_name": candidate_char.name,
                        "reason": match_reason,
                        "aliases_added": [a.alias for a in cand_aliases_db]
                    })

                    existing_aliases_on_target = {a.lower() for a in target_aliases}
                    candidates_to_add = {candidate_char.name.lower()}
                    for alias in cand_aliases_db:
                        candidates_to_add.add(alias.alias.lower())

                    new_aliases_to_create = candidates_to_add - existing_aliases_on_target

                    for alias in cand_aliases_db:
                        session.delete(alias)

                    for new_alias_text in new_aliases_to_create:
                        original_case = candidate_char.name
                        if candidate_char.name.lower() != new_alias_text:
                            for alias in cand_aliases_db:
                                if alias.alias.lower() == new_alias_text:
                                    original_case = alias.alias
                                    break
                        
                        new_alias_obj = CharacterAlias(character_id=target_char.id, alias=original_case)
                        session.add(new_alias_obj)
                        target_aliases.append(original_case)

                    # Reparent custom override events
                    cand_evs = session.exec(
                        select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id == candidate_char.id)
                    ).all()
                    for ev in cand_evs:
                        if ev.book_id is None:
                            # Merge legacy profiles into target baseline
                            target_base = session.exec(
                                select(CharacterTimelineEvent)
                                .where(CharacterTimelineEvent.character_id == target_char.id)
                                .where(CharacterTimelineEvent.book_id == None)
                            ).first()
                            if target_base:
                                if not target_base.demographics: target_base.demographics = ev.demographics
                                if not target_base.physical_build: target_base.physical_build = ev.physical_build
                                if not target_base.hair_and_face: target_base.hair_and_face = ev.hair_and_face
                                if not target_base.distinguishing_marks: target_base.distinguishing_marks = ev.distinguishing_marks
                                session.add(target_base)
                            session.delete(ev)
                        else:
                            ev.character_id = target_char.id
                            session.add(ev)

                    session.delete(candidate_char)
                    session.commit()
                    merged_ids.add(candidate_char.id)

            # Re-compile target visual description of baseline state if unlocked
            if target_char.id not in merged_ids and not target_char.locked:
                base_ev = session.exec(
                    select(CharacterTimelineEvent)
                    .where(CharacterTimelineEvent.character_id == target_char.id)
                    .where(CharacterTimelineEvent.book_id == None)
                ).first()
                if base_ev:
                    base_ev.visual_description = compile_character_visual_prompt(base_ev)
                    session.add(base_ev)
                    session.commit()

    save_project_characters_to_json(project_id)
    return merged_log


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
                scene_num=0,
                label="Base State"
            )
            session.add(base_ev)
            session.commit()
            session.refresh(base_ev)
    return compile_character_description_from_event(char, base_ev, enabled_fields, use_sentence_structure)


def replace_character_tags_in_prompt(
    prompt: str, 
    project_id: int, 
    enabled_fields: Dict[str, bool], 
    use_sentence_structure: bool,
    book_id: Optional[int] = None,
    chapter_num: Optional[int] = None,
    scene_num: Optional[int] = None
) -> str:
    """
    Scans a prompt string for bracketed tags, resolves timeline event descriptions chronologically
    based on coordinates, and returns a modified prompt string containing compiled descriptions.
    """
    bracket_regex = re.compile(r"\[(.*?)\]")
    matches = bracket_regex.findall(prompt)
    if not matches:
        return prompt

    modified_prompt = prompt
    expanded_character_ids = set()
    
    with Session(engine) as session:
        session.expire_all()
        
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        book_order_map = {b.id: (b.book_order or 0) for b in books}
        target_book_order = book_order_map.get(book_id, 0) if book_id else 0

        for match in matches:
            tag = match.strip()
            alias = session.exec(
                select(CharacterAlias)
                .join(Character)
                .where(CharacterAlias.alias == tag)
                .where(Character.project_id == project_id)
            ).first()
            
            if not alias:
                char = session.exec(
                    select(Character)
                    .where(Character.project_id == project_id)
                    .where(Character.name == tag)
                ).first()
            else:
                char = session.get(Character, alias.character_id)

            if char:
                try:
                    session.refresh(char)
                except Exception:
                    pass

                if char.id in expanded_character_ids:
                    replacement = char.name
                else:
                    # Resolve active timeline event!
                    events = session.exec(
                        select(CharacterTimelineEvent)
                        .where(CharacterTimelineEvent.character_id == char.id)
                    ).all()
                    
                    base_ev = None
                    matched_evs = []
                    for ev in events:
                        if ev.book_id is None:
                            base_ev = ev
                            continue
                        
                        ev_order = book_order_map.get(ev.book_id, 0)
                        
                        if ev_order < target_book_order:
                            matched_evs.append((ev, ev_order))
                        elif ev_order == target_book_order:
                            if chapter_num is not None and ev.chapter_num < chapter_num:
                                matched_evs.append((ev, ev_order))
                            elif chapter_num is not None and ev.chapter_num == chapter_num:
                                if scene_num is not None and ev.scene_num <= scene_num:
                                    matched_evs.append((ev, ev_order))
                    
                    resolved_ev = base_ev
                    if matched_evs:
                        matched_evs.sort(key=lambda x: (x[1], x[0].chapter_num, x[0].scene_num))
                        resolved_ev = matched_evs[-1][0]
                    
                    if not resolved_ev:
                        resolved_ev = CharacterTimelineEvent(
                            character_id=char.id,
                            book_id=None,
                            chapter_num=0,
                            scene_num=0,
                            label="Base State"
                        )
                        session.add(resolved_ev)
                        session.commit()
                        session.refresh(resolved_ev)
                        
                    replacement = compile_character_description_from_event(
                        char, resolved_ev, enabled_fields, use_sentence_structure
                    )
                    expanded_character_ids.add(char.id)
                
                modified_prompt = modified_prompt.replace(f"[{tag}]", replacement, 1)
            else:
                modified_prompt = modified_prompt.replace(f"[{tag}]", tag, 1)
                
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

def get_character_frequency_map_db(project_name: str, session: Session) -> Dict[str, int]:
    """Scans prompts.csv files to build a map of bracket tag occurrences for a project."""
    frequencies = {}
    bracket_regex = re.compile(r"\[(.*?)\]")
    base_output_dir = Path(get_setting("output_dir", "./output", session)).resolve()
    
    proj = session.exec(select(Project).where(Project.name == project_name)).first()
    if not proj:
        return frequencies
        
    books = session.exec(select(Book).where(Book.project_id == proj.id)).all()
    for b in books:
        csv_path = base_output_dir / project_name / b.name / "prompts.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="|")
                for row in reader:
                    prompt_text = row.get("prompt", "")
                    for match in bracket_regex.findall(prompt_text):
                        clean_tag = match.strip().lower()
                        frequencies[clean_tag] = frequencies.get(clean_tag, 0) + 1
        except Exception:
            pass
    return frequencies


def get_character_import_matches(tgt_project_id: int, src_project_id: int) -> List[Dict[str, Any]]:
    """
    Finds pairings between target project unlocked characters and source project characters.
    Matches if canonical name matches or if aliases intersect.
    Filters out source characters with empty descriptions or fallback "a person named X" descriptions.
    Ensures each source character is matched with at most one target character (best fit).
    Sorts matches descending by target mentions count so important characters are at the top.
    """
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
                    return (tgt_book_order_map.get(ev.book_id, 0), ev.chapter_num, ev.scene_num, ev.id or 0)
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
                return (src_book_order_map.get(ev.book_id, 0), ev.chapter_num, ev.scene_num, ev.id or 0)
            
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
                    return (src_book_order_map.get(ev.book_id, 0), ev.chapter_num, ev.scene_num, ev.id or 0)
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
                    scene_num=0,
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


def reset_project_characters(project_id: int):
    """
    Completely wipes all characters, aliases, and timeline events for a project.
    Resets the characters.json file to an empty state.
    """
    with Session(engine) as session:
        # 1. Find all characters belonging to this project
        chars = session.exec(select(Character).where(Character.project_id == project_id)).all()
        char_ids = [c.id for c in chars]

        if char_ids:
            # 2. Delete Relational Data first
            # Aliases
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id.in_(char_ids))).all()
            for a in aliases:
                session.delete(a)

            # Timeline Events
            events = session.exec(select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id.in_(char_ids))).all()
            for e in events:
                session.delete(e)
            
            # Modifiers (if any)
            modifiers = session.exec(select(CharacterStateModifier).where(CharacterStateModifier.character_id.in_(char_ids))).all()
            for m in modifiers:
                session.delete(m)

            # 3. Delete Characters
            for c in chars:
                session.delete(c)

        session.commit()

    # 4. Sync the now-empty state to the characters.json file
    save_project_characters_to_json(project_id)