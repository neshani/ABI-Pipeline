import re
import json
import csv
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Project, Book, Character, CharacterAlias, CharacterStateModifier
from services.prompt_engine import smart_chunk_text, get_llm_response

def get_characters_json_path(project_id: int) -> Optional[Path]:
    """Retrieves the file-as-source-of-truth characters.json target path."""
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return None
        base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
        return base_output_dir / project.name / "characters.json"


def save_project_characters_to_json(project_id: int):
    """
    Serializes all project characters, aliases, and modifiers to characters.json.
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
            modifiers = session.exec(select(CharacterStateModifier).where(CharacterStateModifier.character_id == char.id)).all()
            
            char_entry = {
                "name": char.name,
                "is_dynamic": char.is_dynamic,
                "locked": char.locked,
                "book_id": char.book_id,
                "profile": {
                    "sex_or_gender": char.sex_or_gender,
                    "approximate_age": char.approximate_age,
                    "ethnicity_or_race": char.ethnicity_or_race,
                    "height_or_stature": char.height_or_stature,
                    "weight_or_build": char.weight_or_build,
                    "hair_color_and_style": char.hair_color_and_style,
                    "facial_features": char.facial_features,
                    "distinguishing_marks": char.distinguishing_marks
                },
                "aliases": [alias.alias for alias in aliases],
                "modifiers": [
                    {
                        "name": mod.name,
                        "modifier_text": mod.modifier_text,
                        "book_id": mod.book_id,
                        "start_chapter": mod.start_chapter,
                        "end_chapter": mod.end_chapter,
                        "is_permanent": mod.is_permanent
                    }
                    for mod in modifiers
                ]
            }
            serialized_data.append(char_entry)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=2, ensure_ascii=False)


def sync_project_characters_from_json(project_id: int):
    """
    Rebuilds the SQLModel character entries from characters.json if the database was wiped.
    Maintains our strict File-as-Source-of-Truth database indexing principles.
    """
    json_path = get_characters_json_path(project_id)
    if not json_path or not json_path.exists():
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Characters] Failed to parse {json_path}: {str(e)}")
        return

    with Session(engine) as session:
        # Clear out existing SQLModel character caches for this project to perform a clean sync
        old_chars = session.exec(select(Character).where(Character.project_id == project_id)).all()
        for oc in old_chars:
            # Cleanly delete aliases via ORM
            aliases_to_del = session.exec(
                select(CharacterAlias).where(CharacterAlias.character_id == oc.id)
            ).all()
            for a in aliases_to_del:
                session.delete(a)
            
            # Cleanly delete state modifiers via ORM
            mods_to_del = session.exec(
                select(CharacterStateModifier).where(CharacterStateModifier.character_id == oc.id)
            ).all()
            for m in mods_to_del:
                session.delete(m)

            session.delete(oc)
        session.commit()

        # Reconstruct tables from file mapping
        for char_data in data:
            profile = char_data.get("profile", {})
            new_char = Character(
                project_id=project_id,
                book_id=char_data.get("book_id"),
                name=char_data["name"],
                sex_or_gender=profile.get("sex_or_gender"),
                approximate_age=profile.get("approximate_age"),
                ethnicity_or_race=profile.get("ethnicity_or_race"),
                height_or_stature=profile.get("height_or_stature"),
                weight_or_build=profile.get("weight_or_build"),
                hair_color_and_style=profile.get("hair_color_and_style"),
                facial_features=profile.get("facial_features"),
                distinguishing_marks=profile.get("distinguishing_marks"),
                is_dynamic=char_data.get("is_dynamic", False),
                locked=char_data.get("locked", False)
            )
            session.add(new_char)
            session.commit()  # commit to acquire character ID for relational attachments

            # Attach extracted aliases
            for alias_text in char_data.get("aliases", []):
                new_alias = CharacterAlias(character_id=new_char.id, alias=alias_text)
                session.add(new_alias)

            # Attach state modifiers
            for mod_data in char_data.get("modifiers", []):
                new_mod = CharacterStateModifier(
                    character_id=new_char.id,
                    book_id=mod_data["book_id"],
                    name=mod_data["name"],
                    modifier_text=mod_data["modifier_text"],
                    start_chapter=mod_data["start_chapter"],
                    end_chapter=mod_data["end_chapter"],
                    is_permanent=mod_data.get("is_permanent", False)
                )
                session.add(new_mod)
                
        session.commit()


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

        # Reconcile discovered bracketed names against existing database records
        for tag in discovered_tags:
            # Check if this name is already mapped to any alias or canonical identity
            existing_alias = session.exec(
                select(CharacterAlias).where(CharacterAlias.alias == tag)
            ).first()
            if existing_alias:
                continue

            existing_char = session.exec(
                select(Character).where(Character.project_id == project_id).where(Character.name == tag)
            ).first()
            if existing_char:
                # Associate tag as an alias automatically if missing
                new_alias = CharacterAlias(character_id=existing_char.id, alias=tag)
                session.add(new_alias)
                continue

            # Otherwise, instantiate a new unassigned Character and self-alias mapping
            new_char = Character(project_id=project_id, name=tag)
            session.add(new_char)
            session.commit()

            new_alias = CharacterAlias(character_id=new_char.id, alias=tag)
            session.add(new_alias)
            session.commit()

    # Flush changes to file backup
    save_project_characters_to_json(project_id)
    return discovered_tags


def merge_character_aliases(project_id: int, target_character_id: int, source_alias_ids: List[int]):
    """
    Merges multiple aliases into a single canonical target Character.
    Cleans up the now empty source characters to keep our database indexed and neat.
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
            
            # Re-associate alias to the new target character
            alias.character_id = target_character_id
            session.add(alias)
            session.commit()

            # If the source character now has zero remaining aliases, remove the character record
            remaining_aliases = session.exec(
                select(CharacterAlias).where(CharacterAlias.character_id == old_char_id)
            ).all()
            if not remaining_aliases:
                old_char = session.get(Character, old_char_id)
                if old_char and old_char.id != target_character_id:
                    session.delete(old_char)
                    session.commit()

    save_project_characters_to_json(project_id)


def get_character_mention_chunks(
    project_name: str, 
    book_name: str, 
    character_id: int, 
    chunk_size_words: int = 800
) -> List[Dict[str, Any]]:
    """
    Helper utility to segment book transcript.txt into discrete chunks,
    returning chunks that contain mentions of the character's mapped aliases.
    Useful for building highly targeted LLM reading context windows.
    """
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    transcript_path = base_output_dir / project_name / book_name / "transcript.txt"
    if not transcript_path.exists():
        return []

    with Session(engine) as session:
        aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == character_id)).all()
        alias_texts = {a.alias.lower() for a in aliases}

    if not alias_texts:
        return []

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()

    cleaned_text = content.replace("==CHAPTER==", " ").strip()
    all_chunks = smart_chunk_text(cleaned_text, chunk_size_words)
    
    mention_chunks = []
    for idx, chunk in enumerate(all_chunks):
        lower_chunk = chunk.lower()
        mentions_count = sum(len(re.findall(re.escape(alias), lower_chunk)) for alias in alias_texts)
        if mentions_count > 0:
            mention_chunks.append({
                "chunk_index": idx,
                "text": chunk,
                "mentions_count": mentions_count
            })

    # Sort chunks primarily by chronological occurrence
    return mention_chunks


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Bulletproof helper to extract and parse a valid JSON block out of raw LLM output,
    ignoring background commentary, descriptions, or markdown fence syntax.
    """
    # 1. Check for standard Markdown JSON block
    markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        try:
            return json.loads(markdown_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Fallback: Find the outermost curly braces
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace+1])
        except json.JSONDecodeError:
            pass

    return {}


def get_default_character_template() -> str:
    """Returns the default system instructions for character visual profiling."""
    return (
        "You are a specialized AI function that performs character visual extraction.\n"
        "Analyze the provided book text chunk and try to extract physical, observable features "
        "for the character named {character_name} (and their known aliases: {aliases}).\n\n"
        "### CURRENT KNOWN DETAILS ###\n"
        "Focus on filling the missing attributes. Do not overwrite or re-extract details already known:\n"
        "{known_traits}\n\n"
        "### LOOK FOR THESE UNKNOWN DETAILS ###\n"
        "Specifically seek information for these fields:\n"
        "{unknown_traits}\n\n"
        "### RULES ###\n"
        "1. Only extract physical visual traits directly mentioned or strongly implied by the text. "
        "Do NOT guess, hypothesize, or invent details.\n"
        "2. Do NOT extract temporary clothing, temporary actions, gestures, facial expressions "
        "(e.g., 'wrinkled her brow', 'frowned', 'smiled'), vocal attributes (e.g. 'soft voice', 'screamed'), "
        "or passing moods. Focus ONLY on stable, permanent physical properties of their body or face.\n"
        "3. Keep descriptions highly concise, descriptive, and clean (e.g., 'blue eyes', 'scar on left cheek'). "
        "Do not extract single words like 'cheeks' unless accompanied by a stable physical adjective.\n"
        "4. Your response MUST be a single, valid JSON block. Do not include conversational prefaces, "
        "explanations, or markdown commentary outside of the JSON block.\n\n"
        "### GENDER TAG RESTRICTION ###\n"
        "For the 'sex_or_gender' attribute, you MUST select EXACTLY one of these four tags if found: "
        "\"man\", \"woman\", \"boy\", \"girl\". If not explicitly clear, specify null.\n\n"
        "### JSON TARGET SCHEMA ###\n"
        "Return ONLY the fields that you have found newly described in this text chunk. "
        "Exclude any fields where no information is found. "
        "Use this strict schema:\n"
        "{{\n"
        "  \"sex_or_gender\": \"man\" | \"woman\" | \"boy\" | \"girl\" | null,\n"
        "  \"approximate_age\": \"string descriptive age range\" | null,\n"
        "  \"ethnicity_or_race\": \"string ethnicity detail\" | null,\n"
        "  \"height_or_stature\": \"string height description\" | null,\n"
        "  \"weight_or_build\": \"string weight or body build\" | null,\n"
        "  \"hair_color_and_style\": \"string hair description\" | null,\n"
        "  \"facial_features\": \"string descriptive facial traits\" | null,\n"
        "  \"distinguishing_marks\": \"string permanent distinguishing features (e.g. wears glasses, scar)\" | null\n"
        "}}\n"
    )


async def run_stateful_character_profiling(
    project_id: int, 
    character_id: int, 
    book_id: int, 
    max_chunks_to_scan: int = 5,
    progress_callback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executes the Code-Led Stateful Extraction Loop for a single character.
    Chronologically scans chunks mentioning the character's aliases.
    Presents the running state to the LLM to fill in the blanks, automatically 
    terminating early once the checklist is filled.
    """
    # Load connection parameters
    llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
    model_name = get_setting("llm_model", "local-model")
    
    # Retrieve customized template from DB if exists, fallback to default
    custom_template = get_setting("character_profiler_template", None)
    if not custom_template or str(custom_template).strip() == "":
        system_instructions_raw = get_default_character_template()
    else:
        system_instructions_raw = str(custom_template)

    with Session(engine) as session:
        project = session.get(Project, project_id)
        book = session.get(Book, book_id)
        char = session.get(Character, character_id)
        
        if not project or not book or not char:
            return {}
            
        if char.locked:
            print(f"[Profiler] Character {char.name} is locked. Skipping.")
            return {}

        project_name = project.name
        book_name = book.name
        aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
        alias_list = [a.alias for a in aliases]

    # Initialize state with any existing data in the DB to allow resuming/incremental profiles
    state_checklist = {
        "sex_or_gender": char.sex_or_gender,
        "approximate_age": char.approximate_age,
        "ethnicity_or_race": char.ethnicity_or_race,
        "height_or_stature": char.height_or_stature,
        "weight_or_build": char.weight_or_build,
        "hair_color_and_style": char.hair_color_and_style,
        "facial_features": char.facial_features,
        "distinguishing_marks": char.distinguishing_marks
    }

    # Retrieve matching chunks where the character is active
    chunks = get_character_mention_chunks(project_name, book_name, character_id, chunk_size_words=800)
    if not chunks:
        print(f"[Profiler] No mention chunks found for character: {char.name}")
        return state_checklist

    scanned_count = 0

    for chunk_data in chunks:
        if scanned_count >= max_chunks_to_scan:
            break

        # Check early exit: Is our profile checklist already completely filled?
        unknown_fields = [k for k, v in state_checklist.items() if v is None or str(v).strip() == ""]
        if not unknown_fields:
            print(f"[Profiler] Success! Checklist for {char.name} is complete. Terminating loop early.")
            break

        scanned_count += 1
        chunk_text = chunk_data["text"]

        # Build clean state checklist mapping
        known_display = "\n".join([f"- {k}: {v}" for k, v in state_checklist.items() if v]) or "None"
        unknown_display = "\n".join([f"- {k}" for k in unknown_fields])

        # Safely format instructions dynamically
        try:
            system_instructions = system_instructions_raw.format(
                character_name=char.name,
                aliases=", ".join(alias_list),
                known_traits=known_display,
                unknown_traits=unknown_display
            )
        except Exception as e:
            # Fallback if the user has custom or malformed brackets
            print(f"[Profiler] Dynamic prompt formatting error: {str(e)}")
            system_instructions = system_instructions_raw\
                .replace("{character_name}", char.name)\
                .replace("{aliases}", ", ".join(alias_list))\
                .replace("{known_traits}", known_display)\
                .replace("{unknown_traits}", unknown_display)

        user_prompt = (
            f"### CURRENT TEXT PASSAGE ###\n"
            f"\"\"\"\n{chunk_text}\n\"\"\"\n\n"
            f"Task: Review the passage. Extract facts for the Unknown details. "
            f"Respond with a single JSON block conforming to the instructions."
        )

        full_prompt = f"{system_instructions}\n\n{user_prompt}"

        try:
            print(f"[Profiler] Scanning Chunk {chunk_data['chunk_index']} for {char.name} ({scanned_count}/{max_chunks_to_scan})...")
            raw_response = await get_llm_response(full_prompt, llm_url, model_name)
            extracted_json = extract_json_from_text(raw_response)

            if extracted_json:
                print(f"[Profiler] Received new data: {extracted_json}")
                # Merge newly discovered non-null traits into our state checklist
                for key in state_checklist.keys():
                    new_val = extracted_json.get(key)
                    if new_val and str(new_val).strip() != "" and str(new_val).lower() != "null":
                        # Enforce gender category formatting restrictions
                        if key == "sex_or_gender":
                            cleaned_gender = str(new_val).lower().strip()
                            if cleaned_gender in ["man", "woman", "boy", "girl"]:
                                state_checklist[key] = cleaned_gender
                        else:
                            state_checklist[key] = str(new_val).strip()

            if progress_callback:
                # Trigger live UI callback update if attached
                progress_callback(char.id, scanned_count, max_chunks_to_scan, state_checklist)

        except Exception as e:
            print(f"[Profiler] Error during chunk scan loop: {str(e)}")

        await asyncio.sleep(0.5)

    # Save finalized checklist profile to the database
    with Session(engine) as session:
        db_char = session.get(Character, character_id)
        if db_char:
            db_char.sex_or_gender = state_checklist["sex_or_gender"]
            db_char.approximate_age = state_checklist["approximate_age"]
            db_char.ethnicity_or_race = state_checklist["ethnicity_or_race"]
            db_char.height_or_stature = state_checklist["height_or_stature"]
            db_char.weight_or_build = state_checklist["weight_or_build"]
            db_char.hair_color_and_style = state_checklist["hair_color_and_style"]
            db_char.facial_features = state_checklist["facial_features"]
            db_char.distinguishing_marks = state_checklist["distinguishing_marks"]
            session.add(db_char)
            session.commit()

    # Flush changes to disk (FaST serialization)
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
    Scans all characters in a project, computes their bracket-mention frequencies,
    and automatically merges sub-characters (like 'Detective Stone', 'Stone's') 
    into their most prominent canonical counterpart (like 'Stone').
    Utilizes difflib sequence matching, title-stripping, and substring rules.
    """
    import difflib
    from database.models import Setting
    
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return []
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        base_output_dir = Path(get_setting("output_dir", "./output")).resolve()

    # 1. Compute frequency map of raw lowercase alias text
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

        # Build alias mapping
        char_aliases = {}
        for char in characters:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
            char_aliases[char.id] = [a.alias for a in aliases]

        # Calculate frequency per character ID
        def get_char_freq(char_id):
            return sum(frequencies.get(a.lower(), 0) for a in char_aliases.get(char_id, []))

        # Sort descending by total frequency (canonical target first)
        sorted_chars = sorted(characters, key=lambda c: get_char_freq(c.id), reverse=True)

        merged_log = []
        merged_ids = set()

        # Normalization and stripping rules (titles & possessives)
        titles = [
            "detective", "officer", "agent", "captain", "mr", "mrs", "ms", "dr", 
            "doctor", "professor", "lieutenant", "sergeant", "colonel", "general", 
            "sheriff", "deputy", "chief", "father", "aunt", "uncle", "miss"
        ]
        
        def normalize(name_str: str) -> str:
            val = name_str.lower().strip()
            if val.endswith("'s"):
                val = val[:-2].strip()
            if val.endswith("’s"):
                val = val[:-2].strip()
            for t in titles:
                if val.startswith(t + " "):
                    val = val[len(t) + 1:].strip()
                elif val.startswith(t + "."):
                    val = val[len(t) + 1:].strip()
            return val

        # Greedy merge matching pass
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

                        # Exact normalized match (e.g. "Detective Stone" -> "Stone" after stripping)
                        if target_norm == cand_norm:
                            is_match = True
                            match_reason = f"Title/Possessive Normalization"
                            break

                        # Substring containment (e.g., "Stone" contained in "Detective Stone")
                        if len(target_norm) >= 4 and len(cand_norm) >= 4:
                            if target_norm in cand_norm or cand_norm in target_norm:
                                is_match = True
                                match_reason = f"Substring Containment"
                                break

                        # Fuzzy ratio sequence comparison
                        if len(target_norm) >= 4 and len(cand_norm) >= 4:
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
                    
                    # Associate candidate name itself as an alias on target if not present
                    if candidate_char.name.lower() not in existing_aliases_on_target:
                        new_alias = CharacterAlias(character_id=target_char.id, alias=candidate_char.name)
                        session.add(new_alias)
                        target_aliases.append(candidate_char.name)

                    # Re-associate other aliases to the target character
                    for alias in cand_aliases_db:
                        if alias.alias.lower() not in existing_aliases_on_target:
                            alias.character_id = target_char.id
                            session.add(alias)
                            target_aliases.append(alias.alias)
                        else:
                            session.delete(alias)

                    # Re-associate any active modifiers
                    cand_mods = session.exec(
                        select(CharacterStateModifier).where(CharacterStateModifier.character_id == candidate_char.id)
                    ).all()
                    for mod in cand_mods:
                        mod.character_id = target_char.id
                        session.add(mod)

                    session.delete(candidate_char)
                    session.commit()
                    merged_ids.add(candidate_char.id)

    # Re-save finalized JSON file state
    save_project_characters_to_json(project_id)
    return merged_log