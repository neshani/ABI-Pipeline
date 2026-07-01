import re
import json
import csv
import asyncio
import difflib
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


def compile_character_visual_prompt(char: Character) -> str:
    """
    Assembles a descriptive, natural language physical prompt from simplified structured traits.
    Optimized for single-stream text encoders like Qwen (Z-Image Turbo).
    """
    pieces = []
    
    if char.demographics and str(char.demographics).strip():
        pieces.append(char.demographics.strip())
    if char.hair_and_face and str(char.hair_and_face).strip():
        pieces.append(char.hair_and_face.strip())
    if char.physical_build and str(char.physical_build).strip():
        pieces.append(char.physical_build.strip())
    if char.distinguishing_marks and str(char.distinguishing_marks).strip():
        pieces.append(char.distinguishing_marks.strip())

    cleaned_pieces = [p.strip() for p in pieces if p and str(p).strip()]
    if not cleaned_pieces:
        return f"a person named {char.name}"
        
    return ", ".join(cleaned_pieces)


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
                "visual_description": char.visual_description,
                "profile": {
                    "demographics": char.demographics,
                    "physical_build": char.physical_build,
                    "hair_and_face": char.hair_and_face,
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
            aliases_to_del = session.exec(
                select(CharacterAlias).where(CharacterAlias.character_id == oc.id)
            ).all()
            for a in aliases_to_del:
                session.delete(a)
            
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
                demographics=profile.get("demographics"),
                physical_build=profile.get("physical_build"),
                hair_and_face=profile.get("hair_and_face"),
                distinguishing_marks=profile.get("distinguishing_marks"),
                visual_description=char_data.get("visual_description"),
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

        for tag in discovered_tags:
            # Fixed: Join with Character table to check if the alias exists in the current project scope
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

            new_alias = CharacterAlias(character_id=new_char.id, alias=tag)
            session.add(new_alias)
            session.commit()

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
            
            alias.character_id = target_character_id
            session.add(alias)
            session.commit()

            remaining_aliases = session.exec(
                select(CharacterAlias).where(CharacterAlias.character_id == old_char_id)
            ).all()
            if not remaining_aliases:
                old_char = session.get(Character, old_char_id)
                if old_char and old_char.id != target_character_id:
                    session.delete(old_char)
                    session.commit()

        # Update target's visual description if unlocked
        if target_char and not target_char.locked:
            target_char.visual_description = compile_character_visual_prompt(target_char)
            session.add(target_char)
            session.commit()

    save_project_characters_to_json(project_id)


def get_character_mention_chunks(
    project_id: int,
    character_id: int,
    book_id: Optional[int] = None,
    chunk_size_words: int = 800
) -> List[Dict[str, Any]]:
    """
    Retrieves chunks of transcript.txt containing character aliases.
    If book_id is provided, limits scan to that book.
    If book_id is None, scans all books in the project, returning them in chronological order.
    """
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return []
            
        aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == character_id)).all()
        alias_texts = {a.alias.lower() for a in aliases}
        if not alias_texts:
            return []

        if book_id:
            books = [session.get(Book, book_id)]
        else:
            # Query all books inside the project, ordered chronologically
            books = session.exec(
                select(Book).where(Book.project_id == project_id).order_by(Book.id)
            ).all()

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

        cleaned_text = content.replace("==CHAPTER==", " ").strip()
        all_chunks = smart_chunk_text(cleaned_text, chunk_size_words)
        
        for idx, chunk in enumerate(all_chunks):
            lower_chunk = chunk.lower()
            mentions_count = sum(len(re.findall(re.escape(alias), lower_chunk)) for alias in alias_texts)
            if mentions_count > 0:
                mention_chunks.append({
                    "book_id": book.id,
                    "book_name": book.name,
                    "chunk_index": idx,
                    "text": chunk,
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


def get_default_character_template() -> str:
    """Returns default visual profiling instructions emphasizing paintable concrete details and banning narrative plots."""
    return (
        "You are a strict, objective AI character profiler. Extract physical features for {character_name} "
        "(aka: {aliases}) from the provided book passage.\n\n"
        "### TARGET SENTENCE SCHEMA ###\n"
        "We inject your output into this exact template:\n"
        "\"{character_name} (a {{demographics}}, {{hair_and_face}}, who is {{physical_build}}, and {{distinguishing_marks}})\"\n\n"
        "Your JSON values must be short, lowercase grammatical fragments:\n"
        "- 'demographics': Noun phrase of age, race, gender (NO articles). E.g., 'middle-aged Caucasian man', 'young Italian woman'.\n"
        "- 'hair_and_face': Prepositional phrase starting with 'with'. E.g., 'with thinning brown hair', 'with sharp blue eyes'.\n"
        "- 'physical_build': Height, posture, and build. E.g., 'tall and athletic', 'short and stocky'.\n"
        "- 'distinguishing_marks': Permanent details only (tattoos, scars, glasses). E.g., 'with a scar on his cheek'.\n\n"
        "### CRITICAL RESTRICTIONS (STRICTLY ENFORCED) ###\n"
        "1. NO CLOTHING: Do not extract suits, jackets, raincoats, hats, or attire. The profile must be entirely clothing-free.\n"
        "2. NO TRANSIENT GESTURES/EXPRESSIONS: Ignore voice, sounds, smiles, frowns, raised eyebrows, jaw drops, parted lips, glances, or momentary physical movements. Focus on stable, lifelong features only.\n"
        "3. ENTITY SHIELD: Often the text describes a perp, suspect, bystander, or corpse (e.g., 'a male Caucasian 5'6\"' or 'the doctor at the bar') while {character_name} reacts or speaks. Do NOT extract these! Only extract traits if they explicitly describe {character_name}.\n"
        "4. ONLY PAINTABLE VISUAL DETAILS: Your extractions must describe direct, concrete physical colors, textures, shapes, and tangible sizes (e.g., 'blonde hair', 'sharp green eyes'). Do NOT extract narrative, abstract, plot-heavy, or relational facts (e.g., 'hair color matching a Jane Doe', 'looked like her mother', 'with a face known to police'). If an artist cannot physically paint it, it is strictly forbidden.\n"
        "5. Output MUST be a single, valid JSON block. No commentary.\n\n"
        "### CURRENT PROFILE STATE ###\n"
        "Currently recorded:\n"
        "{known_traits}\n"
        "Unknown (needs data):\n"
        "{unknown_traits}\n\n"
        "### INSTRUCTIONS ###\n"
        "Fill missing data or correct old provisional traits ONLY if this text passage clearly and explicitly contradicts them with authoritative evidence. "
        "Do not output unchanged fields.\n\n"
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
    Determines if a newly extracted trait value is a valid permanent visual descriptor,
    preventing transient expressions, auditory traits, actions, or overly generic single words.
    """
    val = new_val.lower().strip()
    if not val or val == "null" or val == "none":
        return False
        
    # 1. Surgical ban list targeting ONLY non-visual traits or highly transient action-modifiers
    banned_terms = [
        # Auditory/Vocal (Strictly non-visual)
        "voice", "sound", "accent", "tone", "shout", "whisper", "screamed", "spoken", "spoke", "screaming",
        
        # Pure momentary facial expressions
        "smile", "grin", "frown", "scowl", "pout", "smirk", "laugh", "giggle", "chuckle",
        "twitching", "winking", "blinking", "crying", "tears", "shivering", "shivered", "recoiled",
        
        # Transient states of permanent features (allows 'eyebrows', 'lips', 'jaw', 'teeth' to be permanent)
        "raised eyebrow", "raised eyebrows", "furrowed", "dropped jaw", "parted lip", "parted lips",
        "gritting teeth", "gnashing", "biting lip", "chewing lip",
        
        # Transitive physical action verbs (prevents literal plot interactions from becoming traits)
        "clap", "clapped", "clapping", "slap", "slapped", "slapping", "grab", "grabbed", "grabbing", 
        "hold", "held", "holding", "press", "pressed", "pressing", "touch", "touched", "touching",

        "unknown", "not specified", "unspecified", "unmentioned", "not mentioned", "not described"
    ]
    
    for term in banned_terms:
        if re.search(rf"\b{term}", val):
            print(f"[Profiler Filter] Discarding transient/action/auditory term '{term}' in: '{new_val}'")
            return False
            
    # 2. Block ultra-generic single-word filler overwrites (e.g. overwriting 'six-foot-two and athletic' with just 'tall')
    generic_words = ["tall", "short", "thin", "fat", "man", "woman", "boy", "girl", "hair", "face"]
    if val in generic_words and old_val and len(old_val.strip()) > 15:
        print(f"[Profiler Filter] Discarding generic single-word update '{new_val}' over descriptive: '{old_val}'")
        return False
        
    return True


async def run_stateful_character_profiling(
    project_id: int, 
    character_id: int, 
    book_id: Optional[int] = None, 
    max_chunks_to_scan: int = 5,
    progress_callback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executes the Code-Led Stateful Extraction Loop for a single character.
    Chronologically scans chunks mentioning the character's aliases from book transcripts.
    """
    llm_url = get_setting("llm_url", "http://127.0.0.1:11434")
    model_name = get_setting("llm_model", "local-model")
    
    custom_template = get_setting("character_profiler_template", None)
    if not custom_template or str(custom_template).strip() == "":
        system_instructions_raw = get_default_character_template()
    else:
        system_instructions_raw = str(custom_template)

    with Session(engine) as session:
        project = session.get(Project, project_id)
        char = session.get(Character, character_id)
        
        if not project or not char:
            return {}
            
        if char.locked:
            print(f"[Profiler] Character {char.name} is locked. Skipping.")
            return {}

        # WIPE OLD AUTO-GENERATED TRAITS FOR A CLEAN SLATE PASS
        char.demographics = None
        char.physical_build = None
        char.hair_and_face = None
        char.distinguishing_marks = None
        char.visual_description = None
        session.add(char)
        session.commit()

        aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
        alias_list = [a.alias for a in aliases]

    # Initialize running state with a clean slate
    state_checklist = {
        "demographics": None,
        "physical_build": None,
        "hair_and_face": None,
        "distinguishing_marks": None
    }

    chunks = get_character_mention_chunks(project_id, character_id, book_id, chunk_size_words=800)
    if not chunks:
        print(f"[Profiler] No mention chunks found for character: {char.name}")
        return state_checklist

    scanned_count = 0

    for chunk_data in chunks:
        if scanned_count >= max_chunks_to_scan:
            break

        unknown_fields = [k for k, v in state_checklist.items() if v is None or str(v).strip() == ""]
        scanned_count += 1
        chunk_text = chunk_data["text"]

        known_display = "\n".join([f"- {k}: {v}" for k, v in state_checklist.items() if v]) or "None"
        unknown_display = "\n".join([f"- {k}" for k in unknown_fields]) or "None"

        try:
            system_instructions = system_instructions_raw.format(
                character_name=char.name,
                aliases=", ".join(alias_list),
                known_traits=known_display,
                unknown_traits=unknown_display
            )
        except Exception as e:
            print(f"[Profiler] Dynamic prompt formatting error: {str(e)}")
            system_instructions = system_instructions_raw\
                .replace("{character_name}", char.name)\
                .replace("{aliases}", ", ".join(alias_list))\
                .replace("{known_traits}", known_display)\
                .replace("{unknown_traits}", unknown_display)

        user_prompt = (
            f"### CURRENT TEXT PASSAGE (from {chunk_data['book_name']}) ###\n"
            f"\"\"\"\n{chunk_text}\n\"\"\"\n\n"
            f"Task: Review the passage. Extract facts for the unknown traits or self-correct "
            f"any contradicted provisional details. Respond with a single JSON block."
        )

        full_prompt = f"{system_instructions}\n\n{user_prompt}"

        try:
            print(f"[Profiler] Scanning {chunk_data['book_name']} Chunk {chunk_data['chunk_index']} for {char.name} ({scanned_count}/{max_chunks_to_scan})...")
            raw_response = await get_llm_response(full_prompt, llm_url, model_name)
            extracted_json = extract_json_from_text(raw_response)

            if extracted_json:
                print(f"[Profiler] Received profiling data: {extracted_json}")
                for key in state_checklist.keys():
                    new_val = extracted_json.get(key)
                    if new_val and str(new_val).strip() != "" and str(new_val).lower() != "null":
                        if is_valid_permanent_trait(key, str(new_val), state_checklist[key]):
                            state_checklist[key] = str(new_val).strip()

            if progress_callback:
                progress_callback(char.id, scanned_count, max_chunks_to_scan, state_checklist)

        except Exception as e:
            print(f"[Profiler] Error during chunk scan loop: {str(e)}")

        await asyncio.sleep(0.5)

    with Session(engine) as session:
        db_char = session.get(Character, character_id)
        if db_char:
            db_char.demographics = state_checklist["demographics"]
            db_char.physical_build = state_checklist["physical_build"]
            db_char.hair_and_face = state_checklist["hair_and_face"]
            db_char.distinguishing_marks = state_checklist["distinguishing_marks"]
            
            # Auto-compile visual description if character is unlocked
            if not db_char.locked:
                db_char.visual_description = compile_character_visual_prompt(db_char)
                
            session.add(db_char)
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
    Scans all characters in a project, computes their bracket-mention frequencies,
    and automatically merges sub-characters (like 'Detective Stone', 'Stone's') 
    into their most prominent canonical counterpart (like 'Stone').
    Utilizes difflib sequence matching, title-stripping, and substring rules.
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

                        if target_norm == cand_norm:
                            is_match = True
                            match_reason = f"Title/Possessive Normalization"
                            break

                        if len(target_norm) >= 4 and len(cand_norm) >= 4:
                            if target_norm in cand_norm or cand_norm in target_norm:
                                is_match = True
                                match_reason = f"Substring Containment"
                                break

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

                    # Fixed: Collect all candidate names/aliases and verify uniquely against target
                    existing_aliases_on_target = {a.lower() for a in target_aliases}
                    candidates_to_add = {candidate_char.name.lower()}
                    for alias in cand_aliases_db:
                        candidates_to_add.add(alias.alias.lower())

                    new_aliases_to_create = candidates_to_add - existing_aliases_on_target

                    # Delete candidate aliases entirely to prevent duplicated target aliases
                    for alias in cand_aliases_db:
                        session.delete(alias)

                    # Create fresh non-duplicate aliases assigned directly to the target
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

                    cand_mods = session.exec(
                        select(CharacterStateModifier).where(CharacterStateModifier.character_id == candidate_char.id)
                    ).all()
                    for mod in cand_mods:
                        mod.character_id = target_char.id
                        session.add(mod)

                    session.delete(candidate_char)
                    session.commit()
                    merged_ids.add(candidate_char.id)

            # Re-compile target visual description after all merges if unlocked
            if target_char.id not in merged_ids and not target_char.locked:
                target_char.visual_description = compile_character_visual_prompt(target_char)
                session.add(target_char)
                session.commit()

    save_project_characters_to_json(project_id)
    return merged_log

def compile_character_description(char: Character, enabled_fields: Dict[str, bool], use_sentence_structure: bool) -> str:
    """
    Assembles selected character traits into either a comma-separated list 
    or a parenthetical relative clause to bound traits and prevent bleeding.
    If no traits are populated, returns the character's name directly.
    """
    demo = char.demographics if enabled_fields.get("demographics", True) else None
    build = char.physical_build if enabled_fields.get("physical_build", True) else None
    hair_face = char.hair_and_face if enabled_fields.get("hair_and_face", True) else None
    marks = char.distinguishing_marks if enabled_fields.get("distinguishing_marks", True) else None

    # Check if there are any active, populated visual details at all
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

        cleaned_pieces = [p.strip() for p in pieces if p and str(p).strip()]
        if not cleaned_pieces:
            return char.name
            
        return ", ".join(cleaned_pieces)
    
    else:
        # Prevent blending/cross-contamination via descriptive containment
        base_noun = demo.strip() if demo else "person"
        
        # Determine phonetic a/an
        first_char = base_noun[0].lower() if base_noun else 'p'
        article = "an" if first_char in "aeiou" else "a"
        
        clauses = []
        if hair_face:
            clauses.append(hair_face.strip())
            
        if build:
            b_clean = build.strip()
            # Handle if LLM extracted starting with 'who is' or 'is'
            if not b_clean.lower().startswith("who is ") and not b_clean.lower().startswith("is "):
                clauses.append(f"who is {b_clean}")
            else:
                clauses.append(b_clean)
            
        if marks:
            clauses.append(marks.strip())
            
        if clauses:
            # Construct cohesive natural relative clauses
            if len(clauses) > 1:
                main_clauses = ", ".join(clauses[:-1])
                final_clause = clauses[-1]
                if not final_clause.lower().startswith("and "):
                    final_clause = f"and {final_clause}"
                parenthetical = f"{article} {base_noun}, {main_clauses}, {final_clause}"
            else:
                parenthetical = f"{article} {base_noun}, {clauses[0]}"
            
            # Defensive post-processing cleanup (fix double spaces, duplicate commas, double connectives)
            parenthetical = re.sub(r'\s*,\s*,', ',', parenthetical)
            parenthetical = re.sub(r'\band\s+and\b', 'and', parenthetical)
            parenthetical = re.sub(r'\bwith\s+with\b', 'with', parenthetical)
            parenthetical = re.sub(r'\s+', ' ', parenthetical).strip()
            
            return f"{char.name} ({parenthetical})"
        else:
            return f"{char.name} ({article} {base_noun})"


def replace_character_tags_in_prompt(
    prompt: str, 
    project_id: int, 
    enabled_fields: Dict[str, bool], 
    use_sentence_structure: bool
) -> str:
    """
    Scans a prompt string for bracketed tags, matches aliases to project characters, 
    and returns a modified prompt string containing compiled descriptions.
    If the character occurs multiple times, only the first mention gets expanded 
    to prevent redundancy, prompt bloat, and attribute bleeding.
    """
    bracket_regex = re.compile(r"\[(.*?)\]")
    matches = bracket_regex.findall(prompt)
    if not matches:
        return prompt

    modified_prompt = prompt
    expanded_character_ids = set()  # Track which characters have already been described in this prompt
    
    with Session(engine) as session:
        for match in matches:
            tag = match.strip()
            # Fixed: Match Alias scoped strictly to the current project_id
            alias = session.exec(
                select(CharacterAlias)
                .join(Character)
                .where(CharacterAlias.alias == tag)
                .where(Character.project_id == project_id)
            ).first()
            
            if not alias:
                char = session.exec(
                    select(Character).where(Character.project_id == project_id).where(Character.name == tag)
                ).first()
            else:
                char = session.get(Character, alias.character_id)

            if char:
                # If we've already described this specific character ID in this prompt, just use their name!
                if char.id in expanded_character_ids:
                    replacement = char.name
                else:
                    replacement = compile_character_description(char, enabled_fields, use_sentence_structure)
                    expanded_character_ids.add(char.id)
                
                # Replace ONLY the first single occurrence of this bracketed tag in the string
                modified_prompt = modified_prompt.replace(f"[{tag}]", replacement, 1)
            else:
                # Fallback: Strip brackets for characters/pronouns not in the database
                modified_prompt = modified_prompt.replace(f"[{tag}]", tag, 1)
    return modified_prompt