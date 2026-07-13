import re
import csv
import difflib
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Project, Book, Character, CharacterAlias, CharacterTimelineEvent, CharacterBookLink
from services.character_manager import (
    save_project_characters_to_json,
    compile_character_visual_prompt
)

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

            new_char = Character(project_id=project_id, name=tag, origin="Prompt Scan", merge_checked=False)
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

    recalculate_project_character_hits(project_id)
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
                    cand_evs = session.exec(
                        select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id == old_char_id)
                    ).all()
                    for ev in cand_evs:
                        if ev.book_id is None:
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


# ========================================================
# --- NEW DYNAMIC "ZAP & MAGNET" WORKSPACE ENGINE ---
# ========================================================

MALE_HONORIFICS = {"mr", "sir", "gentleman", "father", "uncle", "brother", "lord", "lieutenant", "captain"}
FEMALE_HONORIFICS = {"mrs", "ms", "miss", "lady", "mother", "aunt", "sister", "madam", "madame"}
TITLES = MALE_HONORIFICS.union(FEMALE_HONORIFICS).union({
    "detective", "officer", "agent", "dr", "doctor", "professor", "sergeant", 
    "colonel", "general", "sheriff", "deputy", "chief"
})

def extract_name_details(name_str: str) -> Dict[str, Any]:
    """Isolates core clean word tokens, sifting out honorific prefixes."""
    val = name_str.lower().strip()
    if val.endswith("'s") or val.endswith("’s"):
        val = val[:-2].strip()
        
    words = val.split()
    extracted_honorifics = []
    clean_words = []
    for w in words:
        w_clean = re.sub(r"^\W+|\W+$", "", w)
        if w_clean in TITLES:
            extracted_honorifics.append(w_clean)
        else:
            if w_clean:
                clean_words.append(w_clean)
                
    first_name = clean_words[0] if clean_words else ""
    last_name = clean_words[-1] if len(clean_words) > 1 else ""
    
    return {
        "original": name_str,
        "clean_words": clean_words,
        "clean_full": " ".join(clean_words),
        "first_name": first_name,
        "last_name": last_name,
        "honorifics": extracted_honorifics
    }

def is_loose_match(cand_name: str, master_name: str, master_aliases: List[str], threshold: float = 0.6) -> bool:
    """Loose matching checks for phonetic/token overlap or loose fuzzy similarity >= threshold."""
    cand_details = extract_name_details(cand_name)
    cand_tokens = cand_details["clean_words"]
    if not cand_tokens:
        return False
        
    # Compile Master Variants
    master_variants = [master_name] + master_aliases
    for var in master_variants:
        var_details = extract_name_details(var)
        var_tokens = var_details["clean_words"]
        
        # 1. Direct overlap on any clean token that is at least 3 characters long (prevents short leaks)
        long_cand_tokens = {t for t in cand_tokens if len(t) >= 3}
        long_var_tokens = {t for t in var_tokens if len(t) >= 3}
        if long_cand_tokens.intersection(long_var_tokens):
            return True
            
        # 2. Loose fuzzy similarity check (captures misspelt transcriptions)
        ratio = difflib.SequenceMatcher(None, cand_details["clean_full"], var_details["clean_full"]).ratio()
        if ratio >= threshold:
            return True
            
    return False

def get_unresolved_singletons(project_id: int) -> List[Dict[str, Any]]:
    """Gets left-pane list of singletons with no aliases or seed headers."""
    with Session(engine) as session:
        chars = session.exec(
            select(Character)
            .where(Character.project_id == project_id)
            .where(Character.merge_checked == False)
            .where(Character.origin != "Goodreads Seed")
        ).all()
        
        results = []
        for c in chars:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == c.id)).all()
            if len(aliases) > 1:
                continue
            results.append({
                "id": c.id,
                "name": c.name,
                "hit_count": c.hit_count or 0
            })
            
    results.sort(key=lambda x: x["name"].lower())
    return results

def get_master_profiles(project_id: int) -> List[Dict[str, Any]]:
    """Gets right-pane consolidated list of master profiles."""
    with Session(engine) as session:
        chars = session.exec(
            select(Character)
            .where(Character.project_id == project_id)
        ).all()
        
        results = []
        for c in chars:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == c.id)).all()
            alias_list = [a.alias for a in aliases]
            
            is_seeded = (c.origin == "Goodreads Seed")
            is_promoted = c.merge_checked
            has_aliases = len(alias_list) > 1
            
            if has_aliases or is_seeded or is_promoted or c.locked:
                results.append({
                    "id": c.id,
                    "name": c.name,
                    "aliases": alias_list,
                    "is_seeded": is_seeded,
                    "is_worked_on": has_aliases,
                    "locked": c.locked,
                    "hit_count": c.hit_count or 0
                })
                
    results.sort(key=lambda x: x["name"].lower())
    return results

def promote_singleton_to_master(project_id: int, singleton_id: int):
    """Flags a left-pane singleton to master-profile status."""
    with Session(engine) as session:
        char = session.get(Character, singleton_id)
        if char:
            char.merge_checked = True
            session.add(char)
            session.commit()
    save_project_characters_to_json(project_id)


def demote_master_to_singleton(project_id: int, master_id: int):
    """Demotes a master profile back to an unresolved singleton."""
    with Session(engine) as session:
        char = session.get(Character, master_id)
        if char:
            char.merge_checked = False
            session.add(char)
            session.commit()
    save_project_characters_to_json(project_id)

def zap_singleton_into_master(project_id: int, candidate_id: int, target_id: int):
    """Consolidates candidate character aliases, links, and overrides into target."""
    with Session(engine) as session:
        tgt_char = session.get(Character, target_id)
        cand_char = session.get(Character, candidate_id)
        if not tgt_char or not cand_char:
            return
            
        cand_aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == candidate_id)).all()
        for alias in cand_aliases:
            dup = session.exec(
                select(CharacterAlias)
                .where(CharacterAlias.character_id == target_id)
                .where(CharacterAlias.alias == alias.alias)
            ).first()
            if dup:
                session.delete(alias)
            else:
                alias.character_id = target_id
                session.add(alias)
                
        dup_canonical = session.exec(
            select(CharacterAlias)
            .where(CharacterAlias.character_id == target_id)
            .where(CharacterAlias.alias == cand_char.name)
        ).first()
        if not dup_canonical and cand_char.name.lower() != tgt_char.name.lower():
            new_alias = CharacterAlias(character_id=target_id, alias=cand_char.name)
            session.add(new_alias)
            
        cand_links = session.exec(select(CharacterBookLink).where(CharacterBookLink.character_id == candidate_id)).all()
        for link in cand_links:
            dup_link = session.exec(
                select(CharacterBookLink)
                .where(CharacterBookLink.character_id == target_id)
                .where(CharacterBookLink.book_id == link.book_id)
            ).first()
            if dup_link:
                session.delete(link)
            else:
                link.character_id = target_id
                session.add(link)
                
        cand_evs = session.exec(select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id == candidate_id)).all()
        for ev in cand_evs:
            if ev.book_id is None:
                tgt_base = session.exec(
                    select(CharacterTimelineEvent)
                    .where(CharacterTimelineEvent.character_id == target_id)
                    .where(CharacterTimelineEvent.book_id == None)
                ).first()
                if tgt_base:
                    if not tgt_base.demographics: tgt_base.demographics = ev.demographics
                    if not tgt_base.physical_build: tgt_base.physical_build = ev.physical_build
                    if not tgt_base.hair_and_face: tgt_base.hair_and_face = ev.hair_and_face
                    if not tgt_base.distinguishing_marks: tgt_base.distinguishing_marks = ev.distinguishing_marks
                    session.add(tgt_base)
                session.delete(ev)
            else:
                ev.character_id = target_id
                session.add(ev)
                
        session.delete(cand_char)
        session.commit()
        
        tgt_char.merge_checked = True
        session.add(tgt_char)
        session.commit()
        
        if not tgt_char.locked:
            base_ev = session.exec(
                select(CharacterTimelineEvent)
                .where(CharacterTimelineEvent.character_id == target_id)
                .where(CharacterTimelineEvent.book_id == None)
            ).first()
            if base_ev:
                base_ev.visual_description = compile_character_visual_prompt(base_ev)
                session.add(base_ev)
                session.commit()
                
    recalculate_project_character_hits(project_id)
    save_project_characters_to_json(project_id)


def remove_alias_from_master_by_text(project_id: int, target_id: int, alias_text: str):
    """Deletes alias row and recreates it as a standalone unresolved singleton on the left."""
    with Session(engine) as session:
        alias_row = session.exec(
            select(CharacterAlias)
            .where(CharacterAlias.character_id == target_id)
            .where(CharacterAlias.alias == alias_text)
        ).first()
        if alias_row:
            session.delete(alias_row)
            
        existing = session.exec(
            select(Character)
            .where(Character.project_id == project_id)
            .where(Character.name == alias_text)
        ).first()
        
        if not existing:
            new_char = Character(
                project_id=project_id,
                name=alias_text,
                origin="Alias Reversion",
                merge_checked=False,
                hit_count=0
            )
            session.add(new_char)
            session.flush()
            
            new_alias = CharacterAlias(character_id=new_char.id, alias=alias_text)
            session.add(new_alias)
            
            base_ev = CharacterTimelineEvent(
                character_id=new_char.id,
                book_id=None,
                chapter_num=0,
                scene_num=0,
                label="Base State"
            )
            session.add(base_ev)
            
        session.commit()
        
    recalculate_project_character_hits(project_id)
    save_project_characters_to_json(project_id)


def auto_merge_project_characters(project_id: int, similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """Legacy compatibility block, does not run default destructive merges."""
    return []


# =======================================================
# --- SANITIZED GOODREADS & PLAIN TEXT SEED GOBBLER ---
# =======================================================

def is_valid_character_name(name: str) -> bool:
    """
    Applies rationality check rules to filter out synopses, reviews, and junk lines.
    Rejects strings > 35 characters, > 4 words, or containing invalid metadata patterns.
    """
    clean_val = name.strip()
    if not clean_val:
        return False

    # Max length threshold
    if len(clean_val) > 35:
        return False

    # Word count limits (reject if > 4 words unless formatted like "Mr. and Mrs. X")
    words = clean_val.split()
    if len(words) > 4:
        if not ("and" in clean_val.lower() and ("mr." in clean_val.lower() or "mrs." in clean_val.lower())):
            return False

    # Filter out common review, synopsis, or web elements
    banned_keywords = [
        "review", "published", "genres", "synopsis", "written by", "author", 
        "http", "www.", "goodreads", "page", "ebook", "paperback", "hardcover",
        "chapter", "rating", "edition", "first published", "original title"
    ]
    val_lower = clean_val.lower()
    for kw in banned_keywords:
        if kw in val_lower:
            return False

    # Disallow pronoun/conjunction leaks
    banned_lone_words = {"he", "she", "they", "them", "was", "were", "the", "with", "this", "that"}
    for word in words:
        if word.lower() in banned_lone_words:
            return False

    # Filter out un-escaped web syntax or punctuation leaks
    if any(char in clean_val for char in ["[", "]", "{", "}", "<", ">", "*", "/", "\\", "_"]):
        return False

    return True


def parse_goodreads_dump(raw_text: str) -> Dict[str, Any]:
    """
    Parses raw pasted text from either structured Goodreads dumps or raw text lists.
    Extracts Book Title, Author, and parsed character names with full sanitization.
    """
    title = None
    author = None
    characters = []

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # Detect if structured Goodreads metadata is present
    text_lower = raw_text.lower()
    is_goodreads_format = "characters" in text_lower or "original title" in text_lower or "first published" in text_lower

    if is_goodreads_format:
        # --- PATHWAY A: GOODREADS STRUCTURED EXTRACTION ---
        # 1. Extract original title
        for idx, line in enumerate(lines):
            if line.lower() == "original title" and idx + 1 < len(lines):
                title = lines[idx + 1]
                break

        # If not found, look for Series index tags (e.g., "Stone Barrington #42")
        if not title:
            for idx, line in enumerate(lines):
                if re.search(r"#\d+", line):
                    cand_title_idx = idx + 1
                    while cand_title_idx < len(lines) and not lines[cand_title_idx]:
                        cand_title_idx += 1
                    if cand_title_idx < len(lines):
                        cand_title = lines[cand_title_idx]
                        if cand_title.lower() not in ["want to read", "rate this book", "show more", "genres"]:
                            title = cand_title
                            
                            # Next non-empty line after title is typically the Author
                            cand_author_idx = cand_title_idx + 1
                            while cand_author_idx < len(lines) and not lines[cand_author_idx]:
                                cand_author_idx += 1
                            if cand_author_idx < len(lines):
                                cand_author = lines[cand_author_idx]
                                if not re.match(r"^\d+(\.\d+)?$", cand_author) and len(cand_author) < 100:
                                    author = cand_author
                            break

        # Fallback to look for Goodreads Author profiles
        if not author:
            for line in lines:
                if "(Goodreads Author)" in line:
                    author = line.replace("(Goodreads Author)", "").strip()
                    break
            if not author:
                for line in lines:
                    by_match = re.search(r"^by\s+([A-Za-z\s\.\-\'\’]+)$", line, re.IGNORECASE)
                    if by_match:
                        author = by_match.group(1).strip()
                        break

        # 2. Extract structured Characters list
        char_index = -1
        for idx, line in enumerate(lines):
            if re.match(r"^Characters\b", line, re.IGNORECASE):
                char_index = idx
                break

        if char_index != -1:
            char_lines = []
            header_content = re.sub(r"^Characters[:\s\t]*", "", lines[char_index], flags=re.IGNORECASE).strip()
            if header_content:
                char_lines.append(header_content)
                
            stop_markers = ["show more", "show less", "this edition", "format", "published", "asin", "isbn", "language", "genres"]
            
            for idx in range(char_index + 1, len(lines)):
                line = lines[idx]
                line_lower = line.lower()
                
                if any(m in line_lower for m in stop_markers):
                    break
                    
                if "\t" in line or ":" in line:
                    parts = re.split(r"[:\t]", line, 1)
                    if len(parts) == 2 and not any("," in p for p in parts):
                        break
                        
                char_lines.append(line)
                
            full_char_block = " ".join(char_lines)
            raw_names = [n.strip() for n in full_char_block.split(",")]
            
            for rn in raw_names:
                cleaned_name = rn
                parentheses = re.findall(r"\((.*?)\)", rn)
                for p_content in parentheses:
                    p_lower = p_content.lower()
                    strip_p = False
                    
                    if title and title.lower() in p_lower:
                        strip_p = True
                    if "goodreads" in p_lower:
                        strip_p = True
                        
                    if strip_p:
                        cleaned_name = cleaned_name.replace(f"({p_content})", "").strip()
                        
                cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip()
                
                if is_valid_character_name(cleaned_name):
                    characters.append(cleaned_name)
    else:
        # --- PATHWAY B: PLAIN TEXT COMA/LINEBREAK GOBBLER ---
        # Parse names split by lines, then further split any comma blocks
        for line in lines:
            sub_parts = [p.strip() for p in line.split(",") if p.strip()]
            for part in sub_parts:
                cleaned_name = re.sub(r"\s+", " ", part).strip()
                if is_valid_character_name(cleaned_name):
                    characters.append(cleaned_name)

    return {
        "success": len(characters) > 0 or title is not None,
        "title": title,
        "author": author,
        "characters": list(dict.fromkeys(characters))
    }


def find_matching_project_book(project_id: int, parsed_title: str, session: Session, similarity_threshold: float = 0.8) -> Optional[Book]:
    """
    Fuzzy and substring matches a parsed Goodreads title to a Book inside the active project.
    """
    if not parsed_title:
        return None
        
    books = session.exec(select(Book).where(Book.project_id == project_id)).all()
    if not books:
        return None
        
    parsed_norm = parsed_title.lower().strip()
    
    # 1. Substring Match Pass
    substring_matches = []
    for book in books:
        candidates = [book.name]
        if book.display_title:
            candidates.append(book.display_title)
            
        for cand in candidates:
            if not cand:
                continue
            cand_norm = cand.lower().strip()
            if parsed_norm in cand_norm or cand_norm in parsed_norm:
                substring_matches.append(book)
                break
                
    if len(substring_matches) == 1:
        return substring_matches[0]
        
    # 2. Fallback to Fuzzy Ratio Matching
    best_match = None
    best_ratio = 0.0
    
    for book in books:
        candidates = [book.name]
        if book.display_title:
            candidates.append(book.display_title)
            
        for cand in candidates:
            if not cand:
                continue
            cand_norm = cand.lower().strip()
            
            ratio = difflib.SequenceMatcher(None, parsed_norm, cand_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = book
                
    if best_ratio >= similarity_threshold:
        return best_match
        
    return None


def update_book_metadata(book_id: int, display_title: Optional[str], author: Optional[str]):
    """Stores secondary Goodreads metadata back to the Book row."""
    with Session(engine) as session:
        book = session.get(Book, book_id)
        if book:
            if display_title and not book.display_title:
                book.display_title = display_title
            if author and not book.author:
                book.author = author
            session.add(book)
            session.commit()


def commit_seeded_characters(project_id: int, staged_data: Dict[str, Set[int]]):
    """
    Persists staged seeded characters into SQLite and characters.json.
    Assigns origin='Goodreads Seed' and merge_checked=True.
    """
    with Session(engine) as session:
        for char_name, book_ids in staged_data.items():
            if not char_name.strip():
                continue
                
            existing_char = session.exec(
                select(Character)
                .where(Character.project_id == project_id)
                .where(Character.name == char_name)
            ).first()
            
            if existing_char:
                existing_char.origin = "Goodreads Seed"
                existing_char.merge_checked = True
                session.add(existing_char)
                char_id = existing_char.id
            else:
                new_char = Character(
                    project_id=project_id,
                    name=char_name,
                    origin="Goodreads Seed",
                    merge_checked=True,
                    hit_count=0
                )
                session.add(new_char)
                session.flush()
                char_id = new_char.id
                
                base_ev = CharacterTimelineEvent(
                    character_id=char_id,
                    book_id=None,
                    chapter_num=0,
                    scene_num=0,
                    label="Base State"
                )
                session.add(base_ev)
                
                new_alias = CharacterAlias(character_id=char_id, alias=char_name)
                session.add(new_alias)
                session.flush()
                
            for book_id in book_ids:
                if book_id is None:
                    continue
                existing_link = session.exec(
                    select(CharacterBookLink)
                    .where(CharacterBookLink.character_id == char_id)
                    .where(CharacterBookLink.book_id == book_id)
                ).first()
                
                if not existing_link:
                    new_link = CharacterBookLink(character_id=char_id, book_id=book_id)
                    session.add(new_link)
                    
        session.commit()
    save_project_characters_to_json(project_id)


# ==========================================
# --- HIT RECALCULATION & AUTO-PRUNING ---
# ==========================================

def recalculate_project_character_hits(project_id: int):
    """
    Scans prompts.csv files and updates cached character hit counts.
    Allows us to track active vs obsolete seeded names.
    """
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return
            
        freq_map = get_character_frequency_map_db(project.name, session)
        
        characters = session.exec(select(Character).where(Character.project_id == project_id)).all()
        for char in characters:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == char.id)).all()
            total_hits = 0
            for a in aliases:
                total_hits += freq_map.get(a.alias.lower().strip(), 0)
                
            if total_hits == 0:
                total_hits = freq_map.get(char.name.lower().strip(), 0)
                
            char.hit_count = total_hits
            session.add(char)
        session.commit()


# services/character_organizer.py

# services/character_organizer.py

def prune_unused_seeded_characters(project_id: int):
    """
    Silently deletes any unlocked character with hit_count == 0.
    Also prunes any unused aliases (0 hits) that differ from the character's canonical name,
    regardless of the character's locked status.
    Safety mechanism: Only triggers if prompt files actually exist in the project,
    ensuring we never prune characters before a scan is run.
    """
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return
            
        # Verify if prompts exist (has scanning occurred yet?)
        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        has_prompts = False
        for b in books:
            csv_path = base_output_dir / project.name / b.name / "prompts.csv"
            if csv_path.exists():
                has_prompts = True
                break
                
        # If no prompts exist across any book, skip pruning completely (safety safeguard)
        if not has_prompts:
            return
            
        # 1. Prune unused characters
        unused_chars = session.exec(
            select(Character)
            .where(Character.project_id == project_id)
            .where(Character.locked == False) # Only unlocked characters can be entirely deleted
            .where(Character.hit_count == 0)
        ).all()
        
        for uc in unused_chars:
            # Drop relational attachments
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == uc.id)).all()
            for a in aliases:
                session.delete(a)
                
            links = session.exec(select(CharacterBookLink).where(CharacterBookLink.character_id == uc.id)).all()
            for l in links:
                session.delete(l)
                
            events = session.exec(select(CharacterTimelineEvent).where(CharacterTimelineEvent.character_id == uc.id)).all()
            for e in events:
                session.delete(e)
                
            session.delete(uc)
            
        # 2. Prune unused aliases for *all* remaining active characters
        freq_map = get_character_frequency_map_db(project.name, session)
        
        # This query now fetches all active characters, regardless of locked status,
        # so their aliases can be cleaned up.
        active_chars = session.exec(
            select(Character)
            .where(Character.project_id == project_id)
        ).all()
        
        for ac in active_chars:
            aliases = session.exec(
                select(CharacterAlias)
                .where(CharacterAlias.character_id == ac.id)
            ).all()
            
            for alias in aliases:
                alias_text = alias.alias.lower().strip()
                canonical_text = ac.name.lower().strip()
                
                # Delete the alias if it is not the canonical name and has 0 hits in prompts.csv
                if alias_text != canonical_text:
                    hits = freq_map.get(alias_text, 0)
                    if hits == 0:
                        session.delete(alias)
                        
        session.commit()
    save_project_characters_to_json(project_id)