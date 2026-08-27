import re
import csv
import functools
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

MALE_HONORIFICS = {"mr", "sir", "gentleman", "father", "uncle", "brother", "lord", "lieutenant", "captain"}
FEMALE_HONORIFICS = {"mrs", "ms", "miss", "lady", "mother", "aunt", "sister", "madam", "madame"}
TITLES = MALE_HONORIFICS.union(FEMALE_HONORIFICS).union({
    "detective", "officer", "agent", "dr", "doctor", "professor", "sergeant", 
    "colonel", "general", "sheriff", "deputy", "chief", "the", "a", "an"
})


@functools.lru_cache(maxsize=8192)
def extract_name_details(name_str: str) -> Dict[str, Any]:
    """Isolates core clean word tokens, sifting out honorific prefixes (cached)."""
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

# In-memory caches for prompt tag frequencies to avoid repeating disk reads on every UI interaction
_PROJECT_FREQ_CACHE: Dict[str, Dict[str, int]] = {}
_BOOK_FREQ_CACHE: Dict[str, Dict[str, int]] = {}


def clear_frequency_map_cache():
    """Clears cached frequency maps, forcing a fresh disk read when prompts update."""
    _PROJECT_FREQ_CACHE.clear()
    _BOOK_FREQ_CACHE.clear()


def get_character_frequency_map_db(project_name: str, session: Session) -> Dict[str, int]:
    """Scans prompts.csv files across all books in a project to build tag occurrence totals (cached)."""
    if project_name in _PROJECT_FREQ_CACHE:
        return _PROJECT_FREQ_CACHE[project_name]

    frequencies: Dict[str, int] = {}
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

    _PROJECT_FREQ_CACHE[project_name] = frequencies
    return frequencies


def get_book_character_frequency_map(project_name: str, book_name: str, session: Session) -> Dict[str, int]:
    """Scans prompts.csv for a single book to get book-specific tag frequencies (cached)."""
    cache_key = f"{project_name}::{book_name}"
    if cache_key in _BOOK_FREQ_CACHE:
        return _BOOK_FREQ_CACHE[cache_key]

    frequencies: Dict[str, int] = {}
    bracket_regex = re.compile(r"\[(.*?)\]")
    base_output_dir = Path(get_setting("output_dir", "./output", session)).resolve()

    csv_path = base_output_dir / project_name / book_name / "prompts.csv"
    if csv_path.exists():
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

    _BOOK_FREQ_CACHE[cache_key] = frequencies
    return frequencies


def extract_characters_from_prompts(project_id: int) -> Set[str]:
    """
    Scans the prompts.csv file of every book in the project, looking for bracketed names [Dino].
    Automatically indexes them in the database, establishes CharacterBookLink associations,
    and serializes the result to characters.json.
    """
    clear_frequency_map_cache()
    discovered_tags: Set[str] = set()
    bracket_regex = re.compile(r"\[(.*?)\]")

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return discovered_tags

        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        base_output_dir = Path(get_setting("output_dir", "./output", session)).resolve()

        # Map: tag_string -> set of book IDs where it appeared
        tag_to_books: Dict[str, Set[int]] = {}

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
                                if clean_tag not in tag_to_books:
                                    tag_to_books[clean_tag] = set()
                                tag_to_books[clean_tag].add(book.id)
            except Exception as e:
                print(f"[Characters] Error reading prompt CSV for {book.name}: {str(e)}")

        if not tag_to_books:
            return discovered_tags

        for tag, book_ids in tag_to_books.items():
            # 1. Check if this tag matches an existing CharacterAlias
            existing_alias = session.exec(
                select(CharacterAlias)
                .join(Character)
                .where(CharacterAlias.alias == tag)
                .where(Character.project_id == project_id)
            ).first()

            char_id = None
            if existing_alias:
                char_id = existing_alias.character_id
            else:
                # 2. Check if it matches canonical name of an existing Character
                existing_char = session.exec(
                    select(Character)
                    .where(Character.project_id == project_id)
                    .where(Character.name == tag)
                ).first()

                if existing_char:
                    char_id = existing_char.id
                    # Ensure alias row exists
                    has_alias = session.exec(
                        select(CharacterAlias)
                        .where(CharacterAlias.character_id == char_id)
                        .where(CharacterAlias.alias == tag)
                    ).first()
                    if not has_alias:
                        session.add(CharacterAlias(character_id=char_id, alias=tag))
                else:
                    # 3. Create brand new Character entity
                    new_char = Character(
                        project_id=project_id,
                        name=tag,
                        origin="Prompt Scan",
                        merge_checked=False,
                        hit_count=0
                    )
                    session.add(new_char)
                    session.flush()
                    char_id = new_char.id

                    # Create Base State timeline event
                    base_ev = CharacterTimelineEvent(
                        character_id=char_id,
                        book_id=None,
                        chapter_num=0,
                        scene_num=0,
                        label="Base State"
                    )
                    session.add(base_ev)

                    # Add canonical name as alias
                    new_alias = CharacterAlias(character_id=char_id, alias=tag)
                    session.add(new_alias)

            # 4. Link character to all books where this tag was found
            if char_id:
                for b_id in book_ids:
                    existing_link = session.exec(
                        select(CharacterBookLink)
                        .where(CharacterBookLink.character_id == char_id)
                        .where(CharacterBookLink.book_id == b_id)
                    ).first()
                    if not existing_link:
                        session.add(CharacterBookLink(character_id=char_id, book_id=b_id))

        session.commit()

    recalculate_project_character_hits(project_id)
    save_project_characters_to_json(project_id)
    return discovered_tags


def merge_character_aliases(project_id: int, target_character_id: int, source_alias_ids: List[int]):
    """
    Merges multiple alias rows or entire characters into a canonical target Character.
    """
    from services.character_manager import merge_character_into_target

    with Session(engine) as session:
        target_char = session.get(Character, target_character_id)
        if not target_char:
            return

        source_char_ids_to_merge = set()

        for alias_id in source_alias_ids:
            alias = session.get(CharacterAlias, alias_id)
            if not alias:
                continue

            old_char_id = alias.character_id
            if old_char_id != target_character_id:
                source_char_ids_to_merge.add(old_char_id)

    # Perform full structural character merge for any distinct characters
    for src_id in source_char_ids_to_merge:
        merge_character_into_target(source_char_id=src_id, target_char_id=target_character_id)

    recalculate_project_character_hits(project_id)
    save_project_characters_to_json(project_id)

def commit_raw_seeded_characters(project_id: int, book_id: Optional[int], raw_text: str):
    """
    Parses comma-separated or newline-delimited character names and attaches them
    directly to the specified book_id (or project base if None).
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    parsed_names: List[str] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",") if p.strip()]
        for p in parts:
            if len(p) <= 40 and not any(c in p for c in ["[", "]", "{", "}", "<", ">"]):
                parsed_names.append(p)

    unique_names = list(dict.fromkeys(parsed_names))
    if not unique_names:
        return

    with Session(engine) as session:
        for name in unique_names:
            existing_char = session.exec(
                select(Character)
                .where(Character.project_id == project_id)
                .where(Character.name == name)
            ).first()

            if existing_char:
                char_id = existing_char.id
            else:
                new_char = Character(
                    project_id=project_id,
                    name=name,
                    origin="Direct Seed",
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

                new_alias = CharacterAlias(character_id=char_id, alias=name)
                session.add(new_alias)

            if book_id:
                existing_link = session.exec(
                    select(CharacterBookLink)
                    .where(CharacterBookLink.character_id == char_id)
                    .where(CharacterBookLink.book_id == book_id)
                ).first()
                if not existing_link:
                    session.add(CharacterBookLink(character_id=char_id, book_id=book_id))

        session.commit()

    recalculate_project_character_hits(project_id)
    save_project_characters_to_json(project_id)


def recalculate_project_character_hits(project_id: int):
    """
    Scans prompts.csv files and updates cached character hit counts across the project.
    """
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return

        freq_map = get_character_frequency_map_db(project.name, session)
        characters = session.exec(select(Character).where(Character.project_id == project_id)).all()

        aliases = session.exec(
            select(CharacterAlias)
            .join(Character)
            .where(Character.project_id == project_id)
        ).all()

        from collections import defaultdict
        char_aliases = defaultdict(list)
        for a in aliases:
            char_aliases[a.character_id].append(a)

        for char in characters:
            aliases_list = char_aliases[char.id]
            total_hits = 0
            for a in aliases_list:
                total_hits += freq_map.get(a.alias.lower().strip(), 0)

            if total_hits == 0:
                total_hits = freq_map.get(char.name.lower().strip(), 0)

            char.hit_count = total_hits
            session.add(char)
        session.commit()


def prune_unused_seeded_characters(project_id: int):
    """
    Silently deletes any unlocked character with hit_count == 0.
    Also prunes unused aliases (0 hits) that differ from the canonical name.
    """
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if not project:
            return

        books = session.exec(select(Book).where(Book.project_id == project_id)).all()
        has_prompts = False
        for b in books:
            csv_path = base_output_dir / project.name / b.name / "prompts.csv"
            if csv_path.exists():
                has_prompts = True
                break

        if not has_prompts:
            return

        # 1. Prune unused unlocked characters
        unused_chars = session.exec(
            select(Character)
            .where(Character.project_id == project_id)
            .where(Character.locked == False)
            .where(Character.hit_count == 0)
        ).all()

        for uc in unused_chars:
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

        # 2. Prune unused aliases for remaining active characters
        freq_map = get_character_frequency_map_db(project.name, session)
        active_chars = session.exec(select(Character).where(Character.project_id == project_id)).all()

        for ac in active_chars:
            aliases = session.exec(select(CharacterAlias).where(CharacterAlias.character_id == ac.id)).all()
            for alias in aliases:
                alias_text = alias.alias.lower().strip()
                canonical_text = ac.name.lower().strip()

                if alias_text != canonical_text:
                    if freq_map.get(alias_text, 0) == 0:
                        session.delete(alias)

        session.commit()

    save_project_characters_to_json(project_id)

def get_suggested_alias_merges(project_id: int, target_character_id: int, book_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Finds unmerged characters in the project that likely belong to target_character_id
    by comparing all master variants (name + all aliases) against candidate variants.
    """
    suggestions = []

    with Session(engine) as session:
        target_char = session.get(Character, target_character_id)
        if not target_char:
            return []

        project = session.get(Project, project_id)
        freq_map = get_character_frequency_map_db(project.name, session) if project else {}

        # 1. Collect all target names & aliases and extract their clean tokens
        target_aliases = session.exec(
            select(CharacterAlias).where(CharacterAlias.character_id == target_character_id)
        ).all()
        target_variant_strings = [target_char.name] + [a.alias for a in target_aliases]
        target_details_list = [extract_name_details(v) for v in target_variant_strings if v.strip()]
        
        target_tokens: Set[str] = set()
        for td in target_details_list:
            for token in td["clean_words"]:
                if len(token) >= 3:
                    target_tokens.add(token)

        # 2. Query candidate characters to evaluate
        query = select(Character).where(Character.project_id == project_id).where(Character.id != target_character_id)
        if book_id:
            query = query.join(CharacterBookLink).where(CharacterBookLink.book_id == book_id)
        candidate_chars = session.exec(query).all()

        for cand in candidate_chars:
            cand_aliases = session.exec(
                select(CharacterAlias).where(CharacterAlias.character_id == cand.id)
            ).all()
            cand_variant_strings = [cand.name] + [a.alias for a in cand_aliases]
            cand_details_list = [extract_name_details(v) for v in cand_variant_strings if v.strip()]

            cand_tokens: Set[str] = set()
            for cd in cand_details_list:
                for token in cd["clean_words"]:
                    if len(token) >= 3:
                        cand_tokens.add(token)

            reason = None
            best_similarity = 0.0

            # Match Check 1: Direct Clean Token Overlap (e.g. 'Barrington' in 'Stone Barrington' and 'Mr. Barrington')
            shared_tokens = target_tokens.intersection(cand_tokens)
            if shared_tokens:
                reason = f"Shares word '{list(shared_tokens)[0].capitalize()}'"
                best_similarity = 0.90

            # Match Check 2: Substring & Sequence Matching across all variant combinations
            if not reason:
                for td in target_details_list:
                    t_full = td["clean_full"]
                    if not t_full:
                        continue
                    for cd in cand_details_list:
                        c_full = cd["clean_full"]
                        if not c_full:
                            continue

                        # Substring match on clean tokens
                        if len(t_full) >= 3 and len(c_full) >= 3 and (t_full in c_full or c_full in t_full):
                            reason = "Partial name match"
                            best_similarity = max(best_similarity, 0.85)
                            break

                        # Fuzzy similarity ratio check with fast mathematical length bound
                        len_c = len(c_full)
                        len_t = len(t_full)
                        if len_c + len_t > 0:
                            max_possible = (2.0 * min(len_c, len_t)) / (len_c + len_t)
                            if max_possible >= 0.80:
                                sim = difflib.SequenceMatcher(None, t_full, c_full).ratio()
                                if sim > best_similarity:
                                    best_similarity = sim

                    if reason:
                        break

                if not reason and best_similarity >= 0.80:
                    reason = f"Similar spelling ({int(best_similarity * 100)}% match)"

            if reason:
                hits = sum(freq_map.get(cv.lower().strip(), 0) for cv in cand_variant_strings)
                suggestions.append({
                    "character_id": cand.id,
                    "name": cand.name,
                    "aliases": [a.alias for a in cand_aliases],
                    "hits": hits,
                    "reason": reason,
                    "similarity": best_similarity
                })

    # Prioritize high hit counts and similarity
    suggestions.sort(key=lambda x: (x["hits"], x["similarity"]), reverse=True)
    return suggestions[:8]