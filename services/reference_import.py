"""Reference documents: one ``## Name`` block per character.

The contract is deliberately tiny, because the point is that a user can produce
the file from anything - a companion volume, a wiki dump, a text editor - with
no field names to learn:

    ## Moiraine Damodred
    Whatever prose the source has about her. Anything, any length.

    ## Perrin Aybara
    ...

The heading is the project's character name. The body is material, not answers:
the LLM sorts it into traits, because a user who has to fill in eleven fields by
hand would rather type them straight into the editor.

Parsers here turn a source into blocks; ``blocks_to_markdown`` writes them back.
The epub reader handles the two shapes that matter in practice: a dictionary with
bold headwords (companion volumes) and a document with real headings. It is not
a general EPUB renderer and does not try to be.
"""
from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Block = Tuple[str, str]

_HEADING = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.M)
_COMMENT = re.compile(r"(?s)<!--.*?-->")
_TAG = re.compile(r"<[^>]+>")
_BOLD_HEAD = re.compile(
    r'^<p[^>]*>\s*(?:<span class="den-b-9pt">|<b>|<strong>)(.*?)(?:</span>|</b>|</strong>)'
    r"\s*(.*?)</p>$", re.S)


def normalise(text: str) -> str:
    """Collapse HTML entities and whitespace without touching the words.

    Typographic apostrophes are folded to ASCII because a headword is a *key*:
    "Perrin t’Bashere Aybara" in the file and "Perrin t'Bashere Aybara" in the
    project are the same character, and a mismatch here silently means "not
    found". The rest of the app does the same in ``clean_character_key``.
    """
    text = html.unescape(text or "")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def read_blocks(text: str) -> List[Block]:
    """``[(name, body)]`` from markdown-style text. Order is preserved."""
    text = _COMMENT.sub("\n", text or "")
    matches = list(_HEADING.finditer(text))
    blocks: List[Block] = []
    for index, match in enumerate(matches):
        name = normalise(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = normalise(text[match.end():end])
        if name:
            blocks.append((name, body))
    return blocks


def blocks_to_markdown(blocks: List[Block], *, source: str = "",
                       header: str = "") -> str:
    """The inverse, with optional provenance lines above each block."""
    lines = []
    if header:
        lines.extend([f"<!-- {header} -->", ""])
    for name, body in blocks:
        lines.append(f"## {name}")
        lines.append(body.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- epub -------------------------------------------------------------------

def _epub_text_fragments(path: Path) -> List[str]:
    """Every spine document as XHTML, in reading order where it can be known."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "META-INF/encryption.xml" in names:
            raise ValueError("this epub is encrypted (DRM); export a text or "
                             "markdown copy instead")
        documents = sorted(n for n in names if re.search(r"\.(x?html?|xml)$", n, re.I))
        # Keep reading order: the OPF spine is authoritative when present.
        order: List[str] = []
        opf = next((n for n in names if n.lower().endswith(".opf")), "")
        if opf:
            try:
                manifest = archive.read(opf).decode("utf-8", "ignore")
                ids = {}
                for item in re.finditer(
                        r'<item\b[^>]*\bid="([^"]+)"[^>]*\bhref="([^"]+)"', manifest):
                    ids[item.group(1)] = item.group(2)
                base = opf.rsplit("/", 1)[0] if "/" in opf else ""
                for ref in re.finditer(r'<itemref\b[^>]*\bidref="([^"]+)"', manifest):
                    href = ids.get(ref.group(1))
                    if not href:
                        continue
                    candidate = f"{base}/{href}" if base else href
                    candidate = re.sub(r"/\./", "/", candidate)
                    if candidate in names:
                        order.append(candidate)
            except Exception:
                order = []
        return [archive.read(n).decode("utf-8", "ignore")
                for n in (order or documents)]


def _paragraphs(markup: str) -> List[str]:
    """``<p>`` and heading elements, as plain text, in document order."""
    markup = re.sub(r"(?is)<(script|style).*?</\1>", " ", markup)
    out: List[str] = []
    for match in re.finditer(r"(?is)<(p|h[1-6]|div|li)\b[^>]*>(.*?)</\1>", markup):
        text = _TAG.sub("", match.group(2))
        text = normalise(text)
        if text:
            out.append(text)
    if not out:
        text = normalise(_TAG.sub(" ", markup))
        out = [line.strip() for line in text.split("\n") if line.strip()]
    return out


def _headword_blocks(markup: str) -> List[Block]:
    """Blocks from markup that marks its headwords, as companion volumes do.

    The headword is usually a bold run at the start of a paragraph, and the entry
    often continues in following paragraphs that carry no heading at all. Both
    shapes are handled here; the generic heuristic in ``epub_blocks`` cannot tell
    a headword from a short sentence.
    """
    blocks: List[Block] = []
    current: Optional[str] = None
    body: List[str] = []

    def flush() -> None:
        nonlocal current, body
        if current:
            blocks.append((current, normalise(" ".join(body))))
        current, body = None, []

    for match in re.finditer(r"(?is)<p[^>]*>(.*?)</p>", markup):
        inner = match.group(1)
        head = re.match(r'\s*<[^>]*class="[^"]*den-b-9pt[^"]*"[^>]*>(.*?)</[^>]+>\s*(.*)$',
                        inner, re.S)
        if head:
            flush()
            current = normalise(_TAG.sub("", head.group(1))).rstrip(" .,;:")
            body = [normalise(_TAG.sub(" ", head.group(2)))]
        elif current:
            text = normalise(_TAG.sub(" ", inner))
            if text:
                body.append(text)
    flush()
    return blocks


def _generic_blocks(markup: str) -> List[Block]:
    """Blocks from a document with headings or bold lead-ins."""
    blocks: List[Block] = []
    current: Optional[str] = None
    body: List[str] = []

    def flush() -> None:
        nonlocal current, body
        if current and body:
            blocks.append((current, normalise(" ".join(body))))
        current, body = None, []

    for paragraph in _paragraphs(markup):
        # A heading element is unambiguous; a bold lead-in is a headword only
        # when it is short and does not end a sentence.
        if re.match(r"^(chapter|part|book|prologue|epilogue)\b", paragraph, re.I) \
                and len(paragraph.split()) <= 6:
            flush()
            current = paragraph
            continue
        match = re.match(r"^(.{2,80}?)[.:]\s+(.*)$", paragraph, re.S)
        if match and len(match.group(1).split()) <= 6:
            candidate = match.group(1).strip(" .,;:")
            if candidate:
                flush()
                current = candidate
                body = [match.group(2)]
                continue
        if current:
            body.append(paragraph)
        else:
            current = paragraph[:80]
    flush()
    return blocks


def epub_blocks(path: Path) -> List[Block]:
    """Headword blocks from an epub, or one block per document as a fallback.

    Marked-up headwords win when the file has them: they are exact, and the
    heuristic below cannot tell a headword from a short sentence, so using it on
    a dictionary adds hundreds of fake entries.
    """
    blocks: List[Block] = []
    for markup in _epub_text_fragments(path):
        if "den-b-9pt" in markup:
            document_blocks = _headword_blocks(markup)
        else:
            document_blocks = _generic_blocks(markup)
        blocks.extend(document_blocks)

    # A document that produced dozens of one-line "headwords" was almost
    # certainly prose, not a glossary; keep it whole rather than inventing names.
    if len(blocks) > 1 and sum(len(b.split()) for _, b in blocks) / len(blocks) < 3:
        merged = [f"{name}. {text}" for name, text in blocks]
        return [("Imported text", normalise(" ".join(merged)))]
    return blocks


def load_blocks(path) -> List[Block]:
    """Blocks from an epub, markdown, or plain-text file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return epub_blocks(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in (".html", ".htm", ".xhtml"):
        return [("Imported text", normalise(_TAG.sub(" ", text)))]
    return read_blocks(text)


def is_cross_reference(body: str) -> bool:
    """``al'Thor, Rand. See Rand al'Thor`` - an alias, not material."""
    return body.strip().lower().startswith("see ")


def cross_references(blocks: List[Block]) -> Dict[str, List[str]]:
    """``{target: [headwords that point at it]}`` from a dictionary's See entries.

    Free alias evidence, which is much better than a string-similarity guess when
    proposing character merges.
    """
    out: Dict[str, List[str]] = {}
    for name, body in blocks:
        if is_cross_reference(body):
            target = normalise(body)[4:].strip(" .,;:")
            out.setdefault(target, []).append(name)
    return out


_HONORIFICS = ("master", "mistress", "lady", "lord", "sir", "madam", "dame",
               "captain", "sergeant", "father", "mother", "mistress.")


def _block_index(blocks: List[Block]) -> Dict[str, Block]:
    index = {}
    for name, body in blocks:
        if is_cross_reference(body):
            continue
        index.setdefault(normalise(name).lower(), (name, body))
    return index


def _name_tokens(text: str) -> set:
    return set(re.findall(r"[a-z']+", normalise(text).lower()))


def _shares_a_name_word(block: Block, character: str) -> bool:
    """Does the entry's headword use a word from the character's own name?

    This is the guard that stops a *related* word being mistaken for a *name*.
    "Loial" carries the alias "Ogier", and a dictionary has an entry called
    "Ogier"; without this check the alias looks up the species entry and the
    character is described as a race. Likewise "Byar" + the alias "Child Byar"
    reached "Child of the Dragon" through "Child".

    Aliases that share nothing with the name are still useful for *searching*
    (a wiki, the web), just not for accepting a block by themselves.
    """
    return bool(_name_tokens(block[0]) & _name_tokens(character))


def _dominant_prefix(index: Dict[str, Block], token: str) -> Optional[Block]:
    """The article when several entries begin with the name, else ``None``.

    A dictionary gives the spear-carrier one line and the protagonist a column:
    "Elayne Traemane" (94 characters) and "Elayne Trakand" (8,384). Refusing that
    as ambiguous is technically correct and practically useless - the character
    with 48 prompt hits and no description is the one with the column.

    Size is the only signal used, and the bar is deliberately high: three times
    the runner-up *and* 200 characters in its own right. Two real entries of
    comparable size stay ambiguous, which is the honest answer.
    """
    hits = sorted((block for key, block in index.items()
                   if key.startswith(token + " ")),
                  key=lambda block: len(block[1]), reverse=True)
    if len(hits) < 2:
        return None
    runner_up = max(len(hits[1][1]), 1)
    if len(hits[0][1]) >= 200 and len(hits[0][1]) >= 3 * runner_up:
        return hits[0]
    return None


def match_blocks(blocks: List[Block], names: Dict[str, List[str]]
                 ) -> Tuple[Dict[str, Block], List[Block]]:
    """``({character: block}, unmatched)``, accepting only what is certain.

    A near-miss guess puts one character's description on another, so the rules
    only accept an unambiguous answer:

    1. the project name, exactly;
    2. an alias, exactly - but only when the entry's headword uses a word from
       the character's own name (see ``_shares_a_name_word``);
    3. a dictionary cross-reference ("Finngar, Ewin. See Ewin Finngar");
    4. a single-token name that is a unique prefix ("Darl" -> "Darl Coplin"), or
       the clearly dominant entry when several share it ("Elayne" -> the column
       for "Elayne Trakand", not the line for "Elayne Traemane");
    5. the same, with an honorific stripped ("Mistress Luhhan" is *not* accepted:
       two Luhhans exist, and guessing the woman would be a coin flip);
    6. a trailing epithet, where the *first* token is a name rather than a rank
       ("Sheriam Sedai" -> "Sheriam Bayanar"; "Master Gill" is refused because
       "Master" is an honorific and "Child of the Dragon" is not about "Byar").

    Anything else comes back as unmatched, which the UI reports so the user can
    add an alias.
    """
    index = _block_index(blocks)
    target_of: Dict[str, str] = {}
    for name, body in blocks:
        if not is_cross_reference(body):
            continue
        target = normalise(body)[4:].strip(" .,;:")
        target_of[normalise(name).lower()] = target

    def lookup(key: str) -> Optional[Block]:
        return index.get(normalise(key).lower())

    def unique_prefix(token: str) -> Optional[Block]:
        hits = [block for key, block in index.items()
                if key.startswith(token + " ")]
        return hits[0] if len(hits) == 1 else None

    def by_prefix(token: str) -> Optional[Block]:
        return unique_prefix(token) or _dominant_prefix(index, token)

    matched: Dict[str, Block] = {}
    used = set()
    for character, aliases in names.items():
        candidates = [character, *aliases]
        found: Optional[Block] = None
        for candidate in candidates:                       # 1 and 2
            block = lookup(candidate)
            if block and block[0].lower() not in used:
                # The exact project name always counts. An alias has to look like
                # a name (two words) or reuse a word from the character's name:
                # "Rand al'Thor" for a row tagged "Master al'Vere" is a real alias
                # from the user's side, while "Ogier" for "Loial" is a species.
                if (candidate == character or len(candidate.split()) > 1
                        or _shares_a_name_word(block, character)):
                    found = block
                    break
        if not found:                                      # 3
            for candidate in candidates:
                target = target_of.get(normalise(candidate).lower())
                block = lookup(target) if target else None
                if block and block[0].lower() not in used:
                    found = block
                    break
        if not found:                                      # 4
            for candidate in candidates:
                if len(candidate.split()) == 1:
                    block = by_prefix(normalise(candidate).lower())
                    if block and block[0].lower() not in used \
                            and _shares_a_name_word(block, character):
                        found = block
                        break
        if not found:                                      # 5
            for candidate in candidates:
                words = normalise(candidate).split()
                if len(words) > 1 and words[0].lower().strip(".") in _HONORIFICS:
                    rest = normalise(" ".join(words[1:])).lower()
                    block = lookup(rest) or (unique_prefix(rest)
                                             if len(rest.split()) == 1 else None)
                    if block and block[0].lower() not in used \
                            and _shares_a_name_word(block, character):
                        found = block
                        break
        if not found:                                      # 6
            # A trailing epithet ("Sheriam Sedai", "Lan Gaidin"): the project tag
            # carries words the reference entry does not. Only the FIRST token is
            # tried, it must not be an honorific, and it must uniquely identify
            # one entry - anything looser starts attaching descriptions to the
            # wrong people.
            for candidate in candidates:
                words = normalise(candidate).split()
                if len(words) > 1 and words[0].lower().strip(".") not in _HONORIFICS:
                    block = unique_prefix(normalise(words[0]).lower())
                    if block and block[0].lower() not in used \
                            and _shares_a_name_word(block, character):
                        found = block
                        break
        if found:
            matched[character] = found
            used.add(found[0].lower())

    unmatched = [block for key, block in index.items() if key not in used]
    return matched, unmatched
