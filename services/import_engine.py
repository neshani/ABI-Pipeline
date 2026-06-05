import os
import re
import csv
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from sqlmodel import Session, select
from database.connection import engine, get_setting
from database.models import Project, Book, Chapter
from services.sync_engine import sync_book_from_disk

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return "".join(self.fed)


def strip_tags(html: str) -> str:
    """Removes HTML/XML tags safely from a text string."""
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def clean_navigation_lines(text: str) -> str:
    """
    Cleans up common EPUB navigation remnants (like '^', '>>', '<<', '[Top]')
    from the first few lines of a chapter's body text.
    """
    lines = text.split("\n")
    cleaned_lines = []
    
    # Matches lines containing only navigation characters or common navigation words
    nav_pattern = re.compile(
        r'^(?:[\^\s><|•·\[\]\-\(\)]|top|prev|next|index|contents|table\s*of\s*contents|home|back|up)+$', 
        re.IGNORECASE
    )
    
    # We scan the first 8 lines of text for any pure navigation artifacts
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i < 8 and stripped and nav_pattern.match(stripped):
            # Skip this navigation artifact line
            continue
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines).strip()


def extract_epub_text_by_chapters(epub_path: str) -> list[tuple[str, str]]:
    """
    Parses an EPUB file using zipfile and xml.etree.ElementTree.
    Returns a list of tuples: [(chapter_title, chapter_text), ...]
    in the correct reading order defined by the OPF spine.
    """
    chapters = []
    with zipfile.ZipFile(epub_path, 'r') as z:
        # 1. Read META-INF/container.xml to find the main OPF file
        try:
            container_xml = z.read("META-INF/container.xml")
        except KeyError:
            raise ValueError("Invalid EPUB: META-INF/container.xml not found.")
            
        root = ET.fromstring(container_xml)
        rootfile = None
        for elem in root.iter():
            if elem.tag.endswith('rootfile'):
                rootfile = elem
                break
                
        if rootfile is None or 'full-path' not in rootfile.attrib:
            raise ValueError("Invalid EPUB: No rootfile full-path found.")
            
        opf_path = rootfile.attrib['full-path']
        opf_dir = "/".join(opf_path.split("/")[:-1])
        if opf_dir:
            opf_dir += "/"
            
        # 2. Read and parse content.opf
        opf_xml = z.read(opf_path)
        opf_root = ET.fromstring(opf_xml)
        
        manifest = {}
        spine = []
        
        for elem in opf_root.iter():
            tag_local = elem.tag.split('}')[-1]
            if tag_local == 'item':
                item_id = elem.attrib.get('id')
                href = elem.attrib.get('href')
                if item_id and href:
                    manifest[item_id] = href
            elif tag_local == 'itemref':
                idref = elem.attrib.get('idref')
                if idref:
                    spine.append(idref)
                    
        # 3. Read XHTML files in correct spine order
        for idx, idref in enumerate(spine):
            if idref in manifest:
                relative_path = manifest[idref]
                import urllib.parse
                relative_path = urllib.parse.unquote(relative_path)
                
                full_item_path = opf_dir + relative_path if not relative_path.startswith('/') else relative_path.lstrip('/')
                full_item_path = "/".join([part for part in full_item_path.split('/') if part and part != '.'])
                
                zip_keys = {k.lower(): k for k in z.namelist()}
                key_lower = full_item_path.lower()
                
                matched_key = None
                if key_lower in zip_keys:
                    matched_key = zip_keys[key_lower]
                else:
                    for k in z.namelist():
                        if k.lower().endswith(key_lower):
                            matched_key = k
                            break
                
                if matched_key:
                    try:
                        content_bytes = z.read(matched_key)
                        html_text = content_bytes.decode('utf-8', errors='ignore')
                        
                        # Isolate the HTML body to ignore head tags during headers heuristics
                        body_search = re.search(r'<body[^>]*>(.*?)</body>', html_text, re.IGNORECASE | re.DOTALL)
                        body_html = body_search.group(1) if body_search else html_text
                        
                        title = None
                        
                        # Heuristic 1: Extract the first visible H1, H2, or H3 heading tag in the body
                        h_match = re.search(r'<h[1-3][^>]*>(.*?)</h[1-3]>', body_html, re.IGNORECASE | re.DOTALL)
                        if h_match:
                            title = strip_tags(h_match.group(1)).strip()
                            
                        # Heuristic 2: Fallback to the document <title> tag only if it contains non-generic short text
                        if not title:
                            title_match = re.search(r'<title>(.*?)</title>', html_text, re.IGNORECASE | re.DOTALL)
                            if title_match:
                                candidate = strip_tags(title_match.group(1)).strip()
                                if candidate and candidate.lower() not in {"unknown", "cdx", "index", "contents"} and len(candidate) < 50:
                                    title = candidate
                                    
                        # Ultimate Fallback
                        if not title:
                            title = f"Chapter {idx + 1}"
                            
                        # Clean and format the body text
                        body_html = re.sub(r'</?(?:p|div|h[1-6]|br)[^>]*>', '\n', body_html, flags=re.IGNORECASE)
                        clean_text = strip_tags(body_html).strip()
                        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
                        
                        # Apply our navigation pattern cleaners
                        clean_text = clean_navigation_lines(clean_text)
                        
                        if len(clean_text.split()) > 10:  # Ignore layout pages or covers
                            chapters.append((title, clean_text))
                    except Exception as e:
                        print(f"[EPUB-Parser] Failed to read {matched_key}: {e}")
                        
    return chapters


def import_text_transcripts(project_name: str, txt_filepaths: list[str]) -> int:
    """
    Imports sorted plain text files containing pre-existing transcription data.
    """
    sorted_paths = sorted(txt_filepaths, key=lambda p: Path(p).name.lower())
    
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    project_dir = base_output_dir / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    with Session(engine) as session:
        project = session.exec(
            select(Project).where(Project.name == project_name)
        ).first()
        
        if not project:
            project = Project(
                name=project_name,
                path=str(project_dir),
                is_batch=(len(sorted_paths) > 1),
                status="Transcribed"
            )
            session.add(project)
            session.flush()
            
        for path_str in sorted_paths:
            path = Path(path_str)
            book_name = path.stem
            book_dir = project_dir / book_name
            book_dir.mkdir(parents=True, exist_ok=True)
            
            content = path.read_text(encoding="utf-8", errors="ignore")
            
            # Clean navigation remnants out of plain transcripts too
            sections = [s.strip() for s in content.split("==CHAPTER==") if s.strip()]
            cleaned_sections = []
            for sec in sections:
                cleaned_sections.append(clean_navigation_lines(sec))
                
            # Compile and save cleaned master transcript.txt on disk
            full_clean_content = "\n\n==CHAPTER==\n\n".join(cleaned_sections)
            transcript_file = book_dir / "transcript.txt"
            transcript_file.write_text(full_clean_content, encoding="utf-8")
            
            book = session.exec(
                select(Book).where(Book.project_id == project.id).where(Book.name == book_name)
            ).first()
            
            if not book:
                book = Book(
                    project_id=project.id,
                    name=book_name,
                    path=str(book_dir),
                    status="Transcribed",
                    progress=0.0
                )
                session.add(book)
                session.flush()
                
            from sqlmodel import delete
            session.exec(delete(Chapter).where(Chapter.book_id == book.id))
            session.flush()
            
            for idx, text_block in enumerate(cleaned_sections):
                lines = [l.strip() for l in text_block.split("\n") if l.strip()]
                title = lines[0] if lines else f"Chapter {idx + 1}"
                if len(title) > 60:
                    title = title[:57] + "..."
                    
                ch = Chapter(
                    book_id=book.id,
                    chapter_num=idx + 1,
                    title=title,
                    input_file=str(path),
                    type='text_only',
                    status="Completed",
                    word_count=len(text_block.split())
                )
                session.add(ch)
            session.flush()
            
            sync_book_from_disk(book.id, session)
            
        session.commit()
        return project.id


def import_epub_novels(project_name: str, epub_filepaths: list[str]) -> int:
    """
    Imports sorted EPUB novel files, extracts raw text chapters, and registers them.
    """
    sorted_paths = sorted(epub_filepaths, key=lambda p: Path(p).name.lower())
    
    base_output_dir = Path(get_setting("output_dir", "./output")).resolve()
    project_dir = base_output_dir / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    with Session(engine) as session:
        project = session.exec(
            select(Project).where(Project.name == project_name)
        ).first()
        
        if not project:
            project = Project(
                name=project_name,
                path=str(project_dir),
                is_batch=(len(sorted_paths) > 1),
                status="Transcribed"
            )
            session.add(project)
            session.flush()
            
        for path_str in sorted_paths:
            path = Path(path_str)
            book_name = path.stem
            book_dir = project_dir / book_name
            book_dir.mkdir(parents=True, exist_ok=True)
            
            chapters_data = extract_epub_text_by_chapters(str(path))
            
            # Compile master transcript.txt with ==CHAPTER== boundaries
            full_transcript_content = ""
            for idx, (title, text) in enumerate(chapters_data):
                if idx > 0:
                    full_transcript_content += "\n\n==CHAPTER==\n\n"
                    
                # Double-Heading Guard: Prevent duplicating the title on disk
                first_lines = [l.strip().lower() for l in text.split("\n") if l.strip()][:3]
                title_clean = title.strip().lower()
                
                already_has_title = False
                for fl in first_lines:
                    if fl == title_clean or fl.startswith(title_clean) or title_clean.startswith(fl):
                        already_has_title = True
                        break
                        
                if already_has_title:
                    full_transcript_content += text
                else:
                    full_transcript_content += f"{title}\n\n{text}"
                
            transcript_file = book_dir / "transcript.txt"
            transcript_file.write_text(full_transcript_content, encoding="utf-8")
            
            book = session.exec(
                select(Book).where(Book.project_id == project.id).where(Book.name == book_name)
            ).first()
            
            if not book:
                book = Book(
                    project_id=project.id,
                    name=book_name,
                    path=str(book_dir),
                    status="Transcribed",
                    progress=0.0
                )
                session.add(book)
                session.flush()
                
            from sqlmodel import delete
            session.exec(delete(Chapter).where(Chapter.book_id == book.id))
            session.flush()
            
            for idx, (title, text) in enumerate(chapters_data):
                ch = Chapter(
                    book_id=book.id,
                    chapter_num=idx + 1,
                    title=title,
                    input_file=str(path),
                    type='text_only',
                    status="Completed",
                    word_count=len(text.split())
                )
                session.add(ch)
            session.flush()
            
            sync_book_from_disk(book.id, session)
            
        session.commit()
        return project.id