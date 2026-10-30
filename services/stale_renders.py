"""Images whose character traits changed after they were rendered.

A trait edit carries forward through the timeline (:func:`propagate_trait_change`
in ``character_manager``) until the book records a different value. Every image
rendered inside that stretch shows the value the user just replaced, so it has to
go before the scenes are re-rendered.

The set is defined by the **span the edit governs**, not by the old value: after
the save the old value no longer exists in the timeline (propagation rewrote every
event that carried it), so there is nothing left to match against. The span is
recoverable - from the edited position until the first later event that sets the
field to something else - which is why the calculation can happen on click rather
than in the save path.

The rule is field-dependent, mirroring what actually renders:

* ``sex`` / ``species`` are written to every event, so the whole project changes.
* a pinned trait always renders from Base, so the whole project changes.
* a Base edit propagates across volumes (and side stories resolve Base directly),
  so it reaches every book.
* ``carried`` in ``scene`` mode renders only at the position the text recorded.
* everything else: from the edited position until the book changes the value.

Only rows whose prompt text carries a tag for the character (name or any alias)
are included - a scene that never names them does not change when their coat does.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlmodel import Session, select

from database.connection import engine
from database.models import Book, Character, CharacterAlias, CharacterTimelineEvent, Project
from services.character_manager import (
    RENDERED_TRAIT_FIELDS,
    clean_character_key,
    get_pinned_fields,
)
from ui import state

IMAGE_SUFFIXES = (".png", ".webp")

# ``NN_NN_slug.png`` (underscore) is current; ``NN-NN_...`` is the legacy form that
# still exists in older output trees.
_IMAGE_NAME = re.compile(r"^(\d+)[_-](\d+)")
_TAG = re.compile(r"\[(.*?)\]")


def _output_root() -> Path:
    return Path("./output")


def _coords(row: Dict[str, Any]) -> Optional[tuple]:
    try:
        return int(float(row.get("chapter"))), int(float(row.get("shot")))
    except (TypeError, ValueError):
        return None


def _reading_key(order_map: Dict[int, int], book_id: Optional[int],
                 chapter: Any, shot: Any) -> tuple:
    return (order_map.get(book_id, 10 ** 6), int(chapter or 0), int(shot or 0))


def _scan_book_images(project_name: str, book_name: str) -> Dict[tuple, List[Path]]:
    """``{(chapter, shot): [files]}`` for one book - both image directories."""
    found: Dict[tuple, List[Path]] = {}
    book_dir = _output_root() / project_name / book_name
    for directory in (book_dir / "images", book_dir):
        if not directory.exists():
            continue
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.name.lower().endswith(IMAGE_SUFFIXES):
                        continue
                    stem = os.path.splitext(entry.name)[0]
                    match = _IMAGE_NAME.match(stem)
                    if not match:
                        continue
                    key = (int(match.group(1)), int(match.group(2)))
                    found.setdefault(key, []).append(Path(entry.path))
        except OSError:
            continue
    return found


def _character_keys(session: Session, character: Character) -> set:
    keys = {clean_character_key(character.name)}
    for alias in session.exec(
            select(CharacterAlias).where(CharacterAlias.character_id == character.id)).all():
        keys.add(clean_character_key(alias.alias))
    keys.discard("")
    return keys


def _row_names_character(row: Dict[str, Any], keys: set) -> bool:
    for tag in _TAG.findall(row.get("prompt") or ""):
        if clean_character_key(tag) in keys:
            return True
    return False


def change_scope(change: Dict[str, Any], *, events: Sequence[CharacterTimelineEvent],
                 books: Sequence[Book], pinned: Iterable[str],
                 carried_mode: str) -> Optional[dict]:
    """Where one trait edit reaches, as ``{"kind": "all"}`` or ``{"kind": "range"}``.

    ``None`` means the field never reaches a rendered prompt (``notes``, legacy
    fields), so no image can be stale. ``change`` carries ``field``, ``book_id``,
    ``chapter``, ``shot`` and the new ``value``.
    """
    field = change.get("field")
    if field not in RENDERED_TRAIT_FIELDS:
        return None
    if field in set(pinned) or field in ("sex", "species"):
        return {"kind": "all"}
    if change.get("book_id") is None:
        # Base edits propagate through every volume, and a side story with no
        # event of its own resolves Base directly - the whole project shifts.
        return {"kind": "all"}

    order_map = {b.id: (b.book_order if b.book_order is not None else 10 ** 6)
                 for b in books}
    side_books = {b.id for b in books if getattr(b, "is_side_story", False)}
    start = _reading_key(order_map, change["book_id"], change["chapter"], change["shot"])

    if field == "carried" and carried_mode == "scene":
        # Rendered only at the position the text recorded, so only that shot moved.
        return {"kind": "range", "start": start,
                "end": (start[0], start[1], start[2] + 1),
                "edit_book": change["book_id"], "side_books": side_books}

    new_norm = (change.get("value") or "").strip().lower()
    end = None
    for event in events:
        if event.book_id is None:
            continue
        if event.book_id in side_books and event.book_id != change["book_id"]:
            continue            # invisible to this book, so it cannot end the span
        key = _reading_key(order_map, event.book_id, event.chapter_num, event.shot_num)
        if key <= start:
            continue
        current = (getattr(event, field, None) or "").strip().lower()
        if current and current != new_norm:
            end = key
            break
    return {"kind": "range", "start": start, "end": end,
            "edit_book": change["book_id"], "side_books": side_books}


def _book_may_match(scope: dict, order_map: Dict[int, int], book_id: int) -> bool:
    if scope["kind"] == "all":
        return True
    if book_id in scope.get("side_books", ()) and book_id != scope.get("edit_book"):
        return False
    order = order_map.get(book_id, 10 ** 6)
    if order < scope["start"][0]:
        return False
    if scope["end"] is not None and order > scope["end"][0]:
        return False
    return True


def _in_scope(scope: dict, order_map: Dict[int, int], book_id: int,
              chapter: int, shot: int) -> bool:
    if scope["kind"] == "all":
        return True
    key = _reading_key(order_map, book_id, chapter, shot)
    if key < scope["start"]:
        return False
    return scope["end"] is None or key < scope["end"]


def plan_stale_renders(project_id: int, character_id: int,
                       changes: Sequence[Dict[str, Any]],
                       current_book_id: Optional[int] = None) -> dict:
    """Work out which images the given trait edits made stale.

    Returns ``{"entries": [...], "by_book": [...]}``. Each entry is one image-
    bearing scene: ``book_id``, ``book_name``, ``chapter``, ``shot``, ``files``.
    Nothing is written - the caller shows the counts and confirms.
    """
    empty = {"entries": [], "by_book": []}
    if not changes:
        return empty

    with Session(engine) as session:
        character = session.get(Character, character_id)
        project = session.get(Project, project_id)
        if character is None or project is None:
            return empty
        books = sorted(
            session.exec(select(Book).where(Book.project_id == project_id)).all(),
            key=lambda b: (b.book_order if b.book_order is not None else 10 ** 6))
        events = session.exec(
            select(CharacterTimelineEvent)
            .where(CharacterTimelineEvent.character_id == character_id)).all()
        pinned = get_pinned_fields(character)
        keys = _character_keys(session, character)
        order_map = {b.id: (b.book_order if b.book_order is not None else 10 ** 6)
                     for b in books}
        project_name = project.name
        carried_mode = str(getattr(state, "carried_mode", "scene") or "scene")

    scopes = [s for s in (change_scope(c, events=events, books=books, pinned=pinned,
                                      carried_mode=carried_mode) for c in changes) if s]
    if not scopes or not keys:
        return empty

    # Local import: ``services`` cannot import a UI module at load time (the
    # workspace imports this one on the way in), and these helpers already own the
    # pipe format and the ScenePrompt cache sync.
    from ui.pages.book_workspace import load_prompts_csv

    entries: List[Dict[str, Any]] = []
    for book in books:
        if not any(_book_may_match(s, order_map, book.id) for s in scopes):
            continue
        rows = load_prompts_csv(project_name, book.name)
        if not rows:
            continue
        images = _scan_book_images(project_name, book.name)
        for row in rows:
            coords = _coords(row)
            if coords is None or coords not in images:
                continue
            chapter, shot = coords
            if not any(_in_scope(s, order_map, book.id, chapter, shot) for s in scopes):
                continue
            if not _row_names_character(row, keys):
                continue
            entries.append({
                "book_id": book.id,
                "book_name": book.name,
                "chapter": chapter,
                "shot": shot,
                "files": [str(p) for p in images[coords]],
            })

    by_book: List[Dict[str, Any]] = []
    for book in books:
        book_entries = [e for e in entries if e["book_id"] == book.id]
        if book_entries:
            by_book.append({
                "book_id": book.id,
                "book_name": book.name,
                "count": sum(len(e["files"]) for e in book_entries),
                "scenes": len(book_entries),
            })
    return {"entries": entries, "by_book": by_book}


def apply_stale_render_purge(project_id: int, entries: Sequence[Dict[str, Any]]) -> dict:
    """Delete the planned image files and clear approval on their scenes.

    Returns ``{"images": deleted files, "scenes": scenes whose approval was
    cleared, "books": books touched, "approved_rows": [...]}``. The approval flag
    lives in ``prompts.csv`` and is cleared through the workspace helpers, so the
    ``ScenePrompt`` cache is resynced the same way every other edit is.
    """
    if not entries:
        return {"images": 0, "scenes": 0, "books": 0, "rows": []}

    from ui.pages.book_workspace import load_prompts_csv, save_prompts_csv

    with Session(engine) as session:
        project = session.get(Project, project_id)
        books = {b.id: b for b in session.exec(
            select(Book).where(Book.project_id == project_id)).all()}
    if project is None:
        return {"images": 0, "scenes": 0, "books": 0, "rows": []}

    if not hasattr(state, "custom_image_timestamps"):
        state.custom_image_timestamps = {}

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(entry["book_id"], []).append(entry)

    deleted_images = 0
    approved_rows = 0
    touched_books = 0
    changed_rows: List[tuple] = []

    for book_id, group in grouped.items():
        book = books.get(book_id)
        if book is None:
            continue
        deleted_before = deleted_images
        cleared_before = approved_rows

        for entry in group:
            changed_rows.append((book_id, entry["chapter"], entry["shot"]))
            for file_path in entry["files"]:
                try:
                    Path(file_path).unlink()
                    deleted_images += 1
                except FileNotFoundError:
                    pass
                except OSError as ex:
                    print(f"[StaleRenders] Could not delete {file_path}: {ex}")
            state.custom_image_timestamps[(entry["chapter"], entry["shot"])] = int(
                time.time() * 1000)

        coords = {(entry["chapter"], entry["shot"]) for entry in group}
        rows = load_prompts_csv(project.name, book.name)
        cleared = 0
        for row in rows:
            if _coords(row) not in coords:
                continue
            if (row.get("approved") or "").strip().lower() != "true":
                continue
            row["approved"] = "False"
            cleared += 1
        if cleared:
            save_prompts_csv(project.name, book.name, rows)
            approved_rows += cleared

        if deleted_images > deleted_before or approved_rows > cleared_before:
            touched_books += 1

    return {
        "images": deleted_images,
        "scenes": approved_rows,
        "books": touched_books,
        "rows": changed_rows,
    }
