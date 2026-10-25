"""One-time repair of trait values written before the vocabulary existed.

Extraction now normalises on the way in (``services/trait_vocab.py``), but every
character already in a project still stores whatever the old extractor wrote:
builds like ``170 pounds``, ages like ``young`` for a nine-year-old. This walks
those rows once and rewrites them to the canonical values.

Dry run by default - nothing is written and no file is touched:

    <conda python> normalize_traits.py --project "The Wheel of Time"

To write:

    <conda python> normalize_traits.py --project "The Wheel of Time" --apply

With ``--apply`` the database is copied to
``abi_pipeline.db.pre_traits_<timestamp>`` first and ``characters.json`` is
regenerated afterwards, so the change is reversible and the on-disk snapshot
cannot go stale. Rows whose value the vocabulary cannot classify are listed and
left alone - a value is never silently dropped from the report.
"""
from __future__ import annotations

import argparse
import collections
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlmodel import Session, select

from database.connection import engine
from database.models import Book, Character, CharacterTimelineEvent, Project
from services import trait_vocab
from services.character_manager import (NEW_TRAIT_FIELDS, RENDERED_TRAIT_FIELDS,
                                        compile_character_visual_prompt,
                                        save_project_characters_to_json)

FIELDS = ("build", "age")
# A stored description is the rendered override, so it has to be recompiled with
# the traits or the change never reaches an image. Everything the compiler needs:
_LEGACY = ("demographics", "physical_build", "hair_and_face", "distinguishing_marks")


def _probe(event, **overrides) -> SimpleNamespace:
    """A detached copy of the trait fields, for compiling hypothetical text."""
    values = {field: getattr(event, field, None) for field in NEW_TRAIT_FIELDS}
    values.update({field: getattr(event, field, None) for field in _LEGACY})
    values.update(overrides)
    return SimpleNamespace(**values)


def _is_compiled(event, name: str, build: str, age: str) -> bool:
    """Is ``visual_description`` the compiler's own output, or hand-written prose?

    Only a compiled description is safe to regenerate. A name prefix or a bracket
    means somebody wrote it deliberately, and it is left alone.
    """
    stored = (getattr(event, "visual_description", "") or "").strip()
    if not stored:
        return False
    if stored.startswith(name) or "(" in stored:
        return False
    expected = compile_character_visual_prompt(_probe(event, build=build, age=age),
                                               name=name)
    return expected.strip() == stored


def find_project(session: Session, name: str, project_id: int | None) -> Project:
    if project_id is not None:
        project = session.get(Project, project_id)
        if project is None:
            raise SystemExit(f"no project with id {project_id}")
        return project
    matches = session.exec(select(Project).where(Project.name.contains(name))).all()
    if not matches:
        raise SystemExit(f"no project matching {name!r}")
    if len(matches) > 1:
        listing = "\n".join(f"  {p.id}  {p.name}" for p in matches)
        raise SystemExit(f"{len(matches)} projects match {name!r}:\n{listing}\n"
                         "Re-run with --project-id.")
    return matches[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="", help="project name (substring)")
    ap.add_argument("--project-id", type=int, default=None)
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default is a dry run)")
    ap.add_argument("--show", type=int, default=40,
                    help="how many individual rewrites to print (default 40)")
    ap.add_argument("--old-db", default="",
                    help="read pre-change trait values from this database copy "
                         "(needed when the trait pass has already been applied)")
    args = ap.parse_args()

    if not args.project and args.project_id is None:
        ap.error("give --project or --project-id")

    # Pre-change values, by event id, when the caller tells us where they were.
    old_values = {}
    if args.old_db:
        if not Path(args.old_db).exists():
            raise SystemExit(f"no such database copy: {args.old_db}")
        con = sqlite3.connect(args.old_db)
        old_values = {row[0]: (row[1], row[2]) for row in con.execute(
            "select id, build, age from charactertimelineevent")}
        con.close()
        print(f"old values : {len(old_values)} rows from {args.old_db}\n")

    with Session(engine) as session:
        project = find_project(session, args.project, args.project_id)
        characters = session.exec(
            select(Character).where(Character.project_id == project.id)).all()
        ids = [c.id for c in characters]
        events = session.exec(
            select(CharacterTimelineEvent)
            .where(CharacterTimelineEvent.character_id.in_(ids))
            .order_by(CharacterTimelineEvent.character_id,
                      CharacterTimelineEvent.book_id,
                      CharacterTimelineEvent.chapter_num,
                      CharacterTimelineEvent.shot_num)).all() if ids else []
        books = {b.id: b.name for b in session.exec(
            select(Book).where(Book.project_id == project.id)).all()}
        names = {c.id: c.name for c in characters}

        print(f"project : {project.name}  (id {project.id})")
        print(f"rows    : {len(characters)} characters, {len(events)} timeline events")
        print(f"mode    : {'APPLY' if args.apply else 'dry run - nothing will be written'}\n")

        changes: collections.Counter = collections.Counter()
        unknown: collections.Counter = collections.Counter()
        by_source: collections.Counter = collections.Counter()
        samples: list[str] = []
        legacy_only = 0
        total_changes = 0
        pending = []
        rewrites: list[str] = []
        compiled_refreshed = 0
        hand_written = 0

        for ev in events:
            name = names.get(ev.character_id, "")
            has_new = any(getattr(ev, f) for f in
                          ("sex", "species", "age", "hair_color", "hair_style",
                           "eye_color", "face", "build", "attire", "carried"))
            if not has_new and ev.physical_build:
                legacy_only += 1
                continue

            # The values in force before this run: what is stored now, unless a
            # previous pass already moved them and a copy says what they were.
            old_build, old_age = old_values.get(ev.id, (ev.build, ev.age))
            was_compiled = _is_compiled(ev, name, old_build, old_age)

            row = []
            for field in FIELDS:
                before = (getattr(ev, field) or "").strip()
                if not before:
                    continue
                after_value = trait_vocab.normalise_value(field, before)
                if after_value is None:
                    unknown[(field, before)] += 1
                    continue
                after = after_value.strip()
                if after == before:
                    continue
                row.append((field, before, after))
                changes[(field, before, after)] += 1

            # A compiled description is regenerated from the traits: the renderer
            # prefers it over them, so skipping it would make the whole pass a
            # no-op. Hand-written wording is reported, never overwritten.
            new_description = None
            if was_compiled:
                after_values = {field: after for field, _b, after in row}
                candidate = compile_character_visual_prompt(
                    _probe(ev, **after_values), name=name).strip()
                if candidate and candidate != (ev.visual_description or "").strip():
                    new_description = candidate
            elif row and ev.visual_description:
                hand_written += 1

            if row or new_description:
                by_source[ev.source or "?"] += 1
                where = (f"{names.get(ev.character_id, '?')} @ "
                         f"{books.get(ev.book_id, 'Base') if ev.book_id else 'Base'}"
                         f" ch{ev.chapter_num} sh{ev.shot_num}")
                for field, before, after in row:
                    total_changes += 1
                    if len(samples) < args.show:
                        samples.append(f"  {where:52s} {trait_vocab.describe(field, before, after)}")
                if new_description:
                    compiled_refreshed += 1
                    if len(rewrites) < args.show:
                        rewrites.append(
                            f"  {where:52s} description -> {new_description[:90]}")
                pending.append((ev, row, new_description))

        if not pending and not unknown:
            print("nothing to change - every stored value is already canonical.")
            return 0

        print("--- field changes ---")
        for (field, before, after), count in sorted(
                changes.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {count:4d}  {trait_vocab.describe(field, before, after)}")

        if samples:
            print(f"\n--- first {len(samples)} rewrites ---")
            print("\n".join(samples))
            if total_changes > len(samples):
                print(f"  ... {total_changes - len(samples)} more rewrites")

        print(f"\n--- descriptions ---")
        print(f"  {compiled_refreshed} compiled descriptions regenerated from the "
              f"normalised traits")
        print(f"  {hand_written} rows keep hand-written wording (left alone)")
        if rewrites:
            print("\n".join(rewrites))
            if compiled_refreshed > len(rewrites):
                print(f"  ... {compiled_refreshed - len(rewrites)} more")

        if unknown:
            print(f"\n--- left alone: {sum(unknown.values())} values the vocabulary "
                  f"does not classify ---")
            for (field, value), count in unknown.most_common():
                print(f"  {count:4d}  {field}: {value!r}")

        if legacy_only:
            print(f"\nnote: {legacy_only} rows carry only the legacy profile "
                  f"(demographics/physical_build). Those are not rewritten - the "
                  f"renderer falls back to them verbatim.")

        print(f"\nrows touched by source: " + ", ".join(
            f"{k}={v}" for k, v in sorted(by_source.items())))

        if not args.apply:
            print("\ndry run: re-run with --apply to write these changes.")
            return 0

        # --- write -----------------------------------------------------------
        db_path = Path("abi_pipeline.db")
        if db_path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = db_path.with_name(f"abi_pipeline.db.pre_traits_{stamp}")
            shutil.copy2(db_path, backup)
            print(f"\ndatabase backed up to {backup.name}")

        for ev, row, new_description in pending:
            for field, _before, after in row:
                setattr(ev, field, after or None)
            if new_description:
                ev.visual_description = new_description
            session.add(ev)
        session.commit()

        save_project_characters_to_json(project.id)
        print("characters.json regenerated.")
        print(f"wrote {len(pending)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
