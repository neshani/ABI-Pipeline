"""Idempotent schema migrations for existing user databases.

The project's Files-as-Source-of-Truth pattern heals *data*: delete
``abi_pipeline.db`` and it re-syncs from ``output/``. It does **not** heal
*schema*. A user upgrading the app keeps their existing database, and
``SQLModel.metadata.create_all()`` only creates tables that are missing -- it
never renames a column that already exists. So every column rename gets an
entry here, and every entry must be safe to run on every startup.

This module is the one place that knows about retired column names. Anything
that reads *files* written before a rename is handled separately by
``services/compat.py``.

Vocabulary (do not blur these):
    shot  -- the image unit. One chunk of text renders to one image. -> shot_num
    scene -- the narrative unit. One or more shots belong to a scene. -> scene_num

Retired names:
    scene_num (on sceneprompt, charactertimelineevent) -> shot_num
        Historic naming called the image unit a "scene". It is a shot.
"""

from sqlalchemy.exc import SQLAlchemyError

# (table, retired_column, current_column)
COLUMN_RENAMES = [
    ("sceneprompt", "scene_num", "shot_num"),
    ("charactertimelineevent", "scene_num", "shot_num"),
]

# Retired index NAMES. When SQLite renames a column it rewrites the SQL of any
# index that covers it but leaves the index's own name alone, so a database can
# end up with ``ix_..._scene_num`` pointing at ``shot_num``. Cosmetic, but a
# stale name in the schema is exactly what misleads a reader later.
# The stored SQL is reused verbatim with only the name substituted.
INDEX_RENAMES = [
    ("ix_sceneprompt_scene_num", "ix_sceneprompt_shot_num"),
    ("ix_charactertimelineevent_scene_num", "ix_charactertimelineevent_shot_num"),
]


def _table_exists(conn, table: str) -> bool:
    row = conn.exec_driver_sql(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _columns(conn, table: str) -> set:
    rows = conn.exec_driver_sql(f'PRAGMA table_info("{table}")').fetchall()
    return {r[1] for r in rows}


def _rename_indexes(conn) -> list:
    """Recreate retired index names, reusing their stored SQL verbatim."""
    applied = []
    for old, new in INDEX_RENAMES:
        row = conn.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name = ?", (old,)
        ).fetchone()
        if not row or not row[0]:
            continue
        if conn.exec_driver_sql(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name = ?", (new,)
        ).fetchone():
            continue
        conn.exec_driver_sql(f'DROP INDEX "{old}"')
        conn.exec_driver_sql(row[0].replace(old, new, 1))
        applied.append(f"index {old} -> {new}")
    return applied


def apply_migrations(engine) -> list:
    """Rename retired columns in place. Returns a list of human-readable changes.

    Safe to call on every startup:
      * a fresh database already has the current column, so nothing matches;
      * an already-migrated database has no retired column, so nothing matches;
      * a half-migrated table (both columns present) is left alone rather than
        guessed at.

    A failure here must never prevent the app from starting -- callers treat the
    return value as informational and the caller wraps this in try/except.
    """
    applied = []
    try:
        with engine.begin() as conn:
            for table, retired, current in COLUMN_RENAMES:
                if not _table_exists(conn, table):
                    continue
                cols = _columns(conn, table)
                if retired not in cols or current in cols:
                    continue
                conn.exec_driver_sql(
                    f'ALTER TABLE "{table}" RENAME COLUMN "{retired}" TO "{current}"'
                )
                applied.append(f"{table}.{retired} -> {current}")
            applied.extend(_rename_indexes(conn))
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        print(f"[Migrations] column rename failed, continuing: {exc}")
    return applied
