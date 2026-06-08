import json
import os
from sqlmodel import SQLModel, create_engine, Session, select
from .models import Setting
from typing import Optional

DATABASE_FILE = "abi_pipeline.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

# Connect sqlite with write_lock timeout for stability during concurrent operations
engine = create_engine(DATABASE_URL, connect_args={"timeout": 15})

def init_db():
    """Initializes the SQLite database and creates the tables if they don't exist."""
    SQLModel.metadata.create_all(engine)

def get_setting(key: str, default=None, session: Optional[Session] = None):
    """Retrieves a setting by key. Automatically parses JSON strings into Python dicts/lists."""
    # Reuse open active session if provided to avoid connection locks
    if session is not None:
        statement = select(Setting).where(Setting.key == key)
        setting = session.exec(statement).first()
        if not setting:
            return default
        try:
            return json.loads(setting.value)
        except (json.JSONDecodeError, TypeError):
            return setting.value

    # Standalone connection block fallback
    with Session(engine) as standalone_session:
        statement = select(Setting).where(Setting.key == key)
        setting = standalone_session.exec(statement).first()
        if not setting:
            return default
        try:
            return json.loads(setting.value)
        except (json.JSONDecodeError, TypeError):
            return setting.value

def set_setting(key: str, value) -> None:
    """Saves or updates a setting. Automatically serializes dicts/lists into JSON strings."""
    # Convert dictionaries or lists to JSON strings for database storage
    if isinstance(value, (dict, list)):
        serialized_value = json.dumps(value)
    else:
        serialized_value = str(value)

    with Session(engine) as session:
        statement = select(Setting).where(Setting.key == key)
        setting = session.exec(statement).first()
        
        if setting:
            setting.value = serialized_value
        else:
            setting = Setting(key=key, value=serialized_value)
            
        session.add(setting)
        session.commit()

def touch_project(project_id: int, session: Optional[Session] = None) -> None:
    """Updates the project's modified_at timestamp to the current time both in DB and on disk."""
    import time
    from .models import Project

    def _touch(sess: Session):
        project = sess.get(Project, project_id)
        if project:
            project.modified_at = time.time()
            sess.add(project)
            sess.commit()

    if session:
        _touch(session)
    else:
        with Session(engine) as standalone_session:
            _touch(standalone_session)

    # Persist the updated modified_at directly into the on-disk config JSON
    try:
        from services.project_settings import save_project_settings_to_disk
        save_project_settings_to_disk(project_id)
    except Exception:
        pass