import json
import os
from sqlmodel import SQLModel, create_engine, Session, select
from .models import Setting

DATABASE_FILE = "abi_pipeline.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

# Connect sqlite with write_lock timeout for stability during concurrent operations
engine = create_engine(DATABASE_URL, connect_args={"timeout": 15})

def init_db():
    """Initializes the SQLite database and creates the tables if they don't exist."""
    SQLModel.metadata.create_all(engine)

def get_setting(key: str, default=None):
    """Retrieves a setting by key. Automatically parses JSON strings into Python dicts/lists."""
    with Session(engine) as session:
        statement = select(Setting).where(Setting.key == key)
        setting = session.exec(statement).first()
        if not setting:
            return default
        
        # Try to parse as JSON, otherwise return the raw string
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