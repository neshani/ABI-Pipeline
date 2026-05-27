from typing import Optional
from sqlmodel import SQLModel, Field

class Setting(SQLModel, table=True):
    key: str = Field(primary_key=True, description="The unique name of the setting configuration.")
    value: str = Field(description="The string or JSON-serialized value of the setting.")

class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    path: str
    status: str = Field(default="Imported")
    is_batch: bool = Field(default=False)

class Book(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")
    name: str = Field(index=True)
    path: str
    cover_path: Optional[str] = Field(default=None)
    status: str = Field(default="Imported")
    progress: float = Field(default=0.0)

class Chapter(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    book_id: int = Field(foreign_key="book.id")
    chapter_num: int
    title: str
    input_file: str  # Path to original source audio file
    type: str        # 'file' (multi-file) or 'segment' (m4b chapters)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = Field(default="Pending") # Pending, Transcribing, Completed, Failed