import time
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
    modified_at: float = Field(default_factory=time.time)

class Book(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")
    name: str = Field(index=True)
    path: str
    cover_path: Optional[str] = Field(default=None)
    status: str = Field(default="Imported")
    progress: float = Field(default=0.0)
    word_count: Optional[int] = Field(default=None)
    total_images: Optional[int] = Field(default=None)
    completed_images: Optional[int] = Field(default=None)
    prompts_mtime: Optional[float] = Field(default=None)  # Dynamic cache syncing timestamp
    duration: Optional[float] = Field(default=None)       # Cached audiobook total duration in seconds

class Chapter(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    book_id: int = Field(foreign_key="book.id")
    chapter_num: int
    title: str
    input_file: Optional[str] = Field(default=None)  # Optional path to original source audio (None for EPUB/text)
    type: str = Field(default="segment")            # 'file', 'segment', or 'text'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = Field(default="Pending")          # Pending, Transcribing, Completed, Failed
    word_count: Optional[int] = Field(default=None)
    total_images: Optional[int] = Field(default=None)
    completed_images: Optional[int] = Field(default=None)

class ScenePrompt(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    book_id: int = Field(foreign_key="book.id", index=True)
    chapter_num: int = Field(index=True)
    scene_num: int = Field(index=True)
    prompt: str
    quote: str
    approved: bool = Field(default=False)
    timestamp: Optional[str] = Field(default="00:00:00")

class Character(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    book_id: Optional[int] = Field(default=None, foreign_key="book.id", index=True) # Null if global/static, Book ID if book-specific/dynamic
    name: str = Field(index=True)
    
    # Simplified 4-Field Physical Descriptor Schema (Cohesive buckets)
    demographics: Optional[str] = Field(default=None)         # e.g., "middle-aged Caucasian man" or "young Italian woman"
    physical_build: Optional[str] = Field(default=None)       # e.g., "six-foot-two and athletic" or "petite and slender"
    hair_and_face: Optional[str] = Field(default=None)        # e.g., "with thinning brown hair and a clean-shaven face"
    distinguishing_marks: Optional[str] = Field(default=None) # e.g., "wearing wire-rimmed glasses" or "with a scar on his cheek"

    # The compiled natural-language text-to-image replacement string
    visual_description: Optional[str] = Field(default=None)
    
    is_dynamic: bool = Field(default=False) # True if scoped to a book, False if static/project-global
    locked: bool = Field(default=False)     # True if manual curation should protect this profile from LLM extraction or compile overwrites

class CharacterAlias(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    character_id: int = Field(foreign_key="character.id", index=True)
    alias: str = Field(index=True)  # Raw tag text without brackets, e.g., 'Dino' or 'Stone'

class CharacterStateModifier(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    character_id: int = Field(foreign_key="character.id", index=True)
    book_id: int = Field(foreign_key="book.id", index=True) # Modifiers are inherently chapter-based inside a specific book
    name: str                      # e.g., 'Broken Arm Cast' or 'Gandalf the White'
    modifier_text: str             # e.g., 'wearing a plaster cast on his left arm'
    start_chapter: int
    end_chapter: int
    is_permanent: bool = Field(default=False) # If True, active from start_chapter through the remainder of the book