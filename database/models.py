from typing import Optional
from sqlmodel import SQLModel, Field

class Setting(SQLModel, table=True):
    key: str = Field(primary_key=True, description="The unique name of the setting configuration.")
    value: str = Field(description="The string or JSON-serialized value of the setting.")