"""Pydantic request/response models for the Agent Zero API."""

from pydantic import BaseModel, ConfigDict, model_validator


# -- Response models --

class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    version: str
    voice: str | None = None
    ctx_size: int | None = None


class FileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    tags: list[str]
    project: str
    created: str
    last_modified: str
    source: str


class FileContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    content: str
    source: str


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    source: str
    heading: str = ""
    summary: str = ""
    matching_lines: list[str] = []


class SaveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    message: str


class ClaudeMdGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_name: str
    content: str


class ClaudeMdWriteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str


# -- Request models --

class SaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    content: str
    tags: list[str]
    project: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str
    session_id: str | None = None
    agent: str = "fast"  # "fast" | "heavy"


class ClaudeMdGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_name: str


class ClaudeMdWriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_path: str
    project_name: str | None = None
    content: str | None = None

    @model_validator(mode="after")
    def check_exclusive(self):
        if self.project_name and self.content:
            raise ValueError("provide project_name or content, not both")
        if not self.project_name and not self.content:
            raise ValueError("provide either project_name or content")
        return self
