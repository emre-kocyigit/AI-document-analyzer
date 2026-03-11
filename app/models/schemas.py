from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    timestamp: datetime


class Entity(BaseModel):
    text: str
    label: str
    confidence: float


class ClassificationResult(BaseModel):
    label: str
    confidence: float


class DocumentAnalysis(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    char_count: int
    entities: list[Entity] = []
    classifications: list[ClassificationResult] = []
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    message: str
    analysis: DocumentAnalysis


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Question about the document")


class QuestionResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    source_chunks: list[str] = []
