"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class HealthStatus(str, Enum):
    """Health status enum"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Health check response"""
    status: HealthStatus
    model_loaded: bool
    version: str
    device: str


class GenerateRequest(BaseModel):
    """Text generation request"""
    prompt: str = Field(..., min_length=1, description="Input prompt for generation")
    max_new_tokens: Optional[int] = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: Optional[int] = Field(50, ge=0, description="Top-k sampling parameter")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    repetition_penalty: Optional[float] = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    stream: Optional[bool] = Field(False, description="Stream response")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Apa itu kecerdasan buatan?",
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }


class GenerateResponse(BaseModel):
    """Text generation response"""
    generated_text: str
    prompt: str
    tokens_generated: int
    finish_reason: str


class EmbeddingRequest(BaseModel):
    """Embedding generation request"""
    texts: List[str] = Field(..., min_length=1, description="List of texts to embed")
    normalize: Optional[bool] = Field(True, description="Normalize embeddings")

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Halo dunia", "Kecerdasan buatan"],
                "normalize": True
            }
        }


class EmbeddingResponse(BaseModel):
    """Embedding generation response"""
    embeddings: List[List[float]]
    model: str
    dimensions: int


class ChunkRequest(BaseModel):
    """Text chunking request for RAG"""
    text: str = Field(..., min_length=1, description="Text to chunk")
    chunk_size: Optional[int] = Field(512, ge=100, le=2048, description="Size of each chunk")
    chunk_overlap: Optional[int] = Field(50, ge=0, le=500, description="Overlap between chunks")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Long document text here...",
                "chunk_size": 512,
                "chunk_overlap": 50
            }
        }


class ChunkResponse(BaseModel):
    """Text chunking response"""
    chunks: List[str]
    num_chunks: int


class RAGRequest(BaseModel):
    """RAG query request"""
    query: str = Field(..., min_length=1, description="Query text")
    documents: List[str] = Field(..., min_length=1, description="List of documents to search")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of top documents to retrieve")
    max_new_tokens: Optional[int] = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Apa itu pembelajaran mesin?",
                "documents": [
                    "Pembelajaran mesin adalah cabang dari kecerdasan buatan.",
                    "Python adalah bahasa pemrograman yang populer.",
                    "Neural network adalah model pembelajaran mesin yang terinspirasi dari otak manusia."
                ],
                "top_k": 2
            }
        }


class RAGResponse(BaseModel):
    """RAG query response"""
    answer: str
    retrieved_documents: List[str]
    relevance_scores: List[float]


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    status_code: int
