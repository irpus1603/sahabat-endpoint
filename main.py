"""
FastAPI application for Sahabat-9B model endpoint
"""
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import time
import json
from typing import Dict, Any

from config import get_settings
from logger import setup_logger
from model_manager import model_manager
from models import (
    HealthResponse, HealthStatus, ErrorResponse,
    GenerateRequest, GenerateResponse,
    EmbeddingRequest, EmbeddingResponse,
    ChunkRequest, ChunkResponse,
    RAGRequest, RAGResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionChoice, ChatCompletionUsage, ChatMessage
)
from rag_utils import chunk_text, retrieve_relevant_documents, create_rag_prompt

settings = get_settings()
logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    try:
        model_manager.load_model()
        model_manager.load_embedding_model()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API endpoint for Sahabat-9B model with RAG capabilities",
    lifespan=lifespan
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
            status_code=exc.status_code
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.DEBUG else None,
            status_code=500
        ).model_dump()
    )


# Health endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status=HealthStatus.HEALTHY if model_manager.is_loaded() else HealthStatus.UNHEALTHY,
        model_loaded=model_manager.is_loaded(),
        version=settings.APP_VERSION,
        device=model_manager.device
    )


# Generation endpoint
@app.post(
    f"{settings.API_PREFIX}/generate",
    response_model=GenerateResponse,
    tags=["Generation"]
)
async def generate_text(request: GenerateRequest):
    """
    Generate text using Sahabat-9B model

    - **prompt**: Input text prompt
    - **max_new_tokens**: Maximum number of tokens to generate (default: 512)
    - **temperature**: Sampling temperature (default: 0.7)
    - **top_p**: Nucleus sampling probability (default: 0.9)
    - **top_k**: Top-k sampling parameter (default: 50)
    - **do_sample**: Whether to use sampling (default: true)
    - **repetition_penalty**: Penalty for repetition (default: 1.1)
    """
    try:
        result = model_manager.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
            repetition_penalty=request.repetition_penalty
        )

        return GenerateResponse(
            generated_text=result["generated_text"],
            prompt=request.prompt,
            tokens_generated=result["tokens_generated"],
            finish_reason=result["finish_reason"]
        )

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


# Embedding endpoint
@app.post(
    f"{settings.API_PREFIX}/embeddings",
    response_model=EmbeddingResponse,
    tags=["RAG"]
)
async def generate_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for texts

    - **texts**: List of texts to embed
    - **normalize**: Whether to normalize embeddings (default: true)
    """
    try:
        embeddings = model_manager.get_embeddings(
            texts=request.texts,
            normalize=request.normalize
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=settings.EMBEDDING_MODEL,
            dimensions=len(embeddings[0]) if embeddings else 0
        )

    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )


# Chunking endpoint
@app.post(
    f"{settings.API_PREFIX}/chunk",
    response_model=ChunkResponse,
    tags=["RAG"]
)
async def chunk_document(request: ChunkRequest):
    """
    Chunk text into smaller pieces for RAG

    - **text**: Text to chunk
    - **chunk_size**: Size of each chunk in characters (default: 512)
    - **chunk_overlap**: Overlap between chunks in characters (default: 50)
    """
    try:
        chunks = chunk_text(
            text=request.text,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        return ChunkResponse(
            chunks=chunks,
            num_chunks=len(chunks)
        )

    except Exception as e:
        logger.error(f"Chunking failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunking failed: {str(e)}"
        )


# RAG endpoint
@app.post(
    f"{settings.API_PREFIX}/rag",
    response_model=RAGResponse,
    tags=["RAG"]
)
async def rag_query(request: RAGRequest):
    """
    Perform RAG (Retrieval-Augmented Generation) query

    - **query**: User query
    - **documents**: List of documents to search
    - **top_k**: Number of top documents to retrieve (default: 3)
    - **max_new_tokens**: Maximum tokens to generate (default: 512)
    - **temperature**: Sampling temperature (default: 0.7)
    """
    try:
        # Generate embeddings for query and documents
        all_texts = [request.query] + request.documents
        embeddings = model_manager.get_embeddings(texts=all_texts, normalize=True)

        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]

        # Retrieve relevant documents
        retrieved_docs, scores = retrieve_relevant_documents(
            query_embedding=query_embedding,
            document_embeddings=doc_embeddings,
            documents=request.documents,
            top_k=request.top_k
        )

        # Create RAG prompt
        rag_prompt = create_rag_prompt(request.query, retrieved_docs)

        # Generate answer
        result = model_manager.generate(
            prompt=rag_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )

        return RAGResponse(
            answer=result["generated_text"],
            retrieved_documents=retrieved_docs,
            relevance_scores=scores
        )

    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {str(e)}"
        )


# OpenAI-compatible Chat Completions endpoint
@app.post(
    "/v1/chat/completions",
    tags=["Chat Completions"]
)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint

    - **model**: Model name (e.g., "Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-instruct")
    - **messages**: List of messages with role (system/user/assistant) and content
    - **max_tokens**: Maximum tokens to generate (default: 1024)
    - **temperature**: Sampling temperature (default: 0.7)
    - **top_p**: Nucleus sampling (default: 0.9)
    - **top_k**: Top-k sampling (default: 50)
    - **stream**: Enable streaming response (default: false)
    """
    try:
        import uuid
        from datetime import datetime

        # Convert Pydantic messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # Debug logging
        logger.info(f"Stream parameter received: {request.stream}")
        logger.info(f"Stream type: {type(request.stream)}")

        # Handle streaming
        if request.stream:
            async def generate_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                try:
                    for token in model_manager.chat_completion_stream(
                        messages=messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repetition_penalty=request.repetition_penalty
                    ):
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(datetime.now().timestamp()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # Send final chunk with finish_reason
                    final_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now().timestamp()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Streaming failed: {str(e)}")
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "server_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )

        # Non-streaming response
        result = model_manager.chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            stream=False,
            stream_options=request.stream_options
        )

        # Build OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=result["generated_text"]
                    ),
                    finish_reason=result["finish_reason"]
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                total_tokens=result["total_tokens"]
            )
        )

        return response

    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {str(e)}"
        )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "chat_completions": "/v1/chat/completions",
            "generate": f"{settings.API_PREFIX}/generate",
            "embeddings": f"{settings.API_PREFIX}/embeddings",
            "chunk": f"{settings.API_PREFIX}/chunk",
            "rag": f"{settings.API_PREFIX}/rag"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
