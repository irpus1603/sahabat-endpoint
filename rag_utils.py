"""
RAG (Retrieval-Augmented Generation) utilities
"""
from typing import List, Tuple
import numpy as np
from logger import setup_logger

logger = setup_logger(__name__)


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks

    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # If this is not the last chunk, try to break at a sentence or word boundary
        if end < text_length:
            # Look for sentence boundaries
            for delimiter in ['. ', '! ', '? ', '\n\n', '\n']:
                last_delimiter = text.rfind(delimiter, start, end)
                if last_delimiter != -1 and last_delimiter > start + chunk_size // 2:
                    end = last_delimiter + len(delimiter)
                    break
            else:
                # If no sentence boundary, look for word boundary
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start + chunk_size // 2:
                    end = last_space + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = end - chunk_overlap if end < text_length else text_length

    return chunks


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_relevant_documents(
    query_embedding: List[float],
    document_embeddings: List[List[float]],
    documents: List[str],
    top_k: int = 3
) -> Tuple[List[str], List[float]]:
    """
    Retrieve most relevant documents based on embeddings

    Args:
        query_embedding: Embedding of the query
        document_embeddings: List of document embeddings
        documents: List of original documents
        top_k: Number of top documents to retrieve

    Returns:
        Tuple of (retrieved_documents, relevance_scores)
    """
    query_emb = np.array(query_embedding)
    doc_embs = np.array(document_embeddings)

    # Calculate similarities
    similarities = []
    for doc_emb in doc_embs:
        sim = cosine_similarity(query_emb, doc_emb)
        similarities.append(sim)

    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    retrieved_docs = [documents[i] for i in top_indices]
    scores = [float(similarities[i]) for i in top_indices]

    return retrieved_docs, scores


def create_rag_prompt(query: str, context_documents: List[str]) -> str:
    """
    Create a prompt for RAG-based generation

    Args:
        query: User query
        context_documents: Retrieved relevant documents

    Returns:
        Formatted prompt
    """
    context = "\n\n".join([f"Dokumen {i+1}: {doc}" for i, doc in enumerate(context_documents)])

    prompt = f"""Berikut adalah beberapa dokumen yang relevan:

{context}

Berdasarkan dokumen di atas, jawab pertanyaan berikut:
{query}

Jawaban:"""

    return prompt
