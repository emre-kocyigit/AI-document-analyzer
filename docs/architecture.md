# Architecture Documentation

## System Overview

AI Document Analyzer is a production-grade document analysis API that combines traditional NLP with modern LLM capabilities. The system processes uploaded documents through multiple analysis pipelines and provides a RAG-based question-answering interface.

## Core Components

### 1. FastAPI Application Layer
The API gateway handles request validation, file uploads, and response formatting. Built with FastAPI for automatic OpenAPI documentation, async support, and Pydantic-based request/response validation.

### 2. Document Processor
Responsible for file handling, text extraction (PDF and plaintext), and text chunking using LangChain's RecursiveCharacterTextSplitter. Chunks are sized for optimal embedding and retrieval performance.

### 3. ONNX Inference Engine
Runs Named Entity Recognition (NER) and zero-shot text classification using ONNX-optimized transformer models. ONNX Runtime provides significant inference speedups over standard PyTorch/TensorFlow, making the system suitable for production and edge deployment scenarios.

**Models used:**
- NER: `dslim/bert-base-NER` (exported to ONNX)
- Classification: `typeform/distilbert-base-uncased-mnli` (exported to ONNX)

### 4. RAG Pipeline
Implements Retrieval-Augmented Generation using:
- **Embedding**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Vector Store**: ChromaDB with cosine similarity
- **Generation**: Ollama running Mistral

The pipeline indexes document chunks, retrieves relevant context for queries, and constructs grounded prompts for the LLM.

### 5. LLM Service
Communicates with Ollama via HTTP API for document summarization and RAG-based question answering. Designed with async support and configurable timeouts.

## Data Flow

```
Upload → Extract Text → Chunk Text ─┬─→ ONNX NER → Entities
                                     ├─→ ONNX Classification → Categories
                                     ├─→ ChromaDB Indexing → Vector Store
                                     └─→ LLM Summarization → Summary

Question → Embed Query → ChromaDB Retrieval → Context + Prompt → Ollama → Answer
```

## Deployment Architecture

### Local / Docker Compose
- `app` container: FastAPI + all NLP services
- `ollama` container: LLM inference server
- Shared network for inter-service communication
- Volume mounts for persistent storage

### Azure (Stage 2)
- Azure Container Apps for application hosting
- Azure Blob Storage for document persistence
- Azure Functions for event-driven processing
- Azure Container Registry for image management

## Design Decisions

1. **ONNX over raw PyTorch**: Chosen for inference optimization — ONNX Runtime provides 2-3x speedup for transformer models, critical for production latency requirements.

2. **ChromaDB over FAISS**: Simpler API, built-in persistence, and metadata filtering. Suitable for document-scale collections without the complexity of FAISS index management.

3. **Ollama over direct API calls**: Enables fully local LLM inference with no external API dependencies, important for data privacy and cost control.

4. **Lazy model loading**: Models are loaded on first use rather than at startup, reducing cold-start time and memory usage when not all features are needed.
