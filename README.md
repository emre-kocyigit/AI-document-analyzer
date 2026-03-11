# AI Document Analyzer

An end-to-end LLM-powered document analysis API built with FastAPI, LangChain, Ollama, and ONNX Runtime. Upload documents, extract structured insights, and ask questions using RAG (Retrieval-Augmented Generation).

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│   Client     │     │              FastAPI Application                 │
│  (REST API)  │────▶│                                                  │
└─────────────┘     │  ┌─────────────┐  ┌───────────────────────────┐  │
                    │  │  Document    │  │   RAG Pipeline            │  │
                    │  │  Processor   │  │  ┌─────────┐ ┌────────┐  │  │
                    │  │  - PDF/Text  │  │  │ChromaDB │ │ Ollama │  │  │
                    │  │  - Chunking  │  │  │(Vectors)│ │(Mistral)│  │  │
                    │  └──────┬──────┘  │  └─────────┘ └────────┘  │  │
                    │         │         └───────────────────────────┘  │
                    │         ▼                                        │
                    │  ┌─────────────┐  ┌───────────────────────────┐  │
                    │  │    ONNX     │  │   NLP Pipeline            │  │
                    │  │  Inference  │  │  - NER Extraction         │  │
                    │  │  (Optimized)│  │  - Text Classification    │  │
                    │  └─────────────┘  └───────────────────────────┘  │
                    └──────────────────────────────────────────────────┘
                    
                    ┌──────────────────────────────────────────────────┐
                    │              Docker Compose                       │
                    │  ┌──────────────┐  ┌──────────────────────────┐  │
                    │  │  app (FastAPI)│  │  ollama (LLM Server)    │  │
                    │  │  Port: 8000   │  │  Port: 11434            │  │
                    │  └──────────────┘  └──────────────────────────┘  │
                    └──────────────────────────────────────────────────┘
```

## Features

- **Document Upload & Processing**: Upload PDF or text files for automated analysis
- **Named Entity Recognition**: Extract entities (persons, organizations, locations, dates) using ONNX-optimized models
- **Text Classification**: Classify document type and content category
- **RAG-based Q&A**: Ask natural language questions about uploaded documents using LangChain + ChromaDB + Ollama (Mistral)
- **Production-Ready API**: FastAPI with automatic OpenAPI/Swagger documentation, input validation, and error handling
- **Containerized Deployment**: Docker and Docker Compose for reproducible environments
- **CI/CD Pipeline**: GitHub Actions for automated linting, testing, and Docker image builds

## Tech Stack

| Category | Tools |
|----------|-------|
| API Framework | FastAPI, Uvicorn |
| LLM | Ollama (Mistral), LangChain |
| Vector Store | ChromaDB |
| NLP/ML | Hugging Face Transformers, ONNX Runtime |
| Document Processing | PyPDF2, python-multipart |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest, httpx |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- (Optional) Python 3.11+ for local development

### Run with Docker Compose

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-document-analyzer.git
cd ai-document-analyzer

# Start all services
docker compose up --build

# The API will be available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Run Locally (Development)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running with Mistral
ollama pull mistral

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload and process a document |
| `GET` | `/api/v1/documents/{doc_id}` | Get document analysis results |
| `POST` | `/api/v1/documents/{doc_id}/ask` | Ask a question about a document (RAG) |
| `GET` | `/api/v1/health` | Health check endpoint |

## Project Structure

```
ai-document-analyzer/
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI/CD pipeline
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # API route definitions
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_processor.py  # PDF/text extraction & chunking
│   │   ├── rag_pipeline.py        # RAG with LangChain + ChromaDB
│   │   ├── onnx_inference.py      # ONNX-optimized NER & classification
│   │   └── llm_service.py         # Ollama/LLM integration
│   └── models/
│       ├── __init__.py
│       └── schemas.py          # Pydantic request/response models
├── azure/                      # Stage 2: Azure deployment configs
│   └── README.md
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_services.py
├── docs/
│   └── architecture.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Stage 2: Azure Deployment (Coming Soon)

- [ ] Azure Container Registry + Container Apps deployment
- [ ] Azure Blob Storage for document uploads
- [ ] Azure Function for event-driven document processing
- [ ] Model optimization benchmarks (ONNX quantization)
- [ ] Monitoring and observability

## License

MIT License
