import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from datetime import datetime
from app.config import get_settings
from app.models.schemas import (
    HealthResponse,
    DocumentUploadResponse,
    DocumentAnalysis,
    QuestionRequest,
    QuestionResponse,
)
from app.services.document_processor import DocumentProcessor
from app.services.onnx_inference import ONNXInferenceService
from app.services.llm_service import LLMService
from app.services.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1", tags=["Document Analysis"])

# Initialize services
doc_processor = DocumentProcessor()
onnx_service = ONNXInferenceService()
llm_service = LLMService()
rag_pipeline = RAGPipeline(llm_service)

# In-memory document store (replace with database in production)
document_store: dict[str, DocumentAnalysis] = {}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check application health and Ollama connectivity."""
    ollama_status = await llm_service.is_available()
    return HealthResponse(
        status="healthy" if ollama_status else "degraded (Ollama unavailable)",
        version=settings.app_version,
        timestamp=datetime.utcnow(),
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for analysis: text extraction, NER, classification, and RAG indexing."""

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = {"pdf", "txt", "md", "text"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Allowed: {allowed_extensions}",
        )

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max: {settings.max_file_size_mb}MB",
        )

    try:
        # Save and process document
        doc_id = doc_processor.generate_doc_id()
        file_path = await doc_processor.save_upload(content, file.filename)
        text, page_count = doc_processor.extract_text(file_path, file.filename)

        if not text.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted from the document")

        # Run NLP analysis
        entities = onnx_service.extract_entities(text)
        classifications = onnx_service.classify_text(text)

        # Generate summary via LLM
        summary = await llm_service.summarize(text)

        # Index for RAG
        chunks = doc_processor.chunk_text(text)
        rag_pipeline.index_document(doc_id, chunks)

        # Store analysis result
        analysis = DocumentAnalysis(
            doc_id=doc_id,
            filename=file.filename,
            page_count=page_count,
            char_count=len(text),
            entities=entities,
            classifications=classifications,
            summary=summary,
        )
        document_store[doc_id] = analysis

        logger.info(f"Document {doc_id} processed successfully")
        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            message="Document processed successfully",
            analysis=analysis,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/documents/{doc_id}", response_model=DocumentAnalysis)
async def get_document(doc_id: str):
    """Retrieve analysis results for a previously uploaded document."""
    if doc_id not in document_store:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return document_store[doc_id]


@router.post("/documents/{doc_id}/ask", response_model=QuestionResponse)
async def ask_question(doc_id: str, request: QuestionRequest):
    """Ask a question about a document using RAG (Retrieval-Augmented Generation)."""
    if doc_id not in document_store:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    try:
        answer, source_chunks = await rag_pipeline.answer_question(doc_id, request.question)
        return QuestionResponse(
            doc_id=doc_id,
            question=request.question,
            answer=answer,
            source_chunks=source_chunks,
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")
