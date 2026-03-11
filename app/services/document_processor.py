import os
import uuid
import logging
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentProcessor:
    """Handles document upload, text extraction, and chunking."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        os.makedirs(settings.upload_dir, exist_ok=True)

    def generate_doc_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def extract_text_from_pdf(self, file_path: str) -> tuple[str, int]:
        """Extract text from a PDF file. Returns (text, page_count)."""
        try:
            reader = PdfReader(file_path)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
            full_text = "\n\n".join(pages)
            logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")
            return full_text, len(reader.pages)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise ValueError(f"Failed to extract text from PDF: {e}")

    def extract_text_from_txt(self, file_path: str) -> tuple[str, int]:
        """Extract text from a plain text file. Returns (text, page_count=1)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.info(f"Read {len(text)} characters from text file")
            return text, 1
        except Exception as e:
            logger.error(f"Text file reading failed: {e}")
            raise ValueError(f"Failed to read text file: {e}")

    def extract_text(self, file_path: str, filename: str) -> tuple[str, int]:
        """Route to appropriate extractor based on file extension."""
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext == "pdf":
            return self.extract_text_from_pdf(file_path)
        elif ext in ("txt", "md", "text"):
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: .{ext}. Supported: .pdf, .txt, .md")

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks for RAG."""
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    async def save_upload(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file to disk and return the file path."""
        doc_id = self.generate_doc_id()
        safe_filename = f"{doc_id}_{filename}"
        file_path = os.path.join(settings.upload_dir, safe_filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
        logger.info(f"Saved upload: {safe_filename}")
        return file_path
