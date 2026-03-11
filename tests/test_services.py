import pytest
from app.services.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor()


def test_generate_doc_id(processor):
    """Test document ID generation."""
    doc_id = processor.generate_doc_id()
    assert isinstance(doc_id, str)
    assert len(doc_id) == 8


def test_generate_unique_ids(processor):
    """Test that generated IDs are unique."""
    ids = {processor.generate_doc_id() for _ in range(100)}
    assert len(ids) == 100


def test_chunk_text(processor):
    """Test text chunking produces overlapping chunks."""
    # Create text longer than chunk_size
    text = "This is a test sentence. " * 200
    chunks = processor.chunk_text(text)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) > 0 for c in chunks)


def test_chunk_short_text(processor):
    """Test that short text produces a single chunk."""
    text = "Short text."
    chunks = processor.chunk_text(text)
    assert len(chunks) == 1


def test_extract_text_unsupported_format(processor):
    """Test that unsupported formats raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        processor.extract_text("/fake/path.docx", "test.docx")
