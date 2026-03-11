import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_root(client):
    """Test root endpoint returns app info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert "version" in data
    assert data["docs"] == "/docs"


@pytest.mark.anyio
async def test_health(client):
    """Test health endpoint returns status."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "timestamp" in data


@pytest.mark.anyio
async def test_upload_invalid_file_type(client):
    """Test that unsupported file types are rejected."""
    response = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.jpg", b"fake image content", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.anyio
async def test_get_nonexistent_document(client):
    """Test 404 for missing document."""
    response = await client.get("/api/v1/documents/nonexistent")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_ask_nonexistent_document(client):
    """Test 404 when asking question about missing document."""
    response = await client.post(
        "/api/v1/documents/nonexistent/ask",
        json={"question": "What is this about?"},
    )
    assert response.status_code == 404
