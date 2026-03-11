import logging
import httpx
from typing import Optional
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMService:
    """Handles communication with Ollama for LLM inference."""

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model

    async def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    available = any(m["name"].startswith(self.model) for m in models)
                    if not available:
                        logger.warning(f"Model '{self.model}' not found. Available: {[m['name'] for m in models]}")
                    return available
                return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from Ollama."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "No response generated.")

        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            return "Request timed out. Please try again."
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"LLM service error: {str(e)}"

    async def summarize(self, text: str) -> str:
        """Generate a concise summary of the document text."""
        system_prompt = (
            "You are a document analysis assistant. Provide clear, concise summaries "
            "that capture the key points, main arguments, and important details."
        )
        prompt = f"Summarize the following document in 3-5 sentences:\n\n{text[:4000]}"
        return await self.generate(prompt, system_prompt)
