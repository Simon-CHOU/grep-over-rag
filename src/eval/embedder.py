import os
import time
from openai import OpenAI


class DeepSeekEmbedder:
    """Embedding client for DeepSeek's embedding API."""

    def __init__(self, base_url=None, api_key=None, model=None, batch_size=20):
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model or "text-embedding-3-small"
        self.batch_size = batch_size
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching if needed. Returns list of embedding vectors."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in response.data])
            if i + self.batch_size < len(texts):
                time.sleep(0.1)  # rate limit courtesy
        return all_embeddings
