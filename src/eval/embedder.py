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


class DashScopeEmbedder:
    """Embedding client for Alibaba DashScope (text-embedding-v4).
    Uses OpenAI-compatible API format."""

    def __init__(self, base_url=None, api_key=None, model=None, batch_size=6):
        self.base_url = base_url or os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is required")
        self.model = model or "text-embedding-v4"
        self.batch_size = batch_size
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching if needed."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_results = [d.embedding for d in response.data]
            all_embeddings.extend(batch_results[:len(batch)])
            if i + self.batch_size < len(texts):
                time.sleep(0.2)
        return all_embeddings
