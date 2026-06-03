# tests/test_embedder.py
from unittest.mock import MagicMock, patch
from src.eval.embedder import DashScopeEmbedder


def test_dashscope_embedder_uses_env_vars():
    with patch.dict("os.environ", {
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_BASE_URL": "https://test.example.com/v1",
    }):
        embedder = DashScopeEmbedder()
        assert embedder.api_key == "test-key"
        assert embedder.base_url == "https://test.example.com/v1"
        assert embedder.model == "text-embedding-v4"


def test_dashscope_embedder_default_base_url():
    with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=False):
        import os
        os.environ.pop("DASHSCOPE_BASE_URL", None)
        embedder = DashScopeEmbedder()
        assert embedder.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_dashscope_embedder_batch():
    embedder = DashScopeEmbedder(api_key="fake-key", batch_size=2)
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(2)
    ]
    embedder._client = MagicMock()
    embedder._client.embeddings.create.return_value = mock_response

    results = embedder.embed(["text1", "text2", "text3"])
    assert len(results) == 3
    assert embedder._client.embeddings.create.call_count == 2
