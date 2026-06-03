from unittest.mock import MagicMock
from src.eval.llm_client import ChatResult, ToolCall
from src.eval.rag_agent import RAGAgent


class FakeIndexStore:
    def search(self, query_embedding, top_k=5):
        return ["File: NamespaceService.java\nScore: 0.95\n```java\npublic Namespace findNamespace() {\n```"]


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 128 for _ in texts]


def make_mock_llm(responses):
    mock = MagicMock()
    mock.chat = MagicMock(side_effect=responses)
    return mock


def test_rag_agent_finds_symbol_with_vector_search():
    llm = make_mock_llm([
        ChatResult(
            content=None,
            tool_calls=[ToolCall(id="1", name="vector_search", arguments={"query": "NamespaceService.findNamespace"})]
        ),
        ChatResult(content="NamespaceService.java:42"),
    ])
    agent = RAGAgent(
        llm_client=llm,
        codebase_root="/tmp",
        index_store=FakeIndexStore(),
        embedder=FakeEmbedder(),
    )
    result = agent.run(symbol="NamespaceService.findNamespace", symbol_type="method")
    assert result.file_path == "NamespaceService.java"
    assert result.line_number == 42
    assert result.steps == 2


def test_rag_agent_direct_answer():
    llm = make_mock_llm([
        ChatResult(content="NamespaceService.java:87"),
    ])
    agent = RAGAgent(
        llm_client=llm,
        codebase_root="/tmp",
        index_store=FakeIndexStore(),
        embedder=FakeEmbedder(),
    )
    result = agent.run(symbol="NamespaceService.findNamespace", symbol_type="method")
    assert result.line_number == 87
    assert result.steps == 1


def test_rag_agent_strips_markdown_backticks():
    llm = make_mock_llm([
        ChatResult(content="`apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/service/ReleaseService.java:98`"),
    ])
    agent = RAGAgent(llm_client=llm, codebase_root="/tmp",
                     index_store=FakeIndexStore(), embedder=FakeEmbedder())
    result = agent.run(symbol="ReleaseService.findOne", symbol_type="method")
    assert result.file_path == "apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/service/ReleaseService.java"
    assert result.line_number == 98


def test_rag_agent_strips_bold_markdown():
    llm = make_mock_llm([
        ChatResult(content="**apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/service/ReleaseService.java:195**"),
    ])
    agent = RAGAgent(llm_client=llm, codebase_root="/tmp",
                     index_store=FakeIndexStore(), embedder=FakeEmbedder())
    result = agent.run(symbol="ReleaseService.publish", symbol_type="method")
    assert result.file_path == "apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/service/ReleaseService.java"
    assert result.line_number == 195


def test_rag_agent_strips_answer_prefix():
    llm = make_mock_llm([
        ChatResult(content="**Answer:** `apollo-portal/src/main/java/com/ctrip/framework/apollo/portal/service/NamespaceService.java:168`"),
    ])
    agent = RAGAgent(llm_client=llm, codebase_root="/tmp",
                     index_store=FakeIndexStore(), embedder=FakeEmbedder())
    result = agent.run(symbol="NamespaceService.deleteNamespace", symbol_type="method")
    assert result.file_path == "apollo-portal/src/main/java/com/ctrip/framework/apollo/portal/service/NamespaceService.java"
    assert result.line_number == 168
