"""Shared Islamic history RAG pipeline (retrieve → generate)."""

from __future__ import annotations

import os
from typing import Generator, List, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient

load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768
CHAT_MODEL = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = (
    "You are a careful Islamic history assistant. "
    "Answer ONLY from the provided context about the Rashidun era "
    "(Abu Bakr, Umar, Uthman, Ali, etc.). "
    "If the context is insufficient, say you don't know. "
    "Write in clear plain prose with short paragraphs. "
    "Do NOT use markdown (no asterisks, bold, bullets with *, or headings). "
    "Do NOT list sources or PDF names in the answer — the interface shows references separately."
)


def _text(content) -> str:
    """Normalize Gemini / LangChain content blocks to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(getattr(block, "text", str(block)))
        return "\n".join(p for p in parts if p)
    return str(content)


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def _require_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")
    return api_key


_pipeline = None


def _build_pipeline():
    api_key = _require_api_key()

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
        output_dimensionality=EMBEDDING_DIM,
    )
    client = QdrantClient(url=os.getenv("QDRANT_URL"))
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION", "islamic_history"),
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0,
        google_api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    def retrieve(state: State):
        docs = retriever.invoke(state["question"])
        return {"context": docs}

    def generate(state: State):
        ctx = "\n\n".join(
            f"[{d.metadata.get('source')} p.{d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in state["context"]
        )
        msg = prompt.invoke({"context": ctx, "question": state["question"]})
        resp = llm.invoke(msg)
        return {"answer": _text(resp.content)}

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return {
        "graph": graph.compile(),
        "retriever": retriever,
        "llm": llm,
        "prompt": prompt,
    }


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


def get_app():
    return get_pipeline()["graph"]


def source_list(docs: List[Document]) -> list[dict]:
    """Unique sources with PDF name and page."""
    seen = set()
    sources = []
    for d in docs:
        name = d.metadata.get("source") or "unknown"
        page = d.metadata.get("page", "?")
        key = (name, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": name, "page": page})
    return sources


def ask(question: str) -> dict:
    """Run RAG and return plain-text answer plus sources."""
    result = get_app().invoke(
        {"question": question, "context": [], "answer": ""}
    )
    answer = _text(result["answer"])
    return {
        "answer": answer,
        "sources": source_list(result.get("context") or []),
    }


def ask_stream(question: str) -> Generator[dict, None, None]:
    """Yield SSE-friendly events: status → token* → sources → done."""
    pipe = get_pipeline()
    retriever = pipe["retriever"]
    llm = pipe["llm"]
    prompt = pipe["prompt"]

    yield {"type": "status", "message": "Searching the archives…"}
    docs = retriever.invoke(question)

    ctx = "\n\n".join(
        f"[{d.metadata.get('source')} p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )
    msg = prompt.invoke({"context": ctx, "question": question})

    yield {"type": "status", "message": "Writing…"}
    for chunk in llm.stream(msg):
        token = _text(chunk.content)
        if token:
            yield {"type": "token", "text": token}

    yield {"type": "sources", "sources": source_list(docs)}
    yield {"type": "done"}
