"""Multi-document agentic RAG pipeline (route → retrieve → generate).

1. route:    decide which manuscript(s) the question is about.
             - explicit doc_ids from the caller win;
             - otherwise a free alias match ("Abu Bakar" -> abu_bakr);
             - otherwise an LLM router over the catalog descriptions
               (e.g. "Battle of Jamal" -> ali).
2. retrieve: embed the question once, then run one Qdrant search per routed
             document, filtered on the indexed ``doc_id`` payload, so only that
             document's chunks are ever scored.
3. generate: answer strictly from the retrieved passages.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Generator, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from catalog import DOC_IDS, DOCS_BY_ID, catalog_prompt, match_aliases

load_dotenv()

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768
CHAT_MODEL = "gemini-3.1-flash-lite"
DOC_ID_FIELD = "metadata.doc_id"

TOP_K = 6  # total passages for a single-document or unscoped question
MIN_PER_DOC = 3  # floor per document when a question spans several

SYSTEM_PROMPT = (
    "You are a careful Islamic history assistant. "
    "Answer ONLY from the provided context about the Rashidun era "
    "(Abu Bakr, Umar, Uthman, Ali, etc.). "
    "If the context is insufficient, say you don't know. "
    "Write in clear plain prose with short paragraphs. "
    "Do NOT use markdown (no asterisks, bold, bullets with *, or headings). "
    "Do NOT list sources or PDF names in the answer — the interface shows references separately."
)

ROUTER_PROMPT = (
    "You route questions about the Rashidun caliphate to the source manuscripts "
    "that can answer them. Each manuscript covers one caliph's life and era:\n\n"
    "{catalog}\n\n"
    "Pick the smallest set of manuscripts needed to answer the question. "
    "Pick one when the question concerns a single caliph, person or event of one era; "
    "pick several only when it genuinely spans eras (comparisons, successions, "
    "the whole Rashidun period). Return an empty list only if no manuscript is relevant."
)

DocId = Literal[DOC_IDS]  # type: ignore[valid-type]


class RouteDecision(BaseModel):
    """Which manuscripts to search."""

    doc_ids: List[DocId] = Field(
        default_factory=list,
        description="IDs of the manuscripts to search, most relevant first.",
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


class State(TypedDict, total=False):
    question: str
    doc_ids: List[str]  # input: optional explicit scope; output: routed scope
    route_method: str  # explicit | alias | llm | fallback
    context: List[Document]
    answer: str


def _require_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")
    return api_key


def _doc_filter(doc_id: str) -> qm.Filter:
    return qm.Filter(
        must=[qm.FieldCondition(key=DOC_ID_FIELD, match=qm.MatchValue(value=doc_id))]
    )


def format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('title') or d.metadata.get('source')} "
        f"p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


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

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0,
        google_api_key=api_key,
    )
    router_llm = llm.with_structured_output(RouteDecision)
    router_prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), ("human", "{question}")]
    ).partial(catalog=catalog_prompt())

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    def route(state: State):
        explicit = [d for d in state.get("doc_ids") or [] if d in DOCS_BY_ID]
        if explicit:
            return {"doc_ids": explicit, "route_method": "explicit"}

        matched = match_aliases(state["question"])
        if matched:
            return {"doc_ids": matched, "route_method": "alias"}

        try:
            decision = router_llm.invoke(router_prompt.invoke({"question": state["question"]}))
            chosen = list(dict.fromkeys(decision.doc_ids)) if decision else []
        except Exception:
            log.exception("LLM router failed; searching all documents")
            chosen = []
        if chosen:
            return {"doc_ids": chosen, "route_method": "llm"}
        # Nothing matched: search the whole collection rather than return nothing.
        return {"doc_ids": [], "route_method": "fallback"}

    def retrieve(state: State):
        doc_ids = state.get("doc_ids") or []
        vector = embeddings.embed_query(state["question"])  # embed once, reuse per doc

        if not doc_ids:
            hits = vectorstore.similarity_search_with_score_by_vector(vector, k=TOP_K)
        else:
            per_doc = max(MIN_PER_DOC, math.ceil(TOP_K / len(doc_ids)))
            hits = []
            for doc_id in doc_ids:
                hits.extend(
                    vectorstore.similarity_search_with_score_by_vector(
                        vector, k=per_doc, filter=_doc_filter(doc_id)
                    )
                )
            hits.sort(key=lambda h: h[1], reverse=True)

        return {"context": [doc for doc, _ in hits]}

    def generate(state: State):
        msg = prompt.invoke(
            {"context": format_context(state["context"]), "question": state["question"]}
        )
        resp = llm.invoke(msg)
        return {"answer": _text(resp.content)}

    graph = StateGraph(State)
    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return {
        "graph": graph.compile(),
        "route": route,
        "retrieve": retrieve,
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
        sources.append({"source": name, "page": page, "doc_id": d.metadata.get("doc_id")})
    return sources


def _route_info(state: State) -> dict:
    doc_ids = state.get("doc_ids") or []
    return {
        "doc_ids": doc_ids,
        "titles": [DOCS_BY_ID[d].title for d in doc_ids],
        "method": state.get("route_method"),
    }


def _route_message(route: dict) -> str:
    if not route["titles"]:
        return "Searching all archives…"
    return "Consulting " + ", ".join(route["titles"]) + "…"


def ask(question: str, doc_ids: Optional[List[str]] = None) -> dict:
    """Run RAG and return plain-text answer, sources and the routing decision."""
    result = get_app().invoke({"question": question, "doc_ids": doc_ids or []})
    return {
        "answer": _text(result["answer"]),
        "sources": source_list(result.get("context") or []),
        "route": _route_info(result),
    }


def ask_stream(
    question: str, doc_ids: Optional[List[str]] = None
) -> Generator[dict, None, None]:
    """Yield SSE-friendly events: status → route → status → token* → sources → done."""
    pipe = get_pipeline()
    state: State = {"question": question, "doc_ids": doc_ids or []}

    yield {"type": "status", "message": "Choosing the right manuscript…"}
    state.update(pipe["route"](state))
    route = _route_info(state)
    yield {"type": "route", **route}

    yield {"type": "status", "message": _route_message(route)}
    state.update(pipe["retrieve"](state))
    docs = state["context"]

    msg = pipe["prompt"].invoke({"context": format_context(docs), "question": question})

    yield {"type": "status", "message": "Writing…"}
    for chunk in pipe["llm"].stream(msg):
        token = _text(chunk.content)
        if token:
            yield {"type": "token", "text": token}

    yield {"type": "sources", "sources": source_list(docs)}
    yield {"type": "done"}
