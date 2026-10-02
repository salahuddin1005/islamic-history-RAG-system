"""Corrective multi-document RAG (CRAG-style) with LangGraph.

Flow:
  route → retrieve → grade_documents
       ↳ (weak) rewrite_query → retrieve  (max RETRIEVAL_RETRIES)
       → generate → grade_answer
       ↳ (weak) rewrite_query → retrieve → …  (max GENERATION_RETRIES)
       → final answer
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

TOP_K = 6
MIN_PER_DOC = 3
MIN_RELEVANT_DOCS = 2
RETRIEVAL_RETRIES = 2
GENERATION_RETRIES = 2

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

GRADE_DOCS_PROMPT = (
    "You grade retrieved history passages for usefulness.\n"
    "Question: {question}\n\n"
    "Passages (indexed from 0):\n{passages}\n\n"
    "Return the indices of passages that contain information useful for answering "
    "the question. Be strict: exclude off-topic or barely related text."
)

REWRITE_PROMPT = (
    "Rewrite the user's question to improve search over Islamic history manuscripts "
    "about the Rashidun era (Abu Bakr, Umar, Uthman, Ali).\n"
    "Expand names, alternate spellings, and event titles "
    "(e.g. Jamal -> Battle of the Camel / Battle of Jamal).\n"
    "Keep the same intent. Return only the rewritten question, no explanation.\n\n"
    "Original question: {question}\n"
    "Reason retrieval/answer was weak: {reason}"
)

GRADE_ANSWER_PROMPT = (
    "You grade an answer against retrieved context for an Islamic history Q&A system.\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:\n{answer}\n\n"
    "Set grounded=true only if the answer's claims are supported by the context "
    "(or the answer honestly says it does not know).\n"
    "Set addresses_question=true only if the answer actually responds to the question."
)

DocId = Literal[DOC_IDS]  # type: ignore[valid-type]


class RouteDecision(BaseModel):
    """Which manuscripts to search."""

    doc_ids: List[DocId] = Field(
        default_factory=list,
        description="IDs of the manuscripts to search, most relevant first.",
    )


class DocGrade(BaseModel):
    """Which retrieved passages are relevant."""

    relevant_indices: List[int] = Field(
        default_factory=list,
        description="0-based indices of useful passages.",
    )
    reason: str = Field(default="", description="Short reason for the grading.")


class AnswerGrade(BaseModel):
    """Whether the generated answer is acceptable."""

    grounded: bool = Field(description="Answer is supported by the context.")
    addresses_question: bool = Field(description="Answer addresses the user question.")
    reason: str = Field(default="", description="Short critique.")


class RewriteResult(BaseModel):
    rewritten_question: str = Field(description="Improved search question.")


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
    question: str  # current search / answer question (may be rewritten)
    original_question: str
    doc_ids: List[str]
    route_method: str
    context: List[Document]
    answer: str
    retrieval_retries: int
    generation_retries: int
    docs_ok: bool
    answer_ok: bool
    grade_reason: str
    steps: List[str]


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


def _append_step(state: State, step: str) -> List[str]:
    return list(state.get("steps") or []) + [step]


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
        # Skip dummy embed on init (saves API calls / avoids startup failures).
        validate_embeddings=False,
        validate_collection_config=False,
    )

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0,
        google_api_key=api_key,
    )
    router_llm = llm.with_structured_output(RouteDecision)
    docs_grader = llm.with_structured_output(DocGrade)
    answer_grader = llm.with_structured_output(AnswerGrade)
    rewriter = llm.with_structured_output(RewriteResult)

    router_prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), ("human", "{question}")]
    ).partial(catalog=catalog_prompt())

    grade_docs_prompt = ChatPromptTemplate.from_messages(
        [("system", GRADE_DOCS_PROMPT), ("human", "Grade the passages.")]
    )
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [("system", REWRITE_PROMPT), ("human", "Rewrite the question.")]
    )
    grade_answer_prompt = ChatPromptTemplate.from_messages(
        [("system", GRADE_ANSWER_PROMPT), ("human", "Grade the answer.")]
    )
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    def route(state: State):
        original = state.get("original_question") or state["question"]
        explicit = [d for d in state.get("doc_ids") or [] if d in DOCS_BY_ID]
        if explicit:
            return {
                "original_question": original,
                "question": original,
                "doc_ids": explicit,
                "route_method": "explicit",
                "retrieval_retries": 0,
                "generation_retries": 0,
                "steps": _append_step(state, "route:explicit"),
            }

        matched = match_aliases(original)
        if matched:
            return {
                "original_question": original,
                "question": original,
                "doc_ids": matched,
                "route_method": "alias",
                "retrieval_retries": 0,
                "generation_retries": 0,
                "steps": _append_step(state, "route:alias"),
            }

        try:
            decision = router_llm.invoke(router_prompt.invoke({"question": original}))
            chosen = list(dict.fromkeys(decision.doc_ids)) if decision else []
        except Exception:
            log.exception("LLM router failed; searching all documents")
            chosen = []
        if chosen:
            return {
                "original_question": original,
                "question": original,
                "doc_ids": chosen,
                "route_method": "llm",
                "retrieval_retries": 0,
                "generation_retries": 0,
                "steps": _append_step(state, "route:llm"),
            }
        return {
            "original_question": original,
            "question": original,
            "doc_ids": [],
            "route_method": "fallback",
            "retrieval_retries": 0,
            "generation_retries": 0,
            "steps": _append_step(state, "route:fallback"),
        }

    def retrieve(state: State):
        doc_ids = state.get("doc_ids") or []
        query = state.get("question") or state.get("original_question") or ""
        vector = embeddings.embed_query(query)

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

        return {
            "context": [doc for doc, _ in hits],
            "steps": _append_step(state, f"retrieve:{len(hits)}"),
        }

    def grade_documents(state: State):
        docs = state.get("context") or []
        question = state.get("original_question") or state.get("question") or ""
        if not docs:
            return {
                "docs_ok": False,
                "context": [],
                "grade_reason": "No passages retrieved.",
                "steps": _append_step(state, "grade_docs:empty"),
            }

        passages = "\n\n".join(
            f"[{i}] {d.page_content[:900]}" for i, d in enumerate(docs)
        )
        try:
            grade = docs_grader.invoke(
                grade_docs_prompt.invoke({"question": question, "passages": passages})
            )
            indices = sorted(
                {
                    i
                    for i in (grade.relevant_indices if grade else [])
                    if isinstance(i, int) and 0 <= i < len(docs)
                }
            )
            reason = (grade.reason if grade else "") or ""
        except Exception:
            log.exception("Document grading failed; keeping all passages")
            indices = list(range(len(docs)))
            reason = "Grader failed; kept all retrieved passages."

        filtered = [docs[i] for i in indices] if indices else []
        # Keep a little context if grader was overly strict but we still have docs.
        if not filtered and docs:
            filtered = docs[: min(2, len(docs))]
            reason = (reason + " ").strip() + "Kept top passages as fallback."
            docs_ok = False
        else:
            docs_ok = len(filtered) >= MIN_RELEVANT_DOCS or (
                len(filtered) >= 1 and len(docs) == 1
            )

        return {
            "context": filtered,
            "docs_ok": docs_ok,
            "grade_reason": reason.strip(),
            "steps": _append_step(
                state, f"grade_docs:{'ok' if docs_ok else 'weak'}:{len(filtered)}"
            ),
        }

    def rewrite_query(state: State):
        original = state.get("original_question") or state.get("question") or ""
        reason = state.get("grade_reason") or "Retrieved context was weak."
        try:
            result = rewriter.invoke(
                rewrite_prompt.invoke({"question": original, "reason": reason})
            )
            rewritten = (result.rewritten_question if result else "").strip() or original
        except Exception:
            log.exception("Query rewrite failed")
            rewritten = original

        retrieval_retries = int(state.get("retrieval_retries") or 0) + 1
        generation_retries = int(state.get("generation_retries") or 0)
        # If we reached rewrite from a bad answer, count generation retry too.
        if state.get("answer") and not state.get("answer_ok", True):
            generation_retries += 1

        return {
            "question": rewritten,
            "retrieval_retries": retrieval_retries,
            "generation_retries": generation_retries,
            "answer": "",
            "answer_ok": False,
            "steps": _append_step(state, f"rewrite:{rewritten[:80]}"),
        }

    def generate(state: State):
        question = state.get("original_question") or state.get("question") or ""
        msg = answer_prompt.invoke(
            {"context": format_context(state.get("context") or []), "question": question}
        )
        resp = llm.invoke(msg)
        return {
            "answer": _text(resp.content),
            "steps": _append_step(state, "generate"),
        }

    def grade_answer(state: State):
        question = state.get("original_question") or state.get("question") or ""
        answer = state.get("answer") or ""
        context = format_context(state.get("context") or [])
        try:
            grade = answer_grader.invoke(
                grade_answer_prompt.invoke(
                    {"question": question, "context": context, "answer": answer}
                )
            )
            grounded = bool(grade.grounded) if grade else True
            addresses = bool(grade.addresses_question) if grade else True
            reason = (grade.reason if grade else "") or ""
        except Exception:
            log.exception("Answer grading failed; accepting answer")
            grounded, addresses, reason = True, True, "Grader failed; accepted answer."

        ok = grounded and addresses
        return {
            "answer_ok": ok,
            "grade_reason": reason.strip(),
            "steps": _append_step(state, f"grade_answer:{'ok' if ok else 'weak'}"),
        }

    def after_grade_docs(state: State) -> str:
        if state.get("docs_ok"):
            return "generate"
        if int(state.get("retrieval_retries") or 0) < RETRIEVAL_RETRIES:
            return "rewrite_query"
        return "generate"

    def after_grade_answer(state: State) -> str:
        if state.get("answer_ok"):
            return "end"
        if int(state.get("generation_retries") or 0) < GENERATION_RETRIES:
            return "rewrite_query"
        return "end"

    graph = StateGraph(State)
    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer)

    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        after_grade_docs,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "grade_answer")
    graph.add_conditional_edges(
        "grade_answer",
        after_grade_answer,
        {"rewrite_query": "rewrite_query", "end": END},
    )

    return {
        "graph": graph.compile(),
        "route": route,
        "retrieve": retrieve,
        "grade_documents": grade_documents,
        "rewrite_query": rewrite_query,
        "generate": generate,
        "grade_answer": grade_answer,
        "after_grade_docs": after_grade_docs,
        "after_grade_answer": after_grade_answer,
        "llm": llm,
        "prompt": answer_prompt,
        "embeddings": embeddings,
        "vectorstore": vectorstore,
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
        "titles": [DOCS_BY_ID[d].title for d in doc_ids if d in DOCS_BY_ID],
        "method": state.get("route_method"),
    }


def _route_message(route: dict) -> str:
    if not route["titles"]:
        return "Searching all archives…"
    return "Consulting " + ", ".join(route["titles"]) + "…"


def _correction_info(state: State) -> dict:
    return {
        "docs_ok": bool(state.get("docs_ok")),
        "answer_ok": bool(state.get("answer_ok")),
        "retrieval_retries": int(state.get("retrieval_retries") or 0),
        "generation_retries": int(state.get("generation_retries") or 0),
        "rewritten_question": state.get("question"),
        "original_question": state.get("original_question") or state.get("question"),
        "grade_reason": state.get("grade_reason") or "",
        "steps": list(state.get("steps") or []),
    }


def ask(question: str, doc_ids: Optional[List[str]] = None) -> dict:
    """Run Corrective RAG and return answer, sources, route, and correction metadata."""
    result = get_app().invoke(
        {
            "question": question,
            "original_question": question,
            "doc_ids": doc_ids or [],
            "retrieval_retries": 0,
            "generation_retries": 0,
            "steps": [],
        }
    )
    return {
        "answer": _text(result.get("answer") or ""),
        "sources": source_list(result.get("context") or []),
        "route": _route_info(result),
        "correction": _correction_info(result),
    }


def ask_stream(
    question: str, doc_ids: Optional[List[str]] = None
) -> Generator[dict, None, None]:
    """Yield SSE events while running the Corrective RAG loop, then stream tokens."""
    pipe = get_pipeline()
    state: State = {
        "question": question,
        "original_question": question,
        "doc_ids": doc_ids or [],
        "retrieval_retries": 0,
        "generation_retries": 0,
        "steps": [],
    }

    yield {"type": "status", "message": "Choosing the right manuscript…"}
    state.update(pipe["route"](state))
    route = _route_info(state)
    yield {"type": "route", **route}

    # Retrieval / correction loop
    while True:
        yield {"type": "status", "message": _route_message(route)}
        if state.get("question") and state.get("question") != state.get("original_question"):
            yield {
                "type": "status",
                "message": f"Refining search: {state['question'][:120]}",
            }

        state.update(pipe["retrieve"](state))
        yield {"type": "status", "message": "Evaluating retrieved passages…"}
        state.update(pipe["grade_documents"](state))

        decision = pipe["after_grade_docs"](state)
        if decision == "rewrite_query":
            yield {"type": "status", "message": "Passages weak — rewriting the question…"}
            state.update(pipe["rewrite_query"](state))
            continue
        break

    # Generation / answer-correction loop
    while True:
        yield {"type": "status", "message": "Writing…"}
        # Stream tokens for this generation attempt
        msg = pipe["prompt"].invoke(
            {
                "context": format_context(state.get("context") or []),
                "question": state.get("original_question") or question,
            }
        )
        answer_parts: List[str] = []
        for chunk in pipe["llm"].stream(msg):
            token = _text(chunk.content)
            if token:
                answer_parts.append(token)
                yield {"type": "token", "text": token}
        state["answer"] = "".join(answer_parts)
        state["steps"] = _append_step(state, "generate")

        yield {"type": "status", "message": "Checking answer against the sources…"}
        # Clear streamed bubble sense: grading status only (UI keeps tokens until done)
        state.update(pipe["grade_answer"](state))

        decision = pipe["after_grade_answer"](state)
        if decision == "rewrite_query":
            yield {
                "type": "status",
                "message": "Answer needs correction — refining and trying again…",
            }
            yield {"type": "reset"}  # UI should clear partial answer before retry
            state.update(pipe["rewrite_query"](state))
            # Re-retrieve with rewritten query before next generate
            yield {"type": "status", "message": "Searching again with a clearer question…"}
            state.update(pipe["retrieve"](state))
            state.update(pipe["grade_documents"](state))
            continue
        break

    yield {"type": "sources", "sources": source_list(state.get("context") or [])}
    yield {"type": "correction", **_correction_info(state)}
    yield {"type": "done"}
