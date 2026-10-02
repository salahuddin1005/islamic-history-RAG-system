"""Web UI for Islamic history RAG Q&A.

Run:
    .venv\\Scripts\\python -m uvicorn app:app --reload --port 8000
Then open http://localhost:8000
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from catalog import DOCUMENTS
from rag import DocId, ask, ask_stream

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Rashidun Archive", docs_url=None, redoc_url=None)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    # Optional explicit scope; when omitted the router picks the manuscript(s).
    doc_ids: list[DocId] | None = None


class Source(BaseModel):
    source: str
    page: object
    doc_id: str | None = None


class Route(BaseModel):
    doc_ids: list[str]
    titles: list[str]
    method: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    route: Route


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/documents")
def api_documents():
    return [{"doc_id": d.doc_id, "title": d.title, "file": d.file} for d in DOCUMENTS]


@app.post("/api/ask", response_model=AskResponse)
def api_ask(body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    try:
        result = ask(question, body.doc_ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = result["answer"]
    if not isinstance(answer, str):
        answer = str(answer)

    return AskResponse(answer=answer, sources=result["sources"], route=result["route"])


@app.post("/api/ask/stream")
async def api_ask_stream(body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    def event_gen():
        try:
            for event in ask_stream(question, body.doc_ids):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Mount static last so it never shadows /api routes
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
