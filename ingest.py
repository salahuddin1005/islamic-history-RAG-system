from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, VectorParams

from catalog import DOCS_BY_FILE

load_dotenv()

EMBEDDING_DIM = 768  # models/gemini-embedding-001 (truncated)
DOC_ID_FIELD = "metadata.doc_id"  # langchain_qdrant nests metadata under "metadata"

pdf_dir = Path("data/pdfs")
pdfs = sorted(pdf_dir.glob("*.pdf"))
if not pdfs:
    raise SystemExit(f"No PDFs found in {pdf_dir.resolve()}. Add .pdf files and retry.")

unknown = [p.name for p in pdfs if p.name not in DOCS_BY_FILE]
if unknown:
    raise SystemExit(
        "These PDFs are not registered in catalog.py, so they could not be routed to:\n  "
        + "\n  ".join(unknown)
    )

docs = []
for pdf in pdfs:
    entry = DOCS_BY_FILE[pdf.name]
    pages = PyPDFLoader(str(pdf)).load()
    for p in pages:
        p.metadata["source"] = pdf.name
        p.metadata["doc_id"] = entry.doc_id
        p.metadata["title"] = entry.title
        p.metadata["era"] = "rashidun"
    docs.extend(pages)
    print(f"Loaded {len(pages):>3} pages  [{entry.doc_id}] {pdf.name}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(docs)

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
    output_dimensionality=EMBEDDING_DIM,
)

client = QdrantClient(url=os.getenv("QDRANT_URL"))
collection = os.getenv("QDRANT_COLLECTION", "islamic_history")

# Recreate collection so vector size matches Gemini (not OpenAI 1536)
if client.collection_exists(collection):
    client.delete_collection(collection)
    print(f"Deleted existing collection: {collection}")

client.create_collection(
    collection_name=collection,
    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
)
# Keyword index on doc_id: filtered searches only touch that document's points.
client.create_payload_index(
    collection_name=collection,
    field_name=DOC_ID_FIELD,
    field_schema=PayloadSchemaType.KEYWORD,
)
print(f"Created collection: {collection} (dim={EMBEDDING_DIM}, indexed {DOC_ID_FIELD})")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection,
    embedding=embeddings,
)

vectorstore.add_documents(chunks, batch_size=64)
print(f"Indexed {len(chunks)} chunks from {len(pdfs)} PDFs")
