from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

EMBEDDING_DIM = 768  # models/text-embedding-004

pdf_dir = Path("data/pdfs")
docs = []
for pdf in pdf_dir.glob("*.pdf"):
    loader = PyPDFLoader(str(pdf))
    pages = loader.load()
    for p in pages:
        p.metadata["source"] = pdf.name
        p.metadata["era"] = "rashidun"
    docs.extend(pages)

if not docs:
    raise SystemExit(f"No PDFs found in {pdf_dir.resolve()}. Add .pdf files and retry.")

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
print(f"Created collection: {collection} (dim={EMBEDDING_DIM})")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection,
    embedding=embeddings,
)

vectorstore.add_documents(chunks)
print(f"Indexed {len(chunks)} chunks from {len(list(pdf_dir.glob('*.pdf')))} PDFs")
