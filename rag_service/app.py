import os
from fastapi import FastAPI
from pydantic import BaseModel

from rag_core import (
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    LLMIntegration,
    RAGPipeline,
    build_or_rebuild_index,
)

app = FastAPI(title="Chemistry RAG Service")

DOCS_FOLDER = os.getenv("RAG_DOCS_FOLDER", "./chemistry_docs")
PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "./data/vector_store")
COLLECTION = os.getenv("RAG_COLLECTION", "chemistry_docs")
EMB_MODEL = os.getenv("RAG_EMB_MODEL", "all-MiniLM-L6-v2")

# Load components once on startup
embedding_manager = EmbeddingManager(model_name=EMB_MODEL)
vector_store = VectorStore(collection_name=COLLECTION, persist_directory=PERSIST_DIR)
retriever = RAGRetriever(embedding_manager=embedding_manager, vector_store=vector_store)
llm = LLMIntegration()
pipeline = RAGPipeline(retriever=retriever, llm=llm)


class AskRequest(BaseModel):
    query: str
    k: int = 5


class BuildIndexRequest(BaseModel):
    docs_folder: str | None = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/rag/ask")
def rag_ask(req: AskRequest):
    return pipeline.generate_answer(req.query, k=req.k)


# Optional: protect this in real deployments (API key / internal-only)
@app.post("/rag/build_index")
def rag_build_index(req: BuildIndexRequest):
    folder = req.docs_folder or DOCS_FOLDER
    return build_or_rebuild_index(
        docs_folder=folder,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
        embedding_model=EMB_MODEL,
    )
