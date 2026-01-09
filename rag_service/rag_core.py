import os
import uuid
from typing import List, Dict, Any, Optional

import requests
from fastapi import HTTPException

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

from sentence_transformers import SentenceTransformer
import chromadb


def process_all_pdfs(folder_path: str):
    all_documents = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            loader = PyMuPDFLoader(full_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = splitter.split_documents(documents)

            # Add source metadata
            for ch in chunks:
                ch.metadata = ch.metadata or {}
                ch.metadata["source_file"] = filename

            all_documents.extend(chunks)
    return all_documents


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False)


class VectorStore:
    def __init__(self, collection_name: str = "chemistry_docs", persist_directory: str = "./data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, documents, embeddings):
        ids = [str(uuid.uuid4()) for _ in documents]
        metadatas = [doc.metadata or {} for doc in documents]
        texts = [doc.page_content for doc in documents]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, query_embedding, n_results: int = 5):
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return res


class RAGRetriever:
    def __init__(self, embedding_manager: EmbeddingManager, vector_store: VectorStore):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5):
        q_emb = self.embedding_manager.generate_embeddings([query])[0]
        res = self.vector_store.query(q_emb, n_results=k)

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        retrieved = []
        for text, meta, dist in zip(docs, metas, dists):
            retrieved.append(
                {
                    "text": text,
                    "metadata": meta,
                    "distance": dist,
                }
            )
        return retrieved


class LLMIntegration:
    """
    Hugging Face OpenAI-compatible router (matches your notebook).
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "https://router.huggingface.co/v1",
    ):
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_API_TOKEN")
        if not self.api_key:
            raise ValueError("Missing Hugging Face token. Set HF_TOKEN or HUGGING_FACE_API_TOKEN.")

        self.model_name = model_name
        self.api_url = f"{base_url}/chat/completions"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def generate(self, prompt: str, max_new_tokens: int = 500, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful high school chemistry tutor."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

        try:
            resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")


class RAGPipeline:
    def __init__(self, retriever: RAGRetriever, llm: LLMIntegration):
        self.retriever = retriever
        self.llm = llm

    def generate_answer(self, query: str, k: int = 5):
        retrieved = self.retriever.retrieve(query, k=k)

        context_blocks = []
        sources = []
        for i, item in enumerate(retrieved, start=1):
            src = item["metadata"].get("source_file", "unknown")
            sources.append(src)
            context_blocks.append(f"[Source {i}: {src}]\n{item['text']}")

        context = "\n\n".join(context_blocks)

        prompt = f"""You are a secondary school and junior college chemistry tutor.

Rules:
- Use the provided context first.
- If the context is insufficient, say what is missing and then answer using general chemistry knowledge.
- Show steps for calculations and include units.
- Finish by giving an example of a similar exam question and its answer from the context.

Context:
{context}

Question:
{query}
"""

        answer = self.llm.generate(prompt)
        return {
            "query": query,
            "answer": answer,
            "sources": list(dict.fromkeys(sources)),  # unique, keep order
        }


def build_or_rebuild_index(
    docs_folder: str,
    persist_directory: str = "./data/vector_store",
    collection_name: str = "chemistry_docs",
    embedding_model: str = "all-MiniLM-L6-v2",
):
    docs = process_all_pdfs(docs_folder)
    if not docs:
        raise ValueError(f"No PDFs found in {docs_folder}")

    emb = EmbeddingManager(model_name=embedding_model)
    store = VectorStore(collection_name=collection_name, persist_directory=persist_directory)

    texts = [d.page_content for d in docs]
    vectors = emb.generate_embeddings(texts)
    store.add_documents(docs, vectors)
    return {"num_chunks": len(docs), "collection": collection_name, "persist_directory": persist_directory}
