# ingest.py
import os
import json
import pickle
from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
from local_docstore import LocalDocStore

class LocalHfEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        if vectors.ndim == 1:
            vectors = np.expand_dims(vectors, 0)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0].tolist()


def run_ingestion():
    os.makedirs(Config.DB_DIR, exist_ok=True)
    os.makedirs(Config.DOCSTORE_DIR, exist_ok=True)

    if not os.path.exists(Config.JSON_PATH):
        raise FileNotFoundError(f"JSON not found: {Config.JSON_PATH}")

    with open(Config.JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    parent_docs: List[Document] = []

    # Parse JSON into parent documents
    for book in data.get("books", []):
        title = book.get("book_metadata", {}).get("title") or book.get("book_id") or "Unknown"
        for chunk in book.get("chunks", []):
            question = chunk.get("question") or ""
            answer = chunk.get("answer") or chunk.get("content") or ""
            content = f"Question: {question}\nAnswer: {answer}"
            metadata = {
                "id": chunk.get("chunk_id") or chunk.get("id"),
                "book_id": chunk.get("book_id") or book.get("book_id"),
                "source": title
            }
            doc = Document(page_content=content, metadata=metadata)
            parent_docs.append(doc)

    if not parent_docs:
        raise RuntimeError("No documents created from JSON — check JSON structure.")

    # --- Split children for Chroma embeddings ---
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    child_docs = []
    for parent in parent_docs:
        splits = child_splitter.split_text(parent.page_content)
        parent_id = parent.metadata["id"]
        for chunk in splits:
            child_docs.append(Document(
                page_content=chunk,
                metadata={"parent_id": parent_id, "source": parent.metadata.get("source")}
            ))

    # --- Chroma vectorstore for children ---
    hf_embeddings = LocalHfEmbeddings()
    vectorstore = Chroma(
        collection_name="islamic_rag",
        embedding_function=hf_embeddings,
        persist_directory=Config.DB_DIR
    )
    vectorstore.add_documents(child_docs)
    vectorstore.persist()

    # --- DocStore for parents (BM25 / retrieval) ---
    store = LocalDocStore(Config.DOCSTORE_DIR)
    for doc in parent_docs:
        key = doc.metadata["id"]
        store.set(key, pickle.dumps(doc))

    print(f"Ingestion completed: {len(parent_docs)} parents, {len(child_docs)} children.")
    print(f"Chroma DB: {Config.DB_DIR}, DocStore: {Config.DOCSTORE_DIR}")


if __name__ == "__main__":
    run_ingestion()
