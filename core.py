# core.py - WITH FEW-SHOT LEARNING SUPPORT
import logging
import pickle
import json
from typing import Tuple, List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Local imports
from config import Config
from local_docstore import LocalDocStore
from ingest import LocalHfEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_examples(path="few_shot_examples.json"):
    """Load few-shot examples for the model"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load few-shot examples from {path}: {e}")
        return []


class SimpleHybridRetriever(BaseRetriever):
    """
    Custom hybrid retriever combining dense (vector) and sparse (BM25) retrieval.
    Replaces the deprecated EnsembleRetriever with backward compatibility.
    """
    dense_retriever: object
    sparse_retriever: BM25Retriever
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Internal method called by invoke()"""
        # Get results from both retrievers
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)
        
        # Simple merging: combine and deduplicate by content
        seen = set()
        merged = []
        
        # Add dense results first (weighted higher by default)
        for doc in dense_docs:
            doc_id = doc.page_content[:100]  # Use first 100 chars as ID
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc)
        
        # Add sparse results
        for doc in sparse_docs:
            doc_id = doc.page_content[:100]
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc)
        
        # Return top K results
        total_k = getattr(Config, 'K_DENSE', 3) + getattr(Config, 'K_SPARSE', 2)
        return merged[:total_k]
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Backward compatible method for older LangChain code.
        This is what your api.py is calling.
        """
        return self._get_relevant_documents(query, run_manager=None)


def load_rag_system() -> Tuple[SimpleHybridRetriever, object]:
    """
    Load the RAG system with a simplified retriever architecture.
    """
    # --- 1. Load DocStore and Parent Docs ---
    store = LocalDocStore(Config.DOCSTORE_DIR)
    parent_docs: List[Document] = []
    
    for key in store.yield_keys():
        raw = store.get(key)
        if raw:
            parent_docs.append(pickle.loads(raw))
            
    if not parent_docs:
        raise RuntimeError("DocStore empty. Run ingest.py first.")

    logger.info(f"Loaded {len(parent_docs)} parent documents from docstore")

    # --- 2. Dense retriever (Chroma for vector search) ---
    vectorstore = Chroma(
        collection_name="islamic_rag",
        embedding_function=LocalHfEmbeddings(),
        persist_directory=Config.DB_DIR
    )
    
    # Use vectorstore as retriever directly
    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": Config.K_DENSE}
    )

    # --- 3. Sparse retriever (BM25 for keyword search) ---
    sparse_retriever = BM25Retriever.from_documents(parent_docs)
    sparse_retriever.k = Config.K_SPARSE

    # --- 4. Create hybrid retriever ---
    hybrid_retriever = SimpleHybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        dense_weight=Config.DENSE_W,
        sparse_weight=Config.SPARSE_W
    )

    logger.info("Hybrid retriever initialized successfully")

    # --- 5. LLM Setup (Gemini) ---
    llm = None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=Config.LLM_TEMP,
            top_p=Config.TOP_P,
            max_output_tokens=Config.MAX_TOKENS

        )
        logger.info("LLM initialized successfully")
    except Exception as e:
        logger.warning("LLM not available: %s", e)
        llm = None

    return hybrid_retriever, llm
