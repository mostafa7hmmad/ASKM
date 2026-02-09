# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    JSON_PATH = os.getenv("JSON_PATH", "final_rag_corpus.json")
    DB_DIR = os.getenv("DB_DIR", "Data/chroma_db")
    DOCSTORE_DIR = os.getenv("DOCSTORE_DIR", "Data/docstore")

    DENSE_W = float(os.getenv("DENSE_WEIGHT"))
    SPARSE_W = float(os.getenv("SPARSE_WEIGHT"))

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))

    K_DENSE = int(os.getenv("TOP_K_DENSE"))
    K_SPARSE = int(os.getenv("TOP_K_SPARSE"))

    LLM_TEMP = float(os.getenv("LLM_TEMPERATURE"))
    TOP_P = float(os.getenv("TOP_P"))
    MAX_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS"))
