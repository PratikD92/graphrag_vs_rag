import os
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

# Weaviate configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM model for RAG
RAG_LLM_MODEL = os.getenv("RAG_LLM_MODEL")

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Chunking strategy
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))


# Collection nameing convention
COLLECTION_NAME = (
    f"ICICI_Policy_documents_{EMBEDDING_MODEL}_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
).replace("-", "_")
