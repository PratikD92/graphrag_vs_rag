from pathlib import Path
from dotenv import load_dotenv
import os

parent_dir = Path(__file__).parent.parent
load_dotenv(parent_dir / ".env")

EVALUATION_LLM_MODEL = os.getenv("EVALUATION_LLM_MODEL")
EVALUATION_EMBEDDING_MODEL = os.getenv("EVALUATION_EMBEDDING_MODEL")
EVALUATION_API_KEY = os.getenv("OPENAI_API_KEY")
