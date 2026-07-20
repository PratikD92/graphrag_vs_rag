from pathlib import Path
from dotenv import load_dotenv
import os

parent_dir = Path(__file__).parent.parent
load_dotenv(parent_dir / ".env")

LLM_MODEL = os.getenv("EVALUATION_LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EVALUATION_EMBEDDING_MODEL")

print(f"LLM_MODEL: {LLM_MODEL}")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
