from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_chunks():
    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    input_dir = PROJECT_ROOT / "input"

    for txt_file in input_dir.glob("*.txt"):
        print(f"Processing {txt_file.name}")
        text = txt_file.read_text(encoding="utf-8")

        for chunk in splitter.split_text(text):
            chunks.append({"text_content": chunk, "source": txt_file.name})
    print(f"Created {len(chunks)} chunks")

    return chunks
