import asyncio
import os
import pandas as pd
import time
from pathlib import Path

from RAG_pipeline.query_RAG import generate_rag_answer

current_dir = Path(__file__).parent
gold = pd.read_csv(current_dir / "golden_dataset.csv")
rows = []
questions = len(gold)

for _, row in gold.iterrows():
    if _ >= 1:  # Process only first 5 questions
        break

    print(f"Processing query {_+1}/{questions}")

    question = row["question"]
    expected_answer = row["expected_answer"]

    # Generate RAG answer
    (
        rag_answer,
        source_docs,
        context,
        retrieval_latency,
        llm_latency,
        total_latency,
        total_tokens,
        prompt_tokens,
        completion_tokens,
    ) = generate_rag_answer(question)

    # Store results
    rows.append(
        {
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "source": row["source"],
            "retrieved_context": context,
            "generated_answer": rag_answer,
            "retrieval_latency_ms": retrieval_latency,
            "llm_latency_ms": llm_latency,
            "total_latency_ms": total_latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    )

    print(f"Time -> {total_latency/1000:.2f} s\n")


df = pd.DataFrame(rows)

df.to_csv("evaluation/rag_results.csv", index=False)
