import asyncio
import os
import pandas as pd
import time
from pathlib import Path

from query_graph import generate


def stringify_graphrag_context(context):

    parts = []

    for key, df in context.items():

        if df is None or len(df) == 0:
            continue

        parts.append(f"===== {key.upper()} =====")

        parts.append(df.to_string())

    return "\n\n".join(parts)


current_dir = Path(__file__).parent
gold = pd.read_csv(current_dir / "golden_dataset.csv")
rows = []
questions = len(gold)

for _, row in gold.iterrows():
    if _ >= 5:  # Process only first 5 questions
        break
    print(f"Processing query {_+1}/{questions}")
    question = row["question"]
    answer, context, total_latency = asyncio.run(generate(query=question))

    rows.append(
        {
            "question": question,
            "expected_answer": row["expected_answer"],
            "source": row["source"],
            "retrieved_context": stringify_graphrag_context(context),
            "generated_answer": answer,
            "total_latency_ms": f"{total_latency:.2f}",
            # GraphRAG currently doesn't expose token usage
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    )
    print(f"Time -> {total_latency/1000:.2f} s\n")
df = pd.DataFrame(rows)

df.to_csv(current_dir / "graphrag_results.csv", index=False)
