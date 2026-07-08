import pandas as pd
import asyncio

gold = pd.read_csv("evaluation/golden_dataset.csv")

rows = []

for _, row in gold.iterrows():

    question = row["question"]

    answer, context, latency = await generate(question)

    rows.append(
        {
            "question": question,
            "expected_answer": row["expected_answer"],
            "source": row["source"],
            "retrieved_context": stringify_graphrag_context(context),
            "generated_answer": answer,
            "total_latency_ms": latency,
            # GraphRAG currently doesn't expose token usage
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    )

df = pd.DataFrame(rows)

df.to_csv("evaluation/graphrag_results.csv", index=False)
