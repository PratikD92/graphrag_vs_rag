from ragas import evaluate
from datasets import Dataset

# from ragas.metrics import (
#     Faithfulness,
#     AnswerCorrectness,
#     AnswerRelevancy,
#     ContextPrecision,
#     ContextRecall,
# )
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import os
import asyncio
import time
import yaml
from datetime import datetime

from RAG_pipeline.query_RAG import generate_rag_answer
from query_graph import generate
from .utils import (
    save_intermediate_run_results,
    score_and_save,
    stringify_graphrag_context,
    get_run_dir,
    rag_query_cost_calculator,
)
from .config import EVALUATION_LLM_MODEL, EVALUATION_EMBEDDING_MODEL

current_dir = Path(__file__).parent
# parent_dir = Path(__file__).parent.parent
# load_dotenv(parent_dir / ".env")


# Create and use run directory in incremental fashion to store runs
current_run_dir = get_run_dir(current_dir)
golden_csv = "golden_dataset.csv"

# modes = ["graphrag", "rag"]
modes = ["rag"]

# Load golden dataset
gold = pd.read_csv(current_dir / golden_csv)
questions = len(gold)


async def generate_async(query: str):
    return await generate(query=query)


for mode in modes:
    rows = []
    print(f"Evaluating for: {mode}")
    if mode == "rag":
        for _, row in gold[:1].iterrows():
            # print(f"Processing query {_+1}/{questions}")
            print(f"Processing query {_+1}/{questions}".ljust(50), end="\r", flush=True)

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
            # print(f"Time -> {total_latency/1000:.2f} s\n")

        # Save intermediate run results
        run_df = save_intermediate_run_results(current_run_dir, mode, rows)

        # Calculate costs
        cost_df = rag_query_cost_calculator(run_df)

        # Evaluate, Score and save results
        score_and_save(cost_df, current_run_dir, mode)

    else:
        for _, row in gold.iterrows():

            # if _ >= 5:  # Process only first 5 questions
            #     break
            # print(f"Processing query {_+1}/{questions}")
            print(f"Processing query {_+1}/{questions}".ljust(50), end="\r", flush=True)
            question = row["question"]
            # answer, context, total_latency = asyncio.run(generate(query=question))
            answer, context, total_latency = asyncio.run(generate_async(query=question))

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
            # print(f"Time -> {total_latency/1000:.2f} s\n")

        # Save intermediate run results
        run_df = save_intermediate_run_results(current_run_dir, mode, rows)

        # Evaluate, Score and save results
        score_and_save(run_df, current_run_dir, mode)

# Save yaml configuration for this run
configuration = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "golden_csv": golden_csv,
    "questions": questions,
    "evaluation_llm_model": EVALUATION_LLM_MODEL,
    "evaluation_embedding_model": EVALUATION_EMBEDDING_MODEL,
    "LLM model (GraphRAG & RAG)": EVALUATION_LLM_MODEL,
    "Embedding model (GraphRAG & RAG)": EVALUATION_EMBEDDING_MODEL,
    "Costing": {
        "RAG": cost_df["cost"].sum(),
        "RAG_Evaluation": "To be calculated",
        "GraphRAG": "To be calculated",
        "GraphRAG_Evaluation": "To be calculated",
    },
}
with open(current_run_dir / "config.yaml", "w") as f:
    yaml.dump(configuration, f)
