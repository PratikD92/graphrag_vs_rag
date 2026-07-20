from ragas import evaluate
from datasets import Dataset

from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import os
import asyncio
import time

from RAG_pipeline.query_RAG import generate_rag_answer
from query_graph import generate
from .utils import (
    save_intermediate_run_results,
    score_and_save,
    stringify_graphrag_context,
    get_run_dir,
)

current_dir = Path(__file__).parent
parent_dir = Path(__file__).parent.parent
load_dotenv(parent_dir / ".env")


# Create and use run directory in incremental fashion to store runs
current_run_dir = get_run_dir(current_dir)

modes = ["graphrag", "rag"]

# Load golden dataset
gold = pd.read_csv(current_dir / "golden_dataset.csv")
questions = len(gold)

for mode in modes:
    rows = []
    print(f"Evaluating for: {mode}")
    if mode == "rag":
        for _, row in gold.iterrows():
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

        # Evaluate, Score and save results
        score_and_save(run_df, current_run_dir, mode)

    else:
        for _, row in gold.iterrows():

            # if _ >= 5:  # Process only first 5 questions
            #     break
            # print(f"Processing query {_+1}/{questions}")
            print(f"Processing query {_+1}/{questions}".ljust(50), end="\r", flush=True)
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
            # print(f"Time -> {total_latency/1000:.2f} s\n")

        # Save intermediate run results
        run_df = save_intermediate_run_results(current_run_dir, mode, rows)

        # Evaluate, Score and save results
        score_and_save(run_df, current_run_dir, mode)
