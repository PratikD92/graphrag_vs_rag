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
from .utils import save_run_results, score_and_save, stringify_graphrag_context


current_dir = Path(__file__).parent
parent_dir = Path(__file__).parent.parent
load_dotenv(parent_dir / ".env")

runs_dir = current_dir / "runs"
if not runs_dir.exists():
    runs_dir.mkdir()

# Find existing run_# directories and get the highest number
existing_run_numbers = []
for d in runs_dir.iterdir():
    if d.is_dir() and d.name.startswith("run_"):
        suffix = d.name[len("run_") :]
        if suffix.isdigit():
            existing_run_numbers.append(int(suffix))

last_run = max(existing_run_numbers, default=0)
last_run_dir = runs_dir / f"run_{last_run}"

# Reuse the last run directory if it exists and is empty
if last_run > 0 and last_run_dir.exists() and not any(last_run_dir.iterdir()):
    current_run = last_run
    current_run_dir = last_run_dir
else:
    current_run = last_run + 1
    current_run_dir = runs_dir / f"run_{current_run}"
    current_run_dir.mkdir()


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
        run_df = save_run_results(current_run_dir, mode, rows)

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
        run_df = save_run_results(current_run_dir, mode, rows)

        # Evaluate, Score and save results
        score_and_save(run_df, current_run_dir, mode)
