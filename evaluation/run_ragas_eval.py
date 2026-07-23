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
from query_graph import generate, generate_with_cost
from .utils import (
    save_intermediate_run_results,
    score_and_save,
    stringify_graphrag_context,
    get_run_dir,
    rag_query_cost_calculator,
)
from .config import (
    EVALUATION_LLM_MODEL,
    EVALUATION_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

current_dir = Path(__file__).parent
# parent_dir = Path(__file__).parent.parent
# load_dotenv(parent_dir / ".env")


# Create and use run directory in incremental fashion to store runs
current_run_dir = get_run_dir(current_dir)
golden_csv = "golden_dataset.csv"

# modes = ["graphrag"]

# Load golden dataset
gold = pd.read_csv(current_dir / golden_csv)
questions = len(gold)


async def generate_async(query: str):
    return await generate(query=query)


async def generate_with_cost_async(query: str, model: str | None = None):
    # return await generate_with_cost(query=query)
    return await generate_with_cost(query=query, model=model)


def evaluate_ragas(llm_model: str | None = None, sample_size: int = 35):

    streamlit_model = None
    if llm_model:
        streamlit_model = llm_model

    modes = ["rag", "graphrag"]

    for mode in modes:
        rows = []
        print(f"Evaluating for: {mode}")
        if mode == "rag":
            for _, row in gold[:sample_size].iterrows():
                # print(f"Processing query {_+1}/{questions}")
                print(
                    f"Processing query {_+1}/{questions}".ljust(50),
                    end="\r",
                    flush=True,
                )

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
                    rag_model_used,
                ) = generate_rag_answer(question, llm_model=streamlit_model)

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
            rag_cost_df = rag_query_cost_calculator(run_df)

            # Evaluate, Score and save results
            rag_eval_cost = score_and_save(rag_cost_df, current_run_dir, mode)

        else:
            for _, row in gold[:sample_size].iterrows():

                # if _ >= 5:  # Process only first 5 questions
                #     break
                # print(f"Processing query {_+1}/{questions}")
                print(
                    f"Processing query {_+1}/{questions}".ljust(50),
                    end="\r",
                    flush=True,
                )
                question = row["question"]
                answer, context, total_latency, query_cost = asyncio.run(
                    generate_with_cost_async(query=question, model=streamlit_model)
                )

                rows.append(
                    {
                        "question": question,
                        "expected_answer": row["expected_answer"],
                        "source": row["source"],
                        "retrieved_context": stringify_graphrag_context(context),
                        "generated_answer": answer,
                        "total_latency_ms": f"{total_latency:.2f}",
                        "prompt_tokens": query_cost["prompt_tokens"],
                        "completion_tokens": query_cost["completion_tokens"],
                        "total_tokens": query_cost["total_tokens"],
                        "cost": query_cost["total_cost"],
                    }
                )
                # print(f"Time -> {total_latency/1000:.2f} s\n")

            # Save intermediate run results
            graphrag_run_df = save_intermediate_run_results(current_run_dir, mode, rows)

            # Evaluate, Score and save results
            graphrag_eval_cost = score_and_save(graphrag_run_df, current_run_dir, mode)

    # Save yaml configuration for this run
    configuration = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "golden_csv": golden_csv,
        "questions": questions,
        "evaluation_llm_model": EVALUATION_LLM_MODEL,
        "evaluation_embedding_model": EVALUATION_EMBEDDING_MODEL,
        "LLM model (GraphRAG & RAG)": rag_model_used,
        "Embedding model (GraphRAG & RAG)": EVALUATION_EMBEDDING_MODEL,
        "Costing": {
            "RAG": f"${float(rag_cost_df['cost'].sum()):.4f}",
            "RAG_Evaluation": f"${rag_eval_cost:.4f}",
            "GraphRAG": f"${float(graphrag_run_df['cost'].sum()):.4f}",
            "GraphRAG_Evaluation": f"${graphrag_eval_cost:.4f}",
        },
        "Chunk Size": CHUNK_SIZE,
        "Chunk Overlap": CHUNK_OVERLAP,
    }
    with open(current_run_dir / "config.yaml", "w") as f:
        yaml.dump(configuration, f)
