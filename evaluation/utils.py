import pandas as pd
from datasets import Dataset

import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=ResourceWarning, module="ragas")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from openai import AsyncOpenAI
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# COSTING
# For RAG
from litellm import cost_per_token
from .config import EVALUATION_EMBEDDING_MODEL, EVALUATION_LLM_MODEL, EVALUATION_API_KEY

# For ragas
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# Variables
eval_cost = 0.0


def list_run_dirs() -> list[Path]:
    current_dir = Path(__file__).parent
    run_dirs = current_dir / "runs"
    runs_dir = []
    # run_dirs = []
    for d in run_dirs.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            suffix = d.name[len("run_") :]
            if suffix.isdigit() and (d / "config.yaml").exists():
                runs_dir.append((int(suffix), d))

    runs_dir.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in runs_dir]


def get_latest_run_dir():
    current_dir = Path(__file__).parent
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
    return last_run_dir


def get_run_dir(current_dir):
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
    return current_run_dir


def save_intermediate_run_results(current_run_dir, mode, rows):
    """
    Saves intermediate results of running graphRAG/RAG over golden dataset to capture
    question, expected_answer, source, retrieved_context, generated_answer,
    total_latency_ms, prompt_tokens, completion_tokens, total_tokens
    in a csv
    """
    df = pd.DataFrame(rows)
    df.to_csv(current_run_dir / f"{mode}_run_results.csv", index=False)
    print(f"{mode} runs saved to {current_run_dir / f'{mode}_run_results.csv'}")
    return df


def stringify_graphrag_context(context):

    parts = []

    for key, df in context.items():

        if df is None or len(df) == 0:
            continue

        parts.append(f"===== {key.upper()} =====")

        parts.append(df.to_string())

    return "\n".join(parts)


def create_dataset(df):
    dataset = Dataset.from_dict(
        {
            "user_input": df["question"].tolist(),
            "response": df["generated_answer"].tolist(),
            "reference": df["expected_answer"].tolist(),
            "retrieved_contexts": [[ctx] for ctx in df["retrieved_context"]],
        }
    )

    return dataset


def score_and_save(df, current_run_dir, mode):
    # df = pd.read_csv(current_run_dir / f"{mode}_run_results.csv")
    dataset = create_dataset(df)
    required_cols = [
        "retrieval_latency_ms",
        "llm_latency_ms",
        "total_latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
    ]

    # Add missing columns with value 0
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    metadata = df[required_cols].reset_index(drop=True)

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=EVALUATION_LLM_MODEL))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=EVALUATION_EMBEDDING_MODEL)
    )

    # client = AsyncOpenAI(api_key=EVALUATION_API_KEY)
    # evaluator_llm = llm_factory(EVALUATION_LLM_MODEL, client=client)
    # evaluator_embeddings = embedding_factory(
    #     provider="openai", model=EVALUATION_EMBEDDING_MODEL, client=client
    # )

    with get_openai_callback() as cb:
        results = evaluate(
            dataset,
            metrics=[
                AnswerCorrectness(),
                AnswerRelevancy(),
                Faithfulness(),
                ContextPrecision(),
                ContextRecall(),
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
    eval_cost = cb.total_cost
    scores = results.to_pandas().reset_index(drop=True)
    final_df = pd.concat([scores, metadata], axis=1)
    final_df.to_csv(current_run_dir / f"{mode}_scores.csv", index=False)

    print(f"{mode} scores saved to {current_run_dir / f'{mode}_scores.csv'}")

    return eval_cost


def score_and_save_without_cost(df, current_run_dir, mode):
    # df = pd.read_csv(current_run_dir / f"{mode}_run_results.csv")
    dataset = create_dataset(df)

    results = evaluate(
        dataset,
        metrics=[
            AnswerCorrectness(),
            AnswerRelevancy(),
            Faithfulness(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
    )

    results.to_pandas().to_csv(current_run_dir / f"{mode}_scores.csv", index=False)

    print(f"{mode} scores saved to {current_run_dir / f'{mode}_scores.csv'}")


def rag_query_cost_calculator(run_df):
    # To be integrated with run_ragas_eval while iteration happens there
    # df = pd.read_csv("runs/run_1/rag_run_results.csv")

    for _, row in run_df.iterrows():
        prompt_tokens = row["prompt_tokens"]
        completion_tokens = row["completion_tokens"]

        prompt_cost, completion_cost_ = cost_per_token(
            model=EVALUATION_LLM_MODEL,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        total_cost = prompt_cost + completion_cost_
        run_df.at[row.name, "cost"] = round(total_cost, 4)
    # df.to_csv("runs/run_1/rag_costs.csv", index=False)
    # return total_cost
    return run_df


def rag_per_query_cost_calculator(run_df):
    # To be integrated with run_ragas_eval while iteration happens there
    # df = pd.read_csv("runs/run_1/rag_run_results.csv")

    for _, row in run_df.iterrows():
        prompt_tokens = row["prompt_tokens"]
        completion_tokens = row["completion_tokens"]

        prompt_cost, completion_cost_ = cost_per_token(
            model=EVALUATION_LLM_MODEL,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        total_cost = prompt_cost + completion_cost_
        run_df.at[row.name, "cost"] = round(total_cost, 4)
    # df.to_csv("runs/run_1/rag_costs.csv", index=False)
    # return total_cost
    return run_df


# Checking cost calculation
# cost_df = rag_query_cost_calculator()
# print(cost_df)
