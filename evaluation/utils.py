import pandas as pd
from datasets import Dataset
from ragas import evaluate

from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    AnswerRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

# COSTING
# For RAG
from litellm import cost_per_token

# For ragas
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def save_run_results(current_run_dir, mode, rows):
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

    return "\n\n".join(parts)


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

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    with get_openai_callback() as cb:
        results = evaluate(
            dataset,
            metrics=[
                AnswerCorrectness(),
                AnswerRelevancy(),
                Faithfulness(),
                LLMContextPrecisionWithReference(),
                LLMContextRecall(),
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
    print(f"{mode} eval cost: ${cb.total_cost:.4f}")

    results.to_pandas().to_csv(current_run_dir / f"{mode}_scores.csv", index=False)

    print(f"{mode} scores saved to {current_run_dir / f'{mode}_scores.csv'}")


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


def rag_query_cost_calculator():
    # To be integrated with run_ragas_eval while iteration happens there
    df = pd.read_csv("runs/run_1/rag_run_results.csv")

    for _, row in df.iterrows():
        prompt_tokens = row["prompt_tokens"]
        completion_tokens = row["completion_tokens"]

        prompt_cost, completion_cost_ = cost_per_token(
            model="gpt-4o",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        total_cost = prompt_cost + completion_cost_
        df.at[row.name, "cost"] = f"${round(total_cost, 4)}"
    df.to_csv("runs/run_1/rag_costs.csv", index=False)
    # return total_cost
    return df


# Checking cost calculation
# cost_df = rag_query_cost_calculator()
# print(cost_df)
