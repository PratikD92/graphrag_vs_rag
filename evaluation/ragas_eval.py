from ragas import evaluate
from datasets import Dataset

from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    AnswerRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import time

# For ragas
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

current_dir = Path(__file__).parent
parent_dir = Path(__file__).parent.parent
load_dotenv(parent_dir / ".env")

# mode = "graphrag"
mode = "rag"

# df = pd.read_csv(current_dir / "rag_run_dataset.csv")
# df = pd.read_csv(current_dir / f"{mode}_numerical_run_dataset2.csv")
df = pd.read_csv("runs/run_1/rag_run_results.csv")
# print(df.head())

# df = df[:10]
dataset = Dataset.from_dict(
    {
        "user_input": df["question"].tolist(),
        "response": df["generated_answer"].tolist(),
        "reference": df["expected_answer"].tolist(),
        "retrieved_contexts": [[ctx] for ctx in df["retrieved_context"]],
    }
)

# print(dataset[0])
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
df.to_csv("runs/run_1/ragas_costs.csv", index=False)

# print(results)
# timestamp = time.strftime("%Y%m%d_%H%M")
# results.to_pandas().to_csv(
#     # current_dir / "eval_scores" / "ragas_rag_scores.csv",
#     current_dir / "eval_scores" / f"ragas_{mode}_numerical_{timestamp}_scores2.csv",
#     index=False,
# )
