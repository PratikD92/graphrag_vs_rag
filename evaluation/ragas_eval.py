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
import time

current_dir = Path(__file__).parent
parent_dir = Path(__file__).parent.parent
load_dotenv(parent_dir / ".env")

# mode = "graphrag"
mode = "rag"

# df = pd.read_csv(current_dir / "rag_run_dataset.csv")
df = pd.read_csv(current_dir / f"{mode}_numerical_run_dataset2.csv")
# print(df.head())

dataset = Dataset.from_dict(
    {
        "user_input": df["question"].tolist(),
        "response": df["generated_answer"].tolist(),
        "reference": df["expected_answer"].tolist(),
        "retrieved_contexts": [[ctx] for ctx in df["retrieved_context"]],
    }
)

# print(dataset[0])
results = evaluate(
    dataset,
    metrics=[
        AnswerCorrectness(),
        AnswerRelevancy(),
        Faithfulness(),
        ContextPrecision(),
        ContextRecall(),
    ],
)
print(results)
timestamp = time.strftime("%Y%m%d_%H%M")
results.to_pandas().to_csv(
    # current_dir / "eval_scores" / "ragas_rag_scores.csv",
    current_dir / "eval_scores" / f"ragas_{mode}_numerical_{timestamp}_scores2.csv",
    index=False,
)
