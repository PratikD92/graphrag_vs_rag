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

current_dir = Path(__file__).parent
parent_dir = Path(__file__).parent.parent
load_dotenv(parent_dir / ".env")


# df = pd.read_csv(current_dir / "rag_results.csv")
df = pd.read_csv(current_dir / "graphrag_results.csv")
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
        Faithfulness(),
        ContextPrecision(),
        ContextRecall(),
    ],
)
print(results)
results.to_pandas().to_csv(
    current_dir / "ragas_graphrag_scores.csv",
    index=False,
)
