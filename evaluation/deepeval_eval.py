from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path

current_dir = Path(__file__).parent
parent_dir = Path(__file__).parent.parent
load_dotenv(parent_dir / ".env")

metric = HallucinationMetric()

# df = pd.read_csv(current_dir / "rag_run_dataset.csv")
df = pd.read_csv(current_dir / "graphrag_run_dataset.csv")

test_cases = []

for _, row in df.iterrows():

    tc = LLMTestCase(
        input=row["question"],
        actual_output=row["generated_answer"],
        expected_output=row["expected_answer"],
        context=[row["retrieved_context"]],
    )

    test_cases.append(tc)

evaluate(
    test_cases=test_cases,
    metrics=[metric],
)
