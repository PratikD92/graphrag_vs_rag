import streamlit as st

st.header("Evaluation", divider="rainbow")
st.set_page_config(page_title="GraphRAG vs RAG | Eval", layout="wide")

from evaluation.utils import get_latest_run_dir
import yaml
import pandas as pd

current_run_dir = get_latest_run_dir()

# Read config.yaml
config_path = current_run_dir / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

rows = []
for key, value in config.items():
    if key == "timestamp":
        continue
    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            rows.append(
                {
                    "Parameter": f"{key} → {subkey}",
                    "Value": str(subvalue),
                }
            )
    else:
        rows.append(
            {
                "Parameter": key,
                "Value": str(value),
            }
        )

config_df = pd.DataFrame(rows)
config_df["Parameter"] = config_df["Parameter"].str.replace("_", " ").str.title()
config_df["Parameter"] = (
    config_df["Parameter"].str.replace("Llm", "LLM").str.replace("Csv", "CSV")
)


# Scores table

graphrag_scores = pd.read_csv(current_run_dir / "graphrag_scores.csv")
rag_scores = pd.read_csv(current_run_dir / "rag_scores.csv")

metrics = [
    "faithfulness",
    "answer_correctness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
comparison = pd.DataFrame(
    {
        "RAG": rag_scores[metrics].mean(),
        "GraphRAG": graphrag_scores[metrics].mean(),
    }
).T

comparison = comparison.round(3)
comparison.columns = [col.replace("_", " ").title() for col in comparison.columns]


st.subheader(f"Latest Evaluation Scores | {config['timestamp']}")
st.table(comparison)


st.subheader(f"Latest Benchmark and Configuration")
st.table(config_df)
