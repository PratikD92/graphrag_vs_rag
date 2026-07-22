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
    "cost",
]
comparison = pd.DataFrame(
    {
        "RAG": (rag_scores[metrics].mean() * 100),
        "GraphRAG": (graphrag_scores[metrics].mean() * 100),
    }
).T

comparison = comparison.map(lambda x: f"{x:.2f}%")
comparison["cost"] = comparison["cost"].map(
    lambda x: f"${float(x.replace('%', ''))/100:.4f}"
)

comparison = comparison.round(3)
comparison.columns = [col.replace("_", " ").title() for col in comparison.columns]


st.subheader(f"Latest Average Evaluation Scores | {config['timestamp']}")
st.table(comparison)


st.subheader(f"Latest Benchmark and Configuration")
st.table(config_df)


#######################################################
# RUN EVALUATION FORM
#######################################################

st.divider()
with st.form("run_evaluation"):
    st.header("Run Evaluation")
    st.subheader("Evaluation Configuration")
    st.write("Applicable for both RAG and GraphRAG")

    col1, col2 = st.columns(2)

    with col1:

        embedding_model = st.selectbox(
            "Embedding Model",
            [
                # OpenAI
                "openai/text-embedding-3-small",
                "openai/text-embedding-3-large",
                # Google
                "google/text-embedding-004",
                "google/gemini-embedding-001",
            ],
        )

        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=5000,
            value=500,
            step=100,
        )

        prompt_version = st.selectbox(
            "Prompt Version",
            ["v1", "v2"],
        )

    with col2:
        llm_model = st.selectbox(
            "LLM Model",
            [
                # OpenAI
                "openai/gpt-4o-mini",
                "openai/gpt-5.5",
                # Anthropic
                "anthropic/claude-opus-4",
                "anthropic/claude-sonnet-4",
                # Google
                "google/gemini-2.5-pro",
                "google/gemini-2.5-flash",
                # Meta
                "meta-llama/llama-3.3-70b-instruct",
                "meta-llama/llama-3.1-8b-instruct",
            ],
        )

        chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=1000,
            value=100,
            step=50,
        )

    st.markdown("### Evaluation Metrics")

    mcol1, mcol2 = st.columns(2)

    with mcol1:
        st.checkbox("Faithfulness", value=True, disabled=True)
        st.checkbox("Answer Correctness", value=True, disabled=True)
        st.checkbox("Answer Relevancy", value=True, disabled=True)

    with mcol2:
        st.checkbox("Context Precision", value=True, disabled=True)
        st.checkbox("Context Recall", value=True, disabled=True)

    submitted = st.form_submit_button(
        "Run Evaluation",
        use_container_width=True,
        type="primary",
    )

if submitted:
    st.success("Evaluation started!")
