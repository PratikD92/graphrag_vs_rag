import streamlit as st

st.set_page_config(page_title="GraphRAG vs RAG | Eval", layout="wide")
import pandas as pd

st.header("Evaluation", divider="rainbow")

from evaluation.utils import get_latest_run_dir, list_run_dirs
from evaluation.run_ragas_eval import evaluate_ragas
import yaml

available_runs = list_run_dirs()

if not available_runs:
    st.warning("No completed evaluation runs found yet. Run an evaluation first.")
    st.stop()

run_labels = [d.name for d in available_runs]
selected_label = st.selectbox("Select run", run_labels, index=0)
current_run_dir = available_runs[run_labels.index(selected_label)]

# Read config.yaml
config_path = current_run_dir / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


parameters, values = [], []
for key, value in config.items():
    if key == "timestamp":
        continue
    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            parameters.append(f"{key} → {subkey}")
            values.append(str(subvalue))
    else:
        parameters.append(key)
        values.append(str(value))

config_df = pd.DataFrame({"Parameter": parameters, "Value": values})

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
            ],
        )

        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=3000,
            value=1200,
            step=100,
            disabled=True,
        )

        prompt_version = st.selectbox(
            "Prompt Version",
            ["v1", "v2"],
        )

    with col2:
        generation_llm_model = st.selectbox(
            "Generation LLM Model",
            [
                # OpenAI
                "gpt-4o-mini",
                "gpt-4.1-nano",
                "gpt-5.4-nano",
            ],
        )

        chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=1000,
            value=200,
            step=50,
            disabled=True,
        )

        sample_size = st.number_input(
            "Sample Size",
            min_value=1,
            max_value=35,
            value=35,
            step=1,
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
    try:
        evaluate_ragas(
            streamlit_prompt_version=prompt_version,
            llm_model=generation_llm_model,
            sample_size=sample_size,
        )
        st.success("Evaluation Successful!")
        st.rerun()
    except Exception as e:
        st.error(f"Evaluation failed! {e}")
