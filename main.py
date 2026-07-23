import streamlit as st

st.set_page_config(page_title="GraphRAG vs RAG", layout="wide")
import asyncio
import os
import pandas as pd
import time

from query_graph import generate
from RAG_pipeline.query_RAG import generate_rag_answer
from generate_visual_graph import generate_graph, generate_full_graph
from graphrag_source_docs import get_graphrag_source_docs

st.title("Intelligent Insurance Policy Assistant")
st.write(
    "Benchmark GraphRAG and traditional RAG on insurance policy documents using retrieval quality, response quality, latency, token usage, and cost"
)
st.divider()


# SESSION STATE VARIABLES INITIALIZATION
def initialize_session_state():
    session_vars = [
        "graphrag_result",
        "graphrag_loading",
        "graphrag_context",
        "rag_result",
        "rag_source_docs",
        "documents",
        "graphrag_time_elapsed",
        "rag_time_elapsed",
    ]

    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None


initialize_session_state()


# RESET SESSION
def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()


result = None
context = None

# List documents and store in session state
st.session_state.documents = os.listdir("input")


####################################
# DIALOGS
####################################


# GraphRAG Answer
@st.dialog("GraphRAG Answer", width="large")
def show_graphrag_dialog():
    st.write(st.session_state.graphrag_result)


# Traditional RAG Answer
@st.dialog("Traditional RAG Answer", width="large")
def show_rag_dialog():
    st.write(st.session_state.rag_result)


# All Documents
@st.dialog("All Documents", width="large")
def show_all_documents():
    st.json(st.session_state.documents)


# Retrieved Graph visualization
@st.dialog("GraphRAG Graph", width="large")
def show_graph_visuals():
    if st.session_state.graphrag_context is None:
        st.write("No context available")
        return
    html = generate_graph(st.session_state.graphrag_context)
    st.components.v1.html(html, height=600)


# Full Graph visualization
@st.dialog("Full GraphRAG Graph", width="large")
def show_full_graph_visuals():
    html = generate_full_graph()
    st.components.v1.html(html, height=600)


# GraphRAG Source documents
@st.dialog("GraphRAG Result's Source Documents", width="large")
def show_graphrag_source_documents():
    if st.session_state.graphrag_context is None:
        st.write("No context available")
        return
    docs = get_graphrag_source_docs(st.session_state.graphrag_context)
    st.table(docs)


# RAG Source documents
@st.dialog("Traditional RAG Result's Source Documents", width="large")
def show_rag_source_documents():
    if st.session_state.rag_source_docs is None:
        st.write("No source documents available")
        return
    df = pd.DataFrame(st.session_state.rag_source_docs)
    st.dataframe(df["source"].unique())


query = st.text_area(
    "Enter your query",
    placeholder="A truck driver is injured during an accident. Which policy documents are relevant and why?",
)
disable_submit = not query


# SUBMIT BUTTON
acols = st.columns(6)
with acols[0]:
    if st.button("Submit", key="graphrag_submit", type="primary"):
        # reset_session()
        st.session_state.graphrag_loading = True
with acols[-2]:
    if st.button("View Documents Indexed", key="view_documents"):
        show_all_documents()
with acols[-1]:
    if st.button("View Full Graph", key="view_full_graph"):
        show_full_graph_visuals()

st.markdown("---")

cols = st.columns(6)
with cols[-1]:
    st.button(
        "Reset Results",
        on_click=lambda: reset_session(),
        type="secondary",
    )

columns = st.columns(2, border=True)

##########################################
# GRAPHRAG SECTION
##########################################

with columns[0]:

    gcols = st.columns([4, 1])

    # with gcols[0]:
    #     gcols0_cols = st.columns([1.5, 1, 0.5])
    with gcols[0]:
        st.subheader("GraphRAG")
    with gcols[1]:
        if st.session_state.graphrag_time_elapsed:
            st.info(f":clock3: {round(st.session_state.graphrag_time_elapsed, 2)}s")

    gcols = st.columns([1, 1, 1, 1.2])
    with gcols[0]:
        if st.button("Source", key="graphrag_source_documents"):
            show_graphrag_source_documents()

    with gcols[1]:
        if st.button("Graph", key="view_graphrag_graph", type="secondary"):
            show_graph_visuals()

    with gcols[2]:
        if st.button("Expand", key="graph_dialog"):
            show_graphrag_dialog()

    # GENERATE GRAPH RAG ANSWER
    if st.session_state.graphrag_loading:
        if len(query.strip()) == 0:
            st.warning("Please enter a query")
            st.stop()
        if st.session_state.graphrag_result is None:
            start_time = time.time()
            with st.spinner("Generating GraphRAG answer..."):
                result, context, total_latency = asyncio.run(generate(query))
                st.session_state.graphrag_result = result
                st.session_state.graphrag_context = context
                st.session_state.graphrag_loading = False
                end_time = time.time()
                st.session_state.graphrag_time_elapsed = end_time - start_time
                st.rerun()

    if st.session_state.graphrag_result:
        st.write(st.session_state.graphrag_result)


##########################################
# TRADITIONAL RAG SECTION
##########################################
with columns[1]:
    rcols = st.columns([4, 1])

    with rcols[0]:
        st.subheader("Traditional RAG")
    with rcols[1]:
        if st.session_state.rag_time_elapsed:
            st.info(f":clock3: {round(st.session_state.rag_time_elapsed, 2)}s")

    rcols = st.columns([1, 1, 2.5])
    with rcols[0]:
        if st.button("Source", key="rag_source_documents"):
            show_rag_source_documents()

    with rcols[1]:
        if st.button("Expand", key="rag_dialog"):
            show_rag_dialog()

    # GENERATE TRADITIONAL RAG ANSWER
    if st.button("Submit", key="rag_submit", type="primary"):
        start_time = time.time()
        if len(query.strip()) == 0:
            st.warning("Please enter a query")
            st.stop()
        with st.spinner("Generating RAG answer..."):
            if st.session_state.rag_result is None:
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
                ) = generate_rag_answer(query)

                st.session_state.rag_result = rag_answer
                st.session_state.rag_source_docs = source_docs
                end_time = time.time()
                st.session_state.rag_time_elapsed = end_time - start_time
                st.rerun()
    if st.session_state.rag_result:
        if len(st.session_state.rag_source_docs) > 0:
            st.write(st.session_state.rag_result)
        else:
            st.error(st.session_state.rag_result)
