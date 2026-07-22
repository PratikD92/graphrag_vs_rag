from pyvis.network import Network
import pandas as pd
import streamlit as st

PROJECT_DIRECTORY = "."
entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
relationships = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/relationships.parquet")


def generate_graph(context):
    # --------------------------
    # Retrieved nodes/edges
    # --------------------------

    print(f"CONTEXT = {context}")

    retrieved_entities = set(context["entities"]["entity"].astype(str))

    retrieved_edges = set()

    for _, row in context["relationships"].iterrows():
        retrieved_edges.add((str(row["source"]), str(row["target"])))

    # --------------------------
    # Entity descriptions
    # --------------------------

    entity_desc = {}

    for _, row in entities.iterrows():

        entity_name = str(row.get("title", row.get("entity", "")))

        entity_desc[entity_name] = str(row.get("description", ""))

    # --------------------------
    # Create graph
    # --------------------------

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="black",
    )

    # --------------------------
    # Collect all nodes
    # --------------------------

    all_nodes = set()

    for _, row in relationships.iterrows():

        source = str(row["source"])
        target = str(row["target"])

        all_nodes.add(source)
        all_nodes.add(target)

    # --------------------------
    # Add nodes
    # Red = retrieved
    # Grey = not retrieved
    # --------------------------

    for node in all_nodes:

        if node in retrieved_entities:

            net.add_node(
                node,
                label=node,
                title=entity_desc.get(node, ""),
                color="#ff4444",
                size=20,
            )

        else:

            net.add_node(
                node,
                label=node,
                title=entity_desc.get(node, ""),
                color="#d3d3d3",
                size=20,
            )

    # --------------------------
    # Add edges
    # Red = retrieved
    # Grey = not retrieved
    # --------------------------

    for _, row in relationships.iterrows():

        source = str(row["source"])
        target = str(row["target"])

        edge = (source, target)

        if edge in retrieved_edges:

            net.add_edge(
                source,
                target,
                color="red",
                width=2,
                title=str(row.get("description", "")),
            )

        else:

            net.add_edge(
                source,
                target,
                color="#dddddd",
                width=1,
                title=str(row.get("description", "")),
            )

    return net.generate_html()


@st.cache_resource()
def generate_full_graph():

    # --------------------------
    # Entity descriptions
    # --------------------------

    entity_desc = {}

    for _, row in entities.iterrows():

        entity_name = str(row.get("title", row.get("entity", "")))

        entity_desc[entity_name] = str(row.get("description", ""))

    # --------------------------
    # Create graph
    # --------------------------

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="black",
    )

    # --------------------------
    # Collect all nodes
    # --------------------------

    all_nodes = set()

    for _, row in relationships.iterrows():

        source = str(row["source"])
        target = str(row["target"])

        all_nodes.add(source)
        all_nodes.add(target)

    for node in all_nodes:

        net.add_node(
            node,
            label=node,
            title=entity_desc.get(node, ""),
            color="#d3d3d3",
            size=20,
        )

    for _, row in relationships.iterrows():

        source = str(row["source"])
        target = str(row["target"])

        net.add_edge(
            source,
            target,
            color="#dddddd",
            width=1,
            title=str(row.get("description", "")),
        )

    return net.generate_html()
