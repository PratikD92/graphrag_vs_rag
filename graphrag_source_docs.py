import pandas as pd

PROJECT_DIRECTORY = "."

# graphrag_config = load_config(Path(PROJECT_DIRECTORY))

text_units = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/text_units.parquet")


def get_graphrag_source_docs(context):
    retrieved_docs = set()

    for _, source_row in context["sources"].iterrows():

        source_text = source_row["text"]

        matches = text_units[text_units["text"] == source_text]

        if not matches.empty:

            retrieved_docs.update(matches["document_id"].tolist())

    print(retrieved_docs)
    print(len(retrieved_docs))
    documents = pd.read_parquet("output/documents.parquet")

    # print(documents.head())
    # documents[
    #     documents["id"].isin(retrieved_docs)
    # ][["id", "title"]]
    return documents[documents["id"].isin(retrieved_docs)][["id", "title"]]
