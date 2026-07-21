from pathlib import Path
from pprint import pprint
import asyncio
import graphrag.api as api
import pandas as pd
from graphrag.config.load_config import load_config
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
import time

PROJECT_DIRECTORY = "."

graphrag_config = load_config(Path(PROJECT_DIRECTORY))

entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
community_reports = pd.read_parquet(
    f"{PROJECT_DIRECTORY}/output/community_reports.parquet"
)
relationships = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/relationships.parquet")
text_units = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/text_units.parquet")

# query = "A truck driver is injured during an accident.Which policy documents are relevant and why?"


async def generate(query: str):
    start = time.perf_counter()

    try:
        result, context = await api.local_search(
            config=graphrag_config,
            entities=entities,
            relationships=relationships,
            text_units=text_units,
            communities=communities,
            community_reports=community_reports,
            community_level=2,
            covariates=None,
            response_type="Multiple Paragraphs",
            query=query,
        )

    total_latency = (time.perf_counter() - start) * 1000

    # print("\nCONTEXT:")
    # print(context)

    # print(type(result))
    return result, context, total_latency


# async def main():

#     search_engine = await load_search_engine(root_dir="./output", method="local")

#     result = await search_engine.asearch("What is covered under health insurance?")

#     print("\nANSWER:")
#     print(result.response)

#     print("\nCONTEXT:")
#     print(result.context_data)


# asyncio.run(main())
