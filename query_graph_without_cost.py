from pathlib import Path
from pprint import pprint
import asyncio
import graphrag.api as api
import pandas as pd
from graphrag.config.load_config import load_config
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
import time
import contextvars
import litellm
from litellm.integrations.custom_logger import CustomLogger

PROJECT_DIRECTORY = "."


_cost_accumulator = contextvars.ContextVar("cost_accumulator", default=None)

_orig_acompletion = litellm.acompletion
_orig_completion = litellm.completion


def _capture_cost(response, kwargs):
    acc = _cost_accumulator.get()
    if acc is None:
        return
    try:
        cost = litellm.completion_cost(
            completion_response=response, model=kwargs.get("model")
        )
    except Exception as e:
        print("cost calc failed:", e, "model:", kwargs.get("model"))
        cost = 0
    acc.append(cost)


async def _patched_acompletion(*args, **kwargs):
    response = await _orig_acompletion(*args, **kwargs)
    _capture_cost(response, kwargs)
    return response


def _patched_completion(*args, **kwargs):
    print("PATCHED ACOMPLETION CALLED, model:", kwargs.get("model"))
    response = _orig_completion(*args, **kwargs)
    print("USAGE:", getattr(response, "usage", None))
    _capture_cost(response, kwargs)
    return response


litellm.acompletion = _patched_acompletion
litellm.completion = _patched_completion

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
    acc = []
    token = _cost_accumulator.set(acc)

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
    finally:
        _cost_accumulator.reset(token)

    total_latency = (time.perf_counter() - start) * 1000
    query_cost = sum(acc)

    # print("\nCONTEXT:")
    # print(context)

    # print(type(result))
    return result, context, total_latency, query_cost


# async def main():

#     search_engine = await load_search_engine(root_dir="./output", method="local")

#     result = await search_engine.asearch("What is covered under health insurance?")

#     print("\nANSWER:")
#     print(result.response)

#     print("\nCONTEXT:")
#     print(result.context_data)


# asyncio.run(main())
