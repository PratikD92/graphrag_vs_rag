from pathlib import Path
import graphrag.api as api
import pandas as pd
from graphrag.config.embeddings import entity_description_embedding
from graphrag.config.load_config import load_config
from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.utils.api import get_embedding_store, load_search_prompt
from graphrag_llm.model_cost_registry import model_cost_registry
import time
from typing import Any

PROJECT_DIRECTORY = "."
COMMUNITY_LEVEL = 2
RESPONSE_TYPE = "Multiple Paragraphs"

graphrag_config = load_config(Path(PROJECT_DIRECTORY))
print("+" * 50)
print(graphrag_config)
print("+" * 50)

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

    result, context = await api.local_search(
        config=graphrag_config,
        entities=entities,
        relationships=relationships,
        text_units=text_units,
        communities=communities,
        community_reports=community_reports,
        community_level=COMMUNITY_LEVEL,
        covariates=None,
        response_type=RESPONSE_TYPE,
        query=query,
    )

    total_latency = (time.perf_counter() - start) * 1000

    # print("\nCONTEXT:")
    # print(context)

    # print(type(result))
    return result, context, total_latency


def _get_costs(provider: str, model: str) -> tuple[str, dict[str, float] | None]:
    provider_model_id = f"{provider}/{model}"
    costs = model_cost_registry.get_model_costs(provider_model_id)
    if costs:
        print(f"Costs provider_model {provider_model_id}: {costs}")
        return provider_model_id, costs

    costs = model_cost_registry.get_model_costs(model)
    if costs:
        print(f"Costs for only model {model}: {costs}")
        return model, costs

    return provider_model_id, None


def _model_costs(model_config_id: str) -> tuple[str, dict[str, float] | None]:
    model_config = graphrag_config.get_completion_model_config(model_config_id)
    return _get_costs(model_config.model_provider, model_config.model)


def _embedding_model_costs(model_config_id: str) -> tuple[str, dict[str, float] | None]:
    model_config = graphrag_config.get_embedding_model_config(model_config_id)
    return _get_costs(model_config.model_provider, model_config.model)


def _completion_cost(
    prompt_tokens: int, output_tokens: int
) -> tuple[float, str | None]:
    model_id, costs = _model_costs(graphrag_config.local_search.completion_model_id)
    if not costs:
        return 0.0, f"No cost data found for completion model {model_id}"
    print(f"_completion_cost: {costs}")
    return (
        prompt_tokens * costs["input_cost_per_token"]
        + output_tokens * costs["output_cost_per_token"],
        None,
    )


def _embedding_cost_from_metrics(
    embedding_metrics: dict[str, Any],
    query: str,
    tokenizer: Any,
) -> tuple[float, int, str | None]:
    metric_cost = float(embedding_metrics.get("total_cost", 0) or 0)
    metric_tokens = int(embedding_metrics.get("total_tokens", 0) or 0)
    if metric_cost:
        return metric_cost, metric_tokens, None

    model_id, costs = _embedding_model_costs(
        graphrag_config.local_search.embedding_model_id
    )
    if not costs:
        return 0.0, 0, f"No cost data found for embedding model {model_id}"

    query_tokens = metric_tokens or len(tokenizer.encode(query))
    return query_tokens * costs["input_cost_per_token"], query_tokens, None


def _clear_metrics_store(model: Any) -> None:
    clear_metrics = getattr(
        getattr(model, "metrics_store", None), "clear_metrics", None
    )
    if callable(clear_metrics):
        clear_metrics()


def _get_local_search_engine():
    description_embedding_store = get_embedding_store(
        config=graphrag_config.vector_store,
        embedding_name=entity_description_embedding,
    )
    prompt = load_search_prompt(graphrag_config.local_search.prompt)

    return get_local_search_engine(
        config=graphrag_config,
        reports=read_indexer_reports(
            community_reports,
            communities,
            COMMUNITY_LEVEL,
        ),
        text_units=read_indexer_text_units(text_units),
        entities=read_indexer_entities(entities, communities, COMMUNITY_LEVEL),
        relationships=read_indexer_relationships(relationships),
        covariates={"claims": []},
        description_embedding_store=description_embedding_store,
        response_type=RESPONSE_TYPE,
        system_prompt=prompt,
    )


async def generate_with_cost(query: str):
    """Run local search and return GraphRAG token/cost metadata.

    GraphRAG's public api.local_search streams the final chat completion and
    returns only response/context. This mirrors the same local-search setup but
    calls LocalSearch.search(), which exposes token counts. Completion cost is
    then calculated with GraphRAG/LiteLLM's model_cost_registry.
    """
    start = time.perf_counter()
    search_engine = _get_local_search_engine()
    _clear_metrics_store(search_engine.model)
    _clear_metrics_store(search_engine.context_builder.text_embedder)

    search_result = await search_engine.search(query=query)
    total_latency = (time.perf_counter() - start) * 1000

    prompt_tokens = search_result.prompt_tokens + len(
        search_engine.tokenizer.encode(query)
    )
    output_tokens = search_result.output_tokens
    completion_cost, completion_warning = _completion_cost(prompt_tokens, output_tokens)

    embedding_metrics = (
        search_engine.context_builder.text_embedder.metrics_store.get_metrics()
    )
    embedding_cost, embedding_tokens, embedding_warning = _embedding_cost_from_metrics(
        embedding_metrics,
        query,
        search_engine.context_builder.text_embedder.tokenizer,
    )

    cost = {
        "total_cost": completion_cost + embedding_cost,
        "completion_cost": completion_cost,
        "embedding_cost": embedding_cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "embedding_tokens": embedding_tokens,
        "total_tokens": prompt_tokens + output_tokens + embedding_tokens,
        "llm_calls": search_result.llm_calls,
        "completion_model": _model_costs(
            graphrag_config.local_search.completion_model_id
        )[0],
        "embedding_model": _embedding_model_costs(
            graphrag_config.local_search.embedding_model_id
        )[0],
        "embedding_metrics": embedding_metrics,
        "warnings": [
            warning
            for warning in (completion_warning, embedding_warning)
            if warning is not None
        ],
    }

    return search_result.response, search_result.context_data, total_latency, cost


# async def main():

#     search_engine = await load_search_engine(root_dir="./output", method="local")

#     result = await search_engine.asearch("What is covered under health insurance?")

#     print("\nANSWER:")
#     print(result.response)

#     print("\nCONTEXT:")
#     print(result.context_data)


# asyncio.run(main())
