from weaviate.classes.query import MetadataQuery
from .weaviate_client import get_weaviate_client
from .prompt_gen import generate_prompt
from dotenv import load_dotenv
from openai import OpenAI
import os, time
from .config import RAG_LLM_MODEL, COLLECTION_NAME

# from config
load_dotenv("../.env")

TOP_K = 5
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# print(os.getenv("WEAVIATE_API_KEY"))


def retrieve_chunks(query_text: str):
    client = get_weaviate_client()

    try:
        collection = client.collections.use(COLLECTION_NAME)

        raw = collection.query.hybrid(
            query=query_text,
            alpha=0.75,
            return_metadata=MetadataQuery(score=True),
            return_properties=["text_content", "source"],
            limit=TOP_K,
        )

        results = []

        for obj in raw.objects:
            results.append(
                {
                    "text_content": obj.properties["text_content"],
                    "source": obj.properties["source"],
                    "score": obj.metadata.score,
                }
            )

        return results

    except Exception as exc:
        print(f"Weaviate query failed: {exc}")
        return []
        # raise Exception(f"Weaviate query failed: {exc}")

    # finally:
    #     client.close()


# query = """A truck driver is injured during an accident.
#     Which policy documents are relevant and why?"""


def generate_rag_answer(query, llm_model):

    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()

    chunks = retrieve_chunks(query)

    retrieval_latency = (time.perf_counter() - retrieval_start) * 1000

    if len(chunks) == 0:
        return "Weaviate query failed. Please refresh the page and try again.", []

    context = "\n\n".join(f"Source: {c['source']}\n{c['text_content']}" for c in chunks)

    prompt = generate_prompt(query, context)

    llm_start = time.perf_counter()

    model = RAG_LLM_MODEL  # fall back to .env default if not overridden
    if llm_model != RAG_LLM_MODEL or llm_model is not None:
        model = llm_model
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    llm_latency = (time.perf_counter() - llm_start) * 1000

    total_latency = (time.perf_counter() - total_start) * 1000

    usage = response.usage
    total_tokens = usage.total_tokens
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    # print(response.choices[0].message.content)
    return (
        response.choices[0].message.content,
        chunks,
        context,
        retrieval_latency,
        llm_latency,
        total_latency,
        total_tokens,
        prompt_tokens,
        completion_tokens,
        model,
    )
