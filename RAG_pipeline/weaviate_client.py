import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from functools import lru_cache
from typing import Any
from fastapi import HTTPException

from .config import WEAVIATE_API_KEY, WEAVIATE_URL, OPENAI_API_KEY


@lru_cache(maxsize=1)
def get_weaviate_client() -> Any:

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={"X-Openai-Api-Key": OPENAI_API_KEY},
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)  # seconds
        ),
    )

    if client.is_ready():
        # print("Weaviate client is ready.")
        return client
    else:
        raise HTTPException(
            status_code=500,
            detail="Weaviate client is not ready. Please check your Weaviate Cloud Services configuration.",
        )
