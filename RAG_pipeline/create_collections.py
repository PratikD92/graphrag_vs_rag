"""
Creates weaviate collections as per the embedding model, chunk size and overlap chosen.
"""

from .weaviate_client import get_weaviate_client
from .config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from weaviate.classes.config import Configure, Property, DataType
from .upload_batch_weaviate import upload_batch


print(
    f"Creating collection with embedding model: {EMBEDDING_MODEL}\n chunk size: {CHUNK_SIZE}\n chunk overlap: {CHUNK_OVERLAP}"
)


def create_collection(EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP):
    client = get_weaviate_client()

    collection_name = (
        f"ICICI_Policy_documents_{EMBEDDING_MODEL}_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
    ).replace("-", "_")

    if client.collections.exists(collection_name):
        print(f"Collection {collection_name} already exists")
        client.close()
        return 2, collection_name

    try:
        # Create collection
        client.collections.create(
            name=collection_name,
            properties=[
                Property(name="text_content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
            ],
            vector_config=Configure.Vectors.text2vec_openai(
                model=EMBEDDING_MODEL, source_properties=["text_content"]
            ),
        )
        print("Collection created successfully")
    except Exception as e:
        print(f"Error creating collection: {e}")
        client.close()
        return 0, None
    client.close()
    return 1, collection_name


# Create collection
created, collection_name = create_collection(EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)
if created == 0:
    print("Error creating collection, please see logs for details")
    exit(1)
elif created == 2:
    print("Collection already exists")
    upload_batch(collection_name)
else:
    upload_batch(collection_name)
