from .chunk_documents import get_chunks
from .weaviate_client import get_weaviate_client

client = get_weaviate_client()

chunks = get_chunks()

collection = client.collections.use("ICICI_Policy_documents")

with collection.batch.fixed_size(batch_size=5) as batch:
    for chunk in chunks:
        batch.add_object(properties=chunk)

        if batch.number_errors > 10:
            print("Batch import stopped due to excessive errors.")
            break
client.close()
failed_objects = collection.batch.failed_objects
if failed_objects:
    print(f"Number of failed imports: {len(failed_objects)}")
    print(f"First failed object: {failed_objects[0]}")
