import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

COLLECTION_NAME = "ai_diary_async"
VECTOR_DIM = 1024
EMBEDDING_CONCURRENCY = 5


def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        dimensions=VECTOR_DIM,
    )


def create_collection_if_needed() -> Collection:
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists, reusing it")
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            max_length=50,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="mood", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(
            name="tags",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=10,
            max_length=50,
        ),
    ]

    schema = CollectionSchema(fields=fields, description="AI diary entries")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print("Collection created")
    return collection


def create_index_if_needed(collection: Collection) -> None:
    if collection.has_index():
        print("Index already exists, skipping")
        return

    index_params: dict[str, Any] = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1024},
    }
    collection.create_index(field_name="vector", index_params=index_params)
    print("Index created")


async def build_diary_data(
    embeddings: OpenAIEmbeddings, diary_contents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

    async def embed_one(diary: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            vector = await embeddings.aembed_query(diary["content"])
            return {**diary, "vector": vector}

    return await asyncio.gather(*(embed_one(diary) for diary in diary_contents))


async def async_main() -> None:
    load_dotenv()

    try:
        print("Connecting to Milvus...")
        connections.connect(alias="default", host="localhost", port="19530")
        print("✓ Connected\n")

        print("Creating collection...")
        collection = create_collection_if_needed()

        print("\nCreating index...")
        create_index_if_needed(collection)

        print("\nLoading collection...")
        collection.load()
        print("Collection loaded")

        print("\nInserting diary entries...")
        diary_contents = [
            {
                "id": "diary_001",
                "content": "今天天气很好，去公园散步了，心情愉快。看到了很多花开了，春天真美好。",
                "date": "2026-01-10",
                "mood": "happy",
                "tags": ["生活", "散步"],
            },
            {
                "id": "diary_002",
                "content": "今天工作很忙，完成了一个重要的项目里程碑。团队合作很愉快，感觉很有成就感。",
                "date": "2026-01-11",
                "mood": "excited",
                "tags": ["工作", "成就"],
            },
            {
                "id": "diary_003",
                "content": "周末和朋友去爬山，天气很好，心情也很放松。享受大自然的感觉真好。",
                "date": "2026-01-12",
                "mood": "relaxed",
                "tags": ["户外", "朋友"],
            },
            {
                "id": "diary_004",
                "content": "今天学习了 Milvus 向量数据库，感觉很有意思。向量搜索技术真的很强大。",
                "date": "2026-01-12",
                "mood": "curious",
                "tags": ["学习", "技术"],
            },
            {
                "id": "diary_005",
                "content": "晚上做了一顿丰盛的晚餐，尝试了新菜谱。家人都说很好吃，很有成就感。",
                "date": "2026-01-13",
                "mood": "proud",
                "tags": ["美食", "家庭"],
            },
        ]

        embeddings = get_embedding_model()
        print("Generating embeddings concurrently...")
        diary_data = await build_diary_data(embeddings, diary_contents)

        insert_result = collection.insert(diary_data)
        collection.flush()

        inserted_count = getattr(insert_result, "insert_count", len(diary_data))
        print(f"✓ Inserted {inserted_count} records\n")

    except MilvusException as error:
        print(f"Milvus error: {error}")
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}")
    finally:
        connections.disconnect(alias="default")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
