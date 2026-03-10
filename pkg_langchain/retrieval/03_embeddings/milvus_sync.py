import os

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import OpenAIEmbeddings
from pymilvus import (
    DataType,
    MilvusClient,
    MilvusException,
)

COLLECTION_NAME = "ai_diary"
VECTOR_DIM = 1024
MILVUS_ADDRESS = "http://140.143.164.230:19530"


def get_embedding_model() -> DashScopeEmbeddings:
    # return DashScopeEmbeddings(
    #     model=os.getenv("EMBEDDINGS_MODEL_NAME"),
    #     dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    # )
    return OpenAIEmbeddings(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        dimensions=VECTOR_DIM,
    )


def create_collection_if_needed() -> None:
    if client.has_collection(collection_name=COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists, reusing it")
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="id", datatype=DataType.VARCHAR, max_length=50, is_primary=True
    )
    schema.add_field(
        field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM
    )
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=5000)
    schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="mood", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(
        field_name="tags",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=10,
        max_length=50,
    )
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 1024},
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print("Collection created")


client = MilvusClient(
    uri=MILVUS_ADDRESS,
    token="root:Milvus",
)


def main() -> None:
    load_dotenv()

    try:
        print("Connecting to Milvus...")
        collection_list = client.list_collections()
        print(f"list: {collection_list}")
        print("✓ Connected\n")

        print("Creating collection...")
        create_collection_if_needed()

        print("\nLoading collection...")
        client.load_collection(collection_name=COLLECTION_NAME)
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
        print("Generating embeddings...")

        # contents = [str(diary["content"]) for diary in diary_contents]
        # vectors = embeddings.embed_documents(contents)
        # if len(vectors) != len(diary_contents):
        #     raise ValueError("Embedding count does not match diary entry count")

        # diary_data = [
        #     {**diary, "vector": vector}
        #     for diary, vector in zip(diary_contents, vectors, strict=False)
        # ]

        diary_data = []
        for diary in diary_contents:
            print(type(diary["content"]))
            vector = embeddings.embed_query(diary["content"])
            print("vector")
            diary_data.append({**diary, "vector": vector})

        insert_result = client.insert(collection_name=COLLECTION_NAME, data=diary_data)
        client.flush(collection_name=COLLECTION_NAME)

        inserted_count = insert_result.get("insert_count", len(diary_data))
        print(f"✓ Inserted {inserted_count} records\n")

    except MilvusException as error:
        print(f"Milvus error: {error}")
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
