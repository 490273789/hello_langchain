import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymilvus import (
    DataType,
    MilvusClient,
    MilvusException,
)

from models import qwen
from tools.format_print import pretty_print

from .data import diary_contents

COLLECTION_NAME = "ai_diary"
VECTOR_DIM = 1024


@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        dimensions=VECTOR_DIM,
        check_embedding_ctx_length=False,
    )


# 创建Milvus客户端
client = MilvusClient(
    uri=os.getenv("MILVUS_ADDRESS"),
    user=os.getenv("MILVUS_USER"),
    password=os.getenv("MILVUS_PWD"),
)


def get_embedding(text: str) -> list[float]:
    """根据文本获取向量"""
    return get_embedding_model().embed_query(text)


def validate_vector(vector: list[float], *, field_name: str) -> None:
    if not isinstance(vector, list):
        raise TypeError(
            f"{field_name} must be a list[float], got {type(vector).__name__}"
        )

    if len(vector) != VECTOR_DIM:
        raise ValueError(
            f"{field_name} dimension mismatch: expected {VECTOR_DIM}, got {len(vector)}"
        )


def create_collection_if_needed() -> None:
    """创建collection"""
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


def add_data():
    """插入数据"""

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
        vector = get_embedding(diary["content"])
        print("vector")
        diary_data.append({**diary, "vector": vector})

    insert_result = client.insert(collection_name=COLLECTION_NAME, data=diary_data)
    client.flush(collection_name=COLLECTION_NAME)

    inserted_count = insert_result.get("insert_count", len(diary_data))
    print(f"✓ Inserted {inserted_count} records\n")


def update_data():
    # 更新数据（Milvus 通过 upsert 实现更新）
    update_id = "diary_001"
    update_content = {
        "id": update_id,
        "content": "今天下了一整天的雨，心情很糟糕。工作上遇到了很多困难，感觉压力很大。一个人在家，感觉特别孤独。",
        "date": "2026-01-10",
        "mood": "sad",
        "tags": ["生活", "散步", "朋友"],
    }

    vector = get_embedding(update_content.get("content"))

    update_content["vector"] = vector

    result = client.upsert(collection_name=COLLECTION_NAME, data=update_content)

    print(f"✓ Updated diary entry: {update_id}")
    print(result)
    # print(f"日期：{item.date}")
    # print(f"心情：{item.mood}")
    # print(f"标签：{item.tags}")
    # print(f"内容：{item.content}")


def delete_data():
    """删除一条向量数据"""
    delete_id = "diary_005"
    # 删除单条数据
    result = client.delete(
        collection_name=COLLECTION_NAME, filter=f"id == '{delete_id}'"
    )

    print(f"ID: {delete_id}: {result}\n`")

    # 批量删除
    delete_ids = ["diary_002", "diary_003"]
    ids_str = ",".join(map(lambda x: f"'{x}'", delete_ids))
    print(ids_str)
    batch_result = client.delete(
        collection_name=COLLECTION_NAME, filter=f"id in [{ids_str}]"
    )

    print(f"ID: {ids_str}: {batch_result}\n`")

    # 条件删除
    condition_result = client.delete(
        collection_name=COLLECTION_NAME, filter="mood == 'sad'"
    )

    print(f"mood == 'sad': {ids_str}: {condition_result}\n`")


def search_data(query: str, limit: int = 2) -> list:
    """根据输入转化为向量查询"""
    result = []
    try:
        query_vector = get_embedding(query)
        validate_vector(query_vector, field_name="query_vector")

        search_result = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=limit,
            search_params={"metric_type": "COSINE"},
            output_fields=["id", "content", "date", "mood", "tags"],
        )
        print(f"Found {len(search_result)} results:\n")
        print(search_result)

        for item in search_result[0]:
            result.append(item.entity)
    except Exception as err:
        print(err)
    finally:
        return result


def search_rag(question: str, limit: int = 2):
    """查询最终答案"""
    print("=" * 50)
    print(f"query: {question}\n")

    # 在向量数据库检索相关日记
    retrieval_result = search_data(question, limit)
    if len(retrieval_result) == 0:
        print("未找到相关日记")
        return "抱歉，未找到相关日记"

    # 打印日记
    for item in retrieval_result:
        print(f"日期：{item.date}")
        print(f"心情：{item.mood}")
        print(f"标签：{item.tags}")
        print(f"内容：{item.content}")

    # 构建提示词上下文
    context_list = []
    for i, item in enumerate(retrieval_result):
        context_list.append(f"""
                       日记{i}
                       日期：{item.date}
                       心情：{item.mood}
                       标签：{item.tags}
                       内容：{item.content}
                       """)

    context = "\n\n-----\n\n".join(context_list)

    prompt = f"""你是一个温暖贴心的 AI 日记助手。基于用户的日记内容回答问题，用亲切自然的语言。

    请根据以下日记内容回答问题：
    {context}

    用户问题:${question}

    回答要求：
    1. 如果日记中有相关信息，请结合日记内容给出详细、温暖的回答
    2. 可以总结多篇日记的内容，找出共同点或趋势
    3. 如果日记中没有相关信息，请温和地告知用户
    4. 用第一人称"你"来称呼日记的作者
    5. 回答要有同理心，让用户感到被理解和关心

    AI 助手的回答:"""

    result = qwen.invoke(prompt)

    pretty_print(result, view="content")


def main() -> None:
    load_dotenv()

    try:
        print("Connecting to Milvus...")
        collection_list = client.list_collections()
        print(f"list: {collection_list}")
        print("✓ Connected\n")

        print("Creating collection if needed...")
        create_collection_if_needed()

        # print("\nLoading collection...")
        client.load_collection(collection_name=COLLECTION_NAME)
        # print("Collection loaded")

        # add_data()

        # search_data("我想看看关于户外活动的日记")

        # search_rag("我最近做了什么让我感到快乐的事情？")

        # update_data()

        # delete_data()

    except MilvusException as error:
        print(f"Milvus error: {error}")
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}")
    finally:
        client.release_collection(collection_name=COLLECTION_NAME)


if __name__ == "__main__":
    main()
