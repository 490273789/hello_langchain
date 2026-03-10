from pymilvus import DataType, IndexType, MetricType, MilvusClient

from models import qwen_eb

COLLECTION_NAME = "ai_diary_2"
VECTOR_DIM = 1024

MILVUS_ADDRESS = "https://140.143.164.230:19530"


# 初始化Milvus客户端
client = MilvusClient(uri=MILVUS_ADDRESS)


def get_embedding(text: str) -> str:
    """获取单个文本的向量潜入"""
    # 返回一个列列表（vector）
    return qwen_eb.embed_query(text)


def main():
    try:
        print("Connecting to Milvus...")
        # MilvusClient 初始化时通常已尝试连接，但我们可以通过检查集合是否存在来验证
        # 如果只是想确认连接，可以调用 list_collections()
        client.list_collections()
        print("Connected")

        # 检查集合是否已存在，如果存在则删除以便重新运行演示（可选）
        if client.has_collection(collection_name=COLLECTION_NAME):
            print(
                f"Collection '{COLLECTION_NAME}' already exists. Dropping it for fresh start..."
            )
            client.drop_collection(COLLECTION_NAME)

        # 创建集合
        print("Creating collection...")

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, max_length=50, is_primary=True
        )
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM
        )
        schema.add_field(
            field_name="content", datatype=DataType.VARCHAR, max_length=5000
        )
        schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="mood", datatype=DataType.VARCHAR, max_length=50)
        # 添加 Array 字段 (tags)
        # element_type 对应 DataType.VARCHAR
        # max_capacity 对应数组最大元素个数
        # element_params 用于指定数组内元素的参数（如 VARCHAR 的 max_length）
        schema.add_field(
            field_name="tags",
            datatype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=10,
            element_params={"max_length": 50},
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=IndexType.IVF_FLAT,
            metric_type=MetricType.COSINE,
            params={"nlist": 1024},
        )

        # 执行创建集合操作
        client.create_collection(
            collection_name=COLLECTION_NAME, schema=schema, index_params=index_params
        )

        # 加载集合 (在 Milvus 2.x 中，创建后通常需要 load 才能搜索，但 insert 不需要 load)
        # 为了后续可能的搜索操作，这里显式加载
        print("\nLoading collection...")
        client.load_collection(collection_name=COLLECTION_NAME)
        print("Collection loaded")

        # 准备日记数据
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

        # 生成向量
        print("Generating embeddings...")
        diary_data = []
        for diary in diary_contents:
            vector = get_embedding(diary["content"])
            # 构造符合 Milvus 插入格式的数据字典
            entity = {
                "id": diary["id"],
                "vector": vector,
                "content": diary["content"],
                "date": diary["date"],
                "mood": diary["mood"],
                "tags": diary["tags"],
            }
            diary_data.append(entity)

        # 插入数据
        insert_result = client.insert(collection_name=COLLECTION_NAME, data=diary_data)

        print(f"✓ Inserted {insert_result['insert_count']} records\n")

    except Exception as err:
        print(err)
