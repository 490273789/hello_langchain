# 创建Milvus客户端
import os

import dotenv
from pymilvus import MilvusClient

dotenv.load_dotenv()

client = MilvusClient(
    uri=os.getenv("MILVUS_ADDRESS"),
    user=os.getenv("MILVUS_USER"),
    password=os.getenv("MILVUS_PWD"),
)


def main():
    try:
        print("=" * 66)
        print("e-book handling!")
        print("=" * 66)
        book_id = 1

    except Exception as err:
        print(err)
