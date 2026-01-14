
from langchain_chroma import Chroma

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

# 嵌入模型
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

score_measure = [
    "default", # l2
    "cosine", # 余弦相似度 1-cos(角度) 0-2
    "l2", # 欧氏距离
    "ip" # 点积 和cosine
]

# 创建db
db = Chroma(
    collection_name="collection_name",
    embedding_function=embeddings,
    persist_directory="./chroma_db1",
    collection_metadata={"hnsw:space": "l2"}
)

# 提供文档
documents = [
    Document(page_content = "这个苹果手机很好用"),
    Document(page_content = "我国大连盛产苹果"),
]


# 添加文档
ids = db.add_documents(documents)
print(ids)
print("-"*15)

# 检索
result = db.similarity_search_with_score("我想买个手机")
for doc, score in result:
    print(doc.page_content, end='\t')
    print(f"score: {score}")