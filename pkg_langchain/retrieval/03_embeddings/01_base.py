import os

from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)
# 使用嵌入模型
# 对单个文本进行嵌入
text = "这是一个测试文本"
embedding = embeddings.embed_query(text)
print(f"嵌入向量：{embedding}")
print(f"单个文本嵌入维度：{len(embedding)}")
