import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v4", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
# )

# 具体来说：LangChain 默认会做长度检查/分词，可能把输入变成 token 列表发出去；
# 但 DashScope 这个接口期望的是原始文本字符串（或字符串列表），于是报
# 设置了 check_embedding_ctx_length=False，让它发送原始文本而不是 token 数组（接口兼容问题解决）
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model=os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-v4"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    dimensions=1024,
    check_embedding_ctx_length=False,
)
# 使用嵌入模型
# 对单个文本进行嵌入
text = "这是一个测试文本"
embedding = embeddings.embed_query(text)
print(f"嵌入向量：{embedding}")
print(f"单个文本嵌入维度：{len(embedding)}")
