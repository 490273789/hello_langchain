from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import qwen

""" 
索引化，构建知识库
"""

# 家在文档

loader = Docx2txtLoader("./datas/行业.docx")

docs = loader.load()

# 文档切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=30, separators=["\n\n\n\n", ""]
)
documents = text_splitter.split_documents(docs)
# print(f"documents: {documents}")
# print("-" * 88)

# 文档嵌入模型
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

# 向量数据库
db = Chroma.from_documents(
    collection_name="document1", documents=documents, embedding=embeddings
)

# 检索
# 相似度查询：k=2表示得到查询的前两个相似度最高的文档，默认k=4
# print(db.similarity_search("建筑行业的申标系统应用的业务场景有哪些？", k=2))
# print("-" * 88)
# 相似度查询带得分：按照相似度的分数排序，分数值越小越相似（L2距离）
# print(db.similarity_search_with_score("建筑行业审标系统应用的业务场景有哪些？"))
# print("-" * 88)

# 检索：返回相似度最高的三个
docs_find = RunnableLambda(db.similarity_search).bind(k=2)
# print(f"docs_find: {docs_find}")
# print("-" * 88)

print(docs_find.invoke("建筑行业的审标系统应用的业务场景有哪几条？"))
print("-" * 88)

message = """
仅使用提供的上下文回答下面的问题：
{question}
上下文：
{context}
"""

prompt_template = ChatPromptTemplate([("human", message)])

# 定义这个链的时候，还不知道问题是什么
# 用RunnablePassthrough允许我们将用户的具体问题在实际的使用过程中进行动态的传入
chain = (
    {"question": RunnablePassthrough(), "context": docs_find} | prompt_template | qwen
)

# 用大模型生成答案
response = chain.invoke("建筑行业的审标系统应用的业务场景有哪几条？")
print(response.content)
