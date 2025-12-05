# LangChain Retrieval 模块

## 概述

Retrieval（检索）模块是 LangChain 中用于从外部数据源获取信息的核心组件。它是构建 RAG（Retrieval-Augmented Generation，检索增强生成）应用的基础。

RAG 的核心思想是：在调用 LLM 之前，先从外部知识库中检索相关信息，然后将这些信息作为上下文提供给 LLM，从而生成更准确、更具针对性的回答。

## 核心组件

Retrieval 模块主要包含以下几个核心组件：

```
┌─────────────────────────────────────────────────────────────┐
│                    Retrieval 流程                            │
├─────────────────────────────────────────────────────────────┤
│  1. Document Loaders (文档加载器)                            │
│         ↓                                                    │
│  2. Text Splitters (文本分割器)                              │
│         ↓                                                    │
│  3. Embeddings (向量嵌入)                                    │
│         ↓                                                    │
│  4. Vector Stores (向量存储)                                 │
│         ↓                                                    │
│  5. Retrievers (检索器)                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Document Loaders（文档加载器）

Document Loaders 用于从各种数据源加载文档。LangChain 提供了丰富的加载器支持。

### 1.1 常见的文档加载器

| 加载器 | 用途 | 包 |
|--------|------|-----|
| `PyPDFLoader` | 加载 PDF 文件 | `langchain_community` |
| `CSVLoader` | 加载 CSV 文件 | `langchain_community` |
| `WebBaseLoader` | 加载网页内容 | `langchain_community` |
| `TextLoader` | 加载纯文本文件 | `langchain_community` |
| `UnstructuredPowerPointLoader` | 加载 PPT 文件 | `langchain_community` |
| `S3FileLoader` | 加载 S3 文件 | `langchain_community` |

### 1.2 代码示例

#### 加载 PDF 文件

```python
from langchain_community.document_loaders import PyPDFLoader

file_path = "example.pdf"
loader = PyPDFLoader(file_path)

# 加载所有页面
docs = loader.load()

print(f"加载了 {len(docs)} 个文档")
print(f"第一页内容预览: {docs[0].page_content[:200]}")
print(f"元数据: {docs[0].metadata}")
```

#### 加载 CSV 文件

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path="data.csv")

# 加载所有文档
documents = loader.load()

# 对于大数据集，使用懒加载
for document in loader.lazy_load():
    print(document)
```

#### 加载网页内容

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://example.com/article",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

print(f"总字符数: {len(docs[0].page_content)}")
```

### 1.3 Document 对象结构

```python
from langchain_core.documents import Document

# Document 对象包含两个主要属性
document = Document(
    page_content="文档的文本内容...",  # 文档内容
    metadata={"source": "example.pdf", "page": 1}  # 元数据
)
```

---

## 2. Text Splitters（文本分割器）

由于 LLM 有 token 限制，且检索时需要找到最相关的片段，因此需要将长文档分割成小块。

### 2.1 常见的文本分割器

| 分割器 | 特点 | 适用场景 |
|--------|------|----------|
| `RecursiveCharacterTextSplitter` | 递归分割，保持语义完整性 | 通用文本（推荐默认使用） |
| `CharacterTextSplitter` | 按字符数分割 | 简单文本 |
| `TokenTextSplitter` | 按 token 数分割 | 需要精确控制 token 数量 |

### 2.2 代码示例

#### RecursiveCharacterTextSplitter（推荐）

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # 每个块的最大字符数
    chunk_overlap=200,      # 块之间的重叠字符数
    add_start_index=True    # 记录每个块在原文档中的起始位置
)

# 分割文档列表
all_splits = text_splitter.split_documents(docs)

print(f"原文档被分割成 {len(all_splits)} 个子文档")
```

#### CharacterTextSplitter（基于 Token）

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",  # OpenAI 使用的编码
    chunk_size=100,
    chunk_overlap=0
)
texts = text_splitter.split_text(document)
```

### 2.3 分割参数说明

- **chunk_size**: 每个文本块的目标大小（字符数或 token 数）
- **chunk_overlap**: 相邻块之间的重叠部分，确保上下文连续性
- **separators**: 分割时使用的分隔符优先级列表

```python
# RecursiveCharacterTextSplitter 的默认分隔符顺序
# ["\n\n", "\n", " ", ""]
# 优先按段落分割 → 按行分割 → 按空格分割 → 按字符分割
```

---

## 3. Embeddings（向量嵌入）

Embeddings 将文本转换为数值向量，使得语义相似的文本在向量空间中距离更近。

### 3.1 常见的 Embedding 模型

| 模型 | 提供商 | 包 |
|------|--------|-----|
| `OpenAIEmbeddings` | OpenAI | `langchain_openai` |
| `AzureOpenAIEmbeddings` | Azure | `langchain_openai` |
| `HuggingFaceEmbeddings` | HuggingFace | `langchain_huggingface` |
| `OllamaEmbeddings` | Ollama（本地） | `langchain_community` |

### 3.2 代码示例

#### OpenAI Embeddings

```python
import os
from langchain_openai import OpenAIEmbeddings

# 设置 API Key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 对单个文本生成向量
vector = embeddings.embed_query("Hello, world!")
print(f"向量维度: {len(vector)}")
print(f"向量前10个元素: {vector[:10]}")

# 对多个文本批量生成向量
vectors = embeddings.embed_documents([
    "第一个文档",
    "第二个文档"
])
```

#### Azure OpenAI Embeddings

```python
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
)
```

### 3.3 向量相似度对比

```python
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

# 确保向量维度一致
assert len(vector_1) == len(vector_2)
print(f"生成的向量维度: {len(vector_1)}")
```

---

## 4. Vector Stores（向量存储）

Vector Stores 用于存储文档的向量表示，并支持基于向量的相似度搜索。

### 4.1 常见的向量数据库

| 向量数据库 | 特点 | 包 |
|------------|------|-----|
| `InMemoryVectorStore` | 内存存储，适合测试 | `langchain_core` |
| `FAISS` | Facebook 开源，高效本地存储 | `langchain_community` |
| `Chroma` | 开源嵌入式数据库 | `langchain_chroma` |
| `Pinecone` | 云端托管服务 | `langchain_pinecone` |
| `Qdrant` | 高性能向量搜索引擎 | `langchain_qdrant` |
| `Milvus` | 分布式向量数据库 | `langchain_milvus` |

### 4.2 代码示例

#### InMemoryVectorStore（测试用）

```python
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

# 添加文档
document_ids = vector_store.add_documents(documents=all_splits)
print(f"添加了 {len(document_ids)} 个文档")
```

#### FAISS

```python
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# 获取向量维度
embedding_dim = len(embeddings.embed_query("hello world"))

# 创建 FAISS 索引
index = faiss.IndexFlatL2(embedding_dim)

# 初始化向量存储
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# 添加文档
vector_store.add_documents(documents=all_splits)
```

#### Chroma（支持持久化）

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # 本地持久化目录
)

# 添加文档
vector_store.add_documents(documents=all_splits)
```

#### Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

# 初始化客户端（内存模式）
client = QdrantClient(":memory:")

# 获取向量维度
vector_size = len(embeddings.embed_query("sample text"))

# 创建集合
if not client.collection_exists("my_collection"):
    client.create_collection(
        collection_name="my_collection",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

# 初始化向量存储
vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_collection",
    embedding=embeddings,
)
```

### 4.3 相似度搜索

```python
# 基本相似度搜索
results = vector_store.similarity_search(
    "What is the main topic?",
    k=3  # 返回最相似的3个文档
)
print(results[0].page_content)

# 带分数的相似度搜索
results = vector_store.similarity_search_with_score(
    "What is the main topic?"
)
doc, score = results[0]
print(f"相似度分数: {score}")
print(f"文档内容: {doc.page_content}")

# 基于向量的相似度搜索
embedding = embeddings.embed_query("What is the main topic?")
results = vector_store.similarity_search_by_vector(embedding)

# 带过滤条件的搜索
results = vector_store.similarity_search(
    "query",
    k=3,
    filter={"source": "example.pdf"}  # 只搜索特定来源的文档
)
```

---

## 5. Retrievers（检索器）

Retrievers 是对 Vector Stores 的高级封装，提供统一的检索接口。

### 5.1 从 Vector Store 创建 Retriever

```python
# 创建基本检索器
retriever = vector_store.as_retriever(
    search_type="similarity",      # 搜索类型
    search_kwargs={"k": 3}         # 返回3个结果
)

# 执行检索
docs = retriever.invoke("What is the main topic?")

# 批量检索
results = retriever.batch([
    "Question 1?",
    "Question 2?"
])
```

### 5.2 检索类型

```python
# 1. similarity - 相似度搜索（默认）
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 2. mmr - 最大边际相关性（减少结果冗余）
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20  # 先获取20个，再从中选4个最不相似的
    }
)

# 3. similarity_score_threshold - 基于分数阈值过滤
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8  # 只返回相似度 > 0.8 的结果
    }
)
```

### 5.3 自定义 Retriever

```python
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

@chain
def custom_retriever(query: str) -> List[Document]:
    """自定义检索逻辑"""
    # 可以在这里添加预处理、后处理等逻辑
    return vector_store.similarity_search(query, k=3)

# 使用自定义检索器
docs = custom_retriever.invoke("What is RAG?")
```

---

## 6. 完整 RAG 流程示例

### 6.1 构建知识库

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# 1. 加载文档
loader = WebBaseLoader(
    web_paths=("https://example.com/article",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title"))
    ),
)
docs = loader.load()

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
all_splits = text_splitter.split_documents(docs)

# 3. 创建向量存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# 4. 索引文档
document_ids = vector_store.add_documents(documents=all_splits)
print(f"索引了 {len(document_ids)} 个文档块")
```

### 6.2 创建 RAG 检索工具

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """检索相关信息来回答问题。"""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
```

### 6.3 构建 RAG Agent

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# 初始化模型
model = init_chat_model("gpt-4")

# 定义工具
tools = [retrieve_context]

# 创建 Agent
prompt = """你可以使用检索工具从知识库中获取信息。
在回答用户问题之前，请先检索相关内容。"""

agent = create_agent(model, tools, system_prompt=prompt)

# 使用 Agent
query = "What is the main topic of the article?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

---

## 7. 安装依赖

```bash
# 核心依赖
pip install langchain langchain-core langchain-community

# 文本分割
pip install langchain-text-splitters

# OpenAI 集成
pip install langchain-openai

# 向量数据库（按需安装）
pip install langchain-chroma      # Chroma
pip install langchain-pinecone    # Pinecone
pip install langchain-qdrant      # Qdrant
pip install faiss-cpu             # FAISS (CPU版本)

# 网页加载
pip install beautifulsoup4
```

---

## 8. 最佳实践

### 8.1 文档分割

- 使用 `RecursiveCharacterTextSplitter` 作为默认选择
- `chunk_size` 通常设置为 500-1500 字符
- `chunk_overlap` 设置为 chunk_size 的 10-20%
- 对于代码，可以使用语言特定的分割器

### 8.2 向量存储选择

| 场景 | 推荐方案 |
|------|----------|
| 开发测试 | `InMemoryVectorStore` 或 `FAISS` |
| 小规模生产 | `Chroma` (本地持久化) |
| 大规模生产 | `Pinecone`, `Qdrant`, `Milvus` |

### 8.3 检索优化

1. **使用 MMR** 减少结果冗余
2. **设置合适的 k 值**：太小可能遗漏信息，太大增加噪声
3. **利用元数据过滤**：缩小搜索范围
4. **分数阈值过滤**：只保留高相关性结果

---

## 9. 参考资源

- [LangChain 官方文档 - Retrieval](https://python.langchain.com/docs/concepts/retrieval/)
- [LangChain 官方文档 - RAG](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain 集成列表](https://python.langchain.com/docs/integrations/)
