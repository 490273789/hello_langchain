from langchain_community.document_loaders import PyPDFDirectoryLoader
# 索引
# 1. 读取PDF
# 2. 分割 文本，文本段（chunk）
# 3. 向量化：文本段 <-> 向量，需要嵌入模型来辅助
# 4. 向量库：把多个文本段/向量存到向量库

# 读取PDF

file_path = "test.pdy"

loader = PyPDFDirectoryLoader(file_path)
docs = loader.load()

print(f"FDF页数：{len(docs)}")