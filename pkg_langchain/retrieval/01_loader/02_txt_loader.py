from langchain_community.document_loaders import TextLoader

# txt 文档加载器
file_path = "./datas/deepseek.txt"
# 创基文本加载器
loader = TextLoader(file_path)
# 得到真正的文档对象
docs = loader.load()
print(docs[0].metadata)
print(docs[0].page_content)
