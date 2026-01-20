from langchain_community.document_loaders import TextLoader

# txt 文档加载器
file_path = "./datas/deepseek.txt"
loader = TextLoader(file_path)
docs = loader.load()
print(docs)
