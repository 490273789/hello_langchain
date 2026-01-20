from langchain_community.document_loaders import Docx2txtLoader

# 需要安装docx2txt
file_path = "./datas/行业.docx"

loader = Docx2txtLoader(file_path)
docs = loader.load()
print(docs)
