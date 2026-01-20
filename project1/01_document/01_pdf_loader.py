from langchain_community.document_loaders import PyMuPDFLoader

# 需要安装pymupdf
file_path = "./datas/Reflexion.pdf"

loader = PyMuPDFLoader(file_path)

docs = loader.load()

print(docs)
