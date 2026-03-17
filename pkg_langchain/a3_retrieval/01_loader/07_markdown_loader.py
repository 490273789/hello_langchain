from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 需要安装markdown和unstructured
file_path = "pkg_langchain/retrieval/datas/test.md"

# mode=elements 可按照md的标准格式去进行加载
loader = UnstructuredMarkdownLoader(file_path=file_path, mode="elements")

data = loader.load()

print(data)
