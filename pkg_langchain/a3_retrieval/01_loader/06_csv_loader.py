from langchain_community.document_loaders import CSVLoader

file_path = "pkg_langchain/retrieval/datas/test.csv"
# content_columns筛选列的信息
# metadata_columns 添加metadata的信息
loader = CSVLoader(file_path=file_path, content_columns=["1"], metadata_columns=["1"])

data = loader.load()

print(data[0].metadata)
print(data[0].page_content)
