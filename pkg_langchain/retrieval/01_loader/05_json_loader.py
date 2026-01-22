from langchain_community.document_loaders import JSONLoader

file_path = "pkg_langchain/retrieval/datas/test.json"
# jq_schema 提取json中对象的一种特定方式，特定的语法
# "."是提取所有内容， ".status"就是status的内容,".data.page",".data.items[].id"
# text_content: 告诉加载器读取的内容是不是text，当jq_schema为"."的时候如果设置True会报错
loader = JSONLoader(
    file_path=file_path, jq_schema=".data.items[].id", text_content=False
)

data = loader.load()

print(data[0].metadata)
print(data[0].page_content)
