from langchain_text_splitters import MarkdownHeaderTextSplitter

# CharacterTextSplitter 简单切分
markdown = """ 
# 第一章
## 第一节
第一节内容，你好
## 第二节
第二节内容，你好
"""

headers = [("#", "Header1"), ("##", "Header2")]  # 设置切分规则
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

chunks = splitter.split_text(markdown)
print(chunks)
print("-" * 88)
print(len(chunks))
