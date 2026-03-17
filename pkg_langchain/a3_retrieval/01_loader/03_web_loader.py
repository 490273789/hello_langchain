import bs4  # beautiful soup
from langchain_community.document_loaders import WebBaseLoader

# 1. 网页 文档加载器

web_path = (
    "https://www.news.cn/fortune/20251218/93d4a4212fd34985ad05d178a27c7c75/c.html"
)

loader = WebBaseLoader(
    web_path=[web_path],
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("main-left left", "title"))),
)

docs = loader.load()

print(docs)
