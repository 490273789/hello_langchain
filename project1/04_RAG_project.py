from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import DashScopeEmbeddings
from models import *

""" 
索引化，构建知识库
"""

# 家在文档
loader = Docx2txtLoader("../datas/1.pdf")