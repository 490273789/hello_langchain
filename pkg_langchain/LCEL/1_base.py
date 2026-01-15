from models import qwen
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate([("system", "把用户输入翻译成{language}"), ("user", "{text}")])

parser = StrOutputParser()

_chain = prompt | qwen | parser

result = _chain.invoke({"language": "英文", "text": "朝花夕拾"})

print(result)