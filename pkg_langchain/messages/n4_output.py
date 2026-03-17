from models import qwen
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate([("system", "把用户输入的中文翻译成{language}"), ("user", "{text}")])

prompt = template.format(language="英文", text="朝花夕拾")

result = qwen.invoke(prompt)
print(result)
print("-" * 50)

# 使用输出解释器StrOutputParser
str_parser = StrOutputParser()
str_result = str_parser.invoke(result)
print(f"str_result: {str_result}")
print("-" * 50)
# v1.x 版本新增
print(f"content_blocks:{result.content_blocks}")