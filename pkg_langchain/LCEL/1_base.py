from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models import qwen

# Runnable类是一个抽象类（接口），继承了此接口的类都会重写抽象类中的相关方法
# 他会强制要求LCEL组建实现以下几个标准方法
# invoke/ainvoke  batch/abactch  stream/astream
prompt = ChatPromptTemplate(
    [("system", "把用户输入翻译成{language}"), ("user", "{text}")]
)

parser = StrOutputParser()

# LCEL
_chain = prompt | qwen | parser

result = _chain.invoke({"language": "英文", "text": "朝花夕拾"})

print(result)


# | 关到符的底层实现方式：想要两个类的实例之间能够进行 管道符操作，需要在类当中实现 __or__方法
class MyClass:
    def __init__(self, age):
        self.age = age

    def __or__(self, other):
        if type(other) != MyClass:
            return NotImplemented
        return self.age + other.age

    def __ror__(self, other):
        if type(other) == MyClass:
            return NotImplemented
        return self.age + other.age
