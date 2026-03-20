from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from tools import init_llm_client

# 统一接口 + 调用方法统一
# 类似于Linux的管道操作符，前一个表达式返回值，是下个的入参
# Runnable(ABC) ABC - Abstract Base Class
# Runnable类是一个抽象类（接口），继承了此接口的类都会重写抽象类中的相关方法
# 他会强制要求LCEL组建实现以下几个标准方法
# invoke/ainvoke  batch/abatch  stream/astream
prompt = ChatPromptTemplate(
    [("system", "把用户输入翻译成{language}"), ("user", "{text}")]
)

llm = init_llm_client()
parser = StrOutputParser()

# LCEL:  LangChain Expression Language
# 通过LCEL（| 运算符，RunnableSequence，RunnableParallel等）快速的拼接多个Runnable为复杂的工作流，支持条件分枝、并行执行等
_chain = prompt | llm | parser

result = _chain.invoke({"language": "英文", "text": "朝花夕拾"})

print(result)


# | 关到符的底层实现方式：想要两个类的实例之间能够进行 管道符操作，需要在类当中实现 __or__方法
class MyClass:
    def __init__(self, age):
        self.age = age

    def __or__(self, other):
        if type(other) is not MyClass:
            return NotImplemented
        return self.age + other.age

    def __ror__(self, other):
        if type(other) is MyClass:
            return NotImplemented
        return self.age + other.age


# RunnableSequence - 顺序链
# RunnableBranch - 分支链
# RunnableSerializable - 串行链
# RunnableParallel - 并行链
# RunnableLambda - 函数链
