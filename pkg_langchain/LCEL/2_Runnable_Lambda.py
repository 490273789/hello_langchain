from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, chain

from models import qwen

# 可执行匿名函数 RunnableLambda

# itemgetter
d = {"name": "wsn", "age": "18"}
res = itemgetter("name")(d)

print(f"res: {res}")
print("-" * 50)
# exit()

template = ChatPromptTemplate.from_template("{a} + {b}是多少？")


# 获得字符串长度
def length(t):
    return len(t)


def mul(t1, t2):
    return len(t1) * len(t2)


print(mul(t1="aa", t2="ss"))
print(mul("aa", "ss"))


# @chain是RunnableLambda另一种写法：把函数转换为与LCEL兼容的组建
@chain
def mul_length(d):
    return mul(d["t1"], d["t2"])


chain1 = template | qwen

chain2 = (
    {
        "a": itemgetter("name")
        | RunnableLambda(length),  # a = 3 RunnableLambda 一个参数写法
        "b": {"t1": itemgetter("name"), "t2": itemgetter("sex")}
        | RunnableLambda(lambda x: mul(**x)),  # c = 12 RunnableLambda 多参数写法
        "c": {"t1": itemgetter("name"), "t2": itemgetter("sex")}
        | mul_length,  # b = 12  chain写法
    }
    | chain1
    | StrOutputParser()
)

print(chain2.invoke({"name": "wsn", "sex": "male"}))
