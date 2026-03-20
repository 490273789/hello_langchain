# RunnableBranch 是一个if else 的操作，能够根据我们所写的判断条件，去具体执行某一个分支的逻辑
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch

from tools import init_llm_client, pretty_print

prompt = ChatPromptTemplate.from_messages(
    [("system", "你是助手小王"), ("human", "{query}")]
)
llm = init_llm_client()
parser = StrOutputParser()

# 条件的参数是元组，第一个参数是判断条件，第二个参数是命中条件后，所执行的逻辑
branch = RunnableBranch(
    (
        lambda x: isinstance(x, dict) and isinstance(x.get("query"), str),
        prompt | llm | parser,
    ),
    (lambda x: isinstance(x, int), lambda x: x + 1),
    (lambda x: isinstance(x, float), lambda x: x * 2),
    lambda x: "goodbye",
)
pretty_print(branch.invoke({"query": "hello"}), "result")
