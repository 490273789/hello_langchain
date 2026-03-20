from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

from models import gpt4mini, gpt4o
from tools import init_llm_client, pretty_print

# def func1(a1):
#     return a1 + "__func1_output"

# def func2(a2):
#     return a2 + "__func2_output"

# runnable_parallel = RunnableParallel({"key1": func1, "key2": func2})

# runnable_parallel.invoke("你好！")
messages1 = [
    ("system", "你是我的老师，用中文回答我的问题"),
    ("user", "{user_question}"),
]
prompt1 = ChatPromptTemplate.from_messages(messages1)

messages2 = [
    ("system", "你是我的老师，用英文回答我的问题"),
    ("user", "{user_question}"),
]
prompt2 = ChatPromptTemplate.from_messages(messages2)

llm = init_llm_client()


parser = StrOutputParser()

pretty_print("模型响应中。。。")
runnable_parallel1 = prompt1 | RunnableParallel({"ds": gpt4mini, "qwen": gpt4o})
res1 = runnable_parallel1.invoke({"user_question": "purpose 和 propose有什么区别？"})
pretty_print(parser.invoke(res1["ds"]), "ds")
pretty_print(res1["qwen"], "qwen")
runnable_parallel1.get_graph().print_ascii()

chain1 = prompt1 | llm | parser
chain2 = prompt2 | llm | parser


pretty_print("模型响应中。。。")
runnable_parallel2 = prompt1 | RunnableParallel({"中文": chain1, "英文": chain2})
res2 = runnable_parallel2.invoke({"user_question": "purpose 和 propose有什么区别？"})
pretty_print(res2["中文"], "中文")
pretty_print(res2["英文"], "英文")
runnable_parallel2.get_graph().print_ascii()
