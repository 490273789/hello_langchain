from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from tools import init_llm_client

"""
作用：构造一个串行的执行链，通过runnable_sequence的实例，调用invoke方法，
就等于链当中每一个组件去调用invoke，然后将调用结果传递给下一个组件
"""

prompt_template = PromptTemplate(input_variables=["user_info"], template="{user_info}")
llm = init_llm_client()
output_parser = StrOutputParser()

# runnable_sequence = RunnableSequence(*[prompt_template, llm, output_parser])
runnable_sequence = RunnableSequence(prompt_template, llm, output_parser)

runnable_sequence.invoke("你好，你是谁？")
