from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# RunnableWithFallbacks
# 在前面的runnable出现异常时，就会执行RunnableWithFallbacks

llm = init_chat_model(model="deepseek-chat", model_provider="deepseek")

chain = PromptTemplate.from_template("hello") | llm
# 通过调用Runnable组件的with_fallbacks方法就可以得到一个RunnableWithFallbacks实例
# 参数是一个列表，默认使用第一个，如果第一个报错了就会走到下一个
chain_with_fallbacks = chain.with_fallbacks([RunnableLambda(lambda x: "sorry")])

result = chain_with_fallbacks.invoke("1")

print(result)
