from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 可运行透传
# 接受输入，并原样输出，是langchain中的“无操作节点”，用于在流水线中透传输入或保留上下文，也可以用于向输出中添加键

chain = RunnableParallel(
    original=RunnablePassthrough(),  # 保留中间结果
    word_count=lambda x: len(x),
)

result = chain.invoke("hello world")
print(result)  # {'original': 'hello world', 'word_count': 11}
