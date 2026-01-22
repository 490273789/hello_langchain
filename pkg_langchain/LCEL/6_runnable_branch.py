# RunnableBranch 是一个if else 的操作，能够根据我们所写的判断条件，去具体执行某一个分支的逻辑
from langchain_core.runnables import RunnableBranch

# 条件的参数是元组，第一个参数是判断条件，第二个参数是命中条件后，所执行的逻辑
branch = RunnableBranch(
    (lambda x: isinstance(x, str), lambda x: x.upper()),
    (lambda x: isinstance(x, int), lambda x: x + 1),
    (lambda x: isinstance(x, float), lambda x: x * 2),
    lambda x: "goodbye",
)
print(branch.invoke(1))
