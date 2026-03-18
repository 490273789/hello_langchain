# 1.导入依赖
from dotenv import load_dotenv

from tools import init_llm_client, pretty_print

# 通过 python-dotenv 库读取 env 文件中的环境变量，并加载到当前运行的环境中
load_dotenv()

# 2.实例化模型
model = init_llm_client()

# 问题列表
questions = [
    "什么是redis?简洁回答，字数控制在100以内",
    "Python的生成器是做什么的？简洁回答，字数控制在100以内",
    "解释一下Docker和Kubernetes的关系?简洁回答，字数控制在100以内",
]

pretty_print("模型响应中。。。")
# 批量调用大模型 model.batch()
response = model.batch(questions)
pretty_print(f"响应类型：{type(response)}")
print()
for q, r in zip(questions, response):
    pretty_print(f"问题：{q}\n回答：{r.content}\n")
