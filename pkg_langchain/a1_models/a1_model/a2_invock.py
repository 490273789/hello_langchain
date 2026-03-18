# 1.导入依赖
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage

from tools import init_llm_client, pretty_print_ai

load_dotenv()

# 2.实例化模型
model = init_llm_client()

# 构建消息列表
messages = [
    SystemMessage(
        content="你是一个法律助手，只回答法律问题，超出范围的统一回答，非法律问题无可奉告"
    ),
    HumanMessage(content="简单介绍下广告法，一句话告知50字以内"),
    # HumanMessage(content="2+3等于几?")
]

# 3.调用模型
response = model.invoke(messages)  # ainvoke
print(f"响应类型：{type(response)}")
# 打印结果
pretty_print_ai(response, view="zen")
