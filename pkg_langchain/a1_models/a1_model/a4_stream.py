# 1.导入依赖
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage

from tools import init_llm_client, pretty_print

load_dotenv()

# 2.实例化模型
model = init_llm_client()

# 构建消息列表
messages = [
    SystemMessage(content="你叫小问，是一个乐于助人的AI人工助手"),
    HumanMessage(content="你是谁"),
]

# 3.流式调用大模型
response = model.stream(messages)
pretty_print(f"响应类型：{type(response)}")
# 流式打印结果
for chunk in response:
    # 刷新缓冲区 (无换行符，缓冲区未刷新，内容可能不会立即显示)
    print(chunk.content, end="", flush=True)
print("\n")
