from langchain_core.messages import AIMessage, HumanMessage, SystemMessage 
import models
# v1.x 新提供
# ChatPromptTemplate  角色设置
# system 系统角色  user human 用户角色 assistant 大模型回复

# 在 LangChain 中，消息是模型的基本上下文单元。它们代表模型的输入和输出，携带与 LLM 交互时表示对话状态所需的内容和元数据。
# 消息是包含以下内容的对象：
#   角色 —— 标识消息类型（例如 system，user）
#   内容 —— 指消息的实际内容（例如文本、图像、音频、文档等）。
#   元数据 —— 可选字段，例如响应信息、消息 ID 和令牌使用情况
# LangChain 提供了一种适用于所有模型提供程序的标准消息类型，确保无论调用哪个模型，行为都保持一致。

message = [
    ("system", "把用户输入翻译成{language}"),
    ("user", "{text}")
]

message = [
    SystemMessage("你是一个诗歌专家"),
    HumanMessage("写一篇关于春天的七言绝句"),
    AIMessage("...")
]

print(models.gpt)
result = models.gpt.qwen.invoke(message)
print(f"result: {result}")