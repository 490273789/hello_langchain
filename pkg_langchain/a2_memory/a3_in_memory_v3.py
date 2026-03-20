from langchain_core.chat_history import InMemoryChatMessageHistory

from tools import init_llm_client, logger, pretty_print

# 设置本地模型，不使用深度思考
model = init_llm_client()


# 创建内存聊天历史记录实例，用于存储对话消息
history = InMemoryChatMessageHistory()

# 添加用户消息到聊天历史记录
history.add_user_message("我叫张三，我的爱好是学习")
pretty_print("我叫张三，我的爱好是学习", "提问")
logger.info("思考中。。。")
# 调用语言模型处理聊天历史中的消息
ai_message = model.invoke(history.messages)

# 记录并输出AI回复的内容
pretty_print(ai_message.content, "第一次回答")

# 将AI回复添加到聊天历史记录中
history.add_message(ai_message)

# 添加新的用户消息到聊天历史记录
history.add_user_message("我叫什么？我的爱好是什么？")
pretty_print("我叫什么？我的爱好是什么？", "提问")
logger.info("思考中。。。")
# 再次调用语言模型处理更新后的聊天历史
ai_message2 = model.invoke(history.messages)

# 记录并输出第二次AI回复的内容
pretty_print(ai_message2.content, "第二次回答")

# 将第二次AI回复添加到聊天历史记录中
history.add_message(ai_message2)

print()

logger.info("history 列表：")
# 遍历并输出所有聊天历史记录中的消息内容
for message in history.messages:
    pretty_print(message.content, message.type)
