import os
from langchain.chat_models import init_chat_model

qwen = init_chat_model(
    model="qwen-flash",
    model_provider="openai",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)