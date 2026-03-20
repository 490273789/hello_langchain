import os

import dotenv
from langchain.chat_models import BaseChatModel, init_chat_model

dotenv.load_dotenv()


def init_llm_client() -> BaseChatModel:
    api_key = os.getenv("MODEL_NAME")
    if not api_key:
        raise ValueError("环境变量 MODEL_NAME 为配置，请检查.env文件")
    llm = init_chat_model(
        model=os.getenv("MODEL_NAME"),
        model_provider="openai",
        base_url=os.getenv("MODEL_BASE_URL"),
        api_key=os.getenv("MODEL_KEY"),
    )
    return llm
