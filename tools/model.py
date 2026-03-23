import os

import dotenv
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings

dotenv.load_dotenv()


def init_llm_client(model: str = None, is_embedding: bool = False) -> BaseChatModel:
    if is_embedding:
        default_name = os.getenv("EMBEDDINGS_MODEL_NAME")
        model_name = model if model else default_name
        model = DashScopeEmbeddings(
            model=model_name,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
    else:
        default_name = os.getenv("MODEL_NAME")
        model_name = model if model else default_name
        model = init_chat_model(
            model=model_name,
            model_provider="openai",
            base_url=os.getenv("MODEL_BASE_URL"),
            api_key=os.getenv("MODEL_KEY"),
        )
    return model
