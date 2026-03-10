import os

import dotenv
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings

dotenv.load_dotenv()

qwen = init_chat_model(
    model="qwen3.5-flash",
    model_provider="openai",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# qwen3-vl-embedding
# multimodal-embedding-v1
qwen_eb = DashScopeEmbeddings(
    model=os.getenv("EMBEDDINGS_MODEL_NAME"),
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
)
