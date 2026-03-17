import os

import dotenv
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.exceptions import LangChainException

from tools.log_tool import logger

dotenv.load_dotenv()


def init_llm_client() -> BaseChatModel:
    api_key = os.getenv("MODEL_NAME")
    if not api_key:
        raise ValueError("环境变量 MODEL_NAME 为配置，请检查.env文件")
    llm = init_chat_model(
        model=os.getenv("MODEL_NAME"),
        model_provider="openai",
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    return llm


def main():
    try:
        llm = init_llm_client()
        logger.info("LLM客户端初始化成功")

        question = "你是谁"
        response = llm.invoke(question)

        # 格式化输出结果
        logger.info(f"问题：{question}")
        logger.info(f"回答：{response.content}")

        print("======== 流式输出")
        print("*" * 66)
        response_stream = llm.stream("介绍langchain，300字以内")

        for chunk in response_stream:
            print(chunk.content, end="")
    except ValueError as e:
        logger.error(f"配置错误： {str(e)}")
    except LangChainException as e:
        logger.error(f"模型调用失败： {str(e)}")
    except Exception as e:
        logger.error(f"未知错误{str(e)}")


if __name__ == "__main__":
    main()
