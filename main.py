import json

from langchain_core.exceptions import LangChainException

from study.a5_history import summarization_memory_demo
from tools import logger


def test():
    messages = [{"role": "Human", "content": "nihao"}]
    return json.dumps(messages)


def main():
    try:
        logger.info("LLM客户端初始化成功")
        summarization_memory_demo()
    except ValueError as e:
        logger.error(f"配置错误： {str(e)}")
    except LangChainException as e:
        logger.error(f"模型调用失败： {str(e)}")
    except Exception as e:
        logger.error(f"未知错误{str(e)}")


if __name__ == "__main__":
    main()
