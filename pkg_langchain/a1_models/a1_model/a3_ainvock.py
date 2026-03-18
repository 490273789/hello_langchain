# 1.导入依赖
import asyncio

from dotenv import load_dotenv

from tools import init_llm_client, pretty_print, pretty_print_ai

load_dotenv()

# 2.实例化模型
model = init_llm_client()


async def main():
    pretty_print("模型响应中。。。")
    response = await model.ainvoke("解释一下LangChain是什么，简洁回答100字以内")
    pretty_print_ai(response, view="zen")


# 4.运行异步函数
if __name__ == "__main__":
    asyncio.run(main())

"""
LangChain 提供 ainvoke() 异步调用接口，用于在 异步环境（async/await） 中高效并行地执行模型推理。
它的核心作用是：让你同时调用多个模型请求而不阻塞主线程 —— 特别适合大批量请求或 Web 服务场景（如 FastAPI）
"""
