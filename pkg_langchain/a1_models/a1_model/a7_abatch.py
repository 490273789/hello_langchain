# 1.导入依赖（新增 asyncio 用于运行异步程序）
import asyncio

from dotenv import load_dotenv

from tools import init_llm_client, pretty_print

load_dotenv()

# 2.实例化模型
model = init_llm_client()

questions = [
    "什么是redis?简洁回答，字数控制在100以内",
    "Python的生成器是做什么的？简洁回答，字数控制在100以内",
    "解释一下Docker和Kubernetes的关系?简洁回答，字数控制在100以内",
]


# 3.异步批量调用大模型（定义异步函数封装异步操作）
# abatch() 是异步方法，需要基于 async/await 语法构建异步程序，并用 asyncio 驱动运行
async def async_batch_call():
    pretty_print("模型响应中。。。")
    # 调用 model.abatch() 异步批量处理请求，需用 await 修饰（关键）
    response = await model.abatch(questions)
    pretty_print(f"响应类型：{type(response)}")

    # 遍历结果并格式化输出（与原来的同步版本格式一致）
    for q, r in zip(questions, response):
        pretty_print(f"问题：{q}\n回答：{r.content}\n")


# 4.运行异步函数
if __name__ == "__main__":
    asyncio.run(async_batch_call())
