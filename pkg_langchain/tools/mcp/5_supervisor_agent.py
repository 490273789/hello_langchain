import asyncio
import os
import smtplib
from email.mime.text import MIMEText

import dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

dotenv.load_dotenv()

llm = init_chat_model(
    model="gpt-4o",
    model_provider="openai",
)


# ========== 创建一个有搜索功能的子Agent ==========
class SearchSubAgent:
    """带搜索功能的子Agent"""

    def __init__(self):
        self.tools = asyncio.run(
            MultiServerMCPClient(
                {
                    "WebSearch": {
                        "transport": "sse",
                        "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse",
                        "headers": {
                            "Authorization": f"Bearer {os.getenv('DASHSCOPE_MCP_KEY')}"
                        },
                    },  # https://bailian.console.aliyun.com/?tab=mcp#/mcp-market/detail/WebSearch
                    "RailService": {
                        "transport": "sse",
                        "url": "https://dashscope.aliyuncs.com/api/v1/mcps/china-railway/sse",
                        "headers": {
                            "Authorization": f"Bearer {os.getenv('DASHSCOPE_MCP_KEY')}"
                        },
                    },  # https://bailian.console.aliyun.com/?tab=mcp#/mcp-market/detail/WebSearch
                }
            ).get_tools()
        )

        self.agent = create_agent(model=llm, tools=self.tools)

    async def __call__(self, input: str) -> str:
        return await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": input}]}
        )


# ========== 创建一个能发送邮件的子Agent ==========
@tool
async def send_email(to: list[str], subject: str, body: str) -> str:
    """
    发送邮件使用

    Args:
        to: 收件人
        subject: 邮件主题
        body: 邮件正文
    """

    print(f"send_email{to}{subject}{body}")
    SMTP_HOST = "smtp.163.com"
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")  # 需要在邮箱中开启 SMTP 并生成授权码

    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = SMTP_USER
    msg["Subject"] = subject

    try:
        server = smtplib.SMTP_SSL(SMTP_HOST, 465, timeout=10)
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, to, msg.as_string())
        try:
            server.quit()
        except smtplib.SMTPResponseException as e:
            if e.smtp_code == -1 and e.smtp_error == b"\x00\x00\x00":
                pass  # 忽略无害的关闭异常
            else:
                raise
        return "success"
    except Exception as e:
        print(e)
        return f"Send failed: {type(e).__name__} - {e}"


class EmailSubAgent:
    """带发送邮件功能的子代理"""

    def __init__(self):
        self.tools = [send_email]

        self.agent = create_agent(model=llm, tools=self.tools)

    async def __call__(self, input: str) -> str:
        return await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": input}]}
        )


search_subagent = SearchSubAgent()
email_subagent = EmailSubAgent()


# ========== 将子 Agent 包装为工具 ==========
@tool
async def search(input: str) -> str:
    """
    一个具有搜索功能的子Agent，功能包括：
    - 搜索网页
    - 搜索火车票相关信息
    """
    return await search_subagent(input)


@tool
async def email(input: str) -> str:
    """
    一个具有发送邮件功能的子Agent
    """
    return await email_subagent(input)


# ========== 创建主管 Agent ==========
supervisor_agent = create_agent(
    model=llm,
    tools=[search, email],
    system_prompt="你是一个主管，需要调用子Agent来帮助用户",
)


async def main():
    async for chunk in supervisor_agent.astream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "北京明天天气怎么样，要是还不错的话，帮我看看明天上海到北京的车票。如果天气好的话，发送邮件给snpro@qq.com告诉他我明天去北京。如果天气不好的话就告诉他我明天不去北京了。要告诉邮件是否发送成功",
                }
            ]
        }
    ):
        print(chunk, end="\n\n")


asyncio.run(main())
