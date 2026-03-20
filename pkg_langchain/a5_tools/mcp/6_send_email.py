import asyncio
import os
import smtplib
from email.mime.text import MIMEText

import dotenv

dotenv.load_dotenv()


async def send_email(to: list[str], subject: str, body: str) -> str:
    """
    发送邮件。需要自动生成邮件主题。

    Args:
        to: 收件人
        subject: 邮件主题
        body: 邮件正文
    """

    print("send_email")
    SMTP_HOST = "smtp.163.com"
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")  # 需要在邮箱中开启 SMTP 并生成授权码

    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = SMTP_USER
    msg["Subject"] = subject

    try:
        print(f"SMTP_USER:{SMTP_USER}{SMTP_PASS}")
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


asyncio.run(send_email(["snpro@qq.com"], "hello", "测试"))
