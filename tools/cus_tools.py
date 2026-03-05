import asyncio
import os

from langchain.tools import tool
from pydantic import BaseModel, Field


class ReadFileArgs(BaseModel):
    file_path: str = Field(description="'要读取的文件路径'")


@tool(
    "read_file",
    args_schema=ReadFileArgs,
    description="用此工具来读取文件内容。当用户要求读取文件、查看代码、分析文件内容时，调用此工具。输入文件路径（可以是相对路径或绝对路径）。",
)
def read_file(file_path: str):
    try:
        with open(file_path, "r") as file:
            content = file.read()
            print(f"[tool] read_file({file_path}) - read {len(content)} bytes")
            return content
    except Exception as e:
        error_msg = (
            f"[tool] read_file({file_path}) - Error: {type(e).__name__}: {str(e)}"
        )
        print(error_msg)
        return error_msg


class WriteFileArgs(BaseModel):
    file_path: str = Field(description="文件路径")
    content: str = Field(description="要写入文件的内容")


@tool(
    "write_file",
    args_schema=WriteFileArgs,
    description="向指定路径写入文件内容，自动创建目录",
)
def write_file(file_path, content):
    try:
        dir_path = os.path.dirname(file_path)
        # 2. 递归创建目录 (对应 fs.mkdir with recursive: true)
        # exist_ok=True 表示如果目录已存在则不报错
        if dir_path:  # 防止 filePath 只是文件名而没有目录部分时报错
            os.makedirs(dir_path, exist_ok=True)
        # 3. 写入文件 (对应 fs.writeFile with 'utf-8')

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        # 4. 打印日志 (对应 console.log)
        # 注意：Python 中 len(content) 获取的是字符数。
        # 如果要严格对应“字节数”，需要使用 len(content.encode('utf-8'))
        byte_length = len(content.encode("utf-8"))
        print(f'  [工具调用] write_file("{file_path}") - 成功写入 {byte_length} 字节')

        # 5. 返回结果
        return f"文件写入成功: {file_path}"
    except Exception as e:
        return f"文件写入失败: {e}"


class ExecuteCommandArgs(BaseModel):
    command: str = Field(description="要执行的命令")
    working_directory: str = Field(description="工作目录（推荐指定）")


@tool(
    "execute_command",
    args_schema=ExecuteCommandArgs,
    description="执行系统命令，支持指定工作目录，实时显示输出",
)
async def execute_command(command, working_directory):
    cwd = working_directory or os.getcwd()
    print(
        f"[工具调用] execute_command('{command}') {f'- 工作目录: {working_directory}' if working_directory else ''}"
    )
    try:
        # 创建子进程
        # shell=True 对应 JS 的 shell: true
        # stdout=None, stderr=None 对应 JS 的 stdio: 'inherit' (直接输出到父进程的控制台)
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=None,  # 继承父进程.stdout，实现实时输出
            stderr=None,  # 继承父进程.stderr
            stdin=None,  # 继承父进程.stdin
        )

        # 等待进程结束
        return_code = await process.wait()

        if return_code == 0:
            print(f'  [工具调用] execute_command("{command}") - 执行成功')

            cwd_info = ""
            if working_directory:
                cwd_info = (
                    f'\n\n重要提示：命令在目录 "{working_directory}" 中执行成功。'
                    f'如果需要在这个项目目录中继续执行命令，请使用 workingDirectory: "{working_directory}" 参数，不要使用 cd 命令。'
                )

            return f"命令执行成功：{command}{cwd_info}"
        else:
            print(
                f'  [工具调用] execute_command("{command}") - 执行失败，退出码：{return_code}'
            )
            # 注意：由于使用了 stdio: inherit (stdout=None)，错误输出已经直接打印在控制台了。
            # 原 JS 代码尝试捕获 error 事件的 message，但在 close 事件中主要依赖退出码。
            # 这里模拟返回错误信息字符串。
            return f"命令执行失败，退出码：{return_code}"

    except Exception as e:
        # 对应 JS 中 child.on('error', ...)
        error_msg = str(e)
        print(f'  [工具调用] execute_command("{command}") - 发生异常：{error_msg}')
        return f"命令执行失败，退出码：-1\n错误：{error_msg}"


class ListDirectoryArgs(BaseModel):
    directory_path: str = Field(description="目录路径")


@tool(
    "list_directory",
    args_schema=ListDirectoryArgs,
    description="列出指定目录下的所有文件和文件夹",
)
async def list_directory(directory_path: str):
    """
    列出指定目录下的文件和文件夹。
    对应原 JS 代码中的 list_directory 逻辑。
    """
    try:
        # 在 Python 中，os.listdir 是阻塞的。
        # 在异步环境中，我们使用 asyncio.to_thread 将其放入线程池执行，避免阻塞主事件循环。
        files = await asyncio.to_thread(os.listdir, directory_path)

        print(
            f'  [工具调用] list_directory("{directory_path}") - 找到 {len(files)} 个项目'
        )

        # 格式化输出：每个文件前加 "- "
        file_list_str = "\n".join([f"- {f}" for f in files])

        return f"目录内容:\n{file_list_str}"

    except Exception as e:
        # 捕获所有异常 (如 FileNotFoundError, PermissionError 等)
        error_message = str(e)
        print(
            f'  [工具调用] list_directory("{directory_path}") - 错误：{error_message}'
        )
        return f"列出目录失败：{error_message}"
