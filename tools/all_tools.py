from __future__ import annotations

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool


@tool("read_file")
def read_file(file_path: str) -> str:
    """Read a file by path and return its content."""
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        print(f'[tool] read_file("{file_path}") - read {len(content)} bytes')
        return f"文件内容:\n{content}"
    except Exception as exc:  # noqa: BLE001
        print(f'[tool] read_file("{file_path}") - error: {exc}')
        return f"读取文件失败: {exc}"


@tool("write_file")
def write_file(file_path: str, content: str) -> str:
    """Write content to a file and create parent directories automatically."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f'[tool] write_file("{file_path}") - wrote {len(content)} bytes')
        return f"文件写入成功: {file_path}"
    except Exception as exc:  # noqa: BLE001
        print(f'[tool] write_file("{file_path}") - error: {exc}')
        return f"写入文件失败: {exc}"


@tool("execute_command")
def execute_command(command: str, working_directory: str | None = None) -> str:
    """Run a shell command, optionally in a specific working directory."""
    cwd = working_directory or os.getcwd()
    print(
        f'[tool] execute_command("{command}")'
        + (f" - cwd: {working_directory}" if working_directory else "")
    )

    if working_directory and "cd " in command:
        return (
            "命令被拒绝：你已经传入 working_directory，不要在 command 里再使用 cd。"
            f" 当前 command: {command}"
        )

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output_lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)

        return_code = process.wait()
        output = "".join(output_lines).strip()

        if return_code == 0:
            cwd_info = (
                "\n\n重要提示：命令在目录 "
                f'"{working_directory}" 中执行成功。后续继续在该目录执行命令时，请继续传 working_directory，'
                "不要在 command 中使用 cd。"
                if working_directory
                else ""
            )
            return f"命令执行成功: {command}\n输出:\n{output}{cwd_info}"

        return f"命令执行失败，退出码: {return_code}\n命令: {command}\n输出:\n{output}"
    except Exception as exc:  # noqa: BLE001
        return f"命令执行失败: {exc}"


@tool("list_directory")
def list_directory(directory_path: str) -> str:
    """List all items under a directory."""
    try:
        items = sorted(os.listdir(directory_path))
        print(f'[tool] list_directory("{directory_path}") - found {len(items)} items')
        rendered = "\n".join(f"- {item}" for item in items)
        return f"目录内容:\n{rendered}"
    except Exception as exc:  # noqa: BLE001
        print(f'[tool] list_directory("{directory_path}") - error: {exc}')
        return f"列出目录失败: {exc}"


TOOLS = [
    read_file,
    write_file,
    execute_command,
    list_directory,
]
