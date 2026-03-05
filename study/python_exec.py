import os
import subprocess
import sys


def main() -> None:
    command = "ls -la"
    cwd = os.getcwd()

    # Parse command and arguments similar to command.split(' ') behavior.
    parts = command.split(" ")
    cmd, args = parts[0], parts[1:]

    error_msg = ""
    try:
        completed = subprocess.run(
            [cmd, *args],
            cwd=cwd,
            shell=True,
            check=False,
        )
        code = completed.returncode
    except OSError as error:
        error_msg = str(error)
        code = 1

    if code == 0:
        sys.exit(0)

    if error_msg:
        print(f"错误: {error_msg}", file=sys.stderr)
    sys.exit(code or 1)


if __name__ == "__main__":
    main()
