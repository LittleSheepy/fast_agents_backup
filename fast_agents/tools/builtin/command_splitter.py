
import os
import sys
import locale
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Set, Union

from pydantic import BaseModel, Field



from typing import List


def split_windows(command: str) -> List[str]:
    """Windows 安全的简单命令分词，不破坏反斜杠路径。

    规则：
    - 双引号包裹的参数视为一个整体（引号保留）
    - 其余按空白分割
    - 不做反斜杠转义处理（与 POSIX shlex 的关键区别）
    """
    parts: List[str] = []
    current: List[str] = []
    in_quotes = False
    i = 0
    while i < len(command):
        ch = command[i]
        if ch == '"' and not in_quotes:
            in_quotes = True
            current.append(ch)
        elif ch == '"' and in_quotes:
            in_quotes = False
            current.append(ch)
        elif ch in (' ', '\t', '\n') and not in_quotes:
            if current:
                parts.append(''.join(current))
                current = []
        else:
            current.append(ch)
        i += 1
    if current:
        parts.append(''.join(current))
    return parts


def split_windows_commands(command: str) -> List[List[str]]:
    """把含 CMD 操作符的命令行拆成多条命令列表。

    识别的操作符：&&、||、|、&
    - 引号内部的操作符不会被拆分
    - 每条子命令仍通过 split_windows 做安全分词
    - 操作符本身不会出现在结果中
    """
    tokens: List[str] = []
    current: List[str] = []
    in_quotes = False
    i = 0

    while i < len(command):
        ch = command[i]

        if ch == '"':
            in_quotes = not in_quotes
            current.append(ch)
            i += 1
            continue

        if not in_quotes:
            # 尝试匹配双字符操作符 &&、||
            if i + 1 < len(command) and command[i : i + 2] in ("&&", "||"):
                if current:
                    tokens.append(''.join(current))
                    current = []
                i += 2
                continue

            # 匹配单字符操作符 |、&（注意 & 后面紧跟 & 已在上面处理）
            if ch in ("|", "&"):
                if current:
                    tokens.append(''.join(current))
                    current = []
                i += 1
                continue

        current.append(ch)
        i += 1

    if current:
        tokens.append(''.join(current))

    # 对每条子命令做安全分词，过滤空白片段
    commands: List[List[str]] = []
    for token in tokens:
        parts = split_windows(token)
        if parts:
            commands.append(parts)

    return commands


if __name__ == "__main__":
    print("")
    SPLIT_WINDOWS_COMMANDS_CASES = {
        # 单条命令：应返回包裹一层列表
        "dir C:\\Users": [["dir", "C:\\Users"]],
        "git   status": [["git", "status"]],
        "echo\thello\nworld": [["echo", "hello", "world"]],

        # 双引号包裹的路径
        'dir "C:\\Program Files\\App"': [["dir", '"C:\\Program Files\\App"']],
        'copy "a b.txt" "c d.txt"': [["copy", '"a b.txt"', '"c d.txt"']],

        # 反斜杠路径（末尾反斜杠不转义）
        "dir C:\\path\\": [["dir", "C:\\path\\"]],
        'net use "\\\\server\\share"': [["net", "use", '"\\\\server\\share"']],

        # 空输入 / 纯空白
        "": [],
        "   \t  \n  ": [],

        # 引号边界情况
        '""': [['""']],
        'echo "hello': [["echo", '"hello']],
        'echo hello"': [["echo", 'hello"']],

        # && 连接多条命令
        'dir C:\\Users && echo done': [
            ["dir", "C:\\Users"],
            ["echo", "done"],
        ],

        # || 失败回退
        'dir "C:\\Not Exists" || echo failed': [
            ["dir", '"C:\\Not Exists"'],
            ["echo", "failed"],
        ],

        # | 管道
        'type a.txt | findstr "hello"': [
            ["type", "a.txt"],
            ["findstr", '"hello"'],
        ],

        # & 后台/顺序执行
        'echo a & echo b': [
            ["echo", "a"],
            ["echo", "b"],
        ],

        # 链式操作符
        'cmd1 && cmd2 || cmd3': [
            ["cmd1"],
            ["cmd2"],
            ["cmd3"],
        ],

        # 引号内的操作符不拆分
        'echo "a && b"': [
            ["echo", '"a && b"'],
        ],

        # 混合：引号路径 + 操作符
        'cmd /c "dir C:\\Program Files" && echo done': [
            ["cmd", "/c", '"dir C:\\Program Files"'],
            ["echo", "done"],
        ],

        # 首尾多余空格
        "  dir /b  ": [["dir", "/b"]],

        # 单 token
        "whoami": [["whoami"]],

        # 操作符紧贴无空格
        'echo a&&echo b': [
            ["echo", "a"],
            ["echo", "b"],
        ],

        # 连续操作符产生空命令，应被过滤
        'cmd1 && || cmd2': [
            ["cmd1"],
            ["cmd2"],
        ],
    }


    def run_split_tests():
        for command, expected in SPLIT_WINDOWS_COMMANDS_CASES.items():
            result = split_windows_commands(command)
            assert result == expected, (
                f"[{command!r}] failed:\n"
                f"  expected: {expected}\n"
                f"  got:      {result}"
            )
            print(f" {command!r} ==> {result}")


    run_split_tests()

