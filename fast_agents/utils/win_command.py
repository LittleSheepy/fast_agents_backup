import os
import sys
import psutil
from typing import Literal, Tuple
import shutil
import shlex


# ── 分平台的 shell 元字符集合 ──────────────────────────────────────

UNIX_SHELL_METACHARS = frozenset((
    "|", "||", "|&",          # 管道
    ">", ">>", "<", "<<", "<<<", "&>", ">&",  # 重定向（含 fd 号）
    "*", "?",                 # glob
    "&&", ";;", ";", "&",     # 控制操作符
    "$(", "${", "$((" ,       # 命令/变量/算术展开
    "`",                      # 老式命令替换
    "{", "}", "<(", ">(",     # 进程替换
    "~",                      # 家目录展开
))

WIN_SHELL_METACHARS = frozenset((
    "|", ">", ">>", "<",      # 管道 / 重定向
    "&&", "||", "&",          # 控制
    "^", "%", "(", ")",       # cmd 特有：转义 / 变量 / 分组
))


def _platform_shell_metachars() -> frozenset:
    """根据当前平台返回对应的元字符集合。"""
    return WIN_SHELL_METACHARS if sys.platform == "win32" else UNIX_SHELL_METACHARS

def win_deed_shell(command: str) -> bool:
    if not command or not command.strip():
        return False
    return any(sym in command for sym in WIN_SHELL_METACHARS)

def linux_deed_shell(command: str) -> bool:
    """
    判断命令串里是否在「引号 / 转义之外」出现了 shell 元字符。
    只有真正需要 shell 解释的语法才会返回 True。
    """
    if not command or not command.strip():
        return False

    # Unix：用 shlex 做 token 化，只对「引号外」的 token 检查
    try:
        tokens = shlex.split(command)
    except ValueError:
        # 引号未闭合等非法语法，保守地认为需要 shell
        return True

    # 单 token 无特殊符号 → 肯定不需要 shell
    if len(tokens) == 1:
        return False

    # 多 token 时，检查「引号外」是否出现了元字符
    for token in tokens:
        if any(sym in token for sym in UNIX_SHELL_METACHARS):
            return True

    return False


if __name__ == '__main__':
    print(detect_available_shells())
