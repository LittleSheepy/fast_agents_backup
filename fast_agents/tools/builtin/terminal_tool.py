"""安全的命令行工具 - 跨平台优化版

跨平台关键点：
1. 编码：显式指定 encoding="utf-8"，避免 Windows 下系统默认编码（如 GBK）导致乱码
2. shell=True 分支：Windows 上 shell=True 走 cmd.exe，对 Unix 管道/重定向支持有限；
   不含元字符的命令走 shell=False + Windows 安全分词，避免 shlex 吃掉反斜杠
3. 路径沙箱：使用 Path.relative_to() 做越界检查，不依赖字符串拼接，跨平台安全
"""

from __future__ import annotations

import os
import sys
import locale
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Set, Union

from pydantic import BaseModel, Field

from ..tool_base import ToolBase
from ..tool_result import ToolResult

IS_WINDOWS = sys.platform == "win32"

# === 跨平台命令白名单 ===
# Unix 基础命令
_UNIX_READONLY_COMMANDS: Set[str] = {
    "ls", "pwd", "file", "stat", "du", "df",
    "cat", "head", "tail", "less", "more", "wc",
    "find", "grep", "egrep", "fgrep", "rg",
    "sort", "uniq", "cut", "awk", "sed", "diff",
    "echo", "which", "whereis", "whoami",
    "cd",
}

# Windows 特有命令（cmd.exe 内置或常见工具）
_WINDOWS_READONLY_COMMANDS: Set[str] = {
    "dir", "type", "findstr", "where", "tree",
    "more", "copy", "xcopy", "move", "del", "ren",
    "mkdir", "md", "cls", "ver", "date", "time",
    "ping", "ipconfig", "netstat", "tasklist", "powershell"
}

# 可执行命令（需显式开启）
_EXEC_COMMANDS: Set[str] = {"python", "python3", "node", "bash", "sh", "pip"}


def _build_allowed_commands(
    allow_exec: bool = False,
    allowed_commands: Optional[Set[str]] = None,
) -> Set[str]:
    """构建命令白名单，自动包含当前平台的兼容命令。

    跨平台策略：
    - Unix 平台：默认包含 _UNIX_READONLY_COMMANDS
    - Windows 平台：默认包含 _WINDOWS_READONLY_COMMANDS（dir/type/findstr/where/tree/more 等）
    - 两个平台的命令集会叠加，确保 Agent 在任意平台都能使用常见命令
    - 用户可通过 allowed_commands 参数追加自定义命令
    """
    allowed: Set[str] = set()
    if IS_WINDOWS:
        allowed |= _WINDOWS_READONLY_COMMANDS
    else:
        allowed |= _UNIX_READONLY_COMMANDS
    # 另一平台的命令也加入，方便用户在跨平台脚本中混用
    if IS_WINDOWS:
        allowed |= _UNIX_READONLY_COMMANDS
    else:
        allowed |= _WINDOWS_READONLY_COMMANDS
    if allow_exec:
        allowed |= _EXEC_COMMANDS
    if allowed_commands:
        allowed |= allowed_commands
    return allowed


def _split_windows(command: str) -> List[str]:
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

def _split_command(command: str) -> List[str]:
    """跨平台安全的命令分词。

    在 Windows 上，shlex.split() 使用 POSIX 规则，
    反斜杠会被当作转义符吃掉（如 C:\\Users\\a.txt -> C:Usersa.txt）。
    因此 Windows 上对不含 shell 元字符的命令直接按空格分词，
    对含管道/重定向的命令走 shell=True 分支（由 _exec 处理）。
    """
    stripped = command.strip()
    shell_metachars = ("|", ">", "<", "*", "?", "&&", ";")
    needs_shell = any(sym in stripped for sym in shell_metachars)

    if IS_WINDOWS and not needs_shell:
        # Windows 不含元字符的命令：用 split() 避免 shlex 吃掉反斜杠
        # 保留引号内的空格（简单解析双引号包裹的参数）
        return _split_windows(stripped)
    # Unix 或需要 shell 的命令：走标准 shlex
    return shlex.split(stripped)



class TerminalInput(BaseModel):
    command: str = Field(description="要执行的 shell 命令，如 'ls'、'cat README.md'、'findstr \"xxx\" file'")


class TerminalTool(ToolBase):
    name = "terminal"
    description = (
        "执行 shell 命令查看文件与目录。默认支持只读命令（ls/dir, cat/type, head, tail, grep/findstr, find, pwd, wc 等）。"
        "工作目录被限制在指定沙箱内，无法访问上级。跨平台兼容（Linux/macOS/Windows）。"
    )
    args_schema = TerminalInput
    tags = ["builtin", "filesystem"]

    def __init__(self, workspace: str = ".", timeout: int = 30, max_output_chars: int = 20_000,
                 allow_exec: bool = False, allow_cd: bool = True, allowed_commands: Optional[Set[str]] = None):
        super().__init__(max_output_chars=max_output_chars)
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.allow_cd = allow_cd
        self._cwd = self.workspace
        self.allowed_commands: Set[str] = _build_allowed_commands(
            allow_exec=allow_exec, allowed_commands=allowed_commands
        )

    def run(self, command: str) -> Union[str, ToolResult]:
        if not command or not command.strip():
            return ToolResult.fail("命令不能为空")
        try:
            parts = _split_command(command)
        except ValueError as e:
            return ToolResult.fail(f"命令解析失败：{e}")
        if not parts:
            return ToolResult.fail("命令不能为空")
        base = parts[0]
        if base not in self.allowed_commands:
            hint = "（如需启用代码执行，创建 TerminalTool 时传 allow_exec=True）" if base in _EXEC_COMMANDS else ""
            return ToolResult.fail(f"不允许的命令: {base}\n允许的命令: {', '.join(sorted(self.allowed_commands))}{hint}")
        if base == "cd":
            return self._handle_cd(parts)
        return self._exec(parts, command.strip())

    def _handle_cd(self, parts: List[str]) -> ToolResult:
        if not self.allow_cd:
            return ToolResult.fail("cd 命令已禁用")
        target = parts[1] if len(parts) > 1 else str(self.workspace)
        if target == "~":
            new_dir = self.workspace
        elif target.startswith("/") or (IS_WINDOWS and target.startswith("\\")):
            # Unix 绝对路径 或 Windows 绝对路径
            new_dir = (self.workspace / target.lstrip("/").lstrip("\\")).resolve()
        else:
            new_dir = (self._cwd / target).resolve()
        try:
            new_dir.relative_to(self.workspace)
        except ValueError:
            return ToolResult.fail(f"不允许访问工作目录外的路径: {new_dir}")
        if not new_dir.exists():
            return ToolResult.fail(f"目录不存在: {new_dir}")
        if not new_dir.is_dir():
            return ToolResult.fail(f"不是目录: {new_dir}")
        self._cwd = new_dir
        return ToolResult.ok(f"✅ 当前目录: {self._cwd.relative_to(self.workspace)}")

    def _exec(self, parts: List[str], raw_command: str) -> ToolResult:
        """执行命令"""
        shell_metachars = ("|", ">", "<", "*", "?", "&", ";", "$", "`", "{", "(", "^", "%")
        needs_shell = any(sym in raw_command for sym in shell_metachars)

        argv_or_cmd = raw_command if needs_shell else parts
        enc = locale.getpreferredencoding(False) or "utf-8"
        opts = {
            "capture_output": True,
            "text": True,
            "encoding": enc,
            # "errors": "replace",            # error 不会蹦
            "timeout": self.timeout,
            "stdin": subprocess.DEVNULL,    # 防止子进程等待输入
        }
        try:
            proc = subprocess.run(
                argv_or_cmd,
                shell=needs_shell,  # True→raw_command+shell；False→parts+argv
                cwd=str(self._cwd),
                capture_output=True, text=True, timeout=self.timeout,
                encoding="utf-8",
                env={**os.environ, "HOME": str(self.workspace)},
            )
        except subprocess.TimeoutExpired:
            return ToolResult.fail(f"命令执行超时（> {self.timeout}s）")
        except FileNotFoundError:
            # 仅 shell=False(argv) 路径会抛；shell=True 时找不到命令由 shell 内部报错，不会走到这里
            return ToolResult.fail(f"命令未找到：{parts[0]}，请确认已安装")
        except Exception as e:
            return ToolResult.fail(f"执行失败：{e}")

        output = proc.stdout
        if proc.stderr:
            output += f"\n[stderr]\n{proc.stderr}"
        if proc.returncode != 0:
            output = f"[返回码 {proc.returncode}]\n{output}"
        return ToolResult.ok(output or "（命令执行成功，无输出）")

    def reset(self) -> None:
        self._cwd = self.workspace
