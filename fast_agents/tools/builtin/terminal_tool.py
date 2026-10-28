"""安全的Windows命令行工具
fast_agents/tools/builtin/terminal_tool.py

- 拼接的多条命令拆分后每条独立做安全校验
- 自动探测cmd/powershell，智能判断shell=True需求
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Set, Union

from pydantic import BaseModel, Field

from ..tool_base import ToolBase
from ..tool_result import ToolResult
# 按功能拆分的模块导入
from .shell_check import need_shell
from .shell_detector import detect_shell, route_and_run
from .command_splitter import (
    split_windows,
    split_windows_commands,
    validate_command_safety,
)


# === Windows 命令白名单 ===
# 只读安全命令
# 去掉的 命令 "copy", "xcopy", "move", "del", "ren", mkdir, "md"
_WINDOWS_READONLY_COMMANDS: Set[str] = {
    "dir", "type", "findstr", "where", "tree",
    "more", "echo", "cd", "cls", "ver", "date", "time",
    "ping", "ipconfig", "netstat", "tasklist",
    "powershell", "cmd",  # 允许显式调用shell，但参数会被安全校验
}

# 可执行命令（需显式开启allow_exec）
_EXEC_COMMANDS: Set[str] = {"python", "python3", "node", "pip", "npm", "git"}


def _build_allowed_commands(
    allow_exec: bool = False,
    allowed_commands: Optional[Set[str]] = None,
) -> Set[str]:
    """构建命令白名单"""
    allowed: Set[str] = set(_WINDOWS_READONLY_COMMANDS)
    if allow_exec:
        allowed |= _EXEC_COMMANDS
    if allowed_commands:
        allowed |= allowed_commands
    return allowed











class TerminalInput(BaseModel):
    command: str = Field(description="要执行的 shell 命令，如 'dir'、'type README.md'、'findstr \"xxx\" file'")


class TerminalTool(ToolBase):
    name = "terminal"
    description = (
        "执行Windows shell命令查看文件与目录。默认支持只读命令（dir, type, findstr, where, tree等）。"
        "工作目录被限制在指定沙箱内，无法访问上级，自动防止命令注入。"
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
        # 初始化时探测一次shell
        self.detected_shell, self.shell_path = detect_shell()

    def run(self, command: str) -> Union[str, ToolResult]:
        """执行命令入口"""
        if not command or not command.strip():
            return ToolResult.fail("命令不能为空")
        # ========== 新增：多命令拆分，每条命令独立校验安全性 ==========
        raw_command = command.strip()
        sub_commands = split_windows_commands(raw_command)
        if not sub_commands:
            return ToolResult.fail("命令不能为空")

        # 逐条校验每条命令的安全性
        for sub_cmd_str in sub_commands:
            # 分词
            try:
                parts = split_windows(sub_cmd_str)
            except ValueError as e:
                return ToolResult.fail(f"命令解析失败：{e}")
            if not parts:
                continue
            
            # 1. 白名单校验
            base = parts[0].lower().strip('"')
            if base not in self.allowed_commands:
                hint = "（如需启用代码执行，创建 TerminalTool 时传 allow_exec=True）" if base in _EXEC_COMMANDS else ""
                return ToolResult.fail(f"不允许的命令: {base}\n允许的命令: {', '.join(sorted(self.allowed_commands))}{hint}")
            
            # 2. 危险参数校验
            safe, err_msg = validate_command_safety(parts)
            if not safe:
                return ToolResult.fail(f"安全校验失败: {err_msg}")

        # 处理cd命令（cd不会和其他命令拼接执行，单独处理）
        first_parts = split_windows(sub_commands[0])
        if first_parts and first_parts[0].lower() == "cd":
            return self._handle_cd(first_parts)

        # 执行命令
        return self._exec(raw_command)

    def _handle_cd(self, parts: List[str]) -> ToolResult:
        """处理cd命令，微调：更严格的路径越界检查"""
        if not self.allow_cd:
            return ToolResult.fail("cd 命令已禁用")
        target = parts[1] if len(parts) > 1 else str(self.workspace)
        
        # 处理引号
        target = target.strip('"')
        
        if target == "~":
            new_dir = self.workspace
        elif (len(target) >= 2 and target[1] == ":") or target.startswith("\\\\"):
            # Windows 绝对路径（C:\ 开头 或 UNC 路径）
            new_dir = Path(target).resolve()
        else:
            new_dir = (self._cwd / target).resolve()
        
        # 严格路径沙箱检查
        try:
            new_dir.relative_to(self.workspace)
        except ValueError:
            return ToolResult.fail(f"不允许访问工作目录外的路径: {new_dir}")
        
        if not new_dir.exists():
            return ToolResult.fail(f"目录不存在: {new_dir}")
        if not new_dir.is_dir():
            return ToolResult.fail(f"不是目录: {new_dir}")
        
        self._cwd = new_dir
        try:
            rel_path = new_dir.relative_to(self.workspace)
            return ToolResult.ok(f"✅ 当前目录: {rel_path}")
        except ValueError:
            return ToolResult.ok(f"✅ 当前目录: {new_dir}")

    def _exec(self, raw_command: str) -> ToolResult:
        """执行命令，微调：使用新的need_shell智能判断，自动路由shell，每条命令已提前校验"""
        # ========== 新增：智能判断是否需要shell=True ==========
        needs_shell_flag = need_shell(raw_command)

        try:
            rc, output, err, used_shell = route_and_run(
                raw_command,
                cwd=str(self._cwd),
                timeout=self.timeout,
                needs_shell=needs_shell_flag,
                env={**os.environ, "USERPROFILE": str(self.workspace)},
            )
        except subprocess.TimeoutExpired:
            return ToolResult.fail(f"命令执行超时（> {self.timeout}s）")
        except FileNotFoundError:
            parts = split_windows(raw_command.split("&&")[0].split("||")[0].split("|")[0].strip())
            return ToolResult.fail(f"命令未找到：{parts[0] if parts else ''}，请确认已安装")
        except Exception as e:
            return ToolResult.fail(f"执行失败：{e}")

        # 组装输出
        full_output = output
        if err:
            full_output += f"\n[stderr]\n{err}"
        if rc != 0:
            full_output = f"[返回码 {rc}]\n{full_output}"
        
        # 截断过长输出
        if len(full_output) > self.max_output_chars:
            full_output = full_output[:self.max_output_chars] + f"\n... (输出过长，已截断前{self.max_output_chars}字符)"
        
        return ToolResult.ok(full_output or "（命令执行成功，无输出）")

    def reset(self) -> None:
        self._cwd = self.workspace


if __name__ == "__main__":
    # 简单测试
    tool = TerminalTool(workspace="./test_workspace", allow_exec=True)
    print(f"检测到Shell: {tool.detected_shell} -> {tool.shell_path}")
    
    test_cmds = [
        "dir",
        "echo hello && echo world",
        "echo \"a && b\"",
        "dir | findstr .py",
        "del /s /q C:\\Windows",  # 应该被安全校验拦截
    ]
    
    for cmd in test_cmds:
        print(f"\n>>> 执行: {cmd}")
        res = tool.run(cmd)
        print(f"{'✅' if res.success else '❌'} {res.content[:200]}")
