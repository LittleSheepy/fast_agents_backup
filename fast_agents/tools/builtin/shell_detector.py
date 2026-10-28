"""win_shell_detector.py

Shell 探测与路由模块：
1. 优先用 shellingham 探测当前终端环境；
2. 再用命令文本特征判断 CMD / PowerShell；
3. 最后用 subprocess 试跑兜底。
"""

import os
import re
import subprocess
from typing import Literal, Tuple
import shutil

ShellType = Literal["cmd", "powershell", "bash", "unknown"]

# 检测父进程是 cmd 还是 Power Shell
def detect_parent_shell() -> Tuple[ShellType, str]:
    # 兜底：通过父进程判断
    try:
        import psutil

        parent = psutil.Process(os.getpid()).parent()
        if parent is not None:
            pname = parent.name().lower()
            exe_path = parent.exe().lower() if hasattr(parent, "exe") else pname
            if "powershell" in pname or "pwsh" in pname:
                return "powershell", exe_path
            if pname == "cmd.exe" or "cmd.exe" in exe_path:
                return "cmd", exe_path
            if "bash" in pname or "sh" in pname:
                return "bash", exe_path
    except Exception:
        pass
    return "unknown", ""

def detect_shell_by_shellingham() -> Tuple[ShellType, str]:
    try:
        import shellingham
        name, path = shellingham.detect_shell()
        name_lower = name.lower()
        if name_lower in ("powershell", "pwsh"):
            return "powershell", path
        if name_lower == "cmd":
            return "cmd", path
        if name_lower in ("bash", "sh", "zsh", "fish"):
            return "bash", path
        return "unknown", path
    except ImportError:
        pass
    except Exception:
        pass
    return "unknown", ""

def detect_available_shells() -> dict[str, bool]:
    return {
        "cmd": bool(shutil.which("cmd.exe")),
        "powershell": bool(shutil.which("pwsh") or shutil.which("powershell.exe")),
    }

def detect_shell_from_env() -> Tuple[ShellType, str]:
    powershell_path = shutil.which("pwsh") or shutil.which("powershell.exe")
    if powershell_path and "PSMODULEPATH" in os.environ:
        return "powershell", powershell_path
    comspec = os.environ.get("COMSPEC", "").lower()
    if "powershell" in comspec:
        return "powershell", comspec

    cmd_path = shutil.which("cmd.exe")
    if cmd_path and "cmd.exe" in comspec:
        return "cmd", cmd_path

    return "unknown", ""



def detect_shell() -> Tuple[ShellType, str]:
    """探测当前 Shell。"""
    # 通过 shellingham 判断
    name, path = detect_shell_by_shellingham()
    if name != "unknown":
        return name, path

    # 兜底：通过父进程判断
    name, path = detect_parent_shell()
    if name != "unknown":
        return name, path

    # 再兜底：看环境变量
    name, path = detect_shell_from_env()
    if name != "unknown":
        return name, path

    return name, path

def guess_shell_from_command(command: str) -> ShellType:
    """根据命令文本特征猜测 CMD / PowerShell。"""
    cmd = command.strip()
    if not cmd:
        return "unknown"

    ps_score = 0
    cmd_score = 0

    # PowerShell 强特征
    if re.search(r'\$[A-Za-z_\{]', cmd):
        ps_score += 3
    if re.search(r'\$env:', cmd, re.I):
        ps_score += 4
    if re.search(r'\bGet-|Set-|New-|Remove-|Invoke-|Test-|Stop-|Start-|Restart-', cmd):
        ps_score += 3
    if re.search(r'\bWhere-Object\b|\bForEach-Object\b|\bSelect-String\b', cmd):
        ps_score += 2
    if re.search(r'\bImport-Module\b|\bGet-Module\b|\bSet-ExecutionPolicy\b', cmd):
        ps_score += 2
    if re.search(r'--%(\s|$)', cmd):
        ps_score += 3
    if re.search(r'`[nt"\'$]', cmd):
        ps_score += 2

    # CMD 强特征
    if re.search(r'%[A-Za-z_][A-Za-z0-9_]*%', cmd):
        cmd_score += 3
    if re.search(r'\bset\s+\w+=', cmd, re.I):
        cmd_score += 2
    if re.search(r'\bsetlocal\b|\bendlocal\b|\benabledelayedexpansion\b', cmd, re.I):
        cmd_score += 3
    if re.search(r'\bgoto\s+', cmd, re.I):
        cmd_score += 3
    if re.search(r'\b@echo\s+off\b', cmd, re.I):
        cmd_score += 3
    if re.search(r'\bcall\s+', cmd, re.I):
        cmd_score += 2

    # 弱特征
    if re.search(r'\btype\b|\bfindstr\b', cmd):
        cmd_score += 1
    if re.search(r'\bGet-ChildItem\b|\bSet-Location\b|\bGet-Content\b', cmd):
        ps_score += 1

    # 短命令别名弱特征
    ps_aliases = {"pwd", "ls", "cat", "rm", "cp", "mv", "ps", "touch", "which", "mkdir", "rmdir"}
    cmd_builtins = {"dir", "copy", "del", "ren", "move", "findstr", "cls", "echo", "cd", "md", "rd"}

    first_token = cmd.split()[0].lower()

    if first_token in ps_aliases and first_token not in cmd_builtins:
        ps_score += 1
    elif first_token in cmd_builtins and first_token not in ps_aliases:
        cmd_score += 1

    if ps_score > cmd_score + 1:
        return "powershell"
    if cmd_score > ps_score + 1:
        return "cmd"
    return "unknown"

def run_command(
    command: str,
    shell_type: ShellType = "auto",
    timeout: int = 10,
    **kwargs,                       # 透传给 subprocess.run 的额外参数
):
    """执行命令，自动选择 CMD / PowerShell。

    Args:
        command: 要执行的命令字符串。
        shell_type: "auto" / "cmd" / "powershell" / "bash"。
        timeout: 超时秒数（可被 kwargs["timeout"] 覆盖）。
        **kwargs: 其余参数原样透传给 subprocess.run，
                  如 cwd / env / stdin / check / creationflags 等。
                  kwargs 中显式指定的值会覆盖本函数设置的默认值。

    Returns:
        (returncode, stdout, stderr)
    """

    if shell_type == "auto":
        guessed = guess_shell_from_command(command)
        if guessed in ("cmd", "powershell"):
            effective_shell = guessed
        else:
            detected_shell, _ = detect_shell() if shell_type == "auto" else (shell_type, "")
            effective_shell = detected_shell if detected_shell in ("cmd", "powershell") else "powershell"
    else:
        effective_shell = shell_type



    # 1) 组装 subprocess.run 的默认参数
    opts = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
        "stdin": subprocess.DEVNULL,    # 防止子进程等待输入
    }

    # 2) 用调用方传入的 kwargs 覆盖默认值（kwargs 优先级更高）
    opts.update(kwargs)

    # 3) 按目标 shell 构造 argv，再统一把 opts 透传给 subprocess.run
    if effective_shell == "cmd":
        argv_or_cmd = ["cmd", "/c", command]
    elif effective_shell == "powershell":
        argv_or_cmd = ["powershell", "-NoProfile", "-Command", command]
    else:
        # bash / 其他：交给系统 shell 解析
        argv_or_cmd = command
        opts.setdefault("shell", True)   # 字符串命令需要 shell=True；kwargs 已给则尊重 kwargs

    return subprocess.run(argv_or_cmd, **opts)


def route_and_run(command: str, timeout: int = 10) -> Tuple[int, str, str, ShellType]:
    """路由并执行：先试首选 Shell，失败则回退到另一个。"""
    detected_shell, _ = detect_shell()
    guessed = guess_shell_from_command(command)

    # 确定首选 Shell
    if guessed in ("cmd", "powershell"):
        preferred = guessed
    elif detected_shell in ("cmd", "powershell"):
        preferred = detected_shell
    else:
        preferred = "powershell"

    fallback = "cmd" if preferred == "powershell" else "powershell"

    proc = run_command(command, shell_type=preferred, timeout=timeout)
    rc, out, err = proc.returncode, proc.stdout, proc.stderr
    if rc == 0 or not err.strip():
        return rc, out, err, preferred

    # 回退
    proc = run_command(command, shell_type=fallback, timeout=timeout)
    rc2, out2, err2 = proc.returncode, proc.stdout, proc.stderr
    if rc2 == 0 or not err2.strip():
        return rc2, out2, err2, fallback

    # 都失败，返回首选的结果
    return rc, out, err, preferred


if __name__ == "__main__":
    shell_name, shell_path = detect_shell()
    print(f"Detected shell: {shell_name} -> {shell_path}")

    test_commands = [
        "echo %USERNAME%",
        "echo $env:USERNAME",
        "Get-Process | Select-Object -First 3",
        "dir",
    ]

    for cmd in test_commands:
        guessed = guess_shell_from_command(cmd)
        rc, out, err, used = route_and_run(cmd)
        print(f"\nCommand : {cmd}")
        print(f"Guessed : {guessed}")
        print(f"Used    : {used}")
        print(f"RC      : {rc}")
        print(f"Output  : {out.strip()}")
        if err.strip():
            print(f"Error   : {err.strip()}")