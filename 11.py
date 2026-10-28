from sympy import false

from fast_agents.tools.builtin import ReadTool
from fast_agents.utils.win_shell_detector import run_command

import subprocess
def main_ReadTool():
    workspace = "."
    path = r"D:\00sheepy\01code\02alg_code_bjb\agent\fast_agents\tmp/hello.txt"
    read_tool = ReadTool(workspace)
    result = read_tool.run(path)
    print(result)

def main_run_command():
    import locale

    enc = locale.getpreferredencoding(False) or "utf-8"
    opts = {
        "capture_output": True,
        "text": True,
        "encoding": enc,
        "errors": "replace",
        "timeout": 20,
        "stdin": subprocess.DEVNULL,    # 防止子进程等待输入
    }
    # cmd = ["cmd", "/c", "dir", "C:/Program Files", "&&", "echo", "a"]
    cmd = ["cmd", "/c", "dir", "C:/Program Files", "&&", "echo", "a"]
    proc = subprocess.run(cmd,
                          shell=False,
                          **opts)
    print(proc)





if __name__ == '__main__':
    main_run_command()