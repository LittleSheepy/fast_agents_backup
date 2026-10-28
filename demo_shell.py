
from fast_agents.tools.builtin import TerminalTool


cmd_command = [
    "dir | findstr 'txt'",
]

def main_run_command():
    # command = "dir | findstr 'txt'"
    command = "dir"
    tool = TerminalTool()
    result = tool.run(command)
    print(result)




if __name__ == '__main__':
    main_run_command()