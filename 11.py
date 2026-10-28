from fast_agents.tools.builtin import ReadTool

def main(workspace, path):
    read_tool = ReadTool(workspace)
    result = read_tool.run(path)
    print(result)


if __name__ == '__main__':
    workspace = "."
    path = r"D:\00sheepy\01code\02alg_code_bjb\agent\fast_agents\tmp/hello.txt"
    main(workspace, path)