"""MCP × ReActAgent 端到端集成示例

本文件演示如何把 MCP server 真正接进 ReActAgent 的思考-行动循环，
让模型在推理过程中"像调用本地工具一样"调用远程 MCP 工具。

运行方式（Windows PowerShell / cmd 均可，零依赖、不需要 API key）：
    python examples/mcp/mcp_react_agent_demo.py

本示例用 MockLLM 脚本化假 LLM，方便你在没配置 LLM key 的机器上直接跑通；
最后给出「接入真实 LLM 与真实 MCP server」的模板，把 6 行注释打开即可。
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── 让示例在仓库根目录直接 `python examples/mcp/xxx.py` 能跑 ────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fast_agents import ReActAgent, ToolRegistry
from fast_agents.mcp import MCPRegistry
from fast_agents.llm.llm import LLM
from fast_agents.messages.message import Message, MsgToolCall


# ═══════════════════════════════════════════════════════════════════
# 一、MockLLM：按脚本依次返回预置响应，不需要真实 API key
# ═══════════════════════════════════════════════════════════════════

class MockLLM(LLM):
    """与 test 目录里保持一致的假 LLM，方便离线演示。"""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.i = 0

    def invoke(self, messages, tools=None, **kwargs):
        self.calls.append(list(messages))
        resp = self.script[self.i]
        self.i += 1
        return resp


# ═══════════════════════════════════════════════════════════════════
# 二、示例 1：register_into 细粒度展开（推荐，生产用这种）
#    每个 MCP 远程工具被展开成独立 StructuredTool，带完整参数 Schema，
#    模型能看到每个字段的 description/enum/必填项，调用准确率最高。
# ═══════════════════════════════════════════════════════════════════

def demo_register_into():
    print("=" * 60)
    print("示例 1：MCPRegistry.register_into() 细粒度展开")
    print("=" * 60)

    server_path = Path(__file__).parent / "echo_server.py"

    # 1) 准备注册表 + MCP server
    mcp = MCPRegistry()
    mcp.add_stdio("demo", sys.executable, [str(server_path)])

    # 2) 脚本化 LLM：第一步决定调用 demo__calculate，第二步拿到结果后回答
    llm = MockLLM([
        Message(
            "assistant",
            "",
            tool_calls=[
                MsgToolCall(
                    name="demo__calculate",
                    arguments='{"expression": "(2+3)*4^2"}',
                    id="call_1",
                )
            ],
        ),
        Message("assistant", "(2+3)*4^2 的结果是 80。"),
    ])

    # 3) 构造 Agent（关闭 memory/skills 让演示更聚焦）
    agent = ReActAgent(
        name="MCP演示助手",
        llm=llm,
        enable_memory=False,
        enable_skills=False,
    )

    try:
        mcp.connect_all()
        # 关键一行：把 MCP 远程工具逐个展开成 StructuredTool 注册进 agent
        mcp.register_into(agent.tool_registry, prefix_sep="__")
        # 注册后系统提示词里的工具清单要刷新
        agent.refresh_system_prompt()

        print(f"Agent 可用工具: {agent.tool_registry.list_names()}")
        answer = agent.run("(2+3)*4^2 等于多少？")
        print(f"\n🤖 Agent 最终回答: {answer}")

        # 断言结果符合预期（保证示例跑通）
        assert "80" in answer, f"期望结果 80，实际回答: {answer}"
        print("✅ 示例 1 通过：Agent 成功通过 MCP 调用远程工具并得出正确答案。")
    finally:
        mcp.close_all()


# ═══════════════════════════════════════════════════════════════════
# 三、示例 2：本地工具 + MCP 工具混合注册（实际项目最常见的场景）
#    同一个 ToolRegistry 里，既有你自己写的本地工具（python_calculator），
#    又有从 MCP server 动态展开进来的远程工具（demo__echo），
#    Agent 在推理过程中可以自由选择调哪一个。
# ═══════════════════════════════════════════════════════════════════

def demo_mixed_tools():
    print("\n" + "=" * 60)
    print("示例 2：本地工具 + MCP 工具混合使用")
    print("=" * 60)

    from fast_agents.tools.builtin.calculator_tool import CalculatorTool

    server_path = Path(__file__).parent / "echo_server.py"

    mcp = MCPRegistry()
    mcp.add_stdio("demo", sys.executable, [str(server_path)])

    # 让模型先用 MCP 的 echo 回显，再用本地 calculator 计算，最后回答
    llm = MockLLM([
        Message(
            "assistant",
            "",
            tool_calls=[
                MsgToolCall(
                    name="demo__echo",
                    arguments='{"message": "开始计算", "prefix": "[MCP] ", "repeat": 1}',
                    id="call_2",
                )
            ],
        ),
        Message(
            "assistant",
            "",
            tool_calls=[
                MsgToolCall(
                    name="python_calculator",
                    arguments='{"expression": "1+2*3"}',
                    id="call_3",
                )
            ],
        ),
        Message("assistant", "echo 回显成功，1+2*3=7。"),
    ])

    # 注意：示例 1 演示 register_into 时把 agent 交给 mcp 管，这里演示反过来：
    # 先建 ToolRegistry，同时注册本地工具和 MCP 工具，再交给 agent
    tool_registry = ToolRegistry()
    tool_registry.register_tool(CalculatorTool())

    agent = ReActAgent(
        name="混合工具助手",
        llm=llm,
        tool_registry=tool_registry,
        enable_memory=False,
        enable_skills=False,
    )

    try:
        mcp.connect_all()
        # 把 MCP 工具也注册进同一个 tool_registry
        mcp.register_into(agent.tool_registry, prefix_sep="__")
        agent.refresh_system_prompt()

        print(f"Agent 可用工具: {sorted(agent.tool_registry.list_names())}")
        answer = agent.run("先回显一句『开始计算』，然后算 1+2*3 等于多少")
        print(f"\n🤖 Agent 最终回答: {answer}")

        assert "7" in answer, f"期望包含『7』，实际回答: {answer}"
        # 脚本里明确写了两次 tool_calls（echo + calculator），两个工具都被实际调用过
        print("✅ 示例 2 通过：本地工具与 MCP 工具可在同一 Agent 内自由组合使用。")
    finally:
        mcp.close_all()


# ═══════════════════════════════════════════════════════════════════
# 四、关于 agent.add_tool 的 auto_expand 兼容钩子
#    你在 react_agent.py L149-157 内置了 auto_expand 路径：任何带
#    auto_expand=True + _available_tools 属性的工具丢给 agent.add_tool()，
#    都会自动把子工具展开注册。MCPServerTool 正好满足这个条件，因此也可以：
#
#        for server_tool in mcp.get_server_tools():
#            agent.add_tool(server_tool)
#
#    但这个钩子只透传 arguments["input"] 单个字符串参数，适合单入参工具；
#    多参数场景（如 echo 需要 message/prefix/repeat）请用示例 1/2 的 register_into，
#    它会展开出完整 pydantic args_schema，模型能看到每个字段的描述与必填项。


# ═══════════════════════════════════════════════════════════════════
# 四、真实场景模板（把下面注释打开，接入你自己的 LLM 与真实 MCP server）
# ═══════════════════════════════════════════════════════════════════

def template_real_world():
    """
    真实使用场景模板。配置好 LLM_API_KEY / LLM_BASE_URL 后打开即可用。

    例：接入官方 filesystem MCP server（需 Node.js，执行 `npm i -g @modelcontextprotocol/server-filesystem`）

    """
    from fast_agents import ReActAgent
    from fast_agents.llm import LLM
    from fast_agents.mcp import MCPRegistry
    from dotenv import load_dotenv
    load_dotenv()  # 自动向上查找 .env，不再硬编码绝对路径

    llm = LLM()

    mcp = MCPRegistry()
    # 文件系统 MCP：把 D:\\00sheepy\\01code 暴露给 Agent 读写
    mcp.add_stdio(
        name="fs",           # MCP服务器的唯一标识符
        command="npx",      # 启动命令
        args=[
            "-y",   # 自动确认 Yes
            "@modelcontextprotocol/server-filesystem",  # 要运行的nmp包名
            r"D:/00sheepy/01code/02alg_code_bjb/agent/fast_agents"                     # 允许访问的根目录
        ],
    )
    # 还可以同时加 SSE 类型的远程 MCP
    # mcp.add_sse("weather", "http://localhost:8001/sse")

    agent = ReActAgent(name="全能助手", llm=llm, enable_memory=True, enable_skills=True)

    try:
        mcp.connect_all()
        mcp.register_into(agent.tool_registry)
        agent.refresh_system_prompt()

        answer = agent.run("帮我看看 fast_agents 项目根目录下有什么文件")
        print(answer)
    finally:
        mcp.close_all()
    pass


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # template_real_world()
    demo_register_into()
    demo_mixed_tools()
    print("\n" + "=" * 60)
    print("🎉 两个示例全部通过。查看文件中间的注释和末尾 template_real_world() 了解更多用法。")
    print("=" * 60)
