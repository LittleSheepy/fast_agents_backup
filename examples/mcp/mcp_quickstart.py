"""MCP 模块快速上手 —— 用 MockLLM 演示三种接入方式（无需真实 API key）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_agents import ReActAgent, MCPRegistry
from fast_agents.messages.message import Message, MsgToolCall
from fast_agents.tools.builtin.mcp_tool import build_mcp_tools
from fast_agents.tools import builtin


class MockLLM:
    """按预设脚本返回 Message，离线演示 MCP 工具调用链路。

    脚本每一项可以是：
    - str：返回一条 assistant 文本消息（终止调用）
    - list[{"name", "arguments"}]：返回带 tool_calls 的 assistant 消息
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.idx = 0
        self.calls = []

    _tc_seq = 0

    def invoke(self, messages, tools=None, **kwargs):
        self.calls.append(list(messages))
        step = self.script[self.idx]
        self.idx += 1
        if isinstance(step, str):
            return Message("assistant", step)
        tool_calls = []
        for i, t in enumerate(step):
            type(self)._tc_seq += 1
            tool_calls.append(MsgToolCall(
                id=f"call_mock_{type(self)._tc_seq}",
                name=t["name"],
                arguments=json.dumps(t["arguments"], ensure_ascii=False),
            ))
        return Message("assistant", "", tool_calls=tool_calls)

    def stream_invoke(self, messages, tools=None, **kwargs):
        m = self.invoke(messages, tools=tools, **kwargs)
        return m.content or ""


def example_1_expand():
    """示例 1：细粒度展开（生产推荐）—— 每个 MCP 工具独立注册，参数 schema 完整。"""
    print("\n========== 示例 1：细粒度展开 ==========")
    mcp = MCPRegistry()
    py = sys.executable
    server = str(Path(__file__).parent / "echo_server.py")
    mcp.add_stdio_cmd("demo", f'"{py}" "{server}"')
    mcp.connect_all()

    agent = ReActAgent(
        name="MCP助手",
        llm=MockLLM([
            [{"name": "demo__calculate", "arguments": {"expression": "3 * 4 + 1"}}],
            [{"name": "demo__echo", "arguments": {"message": "你好！", "prefix": "[MCP] ", "repeat": 2}}],
            "结果：计算与回显都通过了 MCP 服务器完成。",
        ]),
        enable_memory=False,
        enable_skills=False,
    )
    mcp.register_into(agent.tool_registry)
    agent.refresh_system_prompt()
    print("已注册工具：", agent.tool_registry.list_names())
    result = agent.run("请用 MCP 工具算 3*4+1，再回显两次「你好！」，前缀 [MCP]。")
    # run() 返回最后一条消息
    print("最终回答：", result)
    mcp.close_all()


def example_2_mixed():
    """示例 2：本地工具 + MCP 工具混合使用。"""
    print("\n========== 示例 2：本地 + MCP 混合 ==========")
    mcp = MCPRegistry()
    py = sys.executable
    server = str(Path(__file__).parent / "echo_server.py")
    mcp.add_stdio_cmd("demo", f'"{py}" "{server}"')
    mcp.connect_all()

    agent = ReActAgent(
        name="混合助手",
        llm=MockLLM([
            [{"name": "python_calculator", "arguments": {"expression": "2 ** 10"}}],
            [{"name": "demo__calculate", "arguments": {"expression": "sqrt(2)"}}],
            "本地算出 2^10=1024，远程算出 sqrt(2)≈1.414。",
        ]),
        enable_memory=False,
        enable_skills=False,
    )
    agent.add_tool(builtin.CalculatorTool())
    mcp.register_into(agent.tool_registry)
    agent.refresh_system_prompt()
    print("已注册工具：", agent.tool_registry.list_names())
    result = agent.run("算 2 的 10 次方，再用 MCP 算根号 2。")
    print("最终回答：", result)
    mcp.close_all()


def example_3_builtin_tools():
    """示例 3：启用 MCP 通用管理工具（list/connect/disconnect），模型能查看连接状态。"""
    print("\n========== 示例 3：MCP 通用管理工具 ==========")
    mcp = MCPRegistry()
    agent = ReActAgent(
        name="动态助手",
        llm=MockLLM([
            [{"name": "list_mcp_servers", "arguments": {}}],
            "已通过 list_mcp_servers 确认：当前没有已连接的 MCP 服务器。",
        ]),
        enable_memory=False,
        enable_skills=False,
    )
    for t in build_mcp_tools(mcp, agent.tool_registry, refresh_prompt=agent.refresh_system_prompt):
        agent.add_tool(t)
    agent.refresh_system_prompt()
    print("已注册工具：", agent.tool_registry.list_names())
    result = agent.run("当前有哪些 MCP 服务器？")
    print("最终回答：", result)
    mcp.close_all()


def template_real_world():
    """真实项目接入模板（注释状态；替换 API key 后可运行）。

        from fast_agents import LLM, ReActAgent, MCPRegistry
        from fast_agents.tools.builtin.mcp_tool import build_mcp_tools
        from fast_agents.tools import builtin

        llm = LLM(model="...", api_key="...", base_url="...")

        mcp = MCPRegistry()
        mcp.add_stdio("fs", "npx", args=["-y", "@modelcontextprotocol/server-filesystem", r"D:\00sheepy"])

        agent = ReActAgent(name="助手", llm=llm, enable_memory=True, enable_skills=True)
        agent.add_tool(builtin.CalculatorTool())
        agent.add_tool(builtin.TerminalTool())

        mcp.connect_all()
        mcp.register_into(agent.tool_registry)
        agent.refresh_system_prompt()

        # 可选：让模型能在运行时 list/connect/disconnect 管理 MCP server
        for t in build_mcp_tools(mcp, agent.tool_registry, refresh_prompt=agent.refresh_system_prompt):
            agent.add_tool(t)
        agent.refresh_system_prompt()

        agent.run("帮我看看 D:/00sheepy 下有什么文件")
        mcp.close_all()
    """


if __name__ == "__main__":
    example_1_expand()
    example_2_mixed()
    example_3_builtin_tools()
    print("\n✅ MCP quickstart 全部演示完成。")
