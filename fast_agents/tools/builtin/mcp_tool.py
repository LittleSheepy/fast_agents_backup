"""MCP 通用工具 —— 让 Agent 能在运行时发现、连接、查看 MCP 服务器。

提供三个工具（对齐 memory_tool.py 的三工具模式）：
- list_mcp_servers: 列出当前已注册 / 已连接的 MCP server 及其中可用工具
- connect_mcp_server: 运行时动态连接一个 MCP server（stdio 命令串或 SSE URL）
- disconnect_mcp_server: 断开一个已连接的 MCP server

设计原则：
- 三个工具都是「管理类」工具，不直接代理远端工具调用；真正的远端工具调用
  在细粒度展开后作为独立工具出现在工具表里，模型直接点名调用即可；
- 构造时注入 MCPRegistry 和目标 ToolRegistry（用于 connect 后自动展开），
  不直接依赖 ReActAgent。

使用方式：
    from fast_agents.mcp import MCPRegistry
    from fast_agents.tools.builtin.mcp_tool import build_mcp_tools
    mcp = MCPRegistry()
    agent.add_tools(build_mcp_tools(mcp, agent.tool_registry, refresh_prompt=agent.refresh_system_prompt))
"""

from __future__ import annotations

from typing import Callable, List, Optional, Union

from pydantic import BaseModel, Field

from ..tool_base import ToolBase
from ..tool_result import ToolResult
from ...mcp import MCPRegistry


class ListMcpInput(BaseModel):
    pass  # 无参数


class ConnectMcpInput(BaseModel):
    name: str = Field(description="给这个 MCP server 起一个别名，后续断开/查询用")
    target: str = Field(description="连接目标：stdio 命令串（如 'uvx mcp-server-filesystem D:/code'）或 SSE URL（http(s):// 开头）")
    transport: str = Field(default="auto", description="传输类型：auto（自动判断，URL 开头用 SSE）/ stdio / sse")


class DisconnectMcpInput(BaseModel):
    name: str = Field(description="要断开的 MCP server 别名")


class ListMcpServersTool(ToolBase):
    name = "list_mcp_servers"
    description = "列出当前已注册/已连接的 MCP 服务器及其提供的工具，不确定有哪些外部能力时调用。"
    args_schema = ListMcpInput
    tags = ["mcp"]

    def __init__(self, registry: MCPRegistry):
        super().__init__()
        self.registry = registry

    def run(self) -> Union[str, ToolResult]:
        clients = self.registry.clients
        if not clients:
            return ToolResult.ok("当前没有已连接的 MCP 服务器。可使用 connect_mcp_server 动态添加。")
        lines = []
        for name, client in clients.items():
            info = client.initialize_result
            server_info = info.server_info if info else {}
            tools = self.registry.list_tools(name).get(name, [])
            tool_names = ", ".join(t.name for t in tools) if tools else "(无)"
            lines.append(f"- **{name}** ({server_info.get('name', 'unknown')} v{server_info.get('version', '?')})")
            lines.append(f"  工具：{tool_names}")
        return ToolResult.ok("\n".join(lines))


class ConnectMcpServerTool(ToolBase):
    name = "connect_mcp_server"
    description = "运行时动态连接一个 MCP 服务器。连接成功后，其提供的工具会自动展开到工具表。"
    args_schema = ConnectMcpInput
    tags = ["mcp"]

    def __init__(self, registry: MCPRegistry, tool_registry, refresh_prompt: Optional[Callable[[], None]] = None):
        super().__init__()
        self.registry = registry
        self.tool_registry = tool_registry
        self.refresh_prompt = refresh_prompt

    def run(self, name: str, target: str, transport: str = "auto") -> Union[str, ToolResult]:
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            return ToolResult.fail("name 只能包含字母、数字、下划线、连字符")
        if name in self.registry.clients:
            return ToolResult.fail(f"MCP server '{name}' 已连接；如需重连请先 disconnect_mcp_server")
        # 判断传输类型
        ttype = transport.lower()
        if ttype == "auto":
            ttype = "sse" if target.startswith(("http://", "https://")) else "stdio"
        try:
            if ttype == "sse":
                self.registry.add_sse(name, target)
            elif ttype == "stdio":
                self.registry.add_stdio_cmd(name, target)
            else:
                return ToolResult.fail(f"不支持的传输类型：{transport}")
            self.registry.connect(name)
        except Exception as e:
            # 回滚 factories 中残留
            self.registry._factories.pop(name, None)
            return ToolResult.fail(f"连接失败：{e}")
        # 细粒度展开工具
        self.registry.register_into(self.tool_registry, expand=True)
        added = self.registry.list_tools(name).get(name, [])
        if self.refresh_prompt:
            try:
                self.refresh_prompt()
            except Exception:
                pass
        tool_names = ", ".join(t.name for t in added) if added else "(无)"
        return ToolResult.ok(f"✅ 已连接 MCP server '{name}'，注册了 {len(added)} 个工具：{tool_names}")


class DisconnectMcpServerTool(ToolBase):
    name = "disconnect_mcp_server"
    description = "断开一个已连接的 MCP 服务器，其提供的工具会从工具表中移除。"
    args_schema = DisconnectMcpInput
    tags = ["mcp"]

    def __init__(self, registry: MCPRegistry, tool_registry, refresh_prompt: Optional[Callable[[], None]] = None):
        super().__init__()
        self.registry = registry
        self.tool_registry = tool_registry
        self.refresh_prompt = refresh_prompt

    def run(self, name: str) -> Union[str, ToolResult]:
        if name not in self.registry.clients:
            return ToolResult.fail(f"MCP server '{name}' 未连接")
        prefix = f"{name}__"
        removed = []
        for tool_name in list(self.tool_registry.list_names()):
            if tool_name.startswith(prefix):
                self.tool_registry.unregister(tool_name)
                removed.append(tool_name)
        # 也处理粗粒度名
        coarse = f"mcp_{name}"
        if self.tool_registry.has_tool(coarse):
            self.tool_registry.unregister(coarse)
            removed.append(coarse)
        self.registry.close(name)
        if self.refresh_prompt:
            try:
                self.refresh_prompt()
            except Exception:
                pass
        return ToolResult.ok(f"已断开 '{name}'，移除 {len(removed)} 个工具：{', '.join(removed)}")


def build_mcp_tools(
    registry: MCPRegistry,
    tool_registry,
    refresh_prompt: Optional[Callable[[], None]] = None,
) -> List[ToolBase]:
    """一次性创建三个 MCP 管理工具。

    Args:
        registry: 你的 MCPRegistry 实例。
        tool_registry: 目标 ToolRegistry（connect/disconnect 时自动展开/移除细粒度工具）。
        refresh_prompt: 回调函数，工具表变更后刷新 system prompt；传 ReActAgent.refresh_system_prompt 即可。
    """
    return [
        ListMcpServersTool(registry),
        ConnectMcpServerTool(registry, tool_registry, refresh_prompt=refresh_prompt),
        DisconnectMcpServerTool(registry, tool_registry, refresh_prompt=refresh_prompt),
    ]
