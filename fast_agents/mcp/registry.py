"""MCP 多 server 注册表 —— 管理多连接、批量 connect/close/tool 展开。"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .client import MCPClient
from .errors import MCPError
from .models import ToolInfo
from .tool import MCPToolAdapter, MCPServerTool
from .transports import SSETransport, StdioTransport, make_stdio

logger = logging.getLogger(__name__)


class MCPRegistry:
    """管理多个 MCP server。

    用法：
        reg = MCPRegistry()
        reg.add_stdio("fs", "uvx mcp-server-filesystem D:/code")
        reg.add_sse("remote", "https://example.com/mcp")
        reg.connect_all()
        reg.register_into(agent.tool_registry)   # 细粒度：每个 MCP 工具独立展开
    """

    def __init__(self):
        self._factories: Dict[str, Callable[[], MCPClient]] = {}
        self._clients: Dict[str, MCPClient] = {}
        self._tools: Dict[str, List[ToolInfo]] = {}
        self._closed = False

    # ── 注册 ─────────────────────────────────────────────────

    def add_stdio(self, name: str, command: str, args: Optional[List[str]] = None,
                  env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None) -> None:
        """注册一个 stdio 类型的 server。"""
        def factory():
            t = StdioTransport(command=command, args=args, env=env, cwd=cwd)
            return MCPClient(t)
        self._factories[name] = factory

    def add_stdio_cmd(self, name: str, command_str: str) -> None:
        """便捷方法：直接传命令字符串，自动 shlex.split（Windows 用 posix=False）。"""
        t = make_stdio(command_str)
        self._factories[name] = lambda t=t: MCPClient(t)

    def add_sse(self, name: str, url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0) -> None:
        def factory():
            t = SSETransport(url=url, headers=headers, timeout=timeout)
            return MCPClient(t)
        self._factories[name] = factory

    # ── 生命周期 ─────────────────────────────────────────────

    def connect(self, name: str, timeout: float = 10.0) -> MCPClient:
        if name in self._clients:
            return self._clients[name]
        if name not in self._factories:
            raise MCPError(f"MCP server '{name}' 未注册")
        client = self._factories[name]()
        init = client.connect(timeout=timeout)
        self._clients[name] = client
        self._tools[name] = client.list_tools(timeout=timeout)
        logger.info(
            "MCP[%s] 已连接：%s v%s，工具 %d 个",
            name, init.server_info.get("name", "unknown"),
            init.server_info.get("version", "?"), len(self._tools[name]),
        )
        return client

    def connect_all(self, timeout: float = 10.0) -> None:
        for name in list(self._factories.keys()):
            try:
                self.connect(name, timeout=timeout)
            except Exception as e:
                logger.warning("MCP[%s] 连接失败：%s", name, e)

    def close(self, name: str) -> None:
        c = self._clients.pop(name, None)
        if c:
            c.close()
        self._tools.pop(name, None)

    def close_all(self) -> None:
        for name in list(self._clients.keys()):
            self.close(name)
        self._closed = True

    # ── 查询 ─────────────────────────────────────────────────

    @property
    def clients(self) -> Dict[str, MCPClient]:
        return dict(self._clients)

    def get_client(self, name: str) -> Optional[MCPClient]:
        return self._clients.get(name)

    def list_tools(self, server: Optional[str] = None) -> Dict[str, List[ToolInfo]]:
        if server:
            return {server: self._tools.get(server, [])}
        return dict(self._tools)

    # ── 注册进 ToolRegistry ──────────────────────────────────

    def register_into(self, registry, prefix_sep: str = "__", expand: bool = True) -> List:
        """把所有已连接 server 的工具注册到 ToolRegistry。

        Args:
            registry: 你的 fast_agents ToolRegistry。
            prefix_sep: 工具名前缀分隔符，默认 "__"（如 fs__read_file）。
            expand: True = 细粒度展开（每个 MCP 工具独立一个 ToolBase，参数 schema 完整）；
                    False = 粗粒度，每个 server 一个 MCPServerTool，单 input 参数，靠 auto_expand 再展开一层。
        """
        registered = []
        for server_name, tools in self._tools.items():
            client = self._clients[server_name]
            if expand:
                for info in tools:
                    t = MCPToolAdapter(client=client, server_name=server_name, tool_info=info, prefix_sep=prefix_sep)
                    registry.register_tool(t)
                    registered.append(t)
            else:
                t = MCPServerTool(client=client, server_name=server_name, tools=tools, prefix_sep=prefix_sep)
                registry.register_tool(t)
                registered.append(t)
        return registered
