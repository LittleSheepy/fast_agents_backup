"""MCP 工具适配器 —— 两种形态：细粒度展开 / 粗粒度自动展开。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, create_model

from ..tools.tool_base import ToolBase
from ..tools.tool_result import ToolResult
from .client import MCPClient
from .errors import MCPCallError
from .models import ToolInfo

logger = logging.getLogger(__name__)


def _schema_to_pydantic(schema: Dict[str, Any]) -> type:
    """把 JSON Schema 转成 pydantic BaseModel，作为 args_schema。"""
    props = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: Dict[str, Any] = {}
    _TYPE_MAP = {
        "string": str, "integer": int, "number": float, "boolean": bool,
        "array": list, "object": dict,
    }
    for pname, pdef in props.items():
        json_type = pdef.get("type", "string")
        if isinstance(json_type, list):
            non_null = [t for t in json_type if t != "null"]
            py_type = _TYPE_MAP.get(non_null[0] if non_null else "string", str)
            if "null" in json_type and "null" not in required:
                from typing import Optional as _Opt
                py_type = _Opt[py_type]  # type: ignore
        else:
            py_type = _TYPE_MAP.get(json_type, str)
        if pname not in required:
            from typing import Optional as _Opt
            py_type = _Opt[py_type]  # type: ignore
        field_kwargs: Dict[str, Any] = {"description": pdef.get("description", "")}
        if "enum" in pdef:
            from typing import Literal
            literal_vals = tuple(pdef["enum"])
            py_type = Literal[literal_vals]  # type: ignore
        if "default" in pdef and pname not in required:
            field_kwargs["default"] = pdef["default"]
        elif pname not in required:
            field_kwargs["default"] = None
        fields[pname] = (py_type, Field(**field_kwargs))
    return create_model("MCPArgs", **fields)


class MCPToolAdapter(ToolBase):
    """细粒度适配器：每个 MCP 远端工具独立展开成一个 ToolBase。

    生产推荐用法：模型能看到每个参数的名称、描述、类型、必填、枚举，调用准确率最高。
    """

    tags = ["mcp"]

    def __init__(self, client: MCPClient, server_name: str, tool_info: ToolInfo, prefix_sep: str = "__"):
        self.client = client
        self.server_name = server_name
        self.tool_info = tool_info
        name = f"{server_name}{prefix_sep}{tool_info.name}"
        description = f"[MCP:{server_name}] {tool_info.description}"
        args_schema = _schema_to_pydantic(tool_info.input_schema) if tool_info.has_structured_input else None
        super().__init__(name=name, description=description, args_schema=args_schema)

    def run(self, **kwargs: Any) -> Union[str, ToolResult]:
        try:
            # 如果 args_schema 没有结构化定义，可能存在一个 input 字符串参数
            if not self.tool_info.has_structured_input and "input_text" in kwargs:
                kwargs = {"input": kwargs.pop("input_text")}
            result = self.client.call_tool(self.tool_info.name, kwargs)
        except MCPCallError as e:
            return ToolResult.fail(str(e))
        text = result.as_text()
        if result.is_error:
            return ToolResult.fail(text)
        return ToolResult.ok(text, artifact={"content": [c for c in result.content if c.get("type") == "text"]})


class MCPServerTool(ToolBase):
    """粗粒度适配器：一个 MCP server 对应一个工具，内部通过 auto_expand 再展开一层。

    保留用于与你 react_agent.py L139-155 的 `auto_expand` 协议兼容：
    暴露 `auto_expand=True` 和 `_available_tools` 属性，add_tool 时会自动展开。
    这种形态下每个子工具只有一个 input 字符串参数，适合单入参工具；多参数工具
    建议用 MCPToolAdapter（expand=True）。
    """

    tags = ["mcp"]

    def __init__(self, client: MCPClient, server_name: str, tools: List[ToolInfo], prefix_sep: str = "__"):
        self.client = client
        self.server_name = server_name
        self._tools_list = tools
        self.auto_expand = True
        self._available_tools = [{"name": t.name, "description": t.description} for t in tools]
        name = f"mcp_{server_name}"
        description = f"[MCP:{server_name}] 聚合工具，内部包含 {len(tools)} 个子工具：{', '.join(t.name for t in tools)}"
        super().__init__(name=name, description=description)

    def run(self, payload: Any = None, **kwargs: Any) -> Union[str, ToolResult]:
        """payload 支持 dict 或字符串；也支持把 dict 字段作为关键字参数传入
        （ToolBase.invoke 在 args_schema=None 时会把 dict 拆成 kwargs）。"""
        if payload is None and kwargs:
            payload = kwargs
        if isinstance(payload, dict):
            action = payload.get("action", "call_tool")
            if action == "list_tools":
                lines = []
                for t in self._tools_list:
                    lines.append(f"- {t.name}: {t.description}")
                return ToolResult.ok("\n".join(lines) if lines else "(无可用工具)")
            tool_name = payload.get("tool_name", "")
            arguments = payload.get("arguments", {})
        else:
            return ToolResult.fail("MCPServerTool 需要 dict 格式 payload")

        if not tool_name:
            return ToolResult.fail("需要提供 tool_name")
        # 兼容钩子传 {"input": "..."} 的单入参形态
        if isinstance(arguments, dict) and set(arguments.keys()) == {"input"} and len(self._tools_list) > 0:
            arguments = arguments["input"]
        if not isinstance(arguments, dict):
            arguments = {"input": arguments}

        info = next((t for t in self._tools_list if t.name == tool_name), None)
        if info is None:
            return ToolResult.fail(f"工具 '{tool_name}' 不存在；可用：{', '.join(t.name for t in self._tools_list)}")
        try:
            result = self.client.call_tool(tool_name, arguments)
        except MCPCallError as e:
            return ToolResult.fail(str(e))
        text = result.as_text()
        if result.is_error:
            return ToolResult.fail(text)
        return ToolResult.ok(text)
