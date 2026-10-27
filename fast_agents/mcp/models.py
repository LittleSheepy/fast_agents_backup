"""MCP 数据模型"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ServerCapabilities(BaseModel):
    """服务端在 initialize 响应里声明的能力"""
    tools: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class InitializeResult(BaseModel):
    """initialize 握手结果"""
    protocol_version: str = "2024-11-05"
    server_info: Dict[str, Any] = Field(default_factory=dict)
    capabilities: ServerCapabilities = Field(default_factory=ServerCapabilities)
    instructions: str = ""


class ToolInfo(BaseModel):
    """MCP 工具元信息（来自 tools/list）"""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})

    @property
    def has_structured_input(self) -> bool:
        props = self.input_schema.get("properties", {})
        return bool(props) and set(props.keys()) != {"input"}


class CallResult(BaseModel):
    """tools/call 返回结果"""
    content: List[Dict[str, Any]] = Field(default_factory=list)
    is_error: bool = False
    structured_content: Optional[Any] = None

    def as_text(self) -> str:
        parts = []
        for item in self.content:
            t = item.get("type", "")
            if t == "text":
                parts.append(item.get("text", ""))
            elif t == "image":
                mime = item.get("mimeType", "image")
                parts.append(f"[图片：{mime}]")
            elif t == "resource":
                parts.append(f"[资源] {item.get('resource', {}).get('uri', '')}")
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "(无输出)"
