"""MCP 模块单元测试

运行：python test/test_mcp.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_agents.mcp import (
    MCPRegistry, MCPClient, StdioTransport, MCPToolAdapter, MCPServerTool,
    MCPConnectionError, MCPCallError,
)
from fast_agents.tools import ToolRegistry, ToolResult
from fast_agents.tools.builtin.mcp_tool import (
    ListMcpServersTool, ConnectMcpServerTool, DisconnectMcpServerTool, build_mcp_tools,
)


def _spawn_client() -> tuple[MCPClient, str]:
    py = sys.executable
    server = str(Path(__file__).resolve().parents[1] / "examples/mcp/echo_server.py")
    transport = StdioTransport(command=py, args=[server])
    client = MCPClient(transport)
    return client, f'"{py}" "{server}"'


def test_1_connect_and_list_tools():
    """1. 握手成功，能拿到工具列表"""
    client, _ = _spawn_client()
    try:
        init = client.connect(timeout=10)
        assert init.server_info.get("name") == "demo-echo-server"
        tools = client.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"echo", "calculate"}
        # echo 是多参数，has_structured_input 应为 True
        echo = next(t for t in tools if t.name == "echo")
        assert echo.has_structured_input is True
    finally:
        client.close()


def test_2_call_tool():
    """2. 调用 echo / calculate 拿到正确结果"""
    client, _ = _spawn_client()
    try:
        client.connect(timeout=10)
        r = client.call_tool("echo", {"message": "hi", "prefix": ">", "repeat": 3})
        assert not r.is_error
        assert ">hi>hi>hi" in r.as_text()
        r2 = client.call_tool("calculate", {"expression": "3*4+1"})
        assert "13" in r2.as_text()
    finally:
        client.close()


def test_3_call_nonexistent_tool_raises():
    """3. 调用不存在的工具抛出 MCPCallError"""
    client, _ = _spawn_client()
    try:
        client.connect(timeout=10)
        try:
            client.call_tool("no_such_tool", {})
            assert False, "应该抛异常"
        except MCPCallError:
            pass
    finally:
        client.close()


def test_4_tool_adapter_schema():
    """4. MCPToolAdapter 正确生成 args_schema（多参数）"""
    client, _ = _spawn_client()
    try:
        client.connect(timeout=10)
        tools = client.list_tools()
        echo = next(t for t in tools if t.name == "echo")
        adapter = MCPToolAdapter(client, "demo", echo)
        assert adapter.name == "demo__echo"
        schema = adapter.input_schema
        assert "message" in schema["properties"]
        assert "prefix" in schema["properties"]
        assert "repeat" in schema["properties"]
        assert "message" in schema["required"]
        # prefix/repeat 有 default，不应在 required 里
        assert "prefix" not in schema["required"]
        assert "repeat" not in schema["required"]
        # 能正确调用
        res = adapter.invoke(message="ok", prefix="[x] ", repeat=2)
        assert res.succeeded
        assert "[x] ok[x] ok" in res.output
    finally:
        client.close()


def test_5_coarse_auto_expand_compat():
    """5. MCPServerTool 粗粒度：暴露 auto_expand 和 _available_tools，兼容 add_tool 钩子"""
    client, _ = _spawn_client()
    try:
        client.connect(timeout=10)
        tools = client.list_tools()
        coarse = MCPServerTool(client, "demo", tools)
        assert coarse.auto_expand is True
        assert len(coarse._available_tools) == 2
        res = coarse.invoke({"action": "list_tools"})
        assert "echo" in res.output and "calculate" in res.output
    finally:
        client.close()


def test_6_registry_register_into():
    """6. MCPRegistry.register_into 细粒度展开到 ToolRegistry"""
    client, cmd = _spawn_client()
    try:
        client.connect(timeout=10)
        reg = MCPRegistry()
        reg._clients["demo"] = client
        reg._tools["demo"] = client.list_tools()
        tr = ToolRegistry()
        reg.register_into(tr)
        assert tr.has_tool("demo__echo")
        assert tr.has_tool("demo__calculate")
        res = tr.invoke_tool("demo__calculate", expression="2+3")
        assert res.succeeded and "5" in res.output
    finally:
        client.close()


def test_7_builtin_tools_list_and_connect():
    """7. builtin 三个管理工具：list/connect/disconnect 链路正确"""
    client, cmd = _spawn_client()
    try:
        client.connect(timeout=10)
        reg = MCPRegistry()
        reg._clients["demo"] = client
        reg._tools["demo"] = client.list_tools()
        tr = ToolRegistry()
        list_t, conn_t, disc_t = build_mcp_tools(reg, tr, refresh_prompt=lambda: None)
        res = list_t.invoke()
        assert "demo" in res.output and "echo" in res.output
    finally:
        client.close()


def test_8_builtin_connect_dynamically():
    """8. ConnectMcpServerTool 运行时动态连接"""
    import sys as _sys
    py = _sys.executable
    server = str(Path(__file__).resolve().parents[1] / "examples/mcp/echo_server.py")
    reg = MCPRegistry()
    tr = ToolRegistry()
    refreshed = []
    conn_t = ConnectMcpServerTool(reg, tr, refresh_prompt=lambda: refreshed.append(1))
    target = f'"{py}" "{server}"'
    res = conn_t.invoke(name="dyn", target=target)
    assert res.succeeded, res.output
    assert "dyn" in reg.clients
    assert tr.has_tool("dyn__echo")
    assert tr.has_tool("dyn__calculate")
    assert refreshed, "应该触发 refresh_prompt"
    # 断开
    disc_t = DisconnectMcpServerTool(reg, tr, refresh_prompt=lambda: refreshed.append(1))
    res2 = disc_t.invoke(name="dyn")
    assert res2.succeeded
    assert not tr.has_tool("dyn__echo")
    assert "dyn" not in reg.clients
    reg.close_all()


def test_9_schema_to_openai_and_anthropic():
    """9. 适配器产出的 schema 能正确导出给 OpenAI/Anthropic"""
    client, _ = _spawn_client()
    try:
        client.connect(timeout=10)
        echo = next(t for t in client.list_tools() if t.name == "echo")
        adapter = MCPToolAdapter(client, "demo", echo)
        oai = adapter.to_openai_tool()
        assert oai["function"]["name"] == "demo__echo"
        assert "parameters" in oai["function"]
        ant = adapter.to_anthropic_tool()
        assert ant["name"] == "demo__echo"
        assert "input_schema" in ant
    finally:
        client.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"✅ {fn.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"❌ {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n共 {len(tests)} 项，通过 {passed} 项")
    sys.exit(0 if passed == len(tests) else 1)
