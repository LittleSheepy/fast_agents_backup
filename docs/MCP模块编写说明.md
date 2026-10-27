# MCP 模块编写说明

> 适用框架：`fast_agents` v0.0.1（含 tools / memory / skills）
> 适配系统：Windows / macOS / Linux
> MCP 协议版本：`2024-11-05`

---

## 一、MCP 是什么：和你的 Tools/Skills 的关系

MCP（Model Context Protocol）是 Anthropic 在 2024 年底提出的一个 **"AI 工具的 USB 接口"** 标准。你可以把它理解为：把"工具"这件事从「**写 Python 代码 → 继承 ToolBase → 注册到 ToolRegistry**」这一条路，扩展出第二条「**启动一个独立进程/连一个远程服务 → 自动发现它有哪些工具 → 直接调用**」的路。

| 维度 | 你的 Tools | 你的 Skills | **MCP** |
|------|-----------|-------------|---------|
| 形态 | Python 类，进程内 | 文件夹+SKILL.md/脚本，进程内 | **独立进程/远程 HTTP 服务** |
| 写一个要会 | Python | Markdown | **任何语言**（Python/Node/Go/Rust...） |
| 复用范围 | 本项目内 | 本项目内 | **任何支持 MCP 的客户端都能用**（Claude Desktop、Cursor、LangChain、你的 fast_agents...） |
| 安全边界 | 同一进程，出错会影响 Agent | 同进程 | **进程隔离**，MCP server 崩溃不影响 Agent |
| 安装方式 | pip 装到环境里 | 放到 .skills/ 目录 | `uvx mcp-server-xxx` / `npx @modelcontextprotocol/...` 即装即用 |

**一句话定位**：Tools 是你「自己写的工具」，Skills 是「写死的 SOP 流程」，MCP 是「从外部世界接入的标准化工具集合」。

### 业界主流实现对比

| 框架 | MCP 实现方式 | 优缺点 |
|------|-------------|--------|
| **LangChain** `langchain-mcp-adapters` | 把每个 MCP tool 包装成 `BaseTool` 子类，用 `MultiServerMCPClient` 管理多个连接 | 优点：生态最成熟；缺点：强依赖官方 SDK，`langchain-core` 全家桶带进来 |
| **OpenAI Agents SDK** | 通过 `MCPServer` 类 + `stdio`/`sse` 传输 | 优点：简洁；缺点：只支持 OpenAI 模型 |
| **Anthropic 官方 SDK** | 纯底层 transport，自己拼调用 | 最原始、最贴近协议，但要自己管 handshake、工具列表缓存 |
| **你的 fast_agents.mcp**（本次实现） | 零重型依赖，分 transports / client / registry / tool 四层；同时支持 auto_expand 粗粒度接入 **和** Schema 级细粒度接入 | 优点：不引第三方、Windows 友好、和你现有 ToolBase 无缝；缺点：高级能力（采样/sampling、roots、资源订阅）暂未实现 |

---

## 二、代码架构：完全贴合你现有风格

### 2.1 模块目录

我把 MCP 模块放在单数目录 `fast_agents/mcp/`，和你已有的 `tools/`、`memory/`、`skills/` 命名完全一致：

```
fast_agents/mcp/
├── __init__.py       # 模块出口，导出所有公共符号
├── errors.py         # 异常类体系（连接/协议/调用三层）
├── models.py         # pydantic 数据模型（MCPToolInfo / MCPToolCallResult 等）
├── transports.py     # 传输层抽象 + StdioTransport + SSETransport
├── client.py         # 单个 MCP server 的客户端（握手/发现/调用）
├── registry.py       # 多 server 注册表，批量管理和注册工具
└── tool.py           # 两种工具适配器（MCPServerTool / MCPToolAdapter）
```

**分层对应关系**（和你其他模块一一对应）：

| 层 | tools | memory | skills | **mcp** |
|----|-------|--------|--------|---------|
| 数据模型 | tool_result.py | schema.py | models.py | **models.py** |
| 异常 | tool_exception.py | (在 manager 里) | —— | **errors.py** |
| 实现/加载 | tool_base.py | sqlite_store.py | loader.py | **transports.py + client.py** |
| 注册管理 | tool_registry.py | manager.py | registry.py | **registry.py** |
| 对外工具 | builtin/*.py | tool.py (RememberTool等) | (通过 registry.get_tools 暴露) | **tool.py** |

### 2.2 分层数据流

```
┌─────────────────────────────────────────────────────────┐
│  ReActAgent                                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │  ToolRegistry (你已有的，本地工具 + MCP工具混在一起)│  │
│  └───────────────────────────────────────────────────┘  │
│        ▲                          ▲                     │
│        │ register_tool()          │ invoke_tool()       │
│        │                          │                     │
│  ┌─────┴──────────────────────────┴─────┐               │
│  │  MCPRegistry                         │               │
│  │  ┌──────────────┐ ┌──────────────┐   │               │
│  │  │  MCPClient 1 │ │  MCPClient 2 │   │   连接池       │
│  │  │ (echo)       │ │ (filesystem) │   │               │
│  │  └──────┬───────┘ └──────┬───────┘   │               │
│  └─────────┼────────────────┼───────────┘               │
│            │                │                           │
│  ┌─────────┴────────┐ ┌─────┴──────────┐                │
│  │ StdioTransport   │ │ SSETransport   │   传输层         │
│  │ (子进程 stdio)   │ │ (HTTP+SSE)     │                │
│  └─────────┬────────┘ └─────┬──────────┘                │
└────────────┼────────────────┼───────────────────────────┘
             │                │
      ┌──────┴──────┐   ┌─────┴──────┐
      │ 本地子进程  │   │ 远程HTTP服务│
      │ (npx/uvx/   │   │ (8001端口) │
      │  python)    │   └────────────┘
      └─────────────┘
```

### 2.3 关键设计决策（和 LangChain 方案的差异）

| 决策点 | LangChain 做法 | 本模块做法 | 理由 |
|-------|---------------|-----------|------|
| JSON-RPC 库 | 用官方 `mcp` SDK | **自己实现** | 官方 SDK 有依赖链（anyio、httpx、pydantic-settings、starlette、uvicorn 十几个包），你现在的库连 PyYAML 都有装，没必要为了 MCP 引一堆依赖。完整 JSON-RPC 2.0 客户端只有 100 多行，实现起来很干净。 |
| 工具展开 | 动态创建 BaseTool，所有参数串成一个 dict | **用 pydantic create_model 把 MCP inputSchema 逐字段展开成强类型** | 你的 ToolBase 依赖 args_schema 做多 provider schema 导出，如果不展开就只有一个笼统的 `arguments: dict`，模型看不到每个字段的 description 和 enum，调用准确率会差一大截。 |
| 工具命名 | 直接用远程工具原名 | **`<server_name>__<tool_name>` 前缀式** | 多个 server 可能都提供 `read_file`/`search`，直接用原名会冲突；双下划线既好认也不会和本地工具撞名。 |
| async | 全 async | **同步为主**，SSE 传输后台用线程读事件 | 你的 ReAct 主循环是同步的（tool_registry.invoke_tool 也是同步），做 async 反而要在你的同步循环里加 asyncio.run()，风格不一致。 |
| 进程启动 | 不处理 Windows | **加 CREATE_NO_WINDOW + 强制 UTF-8 环境变量** | 你是 Windows 系统，不开这个标志子进程会弹出一个黑控制台窗口；不强制 UTF-8 中文输出会变成 GBK 乱码。 |
| 兼容性 | 要求改造 Agent 代码 | **同时支持两种接入方式**：① 完全兼容你 `add_tool` 里预留的 `auto_expand` 钩子，零改动也能用；② 推荐的 register_into() 方式，在 Agent 初始化时展开成独立工具。 | 你的 react_agent.py L109-127 已经写好了 `if getattr(tool, "auto_expand", False)` 的处理逻辑，这个钩子就是为 MCP 预留的，必须能直接挂上。 |

---

## 三、核心源码逐文件讲

### 3.1 `errors.py` —— 分层异常

三层异常对应不同的排查方向：

```
MCPError（基类）
├── 连接层：MCPConnectionError / MCPTransportError / MCPTimeoutError
│   → 排查：命令对不对？子进程有没有启动？端口开没开？网络通不通？
├── 协议层：MCPProtocolError / MCPHandshakeError
│   → 排查：协议版本是不是 2024-11-05？返回格式是不是合法 JSON-RPC？
└── 业务层：MCPCallError / MCPToolNotFoundError / MCPResourceNotFoundError
    → 排查：工具名拼错没？参数对不对？server 返回 isError=true 看错误内容
```

### 3.2 `models.py` —— 数据模型

全部用 pydantic BaseModel，对应 MCP 规范字段。关键点：

- `MCPToolInfo.inputSchema` 是服务器声明的 JSON Schema，直接作为 OpenAI function calling 的 `parameters`（OpenAI 的 function calling 本来就是按 JSON Schema 设计的，100% 兼容）；
- `MCPToolCallResult.content` 是一个数组（可以混合 text/image/resource 链接），`to_text()` 方法拍平成纯文本给 ToolResult.output，原始结构化内容放在 artifact 里供下游使用。

### 3.3 `transports.py` —— 传输层

两个核心传输类，都继承自 `MCPTransport` 抽象基类，对外暴露 `connect/send/receive/close` 四个方法。

**StdioTransport（本地子进程）关键点**：

1. **Windows CREATE_NO_WINDOW**：用 `subprocess.CREATE_NO_WINDOW` 标志，避免弹出黑框；
2. **UTF-8 强制**：注入 `PYTHONIOENCODING=utf-8` 和 `PYTHONUTF8=1` 环境变量，即使服务器是中文 Windows 默认 GBK 也不会乱码；
3. **无缓冲管道**：`bufsize=0`，消息一到立刻可读，避免 flush 问题；
4. **后台读线程 + 消息队列**：主线程发请求后阻塞在 queue.get() 等响应，读线程把 stdout 按行切割后推到队列；
5. **stderr 保留**：子进程的 stderr 不会污染 stdout 协议流，但会被缓冲起来，超时时可以打印最近几行帮助排查。

**SSETransport（HTTP Server-Sent Events）关键点**：

1. MCP 的 SSE 协议是**两端点**：GET `/sse` 建立长连接收事件，第一个 `endpoint` 事件告诉客户端 POST 消息到哪个 URL；
2. 依赖 `requests` 库（不强制安装，用的时候才报错提示 `pip install requests`）；
3. 解析 SSE 事件格式（`event:` / `data:` / `id:` / `retry:` / 空行分隔）。

### 3.4 `client.py` —— MCP 客户端

核心逻辑：

1. **JSON-RPC 请求/响应配对**：每个请求分配递增 id，把 (Event, bucket) 存进 pending dict；后台读线程拿到消息按 id 回填并 set event；主线程 wait event 直到超时。这是一个标准的「同步请求 + 异步响应」模式，不需要 asyncio。
2. **initialize 握手**：
   - 发 `initialize` 请求，带上客户端协议版本、能力声明；
   - 检查服务端返回的协议版本（不严格一致只警告不报错）；
   - 握手完成后必须发一个 `notifications/initialized` 通知（**没有 id**，服务器不会回）；
3. **能力发现**：握手完成后一次性调用 `tools/list`、`resources/list`、`prompts/list`，结果缓存起来供后续使用（失败不致命，比如某些 server 没实现 resources 就返回空）；
4. **call_tool**：发 `tools/call`，返回 `MCPToolCallResult`；如果 `isError=true` 抛 `MCPCallError`（被 ToolBase 层捕获并转成 ToolResult.fail）。

### 3.5 `tool.py` —— 两种工具适配器（重点）

#### 方式一：MCPServerTool（兼容你预留的 auto_expand 钩子，快速接入）

你在 `react_agent.py` L109-127 已经写了这段逻辑：

```python
if getattr(tool, "auto_expand", False):
    # 把工具集合拆分为多个子工具，带 _mcp_server_name 前缀
    for tool_info in tool._available_tools:
        sub_tool_name = f"{tool.name}_{tool_info['name']}"
        # ... 动态生成 StructuredTool 并注册
```

`MCPServerTool` 严格按这个接口约定实现：
- `name` 属性：服务器名（比如 `"filesystem"`）
- `auto_expand = True`（类属性，`getattr` 能读到）
- `_available_tools` 属性：返回 `[{name, description, schema}, ...]`
- `run(action=..., tool_name=..., arguments=...)`：支持 `list_tools` 和 `call_tool` 两个 action

**优点**：接入零成本，agent.add_tool(srv_tool) 就完事；
**缺点**：你那段 auto_expand 代码把子工具的参数做成了一个 `input_text: str`（不是结构化参数），模型要自己拼 JSON 字符串，准确率不如方式二。**适合快速验证、临时调试。**

#### 方式二：MCPToolAdapter（推荐，生产用）

把**每个远程工具**展开成独立的 ToolBase 子类，核心是 `_json_schema_to_pydantic()` 函数——把 MCP server 返回的 inputSchema（JSON Schema）动态转成 pydantic BaseModel：

- 支持 `type` / `properties` / `required` / `description` / `enum` / `default`；
- `type: ["string", "null"]` 这种联合类型处理为 Optional[str]；
- `enum` 转 Literal 类型，让 pydantic 自动做取值校验；
- 不支持的高级关键字（`anyOf` / `oneOf` / `$ref`）退化为 Any，保证不解析失败。

展开后工具名是 `filesystem__read_file` 这种形式（双下划线分隔 server 和 tool），description 前面加 `[MCP:filesystem]` 标记，tags 带上 `mcp` 和 server 名。

**优点**：模型能看到每个字段的 description、类型、enum、是否必填，`to_openai_tool()` 导出的 schema 是精确的，调用准确率比方式一高很多；**缺点**：比方式一多几行初始化代码（用 `MCPRegistry.register_into()` 其实一行就搞定）。

### 3.6 `registry.py` —— 多 server 注册表

管理多个 MCPClient，提供三种接入方式：

| 方法 | 对应适配器 | 场景 |
|------|-----------|------|
| `get_server_tools()` | MCPServerTool | 给你 react_agent 现有 add_tool auto_expand 用 |
| `expand_tools(prefix_sep="__")` | MCPToolAdapter | 自己拿到工具列表做定制 |
| `register_into(tool_registry)` | MCPToolAdapter | **最常用**，一行把所有 MCP 工具注册进 ToolRegistry |

还提供 `connect_all()` / `close_all()` 批量连接断开，上下文管理器支持 `with MCPRegistry() as reg: ...`。

---

## 四、5 分钟上手：接入第一个 MCP Server

### 4.1 最快的方式：用示例 echo server 验证

我给你写了一个零依赖的示例 MCP server（`examples/mcp/echo_server.py`），它只实现两个工具：`echo`（回显）和 `calculate`（安全计算）。不需要装任何东西，直接跑：

```python
from pathlib import Path
from fast_agents import ReActAgent
from fast_agents.mcp import MCPRegistry

# 1. 建注册表，加一个 stdio 类型的 MCP server
mcp = MCPRegistry()
server_path = Path("examples/mcp/echo_server.py").resolve()
mcp.add_stdio("demo", "python", [str(server_path)])

# 2. 建 Agent（enable_skills=False 排除干扰，方便看 MCP 效果）
agent = ReActAgent(name="助手", enable_memory=False, enable_skills=False)

# 3. 连接并注册（推荐方式二：register_into 展开成独立工具）
mcp.connect_all()
mcp.register_into(agent.tool_registry, prefix_sep="__")

# 4. 看看工具列表，应该有 demo__echo 和 demo__calculate
print("已注册工具:", agent.tool_registry.list_names())

# 5. 跑起来
agent.run("帮我算一下 (12+8)*3^2 等于多少，然后用 echo 工具把结果回显三遍")

# 6. 退出时断开所有 MCP 连接
mcp.close_all()
```

### 4.2 接真实的公开 MCP Server

MCP 生态里最常用的几个公开 server：

| Server | 安装命令 | 提供的工具 |
|--------|---------|-----------|
| filesystem | `pip install mcp-server-filesystem` 或 `uvx mcp-server-filesystem` | read_file / write_file / list_directory / search_files / ... |
| fetch | `uvx mcp-server-fetch` | fetch_url（抓取网页内容转 markdown） |
| brave-search | `uvx mcp-server-brave-search`（需 API key） | web 搜索 |
| sqlite | `uvx mcp-server-sqlite` | 执行 SQL 查询 |
| github | `uvx mcp-server-github`（需 token） | 操作 GitHub issue/PR/文件 |

接 filesystem server 的代码（Windows）：

```python
from fast_agents import ReActAgent
from fast_agents.mcp import MCPRegistry

mcp = MCPRegistry()
# 给 filesystem server 传一个允许访问的根目录（Windows 路径）
mcp.add_stdio(
    "fs",
    "uvx",
    ["mcp-server-filesystem", "D:\\00sheepy\\01code"],
)

agent = ReActAgent(name="助手", enable_memory=False, enable_skills=False)
mcp.connect_all()
mcp.register_into(agent.tool_registry)
print("已注册工具:", agent.tool_registry.list_names())

agent.run("帮我看看 D:\\00sheepy\\01code 目录下有哪些 Python 项目")
mcp.close_all()
```

> 如果没有 `uv`，用 `pip install uv` 装一下（和 `pip` 一样的 Python 包管理器，但启动 CLI 工具更方便，不用关心脚本在不在 PATH）。也可以用 `npx -y @modelcontextprotocol/server-filesystem ...`（需要 Node.js）。

### 4.3 接远程 SSE 服务

```python
mcp.add_sse("weather", "http://localhost:8001")
mcp.connect_all()
mcp.register_into(agent.tool_registry)
```

SSE 连接会自动处理 endpoint 事件、session 管理。需要装 requests：`pip install requests`。

### 4.4 方式一：用你 add_tool 已有的 auto_expand 钩子（零改动）

如果你想用原来那段 auto_expand 逻辑（所有 MCP 工具参数打成 input_text），也可以，更简洁：

```python
from fast_agents import ReActAgent
from fast_agents.mcp import MCPRegistry

mcp = MCPRegistry()
mcp.add_stdio("demo", "python", [str(server_path)])
mcp.connect_all()

agent = ReActAgent(name="助手", enable_memory=False, enable_skills=False)
for srv_tool in mcp.get_server_tools():
    agent.add_tool(srv_tool)   # 自动检测到 auto_expand=True，展开成 demo_echo / demo_calculate

agent.run("...")
mcp.close_all()
```

### 4.5 不用 Registry，直接用单个 MCPClient

如果只接一个 server 或者要精细控制：

```python
from fast_agents.mcp import MCPClient, MCPToolAdapter

client = MCPClient.from_stdio("python", ["server.py"], name="myserver")
client.connect()
print("工具:", [t.name for t in client.list_tools()])

# 直接调用
r = client.call_tool("echo", {"message": "hi"})
print(r.to_text())

# 或者包装成适配器注册
from fast_agents import global_registry
for info in client.list_tools():
    global_registry.register_tool(MCPToolAdapter(client, info, prefix="myserver"))
```

---

## 五、Windows 专项避坑

1. **Python 命令用 `python` 不要用 `python3`**：Windows 官方安装器只注册 `python.exe`，`python3` 不存在会报 FileNotFoundError。
2. **路径用正斜杠或双反斜杠**：Python 字符串里 `"D:\\data"` 或者 `r"D:\data"`，不要写单反斜杠。
3. **uvx / npx 找不到命令**：
   - `pip install uv` 安装 uv（会把 `uvx.exe` 装到 Scripts 目录下）；
   - 或者 `npm install -g npx` 装 Node 版；
   - 如果刚装完命令还找不到，重启一下终端让 PATH 生效。
4. **子进程弹黑框**：本模块已经加了 `CREATE_NO_WINDOW`，正常情况下不会弹。如果你自己写 MCP server 用 `subprocess` 启动别的进程，记得也加这个标志。
5. **中文乱码**：本模块已经强制 `PYTHONIOENCODING=utf-8`。如果你自己写 server，记得 `print(..., flush=True)`，并且在文件开头别加 `# -*- coding: gbk -*-` 这种奇怪的声明。
6. **uvx 启动慢**：第一次跑 `uvx mcp-server-xxx` 会下载包，耐心等几秒；之后会缓存，启动很快。

---

## 六、自己写一个 MCP Server（以 Python 为例）

MCP server 只要满足三个条件就可以：
1. 启动后从 stdin 按行读 JSON-RPC 消息；
2. 向 stdout 写 JSON-RPC 响应（每行一条）；
3. 实现 `initialize`、`tools/list`、`tools/call` 三个方法。

最小模板（参考 `examples/mcp/echo_server.py`）：

```python
import json, sys

TOOLS = [
    {
        "name": "my_tool",
        "description": "工具描述（写给模型看的，要写清楚什么时候用、参数什么意思）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "查询关键词"},
                "limit": {"type": "integer", "description": "返回条数，默认5", "default": 5},
            },
            "required": ["query"],
        },
    }
]

def handle(msg):
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params", {})
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "my-server", "version": "0.1.0"},
        }}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params["name"]
        args = params.get("arguments", {})
        if name == "my_tool":
            text = f"收到查询: {args.get('query')}, limit={args.get('limit', 5)}"
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": text}], "isError": False,
            }}
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "content": [{"type": "text", "text": f"未知工具: {name}"}], "isError": True,
        }}
    # 通知（无 id）如 initialized 忽略即可
    if mid is None:
        return None
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"未实现: {method}"}}

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            msg = json.loads(line)
        except: continue
        resp = handle(msg)
        if resp is not None:
            print(json.dumps(resp, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
```

**写 MCP server 的关键**：
- **description 要写给模型看**，不是写给人看的。要写清楚「什么时候用」「什么时候不该用」「每个参数的含义和取值范围」，这直接决定模型会不会调对；
- **错误要走 `isError: true` 而不是 JSON-RPC error**：JSON-RPC error 表示协议层错误（方法不存在、参数格式错），工具业务错误（数据找不到、权限不够）应该在 result 里返回并设 `isError: true`；
- **stdout 只能输出 JSON**：任何 print 调试信息必须打到 stderr（`print(..., file=sys.stderr)`），否则客户端会把你的调试日志当成协议消息解析失败；
- **记得 flush**：`print(..., flush=True)`，不然消息可能缓冲在管道里客户端收不到。

---

## 七、测试验收清单

我写了 `test/test_mcp.py`，9 项测试覆盖了从模型到端到端的全部链路，在你的 Windows 机器上可以直接跑：

```powershell
cd D:\00sheepy\01code\02alg_code_bjb\agent\fast_agents
python test\test_mcp.py
```

测试覆盖：
1. 数据模型序列化
2. JSON Schema → pydantic 动态模型（字段类型/默认值/required/enum）
3. stdio 连接 + initialize 握手 + tools/list
4. call_tool 正常调用（echo 重复、calculate 四则运算 + 幂运算）
5. 调用不存在的工具抛 MCPCallError
6. 启动不存在的命令抛 MCPConnectionError
7. MCPToolAdapter 细粒度适配器 invoke
8. MCPServerTool auto_expand 属性和 `_available_tools` / list_tools / call_tool action
9. MCPRegistry.register_into 批量注册到 ToolRegistry 并正常 invoke
10. MCPToolAdapter 导出 OpenAI function schema

```
✅ 数据模型测试通过
✅ JSON Schema 转 pydantic 通过
✅ stdio 客户端连接 + 调用通过
✅ 错误处理通过
✅ 启动失败错误处理通过
✅ MCPToolAdapter 通过
✅ MCPServerTool (auto_expand) 通过
🔌 MCP 'myserver' 已连接，发现 2 个工具
✅ MCPRegistry.register_into 通过
✅ 多 provider schema 导出通过

==================================================
MCP 测试结果: 9/9 通过
```

---

## 八、高级话题（目前没实现，后续可以扩展）

| 能力 | 状态 | 说明 |
|------|------|------|
| Tools 调用 | ✅ 已实现 | 完整链路，最核心能力 |
| Resources 读取 | 🟡 接口已预留 | `MCPClient.read_resource()` 已写，MCPToolAdapter 还没包装（因为 resources 是 pull 模式不是工具调用，需要另一种抽象） |
| Prompts 模板 | 🟡 接口已预留 | `list_prompts()` 可用，但没暴露成工具 |
| tools/list_changed 通知 | ⏳ 未实现 | 服务器动态增减工具时客户端自动刷新；目前只读一次 |
| Sampling（让 MCP server 反过来调用 LLM） | ❌ 暂不做 | 这个要客户端暴露 LLM 能力给 server，安全风险大，等稳定了再加 |
| Roots（客户端声明允许访问的 URI 根） | ❌ 暂不做 | filesystem 这类 server 用命令行参数传目录就行 |
| Streamable HTTP（MCP 新传输） | ❌ 暂不做 | stdio + SSE 已覆盖 99% 场景，等协议稳定再加 |
| 异步客户端 | ❌ 暂不做 | 你的 Agent 主循环是同步的，等你哪天要 async 再补 |

---

## 九、常见问题排查

**Q: `FileNotFoundError: 找不到可执行文件 'uvx'`**
A: 装 uv：`pip install uv`，或者换成 `npx -y @modelcontextprotocol/server-xxx`（装 Node.js）。

**Q: 连接后 `list_tools()` 返回空列表**
A: 可能是子进程启动后立刻崩溃了。用 `client.transport.get_stderr()` 拿到 stderr 输出看错误；常见原因：Python 版本不兼容、缺依赖、命令参数写错。

**Q: 模型调用 MCP 工具时总是参数错**
A: 用方式二（`register_into` / `MCPToolAdapter`）而不是方式一（auto_expand），方式二导出的 schema 每个字段都有 description，模型不容易错。

**Q: `等待 MCP 响应超时（30s）`**
A: 工具执行时间长的话，在创建 MCPClient 时把 timeout 调大：`MCPClient(transport, timeout=120)`；或者在 MCPRegistry.add_stdio 之后 `reg.get("name").timeout = 120`。

**Q: 子进程关闭不彻底，端口被占用**
A: 程序退出时记得调 `mcp.close_all()` 或者用 `with MCPRegistry() as mcp:` 上下文管理器；SSE server 的话检查服务器端有没有正确关闭 /sse 长连接。

**Q: 多个 Agent 能不能共用一个 MCPRegistry？**
A: 可以，一个 MCPClient 维护一个连接，只要不跨线程并发调用就行（本实现是线程安全的请求/响应配对，但多线程同时 call_tool 会排队发送，建议每个 Agent 独立 registry）。

---

## 十、文件清单

本次交付新增的文件（**不修改任何你已有的业务逻辑**，只在 `fast_agents/__init__.py` 加了导出）：

```
fast_agents/mcp/
├── __init__.py
├── errors.py
├── models.py
├── transports.py
├── client.py
├── registry.py
└── tool.py

examples/mcp/
├── echo_server.py            # 零依赖示例 MCP 服务器（echo + calculate）
└── mcp_quickstart.py         # 快速使用示例

test/
└── test_mcp.py               # MCP 模块单元测试（9 项）
```

修改的文件（只有一处）：
- `fast_agents/__init__.py`：在 `__all__` 和顶层 import 里加了 MCP 相关符号的导出，不影响原有导入路径。

---

## 验证步骤（Windows PowerShell）

```powershell
cd D:\00sheepy\01code\02alg_code_bjb\agent\fast_agents

# 1. 跑 MCP 测试
python test\test_mcp.py

# 2. 跑快速示例
$env:PYTHONPATH="."
python examples\mcp\mcp_quickstart.py

# 3. （可选）跑原有测试确认没破坏
python test\test_tools.py
python test\test_react_agent.py
```

全部通过后，你就可以把 MCP 用起来了：`uvx mcp-server-filesystem <目录>` 是最值得先试的一个——接入后 Agent 就能直接读你硬盘上的文件了。
