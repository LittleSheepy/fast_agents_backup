# FastAgents context 模块编写说明

本文档面向 fast_agents 的上下文工程系统，参考 **LangChain / LangGraph 的消息裁剪与 SummarizationNode**、**LlamaIndex 的 Context 构建与召回**、**CrewAI 的上下文传递**、**Anthropic 的 compaction 与 tool result 截断** 等框架的共性设计，给出新架构的完整说明、源码解析和自定义策略教程。
> 面向 `fast_agents` 从「只有 tools / memory」扩展到「带上下文工程」的完整实现指南。
> 本文档配套 `fast_agents/context/` 下的可运行代码，所有代码已实测通过。

---

## 一、更新历史

> **v1 更新摘要**（首次交付，落地路线图模块 F 的 F1–F5）
> 1. **新增 `fast_agents/context/` 模块** —— 七个文件，覆盖 token 计数与预算（F1）、工具结果卸载（F2）、文件系统工具（F3）、可插拔压缩策略（F4）、缓存友好的分层拼装（F5）；
> 2. **`ReActAgent` 接入 context** —— 新增 `ctx` / `enable_context` / `context_budget` / `context_offload_threshold` 四个构造参数，开启后每轮按 token 预算分层拼装上下文、超长工具结果自动卸载；
> 3. **默认关闭** —— `enable_context` 默认 `False`，不改变既有行为；不传 `ctx` 时自动创建默认实例（当前工作目录 + 128k 窗口）；
> 4. **修掉一处接入期的重复注入** —— 开启 context 后，本轮用户输入由 user 层承载，主循环里不再对同一输入二次 `append_user`（详见 3.4）；
> 5. **零新增三方依赖** —— tiktoken 为**可选**依赖，没装时自动降级为启发式估算，`pyproject.toml` / `requirements.txt` 无需变动。

> **v2 预留摘要**（本版未实现的后续变更，先在此占位，避免下次改动忘记录入）
> 1. 语义召回层（`ContextPacket.metadata` 已留位）；
> 2. Prompt caching 断点显式化（`cache_control` 注入点已在 4.4 标注）；
> 3. `SummarizationMiddleware` 中间件化。

---

## 二、主流框架的 context 模块实现对比

### 2.1 主流框架怎么做的

上下文工程（context engineering）是近两年才被单独抽出来的一层。各家的叫法不同、边界不同，但**要解决的问题高度一致**：模型窗口有限，而 Agent 的历史、工具结果、文件内容会无界增长，必须有人负责「算账 + 取舍」。

| 框架 | 核心抽象 | 计量方式 | 压缩/裁剪 | 工具结果处理 | 分层与顺序 | 本实现借鉴 |
|---|---|---|---|---|---|---|
| **LangChain** | `trim_messages` / `MessagesPlaceholder` | `get_num_tokens_from_messages`（tiktoken 包装） | `trim_messages`（按 token 或条数裁剪，可保留 system） | 无内建，靠工具自己截断 | 通过 prompt 模板的占位符手工控制 | **token 计数口径**、**裁剪时保留 system** |
| **LangGraph** | State + `SummarizationNode` | 同 LangChain | SummarizationNode：超阈值就把老消息摘要化，保留最近 N 条 | 无内建 | 图节点决定，state 里存 messages | **「超阈值才摘要、保留最近 N 条」的触发式策略** |
| **LlamaIndex** | `ContextBuilder` / `ChatEngine` | tokenizer 可选 | ChatMemory + 自动压缩 | `ToolOutput` 可截断 | 有 `ContextPack` 概念，把片段组装进 prompt | **「ContextBuilder 做门面、packet 参与预算排队」的思路** |
| **CrewAI** | task context 传递 + 每任务独立上下文 | 无 | 无（把不同任务的上下文隔离来回避） | 无 | 按任务隔离，而非全局拼接 | **「压缩不是唯一解，隔离也是」的取舍意识** |
| **Anthropic Claude** | 无框架，讲实践 | 官方 API 计 token | **compaction**：逼近窗口时把对话总结 + 重启新对话 | **tool result 截断 + 文件回读**（大结果外置到文件，模型按需读取） | 强调「静态在前、动态在后」以命中 prompt cache | **卸载 + 回读机制**、**缓存友好顺序** |
| **DeepAgents** | 记忆即文件 + 上下文中间件 | 有 | 自动 compaction | 文件系统工具读写 | 会话状态外化成文件 | **文件系统作为上下文的「外部存储」** |

### 2.2 本实现与主流的比较

| 对比项 | 主流做法 | 本实现 | 差异说明 |
|---|---|---|---|
| **计数精度** | 强制依赖 tiktoken 或模型自带 tokenizer | tiktoken **可选**，缺失时启发式估算（CJK 1 字≈1 token、拉丁 4 字符≈1 token，再乘 1.15 安全系数） | 保持仓库「零重型依赖」原则；估算宁可多算不卡上限 |
| **是否有预算概念** | 多数只有「窗口大小」 | 显式 `TokenBudget`：`窗口 - 预留输出 - 安全余量 = 可用预算` | 预留输出很关键，不预留会出现「请求成功但回答被截断」 |
| **压缩策略** | 基本单一（trim 或 summarize） | **三级可插拔**：卸载 → 裁剪 → 摘要，逐级降级 | 优先选信息不丢的策略；摘要最慢也最易失真，放最后 |
| **压缩单位** | 多数按「条数」丢 | 按**消息组（turn）**丢，`assistant(tool_calls)` 与 `tool` 结果绝不拆散 | 结构化消息协议的硬约束，拆散直接 400 |
| **工具结果** | 多数靠工具自己截断（丢信息） | **卸载到磁盘 + `read_offloaded` 按需回读**（信息不丢） | 对齐 Anthropic / DeepAgents；也顺带把长任务状态落成文件 |
| **分层** | 用 prompt 模板手工拼 | 显式 `ContextLayer` + `LAYER_ORDER`，每层带 `priority` / `required` | 把「拼装顺序」和「裁剪优先级」变成数据，而非散落在代码里 |
| **管理方式** | 框架自动接管，用户难干预 | 自动拼装 **+ 一组工具让模型自己管理**（回读/读写文件） | 对齐记忆模块「自动 + 工具双路径」的成功经验 |
| **依赖** | langchain / llama-index 等大包 | **零新增依赖**（标准库 `json` / `pathlib` / `uuid`） | 与 `tools` / `memory` 同风格 |
| **暂未实现** | 语义召回、缓存断点显式化 | 预留 `metadata` 扩展位，本轮不做 | 先跑通闭环，接口留口子（见第七节） |

**一句话总结取舍**：拿走了 LangChain/LangGraph 的「分层 + 触发式压缩」、Anthropic 的「卸载 + 回读 + 静态在前」、LlamaIndex 的「门面 + 包」，
**丢掉了**重型的 Runnable 抽象、图编排、以及把压缩做成黑盒的自动接管模式；
CrewAI 的「用隔离代替压缩」这条路线属于 `agents` 模块（子代理）的范畴，本轮不碰。

---

## 三、模块架构

### 3.1 目录结构

```
fast_agents/context/               ← 单数目录，对齐 tools/ memory/ skills/ mcp/
├── __init__.py                    模块出口（只暴露三组概念 + 门面 + 工具）
├── tokens.py                      计量：TokenCounter / TokenBudget / 估算函数
├── layers.py                      分层：ContextLayer / ContextWindow / LAYER_ORDER
├── offload.py                     卸载：ContextOffloader / OffloadRecord
├── compressor.py                  治理：ContextCompressor 三级策略 / CompressionReport
├── builder.py                     门面：ContextBuilder / BuildReport
└── tools.py                       工具：read_offloaded + write/edit/list 四件套

examples/context/
└── context_react_agent_demo.py    5 个示例：分层 / 卸载 / 压缩 / 文件系统 / Agent 接入

test/
└── test_context.py                16 项单元测试（不依赖真实 LLM）
```

设计上完全对齐仓库既有架构约定：

- **门面模式** —— `ContextBuilder` 对应 `MemoryManager`：对外只暴露少量动作，Agent 不需要知道底下分了哪些层、用哪种压缩策略；
- **数据模型用 dataclass** —— `OffloadRecord` / `CompressionReport` / `BuildReport` 用 `@dataclass`，比 pydantic 更轻，因为它们是**内部日志结构**而非需要校验的输入；需要校验的输入（工具参数）仍然用 `pydantic BaseModel`，与 `ToolBase.args_schema` 一致；
- **工具继承 `ToolBase`** —— `ReadOffloadedTool` / `WriteFileTool` / `EditFileTool` / `ListFilesTool` 全部走 `args_schema` + `param_model` 体系，因此 `build_openai_like_tool()`、`|` 运算符、`ToolRegistry` 注册、`ToolResult` 结构化返回**全部直接可用**；
- **安全边界一致** —— 文件工具的路径沙箱思路与 `ReadFileTool` / `TerminalTool` 完全一致（`resolve()` 后 `relative_to(workspace)`，越界直接 `ToolResult.fail`，不做任何 IO）。

### 3.2 数据流总览

context 模块不自己存历史，它做的是**「在消息送进 LLM 之前，把各路来源拼装成一份带预算的上下文」**：

```
                        ┌───────────────────────────────────────────────┐
                        │            ReActAgent.run() 每轮开始           │
                        └───────────────────┬───────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        │                                   ▼                                   │
        │                        ContextBuilder.build()                         │
        │                                                                       │
        │  L1 system   ← self.messages 里的 system（静态，required，缓存锚点）    │
        │  L2 memory   ← memory.retrieve() 长期记忆 + buffer 摘要               │
        │  L3 history  ← memory.buffer.get_messages()   【单一数据源】           │
        │  L4 files    ← 文件片段 / 卸载回读内容                                  │
        │  L5 user     ← 本轮问题（required，永远最后）                          │
        │       │                                                               │
        │       ▼  ContextWindow（按层序，带 priority / required）               │
        │  TokenCounter 算账 → 超预算？                                          │
        │       │                                                               │
        │       ▼  ContextCompressor.compress()                                 │
        │  ① 卸载 offload  →  ② 裁剪 trim（成组）  →  ③ 摘要 summarize           │
        └───────────────────────────────────┬───────────────────────────────────┘
                                            ▼
                                   Messages → LLM
                                            │
                        工具返回超长结果 ────┘
                                            ▼
                              ContextBuilder.observe()
                                            │
                      ContextOffloader.offload()  → 落盘 + 只留引用文本
                                            │
                              模型需要全文 → read_offloaded 工具按 id/行区间回读
```

**四条清晰的边界**：

- **context 不持有历史** —— L3 的历史永远来自 `memory.buffer`，「单一数据源」约定不被打破；
- **context 不改记忆内容** —— 它只读 `memory.retrieve()` 的结果做渲染，不写回 `store`；
- **context 不碰工具执行** —— 它只对「工具已经返回的结果」做卸载，`ToolRegistry` 的执行链路不受影响；
- **卸载是可逆的** —— 落盘的是全文，上下文里留的是引用，信息不丢，只是从「常驻」变成「按需」。

### 3.3 类关系

```
TokenCounter                          ← 计量：tiktoken 优先，缺失降级估算
TokenBudget                           ← 预算：窗口 - 预留输出 - 安全余量

ContextLayer                          ← 一层（name / messages / priority / required）
ContextWindow                         ← 一组层（add_layer / flatten / validate）

ContextOffloader                      ← 卸载：把超长文本写盘，返回引用；支持回读
└── OffloadRecord                     ← 卸载元数据（id / tool_name / 长度 / 预览）

ContextCompressor                     ← 治理：三级策略串联
└── CompressionReport                 ← 压缩结果（卸载数 / 裁剪数 / 是否摘要 / token 前后）

ContextBuilder (门面)                 ← 组合：window + counter + budget + offloader + compressor
└── BuildReport                       ← 一次 build 的结果（分层统计 / 压缩报告 / 告警）

ReadOffloadedTool / WriteFileTool / EditFileTool / ListFilesTool (ToolBase)
build_context_tools()                 ← 工厂，按需组装

ReActAgent                            ← _init_context / _fit_context / _offload_observation
```

### 3.4 ⭐ 接入 ReActAgent 时的两条边界

这两处是「Context 模块」和「Agent 主循环」的接缝，最容易出问题，值得单独讲清。

#### 边界一：本轮用户输入只能出现一次

开启 context 后，本轮问题是由 **user 层**承载的（`ContextBuilder.build(user_input=...)` 会把它拼在最后）。
而 `ReActAgent.run()` 的循环里本来就有一句「第 1 步追加用户输入」的历史逻辑：

```python
# 修复前：两条路径都会加，结果是 user 消息出现两次
if current_step == 1:
    self.messages.append_user(user_input)
```

`_fit_context()` 已经把 `user_input` 拼进工作区了，这里再追加一次，模型第一眼就会看到**两条一模一样的用户输入**——
轻则浪费 token、让模型以为「用户说了两遍」而重复确认，重则诱导模型多做一轮无意义推理。

修复方式是在追加前判一次「是否已由 context 承载」：

```python
# 修复后：开启 context 时，user 层已经放好了本轮问题，不再重复追加
if current_step == 1 and not (self.enable_context and self.ctx is not None):
    self.messages.append_user(user_input)
```

对应的回归断言写在 `test/test_react_agent.py`：

```python
def test_context_user_input_not_duplicated():
    """多工具步场景下，user 输入也只注入一次（不随步数累积）"""
    ...
    for i, call in enumerate(llm.calls):
        users = [m for m in call if m.get("role") == "user"]
        assert len(users) == 1, f"第 {i + 1} 次调用 user 出现 {len(users)} 次"
```

#### 边界二：开启 context 时，记忆不再二次拼进 user

`run()` 里原本的流程是「检索长期记忆 → 拼接到用户输入文本上 → 作为 user 消息发出去」：

```python
memory_text = self._retrieve_long_term_memory(input_text)
user_input = self._merge_memory(input_text, memory_text)
```

context 打开后，长期记忆改由 **L2 memory 层**统一承载（`_collect_context_packets()` 收集），
所以 `_fit_context()` 之后又把 `user_input` 还原成 `input_text`：

```python
if self.enable_context and self.ctx is not None:
    self._fit_context(input_text)
    user_input = input_text          # ← 长期记忆已在 memory 层，不再拼进 user
```

**为什么记忆层用 `role="user"` 而不是 `role="system"`**：记忆是**参考信息**而非**指令**。
放进 system 等于把可能来自用户输入的不可信内容提升到系统指令级别（记忆投毒风险），
这一点与 `记忆模块编写说明.md` 4.9 节的结论完全一致，两层模块用的是同一套注入安全原则：

| 维度 | system | user（本实现的选择） |
|---|---|---|
| 语义权威 | 高，模型更服从 | 低，只作参考 |
| 安全 | ⚠️ 不可信内容升格为指令 | ✅ 层级低，风险可控 |
| 缓存 | 必须在末尾 | 位置灵活 |

为了让模型不把记忆误当成用户刚说的话，memory 层的内容统一带一行前缀：

```python
Message("user", f"[参考记忆，非用户本轮发言]\n{memory_text}", ...)
```

#### 最终的工作区职责划分

| | `self.messages`（工作区） | `memory.buffer`（跨轮历史） | `ContextBuilder.window`（本轮拼装窗口） |
|---|---|---|---|
| 定位 | 本次 run 的推理工作区 | 跨轮会话历史的唯一数据源 | 一次 build 产出的分层快照 |
| 内容 | 分层拼装结果 + 本轮 tool 链 | 历轮 user/assistant（窗口 + 摘要） | system / memory / history / files / user |
| 生命周期 | run 开始重建、结束即可丢弃 | 跨 run 长期存在 | 每次 `build()` 重建 |
| 是否裁剪 | 不裁（已由 builder 裁过） | 滑动窗口 + 滚动摘要 | 按预算裁剪 + 压缩 |

> **为什么不让 context 直接接管 `memory.buffer` 的裁剪？**
> 因为 `buffer` 是「历史存储」，`window` 是「本次送模型的视图」，两者要解决的问题不同：
> 同一条历史，在不同模型窗口（32k / 128k）下该留多少条是不一样的，这属于「视图」的事，不该写进「存储」。

---

## 四、逐个文件讲解

### 4.1 `tokens.py` —— 计量：先知道有多少，才谈得上怎么裁

为什么必须先做这个（对应路线图 F1 的排序）？因为**「裁不裁」的前提是「知道有多少」**。
`BufferMemory` 按「消息条数」裁剪，但一条消息可能 3 个 token，也可能是一整篇文档——
只有把 token 数算出来，才能按**预算**而不是条数管理上下文。

#### TokenCounter：三级降级，零新增依赖

```python
class TokenCounter:
    def __init__(self, model="gpt-4o-mini", use_tiktoken=True, safety_ratio=1.15):
        self._encoder = self._load_encoder(model) if use_tiktoken else None
        self.backend = "tiktoken" if self._encoder is not None else "heuristic"
```

降级链是 **tiktoken → 通用 cl100k 编码 → 启发式估算**，任何一步失败都静默退回下一级：

```python
    @staticmethod
    def _load_encoder(model):
        try:
            import tiktoken
        except Exception:
            return None                      # 没装就是没装，不能影响框架运行
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")   # 模型名不认识 → 通用编码
            except Exception:
                return None                    # 彻底失败 → 上层降级为估算
```

**为什么不把 tiktoken 写进依赖**：它的体积和「零重型依赖」定位不符，而且它只覆盖 OpenAI 系模型。
把它做成**可选增强**，装了更好、不装也能跑，是这个仓库一贯的做法。

#### 启发式估算的分类规则

```python
def estimate_text_tokens(text, safety_ratio=1.15):
    cjk = latin = other = 0
    for ch in text:
        if _is_cjk(ch):                                   # 中日韩：基本 1 字 1 token
            cjk += 1
        elif ch.isascii() and (ch.isalnum() or ch == "_"): # 英文/数字：4 字符 ≈ 1 token
            latin += 1
        else:                                             # 标点/空白：2 字符 ≈ 1 token
            other += 1
    tokens = cjk + latin / 4.0 + other / 2.0
    return int(tokens * safety_ratio) + 1
```

两条关键取舍：

1. **宁可多算，不卡上限** —— 默认乘 `safety_ratio=1.15`。估算少算了 → 请求超窗报错；估算多算了 → 少留几条历史。两害相权，后者代价小得多；
2. **中文单独处理** —— 中文在 BPE 里基本一字一 token，用「4 字符 1 token」的英文规则去估中文，会**低估到 1/4**，这是最常见的估算错误。

#### 消息级计数：别漏掉固定开销和工具 schema

```python
_PER_MESSAGE_OVERHEAD = 4    # 每条消息的 role / 分隔符开销
_PER_REQUEST_OVERHEAD = 3    # 每次请求的固定开销

def count_message_tokens(message, counter=None):
    total = _PER_MESSAGE_OVERHEAD
    content = message.get("content")
    if isinstance(content, str):
        total += counter.count(content)
    elif isinstance(content, list):                # 多模态块：图片按 85 token 粗算
        ...
    for tool_call in message.get("tool_calls") or []:   # function calling 也要算
        total += counter.count(fn.get("name", "")) + counter.count(fn.get("arguments", "")) + 6
    return total
```

还有一处**最容易被忽略的大开销**——工具 schema：

```python
    def count_tool_schemas(self, tools):
        """每轮请求都要带全套 schema，10 个工具轻松吃掉一两千 token"""
        return self.count(json.dumps(list(tools), ensure_ascii=False))
```

工具 schema 是常驻开销，不随对话增长，但会被计入窗口。做预算规划时必须把它算进去，
否则会出现「消息只占 60%，但加上 tools 就超了」的情况。

#### TokenBudget：把窗口切成「可用」和「预留」

```python
@dataclass
class TokenBudget:
    max_context_tokens: int = 128_000    # 模型窗口
    reserve_for_output: int = 4_000      # 给回复留的空间
    safety_margin: int = 1_000           # 估算误差的缓冲

    @property
    def available(self) -> int:
        return max(1, self.max_context_tokens - self.reserve_for_output - self.safety_margin)
```

`reserve_for_output` 是最容易被省略、也最不能省的一项：不预留的话，输入占满窗口后模型没有余量输出，
会出现「请求成功但回答被截断」或者直接 400，而且**报错信息往往很难指向真实原因**。

### 4.2 `layers.py` —— 分层：拼装顺序也是一份数据

上下文不是一根绳子，而是**分层叠起来的**。分层管两件事：**拼装顺序**和**裁剪优先级**。

#### LAYER_ORDER：静态在前、动态在后

```python
LAYER_ORDER = {
    "system": 10,       # 身份 + 工具清单 + 技能索引   静态，required，缓存锚点
    "memory": 20,       # 长期记忆 + 会话摘要          低频变化
    "summary": 25,      # 压缩产生的历史摘要           低频变化
    "history": 30,      # buffer 窗口内的 user/assistant 滚动
    "files": 40,        # 文件片段 / 卸载回读           动态
    "tool_result": 45,  # 工具结果（按需注入时）        动态
    "user": 90,         # 本轮问题                     必须最后
}
```

这个顺序不是随意排的，它同时满足两个硬约束：

1. **prompt caching 是前缀匹配** —— 从请求开头到缓存断点必须逐 token 一致。一旦某轮变动的记忆被写进 system 中段，**后面所有内容和断点全部失效**，长会话成本翻倍。所以「静态在前、动态在后」不是风格偏好，是钱；
2. **指令层级** —— 本轮问题必须离模型最近（最后），保证模型不会被人情世故式的历史带偏。

#### ContextLayer：每层自带裁剪策略

```python
@dataclass
class ContextLayer:
    name: str
    messages: List[Message] = field(default_factory=list)
    priority: int = 50        # 越大越先保留
    required: bool = False    # True 表示绝不裁剪

    @property
    def order(self) -> int:
        return LAYER_ORDER.get(self.name, 60)   # 未知层排在 history 与 user 之间
```

把 `priority` / `required` **做成字段而不是写死在代码里**，好处是裁剪逻辑不用关心谁是 system、
谁是 user，只要看标记就行。`system` 层和本轮 `user` 层都标 `required=True, priority=100`，
因此压缩器天然会跳过它们，不需要在压缩器里写「如果是 system 就……」。

#### ContextWindow：flatten 与 validate

```python
    def flatten(self) -> List[Message]:
        """按层序拼成一条 Message 列表"""
        result = []
        for layer in self.ordered_layers():     # sorted by layer.order，稳定排序
            result.extend(layer.messages)
        return result
```

`validate()` 是**结构化消息协议的自检**，专门用来抓「tool 配对被拆坏」这类隐蔽错误：

```python
    def validate(self) -> List[str]:
        problems = []
        previous_had_tool_calls = False
        for message in self.flatten():
            if message.role == "tool" and not previous_had_tool_calls:
                problems.append(f"tool 消息缺少配对的 assistant(tool_calls)：...")
            previous_had_tool_calls = bool(message.tool_calls)
        return problems
```

**为什么这条检查必须有**：OpenAI 强校验 `tool` 消息必须严格跟在带 `tool_calls` 的 `assistant` 之后，
一旦裁剪把 `assistant(tool_calls)` 挤掉却留下孤立的 `tool` 结果，请求**当场 400**，
而且错误信息只说「tool message 没有对应 tool_call」，不告诉你是哪一步裁坏的。有了 `validate()`，
测试里就能在发请求之前把它抓出来。

### 4.3 `offload.py` —— 卸载：大结果落盘，上下文只留引用

问题很直白：一次 `read_file` 或 `web_search` 可能返回几万字符，直接塞进 tool 消息，几轮下来上下文就爆了——
**而且后面几轮根本不再需要全文**。

方案对齐 Anthropic 的「tool result 截断 + 文件回读」与 DeepAgents 的「会话状态外化成文件」：
**大结果落盘，上下文里只留摘要 + 引用 id，模型真想看全文时再用 `read_offloaded` 按 id 回读（可指定行区间）**。

落盘布局：

```
<workspace>/.fast_agents/context/offload/
├── index.json          # 索引：id → 元数据（工具名、原始长度、摘要、预览）
└── <id>.txt            # 正文（全文，不截断）
```

#### 核心方法：should_offload / offload / read

```python
    def should_offload(self, text, threshold=None) -> bool:
        return len(text or "") > (self.threshold if threshold is None else threshold)

    def offload(self, tool_name, text, *, threshold=None, summary="", metadata=None) -> str:
        text = text or ""
        if not self.should_offload(text, threshold):
            return text                          # ← 没超阈值原样返回，零 IO
        offload_id = uuid4().hex[:16]
        path = self.root / f"{offload_id}.txt"
        try:
            path.write_text(text, encoding="utf-8")
        except Exception as e:
            return self._fallback_truncate(text, str(e))   # ← 落盘失败退回截断，不能让对话中断
        ...
        return self._render_reference(record)
```

两个必须守住的健壮性原则：

- **无脑调用也安全** —— `offload()` 在没超阈值时直接 return 原文，**不产生任何 IO**。正因如此，Agent 主循环里可以「每次工具返回都调一次」，不需要自己判断该不该卸载；
- **卸载失败不能让对话中断** —— 磁盘满、权限不足都可能导致写盘失败，此时退回「截断 + 提示」，对话继续，只是丢了全文。

#### 引用文本：要明确告诉模型「东西去哪了、怎么拿回来」

```python
    def _render_reference(self, record) -> str:
        return (
            f"[工具结果已卸载] 工具 `{record.tool_name}` 返回了 {record.original_chars} 字符，"
            f"全文已写入离线文件，不在上下文中。\n"
            f"offload_id={record.offload_id}\n"
            f"内容开头预览：{record.preview[:200]}\n"
            f"如需查看完整内容，调用工具 read_offloaded(tool_name 的 offload_id="
            f"\"{record.offload_id}\"，可选 start_line/max_lines) 按需读取。"
        )
```

**这段文本是修过一版才定下来的**。只写「已卸载到文件」不够——模型会一头雾水地反复重试原工具。
必须三件事都给全：**卸载了什么**（工具名 + 原长度）、**内容开头长什么样**（预览，让模型判断要不要读）、
**怎么取回**（工具名 + 参数名 + 可选参数）。

#### 回读与行区间

```python
    def read(self, offload_id, start_line=1, end_line=None, max_lines=200) -> str:
        if not _SAFE_ID.match(offload_id or ""):        # id 白名单，杜绝 ../ 穿越
            return f"[读取失败] 非法的 offload_id：{offload_id!r}"
        record = self._index.get(offload_id)
        if record is None:
            return f"[读取失败] 未找到 offload_id={offload_id} 的卸载记录（可能已被淘汰）"
        ...
        numbered = "\n".join(f"{i:>5}|{line}" for i, line in enumerate(selected, start))
        header = f"📦 卸载内容 {offload_id}（来自工具 {record.tool_name}，共 {total} 行，显示 {start}-{stop} 行）"
        if stop < total:
            header += f"\n⚠️ 未读完，如需后续内容把 start_line 设为 {stop + 1}"
        return f"{header}\n{numbered}"
```

三个细节都对齐 `ReadFileTool`：

- **带行号、支持区间、有续读提示** —— 大文件可以分几次读，而不是一次性拉回又爆上下文；
- **失败返回可读文本而不是抛异常** —— 工具链不能因为「id 写错了」就断掉，模型看到错误提示能自己纠正；
- **id 白名单** —— 只允许十六进制字符，路径穿越（`../../etc/passwd`）在入口就被挡住。

#### 有界增长：淘汰而非无限堆积

```python
    def _enforce_limit(self) -> None:
        if self.max_files <= 0 or len(self._index) <= self.max_files:
            return
        ordered = sorted(self._index.values(), key=lambda r: r.created_at)
        for record in ordered[: len(self._index) - self.max_files]:   # 淘汰最早创建
            Path(record.path).unlink(missing_ok=True)
            self._index.pop(record.offload_id, None)
```

卸载目录会随会话一直写下去，必须设上限（默认 200 个文件）。淘汰策略选「最早创建」而不是
「最久未访问」，因为后者需要在 `read()` 里回写时间戳、增加 IO 和并发复杂度，收益不成比例。

### 4.4 `compressor.py` —— 治理：三级策略，逐级降级

预算超了怎么办？不同框架各有各的招（LangChain 的 `trim_messages`、LangGraph 的 `SummarizationNode`、
Anthropic 的 compaction）。这里把它做成**可插拔的三级策略**：

```
第 1 级  卸载 offload   —— 把超长 tool 结果写到磁盘，上下文只留引用（最优，信息不丢）
第 2 级  裁剪 trim      —— 从最老的历史开始成组丢弃 user/assistant（快，会丢信息）
第 3 级  摘要 summarize —— 把丢弃的历史交给 LLM 压成一段话（慢，但保住要点）
```

**为什么是这个顺序**：优先选「信息不丢」的策略。卸载只是把内容挪了位置，模型想要还能拿回来；
裁剪是直接丢；摘要是**不可逆的有损压缩**，而且依赖一次额外的 LLM 调用（慢、可能失败、可能失真）。
把最贵最不可逆的放在最后，只有前两级都救不回来才动用。

#### compress：逐级判断，够用就停

```python
    def compress(self, layer, counter, budget) -> CompressionReport:
        limit = budget.available if hasattr(budget, "available") else int(budget)
        report = CompressionReport()
        report.tokens_before = layer.token_count(counter)
        if report.tokens_before <= limit:
            report.tokens_after = report.tokens_before
            return report                                    # ← 没超就直接返回

        if self.enable_offload and self.offloader is not None:
            self._offload_layer(layer, report)               # 第 1 级
        if self.enable_trim and layer.token_count(counter) > limit:
            self._trim_layer(layer, limit, counter, report)   # 第 2 级（每一级都重新判断）
        if self.enable_summary and layer.token_count(counter) > limit:
            self._summarize_layer(layer, limit, counter, report)  # 第 3 级
        report.tokens_after = layer.token_count(counter)
        return report
```

注意每一级之前都**重新算一次 token**：卸载可能已经解决问题，就不必再裁剪。
这也是「三级策略」而不是「一次全做」的意义。

#### 成组切割：tool 配对绝不拆散

```python
def _group_messages(messages) -> List[List[Message]]:
    """把消息按「原子组」切开：
    - assistant(tool_calls) + 后续 tool 消息 → 一组（不可拆）
    - 其余每条消息各自成组
    """
    groups, buffer = [], None
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            buffer = [message]
            groups.append(buffer)
            continue
        if message.role == "tool" and buffer is not None:
            buffer.append(message)
            continue
        buffer = None
        groups.append([message])
    return groups
```

**这是「不许把 tool 配对拆坏」这条硬规则在代码里的落点。** 裁剪和摘要都以「组」为单位，
所以永远不可能出现「留了 tool 结果、丢了对应的 assistant(tool_calls)」这种 400 组合。
`test_context.py` 里 `test_compressor_trim_keeps_tool_pairing` 专门锁死这一条。

#### 摘要失败必须降级

```python
    def _call_summarize(self, history_text: str) -> str:
        prompt = self.compact_prompt.format(max_words=self.summary_max_words, history=history_text)
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}], temperature=0.0)
            text = (response.content or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"⚠️ 上下文摘要失败，降级为截断：{e}")
        return history_text[: self.summary_max_words * 2]
```

**摘要是一次额外的 LLM 调用，它自己就可能失败**（限流、超时、格式异常）。
如果让它抛异常上去，用户会因为「压缩失败」而拿不到任何回答——这是典型的「为了优化体验反而毁掉体验」。
所以这里是**双层降级**：LLM 调用失败 → 退回截断；截断也拿不到东西 → 上层还有 `keep_recent` 保底。
处理方式与 `BufferMemory._summarize` 保持一致。

#### 摘要写回层里，而不是丢掉

```python
        layer.messages = [
            Message("system", f"[历史已压缩，供参考]\n{summary}",
                    metadata={"type": "compacted_summary", "required": True})
        ] + keep
```

摘要以 `required=True` 的一条消息形式**写回 history 层开头**，后面接保留下来的最近若干条。
这样下一轮再 build 时，这段摘要仍在，不会「压完就忘」。
`self._accumulated_summary` 也留了一份，供 Agent 跨轮注入使用。

#### 关于缓存断点的一处扩展位

如果要做 prompt caching，最自然的断点是**在 system 层末尾**——它是整段上下文里最长、最稳定的一段。
`ContextBuilder.build()` 里 system 层是第一个被 `add_layer` 的、且 `required=True`，
所以按 `LAYER_ORDER` 拼出来它天然在最前、最完整。要接 Anthropic 的 `cache_control`，
只需在 `ContextLayer` 上挂一个 `cache_breakpoint: bool` 字段，在 `flatten()` 时对该层最后一条消息打标即可，
**不需要动压缩和裁剪的任何逻辑**。这就是「把顺序和优先级做成数据」带来的好处。

### 4.5 `builder.py` —— 门面：把上面四件事串成一次调用

`ContextBuilder` 对齐 `MemoryManager` 的写法：**对外只暴露少量动作，Agent 侧不需要知道底下分了哪些层、用哪种压缩策略**。

#### 三个动作

```python
builder.build(system_prompt=..., history=..., user_input=..., memory_packets=...)  # 拼装 + 裁剪
builder.observe(tool_name, content)      # 工具结果回来后：判断是否卸载（无脑调用也安全）
builder.record_turn(user_input, answer)  # 一轮结束：可选地写回工作区
```

#### build：五层依次就位

```python
        # ---- L1 system：required，不参与裁剪 ----
        if system_prompt:
            system_layer = ContextLayer(name="system", required=True, priority=100)
            ...

        # ---- L2 memory：长期记忆 / 会话摘要 ----
        memory_text = self._render_packets(memory_packets) if memory_packets else ""
        if memory_text:
            memory_layer = ContextLayer(name="memory", priority=80)
            memory_layer.add(Message("user", f"[参考记忆，非用户本轮发言]\n{memory_text}",
                                     metadata={"type": "memory", "source": "memory"}))
            self._fit_layer(memory_layer, self._quota_for("memory"), self.counter, warnings)
            ...

        # ---- L3 history：滚动窗口，最容易爆的一块 ----
        history_layer = ContextLayer(name="history", priority=60)
        for message in history or []:
            if message.role == "system":
                continue          # system 由 L1 统一负责，历史里的 system 一律丢弃
            history_layer.add(message)
        report = self.compressor.compress(history_layer, self.counter, self._quota_for("history"))

        # ---- L4 files：卸载回读 / 文件片段 ----
        ...

        # ---- L5 user：required，永远最后 ----
        if user_input:
            user_layer = ContextLayer(name="user", required=True, priority=100)
            user_layer.add(Message("user", user_input, metadata={"type": "user"}))
```

两处值得单独说：

1. **历史里的 system 一律丢弃** —— `buffer` 有可能会把 system 混进来，`self.messages` 里也常驻一条 system。
   如果 L3 原样搬进去，就会出现**两条 system**（L1 一条 + L3 一条），既浪费 token 又可能让模型困惑。
   这条规则把「system 只有一条」从「大家都记得别加」变成了「框架替我保证」；
2. **L2 的配额裁剪走 `_fit_layer`** —— 记忆层不是靠 LLM 摘要压，而是**先尝试卸载、再丢最老的条目**。
   因为记忆条目本来就是独立可丢的，为它单独调一次 LLM 得不偿失。

#### 分层配额：按比例而非绝对值

```python
        self.layer_quotas = layer_quotas or {"system": 0.35, "memory": 0.25, "files": 0.25}

    def _quota_for(self, layer_name: str) -> int:
        if layer_name in self.layer_quotas:
            return int(self.budget.available * self.layer_quotas[layer_name])
        fixed = sum(int(self.budget.available * r) for r in self.layer_quotas.values())
        return max(1, self.budget.available - fixed)    # history 吃剩下的弹性额度
```

设计意图：**给「会疯长」的层设上限，让「天生有界」的层吃剩余**。

- `system` / `memory` / `files` 都是「可能有、也可能没有」的层，各自设 25%~35% 上限，防止某一层独吞预算；
- `history` 是唯一**必然存在且会持续增长**的层，所以让它吃剩余的弹性额度——历史本来就该是主角；
- 比例之和不必为 1（这里 0.85），剩下的 0.15 就是弹性空间。

#### 全局兜底：极端情况再丢一次 history

```python
    def _enforce_global_budget(self, warnings) -> None:
        if not self.window.token_count(self.counter) > self.budget.available:
            return
        history = self.window.get_layer("history")
        if history is None:
            warnings.append("已无可压缩的历史层，上下文可能超出模型窗口")
            return
        groups = _group_messages(history.messages)
        dropped = 0
        while groups and self.window.token_count(self.counter) > self.budget.available:
            if len(groups) <= self.compressor.keep_recent:    # 保底：最近 N 组不丢
                break
            dropped += len(groups.pop(0))
        ...
```

**为什么分层配额之后还要全局兜底**：分层配额是「各管各的」，但整窗之和仍可能超预算
（比如 system 0.35 + memory 0.25 + files 0.25 + history 弹性 = 可能超过 1）。
兜底逻辑再算一次总量，从最老的 history 组继续丢，并且**绝不丢最近的 `keep_recent` 组**——
那是模型理解当前任务最需要的部分。

#### observe 与 BuildReport

```python
    def observe(self, tool_name: str, content: str) -> str:
        """工具结果回来后调一次：超长就卸载，返回可放进 tool 消息的文本"""
        return self.offloader.offload(tool_name, content)
```

`observe()` 只是 `offloader.offload()` 的一层薄封装，存在的意义是**给 Agent 一个稳定的调用点**：
即使将来卸载策略从「落盘」换成「存数据库」或「转摘要」，Agent 侧的调用代码一个字都不用改。

`BuildReport` 是给日志和调试用的：`describe()` 会打印出「本轮上下文由哪几层构成、各占多少 token」，
配合 `verbose=True` 使用，排查「为什么这轮这么贵」时一眼就能看出是哪一层在膨胀：

```
上下文构成：3 层 / 8 条消息 / 1240 token
  system     1 条   820 token  required
  history    6 条   320 token
  user       1 条   100 token  required
```

### 4.6 `tools.py` —— 让模型自己管理上下文

这些工具让「Agent 自己管理自己的上下文」成为可能，也是对齐记忆模块「自动 + 工具双路径」的成功经验。

已有 `ReadFileTool` 承担读文件，这里补上其余三个，凑齐路线图 F3 要求的 **`ls` / `read` / `write` / `edit` 四件套**：

| 工具 | 作用 | 典型场景 |
|---|---|---|
| `read_offloaded` | 按 offload_id 回读被卸载的工具结果 | 上下文里出现「[工具结果已卸载]」时 |
| `write_file` | 写文件（默认覆盖，`append=True` 追加） | 把中间结果、草稿、清单落盘 |
| `edit_file` | 精确字符串替换 | 小步修改文件，比整篇重写省 token |
| `list_files` | 列目录（等价 `ls`） | 不确定文件放哪时先看结构 |

#### 路径沙箱：与 ReadFileTool 完全一致

```python
def _resolve_in_workspace(workspace: Path, path: str) -> Optional[Path]:
    target = (workspace / path).resolve()
    try:
        target.relative_to(workspace)
    except ValueError:
        return None            # 越界 → 调用方返回 ToolResult.fail，不做任何 IO
    return target
```

**这不是可选项而是底线**：模型可以生成任意的路径字符串，`../../.ssh/id_rsa` 这类输入必须在内层就被挡住。
先 `resolve()` 消掉 `..` 和符号链接，再用 `relative_to()` 验证，是这个仓库所有文件类工具的统一做法。

#### 二进制拦截

```python
_BINARY_EXTS = {".png", ".jpg", ..., ".pdf", ".zip", ".so", ".pyd", ".pyc", ".db"}
...
        if target.suffix.lower() in _BINARY_EXTS:
            return ToolResult.fail(f"拒绝以文本方式写入二进制类型文件（{target.suffix}）")
```

以文本模式往 `.png` / `.db` 里写东西，结果是文件被写坏且无法恢复。**宁可拒绝，也不制造损坏文件。**

#### edit_file 的失败提示要能教会模型

```python
        occurrences = original.count(old_text)
        if occurrences == 0:
            return ToolResult.fail(
                "未找到要替换的原文。请先用 read_file 查看文件，"
                "确保 old_text 与文件内容逐字符一致（含缩进与空行）。"
            )
```

模型做精确替换最常见的失败原因是「凭记忆写 old_text」，缩进和空行对不上。
错误提示里直接给出**下一步动作**（先用 `read_file` 看原文），比干巴巴一句「未找到」有效得多——
这也是 `_render_observation` 给错误加前缀「请检查参数或换一个工具/查询方式重试」的同一思路。

#### 工厂：按需组装

```python
def build_context_tools(workspace=".", offloader=None, include_filesystem=True) -> List[ToolBase]:
    tools = []
    if offloader is not None:
        tools.append(ReadOffloadedTool(offloader))     # 没卸载器就不给回读工具
    if include_filesystem:
        tools.extend([ListFilesTool(workspace), WriteFileTool(workspace), EditFileTool(workspace)])
    return tools
```

**只有传了 `offloader` 才注册 `read_offloaded`**：没有卸载器就不可能有卸载内容，注册了只会让模型看到一个永远查不到东西的工具，
白白占用 schema token、还可能诱导模型去调它。这个判断让 `build_context_tools` 在「只要文件系统」和「完整 context」两种场景下都能直接用。

### 4.7 `ReActAgent` —— context 已内置于主类

和记忆一样，context 也是 `ReActAgent` 的**内置能力**，不需要单独的子类。新增四个构造参数：

```python
        ctx: Optional["ContextBuilder"] = None,     # 手动注入的构建器（可选）
        enable_context: bool = False,               # 是否启用（默认关，保持既有行为）
        context_budget: Optional[int] = None,       # 覆盖 ctx 的窗口预算
        context_offload_threshold: int = 4_000,     # 触发卸载的字符数
```

#### 初始化顺序：放在记忆之后就位

```python
        # 系统提示词，全程只注入这一次
        self.refresh_system_prompt()

        # 记忆：默认开启
        self.enable_memory = enable_memory
        self.memory = (memory or MemoryManager(llm=llm)) if enable_memory else None

        # 上下文工程：默认关闭；放在记忆之后就位，因为 build 要用到 memory 的历史与长期记忆
        self.enable_context = enable_context
        if enable_context:
            self._init_context(ctx, context_budget, context_offload_threshold)
        else:
            self.ctx: Optional["ContextBuilder"] = ctx
```

**顺序不能反**：context 的拼装要用 `memory.buffer` 的历史和 `memory.retrieve()` 的长期记忆，
所以必须在记忆初始化之后建立。这也解释了为什么 `_init_context` 里要 `refresh_system_prompt()`——
注册了四个新工具，system 里的工具清单必须同步刷新，否则模型根本不知道有这些工具。

#### `_init_context`：注册工具 + 复用地址

```python
        if ctx is None:
            token_budget = TokenBudget(max_context_tokens=budget) if budget else TokenBudget()
            ctx = ContextBuilder(
                workspace=".", budget=token_budget,
                counter=TokenCounter(model=getattr(self.llm, "model", None) or "gpt-4o-mini"),
                offloader=ContextOffloader(workspace=".", threshold=offload_threshold),
                llm=self.llm,
            )
        self.ctx = ctx
        for t in build_context_tools(workspace=".", offloader=ctx.offloader):
            self.tool_registry.register_tool(t)
        self.refresh_system_prompt()
```

注意 `workspace="."` 与 `offloader` 用的是**同一个 workspace**——卸载目录和文件工具必须指向同一个根，
否则会出现「`write_file` 写在 A 目录、卸载落在 B 目录」的割裂。

> **一个已知的粗糙点**：这里 workspace 硬编码为 `"."`（当前工作目录）。
> 更合理的做法是从配置或构造参数传入，当前保持与 `ReadFileTool` 默认行为一致。
> 如果你的部署需要固定工作目录，建议显式构造 `ContextBuilder(workspace=...)` 后通过 `ctx=` 注入。

#### `_fit_context`：用拼装结果重建工作区

```python
        system_prompt = next((m.content for m in self.messages.get_messages()
                              if m.role == "system"), None)
        history = self.memory.buffer.get_messages() if self.memory else []
        packets = self._collect_context_packets(input_text)

        self.ctx.reset_turn()
        window = self.ctx.build(system_prompt=system_prompt, history=history,
                                user_input=input_text, memory_packets=packets)
        self.messages.clear()
        for message in window.flatten():
            self.messages.append(message)
```

四个要点：

1. **system 从 `self.messages` 里取**（`refresh_system_prompt` 写的那条），保证工具清单、技能索引是最新的；
2. **history 只从 `memory.buffer` 取**，`self.messages` 里的东西一律不参与——严格守住单一数据源；
3. **`reset_turn()` 先清空上一轮残留**，避免窗口越拼越大；
4. **拼完 `clear()` 再整体写回 `self.messages`**，而不是追加。这样工作区的内容**完全由 window 决定**，
   下一步循环里的 tool 链才是在这个干净基础上往上加。

#### `_collect_context_packets`：只收记忆，不收历史

```python
    def _collect_context_packets(self, query: str):
        """把记忆层的「长期记忆 + 会话摘要」整理成 ContextPacket 列表

        注意：buffer 的历史不由这里渲染（避免和 history 层重复注入）
        """
        packets = []
        records = self.memory.retrieve(query, limit=self.memory_limit)
        for r in records:
            packets.append(f"[{r.kind}] {r.content}")
        summary = self.memory.buffer.get_summary()
        if summary:
            packets.append(f"[历史摘要] {summary}")
        return packets
```

**这里的不变量是：`buffer.get_messages()`（历史）只走 history 层，`retrieve()` + `get_summary()`（记忆）只走 memory 层，两者不交叉。**
`记忆模块编写说明.md` 3.4 节费了很大篇幅解决的「历史重复注入」，
在 context 接入后由这条分工**结构性保证**——不再是靠「记得别加」，而是靠「两条路取了不同的东西」。

#### `_offload_observation`：工具返回后的卸载钩子

```python
    def _offload_observation(self, tool_name: str, observation: str) -> str:
        if self.ctx is None or not observation:
            return observation            # 没开 context → 原样返回
        try:
            return self.ctx.observe(tool_name, observation)
        except Exception as e:
            print(f"⚠️ 工具结果卸载失败，保留原文：{e}")
            return observation            # ← 卸载失败必须退回原文，不能让工具链断掉
```

它在主循环里的位置很关键——**在工具结果渲染之后、放进 `tool` 消息之前**：

```python
                        result = self.tool_registry.invoke_tool(tool_call.name, tool_call.arguments)
                        observation = _render_observation(result)          # 按 status 加前缀
                        observation = self._offload_observation(tool_call.name, observation)  # ← 卸载
                        if self.skill_middleware is not None:
                            observation = self.skill_middleware.after_tool_call(...)
                        tool_msg = Message("tool", observation)
                        tool_msg.metadata = {"tool_call_id": tool_call.id, "name": tool_call.name}
```

顺序是刻意安排的：**先渲染前缀、再卸载**。因为 `_render_observation` 会给错误结果加上
「[工具执行错误] …… 请检查参数或换一个工具重试」这样的引导语，这是模型纠错的关键信息，
必须在被卸载之前就拼进文本——否则卸载后模型只看到引用 id，反而丢了纠错线索。

### 4.8 ⭐ 卸载阈值定多少？—— 一个需要实测的参数

`context_offload_threshold` 默认 4000 字符。这个数字不是拍脑袋来的，它同时受两个约束：

| 约束 | 上限方向 | 说明 |
|---|---|---|
| **模型注意力** | 越小越好 | 单条 tool 结果几千字符后，模型对其中部的利用率明显下降，「lost in the middle」 |
| **回读成本** | 越大越好 | 每卸载一次，模型可能要多一轮 `read_offloaded`，多一轮 = 多一次 LLM 调用 = 多一次钱和延迟 |
| **落盘开销** | 越大越好 | 太小的结果落盘，读写文件的开销比省下的 token 更贵 |

实测经验（以 `read_file` 类工具为例）：

- **2000 字符以下**：建议不卸载。这个量级模型能完整消化，卸载属于负优化；
- **4000~8000 字符**：默认区间。一次 `read_file` 读 200 行大约就落在这里；
- **10000 字符以上**：强烈建议卸载。这类结果通常是「整篇文档」或「搜索结果页」，
  模型一次性也读不完，卸载后按需分段读反而更准。

**调参建议**：如果你的任务里模型经常需要「从长结果里找一小段」，阈值可以调小（比如 2000），
迫使模型用 `read_offloaded` + 行区间精准定位；
如果任务里模型倾向于「通读长结果再总结」，阈值调大（比如 8000）能减少一次往返。

---

## 五、上手：四种典型用法

### 用法 1：最简——只要一个带预算的上下文

```python
from fast_agents.context import ContextBuilder, TokenBudget

builder = ContextBuilder(
    workspace=".",
    budget=TokenBudget(max_context_tokens=32_000, reserve_for_output=4_000),
    verbose=True,                     # 每轮打印上下文构成，调优时打开
)

window = builder.build(
    system_prompt="你是一个严谨的技术助手。",
    history=[...],                    # 通常是 memory.buffer.get_messages()
    user_input="帮我看看这个报错怎么解",
)
messages = window.to_openai_messages()    # 直接喂给 LLM
print(builder.stats())
# {'tokens': 1240, 'budget': 27000, 'usage_ratio': 0.046, 'layers': ['system','history','user'], ...}
```

### 用法 2：标准——工具结果自动卸载 + 按需回读

```python
from fast_agents.context import ContextBuilder, ContextOffloader, ReadOffloadedTool

offloader = ContextOffloader(workspace=".", threshold=4_000)
builder = ContextBuilder(workspace=".", offloader=offloader, offload_threshold=4_000)

raw = some_tool.run(...)                       # 假设返回 2 万字符
observation = builder.observe("web_search", str(raw))
# → 落盘 .fast_agents/context/offload/<id>.txt，observation 变成一段带 offload_id 的引用
print(observation)

# 模型想看全文时，通过工具回读（也可以直接调 offloader.read）
tool = ReadOffloadedTool(offloader)
print(tool.run(offload_id=offloader.last_id, start_line=1, max_lines=50))
```

### 用法 3：完整——接进 Agent（推荐）

```python
from fast_agents.agents.react_agent.react_agent import ReActAgent
from fast_agents.memory import MemoryManager
from fast_agents.llm import LLM

llm = LLM()                                    # 读 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL

agent = ReActAgent(
    name="上下文助手",
    llm=llm,
    memory=MemoryManager(llm=llm),             # 记忆与 context 配合使用
    enable_context=True,                       # ← 打开上下文工程
    context_budget=128_000,                    # 模型窗口（按你的模型改）
    context_offload_threshold=4_000,           # 超过 4 千字符的工具结果自动卸载
)
agent.run("把仓库里所有 TODO 找出来，列成清单写到 notes/todo.md")
```

打开后 Agent 会自动：注册 `read_offloaded` / `write_file` / `edit_file` / `list_files`、
每轮按 token 预算分层拼装上下文、超长工具结果落盘并把引用位置告诉模型。

### 用法 4：自定义——注入自己的分层配额与压缩策略

```python
from fast_agents.context import ContextBuilder, ContextCompressor, TokenBudget

# 只卸载 + 摘要，不要裁剪（宁可多花一次 LLM 调用，也不丢中间历史）
compressor = ContextCompressor(
    llm=llm,
    offloader=offloader,
    enable_offload=True,
    enable_trim=False,          # ← 关掉裁剪
    enable_summary=True,
    summary_max_words=300,
    keep_recent=6,              # 至少保留最近 6 条
)

builder = ContextBuilder(
    workspace=".",
    budget=TokenBudget(max_context_tokens=200_000, reserve_for_output=8_000),
    compressor=compressor,
    layer_quotas={"system": 0.30, "memory": 0.30, "files": 0.20},   # history 吃剩余 0.20 弹性
    verbose=True,
)

agent = ReActAgent("助手", llm=llm, ctx=builder, enable_context=True)
```

**`ctx=` 与 `enable_context=True` 要一起用**：`ctx` 只负责「用哪个构建器」，`enable_context` 才决定「开不开」。
只传 `ctx` 不传 `enable_context=True` 时，构建器会被保存但**不生效**（这样你可以先构造好、晚一点再开）。

---

## 六、常见坑与避坑清单

| 坑 | 现象 | 解法 |
|---|---|---|
| **user 输入重复注入** | 开启 context 后，第一次 LLM 调用里出现两条相同的 user 消息 | 主循环追加用户输入前判一次 `not (self.enable_context and self.ctx is not None)`（见 3.4） |
| **记忆重复注入** | 长期记忆既在 memory 层、又拼在 user 文本里 | `_fit_context()` 后把 `user_input` 还原为 `input_text`；记忆只走 memory 层 |
| **system 出现两条** | 提示词变长、模型行为怪异 | L1 负责唯一的 system，history 层里的 system 一律 `continue` 丢弃 |
| **tool 配对被拆坏** | 400：`tool` 消息缺少对应的 `tool_call_id` | 裁剪/摘要一律以**消息组**为单位（`_group_messages`）；用 `window.validate()` 自检 |
| **没预留输出空间** | 请求成功但回答被截断，或直接 400 | `TokenBudget.reserve_for_output` 必须留，别把窗口全给输入 |
| **中文被严重低估** | 明明看着不多，一算超了 | 中文必须按「1 字 ≈ 1 token」估，别套英文的 4 字符规则 |
| **漏算工具 schema** | 消息才占 60%，加 tools 就超了 | 用 `counter.count_tool_schemas(tools)` 单独算一次并计入预算 |
| **卸载提示不清楚** | 模型反复重试原工具，或干脆无视卸载内容 | 引用文本必须给全「卸载了什么 + 预览 + 怎么取回」（见 4.3） |
| **卸载失败拖垮对话** | 磁盘满 / 目录无权限 → 整个 run 崩 | `offload()` 内部 catch，退回「截断 + 提示」；`_offload_observation` 再兜一层 |
| **摘要失败拖垮对话** | 压缩时 LLM 一超时，用户拿不到回答 | `_call_summarize` 双层降级（LLM 失败 → 截断 → `keep_recent` 保底） |
| **卸载目录无限膨胀** | 长跑服务磁盘被写满 | `max_files`（默认 200）按最早创建淘汰 |
| **路径穿越** | 模型生成 `../../etc/passwd` | `resolve()` + `relative_to(workspace)` 双重校验；offload_id 走十六进制白名单 |
| **往二进制文件写文本** | `.png` / `.db` 被写坏且不可恢复 | `_BINARY_EXTS` 白名单拦截，宁可拒绝也不损坏 |
| **工作目录分裂** | `write_file` 写 A 目录、卸载落 B 目录 | `ContextBuilder`、`ContextOffloader`、文件工具必须用同一个 workspace |
| **窗口不重置** | 每轮拼装结果越堆越多 | 每轮 `build()` 前调 `reset_turn()`；`build()` 内部本身也会重建 window |
| **缓存失效** | 成本突然翻倍 | 动态内容（记忆/历史/文件）一律排在静态内容之后；绝不让变动内容进 system 中段 |
| **摘要压完就忘** | 压完的历史下一轮又冒出来 | 摘要以 `required=True` 的消息写回 history 层开头，而不是丢掉 |

---

## 七、下一步演进路线

按优先级排：

1. **语义召回层（记忆与文件统一召回）** —— 现在 memory 层只装 `retrieve()` 的词法召回结果，
   `ContextPacket.metadata` 字段已留好扩展位。下一步给 `MemoryStore` 加 `search_semantic()`，
   让 builder 能按语义把「最相关的 N 个片段」排进 memory 层，而不是按词法重叠；
2. **Prompt caching 断点显式化** —— 给 `ContextLayer` 加 `cache_breakpoint: bool`，
   在 `flatten()` 时对 system 层（或 system + memory）打 `cache_control` 标记。
   需要接入支持缓存的模型服务（Anthropic / 部分兼容代理）。这是当前**收益最明确、改动最小**的一项；
3. **压缩中间件化（Align LangGraph SummarizationNode）** —— 把 `ContextCompressor` 抽象成
   `ContextMiddleware` 接口（`before_build` / `after_build` 钩子），让压缩策略可以像 `SkillMiddleware`
   一样在构造时注入，而不是改 `builder.py`。目前三级策略已经是「可插拔开关」，
   升级成「可插拔实现」是自然的一步；
4. **子代理隔离的上下文（对齐 CrewAI / DeepAgents）** —— 这一项和 `agents` 模块的 E3 重合：
   子代理用**独立窗口**执行子任务，只把结果返回主 Agent。这是「用隔离代替压缩」的另一条路，
   长任务场景下比任何压缩策略都有效；
5. **跨会话的卸载目录治理** —— 现在 `max_files=200` 是简单淘汰。下一步可按 `session_id` 分目录，
   会话结束时可选择归档或清理；`index.json` 里已经存了 `tool_name` / `created_at`，够做进一步统计；
6. **卸载内容的智能摘要** —— 现在卸载时保留的是「原文前 200 字符预览」。
   可以改成「调一次小的 LLM 摘要」，让引用文本更准确地告诉模型「这里面有什么」，
   减少不必要的回读。代价是卸载从 O(1) 变成 O(LLM 调用)，建议做成可选项；
7. **多模态内容块的计数精细化** —— `count_message_tokens` 目前把非文本块统一按 85 token 粗算。
   不同分辨率的图片实际 token 差异很大，可以按图片尺寸分档；
8. **与 `tools` 模块的截断协议统一** —— 路线图 tools 模块计划做「统一返回协议 `ToolResponse` + 工具结果截断」，
   那一层是「工具自己截断」（丢信息），context 这层是「卸载」（不丢信息），
   两者需要明确分工：**工具截断管「单次结果是否可读」，context 卸载管「上下文是否装得下」**。

---

## 八、测试与验证

配套三个可运行文件：

```bash
# 1. context 模块单元测试（16 项，不依赖真实 LLM）
python test/test_context.py

# 2. ReActAgent 集成测试（含 context 接入的 2 项专项回归）
python test/test_react_agent.py

# 3. 端到端演示（5 个示例，全部离线可跑）
python examples/context/context_react_agent_demo.py
```

**`test_context.py`（16 项）** 覆盖：

```
✅ test_token_counter                          计数后端与估算降级
✅ test_token_budget                           预算换算 / 预留输出 / 使用率
✅ test_window_layer_order                     分层顺序：system → memory → history → files → user
✅ test_window_validate_tool_pairing           validate 能抓出孤立的 tool 消息
✅ test_offloader_threshold                    未超阈值不落盘（零 IO）
✅ test_offloader_offload_and_read             卸载 → 引用文本 → 按 id 回读全文
✅ test_offloader_read_line_range              行区间分段读取 + 续读提示
✅ test_offloader_max_files_eviction           超出 max_files 按最早创建淘汰
✅ test_offloader_safety                       offload_id 白名单拦截路径穿越
✅ test_compressor_trim_keeps_tool_pairing     成组裁剪，tool 配对不被拆散
✅ test_compressor_offload_level               第 1 级：卸载超长消息
✅ test_compressor_summary_level               第 3 级：摘要压缩并写回 layer
✅ test_builder_pipeline_and_order             五层拼装顺序 + system 去重
✅ test_builder_budget_trims                   超预算时的分层配额与全局兜底
✅ test_context_filesystem_tools               write / edit / list 的沙箱与二进制拦截
✅ test_read_offloaded_tool_and_factory        read_offloaded 工具 + 工厂按需组装
```

**`test_react_agent.py` 中与 context 相关的 2 项**：

```
✅ test_context_enabled_builds_and_registers_tools  开启后注册 4 个工具、system 仍只 1 条、user 只 1 条
✅ test_context_user_input_not_duplicated           多工具步场景下 user 输入不随步数累积
```

其中 `test_context_user_input_not_duplicated` 是最关键的一条回归断言——
它把「本轮用户输入的出现次数」**锁死在 1 次**，防止 3.4 节那个「user 重复注入」的 bug 被后续改动重新引入。

**`examples/context/context_react_agent_demo.py` 的 5 个示例**：

```
示例 1  分层拼装与 token 预算     看「上下文由哪几层组成、各占多少 token」
示例 2  超长工具结果卸载          2 万字符 → 落盘 → 按需 read_offloaded 回读
示例 3  三级压缩                  卸载 → 裁剪 → 摘要，逐级降级
示例 4  文件系统工具              write / list / edit，会话状态外化
示例 5  ReActAgent 接入 context   脚本化假 LLM，离线跑通完整 ReAct 循环
```

示例 5 会打印出 Agent 每轮实际送给 LLM 的上下文构成，可以直接看到「分层拼装 + 预算裁剪」在真实循环里的效果：

```
Agent 本轮送进 LLM 的上下文构成：
  第 1 次调用：2 条消息 / 约 677 token
    - [system   ] 你是一个具备推理和行动能力的AI助手。...
    - [user     ] 把「上下文工程 = 计量 + 分层 + 治理」这个结论写进 notes/结论.md
  第 2 次调用：4 条消息 / 约 762 token
    - [system   ] ...
    - [user     ] ...
    - [assistant]  (+tool_calls)
    - [tool     ] ✅ 已写入 notes/结论.md（28 字符，文件当前 56 字节）
```

---

## 附：与仓库其他模块的接口速查

| 上游 / 下游 | 接口 | 说明 |
|---|---|---|
| **memory → context** | `memory.buffer.get_messages()` | 进 history 层（历史唯一数据源） |
| **memory → context** | `memory.retrieve(query, limit)` | 进 memory 层（长期记忆参考） |
| **memory → context** | `memory.buffer.get_summary()` | 进 memory 层（会话摘要） |
| **context → messages** | `window.to_openai_messages()` | 按层序转成 OpenAI 格式，直接喂 LLM |
| **context → messages** | `window.flatten()` + `Messages.append()` | 重建 `self.messages` 工作区 |
| **tools → context** | `ToolResult.output` | 经 `_render_observation` 加前缀后交给 `observe()` |
| **context → tools** | `build_context_tools(workspace, offloader)` | 生成 4 个工具，`agent.add_tool()` 或直接注册 |
| **context → skills** | 卸载发生在 `after_tool_call` **之前** | 顺序见 4.7，保证纠错前缀不被卸载吞掉 |

---

> **一句话总结本模块的设计**：上下文工程 = **计量（知道有多少）+ 分层（知道放哪、先砍谁）+ 治理（超了怎么办）**。
> 三者各占一个文件、各解决一个问题，拼起来就是 `ContextBuilder` 的一次 `build()`——**没有魔法，全是账**。
