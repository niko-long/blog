# MCP 面试题

## MCP 协议入门与实践面试题

### 什么是 MCP 协议？为什么有了 Function Call 还需要 MCP？

**MCP（Model Context Protocol，模型上下文协议）\**是由 Anthropic 开源发布的标准协议。它的核心思想是充当\**大模型世界的“USB 接口”**，让任何实现了 MCP 协议的工具都能被任何支持 MCP 的客户端无缝调用，从而实现跨语言、跨平台的即插即用。

有了 Function Call，模型虽然能自己判断调用时机并输出标准化意图，但在**工具规模化**后会暴露出四大痛点，这就是需要 MCP 的原因：

1. **工具定义的维护噩梦**：Function Call 需要在客户端手写大量 JSON Schema 代码来定义工具，工具一旦增多或修改，极难维护且容易出现定义与实现不同步的 Bug。
2. **跨语言跨系统的集成困境**：不同团队（如 Java、Python、第三方 API）提供的工具，需要在客户端写多套适配逻辑。
3. **权限和安全的黑洞**：Function Call 自身无权限机制，业务代码与权限校验混杂，容易产生漏洞。
4. **可观测性的缺失**：Function Call 缺乏统一的协议层面的日志监控和链路追踪。

**总结来说：** Function Call 解决了“模型如何调工具”的问题（相当于发动机），而 MCP 解决了“系统如何管理和统筹工具”的问题（相当于整车框架）。

### MCP 协议是用来替代 Function Call 的吗？它和 Spring AI 这种实现框架的关系是什么？

**不是替代，而是增强。** MCP 底层依然依赖大模型的 Function Call 能力（让模型判断调用时机并输出 JSON 参数），MCP 的作用是在其之上加了一层标准化的管理框架。

**两者的核心区别在于：**

- **工具定义位置**：Function Call 的工具定义在客户端（手写 JSON），MCP 的工具定义在 Server 端（实现代码与定义一体），客户端通过协议动态获取。
- **跨语言支持**：Function Call 每个系统各自实现适配，MCP 只要遵循协议，Java/Python/TS 都能互通。

**MCP 协议与 Spring AI 框架的关系：**

- **MCP 是协议层面（标准制定者）**：定义了工具元数据的 JSON 标准、JSON-RPC 通信协议和工具动态发现机制。
- **Spring AI 是框架层面（协议实现者）**：它提供了 `@McpTool`（1.0版本为 `@Tool`）注解等便利性封装，帮你自动扫描方法生成符合 MCP 标准的 JSON Schema，并自动处理 JSON-RPC 请求解析，让 Java 开发者免于手写底层通信和元数据代码。

### 能详细描述一下 MCP 协议的系统架构和它的三大核心能力吗？

MCP 采用的是 **Host - Client - Server 三层架构**：

1. **Host（宿主应用）**：用户直接交互的应用（如 Claude Desktop、Cursor 或企业自研助手）。负责接收输入、调模型、展示结果。
2. **Client（MCP 客户端）**：Host 内部的通信组件。**一个 Client 只连接一个 Server**，但一个 Host 可以有多个 Client。负责发现工具、发送请求和接收响应。
3. **Server（MCP 服务端）**：提供工具的一方。声明自己拥有哪些工具，并执行实际的工具逻辑。

**工作流程**：Client 启动时会自动从 Server **动态获取**工具列表和参数元数据（解决了手写 JSON Schema 的痛点） -> 传给大模型 -> 模型决定调用 -> Client 发送请求到 Server -> Server 执行并返回。

**MCP 的三大核心能力：**

1. **Tools（工具调用）**：最核心能力。Server 定义具体操作（如查年假），Client 发现并调用。与 Function Call 直接对应，执行具体动作。
2. **Resources（资源访问）**：Server 暴露静态数据供 Client 读取（如文件内容、数据库记录），为模型提供额外上下文。
3. **Prompts（提示词模板）**：Server 提供预定义的交互模板（如“代码审查模板”），规范标准化交互。

### MCP 客户端和服务端之间是如何通信的？在实际项目中应该如何选择？

MCP 协议定义了两种传输方式：**Stdio（标准输入输出）** 和 **Streamable HTTP（可流式 HTTP）**。

| 维度           | Stdio (标准输入输出)                                         | Streamable HTTP (可流式 HTTP)                       |
| -------------- | ------------------------------------------------------------ | --------------------------------------------------- |
| **通信机制**   | 通过本地子进程的 stdin / stdout 通信。                       | 通过 HTTP POST 请求和 SSE（支持流式响应）通信。     |
| **网络与部署** | 无需网络配置，极度安全；Server 必须和 Client 在同一台机器。  | 需网络和端口配置（支持远程）；可跨机器部署。        |
| **客户端支持** | 一对一连接，不支持多客户端共享。                             | 多对一连接，支持团队多个 Client 共享同一个 Server。 |
| **避坑指南**   | **绝对不能向 stdout 输出业务日志或启动 Banner**，否则会干扰 JSON-RPC 协议解析。 | 生产环境中需自行处理 HTTPS 及鉴权问题。             |

**如何选择：**

- **本地开发、个人工具、单机插件**（如接入 Claude Desktop 本地查文件）：选择 **Stdio**，简单安全。
- **团队共享、企业级生产环境、内部工具平台**：选择 **Streamable HTTP**，便于分布式部署和共用。

### 在 Java (Spring Boot) 项目中如何搭建并配置一个 MCP Server？它在 RAG 系统中如何发挥作用？

**搭建与配置步骤（使用 Spring AI）：**

1. **引入依赖**：使用 `spring-ai-starter-mcp-server`（Stdio）或 `spring-ai-starter-mcp-server-webmvc`（HTTP）。
2. **定义工具**：创建一个 `@Service` 类，在方法上加上 `@Tool` (或 `@McpTool`) 注解，并通过 `@ToolParam` 描述参数含义。框架会自动提取它们生成 MCP 协议所需的元数据。
3. **注册工具**：配置 `MethodToolCallbackProvider` 扫描上述工具类。
4. **配置文件 (`application.yml`)**：配置 `spring.ai.mcp.server` 的 `name`、`version` 和 `type`。**注意：如果是 Stdio 模式，必须关闭 Spring Boot Banner 和 控制台日志输出。**

**MCP 在 RAG 系统中的作用：**

1. **统一检索工具**：将“搜索企业知识库”本身封装成一个 MCP 工具。这样不仅自身的系统能用，任何支持 MCP 的 IDE (如 Cursor) 也能直接检索公司知识库。
2. **多工具架构编排**：实现知识检索与业务工具的解耦。RAG 宿主应用并行调用多个职责单一的 MCP Server（如一个处理订单、一个处理知识检索、一个处理 HR 数据），系统扩容只需新增 Server，完全不需要改动现有核心代码。

### 目前 MCP 的生态发展现状如何？在生产环境落地还有哪些局限性？

**生态现状：** 支持非常广泛， Anthropic 官方的 Claude Desktop、热门 AI IDE (Cursor、Windsurf)、各种 VS Code 插件都已完整支持。社区也涌现了大量现成的 Server（如 GitHub、MySQL、Slack、文件系统），开箱即用。

**生产环境落地的局限性：**

1. **协议仍在快速演进**：API 和传输机制（如从 HTTP+SSE 到 Streamable HTTP）变动较快，存在版本兼容问题。
2. **部分客户端支持不完整**：有些客户端仅支持 Stdio 不支持 HTTP，或仅支持 Tools 不支持 Resources。
3. **权限和认证未标准化**：MCP 协议目前在细粒度权限（RBAC 角色控制）和远程 Server 认证（如 OAuth, API Key）上尚未形成统一的标准规范，企业落地时需要自己写额外代码去实现拦截和校验逻辑。
4. **高并发稳定性**：相比于成熟的微服务框架，MCP 协议在极高并发的生产环境中的性能和稳定性还有待大规模验证。

## MCP 详细面试题

### MCP Server的三种部署方式

| 方式      | 适用场景                 | 特点                                                         |
| --------- | ------------------------ | ------------------------------------------------------------ |
| **Stdio** | 本地工具和命令行程序     | Agent启动时拉起MCP Server子进程，通过标准输入输出通信。简单、安全，不需要网络通信 |
| **SSE**   | 远程服务且需要服务端推送 | MCP Server以网络服务形式运行，客户端通过HTTP长连接接入。适合监控告警、实时数据流等 |
| **HTTP**  | 远程服务且无状态调用     | 客户端通过HTTP短连接调用，每次请求-响应后连接断开。适合简单的查询类工具 |

**Stdio方式详解**：

比如在Claude Code的配置文件中，可以配置一个filesystem的MCP Server，启动命令是`"npx -y @modelcontextprotocol/server-filesystem"`。

当Claude Code启动时：

1. 执行这个命令拉起一个子进程
2. 通过stdin把大模型的工具调用请求发给这个子进程
3. 子进程执行完毕后通过stdout返回结果

### MCP协议中的Tools、Resources、Prompts三大核心能力有什么区别？在实际需求中如何做技术选型？

这三者在 MCP 协议中各有定位，互补而非竞争。其核心区别如下：

- **本质与控制方**：
  - **Tools（工具）**：本质是**执行操作**。由**模型驱动**（模型分析意图后决定何时调用、调用哪个）。
  - **Resources（资源）**：本质是**提供数据**。由**应用驱动**（Host 应用或用户决定读取哪些资源作为上下文给模型）。
  - **Prompts（提示词）**：本质是**提供模板**。由**Client 驱动**（通常由用户在 UI 中选择触发）。
- **副作用**：
  - **Tools**：可能有副作用（如写数据库、发邮件、下订单），可读可写。
  - **Resources / Prompts**：无副作用，纯只读。
- **返回内容**：
  - **Tools**：操作结果（结构化内容）。
  - **Resources**：资源数据（文本 text 或二进制 blob）。
  - **Prompts**：封装好的 messages 数组。

**技术选型（如何判断用哪个）：**

- 如果操作会产生副作用，或需要模型自主判断是否执行，使用 **Tools**（类比：去餐厅点菜，触发厨房动作）。
- 如果操作只读无副作用，仅作为固定上下文提供给模型，使用 **Resources**（类比：看餐厅菜单）。
- 如果需要将经过验证的最佳实践（如格式、角色、兜底策略）封装为标准化的交互流程，使用 **Prompts**（类比：使用标准化的点菜表单）。

### 什么是 MCP 中的 Resources（资源）？在预加载上下文场景下，为什么推荐用 Resources 而不是 Tools？

**Resources 的概念**： Resources 的核心作用是 **Server 暴露数据，Client 来读取，给模型提供上下文**。它提供的是只读的参考资料，调用过程没有任何副作用。

**为什么预加载上下文推荐用 Resources 而不是 Tools**： 如果使用 Tools 来获取诸如“用户资料、近期订单、退货政策”等上下文，模型需要先分析并决定调用哪些工具，这会带来两个问题：

1. **额外的推理和延迟**：模型需要先输出 `tool_calls`，经过网络往返后才能给出最终回答。
2. **漏调用的风险**：模型可能无法每次都准确判断出需要调用所有相关工具。

而使用 Resources，Host 应用可以主动、提前把该给的上下文一次性读出并塞给模型，**模型不用猜，没有额外的推理链路和工具调用延迟，也不会漏掉关键信息**。

### MCP 定义了哪两种 Resources 类型？分别适用于什么场景？

MCP 协议定义了两种资源类型：

- **直接资源（Direct Resources）**：
  - **特点**：URI 是固定的，不带参数（如 `docs://return-policy`）。任何时候访问都是指向同一份数据。
  - **适用场景**：数量有限、相对固定的数据。如：系统公告、服务条款、应用配置、系统运行状态等。
- **资源模板（Resource Templates）**：
  - **特点**：使用遵循 RFC 6570 规范的 URI 模板，带有参数占位符（如 `order://users/{userId}/orders/{orderId}`）。Client 必须填入具体参数后才能访问。
  - **适用场景**：数量不固定、需要动态生成或依赖查询参数的数据。如：特定用户的订单详情、特定日期的日志文件、某张数据库表的结构等。

### Resources 的协议交互流程是怎样的？什么时候需要支持资源变更订阅（Subscribe）？

**协议交互流程**主要包含以下三个核心方法：

1. `resources/list`：Client 获取所有可用的直接资源列表。
2. `resources/templates/list`：Client 获取所有资源模板列表。
3. `resources/read`：Client 传入具体 URI 获取资源内容（返回 `contents` 数组，支持文本或 Base64 编码的二进制数据）。

**关于资源变更订阅（Subscribe）**： Client 可以通过 `resources/subscribe` 订阅资源，当资源变化时，Server 发送更新通知，Client 收到通知后再调用 `resources/read` 拉取最新内容。

- **需要支持订阅的场景**：URI 运行时固定，但底层数据会发生变化。例如：存在数据库/配置中心的动态配置、实时销售看板等。
- **不需要支持订阅的场景**：内容写死在代码常量中，进程不重启就不会变的数据。

### 在成熟的 Java 微服务架构中，直接通过 RPC（如 Feign/Dubbo）查询数据和使用 MCP Resources 有什么区别？

- **单 Client 场景（优势不明显）**：如果 AI 应用只有一个 Client（如自己的后端服务），且内部已经有成熟的微服务体系，直接通过 RPC 查询数据库拼装上下文是最直接、高效的做法。此时绕一圈走 MCP Resources 协议反而增加了抽象层，没有明显收益。
- **多 Client 场景（核心优势）**：MCP Resources 的真正优势在于**跨 Client 共享数据源**。同一个 MCP Server 暴露的资源，可以通过统一的 `resources/read` 协议被 Claude Desktop、Cursor 或自定义的 Web 应用同时读取。数据获取逻辑只需在 Server 端写一次，极大提升了多端接入时的复用性。

### 什么是 MCP 中的 Prompts？相比在客户端直接编写 Prompt，它有什么优势？

**Prompts 的概念**： Prompts 允许 Server 预定义标准化、经过验证的提示词模板（包含角色定义、回答规则、引用要求、兜底策略等）。Client 只需要传入相应的参数，就能获取到一组可以直接用于模型调用的 messages 数组。

**相比客户端手写 Prompt 的优势**：

1. **统一管理与版本控制**：所有 Client 获取到的都是同一版本的 Prompt，修改时只需在 Server 端改一次，所有 Client 立即生效。
2. **最佳实践集中沉淀**：Prompt 工程的经验（如复杂的 RAG 规则）不用散落在各个客户端的代码中，而是集中沉淀在 Server 端。
3. **参数统一校验**：Server 端可以统一定义模板所需的必填/非必填参数及其描述，规范化调用方式。

### Prompts 的协议交互流程是怎样的？获取到的 Prompt 格式有什么特点？

**协议交互流程**：

1. `prompts/list`：Client 查询 Server 提供了哪些 Prompt 模板及所需参数。
2. `prompts/get`：Client 传入模板名称和具体参数，获取组装好的 Prompt。

**返回格式的特点**：

- 返回的不是纯文本字符串，而是一个**完整结构化的 `messages` 数组**。
- MCP 协议规范中，消息的 `role` **只支持 `user` 和 `assistant` 两种角色，不支持 `system`**。
- 如果需要设置系统级别的指令，通常将指令内容放在第一条 `user` 消息里，或者由 Client 拿到 messages 数组后自行解析拆分到 `system` 消息中。

## MCP官方Java-SDK面试题

### 请简述MCP官方Java-SDK的背景和定位，以及它与Spring AI的关系？

**答**：MCP官方Java-SDK是由Anthropic主导的modelcontextprotocol GitHub组织维护的，是MCP协议规范在Java语言的官方实现。它定位为轻量级、纯Java的协议实现，不绑定任何框架，为开发者提供构建MCP Server和Client的基础能力。

与Spring AI的关系是：Spring AI MCP是官方SDK的上层封装，Spring AI在SDK基础上做了三件事：

- 注解驱动的工具注册（`@McpTool`、`@McpResource`等注解）
- 自动配置Transport（根据starter依赖自动选择传输方式）
- Spring Boot生命周期管理（Server启动和关闭与容器生命周期绑定）

大多数项目直接用Spring AI就够了，但在非Spring项目、需要自定义Transport或深度控制连接管理时，需要直接使用官方SDK。

### MCP Java-SDK的模块划分是怎样的？为什么要将JSON序列化单独拆分为独立模块？

**答**：MCP Java-SDK分为6个Maven模块：

- `mcp-bom`：BOM模块，统一管理版本号
- `mcp-core`：核心实现，包含Client、Server、Transport、Schema等核心代码
- `mcp-json-jackson2`：Jackson 2.x的JSON序列化实现
- `mcp-json-jackson3`：Jackson 3.x的JSON序列化实现  
- `mcp`：便捷包，等于`mcp-core` + `mcp-json-jackson3`
- `mcp-test`：测试工具和集成测试

JSON序列化单独拆模块的原因是解决Jackson版本冲突问题：

- Java生态中Jackson存在大版本分裂：2.x（包名`com.fasterxml.jackson`）和3.x（包名`tools.jackson`，2025年发布，不向后兼容）
- SDK将JSON序列化抽象为`McpJsonMapper`接口，具体实现按版本拆开
- 项目用Jackson 2时，引入`mcp-core` + `mcp-json-jackson2`；用Jackson 3时，直接引入`mcp`便捷包
- 这种设计避免了与现有项目的依赖冲突，是Java生态中常见的设计模式。

### 请描述MCP Java-SDK的核心架构分层，以及每一层的职责是什么？

**答**：MCP Java-SDK的核心架构分为四层，从下往上依次是：

1. **Schema层**：协议的字典
   - 由`McpSchema`类实现（2786行代码），定义协议里所有消息类型
   - 将协议规范中的消息类型转换为Java Record类（不可变数据类）
   - 例如：`McpSchema.Tool`（工具定义）、`McpSchema.CallToolRequest`（工具调用请求）等
2. **Transport层**：消息的快递公司
   - 负责将JSON-RPC消息从一端送到另一端，不关心消息内容
   - 提供三种实现：Stdio（stdin/stdout）、SSE（Server-Sent Events）、Streamable HTTP（双向HTTP流）
   - 每种Transport有对应的客户端和服务端实现类
3. **Session层**：对话的窗口
   - 管理连接的完整生命周期，负责三件事：
     - 维护连接状态（是否已初始化、是否已关闭）
     - 请求-响应配对（按JSON-RPC的id字段路由响应）
     - 处理通知消息（不需要响应的单向消息）
   - 关键类：`McpClientSession`和`McpServerSession`
4. **Client/Server层**：开发者直接打交道的API
   - 提供Builder模式创建Client和Server
   - 同步API（`McpSyncClient`/`McpSyncServer`）和异步API（`McpAsyncClient`/`McpAsyncServer`）
   - 异步API基于Project Reactor，同步API内部是对异步API的`.block()`封装

### MCP SDK支持哪三种Transport实现？它们的区别是什么，分别适用于什么场景？

**答**：MCP SDK支持三种Transport实现：

1. **Stdio Transport**：
   - 通信方式：通过进程的stdin/stdout传输消息，每条消息是一行JSON
   - 适用范围：仅本地
   - 优点：简单、安全（不暴露网络端口）
   - 缺点：只能本地通信，进程生命周期管理复杂
   - 典型场景：Claude Desktop本地工具、本地开发调试
   - 关键类：`StdioClientTransport`、`StdioServerTransportProvider`
2. **SSE Transport**：
   - 通信方式：SSE推送（Server→Client）+ HTTP POST（Client→Server）
   - 适用范围：本地+远程
   - 优点：基于标准HTTP，穿越防火墙方便，SSE自带断线重连
   - 缺点：Client→Server方向不是流式的
   - 典型场景：传统Web环境的远程通信
   - 关键类：`HttpClientSseClientTransport`、`HttpServletSseServerTransportProvider`
3. **Streamable HTTP Transport**：
   - 通信方式：双向HTTP流，响应体可以是普通JSON或SSE流
   - 适用范围：本地+远程
   - 优点：无需预先建立SSE长连接，按需建立连接，更简洁
   - 缺点：实现相对复杂
   - 典型场景：企业级生产环境部署（官方推荐）
   - 关键类：`HttpClientStreamableHttpTransport`、`HttpServletStreamableServerTransportProvider`

选择建议：本地开发调试用Stdio，生产环境远程部署用Streamable HTTP，SSE作为兼容性较好的过渡方案。

### 用纯SDK构建MCP Server需要完成哪些关键步骤？与Spring AI注解方式相比有什么区别？

**答**：用纯SDK构建MCP Server需要完成四个关键步骤：

1. **创建Transport**：选择合适的传输方式

   ```java
   StdioServerTransportProvider transportProvider = new StdioServerTransportProvider();
   ```

2. **构建工具定义**：手动拼JSON Schema描述参数

   ```java
   String inputSchema = "{ \"type\": \"object\", \"properties\": { ... } }";
   McpSchema.Tool tool = new McpSchema.Tool("toolName", "description", McpSchema.JsonSchema.fromJson(inputSchema));
   ```

3. **编写处理函数**：手动提取参数，返回结果

   ```java
   (exchange, request) -> {
       String param = request.arguments().get("paramName").toString();
       return new McpSchema.CallToolResult(List.of(new McpSchema.TextContent(result)), false);
   }
   ```

4. **阻塞主线程保活**：防止JVM退出

   ```java
   new CountDownLatch(1).await(); // 阻塞主线程
   Runtime.getRuntime().addShutdownHook(new Thread(server::close)); // 优雅关闭
   ```

与Spring AI注解方式的区别：

- **代码量**：SDK方式需要30+行代码实现一个工具，Spring AI只需3行
- **JSON Schema**：SDK需要手动拼写JSON Schema，Spring AI从方法签名自动生成
- **参数映射**：SDK需要手动提取参数，Spring AI自动将请求参数映射到方法参数
- **生命周期管理**：SDK需要手动管理连接和保活，Spring AI与Spring容器生命周期绑定
- **复杂度**：SDK方式更底层更灵活，Spring AI方式更简洁更高效

### MCP Java-SDK从0.x版本升级到1.0.0版本有哪些重要的不兼容变更？未来的2.0.0版本有哪些新特性值得关注？

**答**：从0.x升级到1.0.0版本有三个重要不兼容变更：

1. **Jackson 3成为默认**：
   - `mcp`便捷包默认包含Jackson 3（包名`tools.jackson`）而不是Jackson 2
   - 用Jackson 2的项目需要改为引入`mcp-core` + `mcp-json-jackson2`
2. **Spring集成模块迁移**：
   - Maven坐标从`io.modelcontextprotocol.sdk:mcp-spring-*`变为`org.springframework.ai:mcp-spring-*`
   - 例如：`io.modelcontextprotocol.sdk:mcp-spring-webflux` → `org.springframework.ai:mcp-spring-webflux`
3. **API方法签名调整**：
   - 旧版的`tool()`注册方法在0.18.1已废弃，1.0.0正式移除
   - 改为`toolCall()`方法名

未来2.0.0版本值得关注的新特性：

- **Tasks管理**：支持长时间运行的任务，Client可以查询任务进度、取消任务
- **Elicitation**：Server可以主动向Client请求用户输入（如弹出确认对话框）
- **安全增强**：更完善的认证和授权机制
- **Virtual Threads支持**：利用Java 21的虚拟线程特性提升并发性能

这些演进方向表明SDK正在向更企业级、更安全、更高性能的方向发展，了解底层架构有助于跟上后续版本的变化。

## JSON-RPC_2.0标准面试题

### 什么是JSON-RPC 2.0？它与REST、Dubbo、gRPC等调用方式有什么区别？

**答**：JSON-RPC 2.0是一种轻量的远程过程调用（RPC）协议，使用JSON作为消息格式，目标是用统一的JSON结构描述一次远程方法调用。其设计哲学是"简单至上"。

与REST、Dubbo、gRPC的区别：

- REST/Dubbo/gRPC：偏业务接口调用方式，解决"服务与服务之间如何通信"的问题
- JSON-RPC 2.0：偏协议消息封装规范，定义的是"请求—响应—通知"的统一表达方式
- MCP：在JSON-RPC 2.0之上，进一步约定工具、资源、提示词等能力模型

JSON-RPC 2.0不绑定具体传输层，可以跑在HTTP、WebSocket、stdio等通道之上，只定义消息格式与处理规则，不负责传输层细节、鉴权、服务发现等工程能力。

### JSON-RPC 2.0的核心对象有哪些？请详细说明Request Object的结构和各字段含义。

**答**：JSON-RPC 2.0的核心对象包括：

- Request Object：发起一次调用，请求服务端执行某个方法
- Notification：也是请求，但因为没有id，所以不要求服务端回包
- Response Object：服务端对请求的响应，包含结果或错误
- Batch：把多个请求一起发出去，减少多次交互开销

Request Object结构：

```json
{
  "jsonrpc": "2.0",  // 必需，标识协议版本，必须精确等于"2.0"
  "method": "methodName",  // 必需，要调用的方法名，不能以"rpc."开头
  "params": {...},  // 可选，方法调用参数，可以是Object（命名参数）或Array（位置参数）
  "id": 1  // 条件必需，用于关联请求和响应，不存在时视为Notification
}
```

### 什么是Notification？它与普通Request有什么区别？在什么场景下使用Notification？

**答**：Notification是一种没有id字段的Request，表示客户端不期望收到任何响应。

与普通Request的区别：

- Notification没有id字段，普通Request有id字段
- 服务端不得对Notification返回JSON-RPC Response Object，即使发生错误也不会返回错误对象
- 普通Request服务端必须返回Response Object

使用场景：

- 调用方不关心结果/错误
- 不希望为这次调用付出一次响应的成本（等待、解析、重试、幂等）
- 例如：日志/埋点/旁路异步触发等场景，调用方只要"发出去"即可，不管任务结果或可以在后台处理

在HTTP传输场景下，服务器仍然需要返回HTTP响应（如204 No Content），但不会返回JSON-RPC Response对象。

### Response Object的结构是怎样的？result和error字段有什么规则？

**答**：Response Object结构：

```json
{
  "jsonrpc": "2.0",  // 必需，必须精确等于"2.0"
  "result": {...},  // 成功时必需，调用成功的返回值
  "error": {...},  // 失败时必需，调用失败的错误信息
  "id": 1  // 必需，用于关联请求和响应，必须与Request中的id相同
}
```

result和error字段的核心规则：

- **必须有且只有一个存在**：成功时必须存在result，失败时必须存在error
- 成功时不得存在error字段，失败时不得存在result字段
- id字段在正常情况下必须与Request中的id相同，如果服务端无法识别请求ID（如解析错误），id必须为null

### JSON-RPC 2.0的错误处理机制是怎样的？有哪些预定义的错误码？

**答**：JSON-RPC 2.0的错误处理通过Error Object实现，当RPC调用失败时，Response必须包含error对象。

Error Object结构：

```json
{
  "code": -32601,  // 必需，整数，错误类型
  "message": "Method not found",  // 必需，字符串，简短的人类可读错误描述
  "data": "Method 'getUserInfo' is not registered"  // 可选，任意JSON值，额外错误信息
}
```

预定义错误码（-32768到-32000区间）：

- -32700: Parse error - JSON解析错误
- -32600: Invalid Request - 请求不是有效的Request对象
- -32601: Method not found - 方法不存在
- -32602: Invalid params - 参数错误
- -32603: Internal error - JSON-RPC内部错误
- -32000~-32099: Server error - 保留给实现定义

应用自定义错误应使用其他错误码空间（如1000+），不应滥用协议保留错误码。

### 什么是Batch批处理？它的规则和注意事项有哪些？

**答**：Batch允许客户端一次发送多个Request对象，服务端批量处理后返回多个Response，可以减少网络往返次数，提高吞吐量。

核心规则：

- 请求格式：客户端发送Array，包含多个Request对象，可以混合普通Request和Notification
- 响应格式：服务端返回Array，包含对应的Response对象，Notification不应返回Response
- 响应顺序：不要求与请求顺序一致，客户端必须通过id字段来匹配请求与响应
- 服务端可以并发处理Batch中的请求

注意事项：

- 如果Batch不是有效JSON，服务端必须返回单个错误Response（不是Array）
- 如果Batch是空Array，应返回单个"Invalid Request"错误
- 如果Batch中没有任何需要返回Response的请求（全部是Notification），服务端不应返回JSON-RPC响应
- 每个Batch元素都必须是有效的Request Object

### 在实现JSON-RPC 2.0时，常见的错误有哪些？如何避免？

**答**：常见实现错误及避免方法：

1. **忘记包含jsonrpc: "2.0"字段**
   - 避免：确保所有Request和Response都包含jsonrpc字段且值为"2.0"
2. **Notification返回了响应**
   - 避免：对Notification不返回JSON-RPC Response对象（HTTP场景可返回204）
3. **result和error同时存在**
   - 避免：严格遵守两者只能存在一个的规则
4. **method不是字符串类型**
   - 避免：确保method字段为String类型
5. **params不是Object或Array**
   - 避免：确保params字段为Object或Array类型
6. **命名参数大小写不匹配**
   - 避免：注意JSON字段名大小写敏感，与服务端约定保持一致
7. **Batch全是Notification却返回[]**
   - 避免：无需要返回的Response时，不返回JSON-RPC响应
8. **业务错误使用协议错误码**
   - 避免：-32768~-32000用于协议层错误，业务错误使用自定义错误码（如1000+）

### 在工程实践中，JSON-RPC 2.0需要补充哪些能力？请说明输入校验、超时重试和幂等性处理的最佳实践。

**答**：JSON-RPC 2.0协议本身只定义消息格式，工程实践中需要补充：

**输入校验**：

- 协议层校验：jsonrpc字段、method类型、params类型、id字段、result/error互斥性
- 业务层校验：参数类型、范围、必填项、逻辑关系
- 安全层校验：参数长度限制、内容过滤、方法名白名单、频率限制

**超时与重试**：

- 客户端：设置合理超时时间（5-10秒），幂等方法可重试，使用指数退避策略
- 服务端：设置方法执行超时（30秒），超时返回Server error，记录超时日志

**幂等与去重**：

- 根据id字段做去重，用Redis或内存缓存存储最近处理过的请求ID（5-10分钟）

- 伪代码：

  ```java
  if (cache.has(requestId)) {
    return cache.get(requestId); // 返回缓存的响应
  }
  Response response = executeMethod(request);
  cache.set(requestId, response, 300); // 缓存5分钟
  return response;
  ```

- 只对有id的请求做去重，Notification不需要去重

### JSON-RPC 2.0的调用流程是怎样的？请描述从客户端到服务端的完整过程。

**答**：JSON-RPC 2.0的完整调用流程：

**普通Request（成功）**：

1. 客户端发送带id的Request → 2. 服务端校验格式 → 3. 调用业务逻辑 → 4. 返回带result的Response

**普通Request（失败）**：

1. 客户端发送带id的Request → 2. 业务逻辑抛出异常 → 3. 服务端返回带error的Response，id保持一致

**Notification**：

1. 客户端发送不带id的Request → 2. 服务端执行业务逻辑 → 3. 不返回任何Response

**协议错误**：

1. JSON解析失败 → 2. 服务端返回Parse error，id为null（无法识别请求ID）

**Batch Request**：

1. 客户端发送Array请求 → 2. 服务端并发处理（可选）→ 3. 只为有id的请求返回响应 → 4. 响应顺序可以任意，客户端通过id匹配

关键要点：一次标准调用由Request和Response构成，Notification因无id不要求响应，Response中result和error必须二选一，Batch是否返回响应取决于每个请求是否带id。

### 在JSON-RPC 2.0中，如何处理鉴权、日志追踪和版本兼容性问题？

**答**：

**鉴权处理**：

- 传输层鉴权（推荐）：HTTPS + API Key（Authorization Header）、HTTPS + JWT、mTLS（双向TLS）
- 应用层鉴权（不推荐）：在JSON-RPC请求中加鉴权字段，但会导致鉴权逻辑和业务逻辑耦合
- 签名防篡改：客户端用私钥对请求内容签名，服务端用公钥验证，包含时间戳+nonce防重放

**日志与链路追踪**：

- 记录内容：请求ID、方法名、参数（敏感信息脱敏）、响应结果、执行耗时、客户端信息
- 链路追踪：通过HTTP Header或扩展字段传递Trace ID，每个服务记录并传递
- 工具：使用ELK、Jaeger、Zipkin等工具聚合日志，可视化调用链路

**版本兼容性**：

- 版本管理：方法名加版本号（getUserInfo_v1）、HTTP路径加版本号（/api/v1/jsonrpc）、扩展字段加版本号
- 灰度发布：根据客户端标识路由到不同版本，使用特性开关控制新功能
- 监控：监控新版本错误率、响应时间，出问题快速回滚

错误码分层：协议层错误（-32768\~-32000）、基础设施错误（-32000\~-32099）、业务错误（正数1000~9999）