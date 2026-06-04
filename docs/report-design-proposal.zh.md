# Visual Coding-Agent Multimodal Harness 当前实现汇报

日期：2026-06-04

## 0. 核心目标

我们把 Claude Code / coding agent 的 harness 设计思想迁移到多模态 Agent 中，让 MLLM 不再只是一次性看图或看视频后直接回答，而是在一个可执行、可复核、可收集轨迹的视觉工作环境里主动调用工具、拆分任务、检查证据、压缩上下文，并把完整工具轨迹沉淀为后续训练数据。

一句话概括：

> 做一个面向复杂视觉任务的 Visual Coding-Agent Harness：主 Agent 像 Claude Code 管理代码任务一样，管理图像、视频、视觉工具、证据账本、子 Agent、验证器和训练轨迹。

当前实现已经收敛成三条能力线：

1. 多工具整合的视觉导向 harness。
2. 一套可追踪的证据工作流。
3. training-free 观察模型行为，并把工具轨迹沉淀为后续训练和诊断数据。

## 1. Claude Code 是如何管理任务的

Claude Code 的核心不是复杂 prompt chain，而是一个稳定的 agent loop 加上多层 harness 管理。

最小 loop 是：

```text
用户任务
  -> 模型读取 messages 和 tools
  -> 模型发出 tool_use
  -> harness 执行 tool
  -> 返回 tool_result
  -> tool_result 进入 messages
  -> 继续循环，直到模型不再调用工具
```

这个循环本身很简单，真正的工程能力来自外层管理机制：

| Claude Code 机制 | 管理内容 | 迁移到多模态 Agent |
| --- | --- | --- |
| Agent loop | 模型和真实环境的交互循环 | 保持同样循环，把代码工具换成视觉工具。 |
| Tool registry / dispatch map | 工具 schema、参数校验、handler 分发 | 把视频导航、局部检查、caption、QA、验证等包装成统一 visual tools。 |
| Workspace | 文件、代码、diff、运行结果 | 视觉 evidence workspace：原视频、视频帧、clip、crop、证据记录。 |
| Context compact | 防止命令输出和文件内容污染上下文 | 大图、长视频、长 OCR 不直接塞进 prompt，只保留短 observation 和 artifact path。 |
| Subagent | 用干净上下文处理局部任务 | 用 Segment Inspector 处理局部视频窗口，只向主 Agent 返回 distilled observation。 |
| Event stream | 统一 status、tool_use、tool_result、usage | 统一视觉工具调用事件、artifact 写入事件、observation、verifier 结果。 |

OpenDesign 的 Claude 源码进一步说明了 adapter 思路：它不重新实现 Claude Code，而是把 Claude 原生 `stream-json` 输出解析成统一事件，例如 `tool_use`、`tool_result`、`text_delta`、`usage`，再交给 UI 和持久化系统。多模态 harness 也采用同样的原则：不同视觉 backend 可以不同，但输出必须进入统一事件和证据协议。

## 2. 为什么多模态 Agent 需要这种 harness

长视频 QA、多图推理、OCR-heavy 图像理解、GUI 理解、地理定位、开放世界视觉搜索中，很多失败不是模型完全看不懂，而是模型缺少稳定的视觉操作环境。

典型失败包括：

- 模型凭全局印象回答，没有检查关键区域或关键时间段。
- caption、检索摘要或上一步工具输出变成答案权威，但视觉证据其实不支持。
- 工具轨迹太长，旧观察污染上下文，模型混淆证据。
- 模型知道应该 zoom、crop、OCR、seek、compare，但动作没有真正落到正确区域或时间。
- 工具有很多，但没有统一 schema、provenance 和 observation formatter，无法组合和验证。

所以我们的核心主张不是“工具越多越好”，而是：

> 多模态工具必须被 harness 化：统一 schema、统一状态、统一证据账本、统一上下文压缩、统一子任务协议和统一验证机制。

## 3. 相关论文如何支撑这个方向

| 来源 | 对本项目的启发 |
| --- | --- |
| Learn Claude Code | agent loop 很简单，真正关键是工具、上下文、任务系统、subagent、memory、protocol 等 harness 层。 |
| OpenDesign Claude runtime | adapter + event stream：保留底层 agent 能力，把不同输出解析成统一事件和 artifact 管理。 |
| Visual ChatGPT | Prompt Manager 把多个 Visual Foundation Models 包装成工具，让 ChatGPT 做主调度。 |
| VisProg | training-free 地让 LLM 写 visual program，解释器维护 program state 并调用视觉/知识/逻辑模块。 |
| ReTool-Video | 从粗粒度视频工具扩展到 base/meta tool library，用递归 resolver 把高层视频意图落地为可执行工具链。 |
| ParaVT | 用并行 temporal crop 和共享权重 subagents 减少顺序工具调用的错误传播和上下文污染。 |
| VideoThinker | 用 caption-space 生成工具轨迹，再替换成真实 frame token 做 tool-use training。 |
| VideoSEAL | planner 和 inspector 解耦，最终答案权威来自 inspector 检查过的视觉证据，而不是 noisy trace。 |
| Agentic-MME | 评估不能只看 final answer，还要看工具是否调用、是否用对、是否高效。 |
| Pixel Reasoner / Walk the Talk | zoom-in、select-frame 等 pixel-space 操作本身就是 reasoning primitive；训练要弥合文本推理和视觉动作之间的 gap。 |
| XSkill / IMPACT-CYCLE | 从 rollout 中抽取经验、技能、claim-level correction 和 provenance，供后续复用和监督。 |

## 4. 和 ReTool-Video / ParaVT / VisProg 的差异

这个项目不能被包装成“我们比已有工作多加几个视觉工具”。更强的定位是：已有工作主要解决视觉工具如何被调用，而我们解决视觉工具调用如何被 coding-agent-style harness 管理、审计、压缩、协作，并转成训练数据。

| 工作 | 它主要解决什么 | 我们的差异 |
| --- | --- | --- |
| VisProg | LLM 生成 python-like visual program，解释器调用视觉/知识/逻辑模块。 | VisProg 更像一次性 program execution；我们是持续 agent loop，有 workspace、trace、ledger、verifier 和 context compact。 |
| ReTool-Video | 构建丰富 video tool library，用 recursive grounding 把高层视频意图落成工具链。 | ReTool 强在工具库和递归解析；我们强在 evidence workspace、answer authority、工具轨迹治理和跨工具协作协议。 |
| ParaVT | 用并行 temporal crop 和 RL 解决顺序工具调用的错误传播、格式崩溃和 skip-tool shortcut。 | ParaVT 聚焦 parallel crop-video；我们把它泛化成可扩展的 visual worker / inspector 协议。 |

我们的核心优势有四点：

1. **Evidence discipline 更强。** 每个工具输出都必须绑定 artifact、timestamp、claim、confidence 和 limitation，最终答案只能引用 evidence ledger，而不能直接依赖 planner trace 或 caption shortcut。
2. **更像真实 coding agent。** 我们迁移的是 Claude Code 的完整 harness：tool registry、workspace、context compact、subagent isolation、event stream，而不是只迁移 tool calling。
3. **Subagent 边界更清楚。** 当前 `inspect_segment` 已经作为 Segment Inspector 边界存在：主 Agent 选择时间窗和问题，Inspector 只返回局部 distilled observation。
4. **Training-free 到训练数据的闭环更明确。** Trace 从一开始就按 SFT / preference / RL 可用的数据资产来记录，成功和失败都可以被诊断。

一句话区分：

> VisProg 是“LLM 写视觉程序”；ReTool-Video 是“递归调用更多视频工具”；ParaVT 是“并行视频工具调用训练”；我们是“让多模态 Agent 像 Claude Code 一样工作”。

## 5. 当前系统总览

当前系统结构如下：

```text
用户问题 / benchmark case
  -> Main Agent
  -> question route + task playbook
  -> Visual Tool Registry
  -> ProgramInterpreter / Tool Executor
  -> Evidence Workspace
  -> compact ledger
  -> AnswerAgent
  -> Verifier / Reporter
  -> final answer / follow-up tool calls
```

当前 Main Agent 是 text-only controller：

- 它不直接接收完整视频或全部帧。
- 它看到用户问题、选项、工具 schema、scene index 摘要、已写入的 compact evidence ledger。
- 真正的视频像素访问由工具完成，当前主要是 `global_gist`、`inspect_segment`、`caption_segment`、`qa_segment`。
- 每次工具调用都会写入 workspace：`trace.jsonl` 记录动作过程，`observations.jsonl` 记录结构化视觉证据，`ledger.md` 给后续轮次和 AnswerAgent 使用。

Round 3 后，系统不再把所有问题都强行送进局部检索，而是先做问题路由：

| 路由 | 适用问题 | 当前路径 |
| --- | --- | --- |
| `gist_global` | main idea、overall topic、whole-video summary、信息概括类问题 | 先调用 `global_gist`，形成 sparse whole-video evidence floor。 |
| `temporal_order` | before、after、first、last、sequence、order 等时序比较题 | 先定位候选事件窗口，再用局部视觉事实和 temporal verifier 做顺序比较。 |
| `needle_local` | 局部动作、细节识别、OCR、对象属性、counting、特定人物/物体/场景 | 走 `video_ls/search_segments/read_segment/zoom/inspect_segment` 的局部定位路径。 |

这个路由修复了一个关键问题：VideoMME wrapper 里常有 “answer with the option letter first” 之类格式提示，旧规则容易把它误判成 temporal/needle 问题。当前规则优先识别 `main idea` 等 gist marker，再处理局部/时序 marker。

v4 计划把这套路由进一步收紧成一个硬约束：

```text
global_gist gives the direct floor,
grounding finds candidate clips,
vision reads local facts,
AnswerAgent maps facts to options,
Verifier blocks unsupported or conflicted answers,
and every final answer cites evidence.
```

## 6. 当前工具设计

当前工具不是一组平铺 API，而是四层协作。

| 层级 | 当前工具 / 模块 | 作用 |
| --- | --- | --- |
| Global floor | `global_gist` | 对 gist/global 问题先看稀疏整段视频，给出 whole-video sparse observation，避免整体理解题被误拆成局部检索。 |
| Video workspace | `video_ls`、`search_segments`、`read_segment`、`zoom` | 像代码 Agent 的 `ls/grep/read` 一样管理长视频地图、候选窗口和 coarse-to-fine 定位。 |
| Local visual evidence | `inspect_segment`、`caption_segment`、`qa_segment`、`caption_segments`、image/region tools | 真正访问像素，产生带 artifact、时间窗、claim、confidence、limitations 的 observation。 |
| Evidence arbitration | `EvidenceWorkspace`、AnswerAgent、`verify_ledger_answer`、`report_metrics.py` | 把 observation 变成 evidence table，按 grounding quality 和冲突关系选答案，并记录 unsupported/conflict/direct regression。 |

### 6.1 Global Floor

`global_gist`

- 对整段视频做稀疏采样观察，当前与 `direct_full_video` 使用同类 sparse whole-video 输入。
- 适合 main idea、overall topic、whole-video gist、Information Synopsis 这类问题。
- 输出普通 observation，包含 `tool=global_gist`、`grounding_quality=global_sparse`、whole-video region、claim、confidence 和可选 `supported_option`。
- 它是 global floor，不是定位工具。局部细节、时序顺序、OCR、人物动作或对象属性仍要回到局部工具。
- AnswerAgent 会把受支持的 `global_sparse` row 当成可引用视觉证据；如果后续局部高质量证据与它冲突，需要显式仲裁。

### 6.2 Video Workspace Tools

`video_ls`

- 给出视频总览、segment 数量、时长、可用索引类型。
- 返回候选 segments、outline、coverage、recommended tools。
- 候选包含 `relevance_reason`，推荐下一步通常是 `inspect_segment` 或 `zoom`。
- 对应代码 Agent 的 `ls`。

`search_segments`

- 在 VideoMap 的 caption/asr/ocr/entities 中做检索。
- 返回每个候选的分通道 `matches`、命中字段、score、evidence snippet。
- 当前是 training-free lexical search，embedding retrieval 尚未接入。
- 对应代码 Agent 的 `grep`。

`read_segment`

- 读取某个 segment 的 compact metadata。
- 不访问像素，只读索引。
- 对应代码 Agent 的 `read file/span`。

`zoom`

- 把 coarse segment 物化成稳定 child segments，例如 `seg_0002_z01`。
- 写回 mutable `VideoMapStore`，后续能被 `video_ls` / `search_segments` 召回。
- 用于 coarse-to-fine localization。

### 6.3 Local Visual Evidence Tools

`inspect_segment`

- 当前首选的局部视觉证据工具。
- 语义上是 Segment Inspector subagent boundary：内部看局部时间窗，主 planner 只收到一条 distilled observation。
- 当前仍可接收 `candidate_options`，但只把选项当作“要找什么事实”的提示。
- 默认 prompt 已经要求 worker 不选择 MCQ 选项、不输出 `supported_option`、不写 final answer；选项归因转移到 AnswerAgent / Verifier。
- 旧 trace 里如果出现 `Supported option: X.` 这类 worker 投票，默认 evidence table 会忽略，并由 reporter 统计为 `legacy_worker_vote_rows`。

`caption_segment`

- 低层 VLM 工具：对指定时间段做 caption。
- 返回 claim、clip path、time range、nframes、limitations。
- 当前保留为兼容和诊断工具，但不能让单条 caption shortcut 直接成为答案权威。

`qa_segment`

- 低层 VLM 工具：对指定时间段做 QA。
- 可作为 Inspector 内部能力或兼容路径使用。

`caption_segments`

- 批量 caption 若干 segments，并写回 mutable VideoMap。
- 用于离线或低索引场景，把空地图变成可搜索地图。
- 在线问答里需要谨慎使用，避免把索引构建成本混进回答预算。

`caption_image / qa_image / caption_region / qa_region`

- 图像和区域级工具已经有基础协议。
- 当前不是 VideoMME 主路径，但协议上能接 OCR、detector、SAM、tracking 等模块。

### 6.4 Verification / Answer Tools

`summarize_ledger_evidence`

- 从 ledger 中抽取 claim 列表。

`verify_ledger_answer`

- 当前是 lexical support score + citation gate + 非 navigation 视觉证据 gate。
- `global_gist`、`inspect_segment`、`caption_segment`、`qa_segment` 都可以算视觉证据。
- 仍是弱验证，还不能替代模型式 entailment / temporal verifier。

AnswerAgent / evidence table

- 不是外部视觉工具，而是最终仲裁层。
- 消费 `EvidenceWorkspace.evidence_table()`，按 `candidate_option_relations`、`grounding_quality`、工具类型、confidence、limitations 和冲突关系做结构化选择。
- `global_gist` 的 `supported_option` 暂时允许作为 direct-style whole-video evidence；local worker 的 `supported_option` 或 claim-text option vote 默认不再参与仲裁。
- 当前权重顺序大致是 `visually_confirmed` / `global_sparse` 高于 `inferred`、`weak` 和 `external_knowledge`。
- 对 gist/global 路由保留 global floor：如果 `global_gist` 明确支持某个选项且没有更强反证，可以直接作为最终 citation。

`report_metrics.py`

- 汇总每个 case 的 final rate、accuracy、latency、unsupported final、conflict 和 option support。
- 新增 `direct_regressions`，用于标记 direct baseline 正确但 agent 错误的回归样本。
- 新增 `legacy_worker_vote_rows`，用于暴露旧 worker 投票痕迹，防止局部 worker 的错误 MCQ 票偷偷影响最终答案。

## 7. 当前 Evidence Workspace 和 Observation 协议

每个 run 都会生成 evidence workspace。核心文件是：

```text
runs/<run_id>/
  artifacts/
    frames/
    clips/
    crops/
  observations.jsonl
  trace.jsonl
  ledger.md
```

各部分作用：

| 文件/目录 | 作用 |
| --- | --- |
| `artifacts/` | 保存原视频片段、关键帧、clip、crop 等视觉证据。 |
| `observations.jsonl` | 每次工具调用后的结构化观察，是 AnswerAgent 的主要证据来源。 |
| `trace.jsonl` | 完整工具轨迹，用于调试、指标和后续训练数据。 |
| `ledger.md` | 给主 Agent 和 AnswerAgent 使用的短证据账本。 |

工具输出统一变成 observation：

```json
{
  "observation_id": "obs_0042",
  "tool": "global_gist",
  "claim": "The video mainly explains the process and impact of World War One.",
  "confidence": 0.76,
  "supported_option": "D",
  "grounding_quality": "global_sparse",
  "input_artifacts": ["input/media/video.mp4"],
  "regions": [
    {
      "segment_id": "global",
      "start_sec": 0.0,
      "end_sec": 1782.0,
      "nframes": 64
    }
  ],
  "limitations": "Sparse whole-video sampling may miss fine-grained local events.",
  "raw_output": {}
}
```

字段含义：

- `claim`: 给下一轮 Agent / AnswerAgent 读的自然语言证据。
- `confidence`: 工具自评或启发式置信度。
- `supported_option`: 过渡字段。当前只允许 `global_gist` 作为 direct-style whole-video evidence 使用；局部 worker 默认不应输出，旧 trace 中的局部 worker 选项票会被 `evidence_table()` 忽略并被 reporter 计数。
- `grounding_quality`: 证据强度标签，例如 `visually_confirmed`、`global_sparse`、`inferred`、`weak`、`external_knowledge`。
- `input_artifacts`: 原视频、clip、crop、frame 等证据来源。
- `regions`: 结构化时空定位信息。
- `limitations`: 告诉 AnswerAgent 这条证据有什么限制。
- `raw_output`: 完整工具输出写入文件，默认不直接塞进 prompt。

`compact_ledger_text()` 当前已经把长 trace 压成 bounded context view，主 Agent 看到的是短证据，而不是完整 raw tool output。这一点对应 VideoSEAL 的核心问题：防止 planner trace、caption shortcut 或长上下文污染最终答案权威。

## 8. 当前 Agent Loop

当前回答流程可以分成两条路径。

### 8.1 gist/global 路径

```text
question
  -> classify_question_route() = gist_global
  -> global_gist
  -> EvidenceWorkspace writes global_sparse observation
  -> AnswerAgent reads evidence_table
  -> final answer with citation
```

这条路径解决的是整体理解题。它避免先 `video_ls -> search -> zoom -> inspect`，因此在 `605-1` 这类 main idea 问题上明显缩短工具路径。

### 8.2 temporal/local 路径

```text
question
  -> classify_question_route() = temporal_order or needle_local
  -> video_ls / search_segments
  -> read_segment / zoom
  -> inspect_segment
  -> EvidenceWorkspace writes local observation
  -> AnswerAgent / Verifier
  -> final or targeted follow-up
```

这条路径解决的是局部动作、局部事实、时序、OCR、counting 等问题。当前它已经能稳定调用工具并产出 trace，但准确率瓶颈转移到三处：

- 是否找到正确时间窗。
- `inspect_segment` 是否给出明确、局部、可验证的事实，而不是局部窗口里的 MCQ 选项票。
- AnswerAgent / Verifier 是否能解决冲突证据，而不是引用最近或最顺手的一条 observation。

### 8.3 当前运行策略

- `AgentBudget.free_explore(...)` 支持质量优先的自由探索路径：不做 per-class cost budget，只保留 emergency caps。
- bounded mode 仍可能因为 `max_rounds_reached` incomplete，当前主要作为效率对照。
- planner JSON 解析失败会记录 `planner_json_parse_error`，并 fallback 到局部 `inspect_segment`，避免 quoted option 文本直接中断 agent。
- MCQ 工具调用会把 letter-only `candidate_options` 补成完整选项文本，减少 option grounding drift；但局部 worker 只能用这些选项理解待查事实，不能把选项当作最终答案输出。

## 9. 当前 Answer / Verifier / Reporter 设计

### 9.1 AnswerAgent

AnswerAgent 当前不是再让模型凭直觉综合，而是读取 evidence table 做结构化仲裁：

```text
question
options if MCQ
evidence table
candidate observations
limitations
required citation format
```

当前规则：

- 不能只引用 `video_ls` / `search_segments` 作为视觉证据。
- MCQ final answer 必须以选项字母开头。
- `global_sparse` 可以作为 gist/global 问题的 sparse whole-video visual evidence。
- `visually_confirmed` / `global_sparse` 高于 `inferred` / `weak` / `external_knowledge`。
- 如果 evidence table 中有多个互斥 supported options，需要在 reporter 中显式暴露冲突。
- 对局部 worker 的 legacy option vote 默认不采信；显式 option mapping 应来自 AnswerAgent / Verifier 生成的 `candidate_option_relations`。

### 9.2 Verifier

当前 verifier 是规则基线：

- lexical overlap between answer and compact evidence context。
- citation gate。
- non-navigation visual-evidence gate。

它能拦住一部分没有 citation 或只引用导航工具的答案，但还不能可靠处理 temporal ordering、复杂 entailment、反事实比较和多证据冲突。

### 9.3 Reporter

`report_metrics.py` 当前输出：

- final rate / incomplete rate。
- accuracy。
- latency。
- option support table。
- unsupported final。
- final with conflict。
- direct regressions。
- legacy worker vote rows。

这些指标让失败不再只是“答错了”，而是能定位到 selected option 无结构化支持、caption shortcut、conflicting evidence unresolved、direct baseline regression 等类型。

## 10. 当前实验信号

当前已经验证的事实：

- 本地单测通过：`106 tests OK`。
- 远端 KML 机器单测通过：`106 tests OK`。
- Qwen3-VL remote backend 可以跑通 harness，生成工具调用、trace、observations、ledger 和 summary。

Round 3 最新 sanity probe：

| Case | 方法 | 工具路径 | 答案 | 正确性 | 耗时 |
| --- | --- | --- | --- | --- | --- |
| VideoMME `605-1` | `agent_v2 --free-explore` | `global_gist` | D | 正确 | 3.076s |
| VideoMME `605-1` | `direct_full_video` | direct sparse video | D | 正确 | 5.101s |

对应 artifact：

```text
/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_round3_605_global_fix_20260604/summary.json
```

这一条说明：对 gist/global 问题，`global_gist` 可以把 agent 路径从多轮局部探索缩短成一次 whole-video sparse observation，并保持正确答案。

v4 no-vote 改动同步到 KML 后，3-case anchor 已跑完：

```text
/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_round4_worker_no_vote_3case_20260604/summary.json
```

Compact case result：

| Case | GT | Direct | Agent v2 | 当前诊断 |
| --- | --- | --- | --- | --- |
| VideoMME `605-1` | D | D, correct | D via `global_gist`, correct | global floor 保住。 |
| VideoMME `611-2` | D | A, wrong | C, wrong | unsupported final；需要 temporal grounding/facts 和 `need_more_evidence` 阻断。 |
| VideoMME `612-1` | B | B, correct | A, wrong | direct regression；需要 temporal verifier / option mapping 阻断 unsupported A。 |

Strategy-level metrics：

| Strategy | Accuracy | Final Rate | Direct Regressions | Unsupported Final Rate | Legacy Worker Votes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `agent_v2 --free-explore` | 1/3 | 100% | 1 | 66.7% | 1 |
| `direct_full_video` | 2/3 | 100% | 0 | 0% | 0 |

这一轮说明：

- `605-1` 的 global floor 没有被 worker no-vote / legacy filter 打碎。
- fresh run 里仍出现 `legacy_worker_vote_rows=1`，说明 prompt 约束还不够，需要 worker-output validation 或 `vision_read` schema。
- 两个 temporal case 都 final 了，但都是 unsupported final；AnswerAgent 应该返回 `need_more_evidence`，而不是给出无结构化支持的 C/A。
- `612-1` 是 direct 正确、agent 错误，因此是 v4 定义下的 release blocker。

Round 3 之前的三样本完整 run 保留为诊断参考：

| 方法 | 样本数 | 准确率 | final rate | incomplete rate | 平均耗时 |
| --- | --- | --- | --- | --- | --- |
| `agent_v2 --free-explore` | 3 | 1/3 | 100% | 0% | 304s |
| `direct_full_video` | 3 | 2/3 | 100% | 0% | 5.08s |

这个旧 run 的价值不是证明当前 agent 更强，而是暴露失败形态：

- Agent 已经能稳定调用工具并 final。
- 失败点从“不会调用工具/跑不通”转移到 evidence attribution、AnswerAgent/Verifier 冲突仲裁和局部 observation 质量。
- 真实 Inspector 支持曾经常出现在 claim 文本中，例如 `Supported option: X.`；v4 后默认不再让局部 worker 票参与仲裁，而是单独计入 `legacy_worker_vote_rows`。
- caption shortcut 和 navigation-only citation 会污染最终答案权威。

所以当前结论是：

```text
global/gist 题的路径已经变短，并在 605-1 上保持正确；
temporal/local 题的失败已经从“跑不完”变成“unsupported final / direct regression”；
下一步必须先做 GroundingAgent、VisionFact schema、AnswerAgent blocking 和 temporal verifier。
```

## 11. 当前限制

当前系统已经是一个能跑通、能记录、能诊断的 long-video visual tool harness，但还不是强 AnswerAgent。

主要限制：

- `inspect_segment` 仍是 one-shot isolated VLM call，不是内部可多轮 tool-use 的真正 subagent runtime。
- `inspect_segment` 的 no-vote prompt 和 legacy-vote filter 已经落地，但还没有完整升级成 `vision_read` schema：`facts`、`event_label`、`polarity`、`time_range`、`grounding_quality` 仍需要结构化。
- GroundingAgent 还没有独立成 `ground_question` wrapper；当前仍主要由主 planner 显式调用 `video_ls/search_segments/zoom`。
- Verifier 仍是规则基线，缺少模型式 entailment、temporal ordering verifier 和反事实检查。
- bounded mode 仍可能因预算耗尽 incomplete。
- 局部 needle/temporal 题仍可能出现 caption shortcut、错误时间窗、冲突 evidence 未解决。
- 当前实验样本太小，不能用来声明整体 benchmark 提升。

当前最重要的下一步是：

1. 完成 `ground_question` / GroundingAgent：定位候选 clip，只输出候选，不投选项票。
2. 把 `inspect_segment` 升级到 `vision_read` 风格局部事实 schema。
3. 加强 AnswerAgent / Verifier 的 option consistency、temporal order 和 conflict resolution。
4. 在更大 VideoMME long-video 切片上同时看 accuracy、latency、final rate、direct_regressions、unsupported final 和 legacy_worker_vote_rows。

## 12. v4 开发路线和验收门

v4 不建议继续先堆 OCR、tracking、embedding retrieval、RL 或 SFT。当前瓶颈是 evidence-to-answer reliability，而不是工具库存。

建议的编码顺序：

1. schema 和测试先行：`GroundingCandidate`、`VisionFact`、`EvidenceRowV2`、`CandidateOptionRelation`、`AnswerAgentDecision`、`TemporalVerifierResult`。
2. 加固 global floor：`605-1` 必须继续 `global_gist -> D`，并把 `direct_regressions == 0` 当 release blocker。
3. 实现 GroundingAgent wrapper：内部可以调用导航工具，但对主 Agent 只返回 candidate clips。
4. 实现 VisionAgent no-vote：局部工具只回事实、时间、模态、置信度和限制。
5. 升级 AnswerAgent：由全局层把事实映射到 option relations，无法区分时返回 `need_more_evidence`。
6. 升级 Verifier：冲突检查、direct floor 检查、temporal-order 检查、grounding-quality weighting。
7. 扩展评测：保留 3-case anchor，再加入 5-10 个 needle/local long-video case，并按 route 汇报。

验收门：

| Gate | 要求 |
| --- | --- |
| Stop the bleeding | `605-1 -> D via global_gist`，anchor set `direct_regressions == 0`，`unsupported_final == 0`。 |
| Role split | fresh run 中 local worker 不输出 `supported_option`，`legacy_worker_vote_rows == 0`，GroundingAgent 只输出候选，VisionAgent 只输出事实。 |
| Arbitration | selected option 必须有结构化支持；unresolved conflict 不能 final；`611-2` 应返回 D 或 targeted `need_more_evidence`；`612-1` 应保持 B 并显式解决冲突。 |
| Evaluation honesty | 按 route 汇报；标准 VideoMME 使用 stateless per-QA；stateful memory 实验单独标记；gist/global 保 direct floor，needle/local 才是 agent 增值目标。 |

## 13. 汇报总结

Claude Code 的成功说明：强 Agent 不是单纯靠模型，而是模型被放进一个可执行、可管理、可恢复的工作环境。对于多模态复杂任务，这个工作环境应该是 visual evidence workspace。

当前系统已经实现：

1. typed visual tools + registry + interpreter。
2. long-video workspace tools：`video_ls/search_segments/read_segment/zoom`。
3. global/gist floor：`global_gist`。
4. local visual evidence boundary：`inspect_segment`。
5. evidence workspace：`observations.jsonl`、`trace.jsonl`、`ledger.md`。
6. AnswerAgent evidence-table arbitration。
7. verifier / reporter 指标，包括 unsupported final、conflict、direct regression 和 legacy worker vote rows。

当前最准确的定位是：

```text
A working long-video visual tool harness with traceable evidence,
and an early rule-based answer arbitration layer.
```

下一步不是继续堆更多工具，而是把局部 worker 输出、AnswerAgent 仲裁和 verifier 证据一致性做强，让工具轨迹真正转化成稳定准确率。
