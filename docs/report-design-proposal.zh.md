# Visual Coding-Agent Multimodal Harness 汇报方案

日期：2026-06-02

## 0. 核心目标

我们想把 Claude Code / coding agent 的 harness 设计思想迁移到多模态 Agent 中，让 MLLM 不再只是一次性看图或看视频后直接回答，而是在一个可执行、可复核、可收集轨迹的视觉工作环境里主动调用工具、拆分任务、检查证据、压缩上下文，并把完整工具轨迹沉淀为后续训练数据。

一句话概括：

> 做一个面向复杂视觉任务的 Visual Coding-Agent Harness：主 Agent 像 Claude Code 管理代码任务一样，管理图像、视频、视觉工具、证据账本、子 Agent、验证器和训练轨迹。

这个方向可以分成三条实施主线：

1. 多工具整合的视觉导向 harness。
2. 一套协同合作的 workflow。
3. training-free 观察模型行为，再做工具调用训练和行为改善。

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
| Tool registry / dispatch map | 工具 schema、参数校验、handler 分发 | 把 detector、OCR、crop、zoom、tracking、VQA 等包装成统一 visual tools。 |
| Workspace | 文件、代码、diff、运行结果 | 视觉 evidence workspace：原图、视频帧、crop、mask、OCR patch、证据记录。 |
| Context compact | 防止命令输出和文件内容污染上下文 | 大图、长视频、长 OCR 不直接塞进 prompt，只保留短 observation 和 artifact path。 |
| Subagent | 用干净上下文处理局部任务 | 用 temporal/spatial/OCR/verifier worker 分别检查局部视觉证据。 |
| Task DAG | 持久化任务依赖和状态 | 把复杂视觉任务拆成定位、裁剪、识别、验证、回答等子任务。 |
| Permission / hooks | 控制危险操作、成本、审批 | 控制高成本模型、web search、隐私图像、外部知识检索。 |
| Event stream | 统一 status、tool_use、tool_result、usage | 统一视觉工具调用事件、artifact 写入事件、observation、verifier 结果。 |

OpenDesign 的 Claude 源码进一步说明了 adapter 思路：它不重新实现 Claude Code，而是把 Claude 原生 `stream-json` 输出解析成统一事件，例如 `tool_use`、`tool_result`、`text_delta`、`usage`，再交给 UI 和持久化系统。多模态 harness 也应该这样做：不同视觉 foundation model 可以不同，但输出必须进入统一事件和证据协议。

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
| Visual ChatGPT | Prompt Manager 把 22 个 Visual Foundation Models 包装成工具，让 ChatGPT 做主调度。 |
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
| VisProg | LLM 生成 python-like visual program，解释器调用视觉/知识/逻辑模块。 | VisProg 更像一次性 program execution；我们是持续 agent loop，有 workspace、trace、ledger、subagent、verifier 和 context compact。 |
| ReTool-Video | 构建丰富 video tool library，用 recursive grounding 把高层视频意图落成工具链。 | ReTool 强在工具库和递归解析；我们强在 evidence workspace、answer authority、工具轨迹治理和跨工具协作协议。 |
| ParaVT | 用并行 temporal crop 和 RL 解决顺序工具调用的错误传播、格式崩溃和 skip-tool shortcut。 | ParaVT 聚焦 parallel crop-video；我们把并行扩展成通用 visual workers：temporal、spatial、OCR、entity、verifier 都能协作。 |

我们的核心优势有四点：

1. **Evidence discipline 更强。** 每个工具输出都必须绑定 artifact、timestamp、bbox、claim、confidence 和 limitation，最终答案只能引用 evidence ledger，而不能直接依赖 planner trace 或 caption shortcut。
2. **更像真实 coding agent。** 我们迁移的是 Claude Code 的完整 harness：tool registry、workspace、context compact、subagent isolation、task DAG、permission、event stream，而不是只迁移 tool calling。
3. **Subagent 更一般。** ParaVT 的 subagents 主要并行检查不同 temporal crops；我们的 workers 可以按时间、空间、OCR、实体、动作、验证等维度拆分，返回结构化 observation。
4. **Training-free 到训练数据的闭环更明确。** VisProg 可以 training-free 执行，但不天然产生高质量行为训练数据；我们从一开始就把 trace 设计成 SFT / preference / RL 可用的数据资产。

一句话区分：

> VisProg 是“LLM 写视觉程序”；ReTool-Video 是“递归调用更多视频工具”；ParaVT 是“并行视频工具调用训练”；我们是“让多模态 Agent 像 Claude Code 一样工作”。

## 5. 总体系统设计

整体结构：

```text
用户问题 / benchmark case
  -> Main Agent
  -> Task Planner
  -> Visual Tool Registry
  -> Tool Executor
  -> Evidence Workspace
  -> Evidence Ledger
  -> Answer Agent
  -> Verifier
  -> 最终答案 / 追加工具调用
```

### 5.1 主 Agent 可以是纯文本 Agent

P0 阶段建议把 Main Agent 设计成纯文本 planner/controller。它不直接看图或视频，而是通过工具获得视觉 observation。这和 VisProg 中 GPT-3 不直接看图、只生成 visual program 的思想相近，但我们的执行方式更像 Claude Code 的多轮 loop。

```text
Text-only Main Agent
  -> chooses visual tools
  -> reads structured observations
  -> updates task DAG and ledger
  -> asks verifier whether evidence is enough
  -> answers with cited evidence
```

这样做的好处是：

- 防止主模型偷看图后直接凭直觉回答，保证视觉信息都来自工具轨迹。
- 每个视觉结论都能追溯到具体 tool call、artifact、timestamp、bbox。
- 更容易诊断失败：是 planner 没选对工具、crop 错了、OCR/VQA 错了，还是 verifier 没拦住。
- 轨迹天然是 `state -> tool_call -> observation -> next_tool_call -> answer`，适合后续训练。
- 主 Agent 可以用强文本 LLM，视觉工具可以混合使用 Qwen-VL、InternVL、OCR、SAM、Grounding DINO、tracking model 等。

后续可以做一个重要消融：

| 设置 | 含义 | 目的 |
| --- | --- | --- |
| Text-only Main Agent | 主 Agent 只能读 observation/ledger，不能直接看图。 | 验证 harness 是否足以支撑复杂视觉推理。 |
| Multimodal Main Agent | 主 Agent 可以直接看图/视频帧。 | 测试直接视觉访问是否提升或污染证据链。 |
| Hybrid Main Agent | 默认 text-only，只在 verifier 要求时看 selected artifacts。 | 在证据可控和视觉直观性之间折中。 |

### 5.2 主 Agent 的输入边界

主 Agent 不直接吞完整视频、全部帧、长 OCR 和长 caption。它看到的是：

- 用户问题；
- 当前任务状态；
- 可用工具列表；
- compact evidence ledger；
- verifier 给出的 missing evidence；
- 少量必要 artifact path。

视觉工具输出写入 workspace，主上下文只保留结构化 observation。

## 6. 主线一：多工具整合的视觉导向 Harness

### 6.1 Evidence Workspace

Evidence workspace 是多模态版本的代码工作区。

建议目录：

```text
visual-coding-agent-harness/
  runs/<run_id>/
    input/
      task.json
      media/
    artifacts/
      frames/
      clips/
      crops/
      masks/
      ocr_regions/
      retrieved_images/
    observations.jsonl
    ledger.md
    tasks.json
    trace.jsonl
    answer.json
```

各部分作用：

| 文件/目录 | 作用 |
| --- | --- |
| `artifacts/` | 保存原图、关键帧、clip、crop、mask、OCR patch 等视觉证据。 |
| `observations.jsonl` | 每次工具调用后的结构化观察。 |
| `ledger.md` | 给 Answer Agent 使用的短证据账本。 |
| `tasks.json` | 子任务 DAG：状态、依赖、owner、blocked reason。 |
| `trace.jsonl` | 完整工具轨迹，用于后续训练和分析。 |
| `answer.json` | 最终答案、引用证据、置信度、未解决条件。 |

### 6.2 P0 工具列表

第一版工具不要贪多，先形成闭环。

| 工具 | 作用 | 输入 | 输出 |
| --- | --- | --- | --- |
| `sample_frames` | 粗粒度探索视频 | video path、采样策略 | frame path、timestamp |
| `seek_clip` | 提取目标时间段 | video path、start/end | clip path、关键帧 |
| `crop_region` | 裁剪局部区域 | image/frame path、bbox | crop path |
| `zoom_region` | 放大小目标或文字 | image/frame path、bbox、scale | zoom crop path |
| `ocr_region` | 读取文字、UI、标牌、文档 | image/crop path | text span、bbox、confidence |
| `caption_image` | 粗略语义描述 | image/crop path | 短 caption 和不确定性 |
| `inspect_region` / `verify_local_claim` | 对局部区域做事实检查，而不是让另一个模型替主模型回答整题 | image/crop path、local question 或 claim | artifact-linked local observation |
| `compare_frames` | 比较前后状态或动作变化 | 多个 frame path | temporal/change observation |
| `detect_objects` | 找候选物体/实体 | image path、可选类别 | boxes、labels、confidence |
| `write_observation` | 记录证据 claim | claim、source artifacts | observation / ledger candidate |
| `summarize_evidence` | 压缩观察成证据账本 | observation ids | updated ledger |
| `verify_answer` | 验证答案是否有证据支撑 | answer、ledger | pass/fail、missing evidence |

### 6.3 为什么还需要 OCR / caption / local VQA 类工具

多模态模型本身确实具备 OCR、caption、VQA、物体识别等能力。因此这些工具的价值不是“给模型增加一种它完全没有的能力”，而是把模型隐式感知变成显式、可定位、可复核、可训练的视觉操作。

| 直接问 MLLM | 工具化视觉操作 |
| --- | --- |
| 模型看整张图/整段视频后直接回答。 | 指定时间、区域和局部问题后调用工具。 |
| 输出是一段自然语言，很难知道证据来自哪里。 | 输出带 frame、timestamp、bbox、crop path、confidence。 |
| 错了难诊断：没看到、看错、推理错混在一起。 | 可以定位错误来自时间段、区域、OCR/VQA、ledger 或 answer。 |
| 长视频/多图时上下文容易混。 | 观察写入 evidence ledger，压缩后再回答。 |
| 不容易变成训练信号。 | 每步 tool call 都能标注对错，用于 SFT/preference/RL。 |

因此 `inspect_region` 更准确的定义是：

> 对一个明确视觉区域执行局部事实检查，并返回 artifact-linked observation。

它可以由同一个 MLLM、更小的 VLM、更强的 inspector VLM、专用 classifier，或者 ensemble 实现。重点不在模型大小，而在角色边界：Main Agent 决定看哪里、问什么、证据是否足够；`inspect_region` 只回答指定区域的局部事实。

视觉工具可以分成三类：

| 类别 | 例子 | 价值 |
| --- | --- | --- |
| 模型本身会，但工具化后更可控 | caption、OCR、local VQA、attribute recognition | 定位、约束、证据化、可审计。 |
| 模型本身较弱，专用工具明显补强 | tracking、segmentation、精确 detection、temporal retrieval、chart/layout parser | 专业模块更稳定，也更容易评估。 |
| harness 管理能力 | write_observation、summarize_evidence、verify_answer、compact_trace、spawn_worker | 管理证据、上下文、协作和训练数据。 |

论文贡献应该重点放在第二类和第三类，第一类只作为基础视觉操作 primitive，不能包装成主要贡献。

### 6.4 P1/P2 Foundation Model 工具

后续可以扩展为更完整的 visual foundation model 工具池：

| 工具类别 | 可选 foundation model / 模块 |
| --- | --- |
| Segmentation | SAM / SAM2，mask proposal，object mask refinement。 |
| Detection | Grounding DINO、OWL-style open-vocabulary detector、Florence-style detector、YOLO fast baseline。 |
| OCR / document | PaddleOCR、EasyOCR、TrOCR、layout parser。 |
| Video tracking | SAM2 video propagation、ByteTrack、DeepSORT、optical flow。 |
| Temporal retrieval | CLIP/SigLIP frame embedding、caption index、moment retriever、temporal grounding model。 |
| VQA / caption | Qwen-VL、GPT-4o/4.1-style VLM、LLaVA/InternVL-style open-source VLM。 |
| Depth / pose / geometry | depth estimator、pose estimator、3D/4D spatial understanding tools。 |
| External knowledge | logo/product/place/entity lookup、web search，但必须绑定视觉证据。 |
| Memory / skill | 数据集协议、常见失败模式、成功工具轨迹、任务专用 skill。 |

### 6.5 Observation Formatter

工具结果不能只是一句话。每个工具输出都要变成可复核 observation：

```json
{
  "observation_id": "obs_0042",
  "tool": "ocr_region",
  "input_artifacts": ["artifacts/crops/frame_01240_box_3.png"],
  "time_range": [41.2, 41.2],
  "regions": [
    {"frame": "frame_01240.jpg", "bbox": [312, 180, 498, 330]}
  ],
  "claim": "The sign reads 'EXIT'.",
  "raw_output": "EXIT",
  "confidence": 0.91,
  "supports_question": true,
  "limitations": "Text is partially blurred; verifier may request another frame."
}
```

这样 Answer Agent 不能只依赖 caption 或长 trace，而必须引用具体 observation、frame、timestamp、bbox 和 artifact path。

### 6.6 Context Compact

迁移 Claude Code 的 context compact 机制：

1. micro-compact：旧 tool result 只保留工具名、observation id、artifact path 和一句摘要。
2. ledger compact：把多条 observation 压缩成 answer-ready evidence table。
3. trace archive：完整 raw trace 只写入 `trace.jsonl`，不长期留在活跃上下文。

这对应 VideoSEAL 的核心问题：防止 planner trace、caption shortcut 或长上下文污染最终答案权威。

## 7. 超长视频调度：像查大型代码库一样查视频

超长视频不能直接塞给模型，也不能一次性抽很多帧让模型硬看。更稳的方式是把视频当作一个大型代码仓库：先建立索引，问题来了以后再粗搜定位、局部深挖、证据验证。

### 7.1 初始视频索引

初始 shot / scene segmentation 建议用传统或专用轻量算法，而不是让 MLLM 判断所有边界。TransNetV2、PySceneDetect、ffmpeg scene filter、embedding change score 都可以作为 indexing 层工具。

P0 建议混合三种切分：

| 切分方式 | 作用 |
| --- | --- |
| Uniform windows | 保底覆盖，例如每 30s 或 60s 一段，防止 shot detector 漏掉静态长镜头。 |
| Shot boundary detection | 找视觉镜头切换，适合电影、剪辑、新闻、短视频。 |
| Adaptive keyframe sampling | 在每个 window/shot 内选择代表帧和高变化帧。 |

因为不同视频类型差异很大：

- 电影/剪辑视频：shot boundary 很有用。
- 监控/egocentric/课堂/会议：镜头不切，uniform window 和 motion score 更重要。
- 屏幕录制/GUI：画面变化小，但 OCR change score 很重要。
- 体育/游戏：镜头切换和动作变化都重要。

建议初始 index pipeline：

```text
video
  -> uniform temporal windows
  -> shot boundary detector
  -> motion/change score
  -> sparse keyframes
  -> sparse OCR / caption / embedding
  -> merged timeline_index.json
```

`timeline_index.json` 示例：

```json
{
  "unit_id": "u_0042",
  "time_range": [1260.0, 1290.0],
  "source": ["uniform_window", "shot_boundary"],
  "keyframes": [
    "artifacts/frames/u_0042_k0.jpg",
    "artifacts/frames/u_0042_k1.jpg"
  ],
  "caption": "A person stands near a table with several cups.",
  "ocr": [],
  "embedding_id": "emb_0042",
  "motion_score": 0.37,
  "change_score": 0.52
}
```

这些传统/轻量工具不是核心智能贡献，而是 harness indexing 基础设施。真正贡献在 query-driven retrieval、parallel visual workers、evidence ledger、verifier refinement 和 trace-to-training。

### 7.2 Query-Driven 调度循环

超长视频回答可以采用：

```text
Index -> Retrieve -> Parallel Inspect -> Ledger -> Verify -> Refine/Answer
```

具体步骤：

1. Build 或 load video index。
2. Main Agent 把问题解析成 evidence goals。
3. `retrieve_moments(query, top_k)` 返回候选时间单元。
4. 对 top-k 候选片段启动 parallel visual workers。
5. workers 把局部检查写成 observations。
6. Ledger Agent 更新 evidence ledger。
7. Verifier 检查证据是否足够。
8. 如果不足，扩大 top-k、缩窄时间窗、dense sample、crop/zoom 或 track object。
9. 如果足够，Answer Agent 基于 ledger 回答。

主 Agent 的关键调度决策包括：

- `top_k` 取多少；
- 先查 transcript/OCR，还是先查 visual embedding；
- 是否并行检查多个候选片段；
- 是否扩大或缩小时间窗口；
- 是否从 sparse sampling 切换到 dense sampling；
- 是否调用 tracking、OCR、segmentation；
- 是否让 verifier 先判断证据是否足够；
- 何时停止。

### 7.3 超长视频例子

问题：视频中第一次有人把红色杯子从桌上拿走后，桌上还剩几个杯子？

调度：

```text
1. retrieve_moments("red cup on table person picks up cup", top_k=8)
2. Temporal workers 并行检查 top-8 clips
3. 找到最早可信事件 02:13:42-02:13:50
4. seek_clip(02:13:35, 02:13:55)
5. dense_sample_frames
6. crop_region(table area before and after pickup)
7. inspect_region(after_crop, "How many cups remain on the table?")
8. compare_frames(before, after)
9. write_observation
10. verify_answer
11. answer with cited evidence
```

这比 ParaVT 更一般：ParaVT 主要是 one-turn parallel crop-video；我们的系统是 persistent video index + query-driven retrieval + parallel specialist workers + evidence ledger + verifier-controlled refinement。

## 8. 主线二：协同合作 Workflow

### 8.1 角色划分

| 角色 | 职责 | 上下文边界 |
| --- | --- | --- |
| Main Agent | 理解任务、创建 DAG、选择工具、管理预算、决定是否回答。 | 看问题、工具表、任务状态、compact ledger。 |
| Temporal Worker | 定位相关视频时间段。 | 看视频元数据、采样帧、局部 temporal objective。 |
| Spatial Worker | 检查物体、区域、空间关系。 | 只看选中的 frame/crop。 |
| OCR/UI Worker | 读文字、UI、图表、文档、标牌。 | 只看 OCR 相关 crop。 |
| Entity/Knowledge Worker | 把视觉实体链接到外部知识。 | 只在视觉证据明确后做外部检索。 |
| Ledger Agent | 把 observation 压缩成 evidence ledger。 | 看 observation，不看完整 raw trace。 |
| Answer Agent | 基于 ledger 生成答案。 | 只看问题、ledger、候选约束。 |
| Verifier | 检查答案和证据是否一致。 | 看 answer、ledger，必要时查看局部 artifact。 |

关键是 subagent isolation：每个 worker 可以读很多帧或 crop，但返回给主 Agent 的只有结构化 observation 和短建议，避免污染主上下文。

### 8.2 工作流

1. Main Agent 解析问题：
   - 需要找时间段吗？
   - 需要看局部区域吗？
   - 需要 OCR 吗？
   - 需要外部知识吗？
   - 答案是否要求 temporal order / spatial relation / counting？
2. Main Agent 创建 task DAG：
   - 粗探索；
   - 时间定位；
   - 局部裁剪；
   - OCR / detection / VQA；
   - evidence ledger；
   - answer；
   - verification。
3. 先调用低成本工具：
   - frame sampling；
   - coarse caption；
   - rough OCR；
   - cheap retrieval。
4. 对互不依赖的目标启动 subagents：
   - 不同时间窗并行检查；
   - 不同候选区域并行检查；
   - 不同候选答案并行找证据；
   - OCR、object、action 分开检查。
5. Tool Executor 执行工具，把 artifact 和 observation 写入 workspace。
6. Ledger Agent 汇总相关 observation，更新 `ledger.md`。
7. Answer Agent 只基于 ledger 回答。
8. Verifier 检查：
   - 答案是否引用具体证据；
   - 是否检查了正确时间段；
   - 是否检查了正确区域；
   - 是否有 unsupported claim；
   - 是否 evidence misalignment；
   - 是否工具过度调用。
9. 如果验证失败，Verifier 返回 targeted follow-up tool call，而不是重新开始。

### 8.3 Worker 通信协议

Worker request 示例：

```json
{
  "task_id": "t_temporal_02",
  "role": "temporal_worker",
  "objective": "Find the time window where the person picks up the red cup.",
  "allowed_tools": ["sample_frames", "seek_clip", "compare_frames", "caption_image"],
  "input_artifacts": ["input/media/video.mp4"],
  "budget": {"max_tool_calls": 8, "max_seconds": 90},
  "return_format": "observations_json"
}
```

Worker response 示例：

```json
{
  "task_id": "t_temporal_02",
  "status": "completed",
  "observations": ["obs_0011", "obs_0012"],
  "recommended_next_actions": [
    "crop frame_00480 around the table to confirm cup color"
  ],
  "confidence": 0.72,
  "missing_conditions": [
    "Need local crop because sampled frame is low resolution"
  ]
}
```

这就是 Claude Code team protocol 在视觉任务中的迁移：所有协作都带 task id、role、allowed tools、budget、return format、missing conditions。

### 8.4 一个具体例子

问题：视频中，人在读完标牌之后立刻做了什么？

流程：

1. Main Agent 判断需要两个证据：读标牌的时间点，以及之后的动作。
2. Temporal Worker 采样帧，定位疑似看标牌的时间段。
3. OCR Worker 裁剪标牌，确认文字。
4. Spatial/Action Worker 检查之后几秒的人物动作。
5. Ledger Agent 写入：
   - `obs_07`：00:41.2 标牌可见，crop path，OCR 结果。
   - `obs_11`：00:43.0-00:45.5 人转向右侧并打开门，frame path。
6. Answer Agent 回答：读完标牌后，人物转向右侧并打开门。
7. Verifier 检查时间顺序和证据是否足够；若不足，要求 `compare_frames` 检查 00:41-00:46。

## 9. 主线三：Training-Free 观察到工具调用训练

### 9.1 Stage A：Training-Free Harness

第一阶段不训练模型，先观察模型在 harness 中自然怎么用工具。

需要记录：

- 模型调用了哪些工具；
- 工具参数是否正确：timestamp、bbox、class、local question；
- crop/zoom/seek 是否命中正确证据；
- 是否过早回答；
- 是否过度调用工具；
- final answer 是否被 evidence ledger 支持；
- verifier feedback 后能否恢复。

对比 baseline：

| Baseline | 用途 |
| --- | --- |
| Direct MLLM | 不用工具时的原始能力。 |
| Caption-only agent | 检查文本化视频是否造成 evidence shortcut。 |
| Few-tool agent | 验证少量 zoom/crop/retrieve 是否足够。 |
| Multi-tool no ledger | 验证“堆工具但无 harness”是否增加噪声。 |
| Full visual coding-agent harness | 验证 registry + workspace + subagent + compact + verifier 的完整收益。 |

2026-06-04 实验修正：

- 当前优先跑 no-budget/free-explore 质量路径，而不是先压工具成本。
- 目标是验证工具使用是否提高证据召回、final_rate 和可追溯性。
- 若 free-explore 仍答错，要看是工具没召回证据，还是 AnswerAgent/Verifier 没仲裁证据冲突。
- 成本、latency、调用次数先作为记录指标；等质量路径成立后，再进入效率预算和 Agentic RL。

当前 VideoMME 小样本信号：

- default AnswerAgent 三例：1/3，final_rate 66.7%，incomplete_rate 33.3%。
- `611-2` free-explore：final_rate 100%，incomplete_rate 0%，但 final A vs GT D。
- 该 free-explore trace 找到了支持 GT D 的 observation，但 final 选择了后续 A-supporting observation。

这说明工具使用能增强 evidence recall 和 finality，但 AnswerAgent/Verifier 仍是准确率瓶颈。

### 9.2 Stage B：轨迹过滤和失败分类

每条轨迹都要打标签：

| 标签 | 含义 |
| --- | --- |
| `successful_grounded` | 答案正确，证据正确，工具成本合理。 |
| `answer_correct_evidence_wrong` | 答案看似正确，但 trace 没有支撑，VideoSEAL-style evidence misalignment。 |
| `tool_format_failure` | JSON 或工具格式失败，ParaVT 中 format collapse 类问题。 |
| `wrong_visual_action` | crop/seek/zoom 没有命中目标。 |
| `under_tool_use` | 该看证据时没有调用工具。 |
| `over_tool_use` | 调用了太多无关或高成本工具。 |
| `recovered_by_verifier` | 初始失败，但 verifier 的 targeted follow-up 修复了问题。 |
| `conflicting_evidence_unresolved` | ledger 中存在多个选项支持，final 没有显式仲裁。 |
| `weak_caption_shortcut` | final 依赖 caption/推断型 observation，而 limitation 已说明证据不强。 |
| `option_grounding_drift` | 工具调用只传 letter options 或选项文本不完整，导致 MCQ 证据归因漂移。 |

这些标签就是后续 SFT、preference tuning、RL 和 verifier 训练的数据资产。

### 9.3 Stage C：可训练模块

不必一开始训练完整 MLLM，可以先训练 harness 内部组件。

| 模块 | 训练目标 | 数据来源 |
| --- | --- | --- |
| Tool selection policy | 学会何时调用 crop/OCR/seek/detect/verify。 | 成功与失败轨迹。 |
| Tool argument generator | 学会生成正确 bbox、timestamp、class、local question。 | 工具参数 + verifier/evidence 标签。 |
| Observation summarizer | 把 raw tool output 写成结构化 claim。 | `observations.jsonl` 和 accepted ledger rows。 |
| Verifier | 判断答案是否缺证据、是否有 unsupported claim。 | pass/fail verdict、人类审核、自动指标。 |
| Stopping policy | 判断证据是否足够，可以回答。 | 成功轨迹 vs premature answer。 |
| Recovery policy | 失败后选择 targeted follow-up action。 | recovered trajectories。 |

### 9.4 Stage D：SFT / Preference / RL

训练路线：

0. Trace filtering first：
   - 先过滤 free-explore 轨迹，保留 evidence support、冲突解决、引用一致的样本；
   - 不把长但 unsupported 的探索轨迹当正样本。
1. SFT cold start：
   - 用 `successful_grounded` 轨迹教模型工具格式和基本调用策略。
2. Step-wise preference tuning：
   - 偏好命中正确证据、成本更低、过程更短的工具调用；
   - 惩罚 irrelevant tool、wrong crop、unsupported answer。
3. Verifier-guided improvement：
   - 让 verifier 评价候选工具轨迹；
   - 用 evidence support、missing conditions、cost 做偏好信号。
4. RL / GRPO-style training：
   - reward 同时考虑答案、证据、工具格式、视觉动作命中率和成本；
   - 避免 ParaVT 提到的工具 prior 副作用：format collapse、skip-tool shortcut、parseable-but-useless tool call。

Reward 草图：

```text
R = answer_accuracy
  + evidence_support
  + temporal_grounding
  + spatial_grounding
  + visual_action_hit
  + format_validity
  - unsupported_claim_penalty
  - unnecessary_tool_cost
  - premature_stop_penalty
```

核心是：不能只优化 final answer，要优化 process。

## 10. 评估方案

### 10.1 指标

| 指标 | 检查内容 |
| --- | --- |
| Answer accuracy | 最终答案是否正确。 |
| Evidence support | 答案 claim 是否被 observation 支撑。 |
| Temporal grounding | 是否检查了正确视频时间段。 |
| Spatial grounding | crop/zoom/detect 是否命中正确区域。 |
| Tool-use precision | 调用的工具是否必要。 |
| Tool-use recall | 需要工具时是否真的调用。 |
| Recovery success | verifier 指出问题后能否修复。 |
| Cost / latency | 工具调用次数、耗时、高成本工具使用。 |
| Context pressure | 长 trace 是否降低 evidence support。 |
| Overthinking | 相对人类/oracle 轨迹是否冗余。 |
| Final rate | 是否稳定产出合规 final JSON。 |
| Incomplete rate | 是否因 max rounds / missing evidence 结束。 |
| Conflict rate | ledger 中是否出现多个互斥答案支持。 |
| Option-support consistency | final 选项、rationale、citation 的选项支持是否一致。 |
| Unsupported final rate | final 是否依赖 navigation-only、弱 caption 或外部知识推断。 |

### 10.2 消融实验

| 消融 | 要回答的问题 |
| --- | --- |
| full trace vs compact ledger | 压缩 evidence 是否减少 hallucination/context pollution？ |
| no verifier | 没有 verifier 时 unsupported answer 是否上升？ |
| no subagents | 共享上下文是否伤害长视频/多图任务？ |
| sequential vs parallel workers | ParaVT-style 并行是否减少错误传播？ |
| text-only observation vs artifact-linked observation | 是否必须保留视觉 artifact provenance？ |
| few tools vs full registry | 工具数量是否只有在 harness 化后才有效？ |
| rule router vs model-chosen tools | coding-agent-style 自主调度是否优于规则流程？ |
| no cost budget / free-explore | 没有成本预算时是否提升 evidence recall 和 final_rate？是否暴露更多冲突证据或 wrong final？ |
| bounded budget after verifier | verifier 稳定后，较小预算是否仍能保持 grounded accuracy？ |

## 11. 开发里程碑

### Milestone 0：项目骨架

当前已建立：

```text
visual-coding-agent-harness/
  docs/
  src/
  experiments/
  artifacts/
```

### Milestone 1：最小 Visual Harness

实现：

- `agent_loop.py`
- `registry.py`
- `workspace.py`
- `events.py`
- P0 tools：sample、crop、zoom、OCR、caption/VQA placeholder。
- `trace.jsonl`、`observations.jsonl`、`ledger.md` 写入。

交付：一个图像 case 和一个短视频 case 跑通。

### Milestone 2：Evidence Ledger + Verifier

实现：

- structured observation formatter；
- ledger compact；
- answer-with-citations；
- verifier pass/fail + targeted follow-up。

交付：20 个 case 上比较 direct answer 和 harness answer。

### Milestone 3：Visual Subagents

实现：

- worker request/response schema；
- temporal worker；
- spatial worker；
- OCR worker；
- parallel execution。

交付：长视频任务中展示并行时间窗检查和主上下文压缩。

### Milestone 4：轨迹数据和分析

实现：

- free-explore trace collection；
- trajectory labeling；
- tool-use metrics；
- failure taxonomy；
- conflict / option-consistency metrics；
- SFT/preference 数据导出格式。

交付：第一版 trajectory dataset report，明确哪些 free-explore 轨迹可作为训练正样本，哪些只能作为失败案例。

### Milestone 5：工具调用训练原型

在 AnswerAgent/Verifier 能过滤冲突轨迹后，先训练一个组件：

- tool selection classifier；
- verifier；
- observation summarizer；
- 或小规模 tool-call SFT。

交付：在 held-out cases 上展示行为改善。

## 12. 第一版技术选择

建议第一版保守实现：

- Python harness；
- 本地 artifact workspace；
- JSON schema-like tool definitions；
- OpenCV/PIL/moviepy 做基础图像视频处理；
- OCR 先用可用本地包或 placeholder；
- MLLM/API adapter 先抽象，后续可接 Claude、OpenAI、Qwen-VL、InternVL 等；
- foundation model 工具先走 wrapper，不和主 loop 耦合。

第一版目标不是追求最高 benchmark，而是让所有视觉动作都可执行、可记录、可压缩、可复核、可训练。

## 13. 汇报总结

Claude Code 的成功说明：强 Agent 不是单纯靠模型，而是模型被放进一个可执行、可管理、可恢复的工作环境。对于多模态复杂任务，这个工作环境应该是 visual evidence workspace。

我们的方案是：

1. 把多种 visual foundation models 包装成 typed visual tools，形成多工具视觉 harness。
2. 用 Main Agent、visual workers、ledger、Answer Agent、Verifier 组成协同 workflow。
3. training-free 先观察和收集工具轨迹，再把成功、失败和恢复轨迹转成 SFT/preference/RL 数据，改善工具调用和视觉动作执行能力。

最终贡献不是“加了更多工具”，而是让多模态工具变得可组合、可审计、可压缩、可验证，并能持续产生训练数据。
