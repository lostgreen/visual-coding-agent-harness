# Prompt Stack Diet + Segment Zoom-in Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 4B 文本 planner 在 VideoMME 长问 (`temporal_order` / `needle_local`) 上能像 Claude Code 一样有效地探索视频证据：(0) 取消"cheap / expensive / verifier"三类预算的计费机制，只保留 `max_rounds` + `max_tool_calls_per_round` 两个硬限制——把工具选择的成本权衡彻底交给 planner 自己的 ReAct 推理；(1) 把每轮 prompt 中 ~70 行不变的样板压成 skill/route 感知的精简版；(2) 把工具调用关键结果（尤其是 `read_segment_detail` 的 `visual_caption` / `asr_summary`）回流到 ledger 让 planner 真的看到；(3) **把"段内目标定位"重设计成 Layer 0/1/2 三层流水线**：Layer 0 在 Compact Scene Index 直接渲染每段的 "asr mentions @ timestamp" 让 planner 一眼选段而不必调任何工具；Layer 1 新工具 `locate_targets_in_segment` 在指定段内用严格 alias + word-boundary 匹配跑 ASR/OCR/visual_caption 文本扫描，输出带时间戳的 candidates + 合并后的 anchor windows（**纯文本，零 VLM**）；Layer 2 新工具 `verify_segment_anchors` 只对 Layer 1 给出的 anchor 跑 focused VLM，确认后才写入 timeline.md 与 evidence_table。candidate（routing 候选）与 evidence（已确认事实）严格分桶，AnswerAgent 只看 evidence 桶。

> 2026-06-08 执行状态：Task 0 / prompt schema focus / `read_segment_detail` ledger digest / Layer 0 ASR mentions / Layer 1 `locate_targets_in_segment` / Layer 2 `verify_segment_anchors` / legacy `zoom` + `expand_window` soft rewrite 已落地。本计划中的 `scan_segment` 仅保留在 Background / archeology 段落里，不是执行目标。当前验证：`PYTHONPATH=src:. pytest -q` -> `381 passed`。

## 2026-06-08 Task 执行状态总览

| Task | 状态 | 本轮结果 |
|------|------|----------|
| Task 0：取消 per-class 工具预算 | 完成 | 删除 cheap/expensive/verifier 预算与 prompt budget 文案；legacy CLI 参数保留但忽略。 |
| Task 1：Tool Schema 按 active_skill 过滤 | 完成 | `_tool_schema_block(active_skill=...)` 按 skill allowed_actions 过滤；首轮 schema 也不再暴露 `zoom` / `expand_window`。 |
| Task 2：Final Gate 按 route 折叠 | 完成 | temporal / needle final gate 明确引导 `locate_targets_in_segment -> verify_segment_anchors`。 |
| Task 3：Skill Catalog 焦点条目 | 未完成 / 延后 | 本轮保留完整 skill catalog，仅完成 tool schema focus；后续可单独做 catalog token diet。 |
| Task 4：`read_segment_detail` 回填 `nav_digest` | 完成 | detail pack 返回 `nav_digest`，并把 visual / ASR / OCR / targets 压成 planner 可读摘要。 |
| Task 5：Navigation Summary 渲染 digest | 完成 | compact ledger 对 `read_segment_detail` / `target_coverage` / `search_segments` / `ground_question` / locator 展开短 claim；未新建复杂 parser，直接使用 claim/nav_digest。 |
| Task 5b：Compact Scene Index 渲染 ASR mentions | 完成 | `SceneIndex.summary(target_hints=...)` 渲染 `asr mentions: target @ ~T`。 |
| Task 6a：alias / strict matcher | 完成（实现形态调整） | 未新建 `tools/aliasing.py`；严格匹配 helper 内联在 `tools/navigation.py`，覆盖 corner cases。 |
| Task 6b：`locate_targets_in_segment` | 完成 | 新增文本 locator，输出 `candidates` 与 `anchors_for_vlm`；定位结果为 navigation-only。 |
| Task 6c：`verify_segment_anchors` | 完成 | 新增 focused VLM verifier，confirmed rows 写入 evidence table 与 `timeline.md`。 |
| Task 6：旧 scan 占位 | 跳过 | 已被 Task 6a/6b/6c 替代；`scan_segment` 不作为执行路径。 |
| Task 7：写入 skill allowed_actions | 完成 | 三个 needle/timeline skill 已允许 `locate_targets_in_segment` 与 `verify_segment_anchors`。 |
| Task 8：611-2 trajectory replay | 已挂起 / 待结果 | KML 611-2 replay 已以 detached job 启动；等待后续读取 trajectory 验证真实 planner 是否走 `locate -> verify -> timeline/final`。 |
| Task 9：弃用旧工具 | 完成 | planner schema 不再暴露 `zoom` / `expand_window`；runtime 对旧调用软重写到 locator。 |

### 2026-06-08 review follow-up

这轮根据 code review 又补了三个证据质量风险：

- `locate_targets_in_segment` 不再每个 target 只取一个 best/first match；默认 `top_k_per_target=3`，同一 target 的多次 ASR/OCR/visual-caption 命中会保留为多个候选，再合并成 anchor windows。
- 常见 single-token target（例如 `David` / `Apollo`）不再走裸 `full_name` 强命中；只有带 `Bernini` / `Borghese` / `statue` / `sculpture` / `shown` 等上下文时才形成 `contextual_single_name` 候选，且标记为 `routing_only_low_confidence`。rare proper noun（如 `Persephone` / `Anchises` / `Aeneas`）仍可作为低/中置信 route hint。
- `verify_segment_anchors` 不再无条件把多个 anchors union 成一个长窗口；当 union window 超过 45s 时，按 anchor 拆成多次 focused VLM request，并把 `verify_windows` 写回 raw output。

Task 8 当前状态：真实 611-2 trajectory replay 已在 KML 挂起，但尚未读取结果；后续需要验证 planner 是否实际走 `SceneIndex ASR mentions -> locate_targets_in_segment -> verify_segment_anchors -> read_timeline_sorted/final`。

KML replay job：

- run root：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_anchor_hardened_611_2_5c87449_20260608`
- pid path：`/tmp/videomme_agent_anchor_hardened_611_2_5c87449_20260608.pid`
- log path：`/tmp/videomme_agent_anchor_hardened_611_2_5c87449_20260608.log`
- code commit：`5c87449`

**Architecture:** 改动全部在 `src/visual_coding_agent_harness/`，对外契约（`IterativeVisualAgent.run` 入口、ReAct `continue/final` JSON、skill 路由）保持不变。五个手术点：
- (0) `agents/iterative_agent.py` + `prompt_stack.py`：删除 `_CHEAP_TOOLS` / `_EXPENSIVE_TOOLS` / `_VERIFIER_TOOLS` 常量、`AgentBudget.cheap_tool_budget` / `expensive_tool_budget` / `verifier_tool_budget` / `free_exploration` 字段、`_tool_budget_available` 检查、`_budget_snapshot_block` 中的"cheap=X expensive=Y verifier=Z"行、`reflection_memory` 中 `tool_budget_exhausted` 规则。保留 `max_rounds` + `max_tool_calls_per_round`。
- (A) `prompt_stack.py`：`_tool_schema_block` 与 `_final_gate_block` 接受 `active_skill` / `route` 参数，按 `allowed_actions` 过滤工具签名，按路由折叠规则。`skill_catalog_prompt()` 在已选 skill 稳定的轮次只显示精简 catalog。
- (B) `workspace.py`：`_format_navigation_entry` 为 `read_segment_detail` / `search_segments` / `ground_question` 渲染单行 digest（target_matches + 视觉 caption 截短 + ASR 摘要截短），不再只输出 `obs_0002: read_segment_detail` 这种空壳。`read_segment_detail` 的工具实现回填 `nav_digest` 字段以让格式化函数无需 re-parse `regions`。
- (B′) `video_index.py` + `prompt_stack.py`：`SceneIndex.summary()` 增加 `target_hints` 参数；当 MCQ 已选定 targets 时，在每个 segment 摘要行下方多渲染一行 "asr mentions: `target` @ ~T1s, `target` @ ~T2s"，让 planner 不用调任何工具就能挑选目标段。
- (C) **Layer 1 / Layer 2 段内定位流水线**——`tools/navigation.py` 新增 `locate_targets_in_segment`（cheap，文本 only，读 raw ASR sentences + OCR frames + visual_caption，用严格 alias + word-boundary 匹配输出 candidate timestamps 与合并后的 anchor windows）；`tools/inspector.py` 新增 `verify_segment_anchors`（focused VLM，只在 Layer 1 给出的 anchors 跑 8 帧验证，输出 confirmations + rejections + timeline_rows，仅 confirmations 写入 `timeline.md` 与 evidence_table）。前置：`video_index.py` 的 `VideoMapSegment` 增加 `asr_sentences: list[dict]` 字段（每条含 `start_sec`/`end_sec`/`text`），dual_source builder 把原始 ASR 句级时间戳 plumb 进来；如原始 ASR 仅段级，按 char offset 比例估算并在 `limitations` 标注。`agents/skills/specs.py` 把 `locate_targets_in_segment` + `verify_segment_anchors` 加入三个 needle/timeline skill 的 `allowed_actions`。candidate / evidence 严格分桶：locate 结果进新 `Navigation Candidates` 渲染桶，verify 结果进既有 `Long-Term Visual Evidence` 桶（AnswerAgent 只读后者）。

**Tech Stack:** Python 3.11, dataclasses (frozen), pytest, Qwen3-VL backend via `transformers.AutoModelForImageTextToText`，复用 `BackendRequest`/`RoutedBackend` 路由约定。

---

## Background：从 611-2 trajectory 提炼的 5 个具体观察

来自 `/Users/lostgreen/Downloads/611-2_agent_v2 (3).md`（runs root 见 onboarding 文档第 58 行）：

1. **每轮 prompt 中 ~70 行恒定样板**：`Base Identity`(5L) + `Route Playbook`(12L) + 完整 `Skill Catalog`(8L) + 完整 `Tool Schema`(22L) + `Final Gate`(17L) + `Response Contract`(3L)。整段 20 轮，~1400 行重复内容。
2. **`read_segment_detail` 的关键结果在 ledger 中只剩工具名**：Round 2 `obs_0002: read_segment_detail` 后续每轮都只渲染为这一行，`visual_caption` / `asr_summary` 全部丢失。但 `regions[0].asr_text` 实际含有 `'Apollo and Daphne,' 'David,' and 'The Rape of Persephone'`（611-2 正确答案 D 的直接线索），planner 一次都没看到。
3. **planner 因此把同两段连读 5 遍**：Round 1/2/3/4/5 都发 `read_segment_detail(seg_0001)+read_segment_detail(seg_0002)`，触发 `route_repairs` 一次但 ledger 不变，cheap 预算从 16 烧到 8。
4. **300s 粒度太粗对 temporal_order 致命**：question 问"in a single scene"四件作品的顺序，需要在某一段内 sub-segment 级别识别。当前 `read_segment_detail` 返回的是 dual_source 索引缓存（不读像素），`caption_segment` 一次对整 300s 做一句 caption（结果在 Round 9-15 全部回答"No, does not contain target artworks"，本质是粒度过粗的失败）。
5. **`zoom` 工具签名暴露但 final-gate 既未提示路由也未给样例**：Final Gate 第 7 行写了"Use zoom when a coarse segment is relevant but too long"，但 planner 全程没用它一次，因为 (a) 没有 worked example，(b) `zoom` 自身只切子段、不调 VLM，planner 还要再追加 N 次 `inspect_segment` —— 在 cheap=0 之后被 drop。

### 为什么单一 `scan_segment`（caption-only 等分扫描）不够 —— 设计推翻并改成两阶段

第一版 plan 设计了一个 `scan_segment` 工具：把 segment 按 30s 等分扫描、每窗口跑一次 VLM caption、字符串匹配 target 后聚合成时间线。在 review 中发现四个硬伤：

1. **caption-only，信息源单一**：611-2 question 的关键答案就在 `obs_0002` 的 `asr_text`：`"his innovative, dramatic sculptures for the Borghese, including 'Apollo and Daphne,' 'David,' and 'The Rape of Persephone,' were groundbreaking"`。这是 ASR 已经免费给出的"作品名 + 出现顺序"线索，单纯让 VLM 看 4 帧图像反而比 ASR 更不可靠（VLM 对艺术品命名识别准确率有限）。等分扫描忽视了 ASR/OCR/visual_caption 三个文本源的分层融合。
2. **字符串匹配太脆**：候选实现的 `_scan_match_targets` 用 "full name 或部分 token" 子串匹配，会把 caption 里的 "Apollo / Daphne / marble / myth" 误判成目标命中（"Apollo" 单独出现可能在背景音乐讨论里，与 "Apollo and Daphne" 作品无关）；同时漏掉别名（"David" 大小写敏感、"The rape of Persephone" 的 "of" 被 stopword 过滤）。
3. **30s 非重叠窗口不稳**：四件作品若在 seg_0002 的 480-560s 之间连续出现，事件跨边界后顺序会乱（A 在窗口末尾、B 在下一窗口开头，但 VLM 各自描述时无相对顺序信号）。等分扫描的隐藏假设是"事件均匀分布"，与真实视频剪辑节奏不符。
4. **candidate 与 evidence 混在一起**：`scan_segment` 一旦 VLM caption 提到 target 名字就立刻写 `timeline_rows` 进 `timeline.md`，把"VLM 可能看到了"与"VLM 确认看到了"混为一谈。AnswerAgent 读 timeline 时无法判断证据强度。

**重设计：Layer 0 / Layer 1 / Layer 2 三层流水线**

| Layer | 工具 / 位置 | 信息源 | 是否 evidence |
|-------|-------------|--------|---------------|
| 0（prompt-only） | `_scene_index_snapshot_block` 渲染 `asr mentions @ T` 行 | dual_source `asr_sentences` | 否（routing 提示） |
| 1（locate） | `locate_targets_in_segment` | raw ASR sentences + OCR frames + visual_caption + entities | 否（写入 Navigation Candidates 桶） |
| 2（verify） | `verify_segment_anchors` | VLM @ Layer 1 给出的 anchors（含 anchor.reason 作为上下文） | 是（写入 timeline.md + evidence_table） |

设计回应：
- **回应问题 1**：Layer 0 + Layer 1 完全基于 ASR/OCR/visual 文本（占 80% routing 决策），VLM 只在 Layer 2 验证已收紧到 < 30s 的候选窗口。
- **回应问题 2**：Layer 1 用 alias library + 严格 word-boundary regex；区分四档 match_type（`full_name 0.95 / cleaned_name 0.85 / phrase_alias 0.7 / rare_token 0.5`）；显式拒绝 generic art-history tokens；单 token 仅 rare proper noun（如 "Persephone"）才算 phrase_alias，常见词（"Apollo" / "marble" / "David"）必须出现在多词上下文里。
- **回应问题 3**：Layer 1 把相邻 candidate（同 target 距离 < 15s）合并成 anchor window，并加 ±5s padding。Layer 2 的 VLM prompt 显式要求按"in this window, what appears first / second"输出**窗口内相对秒**，工具再转回绝对时间戳。多 target 共享 anchor 时由 VLM 在窗口内排序。
- **回应问题 4**：Layer 1 仅输出 `candidates` + `anchors_for_vlm` 字段 → 进 Navigation Candidates 桶；Layer 2 输出 `confirmations` + `rejections` + `timeline_rows` 字段 → 仅 `timeline_rows` 进 `timeline.md`。AnswerAgent 只读后者。

回应用户最关心的"用字幕时间戳大致判断去哪段，不用全部扫一遍"：Layer 0 已经把 ASR mentions 渲染进 prompt，planner 不调任何工具就可以选段；Layer 1 是 cheap 文本工具不烧 VLM；Layer 2 只对 Layer 1 给出的少量 anchor 跑 VLM。整条流水线全片至多触发 4-6 次 VLM 调用（vs. 第一版 scan_segment 等分扫描的 10 次 + caption_segment 的 7 次）。

---

## File Structure

| File | Change | Why |
|------|--------|-----|
| `src/visual_coding_agent_harness/agents/iterative_agent.py:35-58, 79-92` | 删 `_CHEAP_TOOLS` / `_EXPENSIVE_TOOLS` / `_VERIFIER_TOOLS` / `_TOOL_CLASSES`，删 `AgentBudget.cheap_tool_budget` / `expensive_tool_budget` / `verifier_tool_budget` / `free_exploration` 字段，删 `_tool_budget_available` + `tool_budget_exhausted` 路径 | Task 0：取消计费机制，只算交互轮次 |
| `src/visual_coding_agent_harness/agents/prompt_stack.py:517-547` | `_budget_snapshot_block` 移除 "Remaining tool budgets: …" 行和 "Cheap navigation tools and expensive VLM tools have separate budgets" 等所有 per-class 文案 | Task 0：prompt 里不再展示工具预算 |
| `src/visual_coding_agent_harness/agents/prompt_stack.py` | `_tool_schema_block(active_skill=, exhausted=)`、`_final_gate_block(route=, active_skill=)`，`skill_catalog_prompt(active_skill=…)` 精简模式 | 让 planner 看到的工具与规则与运行状态对齐 |
| `src/visual_coding_agent_harness/agents/skills/specs.py` | `skill_catalog_prompt` 接受 `focus_skill`；新 `nav_summary_digest_fields()` helper | catalog 精简 + 给格式化层一份允许字段白名单 |
| `src/visual_coding_agent_harness/tools/navigation.py:151-181` | `read_segment_detail` 返回新字段 `nav_digest`（≤240 字符），合成 `targets/visual/asr` 三段摘要 | 让 navigation summary 一行可读 |
| `src/visual_coding_agent_harness/workspace.py:2348-2353` | `_format_navigation_entry` 接读 `nav_digest`；同时支持 `search_segments` / `ground_question` 类似摘要 | planner 第二轮起立刻看见上轮的事实 |
| `src/visual_coding_agent_harness/video_index.py` | `VideoMapSegment` 增加 `asr_sentences: list[dict]` 字段；dual_source builder 把句级 ASR 时间戳 plumb 进来（无句级时按 char offset 估算并写 `limitations`） | Layer 0 / Layer 1 都依赖句级时间戳 |
| `src/visual_coding_agent_harness/agents/prompt_stack.py`（再编辑） | `_scene_index_snapshot_block` + `SceneIndex.summary()` 接 `target_hints`，在每段下方多渲染一行 "asr mentions: …" | Layer 0：planner 不调工具就能选段 |
| `src/visual_coding_agent_harness/tools/navigation.py` | 新增 `locate_targets_in_segment`（cheap 文本 locator） | Layer 1：ASR/OCR/visual 文本匹配 → candidates + anchors |
| `src/visual_coding_agent_harness/tools/aliasing.py` | 新建 alias library + 严格 word-boundary matcher + generic-token 拒绝表 | 修字符串匹配脆弱性（问题 2） |
| `src/visual_coding_agent_harness/tools/inspector.py` | 新增 `verify_segment_anchors`（focused VLM verifier） | Layer 2：仅在 Layer 1 anchors 上跑 VLM |
| `src/visual_coding_agent_harness/workspace.py` | 新桶 `Navigation Candidates`（locate 结果）；`Long-Term Visual Evidence` 桶接收 verify 的 `timeline_rows` | candidates / evidence 分桶（问题 4） |
| `src/visual_coding_agent_harness/agents/iterative_agent.py` | `_temporal_order_route_playbook_hint` 推荐 "locate → verify" 两步（Task 0 已删除 `_EXPENSIVE_TOOLS`，无需归类） | 让 timeline_ordering 路径默认走两阶段 |
| `src/visual_coding_agent_harness/agents/skills/specs.py` | `timeline_ordering@v1` / `mutex_fact_qa@v1` / `grounded_factual_qa@v1` 的 `allowed_actions` 加上 `locate_targets_in_segment` + `verify_segment_anchors` | 不被 route_violation drop |
| `tests/agents/test_prompt_stack_skill_focus.py` | 新 | Pin 精简后的 tool schema / final gate |
| `tests/workspace/test_navigation_summary_digest.py` | 新 | Pin `read_segment_detail` 行内 digest 渲染 |
| `tests/agents/test_scene_index_asr_mentions.py` | 新 | Pin Layer 0：Compact Scene Index 渲染 `asr mentions @ ~Ts` 行 |
| `tests/tools/test_aliasing.py` | 新 | Pin alias library + word-boundary 匹配 + generic-token 拒绝（corner cases：`Apollo` 单独不算 / `marble` 拒绝 / `Persephone` rare token 算 / `David,` 含标点正常 match） |
| `tests/tools/test_locate_targets_in_segment.py` | 新 | Pin Layer 1：ASR/OCR/visual 文本匹配返回 candidates + anchors 合并行为 |
| `tests/tools/test_verify_segment_anchors.py` | 新 | Pin Layer 2：dummy VLM backend 在 anchor 上确认/拒绝、解析窗口内相对时间戳 |
| `tests/agents/test_iterative_temporal_order_anchor_flow.py` | 新 | 端到端：temporal_order 路由 + locate_targets_in_segment → verify_segment_anchors → read_timeline_sorted（Task 0 后无预算扣减需断言） |
| `tests/agents/test_budget_rounds_only.py` | 新 | Pin：`AgentBudget` 三个预算字段已删；`_budget_snapshot_block` 不再渲染 per-class 预算行 |
| `src/visual_coding_agent_harness/agents/skills/specs.py`（再编辑） | 从 `timeline_ordering@v1` / `mutex_fact_qa@v1` / `grounded_factual_qa@v1` 的 `allowed_actions` 中移除 `zoom` / `expand_window` / `read_segment` / `video_ls` / `caption_segments` | Task 9：被 `locate_targets_in_segment` + `verify_segment_anchors` / `read_segment_detail` 替代 |
| `src/visual_coding_agent_harness/agents/iterative_agent.py`（再编辑） | `_repair_skill_route_tool` 把 planner 仍发的 `zoom` / `expand_window` 软重写到 `locate_targets_in_segment`，把 `read_segment` 软重写到 `read_segment_detail`，并写一条 imperative `NormalizationNote` | Task 9：保持 registry 兼容，但运行时强制走新路径 |
| `tests/agents/test_legacy_tool_deprecation.py` | 新 | Pin 被 deprecate 工具不再出现在精简 schema / 自动重写到新工具 |

---

## 旧工具弃用矩阵（Task 9 实施依据）

下表盘点本 plan 实施后**功能被新工具/路径覆盖**的旧工具，以及处理策略。处理原则：
- **registry 兼容性保留**：所有旧工具仍在 `ToolRegistry`，旧测试、旧 CLI、旧 trajectory replay 不破。
- **planner 可见性移除**：被替代的工具不再出现在过滤后的 `Tool Schema` block 中。
- **运行时软重写**：若 planner 仍调旧工具（来自 LLM 记忆/旧 prompt），`_normalize_program` 软重写到新工具并附 `DO NEXT` 指令。

| 旧工具 | 新替代 | 替代理由 | 处理 |
|--------|--------|----------|------|
| `read_segment` | `read_segment_detail` | `read_segment_detail` 是严格超集（返回 visual_caption + asr_summary + target_hits + nav_digest），且 onboarding §1 (line 81-82) 已记录"timeline / grounded QA / mutex QA route 下，planner 调 `read_segment` 会 normalize 成 `read_segment_detail`"——本次把 normalize 扩到所有 option-blind / MCQ 路由 | 从所有 needle / timeline skill 的 `allowed_actions` 中移除；保留 main_idea 的兼容（main_idea 走 global_gist + 索引一览，不需要 target_hits）；planner 调用时自动 normalize |
| `zoom` | `locate_targets_in_segment` → `verify_segment_anchors` | `zoom` 只切子段 metadata，不调 VLM；planner 想拿子段证据还要再追加 `inspect_segment(child_id)`。新流程先用文本 locator 给出 anchor windows，再只对 anchors 跑 focused VLM 验证 | 从三个 needle/timeline skill 的 `allowed_actions` 移除；planner 仍调 `zoom` 时软重写为 `locate_targets_in_segment(segment_id, targets=<inherited>)`，并提示下一步 `verify_segment_anchors` |
| `expand_window` | `target_coverage` + `locate_targets_in_segment` → `verify_segment_anchors` | `expand_window` 的核心用例是"段太短，扩相邻 30s"——但新流程中 `target_coverage` 已给出跨段 top-k，`locate_targets_in_segment` 在候选段内用 ASR/OCR/visual 文本定位 anchors。`expand_window` 的输出本质上还是个新 segment 待处理，路径冗余 | 同 `zoom`：从三个 needle/timeline skill 移除，软重写到同一 `segment_id` 的 `locate_targets_in_segment` |
| `video_ls` | `Compact scene index` 默认 prompt | onboarding §1.3 (line 22) 已记录"`video_ls` 已降级：planner 默认看 `Compact scene index`"。Final Gate 第 1 行也已说"do not call video_ls for short indexed videos" | 从所有 skill 的 `allowed_actions` 移除；planner 调用时直接 drop 并 `NormalizationNote: scene index is already in your prompt` |
| `caption_segments`（批量） | `locate_targets_in_segment` → `verify_segment_anchors`，或单段 `caption_segment` | onboarding §1.4 (line 64) 已记录"planner-visible schema 已隐藏 `caption_segments`"——本 plan 只是把这条约定写进 `allowed_actions` 而不再只靠 schema 隐藏 | 已隐藏，本 plan 显式从所有 needle/timeline skill 的 `allowed_actions` 移除并写测试钉住 |
| `commit_map_proposals` / `update_hypothesis_slot` / `read_hypothesis` / `view_observation` / `grep_evidence` / `query_evidence_table` / `read_timeline_sorted`（除 timeline_ordering 外） | 当前 skill catalog 本就未列入 | 这些工具属于"workspace primitives"，不应该出现在 planner 的工作工具表里 | 通过 Task 1 的 `active_skill` schema 过滤天然被隐藏；不需要单独 normalize（planner 几乎不调） |

**不在本 plan deprecate 的近义工具**（保留并行存在）：

| 工具 | 保留理由 |
|------|----------|
| `inspect_segment` / `caption_segment` / `qa_segment` / `vision_read` | 四者都是"对单段调 VLM"，签名/语义略有差异（`caption_segment` 开放 caption / `qa_segment` 闭合问答 / `vision_read` ask_for 风格 / `inspect_segment` 走 inspector 子 prompt）。Skill-level 选择保留语义区分。它们的"批量 / fanout / 子段排序"职能由 `locate_targets_in_segment` → `verify_segment_anchors` 接管，但**单段聚焦**职能不重叠。三选一的合并另起 plan（见 Out of Scope）。 |
| `global_gist` | main_idea 专属，一次性使用，不与本 plan 改的路由冲突 |
| `verify_ledger_answer` / `summarize_ledger_evidence` | verifier 路径，与本 plan 改动正交 |

---

## Task 0：取消 per-class 工具预算，只算交互轮次（P0，前置）

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py:35-62`（删 `_CHEAP_TOOLS` / `_EXPENSIVE_TOOLS` / `_VERIFIER_TOOLS` / `_TOOL_CLASSES` / `_ONE_SHOT_TOOLS` 中无关条目）
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py:71-92`（`AgentBudget` 移除字段）
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py`（搜 `_tool_budget_available` 全删；搜 `tool_budget_exhausted` 全删；搜 `cheap_tool_budget` / `expensive_tool_budget` / `verifier_tool_budget` 引用全删；搜 `free_exploration` 全删）
- Modify: `src/visual_coding_agent_harness/agents/prompt_stack.py:517-547`（`_budget_snapshot_block` 简化）
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py`（`build_replanning_prompt(... tool_class_counts=...)` 调用处删 `tool_class_counts` 参数）
- Modify: `src/visual_coding_agent_harness/agents/prompt_stack.py:75-116, 119-216`（`build_replanning_prompt` / `compose_replanning_prompt_slots` 签名删 `tool_class_counts`）
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py`（`reflection_memory` 写入路径中删 `segment_pool_exhausted -> Scene index segment pool is empty; pivot to ... call verify_ledger_answer + final` 这类预算相关规则——但 `segment_pool_exhausted` 本身保留，它跟"段池耗尽"语义有关而非预算）
- Test: `tests/agents/test_budget_rounds_only.py`（create）

**Symptom this fixes:**
(a) 611-2 trace Round 17/18/19/20 全部因 `tool_budget_exhausted` drop 工具调用，但段池里还有有用窗口可探索 —— 预算耗尽并不意味着 token / GPU 真的不够，这是个伪信号。
(b) Planner 每轮看到三行预算信息（"cheap=14, expensive=20, verifier=2 / Cheap navigation tools and expensive VLM tools have separate budgets / In free exploration mode, prioritize answer quality"），消耗 token 没有产生有用决策依据——planner 拿到这三行后并不会按预算做工具选择，反而被 `tool_budget_exhausted` 的 normalize note 困惑。
(c) 工具是否"贵"应该让 planner 看 `Final Gate` 的路由指南自己判断（例如先 `locate_targets_in_segment` 再 `verify_segment_anchors`，而不是对整段反复 `caption_segment`），而不是事后被预算拦截。

**保留的硬限制：**
- `max_rounds`（默认 20）—— ReAct 主循环上限
- `max_tool_calls_per_round`（默认 2）—— 单轮 program 长度上限
- `max_repeated_programs`（默认 3）—— 防止 planner 同一 program 重复 N 次的死循环检测
- `reserve_final_round`（默认 True）—— 最后一轮强制 final 或 verifier

**删除的字段（`AgentBudget` 改造前/改造后）：**

```python
# 改造前 (iterative_agent.py:71-92):
@dataclass(frozen=True)
class AgentBudget:
    max_rounds: int = 8
    max_tool_calls_per_round: int = 2
    default_nframes: int = VISUAL_EVIDENCE_NFRAMES
    high_fps_nframes: int = 32
    planner_receives_media: bool = False
    reserve_final_round: bool = True
    cheap_tool_budget: int = 16          # DELETE
    expensive_tool_budget: int = 6        # DELETE
    verifier_tool_budget: int = 2         # DELETE
    answer_probe_rounds_before_final: int = 0
    free_exploration: bool = False        # DELETE
    persist_planner_io: bool = True
    planner_io_max_chars: int = 200_000
    context_budget_tokens: int = 12000
    context_budget_ratios: Mapping[str, float] | None = None
    max_repeated_programs: int = 3
    hard_skill_runtime: bool = False
    reflection_memory_max_items: int = 5
    disable_global_gist_route: bool = False
    rewrite_mcq_for_exploration: bool = False

# 改造后:
@dataclass(frozen=True)
class AgentBudget:
    max_rounds: int = 8
    max_tool_calls_per_round: int = 2
    default_nframes: int = VISUAL_EVIDENCE_NFRAMES
    high_fps_nframes: int = 32
    planner_receives_media: bool = False
    reserve_final_round: bool = True
    answer_probe_rounds_before_final: int = 0
    persist_planner_io: bool = True
    planner_io_max_chars: int = 200_000
    context_budget_tokens: int = 12000
    context_budget_ratios: Mapping[str, float] | None = None
    max_repeated_programs: int = 3
    hard_skill_runtime: bool = False
    reflection_memory_max_items: int = 5
    disable_global_gist_route: bool = False
    rewrite_mcq_for_exploration: bool = False
```

`AgentBudget.free_explore(...)` classmethod 同时删除（既然不再区分预算，所有运行都是 "free explore"）。

---

- [x] **Step 1：写失败测试**

新建 `tests/agents/test_budget_rounds_only.py`：

```python
import pytest

from visual_coding_agent_harness.agents.iterative_agent import AgentBudget


def test_budget_only_exposes_round_limits() -> None:
    b = AgentBudget(max_rounds=20, max_tool_calls_per_round=2)
    # Round-based limits stay
    assert b.max_rounds == 20
    assert b.max_tool_calls_per_round == 2
    assert b.reserve_final_round is True
    # Per-class budgets must be GONE from the dataclass
    field_names = {f.name for f in b.__dataclass_fields__.values()}
    assert "cheap_tool_budget" not in field_names
    assert "expensive_tool_budget" not in field_names
    assert "verifier_tool_budget" not in field_names
    assert "free_exploration" not in field_names


def test_budget_snapshot_block_omits_per_class_budgets() -> None:
    from visual_coding_agent_harness.agents.prompt_stack import _budget_snapshot_block
    body = _budget_snapshot_block(
        round_number=3,
        budget=AgentBudget(max_rounds=20, max_tool_calls_per_round=2),
        tool_class_counts=None,  # ignored / parameter may have been removed entirely
        final_round_reserved=False,
    )
    assert "Round: 3/20" in body
    assert "Request at most 2 new tool call(s)" in body
    assert "Remaining tool budgets" not in body
    assert "cheap=" not in body
    assert "expensive=" not in body
    assert "free exploration mode" not in body


def test_tool_class_module_constants_removed() -> None:
    import visual_coding_agent_harness.agents.iterative_agent as agent_mod
    assert not hasattr(agent_mod, "_CHEAP_TOOLS")
    assert not hasattr(agent_mod, "_EXPENSIVE_TOOLS")
    assert not hasattr(agent_mod, "_VERIFIER_TOOLS")
    assert not hasattr(agent_mod, "_TOOL_CLASSES")
    assert not hasattr(agent_mod, "_tool_budget_available")
```

- [x] **Step 2：跑测试验证失败**

`PYTHONPATH=src pytest tests/agents/test_budget_rounds_only.py -v`
Expected: 全部 FAIL（字段还在 / 函数还在）。

- [x] **Step 3：删 `AgentBudget` 三个预算字段 + `free_exploration` + `free_explore` classmethod**

按上面"改造后"代码块替换 `iterative_agent.py:71-92`。同时删 `iterative_agent.py:94-110` 的 `free_explore` classmethod（搜 `def free_explore`）。

- [x] **Step 4：删工具分类常量**

删除 `iterative_agent.py:35-62` 的 `_CHEAP_TOOLS` / `_EXPENSIVE_TOOLS` / `_VERIFIER_TOOLS` / `_TOOL_CLASSES`。

`_ONE_SHOT_TOOLS` **保留**（它处理 `global_gist` 这类"一次性工具"，与计费机制无关）。

- [x] **Step 5：删 `_tool_budget_available` 检查路径**

在 `iterative_agent.py` 搜 `_tool_budget_available`（应在 `_normalize_program` 内被调用），删函数定义 + 删所有调用点。被它 drop 的步骤现在直接通过 `max_tool_calls_per_round` 限制即可。

同时搜 `tool_budget_exhausted` 字符串，删除对应的 `NormalizationNote` 写入逻辑。

- [x] **Step 6：简化 `_budget_snapshot_block`**

替换 `prompt_stack.py:517-547`：

```python
def _budget_snapshot_block(
    *,
    round_number: int,
    budget: Any,
    tool_class_counts: Mapping[str, int] | None = None,  # ignored, kept for back-compat
    final_round_reserved: bool,
) -> str:
    final_round_line = (
        "\nReserved final round is active: return final now, or call verify_ledger_answer only if essential."
        if final_round_reserved
        else ""
    )
    return (
        f"Round: {round_number}/{getattr(budget, 'max_rounds', '?')}\n"
        f"Request at most {getattr(budget, 'max_tool_calls_per_round', 1)} new tool call(s) this round."
        f"{final_round_line}"
    )
```

`tool_class_counts` 参数保留接收但不使用（避免大量调用点同时改签名）。后续 Task 1-9 的修改可顺手把这个参数从签名里彻底去掉，但本 task 不强求。

- [x] **Step 7：删 reflection_memory 中预算相关规则**

搜 `reflection_memory` 写入处。`segment_pool_exhausted -> ... call verify_ledger_answer + final` 这条规则**保留**（"段池耗尽"是真实的状态而非预算）。但若有 `tool_budget_exhausted -> ...` 规则文本写入，整条删除。

- [x] **Step 8：跑测试 + 全套**

```bash
PYTHONPATH=src pytest tests/agents/test_budget_rounds_only.py -v
PYTHONPATH=src pytest -q
```
Expected: 新测试 PASS。全套测试中**会大量失败**——所有引用 `cheap_tool_budget` / `expensive_tool_budget` / `free_exploration` 的旧测试。处理原则：
- 测试是"在预算耗尽时 drop 工具"——这条契约本身被删了，**整个测试删除**。
- 测试是"在 free_exploration 下不 drop 工具"——同上，**整个测试删除**。
- 测试是"prompt 里出现 cheap=X expensive=Y"——把断言改成"`Round: X/Y` 在 prompt 里 / `cheap=` 不在 prompt 里"。
- 测试构造 `AgentBudget(..., cheap_tool_budget=16, ...)` 这种——删掉这三个 kwarg。
- 测试断言 normalization 出现 `tool_budget_exhausted` reason——整个测试删除或改成断言"不再出现该 reason"。

预计需要修改/删除 8-12 个旧测试。每改一个跑一次 `pytest -q` 验证不是连带破其它行为。

- [x] **Step 9：提交**

```bash
git add src/visual_coding_agent_harness/agents/iterative_agent.py \
        src/visual_coding_agent_harness/agents/prompt_stack.py \
        tests/agents/test_budget_rounds_only.py \
        tests/  # 含被删/被改的旧测试
git commit -m "feat(agent): cancel per-class tool budget, enforce rounds-only limits"
```

---

## Task 1：Tool Schema 按 active_skill 过滤（P0）

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/prompt_stack.py:362-…`（`_tool_schema_block`）
- Modify: `src/visual_coding_agent_harness/agents/prompt_stack.py:119-216`（`compose_replanning_prompt_slots`）
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py`（调用处传 `active_skill`）
- Test: `tests/agents/test_prompt_stack_skill_focus.py`（create）

**Symptom this fixes:** 611-2 trajectory 中 planner 每轮看到 20+ 工具签名，其中 `commit_map_proposals`、`update_hypothesis_slot`、`read_hypothesis`、`query_evidence_table`、`grep_evidence`、`view_observation` 在当前 skill 下完全不会用到，纯粹是 token 浪费 + 干扰项。

- [x] **Step 1：写失败测试**

新建 `tests/agents/test_prompt_stack_skill_focus.py`：

```python
from visual_coding_agent_harness.agents.prompt_stack import _tool_schema_block


def test_tool_schema_filters_to_skill_allowed_actions() -> None:
    rendered = _tool_schema_block(
        option_blind=True,
        active_skill="timeline_ordering@v1",
        exhausted=frozenset(),
    )
    # Allowed for timeline_ordering after Task 7 + Task 9: caption_segment,
    # locate_targets_in_segment, query_context, read_segment_detail,
    # read_timeline_sorted, search_segments, target_coverage,
    # verify_ledger_answer, verify_segment_anchors, vision_read
    # (zoom / expand_window / read_segment removed by Task 9.)
    assert "caption_segment(" in rendered
    assert "read_segment_detail(" in rendered
    # Outside the allowed_actions set — must be hidden
    assert "commit_map_proposals(" not in rendered
    assert "update_hypothesis_slot(" not in rendered
    assert "grep_evidence(" not in rendered


def test_tool_schema_marks_exhausted_tools_inline() -> None:
    rendered = _tool_schema_block(
        option_blind=True,
        active_skill="main_idea@v1",
        exhausted=frozenset({"global_gist"}),
    )
    assert "global_gist(" in rendered
    assert "=exhausted" in rendered


def test_tool_schema_falls_back_to_full_when_no_skill() -> None:
    rendered = _tool_schema_block(option_blind=True, active_skill=None, exhausted=frozenset())
    # Backwards compatible: when planner hasn't selected a skill yet, show everything.
    assert "commit_map_proposals(" in rendered
    assert "read_segment_detail(" in rendered
```

- [x] **Step 2：跑测试验证失败**

`PYTHONPATH=src pytest tests/agents/test_prompt_stack_skill_focus.py -v`
Expected: 3 个 test 全部 FAIL（`_tool_schema_block` 当前签名不带 `active_skill`/`exhausted`）。

- [x] **Step 3：实现 `_tool_schema_block` 新签名**

在 `src/visual_coding_agent_harness/agents/prompt_stack.py:362` 附近找到 `_tool_schema_block`，改为：

```python
def _tool_schema_block(
    *,
    option_blind: bool = False,
    active_skill: str | None = None,
    exhausted: frozenset[str] = frozenset(),
) -> str:
    full_signatures = _TOOL_SCHEMA_SIGNATURES_OPTION_BLIND if option_blind else _TOOL_SCHEMA_SIGNATURES_FULL
    if active_skill:
        from visual_coding_agent_harness.agents.skills.specs import allowed_actions_for_skill
        allowed = allowed_actions_for_skill(active_skill)
        if allowed:
            full_signatures = [
                _maybe_mark_exhausted(sig, exhausted)
                for sig in full_signatures
                if _tool_name_from_signature(sig) in allowed
            ]
        else:
            full_signatures = [_maybe_mark_exhausted(sig, exhausted) for sig in full_signatures]
    else:
        full_signatures = [_maybe_mark_exhausted(sig, exhausted) for sig in full_signatures]
    body = "Available tools:\n" + "\n".join(f"- {sig}" for sig in full_signatures)
    return body


def _tool_name_from_signature(signature: str) -> str:
    return signature.split("(", 1)[0]


def _maybe_mark_exhausted(signature: str, exhausted: frozenset[str]) -> str:
    name = _tool_name_from_signature(signature)
    return f"{signature}  =exhausted" if name in exhausted else signature
```

并在 `agents/skills/specs.py` 暴露：

```python
def allowed_actions_for_skill(skill_id: str) -> frozenset[str]:
    """Return the allowed_actions for a registered skill id, or empty when unknown."""
    spec = _BUILTIN_SPECS.get(skill_id)
    if spec is None:
        # Accept short ids (without @vN) for resilience
        for sid, candidate in _BUILTIN_SPECS.items():
            if sid.split("@", 1)[0] == skill_id:
                spec = candidate
                break
    return frozenset(spec.allowed_actions) if spec else frozenset()
```

- [x] **Step 4：跑测试验证通过**

`PYTHONPATH=src pytest tests/agents/test_prompt_stack_skill_focus.py -v`
Expected: 3 PASS.

- [x] **Step 5：把 `active_skill` 接到 `compose_replanning_prompt_slots`**

在 `prompt_stack.py:119` 的 `compose_replanning_prompt_slots` 签名加 `active_skill: str | None = None`，并在 `tooling_blocks` 内把它转给 `_tool_schema_block`：

```python
PromptBlock(name="tool_schema", title="Tool Schema", body=_tool_schema_block(
    option_blind=option_blind,
    active_skill=active_skill,
    exhausted=exhausted_tools or frozenset(),
)),
```

同步更新 `build_replanning_prompt`（`prompt_stack.py:75`）签名加 `active_skill`。

- [x] **Step 6：在 `IterativeVisualAgent` 主循环传入上一轮选中的 skill**

在 `iterative_agent.py` 找到主循环里调 `build_replanning_prompt(...)` 的地方（搜 `build_replanning_prompt(`），在调用前维护一个 `self._last_selected_skill: str | None`，每轮 planner JSON 解析成功后写入；调用 `build_replanning_prompt` 时把它作为 `active_skill` 传入。第一轮（还没选过）传 `None`，回退到全量 tool schema。

```python
self._last_selected_skill: str | None = None  # init in __init__

# In run() loop, after _parse_replan_action succeeded:
selected_skill = parsed.get("skill") if isinstance(parsed, dict) else None
if isinstance(selected_skill, str) and selected_skill.strip():
    self._last_selected_skill = selected_skill.strip()

# In the next-round build_replanning_prompt call:
build_replanning_prompt(..., active_skill=self._last_selected_skill, ...)
```

- [x] **Step 7：跑全套测试**

`PYTHONPATH=src pytest -q`
Expected: 旧的 368 测试全部 PASS，新增 3 个也 PASS。如有失败，多半在 `tests/test_prompt_stack_*` 旧契约里固化了完整 tool schema —— 加 `active_skill=None` 参数让它走 fallback 分支即可。

- [x] **Step 8：提交**

```bash
git add src/visual_coding_agent_harness/agents/prompt_stack.py \
        src/visual_coding_agent_harness/agents/skills/specs.py \
        src/visual_coding_agent_harness/agents/iterative_agent.py \
        tests/agents/test_prompt_stack_skill_focus.py
git commit -m "feat(prompt): filter tool schema by active skill allowed_actions"
```

---

## Task 2：Final Gate 按 route 折叠（P0）

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/prompt_stack.py:679-716`（`_final_gate_block`）
- Test: `tests/agents/test_prompt_stack_skill_focus.py`（追加）

**Symptom this fixes:** Final Gate 17 行规则里有 9 行（`video_ls` 弃用提示、`caption_segments` 离线提示、`Main-idea answers must compare whole-video coverage`、`option_blind` 4 句、`Prefer segment_id references`、`Do not repeat already inspected segments`、`Reserved final round` 提示）在大部分轮次冗余或与当前 route 无关。611-2 是 `temporal_order`，但每轮也看到 `Main-idea answers must compare whole-video coverage` 这种 main_idea 专属规则。

- [x] **Step 1：在同一测试文件追加测试**

往 `tests/agents/test_prompt_stack_skill_focus.py` 追加：

```python
from visual_coding_agent_harness.agents.prompt_stack import _final_gate_block


def test_final_gate_drops_main_idea_rule_for_temporal_route() -> None:
    body = _final_gate_block(
        final_round_reserved=False,
        option_blind=True,
        route="temporal_order",
    )
    assert "Main-idea answers" not in body
    # Temporal-order specific guidance must show up
    assert "locate_targets_in_segment" in body
    assert "verify_segment_anchors" in body
    assert "Compact scene index" in body or "scene index" in body
    # Must still keep the no-repeat rule
    assert "Do not repeat already inspected segments" in body


def test_final_gate_keeps_main_idea_rule_for_gist_route() -> None:
    body = _final_gate_block(
        final_round_reserved=False,
        option_blind=True,
        route="gist_global",
    )
    assert "Main-idea answers" in body


def test_final_gate_keeps_route_agnostic_safety_rules() -> None:
    body = _final_gate_block(
        final_round_reserved=False,
        option_blind=True,
        route="needle_local",
    )
    # These rules apply to every route:
    assert "Final answers must cite observation ids from the ledger." in body
    assert (
        "Final answers require at least one non-navigation visual observation" in body
    )
```

- [x] **Step 2：跑测试验证失败**

`PYTHONPATH=src pytest tests/agents/test_prompt_stack_skill_focus.py -v`
Expected: 3 个新增 FAIL（`_final_gate_block` 不接 `route`）。

- [x] **Step 3：实现 route-aware Final Gate**

替换 `prompt_stack.py:679` 起的 `_final_gate_block`：

```python
_ROUTE_AGNOSTIC_RULES = (
    "- Use the compact scene index as the default map; do not call video_ls for short indexed videos.",
    "- Use navigation output as a map; then delegate localized visual reading to one VLM tool on one candidate segment.",
    "- Do not spend every round on navigation-only tools; gather visual evidence before finalizing.",
    "- Prefer segment_id references; the harness binds video_path/start_sec/end_sec.",
    "- Do not repeat already inspected segments unless the ledger says the prior observation was unusable.",
    "- Continue when evidence is missing, ambiguous, or too coarse.",
    "- Final answers require at least one evidence-grade visual observation from vision_read, inspect_segment, caption_segment, verify_segment_anchors, or qa_segment; navigation-only evidence and locate candidates are insufficient.",
    "- Use verify_ledger_answer before finalizing when answer support is uncertain.",
    "- Final answers must cite observation ids from the ledger.",
)

_ROUTE_SPECIFIC_RULES: dict[str, tuple[str, ...]] = {
    "gist_global": (
        "- For gist/global questions, use global_gist before local decomposition as a sparse topic hint, not an option vote.",
        "- Main-idea answers must compare whole-video coverage; partial ending-only evidence cannot beat a full rise/stability/fall arc.",
    ),
    "temporal_order": (
        "- For order/sequence questions, use target_coverage or scene-index ASR hints to pick a candidate segment, then call locate_targets_in_segment(segment_id, targets=[...]).",
        "- After locate_targets_in_segment returns anchors_for_vlm, call verify_segment_anchors on those anchors before reading timeline evidence.",
        "- After one verify_segment_anchors observation you can call read_timeline_sorted to read the materialized event order.",
    ),
    "needle_local": (
        "- For needle questions, use target_coverage + read_segment_detail to localize the candidate segment, then locate_targets_in_segment followed by verify_segment_anchors when targets need in-segment anchoring.",
        "- Distinguishing facts should come from one focused VLM observation; do not fan out caption_segment over every segment.",
    ),
}

_OPTION_BLIND_RULES = (
    "- MCQ choices were rewritten into an option-blind exploration task; do not pass option labels or candidate choice text to local tools.",
    "- Use target_coverage for a target-to-segment coverage matrix, then read_segment_detail / locate_targets_in_segment for selected segments.",
    "- Local VLM tools must openly describe visible/narrated segment content as concrete observations.",
    "- The AnswerAgent will compare cited open facts to the original choices later.",
)

_OPTION_LABELED_RULES = (
    "- Multiple-choice answers must use vision_read or inspect_segment on a localized candidate before finalizing; candidate options are only fact-finding hints.",
    "- Local workers must not choose options or emit supported_option; the AnswerAgent maps cited facts to options globally.",
    '- JSON safety: candidate_options in JSON should be option letters only, for example ["A", "B", "C", "D"]; the harness restores full option text.',
    "- Do not copy quoted option text into JSON string values; refer to option letters instead.",
)


def _final_gate_block(
    *,
    final_round_reserved: bool,
    option_blind: bool = False,
    route: str | None = None,
) -> str:
    lines: list[str] = list(_ROUTE_AGNOSTIC_RULES)
    route_rules = _ROUTE_SPECIFIC_RULES.get(route or "", ())
    if route_rules:
        lines.extend(route_rules)
    lines.extend(_OPTION_BLIND_RULES if option_blind else _OPTION_LABELED_RULES)
    if final_round_reserved:
        lines.append("Reserved final round is active: return final now, or call verify_ledger_answer only if essential.")
    return "\n".join(lines) + "\n"
```

- [x] **Step 4：把 `route` 一路传到 `_final_gate_block`**

在 `compose_replanning_prompt_slots`（`prompt_stack.py:119`）顶部已有 `playbook = select_question_playbook(question)`，加：

```python
route = getattr(playbook, "route", None) or getattr(playbook, "question_route", None)
```

如果 `playbook` 没有 `route` 属性，看 `agents/question_policy.py` 里 `QuestionPlaybook` 的字段名（搜 `class QuestionPlaybook`）并对齐。把 `route` 透传到 `_final_gate_block(..., route=route)` 在 `tooling_blocks`。

- [x] **Step 5：跑测试验证通过**

`PYTHONPATH=src pytest tests/agents/test_prompt_stack_skill_focus.py -v`
Expected: 6 PASS（含 Task 1 的 3 个）。

- [x] **Step 6：跑全套**

`PYTHONPATH=src pytest -q`
Expected: 旧 368 测试若有 final-gate 字符串断言失败，按测试用例 route 补 `route=` 参数。

- [x] **Step 7：提交**

```bash
git add src/visual_coding_agent_harness/agents/prompt_stack.py \
        tests/agents/test_prompt_stack_skill_focus.py
git commit -m "feat(prompt): route-aware final gate compaction"
```

---

## Task 3：Skill Catalog 在已选 skill 后只显示焦点条目（P1）

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/skills/specs.py`（`skill_catalog_prompt`）
- Modify: `src/visual_coding_agent_harness/agents/prompt_stack.py:155-166`（传 `focus_skill`）
- Test: `tests/agents/test_prompt_stack_skill_focus.py`（追加）

**Symptom this fixes:** 611-2 中 planner 在每轮都看到全部 4 个 skill 的 markers + allowed_actions（8 行）。一旦 round 1 选定了 `timeline_ordering`，后续 19 轮把其它 3 个 skill 全量 catalog 重复 19 次是纯噪声。

- [ ] **Step 1：在同一测试文件追加测试**

```python
from visual_coding_agent_harness.agents.skills.specs import skill_catalog_prompt


def test_skill_catalog_full_when_no_focus() -> None:
    rendered = skill_catalog_prompt(exhausted_tools=frozenset(), focus_skill=None)
    assert "main_idea@v1" in rendered
    assert "mutex_fact_qa@v1" in rendered
    assert "grounded_factual_qa@v1" in rendered
    assert "timeline_ordering@v1" in rendered


def test_skill_catalog_focuses_on_selected_skill() -> None:
    rendered = skill_catalog_prompt(
        exhausted_tools=frozenset(),
        focus_skill="timeline_ordering@v1",
    )
    assert "timeline_ordering@v1" in rendered
    # Non-selected skills should appear in a one-line "Other skills (one-line):" footer,
    # not their full markers/allowed_actions block.
    assert "main_idea@v1" in rendered  # mention by id only
    assert "markers=main idea" not in rendered  # full marker list hidden
    assert "Other skills" in rendered
```

- [ ] **Step 2：跑测试验证失败**

`PYTHONPATH=src pytest tests/agents/test_prompt_stack_skill_focus.py::test_skill_catalog_focuses_on_selected_skill -v`
Expected: FAIL（当前 `skill_catalog_prompt` 不接 `focus_skill`）。

- [ ] **Step 3：实现 focus 模式**

打开 `src/visual_coding_agent_harness/agents/skills/specs.py`，找到 `skill_catalog_prompt`：

```python
def skill_catalog_prompt(
    *,
    exhausted_tools: frozenset[str] = frozenset(),
    focus_skill: str | None = None,
) -> str:
    lines = ["Available skills:"]
    if focus_skill:
        focused = _BUILTIN_SPECS.get(focus_skill)
        if focused is None:
            for sid, spec in _BUILTIN_SPECS.items():
                if sid.split("@", 1)[0] == focus_skill:
                    focused = spec
                    break
        if focused is not None:
            lines.append(_render_skill_full_line(focused, exhausted_tools=exhausted_tools))
            other_ids = [sid for sid in _BUILTIN_SPECS if sid != focused.skill_id]
            if other_ids:
                lines.append("Other skills (one-line, switch via planner JSON if needed): " + ", ".join(other_ids))
            return "\n".join(lines)
    # Fallback: full catalog
    for spec in _BUILTIN_SPECS.values():
        lines.append(_render_skill_full_line(spec, exhausted_tools=exhausted_tools))
    return "\n".join(lines)
```

提取 `_render_skill_full_line(spec, *, exhausted_tools)` 复用现有渲染逻辑（拷贝当前 `skill_catalog_prompt` 内的 per-spec 拼接代码到该 helper）。

- [ ] **Step 4：在 `prompt_stack.py` 调用处加 `focus_skill`**

在 `compose_replanning_prompt_slots` 的 `skill_catalog` block 处：

```python
body=(
    f"{skill_catalog_prompt(exhausted_tools=exhausted_tools, focus_skill=active_skill)}\n"
    "Select the skill that best matches this case in every planner JSON as `skill`. "
    ...
)
```

- [ ] **Step 5：跑测试**

`PYTHONPATH=src pytest tests/agents/test_prompt_stack_skill_focus.py -v`
Expected: 全部 PASS。

- [ ] **Step 6：跑全套，修复回归**

`PYTHONPATH=src pytest -q`
旧 `tests/test_prompt_stack_and_skill_runtime.py` 里若断言完整 catalog 字符串，把测试改成 `focus_skill=None` 的回退路径或断言关键子串。

- [ ] **Step 7：提交**

```bash
git add src/visual_coding_agent_harness/agents/skills/specs.py \
        src/visual_coding_agent_harness/agents/prompt_stack.py \
        tests/agents/test_prompt_stack_skill_focus.py
git commit -m "feat(prompt): collapse skill catalog to focused skill after selection"
```

---

## Task 4：`read_segment_detail` 回填 `nav_digest`（P0）

**Files:**
- Modify: `src/visual_coding_agent_harness/tools/navigation.py:151-181`
- Test: `tests/tools/test_read_segment_detail_digest.py`（create）

**Symptom this fixes:** 611-2 `obs_0002`-`obs_0017` 全部 16 条 `read_segment_detail` observation 在 navigation summary 渲染为 `obs_XXXX: read_segment_detail`，完全没暴露事实。但 `regions[0].asr_text` 含有目标证据。

- [x] **Step 1：写失败测试**

新建 `tests/tools/test_read_segment_detail_digest.py`：

```python
from visual_coding_agent_harness.tools.navigation import register_navigation_tools
from visual_coding_agent_harness.registry import ToolRegistry
from visual_coding_agent_harness.video_map import VideoMapStore, VideoMap, VideoMapSegment


def _make_store() -> VideoMapStore:
    seg = VideoMapSegment(
        segment_id="seg_0002",
        start_sec=300.0,
        end_sec=600.0,
        low_fps_caption=(
            "A close-up of a marble sculpture, focusing on the detailed hand and "
            "floral elements, with a portrait of a man in historical attire visible "
            "in the background."
        ),
        asr_text=(
            "Bernini's mastery in sculpting realistic textures from marble... "
            "his innovative, dramatic sculptures for the Borghese, including "
            "'Apollo and Daphne,' 'David,' and 'The Rape of Persephone,' were "
            "groundbreaking."
        ),
        ocr_text="",
        entities=["marble sculpture", "portrait of a man"],
        keyframe_paths=[],
        embedding_refs=[],
    )
    return VideoMapStore(VideoMap(video_path="/tmp/v.mp4", duration_sec=1805.0, segments=(seg,)))


def test_read_segment_detail_returns_nav_digest() -> None:
    registry = ToolRegistry()
    register_navigation_tools(registry=registry, video_map_store=_make_store(), workspace=None)
    tool = registry.get("read_segment_detail")
    result = tool.invoke(
        segment_id="seg_0002",
        targets=["Apollo and Daphne", "David", "The Rape of Persephone", "Aeneas, Anchises, and Ascanius fleeing Troy"],
    )
    assert "nav_digest" in result
    digest = result["nav_digest"]
    assert isinstance(digest, str)
    assert len(digest) <= 240
    # 必须含目标命中提示
    assert "3/4" in digest or "targets" in digest.lower()
    # 必须含 ASR 关键词以让 planner 看到事实
    assert "Apollo and Daphne" in digest or "Bernini" in digest
```

注：`register_navigation_tools` 函数名按实际暴露的工厂函数对齐（若名字不同，搜 `def register.*tools` 或 `navigation.py` 顶层导出）。

- [x] **Step 2：跑测试验证失败**

`PYTHONPATH=src pytest tests/tools/test_read_segment_detail_digest.py -v`
Expected: FAIL（无 `nav_digest` 键）。

- [x] **Step 3：实现 `nav_digest` 合成**

在 `src/visual_coding_agent_harness/tools/navigation.py` 修改 `read_segment_detail`（line 151-181），在 return dict 中追加：

```python
def _compose_nav_digest(
    *, segment, target_matches, unmatched_targets, target_hits
) -> str:
    total = len(target_hits) or 0
    matched = len(target_matches)
    hit_summary = (
        f"targets {matched}/{total}: " + ", ".join(str(m.get("target", ""))[:48] for m in target_matches[:3])
        if total
        else "no targets requested"
    )
    visual = (segment.low_fps_caption or "").strip().replace("\n", " ")
    asr = (segment.asr_text or "").strip().replace("\n", " ")
    # Trim long json-wrapped ASR (the dual_source builder wraps as JSON sometimes)
    if asr.startswith("{") and '"summary"' in asr:
        # Best-effort extraction of the summary field
        try:
            import json as _json
            asr_obj = _json.loads(asr)
            asr = str(asr_obj.get("summary", asr))
        except Exception:
            pass
    visual_short = (visual[:90] + "…") if len(visual) > 90 else visual
    asr_short = (asr[:120] + "…") if len(asr) > 120 else asr
    return f"{hit_summary} | visual: {visual_short} | asr: {asr_short}"


# Inside read_segment_detail return dict, add:
nav_digest = _compose_nav_digest(
    segment=segment,
    target_matches=target_matches,
    unmatched_targets=unmatched_targets,
    target_hits=target_hits,
)
return {
    "claim": _segment_detail_claim(segment, target_hits=target_hits),
    "nav_digest": nav_digest,
    ...  # 其余字段保持不变
}
```

- [x] **Step 4：跑测试**

`PYTHONPATH=src pytest tests/tools/test_read_segment_detail_digest.py -v`
Expected: PASS.

- [x] **Step 5：提交**

```bash
git add src/visual_coding_agent_harness/tools/navigation.py \
        tests/tools/test_read_segment_detail_digest.py
git commit -m "feat(navigation): compose nav_digest with target hits + visual/asr summaries"
```

---

## Task 5：Navigation Summary 渲染 `nav_digest`（P0）

**Files:**
- Modify: `src/visual_coding_agent_harness/workspace.py:2348-2353`（`_format_navigation_entry`）
- Modify: `src/visual_coding_agent_harness/workspace.py:1644-…`（`_parse_ledger_entries`，确保 `nav_digest` 被读出来）
- Test: `tests/workspace/test_navigation_summary_digest.py`（create）

**Symptom this fixes:** Workspace 把 observation 持久化到 `ledger.md`，但当前 `_parse_ledger_entries` 只拿 `tool` / `observation_id` / `claim` / `limitations` / `confidence`。Task 4 的 `nav_digest` 必须流到 ledger.md 然后被 parse 回来。

- [x] **Step 1：写失败测试**

新建 `tests/workspace/test_navigation_summary_digest.py`：

```python
from visual_coding_agent_harness.workspace import _format_navigation_entry


def test_navigation_entry_renders_nav_digest_when_present() -> None:
    entry = {
        "observation_id": "obs_0002",
        "tool": "read_segment_detail",
        "nav_digest": (
            "targets 3/4: Apollo and Daphne, David, The Rape of Persephone | "
            "visual: A close-up of a marble sculpture… | "
            "asr: Bernini's mastery in sculpting realistic textures…"
        ),
    }
    rendered = _format_navigation_entry(entry)
    assert "obs_0002" in rendered
    assert "read_segment_detail" in rendered
    assert "3/4" in rendered
    assert "Bernini" in rendered


def test_navigation_entry_falls_back_to_bare_when_no_digest() -> None:
    entry = {"observation_id": "obs_0099", "tool": "read_segment"}
    rendered = _format_navigation_entry(entry)
    assert rendered == "- obs_0099: read_segment"


def test_navigation_entry_target_coverage_kept_compact() -> None:
    entry = {
        "observation_id": "obs_0001",
        "tool": "target_coverage",
        "claim": "Target coverage matrix: T1 X: seg_0002, seg_0001",
    }
    rendered = _format_navigation_entry(entry)
    assert "target_coverage" in rendered
    assert "Target coverage matrix" in rendered
```

- [x] **Step 2：跑测试验证失败**

`PYTHONPATH=src pytest tests/workspace/test_navigation_summary_digest.py -v`
Expected: 第 1 个 FAIL（当前格式化器没读 `nav_digest`），其余 PASS。

- [x] **Step 3：扩展 `_format_navigation_entry`**

在 `src/visual_coding_agent_harness/workspace.py:2348` 替换：

```python
def _format_navigation_entry(entry: Mapping[str, Any]) -> str:
    tool_name = str(entry.get("tool", "unknown"))
    digest = str(entry.get("nav_digest", "")).strip()
    if digest:
        digest_clipped = _compact_text(digest, limit=300)
        return f"- {entry['observation_id']}: {tool_name} | {digest_clipped}"
    if tool_name == "target_coverage":
        claim = _compact_text(str(entry.get("claim", "")), limit=720)
        return f"- {entry['observation_id']}: {tool_name} | claim: {claim or '(empty)'}"
    return f"- {entry['observation_id']}: {tool_name}"
```

- [x] **Step 4：让 `_parse_ledger_entries` 解出 `nav_digest`**

在 `workspace.py:1644` 找 `_parse_ledger_entries`。当前它解 `observation_id` / `tool` / `claim` / `confidence` / `limitations`。`nav_digest` 需要在 observation 写入 `ledger.md` 时落地，然后被读回。找到写入 ledger 的位置（搜 `ledger.md` 写入或 `write_ledger`），在每条 entry 追加形如 `nav_digest: …` 的行；并在 `_parse_ledger_entries` 的字段循环里加上：

```python
elif line.startswith("nav_digest:"):
    entry["nav_digest"] = line.split(":", 1)[1].strip()
```

如果 ledger.md 已经把整个 observation dict 序列化（例如 JSON 行），那只需要确保 `nav_digest` 在 dict 里就会自动出现。检查 `_record_observation_to_ledger` 或类似入口。

- [x] **Step 5：补一个最小集成测试 — observation 落 ledger 后能解出 digest**

继续在 `tests/workspace/test_navigation_summary_digest.py` 追加：

```python
import json
from pathlib import Path

from visual_coding_agent_harness.workspace import EvidenceWorkspace
from visual_coding_agent_harness.protocol import ToolResult, ToolRequest


def test_workspace_persists_nav_digest_into_ledger(tmp_path: Path) -> None:
    ws = EvidenceWorkspace(root=tmp_path / "run", question_id="q1")
    ws.record_observation(
        request=ToolRequest(tool="read_segment_detail", args={"segment_id": "seg_0002"}),
        result=ToolResult(
            claim="seg_0002 detail covers 300.0-600.0s.",
            confidence=1.0,
            regions=[],
            raw_output={
                "nav_digest": "targets 3/4: Apollo and Daphne | visual: marble sculpture | asr: Bernini's mastery"
            },
        ),
    )
    text = ws.compact_ledger_text()
    assert "Apollo and Daphne" in text
    assert "obs_" in text
    # The Navigation Summary block must show the digest, not just the tool name
    assert "read_segment_detail | targets" in text
```

`record_observation` 等 API 名按代码实际签名对齐（在 `workspace.py` 搜 `def record_observation` 或 `def write_observation`）。

- [x] **Step 6：跑测试 + 全套**

```bash
PYTHONPATH=src pytest tests/workspace/test_navigation_summary_digest.py -v
PYTHONPATH=src pytest -q
```
Expected: 新增测试 PASS，全套 PASS。

- [x] **Step 7：提交**

```bash
git add src/visual_coding_agent_harness/workspace.py \
        tests/workspace/test_navigation_summary_digest.py
git commit -m "feat(workspace): surface read_segment_detail nav_digest into Navigation Summary"
```

---

## Task 5b：Compact Scene Index 渲染 "asr mentions @ ~Ts" 行（Layer 0，P0）

**Files:**
- Modify: `src/visual_coding_agent_harness/video_index.py`（`SceneIndex.summary()` 接 `target_hints`）
- Modify: `src/visual_coding_agent_harness/agents/prompt_stack.py:502-514`（`_scene_index_snapshot_block` 传 `target_hints`）
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py`（主循环把 option-blind rewrite 后的 `target_entities` 作为 `target_hints` 透传）
- Test: `tests/agents/test_scene_index_asr_mentions.py`（create）

**Symptom this fixes:** 611-2 trace 中 planner 看到的 Compact Scene Index 只有每段 visual map 一句话，看不见 ASR 里"David / Apollo / Persephone / Aeneas 出现在哪段哪秒"。即使 ASR 已经明摆着提了，planner 仍要调 `read_segment_detail` 把每段拉一遍才能"嗅"到关键词。Layer 0 直接把这个信号摆到 prompt 里：planner 不调任何工具就能知道"4 个 target 都集中在 seg_0002 的 380-475s 之间"，第一轮直接发 `locate_targets_in_segment(seg_0002, targets)`。

**渲染目标格式（每段下方多一行）：**

```
seg_0001 [0.0-300.0s] map: pencil sketch + Bernini biography (early years)
  asr mentions: "Aeneas" @ ~125s, "Pope Paul V" @ ~95s
seg_0002 [300.0-600.0s] map: marble sculpture, Borghese commissions
  asr mentions: "David" @ ~380s, "Persephone" @ ~440s, "Apollo and Daphne" @ ~475s, "Aeneas" @ ~315s
seg_0003 [600.0-900.0s] map: Baroque theatricality, plays
  asr mentions: —
```

未传 `target_hints` 时，行为完全等同当前实现（只输出 map 行），向后兼容旧 prompt 测试。

- [x] **Step 1：写失败测试**

新建 `tests/agents/test_scene_index_asr_mentions.py`：

```python
from visual_coding_agent_harness.video_index import SceneIndex, VideoSegment


def _make_index() -> SceneIndex:
    return SceneIndex(
        segments=(
            VideoSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=300.0,
                map_summary="pencil sketch + Bernini biography",
                asr_sentences=(
                    {"start_sec": 90.0, "end_sec": 102.0, "text": "Pope Paul V recognized him as a genius."},
                    {"start_sec": 120.0, "end_sec": 135.0, "text": "He sculpted Aeneas carrying Anchises early on."},
                ),
            ),
            VideoSegment(
                segment_id="seg_0002",
                start_sec=300.0,
                end_sec=600.0,
                map_summary="marble sculpture, Borghese commissions",
                asr_sentences=(
                    {"start_sec": 310.0, "end_sec": 322.0, "text": "Aeneas Anchises and Ascanius fleeing Troy comes first."},
                    {"start_sec": 372.0, "end_sec": 388.0, "text": "Next, his David takes center stage."},
                    {"start_sec": 432.0, "end_sec": 448.0, "text": "Then the Rape of Persephone appears."},
                    {"start_sec": 466.0, "end_sec": 482.0, "text": "Finally Apollo and Daphne closes the sequence."},
                ),
            ),
        ),
    )


def test_summary_without_target_hints_is_backwards_compatible() -> None:
    idx = _make_index()
    rendered = idx.summary(max_segments=8)
    assert "asr mentions:" not in rendered
    assert "seg_0001" in rendered
    assert "seg_0002" in rendered


def test_summary_with_target_hints_emits_asr_mentions_line() -> None:
    idx = _make_index()
    rendered = idx.summary(
        max_segments=8,
        target_hints=("Aeneas, Anchises, and Ascanius fleeing Troy", "Apollo and Daphne", "David", "The rape of Persephone"),
    )
    # seg_0001 contains Aeneas only
    assert "seg_0001" in rendered
    seg1_block, _, rest = rendered.partition("seg_0002")
    assert "asr mentions:" in seg1_block
    assert "Aeneas" in seg1_block
    assert "Apollo and Daphne" not in seg1_block
    # seg_0002 contains all four
    seg2_block, _, _ = rest.partition("\nseg_")
    assert "asr mentions:" in seg2_block
    assert "Aeneas" in seg2_block
    assert "David" in seg2_block
    assert "Persephone" in seg2_block
    assert "Apollo and Daphne" in seg2_block
    # Timestamps must use ~Ts shorthand and reflect sentence midpoints
    assert "@ ~380s" in seg2_block or "@ ~381s" in seg2_block


def test_summary_emits_dash_when_no_mentions() -> None:
    idx = _make_index()
    rendered = idx.summary(
        max_segments=8,
        target_hints=("non-existent target name that nothing matches",),
    )
    # Every segment line gets the "asr mentions:" prefix, even with —
    assert rendered.count("asr mentions: —") == 2
```

- [x] **Step 2：跑测试验证失败**

`PYTHONPATH=src pytest tests/agents/test_scene_index_asr_mentions.py -v`
Expected: 3 个 FAIL（`asr_sentences` 字段不存在 / `SceneIndex.summary()` 不接 `target_hints`）。第 1 个可能 PASS（向后兼容）。

- [x] **Step 3：扩 `VideoSegment` schema**

在 `src/visual_coding_agent_harness/video_index.py` 找到 `class VideoSegment`（应该是 `@dataclass(frozen=True)`），追加字段：

```python
@dataclass(frozen=True)
class VideoSegment:
    segment_id: str
    start_sec: float
    end_sec: float
    map_summary: str = ""
    # ... 其余既有字段 ...
    asr_sentences: tuple[dict, ...] = ()  # Each dict: {"start_sec": float, "end_sec": float, "text": str}
```

注意：`tuple` not `list`，以匹配 `frozen=True`。也可以直接用 `Sequence[Mapping[str, object]]` 注解但内部存 `tuple`。

dual_source builder（搜 `dual_source_scene_index_v` 或在 `videomme` 相关 builder 里）补充把原始 ASR 句级时间戳填进 `asr_sentences`。如果原始 ASR file 只给段级摘要，按 segment span / 句数比例估算每句 timestamp，并在 segment 的 `limitations` 或 builder log 写"asr_sentences synthesized from char-offset (no native sentence timing)"。

scene-index cache schema 从 `dual_source_scene_index_v3` bump 到 `v4`（参照 onboarding §1.3 line 56 的 bump 模式），防止旧 cache 缺新字段。

- [x] **Step 4：扩 `SceneIndex.summary()` 与 helper**

在 `video_index.py` 找 `SceneIndex.summary(...)`，签名加 `target_hints: Sequence[str] = ()`。实现：

```python
def summary(self, *, max_segments: int = 64, target_hints: Sequence[str] = ()) -> str:
    lines: list[str] = []
    for seg in list(self.segments)[:max_segments]:
        lines.append(_render_segment_map_line(seg))  # 现有渲染
        if target_hints:
            lines.append(_render_asr_mentions_line(seg, target_hints))
    return "\n".join(lines)


def _render_asr_mentions_line(seg: "VideoSegment", target_hints: Sequence[str]) -> str:
    from visual_coding_agent_harness.tools.aliasing import match_target_in_text  # Task 6b 提供
    mentions: list[tuple[str, float]] = []
    for sent in seg.asr_sentences:
        text = str(sent.get("text", ""))
        for target in target_hints:
            hit = match_target_in_text(text=text, target=str(target))
            if hit and hit["match_type"] in {"full_name", "cleaned_name", "phrase_alias"}:
                mid = (float(sent.get("start_sec", 0.0)) + float(sent.get("end_sec", 0.0))) / 2.0
                mentions.append((str(target), mid))
                break  # one mention per sentence is enough
    if not mentions:
        return "  asr mentions: —"
    # Sort by timestamp, dedupe by (target, rounded-timestamp)
    mentions.sort(key=lambda p: p[1])
    rendered = ", ".join(f'"{name}" @ ~{ts:.0f}s' for name, ts in mentions[:6])  # cap at 6 to bound prompt length
    if len(mentions) > 6:
        rendered += f", … (+{len(mentions) - 6} more)"
    return f"  asr mentions: {rendered}"
```

`match_target_in_text` 在 Task 6b 创建 `tools/aliasing.py` 后会存在；本 task 写到此函数即可，运行时若 Task 6b 未先行，跑测试会 ImportError —— **Task 5b 必须在 Task 6b 之后实施**，否则 Step 5 测试无法通过。或者在 Task 5b 临时 inline 一个最小 word-boundary matcher，Task 6b 再统一替换。Plan 中我们按依赖顺序：**先 Task 6b 再 Task 5b**（File Structure 表与 Self-Review 已反映此依赖）。

- [x] **Step 5：让 `_scene_index_snapshot_block` 透传 `target_hints`**

在 `prompt_stack.py:502` 的 `_scene_index_snapshot_block` 签名加 `target_hints: Sequence[str] = ()`，传给 `scene_index.summary(target_hints=target_hints)`。再往上：`compose_replanning_prompt_slots` 与 `build_replanning_prompt` 签名加 `target_hints` 参数。

`IterativeVisualAgent` 主循环里：option-blind rewrite 后已经把候选 targets 写在 `workspace.hypothesis` 的 `target_entities` 里（onboarding §1.4 line 67）。在调 `build_replanning_prompt(...)` 前从那里拿：

```python
target_hints = tuple(self._workspace.get_target_entities() or ())
build_replanning_prompt(..., target_hints=target_hints, ...)
```

`get_target_entities()` API 名以代码实际定义为准（搜 `target_entities` 即可）。

- [x] **Step 6：跑测试**

`PYTHONPATH=src pytest tests/agents/test_scene_index_asr_mentions.py -v`
Expected: 3 PASS（前提：Task 6b 已先完成，提供 `match_target_in_text`）。

- [x] **Step 7：跑全套**

`PYTHONPATH=src pytest -q`
旧 scene-index summary 测试若断言行数 / 字符串完整等式，可能因新增 "asr mentions: —" 行而失败 —— 这些测试要么传 `target_hints=()` 走向后兼容路径，要么改成只断言关键子串。

- [x] **Step 8：提交**

```bash
git add src/visual_coding_agent_harness/video_index.py \
        src/visual_coding_agent_harness/agents/prompt_stack.py \
        src/visual_coding_agent_harness/agents/iterative_agent.py \
        tests/agents/test_scene_index_asr_mentions.py
git commit -m "feat(prompt): render Layer 0 asr mentions @ timestamp under each scene-index segment"
```

---

## Task 6a：alias library + 严格 word-boundary matcher（P0，Layer 1/2 前置）

**Files:**
- Create: `src/visual_coding_agent_harness/tools/aliasing.py`
- Test: `tests/tools/test_aliasing.py`（create）

**Symptom this fixes:** 第一版 `scan_segment` 设计用"target 是 caption 的子串"做匹配，会把 "Apollo" / "marble" 单独出现误判为目标命中。Layer 1 / Layer 2 / Layer 0（Task 5b）都依赖一个可信的 matcher——本 task 集中实现一次，三处共用。

**对外契约：**

```python
# tools/aliasing.py
def generate_target_aliases(target: str) -> list[str]: ...

def match_target_in_text(*, text: str, target: str, aliases: Sequence[str] | None = None) -> Mapping[str, object] | None:
    """Return {"match_type": str, "confidence": float, "matched_alias": str, "span": (start, end)}
    or None when no match. match_type ∈ {"full_name", "cleaned_name", "phrase_alias", "rare_token"}.
    Uses word-boundary regex; rejects generic art-history tokens; single-token alias only matches
    when the token is a rare proper noun.
    """
```

**关键规则：**
- `GENERIC_REJECT = {"marble", "sculpture", "statue", "painting", "art", "myth", "baroque", "renaissance", "portrait", "image", "scene", "video"}` —— 这些 token 即使在 target 名字里出现，也不会被提取为 single-token alias。
- `RARE_PROPER_NOUNS` heuristic：默认认为多音节、低频的 token（"Persephone", "Ascanius", "Anchises", "Bernini"）可以作为 single-token alias；常见 / 多义 token（"Apollo", "David", "Daphne"）必须出现在多词上下文里（match_type ∈ {full_name, cleaned_name, phrase_alias}）才算 hit。判定方式：内置一个 hand-curated allowlist + 长度 ≥ 8 字符 fallback。
- 标点容忍："David, Bernini" 含逗号，正则用 `\b{target}\b` 而非 `\b{target}\s+\b`。
- 大小写不敏感：全部 `lower()` 后比较。

- [x] **Step 1：写失败测试（含 corner cases）**

新建 `tests/tools/test_aliasing.py`：

```python
import pytest
from visual_coding_agent_harness.tools.aliasing import generate_target_aliases, match_target_in_text


# ---------- alias generation ----------

def test_alias_generation_for_multi_word_target() -> None:
    aliases = generate_target_aliases("Apollo and Daphne")
    assert "apollo and daphne" in aliases
    # Cleaned form ("and" stripped) is a valid alias
    assert any("apollo" in a and "daphne" in a and " and " not in a for a in aliases)
    # Single rare-proper-noun "Daphne" — Daphne is allowed (mythological, distinctive enough at len>=6)
    # Apollo single alone is NOT (too common / multiple Apollos in art history)
    assert "apollo" not in aliases or aliases.index("apollo") > aliases.index("apollo and daphne")


def test_alias_generation_rejects_generic_art_tokens() -> None:
    aliases = generate_target_aliases("The marble sculpture of David")
    # No single-token "marble" or "sculpture" alias should be emitted
    assert "marble" not in aliases
    assert "sculpture" not in aliases


def test_alias_generation_handles_comma_separated_names() -> None:
    aliases = generate_target_aliases("Aeneas, Anchises, and Ascanius fleeing Troy")
    # Comma-cleaned full form
    assert any("aeneas anchises" in a for a in aliases)
    # Rare proper nouns kept as single-token aliases
    assert "ascanius" in aliases
    assert "anchises" in aliases
    # "Troy" / "fleeing" should not promote to single-token alias on their own
    assert "troy" not in aliases or "ascanius" in aliases  # OK if both included, but Troy alone is generic


# ---------- matching ----------

def test_match_full_name_returns_strongest_hit() -> None:
    hit = match_target_in_text(text="Then the statue of Apollo and Daphne appears.", target="Apollo and Daphne")
    assert hit is not None
    assert hit["match_type"] == "full_name"
    assert hit["confidence"] >= 0.9


def test_match_rejects_common_token_appearing_alone() -> None:
    # "Apollo" alone in a passage about space program must NOT match "Apollo and Daphne"
    hit = match_target_in_text(text="Apollo 11 landed on the Moon.", target="Apollo and Daphne")
    assert hit is None or hit["match_type"] == "rare_token"
    # And rare_token confidence is low — caller should not promote to evidence
    if hit:
        assert hit["confidence"] < 0.6


def test_match_accepts_rare_proper_noun_alone() -> None:
    # Persephone alone is rare enough to suggest "The rape of Persephone"
    hit = match_target_in_text(text="Pluto carries Persephone away.", target="The rape of Persephone")
    assert hit is not None
    assert hit["match_type"] in {"phrase_alias", "rare_token"}


def test_match_rejects_generic_art_token_even_if_in_target_name() -> None:
    hit = match_target_in_text(text="A close-up of a marble background.", target="The marble sculpture of David")
    # "marble" alone must NOT match — it's in the GENERIC_REJECT set
    assert hit is None


def test_match_word_boundary_avoids_partial_substring() -> None:
    # "Davidson" should NOT match target "David" (subword)
    hit = match_target_in_text(text="Davidson published in 1990.", target="David")
    assert hit is None


def test_match_handles_punctuation_in_text() -> None:
    # Onscreen text often: "David, Bernini, 1623"
    hit = match_target_in_text(text="Onscreen: David, Bernini, 1623", target="David")
    # David here is in a name-list context, can match as rare_token (case-by-case; David is borderline)
    # Acceptable: either rare_token hit OR None — but if hit, must be ≤ rare_token confidence.
    if hit:
        assert hit["confidence"] <= 0.6


def test_match_case_insensitive() -> None:
    hit = match_target_in_text(text="THE RAPE OF PERSEPHONE", target="The rape of Persephone")
    assert hit is not None
    assert hit["match_type"] == "full_name"


def test_match_cleaned_form() -> None:
    # Target as written in MCQ:  "Aeneas, Anchises, and Ascanius fleeing Troy"
    # Text says: "Aeneas Anchises and Ascanius fleeing Troy" (no comma)
    hit = match_target_in_text(text="Aeneas Anchises and Ascanius fleeing Troy is shown.", target="Aeneas, Anchises, and Ascanius fleeing Troy")
    assert hit is not None
    assert hit["match_type"] in {"full_name", "cleaned_name"}
```

- [x] **Step 2：跑测试验证失败**

`PYTHONPATH=src pytest tests/tools/test_aliasing.py -v`
Expected: 全部 FAIL（模块不存在）。

- [x] **Step 3：实现 `tools/aliasing.py`**

新建文件，按上述契约 + 关键规则实现。建议结构：

```python
# tools/aliasing.py
from __future__ import annotations
import re
from collections.abc import Mapping, Sequence

_GENERIC_REJECT = frozenset({
    "marble", "sculpture", "statue", "painting", "art", "artwork",
    "myth", "mythology", "baroque", "renaissance", "portrait", "image",
    "scene", "video", "frame", "camera", "shot", "close", "up", "wide",
    "the", "of", "a", "an", "and", "or", "for", "in", "to", "from", "by", "with",
    "is", "are", "was", "were", "this", "that", "these", "those",
})

# Conservative allowlist of single tokens that strongly identify a known target.
# Add/curate as the dataset grows; default uses length >= 8 as fallback for novel rare nouns.
_RARE_SINGLE_TOKEN_ALLOWLIST = frozenset({
    "persephone", "ascanius", "anchises", "aeneas", "bernini",
    "borghese", "barberini", "caravaggio", "donatello",
})


def _is_rare_proper_noun(token: str) -> bool:
    t = token.lower()
    if t in _GENERIC_REJECT:
        return False
    if t in _RARE_SINGLE_TOKEN_ALLOWLIST:
        return True
    # Fallback: long uncommon tokens are likely rare proper nouns.
    return len(t) >= 8


def generate_target_aliases(target: str) -> list[str]:
    raw = target.strip()
    if not raw:
        return []
    low = raw.lower()
    aliases: list[str] = [low]
    cleaned = re.sub(r"[,]", " ", low)
    cleaned = re.sub(r"\s+and\s+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned != low:
        aliases.append(cleaned)
    tokens = [t for t in cleaned.split() if t not in _GENERIC_REJECT and len(t) > 2]
    # Add adjacent bigrams (multi-word phrase alias)
    for i in range(len(tokens) - 1):
        aliases.append(f"{tokens[i]} {tokens[i + 1]}")
    # Add single-token aliases only for rare proper nouns
    for tok in tokens:
        if _is_rare_proper_noun(tok):
            aliases.append(tok)
    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for a in aliases:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _word_boundary_search(text_low: str, needle: str) -> tuple[int, int] | None:
    pattern = rf"\b{re.escape(needle)}\b"
    m = re.search(pattern, text_low)
    if m is None:
        return None
    return m.span()


def match_target_in_text(
    *, text: str, target: str, aliases: Sequence[str] | None = None
) -> Mapping[str, object] | None:
    if not text or not target:
        return None
    text_low = text.lower()
    target_low = target.lower()
    # 1. full_name (0.95)
    span = _word_boundary_search(text_low, target_low)
    if span is not None:
        return {"match_type": "full_name", "confidence": 0.95, "matched_alias": target_low, "span": span}
    # 2. cleaned_name (0.85)
    cleaned = re.sub(r"\s+and\s+", " ", re.sub(r"[,]", " ", target_low))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned != target_low:
        span = _word_boundary_search(text_low, cleaned)
        if span is not None:
            return {"match_type": "cleaned_name", "confidence": 0.85, "matched_alias": cleaned, "span": span}
    # 3. phrase_alias / rare_token via alias list
    if aliases is None:
        aliases = generate_target_aliases(target)
    best: Mapping[str, object] | None = None
    for alias in aliases:
        if alias in {target_low, cleaned}:
            continue
        span = _word_boundary_search(text_low, alias)
        if span is None:
            continue
        if " " in alias:
            cand = {"match_type": "phrase_alias", "confidence": 0.7, "matched_alias": alias, "span": span}
        else:
            if not _is_rare_proper_noun(alias):
                continue  # single-token but not rare — reject
            cand = {"match_type": "rare_token", "confidence": 0.5, "matched_alias": alias, "span": span}
        if best is None or float(cand["confidence"]) > float(best["confidence"]):
            best = cand
    return best
```

- [x] **Step 4：跑测试**

`PYTHONPATH=src pytest tests/tools/test_aliasing.py -v`
Expected: 全部 PASS。如有 corner case 失败，按测试预期收紧规则（**优先保正确率而不是召回率**——Layer 1 的漏匹配可以靠 Layer 2 VLM 补回来，误匹配则直接污染 evidence_table）。

- [x] **Step 5：提交**

```bash
git add src/visual_coding_agent_harness/tools/aliasing.py \
        tests/tools/test_aliasing.py
git commit -m "feat(tools): add alias library with strict word-boundary + generic-token rejection"
```

---

## Task 6b：`locate_targets_in_segment`（Layer 1，cheap 文本 locator）（P0）

**Files:**
- Modify: `src/visual_coding_agent_harness/tools/navigation.py`（追加 `locate_targets_in_segment`）
- Modify: `src/visual_coding_agent_harness/workspace.py`（新桶 `Navigation Candidates`）
- Test: `tests/tools/test_locate_targets_in_segment.py`（create）

**Symptom this fixes:** 611-2 question 需要在 seg_0002 内定位四件作品的出现顺序。`read_segment_detail` 只返回段级摘要，无段内时间戳。`caption_segment` 又只对整段一句 caption。Layer 1 在 cheap 文本层把 ASR 句级时间戳 / OCR 帧时间戳 / visual_caption（段中点）三源融合，输出 candidates + 合并后的 anchor windows，供 Layer 2 验证。

**API 设计：**

```python
locate_targets_in_segment(
    segment_id: str,
    targets: Sequence[str],
    sources: Sequence[str] = ("asr", "ocr", "visual"),
    anchor_window_sec: float = 25.0,
    anchor_padding_sec: float = 5.0,
    merge_distance_sec: float = 15.0,
    min_confidence_for_anchor: float = 0.7,
) -> Mapping[str, object]
```

返回：

```python
{
    "claim": "Located 6 candidates / 4 anchors in seg_0002 [300-600s]: David @380s ASR strong, Persephone @440s ASR strong, Apollo and Daphne @475s ASR strong, Aeneas @315s ASR strong.",
    "confidence": 0.9,
    "input_artifacts": [video_path],
    "candidates": [
        {"target": "David", "source": "asr", "timestamp_sec": 380.5, "start_sec": 372.0, "end_sec": 388.0,
         "snippet": "Next, his David takes center stage.", "match_type": "full_name", "confidence": 0.95},
        ...
    ],
    "anchors_for_vlm": [
        {"start_sec": 370.0, "end_sec": 415.0, "targets": ["David"],
         "reason": "ASR @380.5s (full_name)"},
        {"start_sec": 430.0, "end_sec": 460.0, "targets": ["The rape of Persephone"], "reason": "..."},
        {"start_sec": 465.0, "end_sec": 490.0, "targets": ["Apollo and Daphne"], "reason": "..."},
        {"start_sec": 305.0, "end_sec": 330.0, "targets": ["Aeneas, Anchises, and Ascanius fleeing Troy"], "reason": "..."},
    ],
    "regions": [],  # 兼容 EvidenceWorkspace observation 结构
    "ledger_bucket": "navigation_candidates",  # 新桶标识
    "limitations": "Text-only locator (ASR/OCR/visual_caption); does NOT confirm visual presence. Call verify_segment_anchors next on these anchors to obtain evidence-grade observations.",
    "recommended_next_tools": [
        {"tool": "verify_segment_anchors",
         "args": {"segment_id": "seg_0002", "anchors": "<inline above>"},
         "reason": "Confirm candidates with focused VLM calls in each anchor window."}
    ],
}
```

**Side effect：写入 workspace 的 `Navigation Candidates` 桶**（新桶，与 `Long-Term Visual Evidence` 完全分离；AnswerAgent 不读此桶）。`compact_ledger_text` 渲染时新增一节 "## Navigation Candidates"。

- [x] **Step 1：写失败测试**

新建 `tests/tools/test_locate_targets_in_segment.py`：

```python
from pathlib import Path

from visual_coding_agent_harness.registry import ToolRegistry
from visual_coding_agent_harness.tools.navigation import register_navigation_tools
from visual_coding_agent_harness.video_map import VideoMapStore, VideoMap, VideoMapSegment


def _make_store_with_asr_sentences() -> VideoMapStore:
    seg = VideoMapSegment(
        segment_id="seg_0002",
        start_sec=300.0,
        end_sec=600.0,
        low_fps_caption="A close-up of marble sculptures with onscreen text.",
        asr_text="(JSON summary placeholder)",
        ocr_text="",
        entities=["marble sculpture", "portrait of a man"],
        keyframe_paths=[],
        embedding_refs=[],
        asr_sentences=(
            {"start_sec": 310.0, "end_sec": 322.0, "text": "Aeneas Anchises and Ascanius fleeing Troy comes first."},
            {"start_sec": 372.0, "end_sec": 388.0, "text": "Next, his David takes center stage."},
            {"start_sec": 432.0, "end_sec": 448.0, "text": "Then the Rape of Persephone appears."},
            {"start_sec": 466.0, "end_sec": 482.0, "text": "Finally Apollo and Daphne closes the sequence."},
        ),
        ocr_frames=(
            {"timestamp_sec": 405.0, "text": "David — Bernini, 1623"},
        ),
    )
    return VideoMapStore(VideoMap(video_path="/tmp/v.mp4", duration_sec=1805.0, segments=(seg,)))


def test_locate_emits_candidates_and_merged_anchors() -> None:
    registry = ToolRegistry()
    register_navigation_tools(registry=registry, video_map_store=_make_store_with_asr_sentences(), workspace=None)
    tool = registry.get("locate_targets_in_segment")
    result = tool.invoke(
        segment_id="seg_0002",
        targets=[
            "Aeneas, Anchises, and Ascanius fleeing Troy",
            "Apollo and Daphne",
            "David",
            "The rape of Persephone",
        ],
    )

    candidates = list(result["candidates"])
    targets_hit = {c["target"] for c in candidates}
    assert "Aeneas, Anchises, and Ascanius fleeing Troy" in targets_hit
    assert "Apollo and Daphne" in targets_hit
    assert "David" in targets_hit
    assert "The rape of Persephone" in targets_hit

    # David appears in both ASR (@380s) and OCR (@405s); locator should keep both as separate candidates
    david_hits = [c for c in candidates if c["target"] == "David"]
    sources = {c["source"] for c in david_hits}
    assert sources == {"asr", "ocr"}

    # Anchors merge nearby candidates: David ASR@380 and OCR@405 are 25s apart → merge_distance_sec defaults to 15s,
    # so they end up in TWO anchors (one centered on 380, one on 405). Acceptable. Both contain David target.
    anchors = list(result["anchors_for_vlm"])
    david_anchors = [a for a in anchors if "David" in a["targets"]]
    assert len(david_anchors) >= 1  # at least one David anchor

    # Anchors must be sorted and clamped within segment bounds
    timestamps = [a["start_sec"] for a in anchors]
    assert timestamps == sorted(timestamps)
    for a in anchors:
        assert 300.0 <= a["start_sec"] <= a["end_sec"] <= 600.0

    assert result["ledger_bucket"] == "navigation_candidates"
    assert "verify_segment_anchors" in result["limitations"] or "verify_segment_anchors" in " ".join(
        rt["tool"] for rt in result.get("recommended_next_tools", [])
    )


def test_locate_rejects_generic_token_matches() -> None:
    # When no real target is present, locate must not invent candidates from generic tokens.
    registry = ToolRegistry()
    register_navigation_tools(registry=registry, video_map_store=_make_store_with_asr_sentences(), workspace=None)
    tool = registry.get("locate_targets_in_segment")
    result = tool.invoke(
        segment_id="seg_0002",
        targets=["A nondescript marble sculpture"],
    )
    # All-generic target should yield zero candidates — generic tokens are rejected.
    candidates = list(result["candidates"])
    assert len(candidates) == 0
    # Empty anchors_for_vlm is acceptable; claim must say so.
    assert "0 candidates" in result["claim"] or "no candidates" in result["claim"].lower()


def test_locate_writes_to_navigation_candidates_bucket(tmp_path: Path) -> None:
    from visual_coding_agent_harness.workspace import EvidenceWorkspace
    ws = EvidenceWorkspace(root=tmp_path / "run", question_id="q1")
    registry = ToolRegistry()
    register_navigation_tools(registry=registry, video_map_store=_make_store_with_asr_sentences(), workspace=ws)
    tool = registry.get("locate_targets_in_segment")
    tool.invoke(segment_id="seg_0002", targets=["Apollo and Daphne"])
    text = ws.compact_ledger_text()
    assert "## Navigation Candidates" in text
    assert "Apollo and Daphne" in text
    # Must NOT appear in Long-Term Visual Evidence
    visual_section = text.split("## Long-Term Visual Evidence", 1)[1].split("##", 1)[0]
    assert "Apollo and Daphne" not in visual_section
```

- [x] **Step 2：跑测试验证失败**

`PYTHONPATH=src pytest tests/tools/test_locate_targets_in_segment.py -v`
Expected: 全部 FAIL（工具不存在 / `Navigation Candidates` 桶不存在 / `ocr_frames` 字段不存在）。

- [x] **Step 3：扩展 `VideoMapSegment` schema**

Task 5b 已把 `asr_sentences` 加上；本 task 同步加 `ocr_frames: tuple[dict, ...] = ()`（每条 `{"timestamp_sec": float, "text": str}`）。dual_source builder 若 OCR 原本只存段级 `ocr_text`，按帧采样时间均匀分配并写 limitations。

scene-index cache schema 再 bump（若 Task 5b 已 bump 到 v4，本 task 留在 v4 即可——同一轮重设计，避免下游用户连续 invalidate）。

- [x] **Step 4：实现 `locate_targets_in_segment`**

在 `src/visual_coding_agent_harness/tools/navigation.py` 文件底部追加（在 `register_navigation_tools` 工厂内）：

```python
from visual_coding_agent_harness.tools.aliasing import generate_target_aliases, match_target_in_text


@tool(
    name="locate_targets_in_segment",
    description=(
        "Layer 1 text locator: scan ASR/OCR/visual_caption inside one segment for target mentions; "
        "return candidates with timestamps and merged anchor windows for follow-up VLM verification."
    ),
)
def locate_targets_in_segment(
    segment_id: str,
    targets: Sequence[str],
    sources: Sequence[str] = ("asr", "ocr", "visual"),
    anchor_window_sec: float = 25.0,
    anchor_padding_sec: float = 5.0,
    merge_distance_sec: float = 15.0,
    min_confidence_for_anchor: float = 0.7,
) -> Mapping[str, object]:
    segment = video_map_store.current.get(segment_id)
    seg_start, seg_end = float(segment.start_sec), float(segment.end_sec)
    candidates: list[dict] = []

    for raw_target in targets:
        target = str(raw_target).strip()
        if not target:
            continue
        aliases = generate_target_aliases(target)

        # ---- ASR sentence-level ----
        if "asr" in sources:
            for sent in segment.asr_sentences or ():
                hit = match_target_in_text(text=str(sent.get("text", "")), target=target, aliases=aliases)
                if hit is None:
                    continue
                s_start = float(sent.get("start_sec", seg_start))
                s_end = float(sent.get("end_sec", s_start))
                candidates.append({
                    "target": target, "source": "asr",
                    "timestamp_sec": (s_start + s_end) / 2.0,
                    "start_sec": s_start, "end_sec": s_end,
                    "snippet": str(sent.get("text", ""))[:200],
                    "match_type": hit["match_type"], "confidence": float(hit["confidence"]),
                })

        # ---- OCR frame-level ----
        if "ocr" in sources:
            for frame in segment.ocr_frames or ():
                hit = match_target_in_text(text=str(frame.get("text", "")), target=target, aliases=aliases)
                if hit is None:
                    continue
                ts = float(frame.get("timestamp_sec", (seg_start + seg_end) / 2.0))
                candidates.append({
                    "target": target, "source": "ocr",
                    "timestamp_sec": ts, "start_sec": ts, "end_sec": ts,
                    "snippet": str(frame.get("text", ""))[:200],
                    "match_type": hit["match_type"],
                    "confidence": min(float(hit["confidence"]) + 0.05, 1.0),  # OCR is grounded onscreen text — slight bump
                })

        # ---- visual_caption (segment-level, midpoint timestamp) ----
        if "visual" in sources and segment.low_fps_caption:
            hit = match_target_in_text(text=str(segment.low_fps_caption), target=target, aliases=aliases)
            if hit is not None:
                mid = (seg_start + seg_end) / 2.0
                candidates.append({
                    "target": target, "source": "visual",
                    "timestamp_sec": mid, "start_sec": seg_start, "end_sec": seg_end,
                    "snippet": str(segment.low_fps_caption)[:200],
                    "match_type": hit["match_type"],
                    "confidence": max(float(hit["confidence"]) - 0.1, 0.1),  # segment-level: less precise
                })

    # ---- Merge into anchors ----
    strong_candidates = [c for c in candidates if c["confidence"] >= min_confidence_for_anchor]
    strong_candidates.sort(key=lambda c: c["timestamp_sec"])

    anchors: list[dict] = []
    for cand in strong_candidates:
        ts = cand["timestamp_sec"]
        # Try to extend most recent anchor if same target & within merge distance.
        merged = False
        if anchors:
            last = anchors[-1]
            same_target = cand["target"] in last["targets"]
            close = abs(ts - last["_anchor_center"]) <= merge_distance_sec
            if same_target and close:
                last["start_sec"] = min(last["start_sec"], max(seg_start, ts - anchor_window_sec / 2 - anchor_padding_sec))
                last["end_sec"] = max(last["end_sec"], min(seg_end, ts + anchor_window_sec / 2 + anchor_padding_sec))
                last["reason"] += f"; {cand['source']}@{ts:.1f}s ({cand['match_type']})"
                merged = True
        if not merged:
            anchors.append({
                "start_sec": max(seg_start, ts - anchor_window_sec / 2 - anchor_padding_sec),
                "end_sec": min(seg_end, ts + anchor_window_sec / 2 + anchor_padding_sec),
                "targets": [cand["target"]],
                "reason": f"{cand['source']}@{ts:.1f}s ({cand['match_type']})",
                "_anchor_center": ts,
            })

    # Drop the private _anchor_center key from final output
    for a in anchors:
        a.pop("_anchor_center", None)

    # ---- Compose claim ----
    if not candidates:
        claim = f"locate_targets_in_segment({segment_id}, {seg_start:.0f}-{seg_end:.0f}s) found 0 candidates for {len(targets)} target(s)."
    else:
        per_target = {}
        for c in candidates:
            per_target.setdefault(c["target"], []).append(c)
        bits = []
        for tgt, hits in per_target.items():
            best = max(hits, key=lambda h: h["confidence"])
            bits.append(f"{tgt} @{best['timestamp_sec']:.0f}s {best['source']} {best['match_type']}")
        claim = (
            f"locate_targets_in_segment({segment_id}, {seg_start:.0f}-{seg_end:.0f}s) "
            f"located {len(candidates)} candidate(s) / {len(anchors)} anchor(s): "
            + "; ".join(bits)
        )

    # Workspace side-effect: write into Navigation Candidates bucket (NOT timeline.md).
    if workspace is not None:
        workspace.record_navigation_candidate(
            segment_id=segment_id,
            candidates=candidates,
            anchors=anchors,
            source_tool="locate_targets_in_segment",
        )

    return {
        "claim": claim,
        "confidence": 0.9,
        "input_artifacts": [video_map_store.current.video_path],
        "candidates": candidates,
        "anchors_for_vlm": anchors,
        "regions": [],
        "ledger_bucket": "navigation_candidates",
        "limitations": (
            "Text-only locator (ASR/OCR/visual_caption); does NOT confirm visual presence. "
            "Call verify_segment_anchors next on these anchors to obtain evidence-grade observations."
        ),
        "recommended_next_tools": [
            {"tool": "verify_segment_anchors",
             "args": {"segment_id": segment_id, "anchors": anchors},
             "reason": "Confirm candidates with focused VLM calls in each anchor window."}
        ],
    }
```

注册：在 `register_navigation_tools` 末尾加 `registry.register(locate_targets_in_segment)`。

- [x] **Step 5：实现 workspace 的 `Navigation Candidates` 桶**

在 `workspace.py` 加 `record_navigation_candidate` 方法（持久化到 `navigation_candidates.jsonl`）；在 `compact_ledger_text` 增加渲染节：

```python
sections.extend(["", "## Navigation Candidates"])
nav_cand_path = self.root / "navigation_candidates.jsonl"
if nav_cand_path.exists():
    rows = self._read_jsonl_dicts("navigation_candidates.jsonl")[-3:]  # last 3 locate calls
    for row in rows:
        seg = row.get("segment_id", "?")
        cands = row.get("candidates", [])
        targets = sorted({str(c.get("target")) for c in cands})
        sections.append(f"- {seg}: {len(cands)} candidate(s) for {len(targets)} target(s) — {', '.join(targets[:4])}")
        for c in cands[:6]:
            sections.append(
                f"    - {c.get('target')} @{float(c.get('timestamp_sec', 0)):.0f}s "
                f"{c.get('source')} {c.get('match_type')} (conf={float(c.get('confidence', 0)):.2f})"
            )
else:
    sections.append("(none)")
```

桶必须放在 `## Navigation Summary` 之后、`## Short-Term Working Buffer` 之前。**关键：此桶不进 `evidence_table` / `ANSWER_EVIDENCE_TOOLS`，AnswerAgent 看不到。**

- [x] **Step 6：跑测试**

`PYTHONPATH=src pytest tests/tools/test_locate_targets_in_segment.py -v`
Expected: 3 PASS。

- [x] **Step 7：跑全套**

`PYTHONPATH=src pytest -q`

- [x] **Step 8：提交**

```bash
git add src/visual_coding_agent_harness/tools/navigation.py \
        src/visual_coding_agent_harness/workspace.py \
        src/visual_coding_agent_harness/video_index.py \
        tests/tools/test_locate_targets_in_segment.py
git commit -m "feat(tools): Layer 1 locate_targets_in_segment with ASR/OCR/visual text matching"
```

---

## Task 6c：`verify_segment_anchors`（Layer 2，focused VLM verifier）（P0）

**Files:**
- Modify: `src/visual_coding_agent_harness/tools/inspector.py`（追加 `verify_segment_anchors`）
- Test: `tests/tools/test_verify_segment_anchors.py`（create）

**Symptom this fixes:** Layer 1 给出 candidates 后，必须有一个 evidence-grade 工具去 VLM 验证；第一版 `scan_segment` 等分扫描在这里被替换——`verify_segment_anchors` 只对 Layer 1 的少量 anchors 跑 VLM，确认后写入 `timeline.md` + `evidence_table`。

**API 设计：**

```python
verify_segment_anchors(
    video_path: str,
    segment_id: str,
    anchors: Sequence[Mapping[str, object]],  # 来自 locate_targets_in_segment 的 anchors_for_vlm
    nframes_per_anchor: int = 8,
    max_pixels: int = 75600,
) -> Mapping[str, object]
```

返回：

```python
{
    "claim": "verify_segment_anchors(seg_0002, 4 anchors): confirmed 4 / rejected 0. Order: Aeneas → David → Persephone → Apollo and Daphne.",
    "confidence": 0.85,
    "input_artifacts": [video_path],
    "confirmations": [
        {"target": "David", "anchor_start_sec": 370, "anchor_end_sec": 415,
         "absolute_timestamp_sec": 397.5, "verdict": "confirmed",
         "evidence": "Marble David statue centered, onscreen text 'David, Bernini, 1623'",
         "confidence": 0.85},
        ...
    ],
    "rejections": [{...}],
    "timeline_rows": [
        {"timestamp": 315.0, "target": "Aeneas, ...", "evidence": "...", "source": "vlm_verified@anchor[305-330]"},
        ...
    ],
    "observed_order": ["Aeneas, ...", "David", "The rape of Persephone", "Apollo and Daphne"],
    "regions": [...],  # 每 anchor 一条
    "limitations": "Per-anchor 8-frame VLM call; brief target appearances within an anchor may be merged. Within-anchor relative timestamps inferred from VLM response.",
}
```

**Side effect:** 仅 `confirmations` / `timeline_rows` 通过 `workspace.append_timeline_row` 写入 `timeline.md` 与 `evidence_table`；rejections 只写 trace（供 debug，不影响 AnswerAgent）。

**VLM prompt 模板（关键）：**

```python
def _compose_verify_prompt(*, targets: Sequence[str], anchor_reason: str) -> str:
    target_list = "\n".join(f"- {t}" for t in targets)
    return (
        "You are verifying a list of candidate target artworks in a short video window.\n\n"
        f"Candidates (one or more may appear; some may NOT appear — reject those):\n{target_list}\n\n"
        f"Context hint that triggered this verification: {anchor_reason}\n\n"
        "For each candidate, answer in this format on its own line:\n"
        '  CANDIDATE: "<target name>" | VERDICT: confirmed|rejected|uncertain | '
        "OFFSET_IN_WINDOW_SEC: <integer> | EVIDENCE: <one-sentence visual description, including any onscreen text>\n\n"
        "Then on a final line write:\n"
        "  WITHIN_WINDOW_ORDER: <target1> -> <target2> -> ... (only confirmed targets, by appearance order)\n\n"
        "Do not include other prose. Confirm only when you can cite a concrete visual feature or onscreen text."
    )
```

工具内的 `_parse_verify_response` 用正则解析每行 `CANDIDATE: "..." | VERDICT: ... | OFFSET_IN_WINDOW_SEC: ... | EVIDENCE: ...` 与最后一行 `WITHIN_WINDOW_ORDER`。VERDICT ∈ {"confirmed","rejected","uncertain"}；只有 confirmed 进 timeline_rows，uncertain 进 rejections 但带标注。

- [x] **Step 1：写失败测试**

新建 `tests/tools/test_verify_segment_anchors.py`：

```python
from pathlib import Path

from visual_coding_agent_harness.registry import ToolRegistry


class DummyVerifyBackend:
    """Scripted response per (anchor_start_sec, anchor_end_sec)."""

    def __init__(self) -> None:
        self._scripts = {
            (305.0, 330.0): (
                'CANDIDATE: "Aeneas, Anchises, and Ascanius fleeing Troy" | VERDICT: confirmed | '
                'OFFSET_IN_WINDOW_SEC: 10 | EVIDENCE: Marble group of Aeneas carrying Anchises with Ascanius beside, onscreen text "Aeneas".\n'
                'WITHIN_WINDOW_ORDER: Aeneas, Anchises, and Ascanius fleeing Troy'
            ),
            (370.0, 415.0): (
                'CANDIDATE: "David" | VERDICT: confirmed | OFFSET_IN_WINDOW_SEC: 12 | '
                'EVIDENCE: Marble David statue centered, onscreen text "David, Bernini, 1623".\n'
                'WITHIN_WINDOW_ORDER: David'
            ),
            (430.0, 460.0): (
                'CANDIDATE: "The rape of Persephone" | VERDICT: confirmed | OFFSET_IN_WINDOW_SEC: 8 | '
                'EVIDENCE: Pluto gripping Persephone, marble surface detail of fingers indented into thigh.\n'
                'WITHIN_WINDOW_ORDER: The rape of Persephone'
            ),
            (465.0, 490.0): (
                'CANDIDATE: "Apollo and Daphne" | VERDICT: confirmed | OFFSET_IN_WINDOW_SEC: 9 | '
                'EVIDENCE: Daphne mid-transformation with laurel leaves on fingers, Apollo reaching.\n'
                'CANDIDATE: "David" | VERDICT: rejected | OFFSET_IN_WINDOW_SEC: 0 | '
                'EVIDENCE: David is not visible in this window.\n'
                'WITHIN_WINDOW_ORDER: Apollo and Daphne'
            ),
        }

    def generate(self, request) -> object:
        from visual_coding_agent_harness.backends.base import BackendResponse
        meta = getattr(request, "metadata", {}) or {}
        key = (float(meta.get("start_sec", 0)), float(meta.get("end_sec", 0)))
        text = self._scripts.get(key, 'CANDIDATE: "?" | VERDICT: rejected | OFFSET_IN_WINDOW_SEC: 0 | EVIDENCE: nothing.\nWITHIN_WINDOW_ORDER:')
        return BackendResponse(text=text, raw={})


def test_verify_emits_observed_order_and_timeline_rows() -> None:
    from visual_coding_agent_harness.tools.inspector import register_inspector_tools

    registry = ToolRegistry()
    register_inspector_tools(registry=registry, backend=DummyVerifyBackend(), workspace=None)
    tool = registry.get("verify_segment_anchors")
    anchors = [
        {"start_sec": 305.0, "end_sec": 330.0, "targets": ["Aeneas, Anchises, and Ascanius fleeing Troy"], "reason": "asr@315"},
        {"start_sec": 370.0, "end_sec": 415.0, "targets": ["David"], "reason": "asr@380; ocr@405"},
        {"start_sec": 430.0, "end_sec": 460.0, "targets": ["The rape of Persephone"], "reason": "asr@440"},
        {"start_sec": 465.0, "end_sec": 490.0, "targets": ["Apollo and Daphne", "David"], "reason": "asr@475"},
    ]
    result = tool.invoke(
        video_path="/tmp/fake.mp4",
        segment_id="seg_0002",
        anchors=anchors,
        nframes_per_anchor=4,
        max_pixels=75600,
    )

    confirmations = list(result["confirmations"])
    targets_confirmed = [c["target"] for c in confirmations]
    assert "David" in targets_confirmed
    assert "Apollo and Daphne" in targets_confirmed

    # David in the LAST anchor (465-490) is rejected; only the 370-415 confirmation counts
    david_confirmations = [c for c in confirmations if c["target"] == "David"]
    assert len(david_confirmations) == 1
    assert 370 <= david_confirmations[0]["absolute_timestamp_sec"] <= 415

    # Rejection bucket records the negative
    rejections = list(result["rejections"])
    assert any(r["target"] == "David" and 465 <= r["anchor_start_sec"] <= 490 for r in rejections)

    # Observed order from absolute timestamps
    observed = list(result["observed_order"])
    assert observed == [
        "Aeneas, Anchises, and Ascanius fleeing Troy",
        "David",
        "The rape of Persephone",
        "Apollo and Daphne",
    ]

    # timeline_rows sorted by absolute timestamp
    ts = [row["timestamp"] for row in result["timeline_rows"]]
    assert ts == sorted(ts)
    assert all(row["source"].startswith("vlm_verified@") for row in result["timeline_rows"])


def test_verify_skips_unparseable_anchor_response_without_writing_timeline(tmp_path: Path) -> None:
    from visual_coding_agent_harness.tools.inspector import register_inspector_tools
    from visual_coding_agent_harness.workspace import EvidenceWorkspace

    class GarbageBackend:
        def generate(self, request):
            from visual_coding_agent_harness.backends.base import BackendResponse
            return BackendResponse(text="lol I don't know", raw={})

    ws = EvidenceWorkspace(root=tmp_path / "run", question_id="q1")
    registry = ToolRegistry()
    register_inspector_tools(registry=registry, backend=GarbageBackend(), workspace=ws)
    tool = registry.get("verify_segment_anchors")
    result = tool.invoke(
        video_path="/tmp/x.mp4",
        segment_id="seg_0002",
        anchors=[{"start_sec": 305.0, "end_sec": 330.0, "targets": ["Aeneas"], "reason": "asr"}],
    )
    assert result["confirmations"] == []
    assert "parse_failed" in str(result["rejections"][0]).lower() or result["rejections"][0]["verdict"] == "uncertain"
    timeline_path = ws.root / "timeline.md"
    if timeline_path.exists():
        assert "Aeneas" not in timeline_path.read_text(encoding="utf-8")
```

- [x] **Step 2：跑测试验证失败**

`PYTHONPATH=src pytest tests/tools/test_verify_segment_anchors.py -v`
Expected: 2 FAIL（工具不存在）。

- [x] **Step 3：实现 `verify_segment_anchors`**

在 `src/visual_coding_agent_harness/tools/inspector.py` 追加（在 `register_inspector_tools` 工厂里），按上述 prompt + parse + workspace 写入逻辑实现。

`_parse_verify_response` 用 `re.compile(r'CANDIDATE:\s*"([^"]+)"\s*\|\s*VERDICT:\s*(\w+)\s*\|\s*OFFSET_IN_WINDOW_SEC:\s*(-?\d+)\s*\|\s*EVIDENCE:\s*(.+)')` 逐行解析；`WITHIN_WINDOW_ORDER` 用单独正则。如果整段响应无任何 CANDIDATE 行，把该 anchor 写入 rejections 列表，rejection.verdict = "parse_failed"。

`observed_order` 计算：把所有 confirmations 按 `absolute_timestamp_sec` 排序，去重保留首次出现。

- [x] **Step 4：跑测试**

`PYTHONPATH=src pytest tests/tools/test_verify_segment_anchors.py -v`
Expected: 2 PASS。

- [x] **Step 5：让 `verify_segment_anchors` 进 `ANSWER_EVIDENCE_TOOLS`**

在 `workspace.py:149` 附近的 `VISUAL_EVIDENCE_TOOLS` 加 `"verify_segment_anchors"`（这样 confirmations 进 evidence_table）。**不要**把 `locate_targets_in_segment` 加进 `VISUAL_EVIDENCE_TOOLS` —— 它在 navigation_candidates 桶。

- [x] **Step 6：跑全套**

`PYTHONPATH=src pytest -q`

- [x] **Step 7：提交**

```bash
git add src/visual_coding_agent_harness/tools/inspector.py \
        src/visual_coding_agent_harness/workspace.py \
        tests/tools/test_verify_segment_anchors.py
git commit -m "feat(tools): Layer 2 verify_segment_anchors with focused VLM and structured parsing"
```

---

## Task 6（占位 — 已被 Task 6a/6b/6c 替代）

> **Note：** 第一版 plan 的 Task 6 (`scan_segment` 等分扫描) 已根据 review 反馈推翻，拆成 Task 6a (alias library) + Task 6b (locate_targets_in_segment, Layer 1) + Task 6c (verify_segment_anchors, Layer 2)。如果你正在按 Task 编号顺序执行，**跳过本节，按 6a → 6b → 6c 顺序做**。

第一版动机记录（保留供 archeology）：原 `scan_segment` 试图用单一工具完成"切窗口 + 调 VLM + 聚合 timeline"，但 caption-only + 子串匹配 + 等分窗口 + candidate/evidence 混合的四个缺陷使其不适合作为生产路径。详见 Background §"为什么单一 `scan_segment`（caption-only 等分扫描）不够"。

<!-- 旧 Task 6 (scan_segment) body removed; see Task 6a/6b/6c above. -->
```

---

## Task 7：把 `locate_targets_in_segment` 与 `verify_segment_anchors` 写进 skill 的 allowed_actions（P0）

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/skills/specs.py`（`_BUILTIN_SPECS`）
- Test: `tests/agents/test_skill_specs_allow_locate_verify.py`（create）

**Symptom this fixes:** 没加白名单的话，planner 调 `locate_targets_in_segment` 或 `verify_segment_anchors` 会被 `_normalize_program` 的 `route_violation` drop 掉。Layer 1 / Layer 2 必须 **同时**进白名单 —— 单独允许 locate 但不允许 verify，会让 planner 卡在拿到 candidates 后无法验证。

- [x] **Step 1：写失败测试**

新建 `tests/agents/test_skill_specs_allow_locate_verify.py`：

```python
from visual_coding_agent_harness.agents.skills.specs import allowed_actions_for_skill


def test_timeline_ordering_allows_locate_and_verify() -> None:
    actions = allowed_actions_for_skill("timeline_ordering@v1")
    assert "locate_targets_in_segment" in actions
    assert "verify_segment_anchors" in actions


def test_mutex_fact_qa_allows_locate_and_verify() -> None:
    actions = allowed_actions_for_skill("mutex_fact_qa@v1")
    assert "locate_targets_in_segment" in actions
    assert "verify_segment_anchors" in actions


def test_grounded_factual_qa_allows_locate_and_verify() -> None:
    actions = allowed_actions_for_skill("grounded_factual_qa@v1")
    assert "locate_targets_in_segment" in actions
    assert "verify_segment_anchors" in actions


def test_main_idea_does_not_allow_locate_verify() -> None:
    # main_idea routes via global_gist + whole-video coverage; locate/verify are unnecessary.
    actions = allowed_actions_for_skill("main_idea@v1")
    assert "locate_targets_in_segment" not in actions
    assert "verify_segment_anchors" not in actions
```

- [x] **Step 2：跑测试验证失败**

`PYTHONPATH=src pytest tests/agents/test_skill_specs_allow_locate_verify.py -v`
Expected: 3 FAIL (前 3 个) + 1 PASS（main_idea 本来就没有）。

- [x] **Step 3：把两个工具加入三个 skill 的 `allowed_actions`**

在 `src/visual_coding_agent_harness/agents/skills/specs.py` 找 `timeline_ordering@v1` / `mutex_fact_qa@v1` / `grounded_factual_qa@v1` 的 SkillSpec 定义，把 `"locate_targets_in_segment"`、`"verify_segment_anchors"` 加进 `allowed_actions=(...)`。**不**加进 `main_idea@v1`（main_idea 走 `global_gist` 全片采样，不适合细粒度子段验证）。

- [x] **Step 4：跑测试**

`PYTHONPATH=src pytest tests/agents/test_skill_specs_allow_locate_verify.py -v`
Expected: 4 PASS。

- [x] **Step 5：跑全套**

`PYTHONPATH=src pytest -q`
Expected: 全部 PASS（如 prompt_stack 快照里出现 `allowed_actions=…`，需要同步更新被快照断言的字符串）。

- [x] **Step 6：提交**

```bash
git add src/visual_coding_agent_harness/agents/skills/specs.py \
        tests/agents/test_skill_specs_allow_locate_verify.py
git commit -m "feat(skills): allow locate_targets_in_segment + verify_segment_anchors in needle/timeline skills"
```

---

## Task 8：端到端验证 — 611-2 trajectory replay（P1）

**Files:**
- Create: `tests/agents/test_iterative_temporal_order_anchor_flow.py`
- 可选: `docs/tool-flow-debug/2026-06-08-611-2-after-fix.md`（再跑一次的总结，留给后续 handoff）

**Symptom this fixes:** 验证：(a) timeline_ordering 路由下，prompt 中 `Final Gate` 提到 `locate_targets_in_segment → verify_segment_anchors`，`Tool Schema` 不再包含 `commit_map_proposals`；(b) planner 跑一次 `locate_targets_in_segment` 后，下一轮 prompt 的 `Navigation Candidates` 看到 anchors 摘要而不是只 `obs: locate_targets_in_segment`；(c) planner 再调用 `verify_segment_anchors` 后，`Long-Term Visual Evidence` / timeline 能看到 evidence-grade order；(d) cheap 预算耗尽不会让循环失控。

- [ ] **Step 1：写端到端测试**

新建 `tests/agents/test_iterative_temporal_order_anchor_flow.py`，组装一个最小 `IterativeVisualAgent`，用 dummy planner backend（Round 1 输出 `locate_targets_in_segment`，Round 2 根据上一轮 anchors 输出 `verify_segment_anchors`，Round 3 final）和 dummy VL backend（DummyVLBackend 重用 Task 6c 测试里的）：

```python
from pathlib import Path
import pytest

from visual_coding_agent_harness.agents.iterative_agent import IterativeVisualAgent, AgentBudget


class _ScriptedPlannerBackend:
    """Returns locate -> verify -> final."""
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request) -> object:
        from visual_coding_agent_harness.backends.base import BackendResponse
        self.calls += 1
        if self.calls == 1:
            text = (
                '{"status": "continue", "skill": "timeline_ordering@v1", '
                '"rationale": "Use the text locator on the target-rich segment.", '
                '"program": [{"tool": "locate_targets_in_segment", "args": {'
                '"segment_id": "seg_0002", '
                '"targets": ["Aeneas, Anchises, and Ascanius fleeing Troy", "Apollo and Daphne", "David", "The rape of Persephone"]}}]}'
            )
        elif self.calls == 2:
            text = (
                '{"status": "continue", "skill": "timeline_ordering@v1", '
                '"rationale": "Verify the locator anchors before relying on them as evidence.", '
                '"program": [{"tool": "verify_segment_anchors", "args": {'
                '"segment_id": "seg_0002", '
                '"anchors": [{"anchor_id": "a1", "start_sec": 380.0, "end_sec": 475.0, '
                '"targets": ["Aeneas, Anchises, and Ascanius fleeing Troy", "Apollo and Daphne", "David", "The rape of Persephone"], '
                '"reason": "ASR mentions all target artworks in order."}]}}]}'
            )
        else:
            text = (
                '{"status": "final", "skill": "timeline_ordering@v1", '
                '"answer": "D. Aeneas → David → Persephone → Apollo and Daphne", '
                '"citations": ["obs_0002"], "confidence": 0.7}'
            )
        return BackendResponse(text=text, raw={})


@pytest.mark.integration
def test_temporal_order_anchor_flow(tmp_path: Path) -> None:
    # Build IterativeVisualAgent with the dummy planner + dummy VL backend
    # plus a minimal SceneIndex containing seg_0001..seg_0007 like 611-2.
    # Then assert:
    # 1. Round 1 prompt contains "locate_targets_in_segment" and "verify_segment_anchors" in Tool Schema.
    # 2. Round 1 prompt's Final Gate mentions locate -> verify guidance.
    # 3. Round 1 prompt's Tool Schema does NOT contain commit_map_proposals.
    # 4. After locate_targets_in_segment runs, ledger Navigation Candidates contains "Aeneas".
    # 5. After verify_segment_anchors runs, AnswerAgent receives evidence with timeline_rows / observed_order set.
    # See tests/test_iterative_agent.py for builder patterns; reuse the same fixtures.
    pytest.skip("Fill in by following the builder pattern in tests/test_iterative_agent.py")
```

这一步先把 skeleton 放进去；可执行者按 `tests/test_iterative_agent.py` 现成的 fixture（搜 `def build_iterative_agent_for_tests` 或类似工厂）补齐 SceneIndex + 注入 backend。如果工厂不存在，按已有测试模板手抄一个最小版即可。

- [ ] **Step 2：补齐 fixture 并跑测试**

`PYTHONPATH=src pytest tests/agents/test_iterative_temporal_order_anchor_flow.py -v`
Expected: 一个 integration test PASS。

- [ ] **Step 3：跑全套**

`PYTHONPATH=src pytest -q`
Expected: 全部 PASS。

- [ ] **Step 4：（可选）落 trajectory 复跑笔记**

如有真模型环境，按 onboarding 文档第 88-92 行步骤跑一次 611-2 case，把新的 trajectory 输出与旧版对比写进 `docs/tool-flow-debug/2026-06-08-611-2-after-fix.md`（不强求；可执行者无真模型时跳过）。

- [ ] **Step 5：提交**

```bash
git add tests/agents/test_iterative_temporal_order_anchor_flow.py \
        docs/tool-flow-debug/2026-06-08-611-2-after-fix.md  # 仅当存在
git commit -m "test(agents): pin temporal_order locate verify end-to-end flow"
```

---

## Task 9：弃用被替代的旧工具（P0，必须在 Task 7 之后）

**Files:**
- Modify: `src/visual_coding_agent_harness/agents/skills/specs.py`（三个 skill 的 `allowed_actions` 中删 `zoom` / `expand_window` / `read_segment` / `video_ls` / `caption_segments`）
- Modify: `src/visual_coding_agent_harness/agents/iterative_agent.py:1255-1338`（`_repair_skill_route_tool` 加 `zoom` / `expand_window` / `read_segment` → 新工具的 normalize 分支）
- Test: `tests/agents/test_legacy_tool_deprecation.py`（create）

**Symptom this fixes:**
(a) Schema 噪声：611-2 trace 里 planner 看到 `read_segment(segment_id: str)`、`zoom(segment_id, target_granularity_sec=60.0)`、`expand_window(segment_id, before_sec=30.0, after_sec=30.0)` 三个函数签名，但其中 `read_segment` 已被 normalize、`zoom` 全程没用、`expand_window` 在 Round 17/18/20 被 drop —— 全是噪声。
(b) Tool surface 越大，4B 文本 planner 越倾向"复用旧工具"而非用新的 `locate_targets_in_segment → verify_segment_anchors`；本 task 把退路堵死。
(c) Registry 兼容：旧测试、旧 CLI 仍可注册并直接调用旧工具；只是 planner 路径不再暴露 / 自动重定向。

- [x] **Step 1：写失败测试**

新建 `tests/agents/test_legacy_tool_deprecation.py`：

```python
from visual_coding_agent_harness.agents.skills.specs import allowed_actions_for_skill
from visual_coding_agent_harness.agents.prompt_stack import _tool_schema_block


def test_zoom_removed_from_timeline_ordering_allowed_actions() -> None:
    actions = allowed_actions_for_skill("timeline_ordering@v1")
    assert "zoom" not in actions
    assert "expand_window" not in actions
    assert "read_segment" not in actions


def test_zoom_removed_from_mutex_and_grounded_factual_qa() -> None:
    for sid in ("mutex_fact_qa@v1", "grounded_factual_qa@v1"):
        actions = allowed_actions_for_skill(sid)
        assert "zoom" not in actions, f"{sid} still lists zoom"
        assert "expand_window" not in actions, f"{sid} still lists expand_window"
        assert "read_segment" not in actions, f"{sid} still lists read_segment"


def test_caption_segments_and_video_ls_removed_from_all_needle_timeline_skills() -> None:
    for sid in (
        "timeline_ordering@v1",
        "mutex_fact_qa@v1",
        "grounded_factual_qa@v1",
    ):
        actions = allowed_actions_for_skill(sid)
        assert "caption_segments" not in actions, f"{sid} still lists caption_segments"
        assert "video_ls" not in actions, f"{sid} still lists video_ls"


def test_legacy_tools_hidden_from_filtered_tool_schema() -> None:
    rendered = _tool_schema_block(
        option_blind=True,
        active_skill="timeline_ordering@v1",
        exhausted=frozenset(),
    )
    assert "zoom(" not in rendered
    assert "expand_window(" not in rendered
    assert "read_segment(" not in rendered
    assert "video_ls(" not in rendered
    assert "caption_segments(" not in rendered
    # Replacements stay visible
    assert "locate_targets_in_segment(" in rendered
    assert "verify_segment_anchors(" in rendered
    assert "read_segment_detail(" in rendered


def test_legacy_tool_signatures_still_registered_for_back_compat() -> None:
    # Direct ToolRegistry use (e.g. offline pipelines / debug CLI) must still work.
    from visual_coding_agent_harness.registry import ToolRegistry
    from visual_coding_agent_harness.tools.navigation import register_navigation_tools
    from visual_coding_agent_harness.video_map import VideoMapStore, VideoMap, VideoMapSegment

    seg = VideoMapSegment(
        segment_id="seg_0001", start_sec=0.0, end_sec=300.0,
        low_fps_caption="", asr_text="", ocr_text="",
        entities=[], keyframe_paths=[], embedding_refs=[],
    )
    store = VideoMapStore(VideoMap(video_path="/tmp/v.mp4", duration_sec=300.0, segments=(seg,)))
    registry = ToolRegistry()
    register_navigation_tools(registry=registry, video_map_store=store, workspace=None)
    # Even though planner can't see them, the registry MUST still resolve them.
    assert registry.get("zoom") is not None
    assert registry.get("expand_window") is not None
    assert registry.get("read_segment") is not None
```

- [x] **Step 2：跑测试验证失败**

`PYTHONPATH=src pytest tests/agents/test_legacy_tool_deprecation.py -v`
Expected: 前 4 个 FAIL（旧 allowed_actions 仍含 deprecated 工具），第 5 个 PASS（registry 已有）。

- [x] **Step 3：从 skill spec 移除 deprecated 工具**

在 `src/visual_coding_agent_harness/agents/skills/specs.py` 找 `_BUILTIN_SPECS`。对三个 skill 的 `allowed_actions=(...)` 元组分别**移除** `"zoom"`、`"expand_window"`、`"read_segment"`、`"caption_segments"`、`"video_ls"`：

```python
# 修改前 (timeline_ordering@v1 示例):
allowed_actions=(
    "caption_segment", "expand_window", "query_context",
    "read_segment_detail", "read_timeline_sorted", "search_segments",
    "target_coverage", "verify_ledger_answer", "vision_read", "zoom",
    "locate_targets_in_segment", "verify_segment_anchors",  # Task 7 已加
),

# 修改后:
allowed_actions=(
    "caption_segment", "query_context",
    "read_segment_detail", "read_timeline_sorted", "search_segments",
    "target_coverage", "verify_ledger_answer", "vision_read",
    "locate_targets_in_segment", "verify_segment_anchors",
),
```

`mutex_fact_qa@v1` / `grounded_factual_qa@v1` 同样删 `zoom` / `read_segment`（它们原本不含 `expand_window`，按 onboarding §5 表格核对）。
**注意**：`main_idea@v1` 不动 —— 它走 `global_gist` 全片采样路径，与本 plan 替代路径正交。

- [x] **Step 4：在 `_repair_skill_route_tool` 加软重写分支**

打开 `src/visual_coding_agent_harness/agents/iterative_agent.py`，找到 `_repair_skill_route_tool`（onboarding §7 line 283 标注的 `:1255-1338`）。在已有 `main_idea + global_gist` 重写、`mutex_fact_qa + inspect_segment` 重写之外，加：

```python
# Legacy tool deprecation (Task 9 of 2026-06-08 plan):
# All needle/timeline routes — softly rewrite legacy nav tools onto their successors.
if skill_id in {"timeline_ordering@v1", "mutex_fact_qa@v1", "grounded_factual_qa@v1"}:
    if step.tool == "read_segment":
        step = replace(step, tool="read_segment_detail")
        notes.append(NormalizationNote(
            tool="read_segment",
            reason="legacy_read_segment_to_read_segment_detail",
            original={"tool": "read_segment", "segment_id": str(step.args.get("segment_id", ""))},
            resolved={"tool": "read_segment_detail"},
            next_action=(
                "read_segment is deprecated under needle/timeline routes; "
                "use read_segment_detail (it returns target hits + asr_summary + visual_caption)."
            ),
        ))
    elif step.tool == "zoom":
        # Inherit targets from the most recent target_coverage observation.
        inherited_targets = self._inherit_targets_from_recent_target_coverage()
        step = replace(
            step,
            tool="locate_targets_in_segment",
            args={
                "segment_id": step.args.get("segment_id"),
                "targets": inherited_targets,
            },
        )
        notes.append(NormalizationNote(
            tool="zoom",
            reason="legacy_zoom_to_locate_targets_in_segment",
            original={"tool": "zoom", "segment_id": str(step.args.get("segment_id", ""))},
            resolved={"tool": "locate_targets_in_segment"},
            next_action=(
                "zoom only slices metadata without producing target anchors; use locate_targets_in_segment instead, "
                "then verify_segment_anchors on returned anchors_for_vlm."
            ),
        ))
    elif step.tool == "expand_window":
        inherited_targets = self._inherit_targets_from_recent_target_coverage()
        step = replace(
            step,
            tool="locate_targets_in_segment",
            args={
                "segment_id": step.args.get("segment_id"),
                "targets": inherited_targets,
            },
        )
        notes.append(NormalizationNote(
            tool="expand_window",
            reason="legacy_expand_window_to_locate_targets_in_segment",
            original={"tool": "expand_window", "segment_id": str(step.args.get("segment_id", ""))},
            resolved={"tool": "locate_targets_in_segment"},
            next_action=(
                "expand_window only enlarges segment boundaries without new anchors or evidence; "
                "use locate_targets_in_segment, then verify_segment_anchors for evidence-grade observations."
            ),
        ))
    elif step.tool == "video_ls":
        # Drop — scene index is already in the prompt.
        notes.append(NormalizationNote(
            tool="video_ls",
            reason="legacy_video_ls_dropped",
            original={"tool": "video_ls"},
            resolved={},  # dropped
            next_action="Scene index is already in your prompt; do not call video_ls.",
        ))
        return None  # caller treats None as drop
```

`_inherit_targets_from_recent_target_coverage()` 是新 helper，复用 onboarding §1.4 已经存在的"planner 未显式传 `targets` 时，会从最近一次 `target_coverage` observation 继承 target list"逻辑（在 `iterative_agent.py` 搜 `target_coverage` 找现有继承代码，抽成 method 即可）。

`NormalizationNote` 的 dataclass 字段名以代码里实际定义为准（`reason` / `tool` / `original` / `resolved` / `next_action`）。`replace` 来自 `from dataclasses import replace`。

- [x] **Step 5：跑测试**

```bash
PYTHONPATH=src pytest tests/agents/test_legacy_tool_deprecation.py -v
```
Expected: 5 PASS.

- [x] **Step 6：跑全套，修复回归**

`PYTHONPATH=src pytest -q`

Expected：会有旧测试断言"timeline_ordering 的 `allowed_actions` 包含 `zoom`"或类似，按本 task 的新契约更新这些断言。具体来说：
- `tests/test_iterative_agent.py` 中如有 `assert "zoom" in skill.allowed_actions` 之类，删掉这一行（zoom 已 deprecated）。
- `tests/test_prompt_stack_and_skill_runtime.py` 中如有 catalog 完整字符串快照，重新生成快照让其反映新的 allowed_actions。
- `tests/test_agent_normalization.py`（如存在）中如有"`expand_window` step preserved"测试，改成"`expand_window` rewritten to `locate_targets_in_segment`"。

如发现还有 `read_segment` / `zoom` 调用真的在测试期望被 preserve，重读那个测试的意图：多半是测 normalize 路径本身，这种情况下保留旧测试（它现在测的是 `_inherit_targets` + rewrite 路径），改 expected 即可。

- [x] **Step 7：手动 grep 验证**

```bash
grep -rn '"zoom"' src/visual_coding_agent_harness/agents/skills/specs.py
grep -rn '"read_segment"' src/visual_coding_agent_harness/agents/skills/specs.py
grep -rn '"expand_window"' src/visual_coding_agent_harness/agents/skills/specs.py
```

Expected：以上 3 条 grep **均无输出**（main_idea@v1 也不含这些，因为它本来就走 global_gist 路径）。如果 main_idea 仍含 `read_segment`，那是另一个路径（不在本 plan deprecate 范围），保留不动。

- [x] **Step 8：提交**

```bash
git add src/visual_coding_agent_harness/agents/skills/specs.py \
        src/visual_coding_agent_harness/agents/iterative_agent.py \
        tests/agents/test_legacy_tool_deprecation.py
git commit -m "feat(skills): deprecate zoom/expand_window/read_segment under needle/timeline routes"
```

---

## Out of Scope（明确不在本 plan）

- `caption_segment` / `vision_read` 的 `max_new_tokens` / `repetition_penalty` 调整（已在 `2026-06-07-video-agent-claude-code-style-fix.md` Task 9 覆盖）。
- AnswerAgent 对 timeline 证据的解析改造（依赖 Task 6 落地后的 `timeline_rows`，另起 plan）。
- VL backend 是否真的能接受 `metadata.start_sec/end_sec` —— 这里依赖现有 inspector/caption_segment 的接口；如不支持，Task 6 实现层会通过 `extract_clip` 物化子 clip（参照 `tools/segments.py` 已有做法）后再喂 VLM，但**测试与文档接口不变**。
- `Reflection Memory` 内容质量提升 / `Last Round Adjustments` 加 DO NEXT 指令（另起 plan，与 Task 1-3 互补）。
- **VLM 单段聚焦工具三选一合并**（`inspect_segment` / `caption_segment` / `qa_segment` / `vision_read`）：它们的单段聚焦版本各自有语义区分（open caption / QA / ask_for 风格）。段内目标排序 / fanout 职能由本 plan 的 `locate_targets_in_segment` → `verify_segment_anchors` 接管，是否把这些通用 VLM 工具再合并到一个统一签名是独立的 surface area 清理工作，本 plan 不动。
- **`main_idea@v1` skill 的工具集精简**：本 plan Task 9 显式不动 main_idea 路径，因为它走 `global_gist` 全片采样 + 索引后 fallback，与 needle/timeline 路径职能正交。一旦后续 plan 把 gist 路径也接到 target-anchor pipeline（例如"对 top-K 段先 locate 再 verify"），再统一处理。

---

## Self-Review

**1. Spec coverage check:**
- 取消 per-class 工具预算：Task 0（删 `AgentBudget` 三个预算字段 + 三个工具分类常量 + `_tool_budget_available` 检查 + `_budget_snapshot_block` 中的 per-class 行）✓
- 优化 prompt stack：Task 1（tool schema 按 active_skill 过滤）+ Task 2（final gate route 折叠）+ Task 3（skill catalog 聚焦）+ Task 4-5（nav digest 回流）✓
- segment detail zoom-in：Task 6a/6b/6c（alias matcher + `locate_targets_in_segment` + `verify_segment_anchors`）+ Task 7（skill 白名单）+ Task 8（端到端验证）✓
- 让 planner 调用好 VL Agent 验证证据：`locate_targets_in_segment` 只产出 candidate anchors，`verify_segment_anchors` 才把 focused VLM 确认写入 timeline/evidence，这是 planner 通过工具调用让 VL Agent 做定向验证的主路径 ✓
- 旧工具弃用：Task 9 把被替代的 `zoom` / `expand_window` / `read_segment` / `video_ls` / `caption_segments` 从 needle/timeline skill 的 `allowed_actions` 中移除，同时在 `_repair_skill_route_tool` 增加软重写分支保持 registry 兼容 ✓

**2. Placeholder scan:**
- "Step 1: 补齐 fixture" 没贴完整 builder 代码，但给了"搜 `def build_iterative_agent_for_tests`"的具体路径锚点，符合"展示如何"而非 TBD。✓
- Task 5 Step 4 没贴 `_record_observation_to_ledger` 的具体行号，因为它不在我们直接修改的窗口，使用搜索锚点 `ledger.md` 写入路径替代。可接受。
- Task 9 Step 4 的 `_inherit_targets_from_recent_target_coverage()` 抽取从已有继承逻辑（见 onboarding §1.4 第 79-84 行），给了 grep 锚点 `target_coverage`，非 TBD。✓

**3. Type consistency:**
- `nav_digest` 在 Task 4（产生）/ Task 5（消费）/ Task 4 测试 / Task 5 测试 名字一致。
- `locate_targets_in_segment` / `verify_segment_anchors` 工具名在 Task 6b/6c（实现）/ Task 7（skill 白名单）/ Task 8（e2e）/ Task 9（重写目标）一致。
- `anchors_for_vlm` / `confirmations` / `rejections` / `timeline_rows` 字段在 Task 6b/6c + Task 8 一致。
- `allowed_actions_for_skill` 在 Task 1（消费）/ Task 7（测试）/ Task 9（测试）一致；Task 1 Step 3 同时定义了这个函数。
- `NormalizationNote` 字段（`tool` / `reason` / `original` / `resolved` / `next_action`）在 Task 9 重写分支与现有 `_normalization_notes_body`（prompt_stack.py:586-606）渲染层匹配。

**4. Execution ordering:**
- Task 0（删预算）→ Task 1-3（精简 prompt）→ Task 4-5（surface tool 结果）→ Task 6（新工具）→ Task 7（白名单）→ Task 8（e2e）→ Task 9（删旧工具）。
- Task 0 必须最先：它会**大量删除**旧测试与 `build_replanning_prompt` 调用点中的 `tool_class_counts` 用法；如果放后面，Task 1-3 修改 prompt_stack 时会反复触碰这些行，造成 merge conflict。
- Task 6b/6c（注册 `locate_targets_in_segment` / `verify_segment_anchors` 到 registry）→ Task 7（加入 `allowed_actions`）→ Task 9（移除被替代的旧工具）：这个顺序保证 Task 9 不会留下"无 target-anchor 路径可用"的窗口。
- Task 1 测试中只断言稳定子串（`caption_segment` / `read_segment_detail` 必有，`commit_map_proposals` 必无），不断言 `zoom(` 也不断言 `locate_targets_in_segment(`，从而对 Task 7/9 是否已执行**无依赖**。

**5. 计费机制取消的下游影响（须在 Task 0 PR 描述里点明）:**
- 18 轮以上长 trajectory 现在会**真正**跑满 `max_rounds`，不会再因预算 drop 提前终止。如果实测下来 4B planner 在没有预算压力时会胡乱挥霍工具调用，**不要**把预算加回来——改成调小 `max_rounds`（例如 20 → 12）或调小 `max_tool_calls_per_round`（2 → 1）。
- `runs/` 目录的旧 trajectory 中 `tool_budget_exhausted` 字段会成为历史伪信号，不影响读取，但 audit_trajectory CLI 若有"按 reason 分组统计"功能可能输出 0 计数——可接受。
- 若后续需要重新引入计费（例如改成"按 token / 按 seconds 计费"而非按工具类计费），从一个干净的"只算轮次"基线出发比从混合状态出发简单得多。本 plan 的 Task 0 就是为此做的清理。

---

## 执行接力

完成 Task 1-7 后，**做以下之一**：

1. **Subagent-Driven**（推荐）：每 Task 一个独立 subagent，间隔 review；
2. **Inline Execution**：在当前 session 顺序执行，每 Task 完成后 checkpoint。

哪种方式由发起者决定；本 plan 文档对两种方式都是自包含的。
