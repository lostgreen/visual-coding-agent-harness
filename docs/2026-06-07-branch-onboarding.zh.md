# 分支 `codex/visual-harness-ticket-plan` Onboarding (2026-06-07)

> 这份文档让接手 Agent 在不重读全部代码的前提下, 快速建立对当前分支的心智模型。
> 看完这里 + `2026-06-05-harness-diagnostic-and-roadmap.md` + `2026-06-06-handoff-video-harness-runtime.md` 就足够动手。
> 配套修复 plan: `docs/superpowers/plans/2026-06-07-video-agent-claude-code-style-fix.md`。

---

## 1. 一句话总结

这是一套 "Claude-Code-Style" 的长视频视觉 Agent 框架: 一个文本 planner LLM 通过 ReAct 循环
调度受限的视觉/导航/验证工具集, 把观察结果写入一个结构化 Evidence Workspace, 最终把答案
合成交给 AnswerAgent。当前实测在 VideoMME 上 fragile —— 短问 (gist) 靠 fallback 救场,
长问 (timeline) 会循环到 max_rounds 拿不到答案。

### 2026-06-08 当前状态覆盖

以下内容覆盖本文后面 2026-06-07 的旧预算/zoom/scan 语义：

- 运行时已取消 `cheap / expensive / verifier` 三类工具预算，只保留 `max_rounds`、`max_tool_calls_per_round`、`max_repeated_programs` 和 final-round 规则。
- 新版目标链路统一为 `locate_targets_in_segment -> verify_segment_anchors`。`scan_segment` 只保留在计划文档的历史背景里，不是执行路径。
- planner-visible schema 已列出 `locate_targets_in_segment` 与 `verify_segment_anchors`，不再列出 `zoom` / `expand_window`；若 planner 旧习惯仍调用旧工具，runtime 会软重写为 locator。
- Layer 0：`Compact scene index` 仍是每轮默认地图，但当 MCQ rewrite 提供 `target_entities` 时，会额外渲染 `asr mentions: target @ ~T`，用于段级定位提示；它不是答案证据。
- Layer 1：`locate_targets_in_segment(segment_id, targets)` 只读 ASR sentence / OCR / visual caption / entities，输出 `candidates` 与 `anchors_for_vlm`，属于 navigation candidate，不写 evidence table。
- Layer 2：`verify_segment_anchors(segment_id, anchors, targets)` 对 Layer 1 anchor 做 focused VLM 验证，解析 `confirmations` / `rejections` / `timeline_rows`；只有确认项写入 `timeline.md` 与 evidence table。
- review follow-up 已补：locator 默认每 target 保留 top-k 候选；`David` / `Apollo` 这类 common single-token target 需要上下文才形成低置信 route hint；`verify_segment_anchors` 在 anchor union 超过 45s 时会拆分为多个 focused VLM request。
- `read_segment_detail` 现在是真正的 cheap/index detail pack，并通过 `nav_digest` 回流到 compact ledger；planner 后续轮次能直接看到 target、visual caption、ASR/OCR 摘要，而不是只看到 `obs_xxx: read_segment_detail`。
- AnswerAgent / sufficiency predicates 已把 `locate_targets_in_segment` 明确排除为导航提示，最终答案必须引用 `vision_read` / `inspect_segment` / `caption_segment` / `qa_segment` / `verify_segment_anchors` 等 evidence-grade 工具。
- 当前本地验证：`PYTHONPATH=src:. pytest -q` -> `381 passed`。

配套工具流程文档：

- `docs/tool-flow-debug/2026-06-08-locate-verify-tool-flow.md`

### 2026-06-07 晚间最新状态

当前主线不再是早期的 `video_ls` / 段池耗尽问题，而是在调 VideoMME MCQ rewrite 与 local VLM prompt：

- hard-skill runtime 已按用户意图先关掉，不继续走硬编码 skill procedure。
- `video_ls` 已降级：planner 默认看 `Compact scene index`，用 `target_coverage` / `read_segment_detail` 做定位和展开。
- VideoMME 默认 scene index 已切到 `dual-source`，Visual caption 优先于 ASR summary。
- `03a4832` 修复了 temporal MCQ rewrite：`exploration_question` 不再列 target names，local VLM 工具只做开放 caption。
- `target_entities` 仍保留为结构化 metadata，供 `target_coverage` / `read_segment_detail` / AnswerAgent 后处理。

当前 KML rewrite audit：

- baseline：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/rewrite_audit_baseline_cf7f185_50_20260607`
  - `target_detector_prompt=22`
  - `temporal_targets_in_tool_question=6`
- 修复版：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/rewrite_audit_open_03a4832_50_20260607`
  - `target_detector_prompt=7`
  - `temporal_targets_in_tool_question=0`
  - `option_surface_leak=0`
  - `missing_temporal_targets=0`

下一步是第二轮收紧：候选答案值只能进入 `target_entities`，不能进入 `exploration_question`；然后用同一批 50 题复跑 audit。

第二轮已提交：

- `522eda4 fix(agent): reject option-value leaks in mcq rewrites`
- KML run：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/rewrite_audit_open_522eda4_50_20260607`
- 完成后读取 `summary.json`，再决定是否启动 611-2 agent rerun。

第三轮已提交：

- `678323c fix(videomme): use one-line scene map in planner prompt`
- 直接修复用户指出的旧 trajectory 问题：`Compact scene index` 不应在每轮默认 prompt 展开 `Visual: ... | ASR: ... | Tags: ... | Entities: ...`。
- 新字段：`VideoSegment.map_summary`。
- dual-source builder 现在每段先生成：
  - `visual_caption`：完整视觉 caption，供按需 detail / navigation 使用。
  - `asr_summary`：字幕摘要，作为补充。
  - `map_summary`：文本模型合成的一句话 Layer 0 地图，供每轮 planner prompt 使用。
- `SceneIndex.summary()` 现在只输出 `map_summary` 或旧 fallback，不再默认展开完整 caption/ASR/tags/entities。
- `read_segment_detail` / `target_coverage` 仍通过 `VideoMap.from_scene_index()` 使用完整 `visual_caption/asr_summary`。
- scene-index cache schema 已从 `dual_source_scene_index_v2` bump 到 `dual_source_scene_index_v3`，KML 复跑会避开旧缓存。
- 新 KML 611-2 run 已挂起：
  - run root：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_no_hard_skill_dual_source_open_rewrite_611_2_678323c_map_summary_20260607`
  - pid path：`/tmp/videomme_agent_no_hard_skill_dual_source_open_rewrite_611_2_678323c_map_summary_20260607.pid`
  - log path：`/tmp/videomme_agent_no_hard_skill_dual_source_open_rewrite_611_2_678323c_map_summary_20260607.log`

第四轮当前本地改动：

- planner-visible schema 已隐藏 `caption_segments` / `ingest_segment_metadata`，只保留 registry 兼容和 debug 使用。
- planner-visible `verify_ledger_answer` schema 已隐藏 `ledger_text`。
- normalization 会剥掉 planner 传入的 `verify_ledger_answer.ledger_text`，避免 `"obs_0009, obs_0010"` 这类字符串覆盖真实 workspace ledger。
- option-blind MCQ rewrite 后，如果有 `target_entities` 且 registry 有 `target_coverage`，会自动在第一轮 planner 前 seed 一条 `target_coverage` observation。
- compact ledger 的 `Navigation Summary` 会对 `target_coverage` 展开短 claim，让 Layer 1 target coverage matrix 每轮进入 planner prompt。

第五轮当前本地改动：

- `read_segment_detail` 不再被后续策略吞掉；导航后需要视觉验证时，agent 保留原 `read_segment_detail`，并在本轮预算允许时追加 `vision_read` follow-up。
- `read_segment_detail(segment_id)` 现在是真正的 cheap/index detail pack：
  - 返回完整 `visual_caption` / `asr_summary` / `raw_asr_excerpt` / `ocr_text` / `entities`。
  - 返回 `target_hits`、`target_matches`、`unmatched_targets`、`recommended_next_tools`。
  - planner 未显式传 `targets` 时，会从最近一次 `target_coverage` observation 继承 target list。
- 保留旧的空 `read_segment` -> `caption_segment` 升级逻辑；该逻辑不适用于 `read_segment_detail`。

第六轮当前本地改动：

- timeline / grounded QA / mutex QA route 下，planner 调 `read_segment` 会 normalize 成 `read_segment_detail`，并自动带上 `target_entities`。
- `target_coverage` 候选现在带 `source` / `snippet` / `directness`，用于区分 direct mention、possible mention、weak overlap。
- option-blind timeline local VLM query 现在是 target-aware open grounding：只传 unordered targets，不传 A/B/C/D 或候选顺序。
- 仍未完成：独立 `ground_targets_in_segment`、结构化 `timeline_event` rows、`build_observed_order`、final-round hard takeover。

当前本地验证：

```bash
PYTHONPATH=src:. pytest -q
# 368 passed
```

详细调试笔记：

- `docs/tool-flow-debug/2026-06-07-611-2-mcq-rewrite-and-tool-flow.md`
- `docs/tool-flow-debug/2026-06-07-scene-index-and-navigation-tools.md`

---

## 2. 目录速览 (只列与运行循环相关的)

```
src/visual_coding_agent_harness/
├── agents/
│   ├── iterative_agent.py    # ★ 主循环。3749 行, IterativeVisualAgent.run() 是入口
│   ├── prompt_stack.py       # ★ 每轮 prompt 组装。改这里 = 改 planner 看到的世界
│   ├── answer_agent.py       # AnswerAgent: 把 evidence_table 投给 A/B/C/D
│   ├── distill.py            # Observation -> EvidenceRow 蒸馏
│   ├── question_policy.py    # 路由分类: gist_global / temporal_order / needle_local
│   ├── open_questions.py     # MCQ -> option-blind exploration question 重写
│   ├── followup.py           # FollowupScheduler: 多事实任务的查询调度
│   ├── output_quality.py     # confidence_signal 判断
│   ├── prompt_stack.py       # (重复, 上面已列)
│   ├── context_budget.py     # ContextBudgetAllocator: token 配额切片
│   ├── contracts.py          # VISUAL_EVIDENCE_NFRAMES 等常量
│   └── skills/
│       ├── specs.py          # ★ SkillSpec + builtin_skill_registry() (4 个内置 Skill)
│       └── predicates.py     # sufficiency / verifier_checks 谓词实现
├── backends/
│   ├── base.py               # BackendRequest / BackendResponse / Protocol
│   ├── qwen_vl.py            # Qwen3-VL 视觉/多模态
│   ├── qwen_text.py          # Qwen3-4B-Instruct 纯文本 (planner / answer / verify)
│   └── routed.py             # RoutedBackend: 按 task 名路由到 text 或 VL
├── tools/                    # @tool 注册的所有可调度工具 (见 §4 工具目录)
├── workspace.py              # ★ EvidenceWorkspace, 2400+ 行, 持久化 8 种 JSONL/MD
├── interpreter.py            # ProgramInterpreter: 执行编译后的 SkillStep[] (foreach/assign)
├── video_index.py            # SceneIndex / VideoSegment, 固定窗口切片
├── video_map.py              # VideoMap / VideoMapStore: 可写的索引+搜索
├── registry.py               # ToolRegistry / @tool 装饰器 / ToolError
├── protocol.py               # ToolRequest / ToolResult dataclass
├── schemas.py                # JSON Schema for tool I/O
├── iterative_smoke.py        # 本地一键跑 mini agent (无真模型)
└── cli/                      # 各 CLI 入口: eval_videomme / run_ablation / ...
```

`tests/` 254 个 pytest 用例, `runs/` 是 artifact 输出, `plans/` 是旧 plan,
`docs/` 是历史 handoff + 设计文档, `docs/superpowers/plans/` 是新约定的 plan 落点。

---

## 3. 一次 `run()` 调用的实际数据流

入口: `IterativeVisualAgent.run(question, video_path)` (`iterative_agent.py:179`)。

```
                       ┌─────────────────────────────────────────────┐
question + video_path  │ 1. _question_for_exploration 重写问题 (option-blind)
                       │ 2. workspace.ensure_hypothesis              │
                       │ 3. _seed_scene_coverage_evidence            │
                       │ 4. (gist_global) _try_global_gist_route ←──── 预先 1 次 global_gist 写入 obs_0001
                       │ 5. (hard) _try_hard_skill_route             │
                       └────────────────────┬────────────────────────┘
                                            │
              ┌──────────────  循环 round 1..max_rounds  ──────────────┐
              │                                                       │
              │ A. workspace.evidence_status_summary()                │
              │ B. build_replanning_prompt(...) ────► prompt_stack.py │
              │    └─ slots: task, trajectory, hypothesis, evidence,  │
              │       scene_index, feedback, budget, tooling          │
              │ C. backend.generate(task="replan", prompt=...) → JSON │
              │ D. _parse_replan_action → {status, skill, program[]}  │
              │ E. _normalize_program(program)                        │
              │    ├─ _repair_skill_route_tool (✗ 当前重写而不告知)   │
              │    ├─ replace_video_path_placeholder                   │
              │    ├─ _resolve_media_segment (✗ 池空时 silent drop)   │
              │    ├─ _tool_budget_available / route_violation        │
              │    └─ append NormalizationNote → 下轮 feedback        │
              │ F. ProgramInterpreter.run(normalized) → observations  │
              │ G. workspace 持久化 observations + evidence + ledger  │
              │ H. status == "final" → _try_answer_agent_final 救场   │
              │                                                       │
              └─────────────────  到达 max_rounds  ──────────────────┘
                              │
                              ▼
            _try_low_confidence_final + max_rounds_reached
```

每轮真正驱动的是 **prompt + planner JSON + 一次 normalize → interpreter** 三段。
全部的修补 (repair / 替换 / drop) 都集中在 `_normalize_program` 内, 这是排查问题的第一站。

---

## 4. 工具目录 (按文件归组)

所有工具通过 `@tool` 装饰器注册到 `ToolRegistry`。Planner 看到的 schema 是
`prompt_stack._tool_schema_block` 硬编码的子集 (一份白名单), 不是 registry 全量。

| 文件 | 工具 | 角色 |
|------|------|------|
| `tools/global_view.py` | `global_gist` | **one-shot** sparse 全片采样, 仅作为 gist 主题暗示 (claim 被隐藏给 planner) |
| `tools/navigation.py` | `video_ls`, `search_segments`, `ground_question`, `read_segment`, `expand_window`, `zoom`, `commit_map_proposals` | 文本/索引层导航, 不读像素 |
| `tools/inspector.py` | `inspect_segment`, `vision_read` | **像素读**, 走 Segment Inspector 子 prompt; vision_read = facts-only |
| `tools/segments.py` | `caption_segment`, `qa_segment` | 也是像素层, 但保留 question→answer 风格 |
| `tools/enrichment.py` | `caption_segments`, `ingest_segment_metadata` | 离线 VideoMap 缓存填充, 在线不推荐 |
| `tools/query_context.py` | `query_context` | 全局 context capsule, 不投票 |
| `tools/verification.py` | `verify_ledger_answer`, `summarize_ledger_evidence` | verifier 类, 独立预算 |
| `tools/workspace_primitives.py` | `view_observation`, `grep_evidence`, `query_evidence_table`, `read_timeline_sorted`, `read_hypothesis`, `update_hypothesis_slot`, `append_to_timeline` | 直接读写 workspace, 几乎无副作用 |
| `tools/vlm.py` | `caption_image`, `qa_image`, `caption_region`, `qa_region`, `caption_video`, `qa_video` | 早期 P0 工具, 现已基本被 inspector 系列取代 |
| `tools/traditional.py`, `tools/image_atomic.py` | `crop_region`, `zoom_region`, `threshold_image`, `enhance_image`, `edge_detect`, `sample_frames`, `extract_clip` | Pillow / ffmpeg 处理, 给离线管道用 |
| `tools/dummy.py` | `caption_image`, `ocr_region`, `verify_answer` (同名 stub) | 测试用 |

工具分类预算 (`iterative_agent.py:35-58`):

- **cheap** (`cheap_tool_budget`, 默认 16): video_ls, search_segments, ground_question, read_segment, expand_window, zoom, summarize_ledger_evidence
- **expensive** (`expensive_tool_budget`, 默认 6): global_gist, inspect_segment, vision_read, caption_segment, qa_segment, caption_segments, caption_region, qa_region
- **verifier** (`verifier_tool_budget`, 默认 2): verify_ledger_answer

每轮 `_normalize_program` 会同时检查 (1) 工具属哪一类, (2) 是否还有该类预算, (3) 该 Skill 的 `allowed_actions` 是否允许该工具。注意第 (3) 项目前仅做软提示, 没有运行时 drop —— **这是诊断 P1 项**, 也是修复 plan Task 6 要解决的事。

---

## 5. Skill 体系 (`agents/skills/specs.py`)

四个 builtin Skill, 由 `select_skill(question)` 按问题路由选出:

| Skill | 路由 | 关键工具白名单 | 触发标记 (markers) |
|-------|------|----------------|---------------------|
| `main_idea@v1` | `gist_global` | `global_gist`, `query_context`, `vision_read`, `video_ls`, `search_segments`, `verify_ledger_answer` | "main idea", "overall", "summary", "mainly about", "synopsis" |
| `mutex_fact_qa@v1` | `needle_local` | `ground_question`, `query_context`, `vision_read`, `zoom`, `video_ls`, `search_segments`, `verify_ledger_answer` | "option", "neither", "true" |
| `grounded_factual_qa@v1` | `needle_local` | `ground_question`, `query_context`, `vision_read`, `zoom`, `video_ls`, `search_segments`, `verify_ledger_answer` | "which", "what", "where", "who" |
| `timeline_ordering@v1` | `temporal_order` | `caption_segment`, `query_context`, `vision_read`, `read_timeline_sorted`, `video_ls`, `search_segments`, `expand_window`, `zoom`, `verify_ledger_answer` | "before", "after", "first", "last", "then", "order", "sequence" |

`SkillSpec.procedure` 还定义了多步执行流程, 但 **当前 `_try_hard_skill_route` 只对 `grounded_factual_qa` / `mutex_fact_qa` / `timeline_ordering` 在前置阶段调度 `FollowupScheduler` 来跑, `main_idea` 走的是 `_try_global_gist_route` 强制一次 `global_gist` 后落到普通 ReAct 循环**。procedure 字段对 planner 是不可见的 — planner 只看到 `skill_catalog_prompt()` 产出的一行 metadata。这是诊断 P0。

---

## 6. Prompt Stack: planner 每轮看到什么

`build_replanning_prompt` (`prompt_stack.py:75`) 调用 `compose_replanning_prompt_slots`, 再让 `ContextBudgetAllocator` 按 token 配额截断。完整 slot 列表 + 渲染顺序:

```
┌──────────────────────────────────────────────────────────────────┐
│ task                                                             │
│   # Base Identity                                                │
│   # Route Playbook         (来自 _route_playbook_body)            │
│   # Skill Catalog          (skill_catalog_prompt() 静态全量列表)  │
├──────────────────────────────────────────────────────────────────┤
│ trajectory                                                       │
│   Round X/Y, Already inspected segments: ...                     │
├──────────────────────────────────────────────────────────────────┤
│ hypothesis                                                       │
│   # Hypothesis             (来自 workspace.read_hypothesis_text)  │
├──────────────────────────────────────────────────────────────────┤
│ evidence                                                         │
│   Evidence status summary  (option_coverage, coverage_pct, ...)   │
│   Evidence ledger          (compact ledger_text)                  │
├──────────────────────────────────────────────────────────────────┤
│ scene_index                                                      │
│   Question (option-blind 重写过), Uninspected candidates,         │
│   每段的 ASR summary + Visual caption + Tags + Entities           │
├──────────────────────────────────────────────────────────────────┤
│ feedback                                                         │
│   # Last Round Adjustments (NormalizationNote 列表)               │
│   # Answer Feedback        (AnswerAgent 反馈的 missing_evidence)   │
│   # Reflection Memory      (历史失败的硬性规则记忆)               │
├──────────────────────────────────────────────────────────────────┤
│ budget                                                           │
│   Round X/Y, max_tool_calls_per_round, cheap/expensive/verifier  │
├──────────────────────────────────────────────────────────────────┤
│ tooling                                                          │
│   # Tool Schema           (硬编码白名单 + signature)              │
│   # Final Gate            (策略提示, 比如 "use video_ls first")   │
│   # Response Contract     (continue/final JSON schema)            │
└──────────────────────────────────────────────────────────────────┘
```

关键点 / 易踩坑:

1. **Skill Catalog 是静态的**, 不感知运行时 (one-shot 已用、预算耗尽), 这是 planner 反复请求 `global_gist` 的直接原因。
2. **Route Playbook ≠ Skill**: `Route Playbook` 是按问题分类的固定指令文本, `Skill` 才是有 `allowed_actions` 约束的程序契约。两者会同时出现, 但 planner 可能误把 Playbook 当作 hard rule。
3. **Last Round Adjustments 当前只是被动列表** (`- global_gist -> vision_read (reason: …)`), 没有 "DO NEXT" 句式, planner 不会改行为。
4. **Reflection Memory 是持久化的失败规则**, 写入 `workspace.write_reflection_memory()`。目前只有 `planner_json_parse_error` 和 `iterative_final_blocked` 会写, repair 路径不写 → planner 无法跨轮学习。
5. **Tool Schema 是硬编码列表**, 不与运行时 `allowed_actions` 联动, 也不会因预算耗尽自动缩减。
6. **Evidence Ledger 把 `global_gist` 的 claim 标成 "claim hidden from planner"** (`workspace.py:2327`), planner 看不到主题, 自然反复想再来一次。

---

## 7. `_normalize_program` 的修补管线 (这是踩坑最多的地方)

按调用顺序: (`iterative_agent.py:990-1220`)

1. **route_violation**: tool 不在 `skill.allowed_actions` → 标记 `route_violation` + drop。
2. **_repair_skill_route_tool** (`iterative_agent.py:1255-1338`):
   - `main_idea` + 重复 `global_gist` (obs_count ≥ 1) → 静默重写为 `vision_read` 下一未访问段。**(造成 605-1 循环)**
   - `main_idea` + 非法 `vision_read` (尚未跑过 global_gist) → 重写回 `global_gist`。
   - `mutex_fact_qa` + `inspect_segment` → 重写为 `vision_read`。
   - `timeline_ordering` + `caption_segments`/批量 `caption_segment` → 重写为单段 `caption_segment`。
3. **replace_video_path_placeholder**: planner 给空字符串或 `{video_path}` 占位 → 注入真实路径。
4. **reserve_final_round**: 最后一轮非 verifier 工具 → drop。
5. **upgrade_empty_read_segment_to_caption**: `read_segment` 段无索引文本 → 升级为 `caption_segment`。
6. **_tool_budget_available**: 该类预算耗尽 → drop。
7. **_resolve_media_segment** (`iterative_agent.py:1487-1558`): 段已被 inspect → 找下一未访问段; 全部用尽 → 返回 None → drop。**(造成 611-2 max_rounds)**
8. **_tool_exploration_question**: vision_read 的 `ask_for` 被 sanitize, 去掉选项文本。
9. 全部 drop 后, fallback: 选一个未访问段 + 当前 Skill 偏好的 visual tool 拼一个最小 program。

每一步都会调 `_append_normalization_note(notes_out, ...)`, 下一轮 prompt 的 `Last Round Adjustments` block 把它们渲染给 planner。但渲染缺 "DO NEXT" → planner 行为不变。

---

## 8. EvidenceWorkspace 持久化文件

`workspace.py` 在 `runs/<run_id>/` 下维护:

| 文件 | 内容 |
|------|------|
| `observations.jsonl` | 每次 tool 调用的原始 raw_output 包 |
| `evidence.jsonl` | 蒸馏后的事实行 (claim/confidence/grounding_quality/regions) |
| `evidence_table.jsonl` | 按 option/事实分桶的 evidence table (AnswerAgent 输入) |
| `ledger.md` | 给 planner 看的人类可读 compact ledger |
| `timeline.md` | 时序 facts, 给 `read_timeline_sorted` 用 |
| `hypothesis.md` | hypothesis slot 状态 |
| `map_proposals.jsonl` | VideoMap 待提交的修改 |
| `reflection_memory.jsonl` | 跨轮的失败规则记忆 (但目前只两处写入) |
| `manifests.jsonl`, `frame_sets/...` | 抽帧/clip artifact 索引 |
| `trace.jsonl` | 所有 trace event (调试金矿) |
| `planner_io/round_*.json` | 每轮完整 prompt + response (调试金矿) |

要排查任何一轮发生了什么, 先看 `trace.jsonl` (各种 `*_route`, `*_repaired`, `_adjustment` event), 再看对应 `planner_io/round_N.json` 看 prompt 和原始 JSON。

---

## 9. Backend 与模型

- `backends/qwen_text.py`: Qwen3-4B-Instruct 跑 planner / AnswerAgent / verifier / 问题重写。
- `backends/qwen_vl.py`: Qwen3-VL-4B 跑视觉工具 (`global_gist`, `vision_read`, `inspect_segment`, `caption_segment`, `qa_segment`, …)。
- `backends/routed.py`: `RoutedBackend` 用 `BackendRequest.task` 名字路由。

注意:
- `qwen_vl.py:66-71` 的 `model.generate` 没有 `repetition_penalty`/`no_repeat_ngram_size`/`top_p`, 是 vision_read 出现 "同一句话重复 7 遍 + 截断" 的根因之一。
- `tools/inspector.py:243` 限定 `max_new_tokens=256` (vision_read 也共用), 导致经常截断未说完。
- `inspector.py:252` 的 confidence 是硬编码 `0.74` —— 每条 vision 观察的可信度都一样, AnswerAgent 没法据此排序。

---

## 10. AnswerAgent / 终态裁决

`AnswerAgent.run` (`answer_agent.py:68`) 流程:
1. 优先看 `evidence_table` 是否已经有 `option_support` → 走 `arbitrate_evidence_table` (规则裁决, 无需 LLM)。
2. 否则把 `evidence_text` 喂给文本 backend 做 `answer_from_evidence`。
3. 解析失败时, `_fallback_main_idea_from_unassigned_evidence` 尝试基于 unassigned evidence 救场 (commit `f0e01e3`)。
4. `AnswerAgentResult.as_low_confidence_final` 在 follow-up 预算耗尽时退化为 best-effort option (commit `a227242`)。

主循环的 `_try_answer_agent_final` / `_try_low_confidence_final` 在以下时机被调用:
- planner 直接 `status=="final"` 但 MCQ 答案需要复核;
- `evidence_table_no_growth` 多轮无新证据;
- `reserved_final_round` 或 `iterative_final_blocked`。

`max_rounds_reached` 时也会兜底再问一次 (`Answer after full segment sweep`, commit `4540cd2`), 但若 evidence 完全没有 visual citation, 仍可能空答案 (这就是 611-2 的失败模式)。

---

## 11. 测试 / 跑通命令

```bash
# 全套
PYTHONPATH=src python3 -m pytest -q
# 关键单元
PYTHONPATH=src python3 -m pytest tests/test_iterative_agent.py -v
PYTHONPATH=src python3 -m pytest tests/test_prompt_stack_and_skill_runtime.py -v

# 本地 mini 烟测
PYTHONPATH=src python3 -m visual_coding_agent_harness.iterative_smoke

# VideoMME 3-case 跑分 (需要 KML 真模型环境)
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.eval_videomme --help
```

跑后看 `runs/<run_id>/trajectories/*_agent_v2.md`, 它由 `cli.audit_trajectory` 渲染, 是最直接的 debug 入口。

---

## 12. 当前已知问题速查表 (来自 605-1 / 611-2 trajectories)

| ID | 现象 | 根因文件:行 | 修复 plan task |
|----|------|-------------|----------------|
| F1 | `global_gist` claim 隐藏 → planner 反复请求 | `workspace.py:2327` | Task 1 |
| F2 | Skill Catalog 静态, 不去掉已 exhausted | `agents/skills/specs.py:74-84` | Task 2 |
| F3 | exhausted_tools 没有传到 prompt | `agents/prompt_stack.py:152-162, 250-260` | Task 3 |
| F4 | run() 没有计算 exhausted_tools | `agents/iterative_agent.py:240-255` | Task 4 |
| F5 | repair_repeated_global_gist 静默重写 + 不写 reflection memory | `agents/iterative_agent.py:1278-1298` | Task 5 |
| F6 | `allowed_actions` 不作为运行时 deny-list | `agents/iterative_agent.py:1040-1057` | Task 6 |
| F7 | 段池耗尽时 drop 无替代 | `agents/iterative_agent.py:1129-1151` | Task 7 |
| F8 | Last Round Adjustments 无祈使句 | `agents/prompt_stack.py:581-598` | Task 8 |
| F9 | VLM 解码无 repetition_penalty + max_new_tokens 偏低 | `tools/inspector.py:243`, `backends/qwen_vl.py:66-71` | Task 9 |
| F10 | hard_skill_runtime 不真正驱动 procedure | `agents/iterative_agent.py:1813-1857` | 留给后续 plan (诊断 P0) |
| F11 | confidence 硬编码 0.74 | `tools/inspector.py:252` | 留给后续 plan |

---

## 13. 给后续 Agent 的建议

1. **改循环行为前, 先在 `tests/test_iterative_agent.py` 里加用例固化期望**。该文件已有大量循环行为契约, 是回归保护层。
2. **改 prompt 前, 先 grep `prompt_stack.py` 里的 block 名称** (`name="…"`), 避免漏掉两份并行的 `compose_replanning_prompt_*`。
3. **改 normalization 前, 想清楚 reason 名**, 因为 `audit_trajectory` CLI、`trace.jsonl` 分析脚本、`prompt_stack` 渲染都依赖 reason 字符串匹配。新增 reason 务必更新 `_normalization_notes_body`。
4. **修复 plan 顺序很重要**: Task 1-4 是"让 planner 看见状态", 5-8 是"让 planner 必须改行为", 9 是"工具本身别废", 10-11 是回归 smoke。从 1 开始顺序做, 不要跳。
5. **当前分支的 baseline trajectory** 在 `/Users/lostgreen/Downloads/trajectories/{605-1, 611-2}_agent_v2.{md,json}` —— 任何"修好了"的判断, 都要对照这两份重跑后验证。
