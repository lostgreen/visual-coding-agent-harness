# 611-2 replay 交互轮次问题分析

日期：2026-06-08

## 当前证据

当前有效 replay：

- run root：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_anchor_hardened_611_2_5c87449_20260608`
- trajectory：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_anchor_hardened_611_2_5c87449_20260608/trajectories/611-2_agent_v2.json`
- markdown trajectory：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_anchor_hardened_611_2_5c87449_20260608/trajectories/611-2_agent_v2.md`
- segment evidence pack：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_anchor_hardened_611_2_5c87449_20260608/analysis/segment_evidence`
- 本地总览图：`/Users/lostgreen/Downloads/611-2_all_segments_sheet_20260608.jpg`

后续验证 replay：

- run root：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_locator_verify_611_2_0417b13_20260608`
- commit：`0417b13 fix(video): route locator anchors into verifier`
- 状态：`max_rounds_reached`
- summary：`choice=""`, `selected_option=None`, `correct=False`

最新验证 replay：

- run root：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_anchor_bound_ordered_asr_611_2_e44f79b_20260608`
- commit：`e44f79b fix(video): keep target anchors segment-bound`
- 状态：`final`
- summary：`choice="C"`, `ground_truth="D"`, `correct=False`, `rounds=3`
- 工具序列：`read_timeline_sorted`, `read_timeline_sorted`, `caption_segment`, `locate_targets_in_segment(seg_0002)`, `locate_targets_in_segment(seg_0003)`
- 新失败指纹：已经定位到 `seg_0002`，但 ordered-list timeline 把 509s 的 `Apollo and Daphne` 预告句排到四件作品清单前面，导致 runtime timeline decision 直接 final 为 C。

旧的无 hard-skill / open-rewrite trajectory 只作为历史对照；本分析以当前 run 为准。

## Replay 摘要

这次 replay 不是正常闭环。

- 评测状态：`max_rounds_reached`
- `selected_option`: `None`
- `accuracy`: `0.0`
- planner 最后一轮输出了自然语言排序，但没有输出 A/B/C/D 选项字母。
- 工具调用中没有出现 `verify_segment_anchors`。
- 主要工具循环：`locate_targets_in_segment(seg_0002/seg_0003)`、`read_timeline_sorted`、`caption_segment`。

典型轮次：

```text
R03 locate_targets_in_segment(seg_0002), locate_targets_in_segment(seg_0003)
R04 locate_targets_in_segment(seg_0002), locate_targets_in_segment(seg_0003)
R05 locate_targets_in_segment(seg_0002), locate_targets_in_segment(seg_0003)
R06 locate_targets_in_segment(seg_0002), caption_segment(seg_0002)
...
R20 final, but answer is not an option letter
```

planner 的 rationale 多次写到“verify temporal order / confirm with visual evidence”，但 program 层仍继续调用 locator，没有进入 verifier。

## 视频与字幕事实

原始视频：`H8fGd3fCJbg.mp4`

原始字幕：`H8fGd3fCJbg.srt`

关键字幕窗口：

```text
530.0-539.1s:
radical and colossal marble statues
"Aeneas, Anchises, and Ascanius fleeing Troy", "David", "The rape of Persephone",

539.3-546.0s:
and "Apollo and Daphne".
```

这段在 `seg_0002` 内一次性列出四件作品，顺序是：

```text
Aeneas, Anchises, and Ascanius fleeing Troy
David
The rape of Persephone
Apollo and Daphne
```

这正好对应选项 D。

后续字幕窗口：

- `820.1s` 起进入 David 的具体讲解。
- `1014.5s` 起进入 Apollo and Daphne 的具体讲解。
- `1200s` 以后继续 Apollo and Daphne 的细节。

因此后续 David / Apollo 的详细展示是 explanation sequence，不应该覆盖 `530s` 同一场景列清单的顺序。

## 当前交互问题

### 1. locator 找到 anchor，但 planner 看不到可执行的 anchor 参数

`locate_targets_in_segment(seg_0002)` 返回的 observation 内部有 `regions[0].anchors_for_vlm`：

```text
anchor_0001: 395.18-417.0s, targets=[Apollo and Daphne]
anchor_0002: 504.4-544.097s, targets=[Apollo and Daphne, David, The rape of Persephone]
```

但 planner-visible ledger 只显示 claim 字符串：

```text
Candidate anchors: Apollo and Daphne@400.2s, Apollo and Daphne@404.4s,
Apollo and Daphne@509.4s, David@530.0s, The rape of Persephone@530.0s.
```

`anchors_for_vlm` 的 JSON 没有作为可复制参数暴露给 planner。于是 planner 虽然被提示“call verify_segment_anchors on anchors_for_vlm”，但没有一个稳定参数包可以填进：

```json
{"tool": "verify_segment_anchors", "args": {"segment_id": "...", "anchors": [...]}}
```

结果就是重复 locate。

### 2. `assign` 不是 durable reference

planner 多次写：

```json
"assign": "anchor_0002"
```

但这个 assign 没有变成后续可引用的 runtime object。planner 不能写：

```json
"anchors": "$anchor_0002"
```

也不能写：

```json
"anchor_ref": "obs_0005#anchor_0002"
```

所以 `assign` 给了“我已经拿到 anchor”的错觉，但 runtime 没有可解析引用。

### 3. target 集合不稳定

R03 对 `seg_0002` 的 locate targets 是：

```text
Apollo and Daphne
David
The rape of Persephone
```

漏掉了：

```text
Aeneas, Anchises, and Ascanius fleeing Troy
```

虽然 `target_coverage` 已经知道 T1 在 `seg_0002`。这会让 T1 只能作为其他 target snippet 的上下文残留出现，而不是一等 candidate。

Temporal MCQ 里，locator 应该默认继承完整 T1-T4 target set，除非 planner 显式说明要做局部排除。

### 4. locator 把“最早提及”当成“顺序证据”

`Apollo and Daphne` 在 `400s`、`509s` 有预告性提及，所以 planner 最终把 Apollo 排到第一。

但题目问的是：

```text
in what order ... four masterpieces ... in a single scene
```

对这类问题，更强的证据不是每个 target 的 earliest mention，而是同一短窗口内同时包含多个 target 的 ordered list mention。

`530.0-546.0s` 的字幕是一个 list-order candidate，应该优先级高于分散的单 target mention。

### 5. 长段 caption 继续产生负例污染

`caption_segment(seg_0004)` 的 claim 说 900-1200s 没有任何 target artwork。

但关键帧和字幕显示：

- `903s` 仍在 David 讲解。
- `1014.5s` 开始 Apollo and Daphne。
- 视觉帧也能看到 David / Apollo and Daphne 相关雕塑。

这说明 300s 长 clip caption 仍然不适合做 target absence evidence。它可以当开放描述，但不应该作为“没有目标项”的强证据进入 timeline。

## 最新 replay 进一步确认的问题

`0417b13` 已经解决“locator 不进入 verifier”的一部分问题，但 replay 显示新的关键失败点属实。

### 6. anchor-bound verifier 被 state machine 换段

Round 4 的调用是正确的：

```text
verify_segment_anchors(seg_0002)
anchors.segment_id = seg_0002
```

但后续 round 因为 `avoid_repeated_segment / segment_pool_exhausted`，runtime 把 planner 期望继续验证的 seg_0002 anchor 改派到其他 segment：

```text
verify_segment_anchors(seg_0005)
anchors.segment_id = seg_0002
window = 395-417s / 492-544s

verify_segment_anchors(seg_0006)
anchors.segment_id = seg_0002

verify_segment_anchors(seg_0007)
anchors.segment_id = seg_0002
```

这些调用拿 seg_0002 的字幕/时间窗去看 seg_0005/6/7，必然产出假 negative。然后系统又把这些失败当成“目标没有确认”，进一步耗尽 segment pool。

结论：`verify_segment_anchors` 是 anchor-bound tool，不应进入普通 media segment 替换逻辑。anchor 自带 `segment_id` 后，只能在该 segment 验证。

### 7. ordered-list 仍未升级成可排序证据

`seg_0002` 的 ASR 在一个短窗口内列出完整四项，实际已经足够支持 MCQ D。但当前 locator 的 `ordered_list_navigation` 仍主要停留在 navigation 层，后续 VLM verifier 又以视觉确认方式拆成多个 target 做 yes/no。

这会导致：

- ASR 清单顺序没有 materialize 成 timeline rows；
- `read_timeline_sorted` 早期读到空 timeline；
- AnswerAgent/ledger verifier 看不到“同一 transcript span 的 ordered list”这种证据形态。

### 8. ledger verifier 支持了无关答案

后半段多次 `verify_ledger_answer` 支持类似答案：

```text
The video segment order is seg_0001, seg_0002, ...
```

这不是原题要求的 A/B/C/D，也不是四件作品顺序。其 lexical score 可到 0.62，但：

- `temporal_order_verdict` 是 Neutral；
- `option_relations` 为空；
- `selected_option` 仍是 None。

结论：MCQ verifier 必须拒绝非选项答案。否则会把“片段顺序描述”误判成 supported。

## 建议修改

## 已实施修复

本节记录 2026-06-08 后续开发已落地的修复，避免后续把旧问题和新问题混在一起。

### 1. locator 输出可执行 verifier 参数

`locate_targets_in_segment` 现在在顶层返回：

```json
{
  "verify_call_args": {
    "segment_id": "seg_0002",
    "anchors": ["..."],
    "targets": ["..."]
  }
}
```

`recommended_next_tools[0].args` 复用同一个 `verify_call_args`，避免 raw 输出和推荐调用不一致。

### 2. compact ledger 暴露 `verify_call_args`

`compact_ledger_text()` 会用 `observations.jsonl` 回填 navigation observation 的 `raw_output`，并在 `locate_targets_in_segment` 的 Navigation Summary 里显示：

```text
next: verify_segment_anchors verify_call_args={...}
```

planner 不再只能看到 `Candidate anchors: target@time` 的自然语言摘要。

### 3. timeline route 重复 locator 自动修到 verifier

当 active skill 是 `timeline_ordering`，且 planner 在已有 locator anchor 后再次请求同一个 `locate_targets_in_segment`，runtime 会把该 tool call normalize 成：

```text
verify_segment_anchors(...)
```

trace reason：

```text
repair_repeated_locator_to_verify_segment_anchors
```

这直接针对 replay 中 R03-R05 反复 locate、不进入 verify 的空转。

### 4. locator 增加 ordered-list navigation candidate

locator 现在会检查相邻 ASR/OCR 文本窗口。如果同一短窗口内出现 3 个以上 target，会生成：

```json
{
  "match_type": "ordered_list_mention",
  "ordered_targets": ["..."]
}
```

对 611-2，`530.0-546.0s` 这种字幕清单会成为高优先级导航候选，而不是被拆成若干 earliest mention。

### 5. target set 自动补全

如果 planner 传入了局部 targets，但 workspace 里已有 `target_coverage` 的完整 T1-T4，`locate_targets_in_segment` 会把显式 targets 与 coverage targets 取并集。

这修复了当前 replay 中 `seg_0002` locate 漏掉 `Aeneas, Anchises, and Ascanius fleeing Troy` 的问题。

### 6. planner final 的自然语言顺序可映射回 MCQ

如果 planner final 没有以 A/B/C/D 开头，但答案文本里的实体顺序唯一匹配某个选项，runtime 会把 final answer 改写成：

```text
D. <原自然语言顺序答案>
```

该 fallback 很保守：必须能从答案文本中找到完整事件序列，且只能唯一匹配一个选项。

### 当前验证

本地全量测试：

```text
PYTHONPATH=src:. pytest -q
390 passed
```

本轮相关模块回归：

```text
PYTHONPATH=src:. pytest -q tests/test_route_validator.py tests/test_video_navigation.py tests/test_caption_qa_tools.py tests/test_verification_tools.py tests/test_timeline.py tests/test_iterative_agent.py
154 passed
```

新增/覆盖测试：

- locator 输出 `verify_call_args`；
- locator 自动并入 target coverage；
- compact ledger 暴露 `verify_call_args`；
- timeline 重复 locator 自动 repair 到 `verify_segment_anchors`；
- 自然语言 temporal answer 唯一映射回 MCQ option；
- ordered-list candidate 识别同窗口 target 顺序。
- anchor-bound verifier 不再被 `avoid_repeated_segment` 换段；
- ordered-list ASR 写入 confirmed indexed transcript timeline rows；
- interactive loop 可从 ordered-list timeline 直接 final；
- MCQ ledger verifier 拒绝非选项答案。

### 7. anchor-bound verifier 不再替换 segment

`verify_segment_anchors` 现在走单独的 normalize 分支，不再进入普通 `_resolve_media_segment()` 的 `avoid_repeated_segment` 替换逻辑。

规则：

- anchor 带 `segment_id` 时，以 anchor 的 `segment_id` 为准；
- planner 传错 `segment_id` 时，runtime 修回 anchor segment，并记录 `repair_verify_anchor_segment_id_from_anchor`；
- anchors 来自多个 segment 时，该调用被拒绝，要求按 source segment 分开验证；
- tool 层也增加硬校验：`verify_segment_anchors(seg_0005, anchors from seg_0002)` 会直接抛出 mismatch。

### 8. ordered-list ASR 升级为 indexed transcript timeline

`locate_targets_in_segment` 仍然是 navigation-only，不写 answer evidence table。但当它发现同一短 ASR/OCR 窗口包含 ordered target list 时，会额外返回：

```json
"ordered_list_timeline_rows": [
  {
    "entity": "Aeneas, Anchises, and Ascanius fleeing Troy",
    "observed_at_sec": 529.0,
    "confidence_signal": "confirmed",
    "grounding_quality": "indexed_transcript"
  }
]
```

workspace 会把这些 rows 写入 `timeline.md`。这条路径的含义是：

- locator 仍不直接支持最终答案；
- ordered ASR/OCR list 可以作为可排序 transcript timeline；
- VLM verifier 后续只负责补视觉确认，不负责否定 ASR list。

### 9. interactive loop 可直接使用 timeline 决策

普通 planner loop 在工具执行后会检查 `read_timeline_sorted()`。若 confirmed timeline rows 唯一匹配某个 MCQ temporal option，会直接 final：

```text
iterative_timeline_temporal_decision
source = interactive_loop
answer = D
```

同时修复了两个匹配细节：

- 同一个 observation 可以写多条 timeline rows，不能再用 `obs_id` 去重阻止四项清单匹配；
- 选项中引号包裹的艺术品名按整体事件处理，避免逗号切碎 `Aeneas, Anchises, and Ascanius fleeing Troy`。

### 10. MCQ verifier 拒绝非选项答案

`verify_ledger_answer` 现在在 MCQ 场景下要求 answer 能解析出选项字母。否则返回：

```text
insufficient: MCQ answer must begin with option letter
```

runtime 也会给 answer-facing verifier 注入原始 question / candidate_options，保证它能识别 MCQ 约束。

### 11. ordered-list 只按清单子串排序，不吃前置预告句

`e44f79b` replay 证明上一版 `ordered_list_timeline_rows` 太激进：它确实把 ASR 清单升级成 timeline evidence，但排序来源仍是长 ASR sentence 的全文首次出现位置。

真实错误形态：

```text
497.1-539.1s:
... same attention to detail that we will see with Apollo and Daphne.
... radical and colossal marble statues "Aeneas...", "David", "The rape of Persephone",

539.3-546.0s:
and "Apollo and Daphne".
```

旧逻辑把第一句的 unquoted `Apollo and Daphne` 当成 ordered list 的第一项，得到：

```text
Apollo and Daphne -> Aeneas -> David -> The rape of Persephone
```

但题目问的是：

```text
four masterpieces created for Borghese in a single scene
```

更强证据是 quoted list 子串：

```text
"Aeneas...", "David", "The rape of Persephone", and "Apollo and Daphne"
```

因此 ordered-list detector 现在：

- 优先从引号内目标标题抽取顺序；
- 用第一个 quoted target 的估计时间作为 list window 起点，而不是长 ASR sentence 的 497s 起点；
- 若已有 quoted list 子串，允许继续合并紧邻字幕里的最后一个 quoted target；
- 候选排序优先 `quoted_target_count` 和紧凑 target span，再考虑起始时间；
- fallback 才使用全文 target 首次出现顺序。

这让 611-2 的 indexed transcript timeline 恢复为：

```text
Aeneas, Anchises, and Ascanius fleeing Troy
David
The rape of Persephone
Apollo and Daphne
```

对应选项 D。

当前验证：

```text
PYTHONPATH=src:. pytest -q
391 passed

PYTHONPATH=src:. pytest -q tests/test_video_navigation.py tests/test_iterative_agent.py tests/test_timeline.py tests/test_route_validator.py tests/test_caption_qa_tools.py tests/test_verification_tools.py
155 passed
```

## 历史建议对照

以下 A-F 是本轮实施前整理的设计建议，保留作对照：

- A-D 已由 `verify_call_args`、重复 locator repair、ordered-list candidate、target set 自动补全覆盖；
- F 已由 planner final option mapping 和 MCQ verifier gate 覆盖；
- E 仍作为后续质量约束：长段 caption 的 absence claim 不应升级为强反证。

### A. 让 locator 输出可执行 verifier 参数

`locate_targets_in_segment` 的 planner-visible claim/nav_digest 里应该包含短 JSON：

```json
"verify_call_args": {
  "segment_id": "seg_0002",
  "anchors": [
    {
      "anchor_id": "anchor_0002",
      "segment_id": "seg_0002",
      "start_sec": 504.4,
      "end_sec": 544.097,
      "targets": [
        "Aeneas, Anchises, and Ascanius fleeing Troy",
        "David",
        "The rape of Persephone",
        "Apollo and Daphne"
      ]
    }
  ]
}
```

或者新增 runtime reference：

```json
{"tool": "verify_segment_anchors", "args": {"anchor_ref": "obs_0005#anchor_0002"}}
```

由 runtime 自动展开 anchor JSON。

### B. timeline route 下强制 locate -> verify

当 active skill 是 `timeline_ordering`，且上一轮 locator 已经返回 anchors：

- 下轮工具 schema 可以暂时隐藏 `locate_targets_in_segment`。
- 或者同一 segment/target hash 的 locate 标记为 exhausted。
- 推荐工具只保留 `verify_segment_anchors` / `read_timeline_sorted` / `verify_ledger_answer`。

这样 planner 不会在 navigation 层空转。

### C. locator 增加 ordered-list candidate

对 ASR/OCR 文本做一个轻量结构识别：

- 同一字幕窗口或相邻字幕窗口内包含 >=3 个 target。
- 按文本出现位置给出 `ordered_targets`。
- candidate type 标成 `ordered_list_mention`。
- 对包含所有 target 的短窗口提升优先级。

对 611-2，应该生成：

```json
{
  "segment_id": "seg_0002",
  "start_sec": 525.1,
  "end_sec": 546.0,
  "ordered_targets": [
    "Aeneas, Anchises, and Ascanius fleeing Troy",
    "David",
    "The rape of Persephone",
    "Apollo and Daphne"
  ],
  "match_type": "ordered_list_mention"
}
```

### D. target set 自动补全

`locate_targets_in_segment` 对 temporal MCQ 应该把 planner 传入 targets 与全局 T1-T4 取并集，或者至少在返回里提示：

```text
planner omitted target(s): Aeneas, Anchises, and Ascanius fleeing Troy
```

否则 planner 可能把完整排序题变成局部排序题。

### E. caption absence 降权

`caption_segment` 对“未看到 target”的 claim 不应作为 strong absence evidence。

建议：

- 只有 focused verifier 可以产出 target rejection。
- 长段 caption 的 negative target 结论标成 `unsupported_absence`。
- `read_timeline_sorted` 不应把这种负例当成 timeline 反证。

### F. MCQ final gate 必须输出选项字母

VideoMME runner 的 MCQ final gate 应硬性要求：

```text
final answer must begin with exactly one option letter A/B/C/D
```

如果 planner 输出自然语言顺序，AnswerAgent 应把它和原始选项比较，而不是让 `selected_option=None` 进入 summary。

## 结论

611-2 的主要失败不是 target extraction 已经完全错，也不是 locator 完全没找到证据；失败点是交互协议和证据升级：

```text
locator candidates 找到了
-> 旧版：anchor JSON 不可复制，planner 反复 navigation
-> 0417b13：进入 verifier，但 anchor 被换到错误 segment
-> e44f79b：anchor-bound 和 timeline evidence 生效，但 ordered-list detector 把长 ASR 句里的前置 Apollo 预告当作清单第一项
```

本轮已完成：

1. `verify_segment_anchors` anchor-bound，不再被 `avoid_repeated_segment` 换段；
2. ordered-list ASR 写入 confirmed indexed transcript timeline rows；
3. ordinary interactive loop 可从 timeline 唯一匹配 MCQ option 后直接 final；
4. `verify_ledger_answer` 拒绝非选项 MCQ answer。
5. ordered-list detector 优先按 quoted list 子串排序，不再把 unquoted 前置预告句混入清单顺序。

下一步挂新版 611-2 replay，观察是否能从 `seg_0002` 的 quoted ordered ASR list 直接收敛到 D。
