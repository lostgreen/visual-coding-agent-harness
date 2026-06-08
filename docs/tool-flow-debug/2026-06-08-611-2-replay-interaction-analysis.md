# 611-2 replay 交互轮次问题分析

日期：2026-06-08

## 当前证据

当前有效 replay：

- run root：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_anchor_hardened_611_2_5c87449_20260608`
- trajectory：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_anchor_hardened_611_2_5c87449_20260608/trajectories/611-2_agent_v2.json`
- markdown trajectory：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_anchor_hardened_611_2_5c87449_20260608/trajectories/611-2_agent_v2.md`
- segment evidence pack：`/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_anchor_hardened_611_2_5c87449_20260608/analysis/segment_evidence`
- 本地总览图：`/Users/lostgreen/Downloads/611-2_all_segments_sheet_20260608.jpg`

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
386 passed
```

新增/覆盖测试：

- locator 输出 `verify_call_args`；
- locator 自动并入 target coverage；
- compact ledger 暴露 `verify_call_args`；
- timeline 重复 locator 自动 repair 到 `verify_segment_anchors`；
- 自然语言 temporal answer 唯一映射回 MCQ option；
- ordered-list candidate 识别同窗口 target 顺序。

### 仍需观察

这轮没有把 `locate_targets_in_segment` 本身升级成 answer evidence。它仍然是 navigation-only。

原因：保持边界清楚，避免 text locator 候选直接变成最终证据。若新版 611-2 replay 仍卡在“字幕清单顺序无法被 verifier/AnswerAgent 使用”，下一步应单独增加 `timeline_asr_summary / indexed_transcript` 证据写入，而不是让 locator 偷渡 evidence。

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

611-2 的主要失败不是 target extraction 已经完全错，也不是 locator 完全没找到证据；当前失败点是交互协议：

```text
locator candidates 找到了
-> anchor JSON 没有 planner-visible / referenceable
-> verifier 没被调用
-> planner 反复 navigation
-> 最后用 earliest mention 做自然语言排序
-> 没有转成 MCQ option
```

下一步应优先修：

1. locator 输出 `verify_call_args` 或 `anchor_ref`；
2. timeline route 下 locate 后强制/偏置进入 verifier；
3. locator 增加 ordered-list candidate；
4. final gate 把自然语言顺序映射回 A/B/C/D。
