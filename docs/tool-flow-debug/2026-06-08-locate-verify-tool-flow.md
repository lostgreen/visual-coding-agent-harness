# locate / verify 工具流程调试记录

日期：2026-06-08

## 当前目标

新版段内目标定位链路统一为：

```text
Layer 0: Compact scene index + asr mentions
Layer 1: locate_targets_in_segment
Layer 2: verify_segment_anchors
```

这里刻意不再使用 `scan_segment`。`scan_segment` 的旧想法把“找候选”和“确认事实”混在一个 VLM 扫描工具里，容易把候选当证据写进 timeline。当前版本把 candidate 与 evidence 分开：Layer 1 只负责定位候选窗口，Layer 2 才负责验证并写证据。

## Layer 0：prompt 内地图

输入来源：

- `SceneIndex` 的每段 `map_summary`。
- VideoMME MCQ rewrite 得到的 `target_entities`。
- dual-source builder 保留下来的 `asr_sentences`。

planner 看到的输出形态：

```text
seg_0004 [900.0-1200.0s] David and Borghese sculpture comparison
  asr mentions: David @ ~930.0s
seg_0005 [1200.0-1500.0s] Apollo and Daphne details
  asr mentions: Apollo and Daphne @ ~1240.0s
```

语义：

- 这是 routing hint，不是 evidence。
- 目的是让纯文本 planner 不用先调 `video_ls`，也能大致知道该读哪段。
- `target_hints` 只用于渲染 ASR mention，不把 A/B/C/D option 顺序泄漏给 local VLM。

## read_segment_detail：cheap/index detail pack

planner 调用：

```json
{"tool": "read_segment_detail", "args": {"segment_id": "seg_0005", "targets": ["Apollo and Daphne"]}}
```

runtime 输出字段：

```json
{
  "segment_id": "seg_0005",
  "visual_caption": "...",
  "asr_summary": "...",
  "raw_asr_excerpt": "...",
  "ocr_text": "...",
  "entities": ["..."],
  "target_hits": ["..."],
  "target_matches": ["..."],
  "unmatched_targets": ["..."],
  "nav_digest": "targets=Apollo and Daphne | visual=... | asr=...",
  "recommended_next_tools": ["vision_read / locate_targets_in_segment / verify_segment_anchors"]
}
```

关键修复：

- `read_segment_detail` 不会被 normalize 成 `caption_segment`。
- 如果 planner 没传 `targets`，工具会从最近的 `target_coverage` observation 或 rewrite metadata 继承 target list。
- compact ledger 的 `Navigation Summary` 会展开 `claim/nav_digest`，所以后续轮次能看到视觉 caption、字幕摘要和 target 命中，而不是只看到 `obs_xxx: read_segment_detail`。

## Layer 1：locate_targets_in_segment

planner 调用：

```json
{
  "tool": "locate_targets_in_segment",
  "args": {
    "segment_id": "seg_0005",
    "targets": ["Aeneas, Anchises, and Ascanius fleeing Troy", "David", "The rape of Persephone", "Apollo and Daphne"]
  }
}
```

输入来源：

- `asr_sentences`：优先，因为有句级时间戳。
- `ocr_frames`：如果有帧级 OCR。
- `visual_caption` / `entities`：兜底文本源。

输出字段：

```json
{
  "targets": ["..."],
  "candidates": [
    {
      "target": "Apollo and Daphne",
      "source": "asr_sentence",
      "snippet": "...Apollo and Daphne...",
      "start_sec": 1240.0,
      "end_sec": 1248.0,
      "score": 0.95
    }
  ],
  "anchors_for_vlm": [
    {
      "segment_id": "seg_0005",
      "start_sec": 1235.0,
      "end_sec": 1253.0,
      "targets": ["Apollo and Daphne"],
      "reason": "asr_sentence match: ..."
    }
  ],
  "recommended_next_tools": ["verify_segment_anchors"],
  "limitations": "Text-only locator; does not confirm visual presence."
}
```

语义：

- 它是文本 locator，不读像素，不做 yes/no 视觉验证。
- 它写入 navigation candidate 桶，不进入 AnswerAgent 的最终证据集合。
- 匹配使用严格 token/word-boundary 逻辑，避免把作品名中的逗号拆成多个事件。

## Layer 2：verify_segment_anchors

planner 调用：

```json
{
  "tool": "verify_segment_anchors",
  "args": {
    "segment_id": "seg_0005",
    "anchors": [{"start_sec": 1235.0, "end_sec": 1253.0, "targets": ["Apollo and Daphne"], "reason": "..."}],
    "targets": ["Apollo and Daphne"]
  }
}
```

local VLM prompt 语义：

- 使用 anchor 的 reason 作为上下文，但不让 local VLM 选择 A/B/C/D。
- 要求输出实际可见/旁白/字幕/OCR 的确认与拒绝。
- 默认 8 帧 focused read，而不是对整段 300s 做粗 caption。

解析后的输出字段：

```json
{
  "confirmations": [
    {
      "target": "Apollo and Daphne",
      "status": "confirmed",
      "observed_at_sec": 1242.0,
      "evidence_type": "narrated_and_visual",
      "claim": "..."
    }
  ],
  "rejections": [
    {
      "target": "David",
      "reason": "not shown in this anchor window"
    }
  ],
  "timeline_rows": [
    {
      "entity": "Apollo and Daphne",
      "observed_at_sec": 1242.0,
      "claim": "..."
    }
  ]
}
```

语义：

- `confirmations` 写入 evidence table。
- `timeline_rows` 写入 `timeline.md`，供 `read_timeline_sorted` 和 AnswerAgent 使用。
- `rejections` 只作为调试/消歧信息，不当作目标不存在的全局结论。

## runtime 约束

- planner-visible schema 会列出 `locate_targets_in_segment` 与 `verify_segment_anchors`，不再列出 `zoom` / `expand_window`。
- active-skill schema 继续过滤工具：timeline / needle 路由只看到与当前 skill 匹配的工具子集。
- planner 若仍调 `zoom` / `expand_window`，runtime 软重写为 `locate_targets_in_segment(segment_id, targets=<inherited>)`。
- `locate_targets_in_segment` 在 AnswerAgent 和 skill predicate 中都被视为 navigation-only。
- 最终答案需要 evidence-grade observation：`vision_read`、`inspect_segment`、`caption_segment`、`qa_segment` 或 `verify_segment_anchors`。

## 本轮验证

本地测试：

```bash
PYTHONPATH=src:. pytest -q
# 378 passed
```

覆盖的关键断言：

- `read_segment_detail` 的 digest 会进入 compact ledger。
- `locate_targets_in_segment` 输出 candidates / anchors，但不写 evidence table。
- `verify_segment_anchors` 解析 confirmations / rejections / timeline_rows，并把确认项写入 evidence 与 timeline。
- `Compact scene index` 可渲染 target ASR mentions。
- `zoom` / `expand_window` 在 timeline skill 下会软重写到 locator。
- `scan_segment` 不出现在 planner prompt 或执行路径中。

## 下一步建议

在 KML 上复跑 611-2 时，重点看三类字段即可，不要拉 raw log：

- trajectory 中是否出现 `locate_targets_in_segment -> verify_segment_anchors`。
- `Navigation Summary` 是否含 `read_segment_detail` 的 `nav_digest`。
- `timeline.md` 是否只出现 `verify_segment_anchors` 的 confirmed rows，而不是 locator candidates。
