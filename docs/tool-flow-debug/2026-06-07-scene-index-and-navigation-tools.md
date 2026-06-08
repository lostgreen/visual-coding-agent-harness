# Scene Index 与导航工具流程调试

日期：2026-06-07

> 2026-06-08 覆盖说明：最新段内目标定位链路已统一为
> `locate_targets_in_segment -> verify_segment_anchors`。本文保留 2026-06-07 的调试脉络；
> 最新工具输入输出与 runtime 契约见
> `docs/tool-flow-debug/2026-06-08-locate-verify-tool-flow.md`。

## 目标

记录 planner 看到的 `Compact scene index`、`video_ls`、`search_segments`、`read_segment` 这几类导航/索引信息的输入输出流程，并解释当前 611-2 中为什么只看到 `ASR/subtitle excerpt` 以及为什么文本里有省略号。

## 原始现象

planner prompt 里出现的片段：

```text
Compact scene index:
seg_0001 [0.0-300.0s] ASR/subtitle excerpt: Just a quick reminder ...
seg_0002 [300.0-600.0s] ASR/subtitle excerpt: believable flesh out of marble. ...
...
```

直观看起来有两个问题：

1. 每段文字末尾有 `...`。
2. 每段似乎只有一种信息：`ASR/subtitle excerpt`，没有独立的 Visual caption、Tags、Entities。

2026-06-07 晚间又出现了第二类现象：

```text
seg_0001 [0.0-300.0s] Visual: ... | ASR: { "summary": ... } | Tags: ... | Entities: ...
```

这说明 dual-source 信息已经进入 scene index，但默认 `Compact scene index` 仍在展开完整 `visual_caption/asr_summary/tags/entities`。这不符合当前三层设计：planner 每轮默认只应该看到 Layer 0 的一句话地图，完整 caption / ASR 应该通过按需 segment detail 工具读取。

对应修复：

- 提交方向：给 `VideoSegment` 增加 `map_summary`。
- `SceneIndex.summary()` 只输出 `map_summary` 或旧 fallback caption，不再默认展开 `Visual:` / `ASR:` / `Tags:` / `Entities:`。
- dual-source builder 先生成 visual caption 和 subtitle summary，再用文本模型生成 `summarize_scene_map_segment` 的一句话 map。
- `summarize_scene_map_segment` 走 text backend，不走 VLM backend。
- scene-index cache schema 从 `dual_source_scene_index_v2` bump 到 `dual_source_scene_index_v3`，避免复用旧缓存。
- 清洗嵌套 JSON 字符串，例如 `ASR: { "summary": "..." }` 会抽成纯文本 summary。

2026-06-07 第四轮机制修复：

- `caption_segments` / `ingest_segment_metadata` 不再出现在 planner-visible schema；它们只保留在 registry 中供离线缓存/debug 使用。
- `verify_ledger_answer` 不再向 planner 暴露 `ledger_text` 参数；normalization 会剥掉 planner 传入的 `ledger_text`，强制 verifier 读 workspace ledger。
- option-blind MCQ rewrite 得到 `target_entities` 后，会在第一轮 planner 前自动执行一次 `target_coverage`。
- compact ledger 的 `Navigation Summary` 会展开 `target_coverage` 的短 claim，因此 Layer 1 target coverage matrix 每轮都会进入 planner prompt。
- 这轮还没有做 local VLM target-aware prompt，也没有给 evidence row 增加 `target_id`。

2026-06-07 第五轮机制修复：`read_segment_detail` 真正成为 cheap/index detail pack。

用户指出的当前新现象：

```text
planner 请求：
read_segment_detail(seg_0002)

runtime 实际执行：
caption_segment / vision_read(seg_0002, question="Openly describe ...")
```

这不是重写问题本身，而是工具设计语义不一致：

- planner 以为 `read_segment_detail` 是“读取已有索引细节”。
- runtime 后续策略把“导航工具后需要视觉证据”实现成了替换整轮 program。
- 结果 cheap/index detail 没有进 ledger，planner 拿不到完整 `visual_caption/asr_summary/entities/target_hits`。

本轮修复边界：

- 不恢复 hard-skill runtime。
- 不把 `video_ls` 放回 planner default。
- 不把 `read_segment_detail` 变成 VLM caption。
- 保留旧的“空 `read_segment` 可升级成 `caption_segment`”行为；这个旧逻辑只针对没有索引文本的 `read_segment`，不适用于 `read_segment_detail`。

修复后的 `read_segment_detail(segment_id, targets=())` 输出语义：

```json
{
  "segment_id": "seg_0002",
  "start_sec": 300.0,
  "end_sec": 600.0,
  "visual_caption": "...",
  "asr_summary": "...",
  "raw_asr_excerpt": "...",
  "ocr_text": "...",
  "entities": ["..."],
  "target_hits": [
    {
      "target": "Apollo and Daphne",
      "matched": true,
      "fields": ["asr_text", "low_fps_caption"],
      "matches": [
        {
          "field": "asr_text",
          "modality": "asr",
          "evidence": "...",
          "score": 0.67
        }
      ]
    }
  ],
  "target_matches": [
    {
      "target": "Apollo and Daphne",
      "source": "asr_text",
      "snippet": "...",
      "score": 0.67,
      "directness": "possible_mention"
    }
  ],
  "unmatched_targets": ["David"],
  "recommended_next_tools": [
    {
      "tool": "vision_read",
      "args": {
        "segment_id": "seg_0002",
        "start_sec": 300.0,
        "end_sec": 600.0,
        "ask_for": "Openly verify ..."
      }
    }
  ]
}
```

当 planner 没有显式传 `targets` 时，工具会从 workspace 中最近一次 `target_coverage` observation 继承 target list。这样 MCQ 自动 seed 的 Layer 1 可以自然流入 Layer 2，不需要 planner 重新拼 target 参数。

agent loop 的修复：

- 旧策略：如果当前 program 全是 navigation，并且策略认为还需要视觉证据，就把整轮 program 替换成 forced visual tool。
- 新策略：保留原 navigation program，在本轮工具预算允许时追加一个 visual follow-up。
- 因此 Round 2 如果 planner 请求 `read_segment_detail(seg_0002)`，实际 program 会变成：

```json
[
  {"tool": "read_segment_detail", "args": {"segment_id": "seg_0002"}},
  {"tool": "vision_read", "args": {"segment_id": "seg_0002", "...": "..."}}
]
```

这满足当前设计：planner 先读已有索引 detail pack，Agent 再验证 planner 期望获取的局部事实，而不是 Agent 自己把 detail 请求改成另一个任务。

2026-06-07 第六轮 P0 修复：把地图到 detail/grounding 的链路继续接上。

这轮依据新的 compact 建议，明确当前问题已经不是“没有地图”，而是“地图没有稳定转成可执行、可验证、可匹配的细节任务”。

旧 run 状态：

- `e5f68bd` 611-2 KML run 已生成 `summary.json` 和 `611-2_agent_v2.{json,md}`。
- 该 run 仍属于第五轮旧行为：可以用来对照，但不能代表本轮第六轮修改。
- 本轮未拉取 raw trajectory / raw log 进入上下文，只记录 artifact 存在状态。

本轮 P0 改动：

1. `read_segment` 在 timeline / grounded QA / mutex QA route 下不再被 drop，也不再被空索引逻辑抢修成 `caption_segment`。
   - 如果 planner 调 `read_segment(seg_x)`，runtime 会 normalize 为 `read_segment_detail(seg_x, targets=target_entities)`。
   - 这样 Round 1 planner “想读 segment”的意图会被保留。

2. `target_coverage` 的候选 segment 现在带可操作字段：
   - `source`：最佳命中字段，例如 `low_fps_caption` / `asr_text`。
   - `snippet`：最佳命中的短证据片段。
   - `directness`：`direct_mention` / `possible_mention` / `weak_overlap`。
   - 这仍然只是 index recall，不是最终 grounding；planner 应继续用 `read_segment_detail` 和 visual tool 验证。

3. local visual tool query 改成 target-aware open grounding：
   - 只传 unordered `target_entities`。
   - 不传 A/B/C/D，不传 option order。
   - prompt 要求对 target-like item 报告 direct visual / narrated / OCR / visually similar、局部 timestamp/order、visible cue，并继续报告其他 artwork/transition。

示例形态：

```text
Openly describe this segment's actual visible artworks...

Pay special attention to these unordered target artwork names or aliases if they appear:
- Aeneas, Anchises, and Ascanius fleeing Troy
- David
- The rape of Persephone
- Apollo and Daphne

For each target-like item, report whether it is directly shown, narrated, visible as onscreen text, or only visually similar.
Report local timestamp/order, exact text if present, and visible cues. Also report other artworks/transitions in order.
Do not choose or compare options.
```

仍未做的 P1/P2：

- 新增独立 `ground_targets_in_segment` 工具。
- 从 target-aware grounding 中抽结构化 `timeline_event` rows。
- `read_timeline_sorted` 动态隐藏或拆成 structured timeline / evidence windows 两种工具。
- `build_observed_order(target_entities, evidence_rows)` + AnswerAgent option matching。

## 先区分两类东西

`Compact scene index` 不是一次普通工具调用输出。它是每轮 planner prompt 内置的 scene-index 区块。

生成位置：

- `src/visual_coding_agent_harness/agents/prompt_stack.py`
- 函数：`_scene_index_snapshot_block()`
- 调用：`scene_index.summary(max_segments=64)`

导航工具是另一套可由 planner 主动调用的工具：

- `search_segments`
- `target_coverage`
- `ground_question`
- `read_segment`
- `read_segment_detail`
- `expand_window`
- `zoom`

`video_ls` 和 `caption_segments` 仍在 registry/工具实现中，但当前普通 planner schema 不再暴露它们。

这些工具在：

- `src/visual_coding_agent_harness/tools/navigation.py`

两者共享同一个底层索引来源，但输出格式不同。

## Compact Scene Index 的输入输出

输入：

- 当前问题 `question`
- 当前 `SceneIndex`
- 已检查过的 segment ids

prompt 中的输出结构：

```text
Question: ...
Uninspected segment candidates: ...
Compact scene index:
{scene_index.summary(max_segments=64)}
```

`SceneIndex.summary()` 的逻辑在：

- `src/visual_coding_agent_harness/video_index.py`

当前 v3 逻辑：

1. 优先输出 `map_summary`。
2. 如果没有 `map_summary`，fallback 到旧 `low_fps_caption`。
3. 再 fallback 到 `keyframe_path`。
4. 最后输出 `no coarse caption yet`。

它不再默认输出：

- `Visual: {visual_caption}`
- `ASR: {asr_summary}`
- `Tags: ...`
- `Entities: ...`

原先 611-2 看到的是：

```text
ASR/subtitle excerpt: ...
```

这说明当时那批 segment 大概率没有填 `asr_summary` / `visual_caption` / tags / entities，而是只有旧格式的 `low_fps_caption`。

## 省略号来源

省略号不是模型生成的，也不是字幕原文一定自带的。

有两层截断：

1. 旧 subtitle scene index 构建时，每个 300s bucket 会做一次 `compact_text(..., limit=720)`。
2. `SceneIndex.summary(max_caption_chars=240)` 又会对每段展示文本调用 `_bounded_text(..., limit=240)`。

`_bounded_text()` 的行为是：

- 如果文本长度不超过 limit，原样返回。
- 如果超过 limit，截到 `limit - 3`，然后追加 `...`。

因此 prompt 里的 `...` 是为了控制 planner prompt 长度。它表示这段索引文本被压缩过，不代表这个 segment 的字幕/视觉信息只有这么多。

## 为什么原来只看到一种信息

旧 VideoMME runner 默认 scene index mode 是 `subtitle`：

- CLI 参数：`--scene-index-mode`
- 旧默认值：`subtitle`
- 新默认值：`dual-source`
- 保留的显式 fallback/ablation 值：`subtitle`

旧 `subtitle` 路径走：

- `src/visual_coding_agent_harness/evals/videomme/runner.py`
- 函数：`subtitle_scene_index()`

这个函数做的事情很简单：

1. 按固定窗口切 segment，当前 611-2 是 300s 一段。
2. 把每段内的 SRT/subtitle 文本拼起来。
3. 压缩成一个 excerpt。
4. 写进 `VideoSegment.low_fps_caption`：

```text
ASR/subtitle excerpt: {excerpt}
```

也就是说，这条旧路径并没有真正填：

- `asr_summary`
- `visual_caption`
- `stage_tags`
- `entities`
- `topic_tags`

所以 `SceneIndex.summary()` 只能 fallback 到 `low_fps_caption`，结果就只显示 `ASR/subtitle excerpt`。

## 当前默认 dual-source 路径是什么

当前默认 scene-index mode 已切到 `dual-source`，代码位置：

- `src/visual_coding_agent_harness/evals/videomme/scene_index_builder.py`
- 类：`SceneIndexBuilder`

它每个 segment 会做两件事：

1. `summarize_subtitle_segment`
   - 输入：segment 时间范围内的 subtitle cues
   - 输出：`summary`、`entities`、`topic_tags`、`confidence`、`raw_asr_ref`
   - 写入：`asr_summary`、`entities`、`topic_tags`
2. `caption_scene_segment`
   - 输入：video segment clip 或带 metadata 的视频
   - 输出：`caption`、`stage_tags`、`entities`、`grounding_quality`
   - 写入：`visual_caption`、`stage_tags`、`entities`、`grounding_quality`

dual-source 路径下，`SceneIndex.summary()` 应该显示类似：

```text
seg_0001 [...] intro / Bernini biography and early sketches
seg_0002 [...] early sculptures / Apollo and Daphne mention
```

设计语义是：

- Layer 0：每轮默认 prompt 只放一句话 map，供 planner 粗定位。
- Layer 1：target coverage matrix 通过 `target_coverage` 生成；option-blind MCQ rewrite 有 `target_entities` 时会自动前置一次，并进入 compact ledger。
- Layer 2：planner 选择 segment 后，通过 `read_segment_detail` 读取完整 `visual_caption`、`asr_summary`、raw ASR ref、OCR、entities、target hits。
- Caption 是主要视觉内容索引；ASR/subtitle 是额外补充；两者都不再直接拼接进每轮默认 prompt。

已有测试覆盖这个预期：

- `tests/test_scene_index_builder.py`
- 测试：`test_summary_uses_one_line_map_not_full_dual_source_detail`
- `tests/test_iterative_agent.py`
- 测试：`test_compact_scene_index_uses_map_summary_not_full_dual_source_detail`

## video_ls 的输入输出

工具位置：

- `src/visual_coding_agent_harness/tools/navigation.py`
- 函数：`video_ls(query="", max_segments=16, top_k=5)`

输入：

- `query`：可选，用于找候选 segment。
- `max_segments`：outline 最多展示多少段。
- `top_k`：返回多少候选。

输出：

- `claim`
  - 视频总段数、总时长、可用索引类型、候选段。
- `coverage`
  - 每类字段覆盖数，例如 `low_fps_caption`、`asr_text`、`ocr_text`、`entities`。
- `outline`
  - sampled segments 的 compact summary。
- `candidates`
  - query 命中的候选段。
- `recommended_next_tools`
  - 通常包括 `read_segment`、`inspect_segment`、`zoom` 等。
- `raw_video_map`
  - 当前 VideoMap 的结构化内容。

注意：`video_ls` 不直接用 `SceneIndex.summary()`。它先把 `SceneIndex` 转成 `VideoMap`。

转换逻辑：

- `VideoMap.from_scene_index()`

字段映射：

- `low_fps_caption = segment.visual_caption or segment.low_fps_caption`
- `asr_text = segment.asr_summary`
- `entities = segment.entities + topic_tags + stage_tags`

注意：`VideoMap.from_scene_index()` 仍然使用完整 `visual_caption/asr_summary`，不是用 `map_summary`。这是刻意区分：

- `map_summary`：每轮 prompt 的短地图。
- `visual_caption/asr_summary`：按需导航和 detail 工具的完整索引内容。

如果当前是旧 subtitle index：

- `visual_caption` 为空；
- `asr_summary` 为空；
- `entities` 为空；
- 只有 `low_fps_caption = "ASR/subtitle excerpt: ..."`。

所以 `video_ls` 的 coverage 也会显示主要只有 `low_fps_caption`，而不是独立 ASR/Visual 多通道。

## search_segments 的输入输出

工具位置：

- `src/visual_coding_agent_harness/tools/navigation.py`
- 函数：`search_segments(query, top_k=5, modalities=())`

输入：

- `query`：检索词。
- `top_k`：候选数量。
- `modalities`：可选通道，例如 `caption`、`asr`、`ocr`、`entities`。

输出：

- `claim`
  - query 返回了哪些 candidate segments。
- `regions`
  - 每个命中 segment 的时间、score、matched fields、matches、summary。
- `modalities`
  - 按通道分组的检索结果。
- `limitations`
  - 明确说明这是 training-free VideoMap retrieval，只基于 caption/ASR/OCR/entity indexes。

旧 subtitle index 下，因为字幕 excerpt 被放在 `low_fps_caption`，检索时它更像 `caption` 字段，而不是真正的 `asr_text` 字段。这会让工具表面上说 caption，但内容实际是 `ASR/subtitle excerpt`。

当前默认 dual-source 下：

- visual caption 进入 `low_fps_caption` / caption 检索通道；
- subtitle summary 进入 `asr_text` / ASR 检索通道；
- entities 和 tags 进入 entity/tag 辅助检索信息。

## read_segment 的输入输出

工具位置：

- `src/visual_coding_agent_harness/tools/navigation.py`
- 函数：`read_segment(segment_id)`

输入：

- `segment_id`

输出：

- `claim`
  - segment 时间范围；
  - `caption: ...`；
  - `ASR: ...`；
  - `OCR: ...`；
  - `entities: ...`。
- `regions`
  - 原始 segment dict。

旧 subtitle index 下，`read_segment` 大概率也只会输出：

```text
caption: ASR/subtitle excerpt: ...
```

而不是：

```text
ASR: ...
Visual: ...
Entities: ...
```

## 已确认根因

611-2 原先看到的 `Compact scene index` 只有 `ASR/subtitle excerpt`，根因是当时 eval 默认路径仍然使用 `scene_index_mode="subtitle"`。

这条路径只是把 SRT/subtitle bucket 压成 excerpt，塞进 `low_fps_caption` 字段，用来给 planner 一个便宜的 coarse index。

它不是多模态 scene index，也没有真正运行 per-segment visual caption。

## 当前代码变更

已完成的调整：

1. VideoMME eval 默认 `scene_index_mode` 从 `subtitle` 改为 `dual-source`。
2. CLI `--scene-index-mode` 默认从 `subtitle` 改为 `dual-source`。
3. `SceneIndex.summary()` 中 Visual caption 排在 ASR summary 前面。
4. 旧 `subtitle_scene_index()` 逻辑保留，但需要显式指定 `--scene-index-mode subtitle` 才会走。
5. planner prompt 中不再默认暴露 `video_ls`，避免短视频场景重复读取同一份地图。
6. 新增 `target_coverage` 和 `read_segment_detail`，承接原来 `video_ls` 的有用部分。
7. option-blind local VLM prompt 从“目标项是否出现”的验证式问题，改为开放描述实际可见/旁白内容。
8. `03a4832` 后，temporal rewrite 的 `exploration_question` 不再列 target names；target 只保留在 `target_entities` metadata 和 coverage/detail 后处理里。

对应测试：

- 默认 CLI/config 现在是 `dual-source`。
- 显式 `scene_index_mode="subtitle"` 仍会走旧 subtitle fallback。
- dual-source builder 仍使用文本模型 summarizer 和 VL captioner。
- summary 输出中 `Visual:` 先于 `ASR:`。

## 三层信息设计

当前建议给纯文本 planner 三层信息：

### Layer 0：全局 compact map

每轮 prompt 都带，但保持短：

```text
seg_0001 intro / Bernini biography
seg_0002 early sculptures / Apollo and Daphne mention
seg_0003 Baroque context / theater
seg_0004 David and Borghese sculpture comparison
seg_0005 Apollo and Daphne details
seg_0006 later career / Fountain
```

代码承载：

- `SceneIndex.summary()`
- prompt 中的 `Compact scene index`

语义：

- 这是默认地图；
- 对短 VideoMME case，不应再调用 `video_ls` 重复拿地图。

### Layer 1：target coverage matrix

每轮可通过工具生成，专门服务 MCQ / QA 的 target 定位：

```text
T1 Aeneas group: candidates [seg_0004? low], missing confirmation
T2 David: candidates [seg_0004 high]
T3 Rape of Persephone: candidates [none], missing
T4 Apollo and Daphne: candidates [seg_0002 ASR, seg_0005 ASR/visual high]
```

代码承载：

- 工具：`target_coverage(targets: list, top_k: int = 3, modalities: list = [])`

输出：

- `coverage`
  - `target_id`
  - `target`
  - `status`
  - `candidates`
  - `missing_confirmation`
- 每个 candidate 包括：
  - `segment_id`
  - `start_sec`
  - `end_sec`
  - `score`
  - `matched_fields`
  - `matches`
  - `summary`

注意：

- 它只是索引覆盖矩阵；
- 不能作为最终视觉证据；
- final 仍需要 `vision_read` / `caption_segment` 等非导航视觉 observation。

### Layer 2：on-demand segment detail

planner 选择 segment 后再展开：

```text
read_segment_detail(seg_0004):
- visual_caption full
- asr_summary full
- raw_asr excerpt clipped
- detected text/OCR
- entities
- target hits
```

代码承载：

- 工具：`read_segment_detail(segment_id: str, targets: list = [])`

输出：

- `segment_id`
- `start_sec`
- `end_sec`
- `visual_caption`
- `asr_summary`
- `raw_asr_excerpt`
- `ocr_text`
- `entities`
- `target_hits`

注意：

- 它仍是索引层 detail；
- 作用是帮 planner 决定下一步读哪个 segment；
- 不替代 fresh visual evidence。

## video_ls 降级

`video_ls` 仍保留在 registry 中，兼容旧代码和测试场景，但不再默认出现在 planner tool schema 里。

原因：

- 对短视频，prompt 里的 `Compact scene index` 已经是一张完整地图；
- 再调用 `video_ls` 通常只是重复拿同一份地图；
- 容易浪费一轮工具调用；
- 还会诱导 planner 停留在 navigation 层，而不是读取真正视觉证据。

替代：

- 用 `target_coverage` 做 target-to-segment matrix；
- 用 `read_segment_detail` 展开单段完整索引；
- 再用 `vision_read` / `caption_segment` 产生 citation-ready 视觉证据。

## local VLM prompt 调整

当前问题：

- 转写后的问题已经保留四个 target；
- 但 local VLM prompt 仍然把 target list 作为验证任务塞给 `caption_segment` / `vision_read`；
- Qwen3-VL 容易保守输出“该段没有这些 target items”；
- 这会生成大量负例 evidence，反而压过 scene index 里的有用线索。

当前调整：

- option-blind + temporal/timeline route 下，local VLM prompt 改为开放描述：
  - 实际可见的 artworks / objects / people；
  - scene changes；
  - onscreen text；
  - narrated events；
  - presentation order；
  - timestamps if possible。
- wording 避免出现 `target list` / `present or absent` 这类否定式检测框架，改成：

```text
Openly describe this segment's actual visible artworks, objects, people, scene changes,
onscreen text, and narrated events in presentation order. Include timestamps if possible.
Focus on concrete observations rather than conclusions.
```

- temporal rewrite 现在要求：

```text
Describe the video segment by segment. Record the actual artworks, sculptures, onscreen text,
narration, and scene transitions in the order they appear, with timestamps when possible;
focus on concrete observations rather than conclusions.
```

目标：

- local VLM 产出可供 AnswerAgent 匹配的事实；
- 不让 local VLM 变成 yes/no target detector。

当前 KML rewrite audit：

- baseline：`runs/rewrite_audit_baseline_cf7f185_50_20260607`
  - `target_detector_prompt=22`
  - `temporal_targets_in_tool_question=6`
- 修复版：`runs/rewrite_audit_open_03a4832_50_20260607`
  - `target_detector_prompt=7`
  - `temporal_targets_in_tool_question=0`
  - `option_surface_leak=0`
  - `missing_temporal_targets=0`

## 设计风险

1. 字段语义混淆：
   - 字幕文本被塞进 `low_fps_caption`；
   - 工具可能把它当 caption/channel；
   - planner 看到 `ASR/subtitle excerpt` 但工具字段叫 caption。
2. 信息被两次压缩：
   - subtitle bucket 压到 720 字符；
   - prompt summary 再压到 240 字符；
   - 对 timeline/order 题，关键实体可能被截掉。
3. 缺少视觉索引：
   - 只靠字幕可能找不到画面里的作品出现时刻；
   - timeline 题很依赖 visual caption 或 targeted vision_read。
4. segment 太粗：
   - 300s 一段会把多个作品混在一起；
   - `Compact scene index` 只能粗定位，不能直接回答顺序。

## 当前建议

短期调试：

1. 对剩余 7 个 rewrite risk samples 做第二轮 prompt / leak 检测收紧。
2. 用同一批 50 题复跑 rewrite audit，确认 candidate-value leak 不再进入 `exploration_question`。
3. 对 611-2 的 no-hard-skill run，使用当前默认 dual-source scene index 和最新提交重新跑。
4. 对照 compact trace：
   - planner prompt 中 `Compact scene index` 是否出现 `ASR | Visual | Tags | Entities`；
   - `target_coverage` 对 `David`、`Persephone`、`Apollo and Daphne`、`Aeneas...` 的 candidates 是否合理；
   - forced `caption_segment` / `vision_read` 的问题是否保持开放 caption，而不是 target detector。

中期设计：

1. 不要把 subtitle excerpt 塞进 `low_fps_caption`，至少在显示和 VideoMap 转换时保持字段语义。
2. 对 prompt 中的 scene index，明确区分：
   - `ASR/subtitle`
   - `Visual caption`
   - `Entities`
   - `Tags`
3. 对 timeline 题，提供不截断或少截断的 entity-aware excerpt。
4. 如果保留 `subtitle` 默认模式，工具描述里要明确说明它是 subtitle-only coarse index，不是 visual scene caption。
