# WP16-7 修订计划：Change-Triggered Entity Occurrence

## 结论先行

WP16-6 证明 Fixed-3 的主要失败来自时间采样覆盖，而不是 RRF 本身。WP16-7A 只回答一个问题：在完全相同的 OCR 与 admission 下，变化触发采样能否比等预算均匀采样覆盖更多官方 anchor 附近的可见实体文字。

本轮不跑 retrieval、RRF、QA 或 judge；只有 WP16-7A 的 coverage gate 通过后，才另开 WP16-7B。视频直接从已有 `/m2v_intern` 路径流式读取，不复制视频，不落全量 1fps 帧。

## Review 后的四项修正

1. **先冻结 `anchor_text_expected`。** 主指标只统计预期确实存在屏幕文字的 case。`no` 与 `uncertain` 仍报告，但不混入严格主分母。
2. **A3 改名为 Tier-0 diagnostic。** 它只对 A2 漏例在官方区间内做 1fps 密集诊断，不是 upper bound，也不参与 endpoint。
3. **A1/A2 共用新 admission。** WP16-6 的词法规则不变；时间支持从“同一个 Fixed-3 Caption packet”改成“同一个 60 秒 entity occurrence”，该变化同时作用于 A1/A2。A0 仅作历史复现，不能声称 admission 完全相同。
4. **效应量优先于显著性。** frozen10/39 都不足以把 McNemar `p<0.05` 设为硬门槛。报告配对差值与 McNemar，决策使用预先冻结的覆盖增量。

## 系统流程

```text
现有 32 段视频
  -> ffmpeg 1fps / 160x90 灰度流（不落盘）
  -> 每秒 full-frame + UI-band 变化分数
  -> 同一份 Tier-0 frame universe
       A1: exact-budget uniform
       A2: 300s bin peak + global change rank + 2s NMS
  -> 只临时解码被选中的高分辨率帧
  -> 同一个 Gemini 2.5 Pro blind entity OCR prompt
  -> 同一个 lexical/admission policy
  -> 60s 内同名实体合并为 occurrence
  -> coverage-only report
  -> 临时帧删除
```

变化分数冻结为：

```text
0.65 * mean(full-frame absolute difference)
+ 0.35 * mean(top/bottom/side UI-band absolute difference)
```

变化分数只决定看哪些帧，不读取题目、选项、答案、Caption 或官方区间。

## Arms

| Arm | 用途 | 采样 | OCR / admission |
| --- | --- | --- | --- |
| A0 Fixed-3 | 历史参照 | 每个 Caption passage 三帧 | WP16-6 历史产物，不重跑 |
| A1 Uniform | 等预算控制 | 从完整 1fps universe 均匀取 B 帧 | 与 A2 完全相同 |
| A2 Change | treatment | 每 300s 至少一个峰值，再按变化分数填满 B | 与 A1 完全相同 |
| A3 Diagnostic | 漏例归因 | 仅 A2 miss 的官方区间 1fps | 诊断，不算 endpoint |

## 存储与执行

- 大文件根目录只用于小型中间结果：`/m2v_intern/xuboshen/zgw/mger_runs/...`。
- 不创建视频副本。
- Tier-0 解码使用 rawvideo pipe；仅保存每秒的时间戳与两个变化分数。
- 高分辨率候选帧按 segment 临时生成，OCR 的结构化结果持久化后立即删除。
- 所有输出保留 source video、segment、source time、virtual time 与 selection reason。

## 冻结 cohort

先做已有完整字段规格的 frozen10。严格文字预期分层为：

- `yes`（主分母 8）：0010、0016、0028、0054、0076、0166、0179、0184。
- `no`：0100，ordinal meditation identity 主要是视觉/事件状态。
- `uncertain`：0108，章节/地点标题可能出现，也可能只靠场景转场表达。

在任何 frozen39 结果出现前，必须另行冻结 39 题的同类分层；没有这个文件就不运行或解释 frozen39 主指标。

## 预算阶梯

1. 两段 smoke：每臂 24 帧，验证时间映射、临时帧清理、blind prompt 与 lineage。
2. frozen10 canary：每臂 256 帧，只看结构和明显覆盖错误。
3. frozen10 primary：每臂 8,873 帧（完整 1fps universe 的约 10%）。
4. 只有 B1 显示正向覆盖增量时，再考虑 17,746 / 26,600 帧的饱和曲线。

每个阶段 A1/A2 必须共享同一 Tier-0 manifest 且选帧数完全相等。

## Gate 与判读

结构 gate 包括：同一 Tier-0 universe、预算完全相等、采样 question/gold blind、模型正确、A1/A2 prompt 和 admission 相同、完整 lineage、零解析失败、零 raw response 持久化、零 dense frame 持久化。

主 endpoint：`anchor_text_expected=yes` 上，官方 anchor 区间前后各扩 60 秒后，是否出现 query-compatible entity occurrence。60 秒容差已写入冻结协议，评测时不得改动。

frozen10 决策：

- GO：A2 比 A1 至少多覆盖 2 个严格 case，且 A2 至少覆盖 4/8。
- PARTIAL：A2 多 1 个，或 A2 为 3/8。
- STOP：A2 不优于 A1，且 A3 也看不到漏掉的可见实体文字。

McNemar 只报告。Endpoint 绝不作为 structural gate。

## Miss audit

A2 漏例在 endpoint 冻结后最多审 10 例，并互斥归类：

1. `ui_text_exists_reader_or_resolution_failure`
2. `no_ui_text_visual_event_or_state`
3. `speech_only_asr_candidate`
4. `annotation_uncertain`
5. `other`

只有第 1 类回到 OCR/Tier-0 reader；第 2 类进入独立 event-state index；第 3 类进入 ASR。不能把所有低 coverage 都解释为 OCR 不够强。

## 当前停止条件

- 不访问 Day-test140 或 Week。
- 不实现 A4.2、A5、WP7。
- WP16-7A 未 GO 前不做 retrieval/QA。
- 60 秒 occurrence gap 是 provisional 超参，只用于本轮共享 admission，不据此做强语义结论。

## 实现与冻结入口

- 协议：`docs/evaluations/mmlifelong_wp16_7_change_triggered_entity_occurrence_v1.json`
- Tier-0 扫描与等预算选帧：`tools/prepare_mmlifelong_change_triggered_entity_sampling.py`
- Blind OCR 与 occurrence admission：`tools/run_mmlifelong_change_triggered_entity_ocr.py`
- Coverage-only 分析：`tools/diagnose_mmlifelong_change_triggered_entity_coverage.py`
- 当前协议 SHA256：`8b84518f8ee1aa4fd30ce0c6e74d23ebb7fcd2915af45417fdf15efe1a4a20aa`

分析器只报告 A1/A2 coverage、配对差值、W/T/L 与 McNemar exact p；它不会执行 retrieval、QA 或 judge。A2 miss 列表在 endpoint 之后生成，供 Tier-0 diagnostic 做互斥原因归类。
