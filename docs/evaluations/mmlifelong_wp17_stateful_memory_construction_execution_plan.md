# MM-Lifelong WP17：Stateful Memory Construction 执行方案

> 状态：执行草案 v0.1，尚未预注册冻结<br>
> 日期：2026-08-28<br>
> 基线代码：`c49426e` 及其之前的 WP16 证据链<br>
> 当前阶段：只定义并验证 Memory Construction；在 construction gate 通过前不运行最终 QA

## 0. 结论先行

WP17 要验证的不是“再做一种长视频 memory”，而是一个更窄、更可证伪的问题：

> MM-Lifelong 的主要召回损失，是否在视频第一次被压成独立 Caption 时就已经发生；先保存屏幕文字和视觉事实，再用带 provenance 的历史状态构造当前事件，能否保留后续检索所需的实体、事件、状态和 occurrence 信息？

计划的核心是一个 question-blind 的构建管线和一个 2×2 因子实验：

```text
Video
  -> dense cheap OCR / visual observation
  -> temporally linked OCR tracks + entity evidence
  -> 30s segment construction
       current frames
       + optional OCR evidence
       + optional inherited compact state
  -> Structured Event Record (SER)
  -> provenance-gated state ledger
  -> occurrence lifecycle / ordinal index
```

必须收紧的研究 claim 是：

> **Screen-text-anchored, provenance-gated stateful memory construction for repeated-event long-video QA.**

不能把“stateful captioning”“hierarchical memory”或“entity memory”本身写成 novelty；这些方向已有多项直接相关工作。

## 1. 为什么从 Retrieval 转到 Construction

WP16-7 的当前证据是：

- 等预算 Uniform 和 Change-Triggered 两个正式臂在 8 个严格文字预期 case 上都是 `0/8`。
- question-blind 的官方区间 1fps dense audit 能恢复 `5/8`：0010、0054、0076、0166、0184。
- 正式臂几乎没有采到官方区间；主要失败是 retrieval-time sampling coverage，不是已有 OCR reader 完全不会读。
- 0016、0028、0179 仍然只能记为 unresolved。旧诊断用 query/token match 反推 reader failure，存在循环定义，WP17-0 必须先重新核实表面形式。
- 0115 因“多次 occurrence 状态比较”被旧协议排除，直接暴露了基于固定 60s gap 的 occurrence 表示缺口。

因此当前最合理的工作假设是：

```text
如果实体名、状态转移或 occurrence 边界没有进入构建期 memory，
下游 retriever 增大 K、扩邻居或临时 OCR 都只能付出更高成本追赶，
且无法突破 caption-bounded recall。
```

这不是说 retrieval 已经解决，而是先测试它的输入是否足够。

## 2. 研究问题与可证伪假设

### RQ1：信息首先丢在哪一层？

把严格 anchor 分解成：

```text
pixel-visible -> OCR-readable -> admitted/linked -> represented in SER -> retrievable
```

假设 H1：`R_pixel × R_read` 明显高于现有独立 Caption 的 representation recall，说明主要损失发生在 Caption 化或构建期 admission，而不是像素层完全没有证据。

### RQ2：OCR evidence 和 state inheritance 各自贡献多少？

假设 H2a：`C10 > C00`，说明 OCR/实体证据在构建期注入有独立收益。<br>
假设 H2b：`C11 > C10`，说明历史状态在 OCR 之外仍有独立收益。<br>
假设 H2c：若 `C10 ≈ C11`，则应停止复杂状态继承，保留更简单的 OCR-enhanced construction。

### RQ3：历史状态会传播错误多久？

假设 H3：硬 provenance write gate 能显著缩短错误状态传播长度，主要价值是降低长期漂移，而不一定提高单段生成质量。

### RQ4：occurrence ordinal 能否在构建期形成？

假设 H4：基于 encounter lifecycle 的 ordinal 比固定时间 gap 更能稳定表达 `#1/#2/#3`、before/after 和 encounter start/end。

### RQ5：屏幕表面形式能否形成 question-independent canonical identity？

假设 H5：保留原始 OCR surface、时间轨迹和 query-blind alias edge，能提高正式实体名覆盖，同时不把问题或答案词泄漏进 memory。

## 3. 贡献边界

建议最终只保留三个主机制：

1. **Screen-text-anchored canonical identity**：在 generic Caption 之前保存 UI/OCR 的实体表面形式和轨迹。
2. **Provenance-gated state inheritance**：状态新增、修改和关闭必须引用当前 segment 的合法证据；carry-forward 保留原始 provenance 和版本链。
3. **Construction-time occurrence identity**：按 encounter lifecycle 建立 repeated occurrence 和 ordinal，而不是用固定秒数聚类。

错误传播实验是重要诊断，但不应扩成第四套系统贡献。

谨慎措辞：ReflectWorld-MM 已明确提出 evidence-backed entity memory，因此不能宣称“首次做 evidence-grounded state”。可主张的是：本项目把 **hard write validation、screen-text identity 和 controlled propagation measurement** 放进同一 MM-Lifelong 构建实验，并逐项消融。

## 4. 系统设计

### 4.1 Question-blind 构建边界

构建期允许输入：

- 当前 segment 的帧及时间戳；
- 当前 segment 的 OCR track slice；
- 若所有 arm 都具备，则可加入完全相同的 ASR cue；
- 当前 arm 自己的上一版 compact state。

构建期禁止输入：

- question、options、gold answer；
- official target interval、gold clue interval；
- frozen case 的 target entity/query aliases；
- 任何由 QA outcome 反向挑选的 frame 或 evidence。

官方区间和标注只在构建完成后用于评估。

### 4.2 Evidence store

所有可引用证据先写入 append-only store：

```json
{
  "evidence_id": "ocr:seg_0012:track_0007",
  "kind": "ocr_track",
  "segment_id": "seg_0012",
  "start_sec": 360.0,
  "end_sec": 366.0,
  "surface": "寅虎",
  "normalized_surface": "寅虎",
  "support_frames": ["frame:seg_0012:0360", "frame:seg_0012:0362"],
  "source": "paddleocr+gemini_verify",
  "confidence": "high"
}
```

Evidence ID 必须稳定、唯一并能反查 segment、时间和原始 source。模型只能从 prompt 中给出的 allowed evidence handles 中引用，不能自己发明 ID。

### 4.3 Dense OCR track

WP17 不再把同名 OCR 行全局去重为一个实体。Track 必须保留时间连续性：

```text
frame OCR rows
  -> normalize text without discarding single English/CJK names
  -> match by text similarity + UI region + bbox/center + short temporal gap
  -> close track after sustained absence
  -> preserve a later reappearance as a new track
```

最小 schema：

```json
{
  "track_id": "track_0007",
  "surfaces": ["寅虎"],
  "canonical_surface": "寅虎",
  "start_sec": 360.0,
  "end_sec": 371.0,
  "ui_regions": ["boss_name_bar"],
  "support_frame_ids": ["..."],
  "bbox_series": [[0.31, 0.08, 0.48, 0.13]],
  "reader_sources": ["paddleocr", "gemini_verify"],
  "lineage_complete": true
}
```

GoMatching 的 long/short-term matching 可作为算法参考，但首版不引入其 Detectron2/旧 CUDA 训练栈。首版使用可测试的确定性匹配；只有 track fragmentation 成为主瓶颈时再升级。

### 4.4 Entity alias

构建 memory 时必须保留原始 surface，不能只留下模型猜测的 canonical name。

```text
surface node: 寅虎
alias edge: 寅虎 <-> Yin Tiger
canonical display: 寅虎
```

规则：

- Unicode、空白、标点和 OCR 常见混淆可机械归一。
- 同语言近似 surface 可由 track 共现和 edit similarity 建边。
- 跨语言 alias 若使用模型或外部词表，必须 question-blind、单独记录来源，并作为独立 ablation。
- 用于评分的 gold alias/equivalence 表与 construction 完全隔离，不能注入生成器。

### 4.5 Structured Event Record

每个 30s segment 输出 SER，而不是只输出一段自然语言：

```json
{
  "segment_id": "seg_0012",
  "entities": [
    {
      "entity_ref": "entity:寅虎",
      "surface": "寅虎",
      "role": "active_boss",
      "evidence_ids": ["ocr:seg_0012:track_0007"]
    }
  ],
  "events": [
    {
      "predicate": "encounter_started",
      "subject_refs": ["entity:寅虎"],
      "time_range_sec": [360.0, 371.0],
      "evidence_ids": ["ocr:seg_0012:track_0007", "frame:seg_0012:0365"]
    }
  ],
  "state_changes": [],
  "relations": [],
  "occurrence_refs": [
    {"encounter_id": "enc:寅虎:0002", "ordinal": 2, "phase": "active"}
  ]
}
```

可检索文本由结构字段机械渲染；保留自然语言 summary 只作辅助，不作为唯一 memory。

### 4.6 Provenance-gated state ledger

状态不是一段可自由续写的 summary，而是 versioned ledger：

```json
{
  "state_version": 12,
  "slots": {
    "active_encounter": {
      "value": "enc:寅虎:0002",
      "lifecycle": "active",
      "origin_evidence_ids": ["ocr:seg_0012:track_0007"],
      "last_update_evidence_ids": ["frame:seg_0012:0365"],
      "carried_from_state_version": 11,
      "last_observed_sec": 371.0
    }
  }
}
```

状态操作只有：

- `observe`：创建新 slot，必须引用当前 segment 证据。
- `update`：修改值或 lifecycle，必须引用当前 segment 证据。
- `close`：结束 encounter/state，必须引用当前证据或显式 runtime timeout policy。
- `retain`：不生成新事实；runtime 原样保留旧值、原始 provenance 和 `carried_from_state_version`。

关键修正：**不是每个 segment 都要求重新看见同一实体。** 只有新增或语义改变必须有当前证据；合法 carry-forward 继承原始证据链。否则 state 无法跨过一个没有重复 UI 的 segment。

### 4.7 Occurrence lifecycle

不再使用“同一实体 + 60s gap”直接计数，而维护：

```text
absent/closed -> entering -> active -> ending -> closed
```

约束：

- 只在 `closed/absent -> active` 的合法转移上递增 ordinal。
- 短时 UI 消失、菜单打开或镜头遮挡不自动递增。
- defeat、victory overlay、loot、boss bar disappearance、scene transition 都可以是 end evidence，但需保留各自 evidence handle。
- generic event boundary detector 只可提出候选 boundary，不能单独决定实体身份或 ordinal。

## 5. 两个直观例子

### 例 1：后续 segment 没有再次出现“寅虎”

```text
Segment t:
  OCR = 寅虎
  visual = fight starts

Segment t+1:
  OCR = none
  visual = player opens equipment/gourd menu
```

`C10` 只能从当前证据写“玩家打开菜单”。<br>
`C11` 可以写“在寅虎第 2 次遭遇期间打开菜单”，但这条记录必须同时引用：

- 当前 segment 的 menu frame；
- inherited `active_encounter=寅虎#2` 的原始 evidence 和 state version。

如果 C11 不能稳定提高 relation/state coverage，说明 state inheritance 不值得复杂化。

### 例 2：boss 倒地和正式胜利是否应合并？

```text
04:56:49-04:57:24：boss 跪倒、被击败并消散。
05:00:00-05:00:47：最后一击/“得胜”确认，随后新角色出现并唱歌。
```

构建期不看题目，也不先决定“这两个 Caption 是否应该合成一个检索单元”。它应记录：

```text
encounter X:
  phase active -> ending   @ 04:56:49
  event boss_down
  phase ending -> closed   @ 05:00:00
  event victory_confirmed

next event:
  new_character_appears / song_started
```

这样 retrieval 可以按问题选择 `boss_down`、`victory_confirmed` 或完整 encounter；不需要用问题反向决定构建时的 merge。

## 6. 2×2 因子实验

### 6.1 Arms

| Arm | 当前 frames | OCR evidence | inherited state | 作用 |
| --- | --- | --- | --- | --- |
| C00 | 有 | 无 | 无 | 30s 独立构建基线；不是现有 5min Caption 的等价复现 |
| C01 | 有 | 无 | 有 | 状态幻觉/漂移探针 |
| C10 | 有 | 有 | 无 | OCR evidence 主效应 |
| C11 | 有 | 有 | 有 | 完整 treatment |

C01 不能省略；它是唯一能区分“state 真有上下文价值”和“state 只是持续复述先验”的臂。

四臂统一使用 30s segment，因此这个 2x2 只估计**固定粒度下** OCR evidence 和 state inheritance 的效应。现有系统的 5min Caption 另记为外部基线 E0a；同时保留 30s 独立自然语言 Caption 校准臂 E0b。只有 E0a 与 E0b 的差异才能归因于 segment 粒度，不能把它计入 OCR 或 state 的 treatment effect。

### 6.2 Matched-control 约束

四臂必须共享：

- 完全相同的 segment 划分、帧 ID、帧预处理和时间戳；
- C10/C11 完全相同的 frozen OCR track；
- 若使用 ASR，则四臂输入完全相同的 ASR；
- 相同 backbone、provider、token budget、JSON schema 和重试策略；
- 每个 arm 独立 state，不允许跨臂读取。

重要限制：现有 `MatchedResponseCacheClient` 只有在 prompt、images、model 等请求完全一致时才能重放。C00/C01/C10/C11 的请求本来就不同，**不能跨 treatment 重放模型响应**，否则会抹掉 treatment。它只用于同臂 resume 或真正相同的 pre-treatment 调用。跨臂匹配的是 evidence packet，不是生成文本。

为降低 provider 时间漂移：按 segment 处理四臂，使用按 segment 轮换的 arm 顺序；C01/C11 各自仍严格按时间顺序推进状态。记录 requested seed 和实际模型元数据，但不假设闭源 API seed 一定生效。

### 6.3 效应量

主配对比较：

```text
OCR direct effect   = C10 - C00
State direct effect = C11 - C10
```

同时报告 factorial estimates：

```text
OCR main effect   = ((C10 - C00) + (C11 - C01)) / 2
State main effect = ((C01 - C00) + (C11 - C10)) / 2
Interaction       = (C11 - C10) - (C01 - C00)
```

frozen10+0115 只有约 11 个 target case，所有区间和 p 值都只能作为 development diagnostic，不能作为无偏论文结论。

## 7. 指标与 gate

### 7.1 Construction endpoints

1. **ARC — Anchor Representation Coverage**<br>
   官方 anchor event 是否在对应 SER/occurrence 中出现。构建时不可见官方区间，评估时才对齐。
2. **CEC — Canonical Entity Coverage**<br>
   目标实体的原始 surface 或冻结 equivalence alias 是否进入 OCR track/SER/state。
3. **RSC — Relation/State Coverage**<br>
   关键 before/after、entity-event、equipment transition、encounter phase 是否被表达。
4. **Occurrence coverage / ordinal accuracy**<br>
   start/end 边界、encounter 数量、ordinal 与人工冻结标注的一致性。
5. **PV — Provenance Validity**<br>
   分成三层，不能混成一个“全自动幻觉率”：
   - `PV_ref`：引用 handle 存在、属于合法 segment/旧 state，机械可判；
   - `PV_text`：文本声明在引用 OCR/ASR 中可归一匹配，机械可判；
   - `PV_visual`：frame 是否真的支持事件/状态，需 blind 人工或独立 verifier 抽检。
6. **Unsupported state write rate**<br>
   被 hard gate 拒绝的 observe/update/close 比例与原因。
7. **Cost**<br>
   每小时视频的 local OCR runtime、VLM calls、prompt/completion/reasoning tokens、临时存储峰值。

### 7.2 Structural gates

这些 gate 判实验是否有效，不包含任何 treatment endpoint：

- exact segment/frame universe aligned；
- construction question/gold/official-interval blind；
- actual model/provider/config 正确；
- OCR/SER/state/occurrence schema 全部可解析；
- 每个 evidence handle 唯一且 lineage 完整；
- `PV_ref=100%`；文本派生状态 `PV_text=100%`；
- state version 单调，四臂无交叉污染；
- occurrence lifecycle 转移合法，ordinal 只在合法 start 上递增；
- 零 silent state overwrite、零 terminal transaction failure；
- finish reason、completion tokens 和 reasoning tokens 完整记录；空可见 content 不得在未检查 token exhaustion 前归因成 abstention；
- 原始视频不复制，临时帧按策略清理；
- endpoint 值不作为 structural gate。

### 7.3 Development decision rules

这些只是下一步工程决策，不是统计显著性声明：

- **OCR GO**：C10 相对 C00 在 ARC/CEC 上净增至少 2 个 case，且 PV 不下降超过 5pp。
- **State GO**：C11 相对 C10 在 RSC、ARC 或 ordinal 上净增至少 2 个 case，且 PV 不下降超过 5pp。
- **Simplify**：C10 改善而 C11≈C10，则停止 state line，只保留 OCR-enhanced construction。
- **State STOP**：C11 没有增益，或 unsupported write / propagation 明显恶化。
- **Construction STOP**：C10、C11 都不优于 C00，回到 frame/event reader，而不是启动 retrieval/QA。

## 8. 分阶段执行

### WP17-0：零 API 再诊断与冻结（约半天）

任务：

1. 对 0016、0028、0179 做 question-independent surface audit，区分像素不存在、reader miss、alias mismatch 和 admission rejection。
2. 离线扫 `support=1/2/3 × lexical on/off × high-value singleton on/off`，输出 pre/post admission recall，不改 endpoint。
3. 固化 sparse archive 的功效分析；不运行 B1 10% frame-budget 实验，因为 uniform baseline 预计已覆盖大多数 case，treatment headroom 小于 frozen10 噪声。
4. 在任何 WP17 outcome 出现前冻结：
   - local timeline；
   - construction annotations；
   - 与 frozen10/frozen39 不重叠的 `wp17_heldout_v1`（至少 20 题）；
   - 0115 与 2-3 个 repeated encounter 的 start/end/ordinal 标注。

输出建议：

```text
tools/diagnose_mmlifelong_wp17_preflight.py
docs/evaluations/mmlifelong_wp17_local_timeline_v1.json
docs/evaluations/mmlifelong_wp17_heldout_v1.json
docs/evaluations/mmlifelong_wp17_occurrence_ordinal_v1.json
```

只有 WP17-0 完成后才冻结正式 protocol JSON。

### WP17-1：Dense OCR track（2-3 天）

范围：frozen10 + 0115 的 anchor ±10min 局部 timeline，约 3.5h；不处理完整 24h。

流程：

```text
ffmpeg 1fps stream
  -> full frame + frozen UI ROIs
  -> local PaddleOCR candidate rows with bbox
  -> temporal linking / dedup
  -> uncertain/high-value candidates only: Gemini 2.5 Pro blind verify
  -> OCR track store
```

预算：约 12,600 个 1fps 时间点；Gemini 定向复核上限 500 calls，尽量批量输入。视频直接复用 KML 现有路径，不复制；帧流式处理，持久化结构化 rows/tracks，不持久化全量帧或完整模型 prose。

WP17-1 mechanism check：track-level strict target recall 不低于 official-interval dense review 的 80%；以实际计数同时报告，不能用小样本百分比掩盖分母。若不通过，先修 ROI/reader/track，不进入 WP17-2。

### WP17-2：2×2 memory construction（5-7 天）

范围：同一 3.5h timeline，固定 30s segment，约 420 segments。

调用量：

```text
420 segments × 4 arms = 1,680 base generation calls
```

重试总上限建议为 10%，总调用 hard cap 1,850；final/SER JSON 调用的 completion budget 不低于 4096，并保留 usage metadata。

执行顺序：先做 3-case vertical canary；结构 gate 通过后再跑全量。分析只做 construction endpoints，不调用 QA judge。

### WP17-3：错误传播与 ordinal（约 3 天）

错误注入：

- 错误 `location`；
- 错误 `active_encounter`；
- 错误 `occurrence_count`。

在随机冻结的 state version 注入错误，向前重放最多 20 segments；比较 hard provenance gate on/off。报告：

- propagation length `L`；
- 自愈、显式关闭、timeout 和持续错误比例；
- 每类 slot 的污染范围；
- gate 的拒绝成本和误杀率。

模型/重放调用 hard cap 800。若 gate 不能缩短传播，不能把 C11 接入长期 memory。

### WP17-4：Retrieval / QA（只在 GO 后）

前置条件：

- WP17-2 至少 OCR GO；
- 若要测试 C11，则同时满足 State GO 和 WP17-3 propagation gate；
- heldout 已在任何 construction outcome 前冻结；
- frozen10/frozen39 明确降级为 development set。

heldout 上至少比较：

```text
E0a current 5min independent Caption / ReMA
E0b 30s independent natural-language Caption（粒度校准）
E1  C10 OCR-enhanced independent structured memory
E2  C11 provenance-gated stateful structured memory（若 State GO）
E3  MAGIC/WorldMM-style structured-memory baseline（可复现时）
```

E0b/E1/E2 共享相同 30s segment 和 frame packet；E0a 只用于回答“相对当前 5min 系统是否整体改善”。相同 retrieval、QA backbone 和 judge；除明示的 segment 粒度校准外，construction memory 是唯一变化。没有新授权时不访问 Day-test140 或 Week。

## 9. 当前仓库的实现映射

### 9.1 原样复用

| 当前文件 | 复用内容 | 注意事项 |
| --- | --- | --- |
| `src/vcah/virtual_video.py` | source/virtual 时间映射、segment/frame lineage | 作为所有 evidence ID 的时间真值 |
| `src/vcah/captioning.py` | chunk spec、frame materialization、缓存摘要、resume | WP17 使用 30s 和结构化 prompt，不覆盖旧 Caption cache |
| `src/vcah/caption_store.py` | 事务状态和可恢复写入 | SER/state 最好使用独立 schema/store |
| `src/vcah/model_client.py` | API、usage metadata、同请求 record/replay | 不跨 treatment replay 不同 prompt |
| `src/vcah/occurrence_ocr.py` | blind OCR prompt/parser、row normalization、lineage 绑定 | 当前全局文本去重不是 track，需替换为时间链接 |
| `src/vcah/occurrence_entity_sidecar.py` | entity OCR schema、region/type/confidence | WP17-0 后重定 admission；不得沿用单字/单词排除 |
| `src/vcah/change_triggered_entity_occurrence.py` | 低成本 ffmpeg 流和 change score | 只作辅助 trigger/诊断，不再是主覆盖机制 |
| `tools/audit_mmlifelong_occurrence_canary.py` | gate/report 风格 | 新建 construction 专用 audit，避免混入旧 endpoint |

### 9.2 只新增四个核心模块

| 新文件 | 责任 | 聚焦测试 |
| --- | --- | --- |
| `src/vcah/ocr_track.py` | bbox/region/text/time 的 track linking；保留短暂 singleton 和重复出现 | fragmentation、reappearance、single-token/CJK、lineage |
| `src/vcah/entity_alias.py` | query-blind surface normalization、alias graph、来源隔离 | 无 question/gold 泄漏、跨语言 ablation、原始 surface 保留 |
| `src/vcah/segment_state.py` | versioned state、delta、retain、hard provenance gate | observe/update/close/retain、跨臂隔离、错误注入 |
| `src/vcah/occurrence_ordinal.py` | encounter lifecycle、start/end、counter、ordinal | UI 短失不递增、结束/重开、0115 多次 occurrence |

SER schema 可放在 `src/vcah/structured_event_record.py`，但它应保持纯数据契约和 validator，不扩成另一个 agent 框架。

### 9.3 工具入口

```text
tools/diagnose_mmlifelong_wp17_preflight.py
tools/build_mmlifelong_dense_ocr_track.py
tools/run_mmlifelong_stateful_memory_2x2.py
tools/audit_mmlifelong_stateful_memory.py
tools/analyze_mmlifelong_memory_construction.py
tools/run_mmlifelong_state_error_injection.py
```

每个工具先有 `--dry-run` 和 compact summary；健康任务只监控 counts/liveness/free space，不读原始日志。

### 9.4 实现顺序

1. WP17-0 diagnostic commit。
2. `ocr_track.py` + dense OCR canary commit。
3. SER + `segment_state.py` + 2×2 vertical slice commit。
4. `occurrence_ordinal.py` + error injection commit。
5. construction GO 后另开 retrieval/QA commit。

不得把 WP17-1/2/3 和 retrieval 改动合进一个 commit。

## 10. 相关工作与可借鉴代码

### 10.1 直接强基线 / 高价值复用

| 工作 | 与 WP17 的重合 | 可借鉴代码 | 不同点与使用决定 |
| --- | --- | --- | --- |
| [ReMA / MM-Lifelong](https://arxiv.org/abs/2603.05484) | benchmark 和 recursive belief baseline | [官方代码](https://github.com/cg1177/Recursive-Multimodal-Agent) | E0 基线；WP17 研究其独立 Caption 之前的信息保留 |
| [MAGIC-Video](https://arxiv.org/abs/2605.08271) | MM-Lifelong、named entities、semantic triples、visual clips、event/topic chains | [官方 Apache-2.0 repo](https://github.com/lijiazheng0917/MAGIC-video)，尤其 `preprocess/mmlifelong/preprocess_video.py` 与 event/topic chain scripts | 最直接外部 baseline；借 artifact schema/评测接口，不整体迁移当前 harness |
| [WorldMM](https://openaccess.thecvf.com/content/CVPR2026/papers/Yeo_WorldMM_Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning_CVPR_2026_paper.pdf) | episodic/semantic/visual memory，多尺度与 consolidation | [官方 Apache-2.0 repo](https://github.com/wgcyeo/WorldMM)，`src/worldmm/memory/{episodic,semantic,visual}` | 可借 semantic consolidation；不替代 WP17 的 pre-caption OCR/state 因子实验 |
| [M3-Agent](https://arxiv.org/abs/2508.09736) | 30s segments、entity-centric multimodal graph、在线 memorization/control | [官方 Apache-2.0 repo](https://github.com/bytedance-seed/m3-agent)，memorization prompts 与 graph schema | schema/prompt 参考；训练模型/RL 栈不是当前 training-free API 管线的直接依赖 |

### 10.2 高度相关但主要借设计

| 工作 | 可借鉴点 | 约束 |
| --- | --- | --- |
| [ReflectWorld-MM](https://arxiv.org/abs/2607.09759) / [repo](https://github.com/addxai/ReflectWorld) | bounded short-term perception、persistent entity memory、evidence-backed segment/entity state | 已经压缩了 broad novelty；其 TypeScript+Python product runtime 较重，首版不 fork。WP17 差异放在 screen-text identity、hard write gate 和 propagation experiment |
| [GROVE](https://arxiv.org/abs/2608.02392) / [repo](https://github.com/SitongGong/GROVE) | causal streaming、observation→moment→episode→pattern、scale-native retrieval、checkpoint | 代码很新，当前 repo 规模小且未确认清晰 license；先借分层和 causal/checkpoint 设计，不设为依赖 |
| [VideoLLaMB](https://arxiv.org/abs/2409.01071) / [repo](https://github.com/bigai-nlco/VideoLLaMB) | recurrent memory bridge、SceneTiling 的语义连续性 | 模型内 recurrent tokens/LLAVA 栈与当前外部结构化 memory 实验不匹配，只作概念对照 |

### 10.3 OCR 与边界工具

| 工作 | 借鉴方式 | 决定 |
| --- | --- | --- |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 本地低成本中英 OCR、bbox、confidence，Apache-2.0 | WP17-1 首选 dense candidate reader；Gemini 只复核难例/高价值 UI |
| [GoMatching](https://arxiv.org/abs/2401.07080) / [repo](https://github.com/Hxyz-123/GoMatching) | 视频文字 long/short-term matching、track rescoring | 借匹配思想；旧 PyTorch/Detectron2/CUDA 和训练需求使其不适合首版直接集成 |
| [GEBD](https://openaccess.thecvf.com/content/ICCV2021/html/Shou_Generic_Event_Boundary_Detection_A_Benchmark_for_Event_Segmentation_ICCV_2021_paper.html) / [GEB+](https://arxiv.org/abs/2204.00486) | generic boundary proposal、状态变化描述与评估 | 只能提供候选边界，不能决定“哪个 boss 的第几次 encounter”；作为 optional proposal baseline，不作核心依赖 |

### 10.4 复用原则

- 外部 repo 只从官方来源获取，首次使用前固定 commit SHA、license 和依赖摘要。
- 优先借数据契约、prompt、consolidation 和评测接口，不 wholesale fork 大型 runtime。
- MAGIC/WorldMM 是强基线，不是 WP17 novelty；ReflectWorld 是最直接的 novelty 边界。
- 任何 external preprocessing 若读取 benchmark question/gold，必须关闭或隔离，不能进入 construction。

## 11. 时间、调用和存储预算

| 阶段 | 范围 | 模型/API 预算 | 预计时间 |
| --- | --- | ---: | ---: |
| WP17-0 | 离线诊断与冻结 | 0 | 0.5 天 |
| WP17-1 | 3.5h、约 12.6k 时间点 | local OCR + Gemini verify ≤500 calls | 2-3 天 |
| WP17-2 | 420 segments × 4 arms | 1,680 base，hard cap 1,850 | 5-7 天 |
| WP17-3 | error injection + ordinal | hard cap 800 | 3 天 |
| WP17-4 | heldout retrieval/QA | GO 后单独预算 | 未计入 |

存储原则：

- 复用 KML 现有视频路径，不复制视频。
- full-frame/ROI 图片只作临时文件，结构化 OCR/SER/state/metrics 持久化。
- 不在报告、ZIP 或 chat 中保存 raw logs、完整 model prose、secrets 或 config。
- 每阶段输出独立 root，失败残留可恢复，不覆盖旧 WP16 结果。

## 12. 风险与控制

| 风险 | 控制 |
| --- | --- |
| OCR false positive 被状态放大 | track confidence + Gemini selective verify + hard state provenance |
| provenance 只证明“有引用”，不证明视觉语义正确 | PV 拆为 ref/text/visual；视觉支持用 blind audit，不夸大自动验证 |
| alias 表泄漏问题实体名 | construction/evaluation alias 物理隔离；query/gold blind audit |
| C11 历史错误持续传播 | versioned delta、retain 不改写、controlled injection、TTL/close policy |
| arm 顺序和 provider 漂移 | segment-level rotating order、相同 config、usage metadata、必要时重复跑 |
| 30s 切片收益被误记为 OCR/state 收益 | 2x2 固定 30s；另设 E0a 5min 与 E0b 30s 粒度校准 |
| frozen10 反复调参 | 明确为 dev；heldout 在 outcome 前冻结 |
| generic boundary 误当 encounter boundary | GEBD 只提 proposal；实体+lifecycle validator 决定 ordinal |
| API 返回成功但无 visible content | 先检查 finish_reason 和 reasoning-token exhaustion；final call ≥4096 tokens |
| 过度工程化 | 只自研四个核心模块；外部层级/graph/consolidation 优先复用 |

## 13. Claim-Evidence Map

| 候选 claim | 必须由什么证据支撑 | 不能用什么替代 |
| --- | --- | --- |
| 原 Caption 在构建期丢失 anchor 信息 | WP17-0 层级 recall + C10-C00 ARC/CEC | 单个 OCR 成功例 |
| OCR evidence 有独立作用 | 2×2 的 C10-C00 和 OCR main effect | C11-C00 总提升 |
| state inheritance 有额外作用 | C11-C10、state main effect、RSC/ordinal 提升 | C11 的自然语言看起来更完整 |
| provenance gate 抑制错误传播 | WP17-3 gate on/off 的 propagation length | 单段 PV_ref=100% |
| occurrence ordinal 在构建期可行 | 0115+额外标注上的 boundary/count/ordinal | 固定 60s gap 聚类 |
| 方法改善最终 QA | 独立 heldout 的 WP17-4 matched QA | frozen10 dev 增益 |

## 14. 执行验收清单

### 开始 WP17-1 前

- [ ] 0016/0028/0179 surface audit 完成。
- [ ] admission ablation 完成。
- [ ] local timeline、heldout、ordinal annotation 冻结并记录 SHA256。
- [ ] protocol 明确 question/gold blind 和不跑 B1。

### 开始 WP17-2 前

- [ ] OCR track lineage 和 schema gate 通过。
- [ ] dense OCR mechanism check 通过。
- [ ] 3-case 2×2 vertical canary 通过。
- [ ] 四臂 prompt 只包含规定 treatment 字段。

### 开始 WP17-4 前

- [ ] OCR GO；若运行 C11，再要求 State GO。
- [ ] error propagation 可控。
- [ ] heldout 从未参与 prompt、threshold 或 stop-rule 调整。
- [ ] external baseline commit/license/config 已冻结。

## 15. 最终决策树

```text
WP17-0 发现 pixel/reader 本身不足
  -> 先修 local OCR/ROI，不做 state

WP17-1 OCR track 不能接近 dense review
  -> 停在 reader/track，不做 2×2

C10 > C00, C11 ≈ C10
  -> 采用 OCR-enhanced independent construction；停止复杂 state

C11 > C10 且 propagation 可控
  -> stateful construction GO；再做 heldout retrieval/QA

C11 > C10 但 propagation 失控
  -> 保留 OCR，修 provenance/lifecycle；不接长期 memory

C10、C11 都不改善
  -> construction representation 仍不足，回到 event/visual reader，不继续调 retrieval sampler
```

## 16. 本文档的自检结论

1. 2×2 能拆开 OCR 和 state，优于原先 baseline/+OCR/+OCR+state 三臂。
2. 贡献已收紧到 screen-text identity、hard provenance write validation 和 occurrence lifecycle；没有把已有的 hierarchical/entity memory 重新包装成 novelty。
3. `PV_ref` 与真实视觉支持被明确分开，避免把“指针存在”误写成“事实正确”。
4. state carry-forward 与新写入被明确分开，避免 provenance gate 把合法持续状态全部删除。
5. frozen10+0115 明确只作 dev；最终 claim 必须来自预先冻结 heldout。
6. 当前最小可行实现只新增四个机制模块，不复制 MAGIC/WorldMM/ReflectWorld 的完整系统。

这份文档可以作为 WP17-0 的执行入口，但在 WP17-0 产出 surface audit、heldout 和 annotation SHA256 前，不能视为已预注册协议。
