# MM-Lifelong Slot Memory Construction 执行计划 v4

## 0. 本版决策

本版取代 `mmlifelong-slot-memory-construction-final-plan-v3-20260830.md` 的执行部分，但不修改已经冻结的 `cbf21d1` 30 秒协议。

核心问题固定为：

> 在相同视觉/OCR evidence 与相同 600-token 历史预算下，有生命周期的 slot capsule 是否优于直接携带上一段 caption？

核心效应：

```text
Delta_slot_over_free = E1C2 - E1C1
```

数据现实已经在 KML 上核实：Week 有 200 条题目和完整视频；Month 只有 train/val 标注，没有 Month 视频。因此 Month 从关键路径移除，Week 改为预先冻结的 dev60/query-holdout140。

## 1. 五项开跑前修正

### 1.1 两个逻辑阶段，不绑定两次调用

方法定义包含两个逻辑阶段：

```text
Perception -> grounded observations
Slot maintenance -> slot operations
```

它们可以由一次模型调用输出：

```json
{
  "observations": [],
  "slot_operations": []
}
```

Runtime 必须先校验 observations，再校验并执行 slot operations。Slot operation 不得引入 observations 中不存在且没有历史 provenance 的事实。

默认执行策略是 single-call fusion。只有结构 canary 失败时才允许退回双调用；不能根据 endpoint 好坏选择调用方式。论文 claim 是两阶段契约，不是调用次数。

### 1.2 只冻结 token budget，不设 slot 数上限

冻结：

```text
Bctx = 600 tokenizer tokens
```

取消 `Smax=12`。Canary 不允许调整 `Bctx`。Capsule 超预算时：

1. 已 CLOSE/ARCHIVE 且最久未验证的 working slot 优先 EVICT；
2. active slot 不得静默截断；
3. 仍超预算则结构失败并进入固定 repair，不允许运行时扩大预算。

ARCHIVE/EVICT 只移出下一个 chunk 的 working context，不删除 SER 或长期记忆。

### 1.3 `active_entities` 收窄为 `active_participants`

冻结 slots：

```text
location
active_encounter
active_participants
equipped_or_held_item
recent_state_change
occurrence_counter
current_activity
```

`active_participants` 只允许保存正在参与当前 active event/encounter 的实体。仅仅被看到、被 OCR 读到或被提及的普通实体写入 SER，但不得进入 working slot。

### 1.4 Never-forget 分成两个版本

```text
never_forget_unbounded
  只报告 token growth / context-window pressure / scalability；
  不参与最终质量比较。

never_forget_budgeted
  同样使用 600-token budget；
  禁止语义 CLOSE/ARCHIVE/EVICT；
  超限时按最老 working entry 做机械 FIFO 移出；
  SER 与长期记忆仍不删除。
```

质量对比使用 `never_forget_budgeted`，从而只比较生命周期策略与机械 FIFO，不混入 prompt 长度。

### 1.5 30 秒实验只作历史方向性对照

`cbf21d1` 的 30 秒 2x2 使用旧 state policy，120 秒使用 slot lifecycle v2。因此两者不能支持严格粒度因果 claim。

30 秒冻结实验可完成并归档为 historical/directional control。若最终必须提出 granularity claim，另跑一个小型 matched control：

```text
30s slot-v2 vs 120s slot-v2
```

两边使用相同 evidence、slot schema、lifecycle、token budget 和调用策略。

## 2. 120 秒主实验

### 2.1 三个第一阶段关键臂

| Arm | 当前 chunk | OCR evidence | 历史 context |
|---|---|---|---|
| E1C0 | frames/ASR | yes | none |
| E1C1 | frames/ASR | yes | previous caption，截断到 600 tokens |
| E1C2 | frames/ASR | yes | slot capsule，最多 600 tokens |

三个臂共享相同 120 秒 segment、frames、ASR 和 frozen OCR packet。执行顺序按 segment 循环平衡。

第一阶段只回答 `E1C2 - E1C1`。Day 局部 timeline 约 114 segments，single-call 基础调用为：

```text
114 x 3 = 342 calls
```

若 slot-over-free 接近零，立即收缩为 dense evidence + bounded previous caption，不补完整 factorial。

### 2.2 第二阶段完整 factorial

只有第一阶段通过预注册 gate，才补：

```text
E0C0 / E0C1 / E0C2
```

形成 evidence 两水平 x context 三水平。`E0` 与 `E1` 只差 OCR evidence；所有其他输入、预算和调用策略相同。

### 2.3 Token matching

E1C1/E0C1 使用与 capsule 相同 tokenizer 和 600-token 上限，保留上一段 caption 的尾部最近信息。报告实际 token 分布，并要求：

```text
abs(mean_tokens(C1) - mean_tokens(C2)) / 600 <= 0.10
```

这只是 aggregate gate；逐 segment 同时记录 token difference，避免少数长 prompt 被均值掩盖。

## 3. Week-dev60 / Week-holdout140

### 3.1 为什么改用 Week subset

Month 视频当前不可用，不能作为 retrieval/QA 调参前置。Week 已完整落盘，且题材与 Day 不同，适合检验 slot/state 是否比 OCR 更具有跨域性。

Week 的 200 个 query 共享同一套约 51.9 小时视频。故 60/140 是 query split，不是 video split：

```text
Week-dev60
  允许 method selection；
  可使用这 60 题的 official intervals 离线构造开发窗口；
  question/options/answer/interval/case_id 均不得进入模型输入。

Week-holdout140
  方法冻结前禁止访问结果、题目和 official intervals；
  只在 full Week question-blind memory 构建完成后一次性评测；
  只能支持 query-level holdout claim，不能写 unseen-video claim。
```

### 3.2 冻结抽样规则

抽样只使用 `case_id + question_type`，固定 seed `20260830`，按 question type 做 largest-remainder 30% 配额，然后在每层内按 SHA256 rank 取样。不得使用答案、clue、模型输出或先前结果。

已核实的 Week 类型分布与 dev60 配额：

| Type | All | Dev |
|---|---:|---:|
| Attribute Recognition | 8 | 2 |
| Causal Reasoning | 17 | 5 |
| Counting | 59 | 18 |
| Entity Recognition | 23 | 7 |
| Event Recognition | 17 | 5 |
| Event Tracking | 6 | 2 |
| Hallucination Detection | 10 | 3 |
| Language Content Recall | 15 | 4 |
| Social Interaction | 9 | 3 |
| Temporal Reasoning | 36 | 11 |

合计 200 / 60，剩余 140 为 complement。

### 3.3 Week 使用顺序

1. 在查看任何 Week outcome 前冻结 dev60/holdout140 manifest 与 SHA256。
2. Week-dev60 只跑 E1C0/E1C1/E1C2，验证 slot-over-free 是否跨域成立。
3. 允许根据 Week-dev60 调整全局方法，但不得产生 Week-only prompt、slot 或阈值。
4. 冻结最终方法 commit。
5. 对完整 Week 视频做 question-blind construction。
6. 一次性评测 Week-holdout140；此后不得回改方法再重报同一 holdout。

## 4. Gates 与 endpoints

### 4.1 Structural gates

- observations 在 slot ops 之前完成 schema validation；
- slot writes 有当前 observation 或历史 slot provenance；
- `active_participants` 全部属于当前 active event；
- slot version 单调、arm state 隔离；
- E1 三臂 OCR packet byte-identical；
- C1/C2 使用相同 tokenizer 与 600-token budget；
- capsule 无静默截断；
- ARCHIVE/EVICT 不删除 SER；
- response usage、finish reason 与 reasoning-token metadata 完整；
- question/options/answer/official interval/case ID 对 construction model 不可见。

Endpoint 数值不是 structural gate。

### 4.2 第一阶段 endpoints

主 endpoint：

```text
Delta_slot_over_free = E1C2 - E1C1
```

分别对以下 construction metrics 报 paired delta、bootstrap CI 和 W/T/L：

- anchor representation coverage；
- canonical entity coverage；
- relation/state coverage；
- occurrence/ordinal correctness；
- provenance validity；
- unsupported state write rate；
- token/context cost。

QA 只作第二阶段辅助指标，不能用单次 raw exact 波动决定 slot 机制。

### 4.3 预注册决策

```text
GO:
  slot-over-free 在至少两个预注册 representation/state endpoint 上为正，
  且 provenance / unsupported-write 不劣化，
  且成本满足 600-token gate。

NO-GO:
  slot-over-free 接近零或只表现为 QA 波动，
  或 provenance/unsupported-write 明显恶化。
```

GO 后才补完整六臂和生命周期消融。NO-GO 则收缩为 dense evidence + bounded previous caption。

## 5. 执行顺序

### Phase A：输入与协议

1. 完成 Day pixel-GT re-audit 和 OCR track reconstruction；
2. 归档旧 30 秒冻结实验，但不据其 endpoint 调整 120 秒方法；
3. 冻结 120 秒 slot-v2 协议；
4. 冻结 Week-dev60/holdout140 manifest；
5. 不等待 Month 视频。

### Phase B：最小机制验证

1. Day 3-segment x 3-arm structural canary：9 calls + retry；
2. Day 114-segment x 3-arm：342 base calls；
3. 根据预注册 GO/NO-GO 决定是否继续。

### Phase C：跨域开发

1. 用 Week-dev60 intervals 构造去重后的局部 120 秒 timeline；
2. 对该 timeline 跑 E1C0/E1C1/E1C2；
3. 只允许全局方法修改；
4. 冻结最终方法。

### Phase D：完整结果

GO 后补 Day 六臂、`never_forget_budgeted`、`never_forget_unbounded` 规模曲线、`no_retain`、`no_provenance_gate`；随后完整构建 Week 并一次性评测 holdout140。

## 6. 红线

- 不修改 `cbf21d1` 30 秒协议；
- 不因 canary endpoint 改 `Bctx=600`；
- 不把 logical roles 写成必须两次模型调用；
- 不让普通可见 entity 进入 `active_participants`；
- 不用 unbounded never-forget 做质量比较；
- 不用旧 30 秒 state policy 支持严格粒度因果 claim；
- Week-holdout140 在方法冻结前保持 sealed；
- 不把共享 Week 视频语料上的 140 query 写成 unseen-video external test；
- Month 视频未到位不阻塞当前机制实验，也不虚构 Month 结果。

## 7. 当前立即动作

1. 提交并测试 Week split 工具；
2. 在 KML 上零调用生成 Week-dev60/holdout140 manifest；
3. 写入 manifest SHA256 与数据可用性审计；
4. 另行冻结 120 秒 slot-v2 protocol；
5. 只启动 3-segment x 3-arm canary，不直接启动完整六臂。
