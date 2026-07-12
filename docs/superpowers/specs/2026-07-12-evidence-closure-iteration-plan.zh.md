# VCAH 证据闭环与候选调度迭代计划

## 1. 背景与当前基线

当前 Virtual Video 多轮框架已经从“拿到局部画面后直接宣称 grounded”推进到“能够区分 forced choice 与 grounded answer，并暴露证据链缺口”。最近一组 VideoMME Long 五题回归结果为：

| 指标 | 结果 |
| --- | --- |
| forced-choice accuracy | 3/5 |
| grounded / verified | 0/5 |
| 正确选项 | 648-3、742-3、799-3 |
| 错误选项 | 702-1、839-1 |
| accepted investigations | 36 |
| evidence records | 56 |

这说明当前问题已经不只是视觉模型“看不清”，而是探索循环内部存在三类系统性损耗：

1. 候选已经召回，但没有被公平调度和验证。
2. 画面或模型文本已经包含答案事实，但没有稳定写入结构化 evidence。
3. 关键条件仍未闭环，Reasoner 已提前进入 answer，后续回合被 gate reject 消耗。

本轮目标不是继续加严最终 gate，也不是针对五道题添加规则，而是建立一个紧凑、可复用的证据闭环：

```text
Query contract
  -> navigation candidates
  -> visual observation
  -> structured facts
  -> condition state
  -> targeted repair or answer
```

## 2. 对 review 建议的判断

### 2.1 应直接吸收

以下诊断有明确代码和轨迹证据，应进入近期实现：

- 拆分 `retrieval_ready`、`choice_ready`、`grounded_ready`。
- ASR 搜索结果只能作为导航候选，不能直接满足 measurement、identity、causal 等语义条件。
- 为 ASR 命中建立有生命周期的 candidate registry，避免高价值候选被模型选择性忽略。
- Investigator 的结构化 JSON 失败必须显式暴露，并允许一次修复及保守 fallback。
- relation condition 必须检查 relation type、subject、object，不能由任意 supported relation 代替。
- audit 应围绕选中答案尚未支持的 claim atoms，而不是重复验证整个题干。
- cache reuse 的运行状态应与 semantic satisfaction 分开。

### 2.2 需要收敛后吸收

以下方向正确，但不宜第一版做成复杂子系统：

- Typed conditions：先使用一个紧凑的 `GapCondition` schema 加类型字段，不立刻拆成多套类层级。
- Candidate scheduler：提供候选覆盖提醒和有限 override，不让框架完全接管 Reasoner 的任务选择。
- Measurement fallback：只抽取显式数字、单位和时钟，不从自由文本猜测人物、事件或因果关系。
- `choice_ready`：第一版表示“存在可比较的 answer evidence / 模型给出候选”，不声称框架能独立判断 option ranking confidence。
- Readiness repair：有明确 repair task 时继续探索；没有可执行修复时允许模型保留 best choice，避免框架卡死。

### 2.3 暂缓到后续

以下能力有价值，但工程量和语义风险较高，不进入首轮 P0：

- 完整跨窗口 identity graph。
- 通用 derived-fact DSL。
- 专用倒计时插值、比分状态机和多帧 OCR 共识。
- 完整的 consumption ledger 与复杂累计推理。
- 自动学习候选分数或训练调度策略。

这些能力应在 P0/P1 证明通用闭环有效后，以插件式 operator 增加。

## 3. 五个 case 暴露的问题

### 3.1 648-3：视觉已读到强数值，但结构化事实丢失

轨迹中先观察到 `>23 trillion`，该事实不能蕴含选项 `>25 trillion`。后续观察文本已经出现 `30 quintillion`，但 Investigator 输出 `measurements=[]`，导致正确答案只能 forced，无法 grounded。

主要问题：

- structured output 缺失时没有 repair/fallback。
- scale normalization 不支持 quadrillion/quintillion。
- MeasurementFact 缺少 `quantity_type`，无法区分 diameter 与 count。
- option claim 仍缺少规范化的数值蕴含比较。

通用修复：结构化解析状态、显式数值 fallback、quantity binding、option numeric entailment。

### 3.2 702-1：正确候选已召回，但调度偏向单一假设

ASR 最早召回了 `dog + father` 窗口，也召回多个 firework 窗口。Reasoner 连续检查 firework 路径，却没有检查区分度更高的 dog/father 候选，最终错误选择 A，gold 为 D。

主要问题：

- navigation hint 没有候选生命周期。
- 全局 top-k 和自由调度可能集中在同一 option hypothesis。
- 未检查高价值候选不会在 finalization 前显式暴露。
- 事件人物与后续人物身份没有建立关系。

通用修复：option-stratified candidate registry、unseen candidate dashboard、有限调度 override。完整 identity graph 后置。

### 3.3 742-3：局部感知改善，但关键时钟状态未闭环

系统已经正确拒绝错误 crop，并观察到 8:13 和 2:57，forced choice 从 A 修正为正确 B；但从未直接观察到关键 5:58，因此 grounded gate 正确拒绝。

主要问题：

- `8:13` 和 `2:57` 的 measurement normalization 不一致。
- 关键 condition 未满足时仍过早 answer。
- 普通窗口搜索没有利用倒计时单调性缩小范围。

通用修复：统一 `M:SS -> seconds`、readiness 驱动 repair。倒计时定位作为后续 temporal operator，不在 P0 写入题目特例。

### 3.4 799-3：视觉链完整，但 JSON 截断和过宽 audit 阻断 grounding

模型观察已包含 `lipstick -> fake spots -> stretcher/ambulance`，足以支持“如何离开飞机”；但 fenced/truncated JSON 解析失败后，raw JSON 被塞入 verbatim，target/relation/conditions 为空。随后 audit 又要求证明“岛在下方”等题干背景。

主要问题：

- structured parse failure 静默退化为空事实。
- audit contract 把题干前提和答案必要原子混在一起。
- 相同 audit 可能重复执行。

通用修复：一次 JSON repair、保守 event fallback、selected-option claim atom audit、audit fingerprint 去重。

### 3.5 839-1：局部数值存在，但身份、边界和累计语义缺失

系统只形成 310-calorie measurement，且 delta/cumulative semantics 未知；没有确认是同一人物、是否发生在 meeting boundary 前，也没有形成可靠聚合，最终 forced A 错误，gold 为 C。

主要问题：

- ASR 搜索曾错误满足 calorie semantic condition。
- measurement 缺少 event/person/object binding。
- delta、cumulative、boundary 未闭环时仍尝试回答。
- 后续搜索围绕 burger 关键词重复，而非修复缺失字段。

通用修复：ASR semantic invariant、typed measurement gap、targeted field repair。完整 identity-bound accumulation 放在 P2。

## 4. 设计原则

### 4.1 框架提供状态，不替代模型推理

框架应回答：

- 已经发现哪些候选？
- 哪些候选尚未观察？
- 哪些事实已结构化？
- 哪些关键条件仍未知？
- 哪个最小修复动作可以补齐缺口？

框架不应直接决定开放语义问题的答案，也不应因为模型生成的 contract 不完整而永久拒答。

### 4.2 ASR 是视频侧的 lexical grep

允许对全局 ASR 做纯词法搜索，返回 cue、时间和 lineage。ASR 可以：

- 生成 navigation candidate。
- 满足显式 `lexical_navigation` condition。

ASR 不可以单独满足：

- 视觉存在与身份。
- 屏幕文字或数值读取。
- 动作、因果和空间关系。
- 选项最终 grounding。

### 4.3 双轨答案保持不变

- `best_choice`：用于与 VideoMME forced-choice accuracy 对比。
- `grounded_answer`：仅在引用证据支持必要 claim atoms 时输出。
- 预算耗尽后允许 forced choice，不返回机械式拒答作为唯一结果。

### 4.4 不允许证据静默丢失

模型输出解析失败时，必须记录：

```text
parsed | repaired | fallback_extracted | failed
```

禁止出现“verbatim 中有完整事实，但 structured facts 为空且无错误标记”的状态。

## 5. 目标架构

### 5.1 Completion dashboard

以紧凑映射或 dataclass 表达：

```python
CompletionStatus(
    retrieval_ready,
    choice_ready,
    grounded_ready,
    unresolved_critical_condition_ids,
    unsupported_claim_atom_ids,
    high_value_uninspected_candidate_ids,
)
```

语义：

- `candidate_available`：已有导航候选，但不代表已完成视觉检查。
- `retrieval_ready`：已有明确 dispatch 后产生、或可复用的视觉 observation；纯 ASR candidate 不计入。
- `choice_ready`：已有可解释的 best choice 或显式 option comparison，仅作为 dashboard 信息，不阻止预算耗尽后的 forced answer。
- `grounded_ready`：scope、critical conditions、required derived facts 和 selected-option atoms 均闭环。

`ready_for_answer` 暂时保留兼容字段，但其值等同 `grounded_ready`。

### 5.2 Navigation candidate registry

```python
NavigationCandidate(
    candidate_id,
    source_task_id,
    time_range,
    hypothesis_ids,
    discriminative_score,
    matched_terms,
    status,                 # unseen / inspected / supported / refuted / inconclusive
    resulting_evidence_ids,
)
```

调度规则：

1. 每个 option hypothesis 至少保留一个最高分候选。
2. 同一假设的第三个重复窗口，优先级低于另一假设的首个未检查候选。
3. Candidate ID 必须沿 `InvestigationTask.source_candidate_ids` 传入 Evidence；只有这条 provenance 链能把 candidate 标成 inspected/supported/refuted/inconclusive。时间重叠最多表示 `possibly_covered`。
4. finalization 前将高价值未检查候选显式提供给 Reasoner。
5. override 每轮最多占一个 task slot，避免框架接管全部探索。

### 5.3 Compact typed conditions

继续使用一个 `GapCondition`，增加可选字段：

```python
condition_type: auto | lexical_navigation | presence | measurement | relation | temporal
target_role: str
quantity_type: str
unit: str
relation_type: str
subject_role: str
object_role: str
required_relation: str
```

旧自然语言 condition 保持兼容；新 condition 一旦有 typed fields，就必须按字段精确校验。

### 5.4 Structured observation pipeline

```text
VLM response
  -> normal JSON parse
  -> balanced-object extraction
  -> one bounded JSON repair
  -> conservative text fallback
  -> normalized facts + parse status
```

fallback 允许：

- 显式数字、scale、单位。
- `M:SS` 时钟规范化。
- 明确列举的 event sequence。

fallback 禁止：

- 从 ASR 推断视觉身份。
- 从模糊文本猜因果或 same-entity。
- 缺少单位时凭选项强行绑定数值。

结构化恢复分为三个信任等级：

- `parsed/repaired`：恢复模型原本输出的结构。
- `fallback_extracted`：从 verbatim/summary 抽取显式数字或事件描述。
- `binding_status`：单独记录 `explicit/contextual/ambiguous/unbound`，禁止把抽取成功等同于语义绑定成功。

### 5.5 Claim-atom audit

只审计选中答案仍未支持的原子：

```text
selected option
  -> required claim atoms
  -> cited evidence support
  -> unsupported atoms only
  -> targeted audit tasks
```

使用 `(option, atom, quantized_range, method)` 作为 fingerprint；等价 audit 已得到相同结果时不再重复。

## 6. 分阶段实施

### Phase 0：轨迹与回归夹具

目标：先固定当前五个 case 的行为，保证后续增量可归因。

- 保存五题 compact replay fixture，不复制大日志和全部图片。
- viewer 增加 parse status、candidate status、readiness 三状态和 unsupported atoms。
- 添加指标聚合脚本。
- 不改变 Reasoner/Investigator 行为。

验收：相同 fixture 可离线生成一致 dashboard；当前 3/5、0/5 基线可复现。

### Phase 1：Readiness 与 ASR invariant

目标：消除“局部 evidence -> 过早回答”和“ASR 命中 -> 语义条件满足”。

- 拆分三种 readiness。
- condition ledger 在同一 condition/context 内按 evidence lineage 单调归并：unknown 不覆盖 satisfied/refuted，矛盾有效证据进入 conflicted，cache reuse 后重新按当前 contract 评估 condition。
- ASR 仅满足 `lexical_navigation`。
- 如果 answer 时存在明确未闭环 condition 且可生成 repair task，在同一轮转为 investigate。
- 无明确 repair 或预算耗尽时保留 best choice，不硬卡死。
- cache reuse 状态改为 `reused`，semantic condition 仍需重新评估。

主要文件：

- `src/vcah/multiround.py`
- `src/vcah/investigator.py`
- `src/vcah/evidence_primitives.py`
- `tools/run_virtual_videomme_interactive.py`

### Phase 2：候选 registry 与有限调度

目标：修复 702 类“召回正确但未检查”。

- ASR cluster 转为 candidate。
- 将 option lexical overlap 仅作为 hypothesis hint，不作为证据。
- 候选按 hypothesis 分层保留。
- 每轮最多注入一个未覆盖 hypothesis 的高价值 candidate task。
- Reasoner 输入显示 candidate 状态和未覆盖假设。
- 记录 `oracle_candidate_available_but_uninspected`。

### Phase 3：结构化解析修复与 measurement normalization

目标：修复 648、799 类“看到了但没写入事实”。

- balanced JSON extraction。
- 一次 bounded repair。
- parse status/error 写入 trace 和 evidence metadata。
- 增加 quadrillion/quintillion。
- 统一 game clock 为 seconds，并保留 raw text。
- MeasurementFact 增加 `quantity_type/predicate/object_id/event_id/extraction_source`。
- 显式数值 fallback；属性绑定必须有 question/task/visible context 支持。

### Phase 4：Typed relation 与 claim-atom audit

目标：减少 relation false positive 和 799 类过宽审计。

- relation type、subject、object 精确匹配。
- 将 selected option 编译成最小 claim atoms。
- 只为 unsupported atoms 创建 audit task。
- audit fingerprint 去重。
- `how` 区分 mechanism、cause、context，不再默认全部需要因果审计。

### Phase 5：可插拔 derived operators

在前四阶段稳定后再实现：

- numeric option entailment。
- scalar delta/cumulative aggregation。
- countdown monotonic navigation。
- temporal transition。
- identity-bound event ledger。

operator 必须由 query contract 能力类型触发，不能按 case ID 或具体答案触发。

## 7. 测试设计

### 7.1 单元测试

- ASR measurement/identity condition 永远保持 unknown。
- lexical navigation condition 可由 ASR 满足。
- typed relation 不接受错误 relation type 或 participant。
- non-full-video 局部 observation 不自动等于 grounded ready。
- candidate scheduler 先覆盖不同 hypothesis，再选重复窗口。
- truncated fenced JSON 能 repair；repair 失败时显式标记。
- `30 quintillion light-years` 规范为 `3e19 light_year`。
- `8:13` 规范为 `493 second/countdown_clock`。
- 相同 audit fingerprint 不重复 dispatch。

### 7.2 五题回归

固定重跑：

- 648-3：检查强 measurement 是否形成并正确蕴含 option。
- 702-1：检查 dog/father candidate 是否被视觉验证。
- 742-3：检查未看到 5:58 时仍不虚假 grounded，同时减少无效 answer 尝试。
- 799-3：检查动作链结构化、audit 不再要求无关背景。
- 839-1：检查 ASR 不越权、未知累计语义触发定向修复。

### 7.3 十题泛化组

五题只用于回归机制，不足以证明准确率提升。随后从 VideoMME Long 选择 10 道不同类型题：

- OCR/小目标读取。
- 时序边界。
- 因果/机制。
- 人物身份。
- scalar quantity。
- event/entity count。

使用并发上限 16 和指数退避。对比：

- Gemini 512-frame direct baseline。
- 当前 agent baseline。
- 每个 phase 后 agent 结果。

## 8. 验收指标

首轮重点不是减少 investigation 数，而是提高有效闭环率：

- `premature_answer_attempts` 下降。
- `semantic_condition_satisfied_by_navigation == 0`。
- `structured_parse_success_rate` 提升。
- `visible_in_verbatim_but_missing_structured_fact` 下降。
- `oracle_candidate_available_but_uninspected` 下降。
- `duplicate_audit_rate` 下降。
- `critical_condition_closure_at_finalization` 提升。
- forced-choice accuracy 不低于当前 3/5。
- grounded/verified 从 0/5 提升，同时不通过降低 forced accuracy 换取。

阶段性预期而非硬承诺：

- Phase 1/2 最可能改善 702 的 choice 和全体 premature answer。
- Phase 3 最可能使 648、799 获得真实 grounded evidence。
- Phase 4 最可能解除 799 的过宽 audit。
- 742 的直接 grounding 可能仍需 Phase 5 countdown operator。
- 839 的完整 grounding 可能仍需 identity-bound accumulation。

## 9. 非目标与风险控制

本轮不做：

- 按 case ID 分支。
- 把 gold、target interval、distractor role 暴露给模型。
- 预先生成视频语义摘要或 embedding memory。
- 将 ASR 当作最终视觉证据。
- 无限 repair loop。
- 因 contract 不完整而永久拒绝 forced answer。

主要风险及控制：

- 框架过严：保留 best-choice/forced-answer 双轨，repair 有界。
- typed condition 编译错误：记录 contract error，允许 Reasoner 修正并留痕。
- candidate override 干扰模型：每轮最多占一个 task slot，支持消融关闭。
- fallback 误绑定：只抽显式值，属性绑定要求上下文一致并记录 extraction source。
- dashboard 膨胀：只保留 unresolved、conflict、high-value unseen 和最近 verified 摘要。

## 10. 建议提交顺序

1. `test: add five-case evidence closure replay fixtures`
2. `refactor: split choice and grounding readiness`
3. `fix: keep ASR search navigation-only`
4. `feat: track and schedule contrastive navigation candidates`
5. `fix: repair structured observations and normalize measurements`
6. `refactor: audit unsupported answer atoms only`
7. `eval: compare five-case and diverse ten-case regressions`

每个提交单独运行相关单测和五题 replay；行为改动不合并成一次大提交，以便判断准确率变化来自哪里。
