# TaskIntentDraft

日期：2026-06-21

## Requested Outcome

按照已确认的离线 ASR 动态合批设计完成所有阶段。每完成一个阶段做一次 review。模型识别准确度是第一优先级，任何性能设计都不能以牺牲准确度为前提。

## Scope

- 阶段 1：拆分 `ASRPipeline`，保持串行行为和结果不变。
- 阶段 2：新增并接入全局 ASR 动态合批调度。
- 阶段 3：补配置、文档、观测和验证。

## Non-goals

- 不优先做模型实例池。
- 不绕过模型推理锁并发调用同一模型实例。
- 不改变公开 API 响应结构。
- 不降低识别准确率或改变音频切块语义。

## Success Evidence

- 相关单元测试通过。
- 新增 scheduler 测试证明跨任务 chunk 可合批。
- review 记录每个阶段的风险和结论。
- 文档记录新配置和架构边界。

## BaselineReadSetHint

- `docs/aegis/baseline/2026-06-21-initial-baseline.md`
- `docs/aegis/specs/2026-06-21-offline-dynamic-batching-design.md`
- `docs/aegis/plans/2026-06-21-offline-dynamic-batching-plan.md`
- `asr-service/app/pipeline/asr_pipeline.py`
- `asr-service/app/runtime/task_manager.py`
- `asr-service/app/engines/qwen_asr_engine.py`

## ImpactStatementDraft

- 影响离线任务处理主链路。
- 影响任务状态、取消、超时、进度、持久化和服务装配。
- 需要保护实时优先逻辑。
- 需要保证 ASR 输入和结果解析等价。
