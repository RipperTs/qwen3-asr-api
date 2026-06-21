# Proof Bundle - 2026-06-21-offline-dynamic-batching

## Method Pack Boundary

This proof bundle is an advisory Aegis Method Pack record. It does not determine evidence sufficiency, produce authoritative `GateDecision`, or grant `completion authority`.

## Task Intent

- Requested outcome: 按照已确认的离线 ASR 动态合批设计完成所有阶段；模型识别准确度第一优先级。
- Scope: standard 模式离线任务队列、ASR pipeline、全局 ASR scheduler、配置、文档和测试。

## Impact

- Compatibility boundary: 新能力只在 offline_worker_count > 1 时启用；公开 API 响应结构保持不变。
- Non-goals:
- 模型实例池
- 真实 GPU 压测参数调优
- vLLM 模式改造

## Evidence Bundle Refs

- docs/aegis/work/2026-06-21-offline-dynamic-batching/evidence-bundle-draft.json

## Drift Check

- Scope status: inside requested dynamic batching scope
- Compatibility status: default serial behavior preserved; new path gated by offline_worker_count > 1
- Retirement status: old serial path intentionally retained as default compatibility path
- Advisory decision: continue
