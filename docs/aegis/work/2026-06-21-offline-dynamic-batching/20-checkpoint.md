# TodoCheckpointDraft

日期：2026-06-21

## Current Todo

- [x] 设计规格已确认。
- [x] 实现计划已写入。
- [x] 阶段 1：拆分 `ASRPipeline` 并保持行为不变。
- [x] 阶段 1 review 与测试。
- [x] 阶段 2：新增并接入动态合批调度。
- [x] 阶段 2 review 与测试。
- [x] 阶段 3：配置、文档、观测和聚焦验证。
- [x] 最终审计。

## Active Slice

完成候选：代码、测试、文档和 Aegis workspace 结构均已验证。

## Evidence Refs

- `docs/aegis/specs/2026-06-21-offline-dynamic-batching-design.md`
- `docs/aegis/plans/2026-06-21-offline-dynamic-batching-plan.md`
- `PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/pipeline/test_asr_pipeline_pure.py -q`：26 passed
- `PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/runtime/test_offline_batch_scheduler.py -q`：7 passed
- `PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/runtime/test_offline_batch_scheduler.py asr-service/tests/unit/pipeline/test_asr_pipeline_pure.py asr-service/tests/unit/runtime/test_task_manager.py asr-service/tests/unit/api/test_main_serve_mode.py asr-service/tests/unit/utils/test_arg_schema.py -q`：244 passed
- `PYTHONPATH=asr-service .venv/bin/python -m py_compile asr-service/app/runtime/offline_batch_scheduler.py asr-service/app/pipeline/asr_pipeline.py asr-service/app/runtime/task_manager.py asr-service/app/main.py asr-service/app/utils/arg_schema.py`：通过
- `git diff --check`：通过
- `PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit -q`：全量 unit 通过
- `.venv/bin/python /Users/wyf/.codex/aegis/scripts/aegis-workspace.py bundle --root /Users/wyf/PythonProject/qwen3-asr-service --work 2026-06-21-offline-dynamic-batching`：通过
- `.venv/bin/python /Users/wyf/.codex/aegis/scripts/aegis-workspace.py check --root /Users/wyf/PythonProject/qwen3-asr-service`：通过

## Blocked On

无。

## Next Step

交付说明；真实 GPU 压测作为后续生产调参步骤。

## DriftCheckDraft

- Scope：仍在动态合批生产化目标内。
- Compatibility：默认 `offline_worker_count=1` 保持旧串行路径；`>1` 才启用多 worker 和全局 ASR scheduler。
- Retirement：旧串行路径保留为默认兼容路径；批量异常仍回退逐条识别，避免牺牲准确度。
- Decision：continue（完成候选，等待用户侧生产压测/调参）。
