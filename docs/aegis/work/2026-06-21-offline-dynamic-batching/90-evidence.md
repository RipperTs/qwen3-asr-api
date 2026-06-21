# EvidenceBundleDraft

日期：2026-06-21

## Evidence

- 阶段 1 RED：`PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/pipeline/test_asr_pipeline_pure.py::test_build_segments_from_batch_preserves_offsets_words_and_split_marker -q`
  - 结果：失败，原因是 `ASRPipeline` 尚无 `_build_segments_from_batch`。
- 阶段 1 GREEN：同一测试通过。
- 阶段 1 regression：`PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/pipeline/test_asr_pipeline_pure.py -q`
  - 结果：23 passed。
- 阶段 1 review：diff 仅抽取 segment 构造辅助方法；`batch_transcribe` 输入、语言参数、结果顺序、words offset、空文本过滤和 fallback 行为不变。
- 阶段 2 RED：`PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/runtime/test_offline_batch_scheduler.py::test_scheduler_batches_chunks_from_multiple_tasks -q`
  - 结果：失败，原因是 `app.runtime.offline_batch_scheduler` 尚不存在。
- 阶段 2 GREEN：`PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/runtime/test_offline_batch_scheduler.py -q`
  - 结果：7 passed。
- 阶段 2 接入验证：
  - `test_transcribe_batched_can_use_global_scheduler` 验证 pipeline 不直接调用 `asr.batch_transcribe`，而是提交 `ChunkJob` 给全局 scheduler。
  - `test_transcribe_batched_falls_back_when_scheduler_chunk_errors` 验证 scheduler 返回错误时，pipeline 回退到 `asr.transcribe`，不直接牺牲识别结果。
  - `test_scheduler_groups_by_language_to_preserve_asr_language_semantics` 验证不同 `language` 不混入同一 ASR 调用。
  - `test_scheduler_submit_many_preserves_single_task_batching` 验证单任务的一组 chunk 优先保持在同一 batch。
- 阶段 2 review：
  - `QwenASREngine` 已有 `_infer_lock`，新增多 worker 不会并发调用非线程安全的模型生成；吞吐提升来自跨任务合批，而不是复制模型实例。
  - scheduler 后台线程捕获 batch 推理异常并返回 chunk error，避免线程死亡导致任务永久等待。
  - 默认 `OFFLINE_WORKER_COUNT=1` 不启用 scheduler，避免单用户路径引入 `batch_wait_ms` 等待。
- 阶段 3 配置/文档/观测：
  - 新增 `--offline-worker-count`、`--offline-asr-batch-size`、`--offline-batch-wait-ms`，同一 schema 覆盖 CLI、YAML 校验和配置日志。
  - `config.example.yaml`、`docs/configuration.md`、`docs/configuration_EN.md`、`docs/architecture.md`、`docs/architecture_EN.md` 已同步说明准确度边界。
  - scheduler 每次成功 ASR batch 记录实际 batch size、language 和累计 batch 数。
- 聚焦回归：
  - `PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/pipeline/test_asr_pipeline_pure.py -q`：26 passed。
  - `PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/runtime/test_offline_batch_scheduler.py -q`：7 passed。
  - `PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/runtime/test_offline_batch_scheduler.py asr-service/tests/unit/pipeline/test_asr_pipeline_pure.py asr-service/tests/unit/runtime/test_task_manager.py asr-service/tests/unit/api/test_main_serve_mode.py asr-service/tests/unit/utils/test_arg_schema.py -q`：244 passed。
  - `PYTHONPATH=asr-service .venv/bin/python -m py_compile asr-service/app/runtime/offline_batch_scheduler.py asr-service/app/pipeline/asr_pipeline.py asr-service/app/runtime/task_manager.py asr-service/app/main.py asr-service/app/utils/arg_schema.py`：通过。
  - `git diff --check`：通过。
- 最终验证：
  - `PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit -q`：全量 unit 通过，退出码 0；仅有既有 `StarletteDeprecationWarning`。
  - `.venv/bin/python /Users/wyf/.codex/aegis/scripts/aegis-workspace.py bundle --root /Users/wyf/PythonProject/qwen3-asr-service --work 2026-06-21-offline-dynamic-batching`：生成 `proof-bundle.md`，退出码 0。
  - `.venv/bin/python /Users/wyf/.codex/aegis/scripts/aegis-workspace.py check --root /Users/wyf/PythonProject/qwen3-asr-service`：通过，退出码 0。

## Not Covered Yet

- 真实 GPU 模型并发压测和最优参数调优；需要生产环境或同等硬件执行。
