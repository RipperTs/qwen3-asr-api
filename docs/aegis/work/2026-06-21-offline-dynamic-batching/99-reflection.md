# Reflection

## Completion Candidate

- 已完成阶段 1：抽取 ASR 结果组装方法，保持串行和单任务 batch 行为不变。
- 已完成阶段 2：新增全局 `ASRBatchScheduler`，并让多离线 worker 共享单模型 batch 入口。
- 已完成阶段 3：新增配置、文档、启动日志和测试覆盖。

## Review Notes

- 准确度边界：动态合批不改变 VAD 切块、chunk 音频、language、对齐开关和结果解析；不同 `language` 不混批。
- 兼容边界：默认 `offline_worker_count=1` 保持旧串行路径；`>1` 才启用动态合批。
- 可靠性边界：scheduler batch 异常不会杀死后台线程；pipeline 对 scheduler 错误回退逐条识别。
- 复杂度边界：新增调度器是跨任务 ASR batch 的唯一 owner，TaskManager 仍只负责任务生命周期，ASRPipeline 仍只负责单任务处理和结果组装。

## Residual Risk

- 未在真实 48G/80G GPU 上做并发压测，最佳 `offline_worker_count`、`offline_asr_batch_size`、`offline_batch_wait_ms` 需要按实际音频长度和用户并发调参。
- OpenVINO CPU 引擎当前没有 `batch_transcribe`，本次动态合批主要服务 standard GPU QwenASREngine 的离线路径。

Method Pack output is evidence, not authority, and does not grant completion authority.
