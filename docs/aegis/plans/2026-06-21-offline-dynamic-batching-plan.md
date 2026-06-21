# 离线 ASR 动态合批实现计划

## Goal

把离线 ASR 从“单任务完整 pipeline 串行执行”演进为“任务状态机 + 全局 ASR 动态合批调度器”，让多个离线任务可以共享一份模型权重并跨任务合批推理。模型识别准确度优先，任何阶段都不得改变音频切块、ASR 输入顺序、语言参数、对齐开关或结果解析语义。

## Architecture

目标架构来自 [离线 ASR 动态合批调度设计](../specs/2026-06-21-offline-dynamic-batching-design.md)：

```text
API 上传任务
  -> TaskManager 创建任务状态
  -> OfflinePreprocessWorker 转换 / VAD / 切 chunk
  -> ASRBatchScheduler 收集多个任务的 chunk
  -> QwenASREngine.batch_transcribe(batch_paths)
  -> TaskAccumulator 回填每个任务的 chunk 结果
  -> OfflineFinalizeWorker 标点 / 分句 / 说话人 / full_text
  -> TaskManager 标记完成
```

## Tech Stack

- Python 3.12
- pytest
- FastAPI
- 当前仓库 `.venv`
- standard GPU 路径以 mock 测试为主，避免真实模型、网络和长耗时依赖

## Baseline/Authority Refs

- [初始架构基线](../baseline/2026-06-21-initial-baseline.md)
- [离线 ASR 动态合批调度设计](../specs/2026-06-21-offline-dynamic-batching-design.md)
- [项目架构说明](../../architecture.md)
- [配置文档](../../configuration.md)

## Compatibility Boundary

- `/v1/asr`、`/v2/asr` 提交响应不变。
- `/tasks` 查询公开字段不变。
- 现有任务终态语义不变。
- 现有 `ASRPipeline.run` 在阶段 1 仍可直接串行执行。
- 动态合批不得牺牲识别准确度：同一个 chunk 必须以相同音频、相同语言、相同 `return_time_stamps` 语义进入 ASR；结果解析和排序必须等价。

## Verification

阶段 1：

```bash
PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/pipeline/test_asr_pipeline_pure.py -q
```

阶段 2：

```bash
PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/runtime/test_offline_batch_scheduler.py asr-service/tests/unit/runtime/test_task_manager.py asr-service/tests/unit/pipeline/test_asr_pipeline_pure.py -q
```

阶段 3：

```bash
PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit -q
```

## Architecture Integrity Lens

- Invariant：准确度优先，ASR 输入、切块、顺序、语言和 words 解析不变。
- Canonical owner / contract：`ASRBatchScheduler` 负责跨任务推理合批；`TaskManager` 负责任务状态；`ASRPipeline` 负责单任务语义的预处理和后处理。
- Responsibility overlap：避免让 `TaskManager` 继续直接拥有完整 pipeline 执行逻辑。
- Higher-level simplification：在调度层解决并发吞吐，不在模型层绕锁或复制实例。
- Retirement / falsifier：如果阶段 2 无法证明跨任务 batch 与旧单任务 batch 输出等价，则回退到阶段 1 解耦结构。
- Verdict：proceed。

## Plan Pressure Test

- Owner / contract / retirement：新增 scheduler owner，旧串行路径先保留作兼容和回退。
- Architecture integrity / higher-level path：动态合批在全局推理入口解决问题，符合设计。
- Verification scope：每阶段必须有新测试和 review。
- Task executability：先拆 pipeline，再实现 scheduler，最后接入配置和文档。
- Pressure result：proceed。

## Plan-Time Complexity Check

- Target files：`asr_pipeline.py`、`task_manager.py`、新增 runtime scheduler 文件、配置和测试。
- Existing size / shape signals：`asr_pipeline.py` 职责较重，`main.py` 装配逻辑较长。
- Owner fit：pipeline 拆分用现有 owner；scheduler 新增独立 owner；避免把调度逻辑塞入 `main.py`。
- Add-in-place risk：直接在 `TaskManager` 内写合批逻辑会混淆任务状态和 GPU 调度。
- Better file boundary：新增 `app/runtime/offline_batch_scheduler.py`，必要时新增 `app/runtime/offline_task_runner.py`。
- Recommendation：split task，先阶段 1 解耦。

## Tasks

### Task 1：拆分 ASRPipeline，保持串行行为不变

Files：

- 修改 `asr-service/app/pipeline/asr_pipeline.py`
- 修改或新增 `asr-service/tests/unit/pipeline/test_asr_pipeline_pure.py`

Why：

为阶段 2 提供可复用的预处理、ASR chunk 转换和后处理边界。该阶段不改变行为。

Impact/Compatibility：

公开 API 不变，`ASRPipeline.run` 签名不变，旧任务执行路径不变。

Verification：

```bash
PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/pipeline/test_asr_pipeline_pure.py -q
```

Steps：

1. Write test：补充或确认单任务 batch 结果顺序、words offset、`split_after` 行为测试。
2. Verify RED：运行目标测试，若新增测试覆盖尚不存在的辅助方法，应先失败。
3. Minimal code：抽出 `prepare_audio_chunks`、`transcribe_chunks`、`build_segments_from_batch`、`finalize_segments` 等小函数，`run` 继续串行调用。
4. Verify GREEN：运行目标测试。
5. Review：检查 diff，确认没有改变 ASR 输入、batch 顺序、结果解析和清理行为。

### Task 2：新增 ASRBatchScheduler 纯 runtime 单元

Files：

- 新增 `asr-service/app/runtime/offline_batch_scheduler.py`
- 新增 `asr-service/tests/unit/runtime/test_offline_batch_scheduler.py`

Why：

建立全局合批调度核心，不接入服务装配前先用 mock ASR 验证调度正确性。

Impact/Compatibility：

不影响现有运行路径。

Verification：

```bash
PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/runtime/test_offline_batch_scheduler.py -q
```

Steps：

1. Write test：覆盖 batch 满立即推理、等待超时小 batch 推理、多任务结果按 task/index 回填、取消跳过。
2. Verify RED：运行新测试，应因模块不存在失败。
3. Minimal code：实现 `ChunkJob`、`ChunkResult`、`ASRBatchScheduler`。
4. Verify GREEN：运行 scheduler 测试。
5. Review：确认 scheduler 是唯一跨任务 ASR owner，不引入模型并发调用。

### Task 3：接入任务状态机和动态合批执行路径

Files：

- 修改 `asr-service/app/runtime/task_manager.py`
- 新增或修改 `asr-service/app/runtime/offline_task_runner.py`
- 修改 `asr-service/app/main.py`
- 修改相关单元测试

Why：

让离线任务实际走预处理并发、全局 ASR 合批、后处理完成的生产路径。

Impact/Compatibility：

内部状态可增加，但对外状态和响应字段必须保持兼容。

Verification：

```bash
PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit/runtime/test_offline_batch_scheduler.py asr-service/tests/unit/runtime/test_task_manager.py asr-service/tests/unit/pipeline/test_asr_pipeline_pure.py -q
```

Steps：

1. Write test：多任务并发提交时 mock ASR 看到跨 task 的同一 batch。
2. Verify RED：测试因未接入 scheduler 失败。
3. Minimal code：新增 runner 并让 `TaskManager` 可注入新的执行策略。
4. Verify GREEN：运行相关测试。
5. Review：检查取消、失败、清理、持久化回调和公开状态映射。

### Task 4：配置、文档和观测

Files：

- 修改 `asr-service/app/config.py`
- 修改 `asr-service/app/utils/arg_schema.py`
- 修改 `asr-service/config.example.yaml`
- 修改 `docs/configuration.md`
- 修改 `docs/architecture.md`
- 修改对应测试

Why：

暴露生产调优旋钮并记录使用边界。

Impact/Compatibility：

新增配置默认值必须保持旧行为尽量稳定；默认可先保守，避免短任务延迟突增。

Verification：

```bash
PYTHONPATH=asr-service .venv/bin/python -m pytest asr-service/tests/unit -q
```

Steps：

1. Write test：配置解析、默认值和 help 文案测试。
2. Verify RED：未加配置前失败。
3. Minimal code：加入配置、日志指标、文档。
4. Verify GREEN：运行单元测试。
5. Review：确认文档明确准确度优先、动态合批和实时优先边界。

## Risks

- 不同语言混合 batch 的上游行为需要验证；若不确定，scheduler 按 language 分组。
- 短任务可能增加 `offline_batch_wait_ms` 级别等待。
- 标点和说话人后处理线程安全性需要验证，不确定时保持单 worker。
- 动态合批增加状态机复杂度，取消和失败路径必须重点测试。

## Retirement

- 阶段 1 保留旧串行执行路径。
- 阶段 2 新 scheduler 先独立存在。
- 阶段 3 后如果动态合批路径稳定，旧的单 worker 完整 pipeline 执行路径可以作为兼容回退保留一版，再评估移除。
