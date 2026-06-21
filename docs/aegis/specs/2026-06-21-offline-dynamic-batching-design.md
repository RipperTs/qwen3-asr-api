# 离线 ASR 动态合批调度设计

日期：2026-06-21

## 背景

当前 standard 离线任务是单工作线程串行执行。即使 GPU 显存从 24G 增加到 48G 或 80G，现有架构仍然只能一次推进一个离线任务；同时，ASR 模型实例内部因 `Qwen3ASRModel.generate` 非线程安全必须加锁，不能通过简单增加线程数获得正确并发。

当前 batch 只来自同一个任务内部的多个 chunk。生产环境需要的是多个用户任务同时推进，并让 GPU 显存和算力用于更大的跨任务 batch，而不是加载多份模型权重。

## 目标

- 用一份 ASR 模型权重服务多个离线任务。
- 将多个任务的待识别 chunk 聚合成全局 batch，提升 GPU 吞吐。
- 保留单模型推理锁，避免并发调用非线程安全模型对象。
- 保持现有 `/v1`、`/v2` 离线 API、任务查询、取消和持久化契约兼容。
- 保留实时任务优先级，避免离线大 batch 长时间阻塞实时转写。
- 让 48G / 80G GPU 能通过更大的 batch 和更多活跃任务获得吞吐收益。

## 非目标

- 不做模型实例池作为首选方案，因为它会复制模型权重，显存利用率差。
- 不绕过 `QwenASREngine._infer_lock` 做同模型对象并发推理。
- 不在第一阶段引入 Redis、外部消息队列或多机调度。
- 不改变转写结果的公开响应结构。
- 不承诺一次重构同时覆盖 vLLM 模式；vLLM 可在 standard 方案稳定后单独评估。

## 第一性原则

- 不可破坏正确性：同一模型实例仍只能有一个推理调用在执行。
- 提升吞吐的正确位置是全局推理调度层，而不是路由层或单任务 pipeline 内部。
- 大显存应优先用于更大的有效 batch 和推理峰值余量，而不是重复加载权重。
- 任务状态、预处理、推理、后处理应解耦，便于观测和调度。

## 推荐架构

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

## 组件职责

### TaskManager

保留任务创建、查询、取消、持久化、TTL 清理职责。移除“直接执行完整 pipeline”的职责，改为维护任务状态机。

建议任务状态：

- `pending`：已提交，等待预处理。
- `preprocessing`：格式转换、时长检查、VAD、切 chunk。
- `queued_asr`：已有 chunk 等待 ASR 调度。
- `transcribing`：至少一个 chunk 已进入 ASR batch。
- `postprocessing`：ASR 完成，正在组装结果。
- `completed` / `failed` / `cancelled`：终态。

对外可继续把内部状态映射为现有 `pending` / `processing` / `completed` / `failed` / `cancelled`，避免破坏客户端。

### OfflinePreprocessWorker

负责 CPU / I/O 密集阶段：

- 上传文件转 WAV。
- 音频时长、大小和最小时长校验。
- FSMN-VAD 检测。
- 合并 VAD 段并切 chunk。
- 为每个 chunk 生成 `ChunkJob`。

该阶段可配置并发，不直接调用 ASR。

### ASRBatchScheduler

全局唯一的 ASR 推理调度器，是本设计的核心所有者。

职责：

- 接收来自多个任务的 `ChunkJob`。
- 按优先级、等待时间和 batch 大小组批。
- 调用 `QwenASREngine.batch_transcribe(audio_paths, language)`。
- 将每个结果回填给对应任务。
- 在每次 batch 前检查取消状态，跳过已取消任务的 chunk。
- 暴露队列深度、batch 大小、等待时间、推理耗时等观测指标。

调度策略：

- `offline_asr_batch_size`：最大 batch chunk 数，默认沿用当前 `ASR_BATCH_SIZE=32`。
- `offline_batch_wait_ms`：为了凑 batch 的最大等待时间，建议默认 `50ms` 到 `200ms`。
- `offline_max_active_tasks`：同时进入预处理 / ASR 的离线任务上限。
- 实时开启时，离线 batch 应受 `REALTIME_PRIORITY_OFFLINE_BATCH_SIZE` 或新配置约束，缩短单次 GPU 锁占用。

### TaskAccumulator

每个任务一个聚合对象，记录：

- chunk 总数和完成数。
- chunk 的 offset、duration、split 标记。
- ASR 文本和 words。
- 进度计算。
- 取消状态。

当任务所有 chunk 完成后，投递到后处理阶段。

### OfflineFinalizeWorker

负责 ASR 后处理：

- 按原顺序组装 segment。
- 按请求处理 `with_words`。
- 标点恢复。
- 说话人分离和声纹识别。
- 句子重组。
- 生成 `segments`、`full_text`、`warnings`。
- 清理临时文件。

该阶段可并发，但说话人和标点是否线程安全需要实现前验证；不确定时先用小并发或单独锁保护。

## 数据结构草案

```python
class ChunkJob:
    task_id: str
    index: int
    path: str
    offset_sec: float
    duration_sec: float
    language: str | None
    split_after: bool
    options: dict
    priority: int
    created_at: float
```

```python
class ChunkResult:
    task_id: str
    index: int
    text: str
    words: list[dict] | None
    error: str | None
```

## 配置项

建议新增：

- `offline_preprocess_workers`：离线预处理并发数，默认 `2`。
- `offline_finalize_workers`：离线后处理并发数，默认 `1`。
- `offline_asr_batch_size`：全局 ASR batch 上限，默认 `32`。
- `offline_batch_wait_ms`：凑 batch 最大等待时间，默认 `100`。
- `offline_max_active_tasks`：同时活跃离线任务上限，默认 `4`。
- `offline_scheduler_queue_size`：chunk 调度队列上限，默认按 `max_queue_size * average_chunks` 推导或显式配置。

已有 `max_queue_size` 继续表示任务提交队列上限，不再等同于 GPU 待推理 chunk 容量。

## 取消与超时

- `pending` / `preprocessing` 阶段取消：尽快停止并清理文件。
- `queued_asr` 阶段取消：从逻辑上跳过未执行 chunk；物理队列可懒删除。
- `transcribing` 阶段取消：当前 batch 无法中断，batch 返回后丢弃该任务后续结果并进入 `cancelled`。
- 超时应按任务总耗时计算，同时记录卡在哪个阶段。

## 实时优先

实时与离线仍共享 ASR 模型锁。为避免离线大 batch 阻塞实时：

- 实时有活跃请求时，离线 scheduler 降低 batch 上限。
- 离线 scheduler 每个 batch 前调用现有 `RealtimePriorityGate`。
- 后续可增加高优先级 chunk 队列，但第一阶段不改变实时协议。

## 兼容性边界

- `/v1/asr`、`/v2/asr` 提交响应不变。
- `/tasks` 查询响应保持现有字段。
- 内部新增状态可以映射到现有公开状态。
- 任务持久化表结构若不扩展，至少保留现有状态和进度；若扩展阶段字段，应提供向后兼容迁移。
- Web UI 可先不展示新增内部状态，只显示现有状态与进度。

## 实施分期

### 阶段 1：结构拆分但保持单任务语义

- 从 `ASRPipeline.run` 中拆出预处理、ASR 输入构造、后处理函数。
- 保持现有串行执行路径不变。
- 补单元测试，确保结果结构不变。

### 阶段 2：引入全局 ASRBatchScheduler

- 新增 chunk job / result / accumulator。
- TaskManager 改为状态机驱动。
- 多任务 chunk 合批调用 `batch_transcribe`。
- 支持取消、失败和进度回填。

### 阶段 3：生产调优与观测

- 增加配置项和配置文档。
- 增加日志指标：平均 batch 大小、队列等待、ASR 耗时、任务端到端耗时。
- 做压测脚本：多个短音频、多个长音频、混合实时与离线。

## 测试策略

- 单元测试：
  - scheduler 在 batch size 达标时立即推理。
  - scheduler 在 wait timeout 后小 batch 推理。
  - 多任务 chunk 结果按 task_id / index 正确回填。
  - 取消任务不会进入后处理或最终 completed。
  - ASR 异常只失败相关任务或相关 chunk，不能卡死 scheduler。

- 集成测试：
  - 多个离线任务并发提交，状态均能完成。
  - 公开 API 响应结构与旧路径一致。
  - 开启任务持久化时，终态能落库。
  - 实时开启时，离线 batch 上限被压低。

- 性能验证：
  - 对比旧串行路径和新动态合批路径的总吞吐。
  - 记录 24G / 48G / 80G 下推荐 `offline_asr_batch_size`。
  - 验证长音频和短音频混合时不存在任务饥饿。

## 风险

- 跨任务合批会增加单个短任务的等待延迟，需要通过 `offline_batch_wait_ms` 控制。
- 后处理阶段如果复用非线程安全模型，也需要独立锁或单 worker。
- 任务状态机复杂度上升，需要严格测试取消、失败、清理和持久化。
- 如果 qwen_asr 的 `batch_transcribe` 对不同语言混合支持有限，scheduler 需要按 language 分组。

## 决策

采用“单模型全局动态合批调度器”作为生产化性能优化主线。模型实例池只作为后续扩展选项，用于显存足够且单模型 batch 已无法继续提升吞吐的场景。
