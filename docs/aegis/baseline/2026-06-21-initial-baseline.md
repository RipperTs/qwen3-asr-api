# 初始架构基线

日期：2026-06-21

## 项目结构

- `asr-service/app/main.py`：服务装配入口，按 `standard` / `vllm` 模式加载引擎、任务管理器和路由。
- `asr-service/app/runtime/task_manager.py`：离线任务队列、任务状态、取消、超时和清理。
- `asr-service/app/pipeline/asr_pipeline.py`：standard 模式离线处理流水线，包含转换、VAD、切块、ASR、标点、说话人和结果合并。
- `asr-service/app/engines/qwen_asr_engine.py`：standard GPU ASR 引擎封装。
- `asr-service/app/runtime/vllm_offline.py`：vLLM 模式离线处理器。
- `asr-service/app/engines/vllm_asr_engine.py`：vLLM 引擎封装。
- `docs/architecture.md`：当前架构说明。
- `docs/configuration.md`：启动参数、配置文件和内置常量说明。

## 当前性能相关事实

- standard 离线任务由 `TaskManager` 单工作线程串行处理。
- `TaskManager` 内部执行池固定为 `ThreadPoolExecutor(max_workers=1)`。
- standard GPU `QwenASREngine` 对 `transcribe` / `batch_transcribe` 使用 `_infer_lock` 串行化，原因是 `Qwen3ASRModel.generate` 非线程安全。
- VAD、标点、说话人 embedding 等共享推理入口也由各自 engine 的 `_infer_lock` 串行保护；多离线 worker 可并发推进 pipeline，但不能绕过这些 engine 边界直接并发调用共享模型实例。
- 当前 `ASRPipeline` 的 batch 只来自单个任务内部切出的 chunk，不合并多个用户任务的 chunk。
- 实时与离线共用 ASR 引擎，实时优先通过 `RealtimePriorityGate` 和离线 batch 缩小实现让路。
- vLLM 模式同样声明进程内同步推理，提升 `concurrency` 不带来吞吐收益。

## 主要约束

- 不能绕过单模型推理锁直接并发调用同一模型实例。
- 增加模型实例会复制模型权重，占用额外 GPU 显存。
- 生产并发目标应优先通过单模型动态合批提升吞吐，而不是简单扩大线程池。

## 已识别架构缺口

- 离线任务状态管理和完整处理流水线耦合，无法跨任务合批。
- GPU 推理入口缺少全局调度器，无法把多个任务的 chunk 组成一个批次。
- 配置只暴露队列长度和单任务内 batch，缺少调度等待窗口、活跃任务数、预处理并发等生产吞吐旋钮。
