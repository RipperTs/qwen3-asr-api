import os
import shutil
import logging
import numpy as np
import soundfile as sf

from app.engines.vad_engine import VADEngine
from app.engines.punc_engine import PuncEngine
from app.pipeline.audio_preprocessor import convert_to_wav, get_audio_duration
from app.pipeline.sentence_segmenter import segment_sentences
from app.utils.result_parser import extract_text, extract_words
from app.config import (
    UPLOADS_DIR,
    AUDIO_CHUNKS_DIR,
    MIN_AUDIO_DURATION,
    MAX_AUDIO_DURATION,
)
import app.config as cfg

logger = logging.getLogger(__name__)


class ASRPipeline:
    def __init__(
        self,
        asr_engine,
        vad_engine: VADEngine,
        punc_engine: PuncEngine | None = None,
        speaker_engine=None,
        speaker_service=None,
        priority_gate=None,
        asr_scheduler=None,
    ):
        self.asr = asr_engine
        self.vad = vad_engine
        self.punc = punc_engine
        self.speaker = speaker_engine
        self.speaker_service = speaker_service    # 声纹库联动（None = 未启用）
        self.priority_gate = priority_gate
        self.asr_scheduler = asr_scheduler

    def run(
        self,
        audio_path: str,
        task_id: str,
        language: str | None = None,
        progress_callback=None,
        cancelled=None,
        identify_speakers: bool = False,
        options: dict | None = None,
    ) -> dict:
        """
        执行完整 ASR Pipeline。

        流程：
        0. ffmpeg 格式转换 → 16kHz WAV
        1. VAD 切片 → 处理用音频块（受 MAX_SEGMENT_DURATION 约束）
        2. 超长 segment 二次切分
        3. ASR 识别
        4. 标点恢复（可选）
        4.5 说话人分离（可选）：全局 VAD 时间轴聚类，先给 ASR 块打 speaker
        4.6 分句：把 ASR 处理块重组为句子（标点/停顿/说话人切换；max_segment 仅显式上限）
        4.7 句子级说话人标签精修 + 声纹识别（可选，进度 0.90→0.95）
        5. 合并结果，回算绝对时间戳
        6. 清理临时文件
        """
        wav_path = None
        chunk_dir = os.path.join(AUDIO_CHUNKS_DIR, task_id)

        # 按请求覆盖（缺省=服务端默认）；降级开关只能关、不能开启未加载模型
        opts = options or {}
        with_punc = opts.get("with_punc", True)
        with_words = opts.get("with_words", True)
        diarize = opts.get("diarize", True)
        max_segment = opts.get("max_segment")            # None → cfg
        id_threshold = opts.get("speaker_id_threshold")
        id_margin = opts.get("speaker_id_margin")
        # 合法但功能未启用的参数 → 软提示（不报错），随 result 返回
        warnings = []
        if opts.get("with_punc") is True and self.punc is None:
            warnings.append("with_punc")
        if opts.get("with_words") is True and not self.asr.align_enabled:
            warnings.append("with_words")
        if opts.get("diarize") is True and self.speaker is None:
            warnings.append("diarize")
        # 声纹识别真正能跑的前提：声纹库 + 说话人引擎 + diarize 同时就位（diarize 关时
        # 不聚类，identify/id 阈值全部失效）——任一缺失即软提示，避免静默丢弃
        spk_id_ready = self.speaker_service is not None and self.speaker is not None and diarize
        if identify_speakers and not spk_id_ready:
            warnings.append("identify_speakers")
        if (id_threshold is not None or id_margin is not None) and not spk_id_ready:
            warnings.append("speaker_id_threshold/margin")

        try:
            os.makedirs(chunk_dir, exist_ok=True)

            # 0. 格式转换
            if progress_callback:
                progress_callback(0.05)
            wav_path = os.path.join(UPLOADS_DIR, f"{task_id}.wav")
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            convert_to_wav(audio_path, wav_path)

            # 检查音频时长
            duration = get_audio_duration(wav_path)
            logger.info(f"[Pipeline] 音频转换完成: 时长={duration:.1f}s, 路径={wav_path}")
            if duration < MIN_AUDIO_DURATION:
                raise ValueError(f"音频过短（{duration:.1f}s），最短要求 {MIN_AUDIO_DURATION}s")
            if duration > MAX_AUDIO_DURATION:
                raise ValueError(
                    f"音频过长（{duration:.0f}s），最大支持 {MAX_AUDIO_DURATION}s，请分段上传"
                )

            # 1. VAD 切片
            if progress_callback:
                progress_callback(0.1)
            vad_segments = self.vad.detect(wav_path)
            logger.info(f"[Pipeline] VAD 检测完成: {len(vad_segments)} 个语音段")

            if not vad_segments:
                logger.info(f"VAD 未检测到语音段: {audio_path}")
                result = {
                    "segments": [],
                    "full_text": "",
                    "language": language,
                    "align_enabled": self.asr.align_enabled,
                    "punc_enabled": self.punc is not None,
                }
                if self.speaker is not None and diarize:
                    result["speakers"] = []
                if warnings:
                    result["warnings"] = warnings
                return result

            # 2. 合并相邻 VAD 段 + 切分写入 chunk 文件
            chunks = self._split_segments_to_chunks(
                wav_path, vad_segments, chunk_dir, max_segment)
            total_chunks = len(chunks)
            logger.info(f"[Pipeline] 切片完成: {len(vad_segments)} 个 VAD 段 -> {total_chunks} 个 chunk")

            # 3. 批量 ASR 识别（按 batch 分批推理，每批之间更新进度）
            segments = []
            if cancelled and cancelled():
                logger.info("[Pipeline] 任务已取消，跳过 ASR 识别")
            elif hasattr(self.asr, "batch_transcribe"):
                segments = self._transcribe_batched(
                    chunks, total_chunks, language, cancelled, progress_callback,
                    task_id=task_id,
                )
            else:
                segments = self._transcribe_sequential(
                    chunks, total_chunks, language, cancelled, progress_callback,
                )

            # 词级时间戳降级：请求 with_words=false 时剥离 ASR 已产出的 words
            if not with_words:
                for seg in segments:
                    seg.pop("words", None)

            # 4. 标点恢复（可选）
            if self.punc and with_punc:
                punc_count = 0
                for seg in segments:
                    if seg["text"] and seg["text"] != "[识别失败]":
                        try:
                            original = seg["text"]
                            seg["text"] = self.punc.restore(seg["text"])
                            if seg["text"] != original:
                                punc_count += 1
                        except Exception as e:
                            logger.warning(f"标点恢复失败，使用原始文本: {e}")
                logger.info(f"[Pipeline] 标点恢复完成: {punc_count}/{len(segments)} 个段落有变化")

            # 4.5 说话人分离（可选；容错：失败只丢标签，不破坏转写）。
            #     先在全局 VAD 时间轴聚类，给每个 ASR 处理块打 speaker——供 4.6 分句
            #     判定说话人切换；最终句子边界确定后（4.7）再按句精修标签。
            diar = None
            speakers_result = None
            speaker_active = (self.speaker is not None and diarize and segments
                              and not (cancelled and cancelled()))
            if speaker_active:
                if progress_callback:
                    progress_callback(0.90)
                try:
                    diar = self._run_diarization(
                        wav_path, vad_segments, cancelled=cancelled)
                    for seg in segments:
                        label = diar.label_for(seg["start"], seg["end"])
                        if label is not None:
                            seg["speaker"] = label
                except InterruptedError:
                    logger.info("[Pipeline] 任务已取消，停止说话人处理")
                    diar = None
                except Exception as e:
                    logger.warning(f"说话人分离失败，跳过: {e}")
                    diar = None

            # 4.6 分句：把 ASR 处理块重组为句子（标点/停顿/说话人切换）。
            #     max_segment 仅在调用方显式给定时作为输出句长上限，缺省不按时长切。
            segments = segment_sentences(segments, max_segment=max_segment)

            # 4.7 句子级说话人标签精修 + 声纹识别/自动登记（可选）
            if diar is not None:
                for seg in segments:
                    label = diar.label_for(seg["start"], seg["end"])
                    if label is not None:
                        seg["speaker"] = label
                speakers_result = diar.labels_in_order
                logger.info(
                    f"[Pipeline] 说话人分离完成: {len(speakers_result)} 人 {speakers_result}"
                )
                # 声纹识别 + 自动登记：speakers 升级为带 speaker_id/name 的映射表；
                # map_and_enroll_clusters 永不抛错（失败退回匿名）
                if (identify_speakers and self.speaker_service is not None
                        and not (cancelled and cancelled())):
                    mapping = self.speaker_service.map_and_enroll_clusters(
                        diar.clusters, id_threshold=id_threshold, id_margin=id_margin,
                        cancelled=cancelled)
                    if not (cancelled and cancelled()):
                        name_of = {m["label"]: m for m in mapping}
                        for seg in segments:
                            m = name_of.get(seg.get("speaker"))
                            if m and m.get("name"):
                                seg["speaker_name"] = m["name"]
                        speakers_result = mapping
                        named = sum(1 for m in mapping if m.get("name"))
                        logger.info(
                            f"[Pipeline] 声纹识别完成: {named}/{len(mapping)} 簇有名")
            if (speaker_active and progress_callback
                    and not (cancelled and cancelled())):
                progress_callback(0.95)

            # 5. 合并全文
            full_text = "".join(
                seg["text"] for seg in segments
                if seg["text"] and seg["text"].strip() and seg["text"] != "[识别失败]"
            )

            if progress_callback:
                progress_callback(1.0)

            result = {
                "segments": segments,
                "full_text": full_text,
                "language": language,
                "align_enabled": self.asr.align_enabled,
                "punc_enabled": self.punc is not None,
            }
            if speakers_result is not None:
                result["speakers"] = speakers_result
            if warnings:
                result["warnings"] = warnings
            return result

        finally:
            # 6. 清理临时文件
            self._cleanup(audio_path, wav_path, chunk_dir)

    def _transcribe_batched(
        self,
        chunks: list[dict],
        total_chunks: int,
        language: str | None,
        cancelled,
        progress_callback,
        task_id: str | None = None,
    ) -> list[dict]:
        """按 batch 分批调用 ASR 推理，每批之间更新进度和检查取消"""
        batch_size = max(1, int(getattr(self.asr, "batch_size", None) or cfg.ASR_BATCH_SIZE))
        if self.priority_gate is not None:
            realtime_batch_size = max(1, int(cfg.REALTIME_PRIORITY_OFFLINE_BATCH_SIZE))
            batch_size = min(batch_size, realtime_batch_size)
        segments: list[dict] = []
        processed = 0

        logger.info(
            f"[Pipeline] ASR 批量处理: {total_chunks} 个 chunk, batch_size={batch_size}"
        )

        for batch_start in range(0, total_chunks, batch_size):
            if cancelled and cancelled():
                logger.info(
                    f"[Pipeline] 任务已取消，已完成 {processed}/{total_chunks} 个 chunk"
                )
                break
            if self.priority_gate is not None:
                if not self.priority_gate.wait_realtime_clear(cancelled=cancelled):
                    logger.info(
                        f"[Pipeline] 任务已取消，已完成 {processed}/{total_chunks} 个 chunk"
                    )
                    break

            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            batch_paths = [c["path"] for c in batch_chunks]

            logger.info(
                f"[Pipeline] ASR 推理批次 {batch_start // batch_size + 1}: "
                f"chunk {batch_start + 1}-{batch_end}/{total_chunks}"
            )

            if self.asr_scheduler is not None:
                segments.extend(
                    self._transcribe_batch_with_scheduler(
                        batch_chunks,
                        batch_start,
                        total_chunks,
                        language,
                        cancelled,
                        progress_callback,
                        task_id,
                    )
                )
            else:
                try:
                    batch_results = self.asr.batch_transcribe(
                        audio_paths=batch_paths,
                        language=language,
                    )
                except Exception as e:
                    logger.error(f"批次推理失败，回退到逐条处理: {e}")
                    fallback = self._transcribe_sequential(
                        chunks[batch_start:], total_chunks, language,
                        cancelled, progress_callback, start_index=batch_start,
                    )
                    segments.extend(fallback)
                    break

                if len(batch_results) != len(batch_chunks):
                    logger.error(
                        f"批次结果数不匹配: 期望 {len(batch_chunks)}, 得到 {len(batch_results)}，"
                        "回退到逐条处理"
                    )
                    fallback = self._transcribe_sequential(
                        chunks[batch_start:], total_chunks, language,
                        cancelled, progress_callback, start_index=batch_start,
                    )
                    segments.extend(fallback)
                    break

                segments.extend(self._build_segments_from_batch(batch_chunks, batch_results))

            processed = batch_end
            logger.info(
                f"[Pipeline] ASR 进度: {processed}/{total_chunks} 个 chunk 完成"
            )
            if progress_callback:
                progress_callback(0.1 + 0.8 * processed / total_chunks)

        return segments

    def _transcribe_batch_with_scheduler(
        self,
        batch_chunks: list[dict],
        batch_start: int,
        total_chunks: int,
        language: str | None,
        cancelled,
        progress_callback,
        task_id: str | None,
    ) -> list[dict]:
        """经全局 ASR scheduler 转写一批 chunk。"""
        from app.runtime.offline_batch_scheduler import ChunkJob

        scheduler_task_id = task_id or "inline"
        jobs = [
            ChunkJob(
                task_id=scheduler_task_id,
                index=batch_start + offset,
                path=chunk_info["path"],
                offset_sec=chunk_info["offset_sec"],
                duration_sec=chunk_info["duration_sec"],
                language=language,
                split_after=bool(chunk_info.get("split_after")),
            )
            for offset, chunk_info in enumerate(batch_chunks)
        ]

        segments = []
        results = self.asr_scheduler.submit_many(jobs)
        for chunk_info, job, result in zip(
            batch_chunks,
            jobs,
            results,
        ):
            if getattr(result, "cancelled", False):
                logger.info(f"chunk {job.index} 调度识别已取消，跳过")
                continue
            if result.error:
                segment = self._retry_scheduler_chunk(chunk_info, job, result.error)
                if segment is not None:
                    segments.append(segment)
                continue
            segment = self._build_segment_from_results(chunk_info, result.results or [])
            if segment is not None:
                segments.append(segment)
        return segments

    def _retry_scheduler_chunk(self, chunk_info: dict, job, error: str) -> dict | None:
        """Retry one failed scheduler chunk without bypassing the scheduler owner."""
        logger.warning(f"chunk {job.index} 调度识别失败，重新提交单条调度: {error}")
        retry = self.asr_scheduler.submit(job)
        if getattr(retry, "cancelled", False):
            logger.info(f"chunk {job.index} 单条调度已取消，跳过")
            return None
        if retry.error:
            logger.error(f"chunk {job.index} 单条调度失败: {retry.error}")
            return {
                "start": chunk_info["offset_sec"],
                "end": chunk_info["offset_sec"] + chunk_info["duration_sec"],
                "text": "[识别失败]",
            }
        return self._build_segment_from_results(chunk_info, retry.results or [])

    def _transcribe_sequential_chunk(
        self,
        chunk_info: dict,
        index: int,
        total_chunks: int,
        language: str | None,
    ) -> dict | None:
        """逐条识别一个 chunk，供串行路径和 scheduler fallback 共用。"""
        logger.info(
            f"[Pipeline] ASR 处理中: chunk {index + 1}/{total_chunks} "
            f"({chunk_info['offset_sec']:.1f}s ~ "
            f"{chunk_info['offset_sec'] + chunk_info['duration_sec']:.1f}s)"
        )
        try:
            results = self.asr.transcribe(
                audio_path=chunk_info["path"],
                language=language,
            )
            return self._build_segment_from_results(chunk_info, results)
        except Exception as e:
            logger.error(f"chunk {index} 识别失败: {e}")
            return {
                "start": chunk_info["offset_sec"],
                "end": chunk_info["offset_sec"] + chunk_info["duration_sec"],
                "text": "[识别失败]",
            }

    def _transcribe_sequential(
        self,
        chunks: list[dict],
        total_chunks: int,
        language: str | None,
        cancelled,
        progress_callback,
        start_index: int = 0,
    ) -> list[dict]:
        """逐 chunk 串行 ASR 识别（fallback 路径）"""
        segments = []
        for offset, chunk_info in enumerate(chunks):
            i = start_index + offset
            if cancelled and cancelled():
                logger.info(f"[Pipeline] 任务已取消，已完成 {i}/{total_chunks} 个 chunk")
                break
            if self.priority_gate is not None:
                if not self.priority_gate.wait_realtime_clear(cancelled=cancelled):
                    logger.info(f"[Pipeline] 任务已取消，已完成 {i}/{total_chunks} 个 chunk")
                    break

            segment = self._transcribe_sequential_chunk(
                chunk_info,
                i,
                total_chunks,
                language,
            )
            if segment is not None:
                segments.append(segment)

            if progress_callback:
                progress_callback(0.1 + 0.8 * (i + 1) / total_chunks)
        return segments

    def _build_segments_from_batch(self, chunks: list[dict], batch_results: list) -> list[dict]:
        """把批量 ASR 输出转换为处理块 segments，保持 chunk 顺序和时间偏移。"""
        segments = []
        for chunk_info, result in zip(chunks, batch_results):
            segment = self._build_segment_from_results(chunk_info, [result])
            if segment is not None:
                segments.append(segment)
        return segments

    def _build_segment_from_results(self, chunk_info: dict, results) -> dict | None:
        """把单个 chunk 的 ASR 结果转换为 segment；空文本保持旧行为：不输出。"""
        text = self._extract_text(results)
        if not text.strip():
            return None

        words = self._extract_words(results, chunk_info["offset_sec"])
        segment = {
            "start": chunk_info["offset_sec"],
            "end": chunk_info["offset_sec"] + chunk_info["duration_sec"],
            "text": text,
        }
        if self.asr.align_enabled and words:
            segment["words"] = words
        if chunk_info.get("split_after"):
            segment["split_after"] = True
        return segment

    def _run_diarization(self, wav_path: str, vad_segments: list[tuple[int, int]],
                         *, cancelled=None):
        """说话人分离：原始 VAD 段（合并前）滑窗 → embedding → 全局聚类。

        返回 DiarizationResult（label_for 投票 + clusters 衔接面，声纹库 V 系列用）。
        延迟导入：speaker 关闭时本模块零额外依赖。
        """
        from app.engines.speaker_embedding_engine import make_windows
        from app.runtime.speaker_cluster import DiarizationResult, cluster_offline

        windows: list[tuple[float, float]] = []
        for start_ms, end_ms in vad_segments:
            windows.extend(make_windows(start_ms / 1000.0, end_ms / 1000.0))
        if not windows:
            return DiarizationResult([], [], [])

        # 窗数上限抽稀：规避超长音频谱聚类 N² 亲和阵内存
        if len(windows) > cfg.SPEAKER_MAX_WINDOWS:
            k = -(-len(windows) // cfg.SPEAKER_MAX_WINDOWS)
            windows = windows[::k]
            logger.info(f"[Pipeline] 说话人滑窗抽稀: 每 {k} 取 1 → {len(windows)} 窗")

        wav, _sr = sf.read(wav_path, dtype="float32")  # 阶段 0 已保证 16k 单声道
        embeddings = self.speaker.embed_windows(
            wav, windows, cancelled=cancelled)
        if cancelled and cancelled():
            raise InterruptedError("说话人处理已取消")
        labels = cluster_offline(embeddings, max_speakers=cfg.SPEAKER_MAX)
        if cancelled and cancelled():
            raise InterruptedError("说话人处理已取消")
        return DiarizationResult(windows, labels, embeddings)

    def _merge_vad_segments(
        self,
        vad_segments: list[tuple[int, int]],
        max_segment_sec: float | None = None,
    ) -> list[tuple[int, int]]:
        """
        贪心合并相邻 VAD 段：从第一段开始，持续追加后续段，
        直到合并后总跨度（首段 start 到末段 end）超过 max_segment_sec（缺省=cfg），
        则切出一组，开始新的一组。保留段间静音以维持时间戳准确性。
        """
        if not vad_segments:
            return []

        max_span_ms = int((max_segment_sec or cfg.MAX_SEGMENT_DURATION) * 1000)
        merged = []
        group_start, group_end = vad_segments[0]

        for start_ms, end_ms in vad_segments[1:]:
            # 如果追加后总跨度仍在阈值内，合并
            if end_ms - group_start <= max_span_ms:
                group_end = end_ms
            else:
                merged.append((group_start, group_end))
                group_start, group_end = start_ms, end_ms

        merged.append((group_start, group_end))
        return merged

    @staticmethod
    def _find_quiet_cut(data, sr, target, window, frame_ms=20, dip_ratio=0.5):
        """在 [target-window, target+window] 内找最安静帧作为切点（落在停顿中点）。

        无明显停顿（区域能量平坦/静音）时回退到名义切点 target，保持确定性。
        返回样本下标。
        """
        lo = max(0, int(target - window))
        hi = min(len(data), int(target + window))
        if hi - lo < int(0.1 * sr):
            return target
        fr = max(1, int(frame_ms / 1000 * sr))
        region = data[lo:hi]
        n = (len(region) - fr) // fr + 1
        if n <= 0:
            return target
        rms = np.empty(n, dtype=np.float64)
        for i in range(n):
            f = region[i * fr:i * fr + fr]
            rms[i] = float(np.sqrt(np.mean(np.square(f, dtype=np.float64)))) if len(f) else 0.0
        med = float(np.median(rms))
        if med <= 1e-6:                       # 平坦/静音 → 名义切点（确定性，单测可控）
            return target
        k = int(np.argmin(rms))
        if rms[k] < dip_ratio * med:          # 存在明显能量低谷（停顿）→ 谷中点下刀
            return lo + k * fr + fr // 2
        return target

    def _split_segments_to_chunks(
        self,
        wav_path: str,
        vad_segments: list[tuple[int, int]],
        chunk_dir: str,
        max_segment_sec: float | None = None,
    ) -> list[dict]:
        """
        合并相邻 VAD 段后切分音频，超长「连续语音段」在最安静处二次切分。

        两个阈值解耦（关键）：
        - 合并跨度上限 merge_max（=max_segment_sec 或 MAX_SEGMENT_DURATION）：相邻 VAD 段
          的合并发生在静音间隙处，安全。
        - 强制二次切分阈值 force_max（恒为 MAX_ASR_CHUNK_DURATION，显存/质量旋钮）：仅当
          「单个连续语音段」超过此值才切，且切在最安静处（停顿），避免把连续语句切在词中
          间导致边界词重复识别/漏字。force_max 远大于 merge_max，故大多数连续语句整段送 ASR。
          注意 force_max 不随显式 max_segment 变动——max_segment 只作句子输出上限（见
          sentence_segmenter），若令其压低 force_max 会重新引入词中切分。

        返回:
            [{"path": str, "offset_sec": float, "duration_sec": float, "split_after"?: bool}, ...]
            split_after=True 标记 force-split 产生的人为切点（供边界去重精准定位）。
        """
        data, sr = sf.read(wav_path)
        merge_max = max_segment_sec if max_segment_sec is not None else cfg.MAX_SEGMENT_DURATION
        force_max = cfg.MAX_ASR_CHUNK_DURATION

        # 先合并碎片段（仅在静音间隙处合并）
        merged = self._merge_vad_segments(vad_segments, merge_max)
        logger.info(
            f"VAD 段合并: {len(vad_segments)} -> {len(merged)} "
            f"(合并阈值={merge_max}s, 强制切分阈值={force_max}s)"
        )

        chunks = []
        idx = 0

        for start_ms, end_ms in merged:
            start_sample = int(start_ms / 1000 * sr)
            end_sample = int(end_ms / 1000 * sr)
            segment_data = data[start_sample:end_sample]
            segment_duration = len(segment_data) / sr

            if segment_duration <= force_max:
                chunk_path = os.path.join(chunk_dir, f"chunk_{idx:04d}.wav")
                sf.write(chunk_path, segment_data, sr)
                chunks.append({
                    "path": chunk_path,
                    "offset_sec": start_ms / 1000,
                    "duration_sec": segment_duration,
                })
                idx += 1
            else:
                # 单段连续语音超长：在最安静处下刀（无明显停顿则回退到名义切点）
                sub_samples = int(force_max * sr)
                window = int(min(force_max * 0.5, 2.5) * sr)
                offset = 0
                while offset < len(segment_data):
                    if len(segment_data) - offset <= sub_samples + window:
                        end = len(segment_data)              # 末块到结尾，免切出过短尾巴
                    else:
                        end = self._find_quiet_cut(
                            segment_data, sr, offset + sub_samples, window)
                        if end <= offset:
                            end = min(offset + sub_samples, len(segment_data))
                    sub_data = segment_data[offset:end]
                    chunk_path = os.path.join(chunk_dir, f"chunk_{idx:04d}.wav")
                    sf.write(chunk_path, sub_data, sr)
                    chunk_offset_sec = start_ms / 1000 + offset / sr
                    chunks.append({
                        "path": chunk_path,
                        "offset_sec": chunk_offset_sec,
                        "duration_sec": len(sub_data) / sr,
                        "split_after": end < len(segment_data),   # 非末子块=force-split 人为切点
                    })
                    offset = end
                    idx += 1

        logger.info(f"切分完成: {len(merged)} 个合并段 -> {len(chunks)} 个 chunk")
        return chunks

    def _extract_text(self, results) -> str:
        """从 qwen_asr transcribe 结果中提取纯文本（委托共享实现）"""
        return extract_text(results)

    def _extract_words(self, results, offset_sec: float) -> list[dict] | None:
        """从 qwen_asr 结果中提取单词级时间戳（委托共享实现）"""
        return extract_words(results, offset_sec)

    def _cleanup(self, original_path: str, wav_path: str | None, chunk_dir: str):
        """清理临时文件"""
        try:
            if original_path and os.path.exists(original_path):
                os.remove(original_path)
        except OSError as e:
            logger.warning(f"清理原始文件失败: {e}")

        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except OSError as e:
            logger.warning(f"清理转换文件失败: {e}")

        try:
            if os.path.exists(chunk_dir):
                shutil.rmtree(chunk_dir, ignore_errors=True)
        except OSError as e:
            logger.warning(f"清理 chunk 目录失败: {e}")
