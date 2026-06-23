"""流式录音文件管理。"""
from __future__ import annotations

import os
import re
import struct
import threading
import time
import uuid
import wave
from pathlib import Path

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def _safe_wav_name(name: str | None) -> str:
    base = Path(name or "stream").name.strip() or "stream"
    base = _SAFE_NAME_RE.sub("_", base).strip(" ._") or "stream"
    stem = Path(base).stem or "stream"
    return f"{stem[:80]}.wav"


class StreamRecordingError(Exception):
    """录音创建/续写失败，可直接映射到 WS error.code。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class _WaveAppendWriter:
    """追加写入已存在的 PCM16 WAV，并同步更新 RIFF/data 长度。"""

    def __init__(self, path: Path, sample_rate: int):
        self._file = open(path, "r+b")
        try:
            self._data_size_pos, self._data_size = self._validate_and_find_data(path, sample_rate)
            self._file.seek(0, os.SEEK_END)
        except Exception:
            self._file.close()
            raise

    @staticmethod
    def _validate_and_find_data(path: Path, sample_rate: int) -> tuple[int, int]:
        try:
            with wave.open(str(path), "rb") as wf:
                if (
                    wf.getnchannels() != 1
                    or wf.getsampwidth() != 2
                    or wf.getframerate() != int(sample_rate)
                    or wf.getcomptype() != "NONE"
                ):
                    raise StreamRecordingError(
                        "recording_mismatch",
                        "录音参数不一致，无法续写",
                    )
        except StreamRecordingError:
            raise
        except (wave.Error, EOFError):
            raise StreamRecordingError("recording_mismatch", "录音文件不是有效 WAV")

        with open(path, "rb") as f:
            header = f.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise StreamRecordingError("recording_mismatch", "录音文件不是有效 WAV")
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(12)
            while True:
                chunk_header_pos = f.tell()
                chunk_header = f.read(8)
                if len(chunk_header) != 8:
                    raise StreamRecordingError("recording_mismatch", "录音文件缺少 data 块")
                chunk_id = chunk_header[:4]
                chunk_size = struct.unpack("<I", chunk_header[4:])[0]
                data_pos = f.tell()
                if chunk_id == b"data":
                    data_end = data_pos + chunk_size + (chunk_size % 2)
                    if data_end != file_size:
                        raise StreamRecordingError("recording_mismatch", "录音文件结构不支持续写")
                    return chunk_header_pos + 4, chunk_size
                f.seek(chunk_size + (chunk_size % 2), os.SEEK_CUR)

    def writeframes(self, pcm_bytes: bytes) -> None:
        self._file.seek(0, os.SEEK_END)
        self._file.write(pcm_bytes)
        self._data_size += len(pcm_bytes)
        file_size = self._file.tell()

        self._file.seek(self._data_size_pos)
        self._file.write(struct.pack("<I", self._data_size))
        self._file.seek(4)
        self._file.write(struct.pack("<I", file_size - 8))
        self._file.seek(0, os.SEEK_END)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class StreamRecorder:
    """单个流式会话的 WAV 写入器。"""

    def __init__(
        self,
        path: Path,
        recording_id: str,
        wav_name: str,
        sample_rate: int,
        *,
        resumed: bool = False,
        on_close=None,
    ):
        self.path = path
        self.info = {"recording_id": recording_id, "wav_name": wav_name, "resumed": resumed}
        self._closed = False
        self._on_close = on_close
        if resumed:
            self._wave = _WaveAppendWriter(path, sample_rate)
        else:
            self._wave = wave.open(str(path), "wb")
            self._wave.setnchannels(1)
            self._wave.setsampwidth(2)
            self._wave.setframerate(int(sample_rate))

    def write(self, pcm_bytes: bytes) -> None:
        if self._closed or not pcm_bytes:
            return
        self._wave.writeframes(pcm_bytes)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._wave.close()
        finally:
            if self._on_close is not None:
                self._on_close()


class StreamRecordingManager:
    """流式录音资源 owner。retention_hours <= 0 表示永不自动清理。"""

    def __init__(self, *, enabled: bool, directory: str, retention_hours: int = 72):
        self.enabled = bool(enabled)
        self.directory = Path(directory)
        self.retention_hours = int(retention_hours)
        self._active_ids: set[str] = set()
        self._lock = threading.Lock()

    def start(
        self,
        *,
        wav_name: str | None,
        sample_rate: int,
        recording_id: str | None = None,
    ) -> StreamRecorder | None:
        if not self.enabled:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)

        if recording_id is not None:
            if not _ID_RE.fullmatch(recording_id or ""):
                raise StreamRecordingError("invalid_recording_id", "recording_id 非法")
            path = self.path_for(recording_id)
            if path is None:
                safe_name = _safe_wav_name(wav_name)
                path = self.directory / f"{recording_id}_{safe_name}"
                resumed = False
            else:
                safe_name = path.name.split("_", 1)[1]
                resumed = True
        else:
            recording_id = uuid.uuid4().hex
            safe_name = _safe_wav_name(wav_name)
            path = self.directory / f"{recording_id}_{safe_name}"
            resumed = False

        self._reserve(recording_id)
        try:
            return StreamRecorder(
                path,
                recording_id,
                safe_name,
                sample_rate,
                resumed=resumed,
                on_close=lambda: self._release(recording_id),
            )
        except Exception:
            self._release(recording_id)
            raise

    def _reserve(self, recording_id: str) -> None:
        with self._lock:
            if recording_id in self._active_ids:
                raise StreamRecordingError("recording_conflict", "录音正在写入中")
            self._active_ids.add(recording_id)

    def _release(self, recording_id: str) -> None:
        with self._lock:
            self._active_ids.discard(recording_id)

    def path_for(self, recording_id: str) -> Path | None:
        if not _ID_RE.fullmatch(recording_id or ""):
            return None
        for path in self.directory.glob(f"{recording_id}_*.wav"):
            if path.is_file():
                return path
        return None

    def filename_for(self, recording_id: str) -> str:
        path = self.path_for(recording_id)
        if path is None:
            return f"{recording_id}.wav"
        return path.name.split("_", 1)[1]

    def delete(self, recording_id: str) -> bool:
        path = self.path_for(recording_id)
        if path is None:
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def cleanup_expired(self) -> int:
        if self.retention_hours <= 0 or not self.directory.exists():
            return 0
        cutoff = time.time() - self.retention_hours * 3600
        removed = 0
        for path in self.directory.glob("*.wav"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    os.remove(path)
                    removed += 1
            except FileNotFoundError:
                continue
        return removed
