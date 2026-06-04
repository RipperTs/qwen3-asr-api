"""app/pipeline/asr_pipeline.py::_split_segments_to_chunks 测试（需真实 16k wav）。

行为依源码确认（asr_pipeline.py:303）：先 _merge_vad_segments 合并，
再按 MAX_SEGMENT_DURATION 二次切分超长段。
"""
import os
import types

import pytest

from app.pipeline.asr_pipeline import ASRPipeline


@pytest.fixture
def pipe():
    dummy = types.SimpleNamespace()
    return ASRPipeline(asr_engine=dummy, vad_engine=dummy, punc_engine=None)


def test_split_merged_single_chunk(pipe, make_wav, tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.MAX_SEGMENT_DURATION", 5)
    wav = make_wav(duration_sec=9.0)
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()

    # (0,3000)+(3500,4000): 跨度 4000<=5000 合并为 (0,4000)，时长 4s<=5 -> 1 chunk
    segs = [(0, 3000), (3500, 4000)]
    chunks = pipe._split_segments_to_chunks(wav, segs, str(chunk_dir))

    assert len(chunks) == 1
    assert chunks[0]["offset_sec"] == pytest.approx(0.0)
    assert chunks[0]["duration_sec"] == pytest.approx(4.0, abs=0.05)
    assert os.path.exists(chunks[0]["path"])


def test_split_oversize_secondary_split(pipe, make_wav, tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.MAX_SEGMENT_DURATION", 5)
    wav = make_wav(duration_sec=9.0)
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()

    # 单段 (0,8000) 时长 8s > 5s -> 二次切分为 5s + 3s
    segs = [(0, 8000)]
    chunks = pipe._split_segments_to_chunks(wav, segs, str(chunk_dir))

    assert len(chunks) == 2
    assert [round(c["offset_sec"], 1) for c in chunks] == [0.0, 5.0]
    assert chunks[0]["duration_sec"] == pytest.approx(5.0, abs=0.05)
    assert chunks[1]["duration_sec"] == pytest.approx(3.0, abs=0.05)
    for c in chunks:
        assert os.path.exists(c["path"])
