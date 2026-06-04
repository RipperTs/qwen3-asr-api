"""app/pipeline/audio_preprocessor.py 测试。

ffmpeg 相关一律 mock subprocess.run（本机/CI 不依赖 ffmpeg）；
get_audio_duration/get_file_size_mb 用真实临时文件。
行为依源码确认（audio_preprocessor.py:9/24/42/48）。
"""
import subprocess

import pytest

from app.pipeline import audio_preprocessor as ap


# ─── get_file_size_mb ───

def test_get_file_size_mb(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\0" * (2 * 1024 * 1024))  # 2 MB
    assert ap.get_file_size_mb(str(f)) == pytest.approx(2.0, abs=0.001)


# ─── get_audio_duration ───

def test_get_audio_duration(make_wav):
    path = make_wav(duration_sec=3.0)
    assert ap.get_audio_duration(path) == pytest.approx(3.0, abs=0.01)


# ─── check_ffmpeg ───

def test_check_ffmpeg_ok(mocker):
    mocker.patch.object(ap.subprocess, "run", return_value=mocker.Mock())
    ap.check_ffmpeg()  # 不抛异常即通过


def test_check_ffmpeg_not_found_raises(mocker):
    mocker.patch.object(ap.subprocess, "run", side_effect=FileNotFoundError())
    with pytest.raises(RuntimeError):
        ap.check_ffmpeg()


def test_check_ffmpeg_called_process_error_raises(mocker):
    err = subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg", "-version"])
    mocker.patch.object(ap.subprocess, "run", side_effect=err)
    with pytest.raises(RuntimeError):
        ap.check_ffmpeg()


# ─── convert_to_wav ───

def test_convert_to_wav_success(mocker):
    run = mocker.patch.object(ap.subprocess, "run", return_value=mocker.Mock())
    ap.convert_to_wav("in.mp3", "out.wav")
    args = run.call_args.args[0]
    assert "ffmpeg" in args
    assert "16000" in args  # 重采样到 16k
    assert "out.wav" in args


def test_convert_to_wav_failure_raises_valueerror(mocker):
    err = subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"], stderr=b"bad input")
    mocker.patch.object(ap.subprocess, "run", side_effect=err)
    with pytest.raises(ValueError) as exc:
        ap.convert_to_wav("in.mp3", "out.wav")
    assert "bad input" in str(exc.value)
