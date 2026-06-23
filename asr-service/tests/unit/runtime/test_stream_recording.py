import os
import time
import wave

import pytest


def test_stream_recording_manager_writes_wav_and_deletes(tmp_path):
    from app.runtime.stream_recording import StreamRecordingManager

    manager = StreamRecordingManager(
        enabled=True,
        directory=str(tmp_path / "recordings"),
        retention_hours=72,
    )

    recorder = manager.start(wav_name="../mic input", sample_rate=8000)
    assert recorder is not None
    assert recorder.info["recording_id"]
    assert recorder.info["wav_name"] == "mic input.wav"

    recorder.write(b"\x01\x00\x02\x00")
    recorder.close()

    path = manager.path_for(recorder.info["recording_id"])
    assert path is not None
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 8000
        assert wf.getnframes() == 2

    assert manager.delete(recorder.info["recording_id"]) is True
    assert manager.delete(recorder.info["recording_id"]) is False


def test_stream_recording_manager_cleanup_uses_hours(tmp_path):
    from app.runtime.stream_recording import StreamRecordingManager

    manager = StreamRecordingManager(
        enabled=True,
        directory=str(tmp_path / "recordings"),
        retention_hours=1,
    )
    recorder = manager.start(wav_name="old", sample_rate=16000)
    recorder.close()
    path = manager.path_for(recorder.info["recording_id"])
    old_ts = time.time() - 7200
    os.utime(path, (old_ts, old_ts))

    assert manager.cleanup_expired() == 1
    assert manager.path_for(recorder.info["recording_id"]) is None


def test_stream_recording_manager_disabled_is_noop(tmp_path):
    from app.runtime.stream_recording import StreamRecordingManager

    manager = StreamRecordingManager(
        enabled=False,
        directory=str(tmp_path / "recordings"),
        retention_hours=72,
    )

    assert manager.start(wav_name="x", sample_rate=16000) is None
    assert manager.cleanup_expired() == 0


def test_stream_recording_manager_resumes_existing_wav(tmp_path):
    from app.runtime.stream_recording import StreamRecordingManager

    manager = StreamRecordingManager(
        enabled=True,
        directory=str(tmp_path / "recordings"),
        retention_hours=72,
    )
    recorder = manager.start(wav_name="meeting", sample_rate=8000)
    recorder.write(b"\x01\x00\x02\x00")
    recorder.close()

    resumed = manager.start(
        wav_name="ignored",
        sample_rate=8000,
        recording_id=recorder.info["recording_id"],
    )
    assert resumed.info["recording_id"] == recorder.info["recording_id"]
    assert resumed.info["wav_name"] == "meeting.wav"
    assert resumed.info["resumed"] is True
    resumed.write(b"\x03\x00\x04\x00")
    resumed.close()

    path = manager.path_for(recorder.info["recording_id"])
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 8000
        assert wf.getnframes() == 4
        assert wf.readframes(4) == b"\x01\x00\x02\x00\x03\x00\x04\x00"


def test_stream_recording_manager_missing_recording_id_creates_new_wav(tmp_path):
    from app.runtime.stream_recording import StreamRecordingManager

    manager = StreamRecordingManager(
        enabled=True,
        directory=str(tmp_path / "recordings"),
        retention_hours=72,
    )

    recorder = manager.start(
        wav_name="restored",
        sample_rate=16000,
        recording_id="a" * 32,
    )
    recorder.write(b"\x01\x00")
    recorder.close()

    assert recorder.info == {
        "recording_id": "a" * 32,
        "wav_name": "restored.wav",
        "resumed": False,
    }
    path = manager.path_for("a" * 32)
    assert path is not None
    assert path.name == f"{'a' * 32}_restored.wav"
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 1


def test_stream_recording_manager_resume_rejects_sample_rate_mismatch(tmp_path):
    from app.runtime.stream_recording import StreamRecordingError, StreamRecordingManager

    manager = StreamRecordingManager(
        enabled=True,
        directory=str(tmp_path / "recordings"),
        retention_hours=72,
    )
    recorder = manager.start(wav_name="meeting", sample_rate=8000)
    recorder.close()

    with pytest.raises(StreamRecordingError) as exc:
        manager.start(
            wav_name="meeting",
            sample_rate=16000,
            recording_id=recorder.info["recording_id"],
        )

    assert exc.value.code == "recording_mismatch"


def test_stream_recording_manager_resume_rejects_concurrent_writer(tmp_path):
    from app.runtime.stream_recording import StreamRecordingError, StreamRecordingManager

    manager = StreamRecordingManager(
        enabled=True,
        directory=str(tmp_path / "recordings"),
        retention_hours=72,
    )
    recorder = manager.start(wav_name="meeting", sample_rate=8000)
    recorder.close()

    active = manager.start(
        wav_name="meeting",
        sample_rate=8000,
        recording_id=recorder.info["recording_id"],
    )
    try:
        with pytest.raises(StreamRecordingError) as exc:
            manager.start(
                wav_name="meeting",
                sample_rate=8000,
                recording_id=recorder.info["recording_id"],
            )
        assert exc.value.code == "recording_conflict"
    finally:
        active.close()
