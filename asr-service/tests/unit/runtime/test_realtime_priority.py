"""实时优先门控单元测试。"""
import threading
import time

from app.runtime.realtime_priority import RealtimePriorityGate


def test_wait_realtime_clear_blocks_until_stream_exits():
    gate = RealtimePriorityGate()
    released = []

    ctx = gate.realtime_section()
    ctx.__enter__()

    t = threading.Thread(target=lambda: (gate.wait_realtime_clear(0.5), released.append(True)))
    t.start()
    time.sleep(0.05)
    assert released == []

    ctx.__exit__(None, None, None)
    t.join(timeout=1)

    assert released == [True]
    assert gate.realtime_active == 0


def test_wait_realtime_clear_default_waits_until_stream_exits():
    gate = RealtimePriorityGate()
    released = []

    ctx = gate.realtime_section()
    ctx.__enter__()

    t = threading.Thread(target=lambda: (gate.wait_realtime_clear(), released.append(True)))
    t.start()
    time.sleep(0.05)
    assert released == []

    ctx.__exit__(None, None, None)
    t.join(timeout=1)

    assert released == [True]


def test_wait_realtime_clear_returns_false_when_cancelled():
    gate = RealtimePriorityGate()
    cancelled = threading.Event()
    result = []

    ctx = gate.realtime_section()
    ctx.__enter__()

    t = threading.Thread(
        target=lambda: result.append(
            gate.wait_realtime_clear(cancelled=cancelled.is_set, poll_interval=0.01)
        )
    )
    t.start()
    time.sleep(0.05)
    assert result == []

    cancelled.set()
    t.join(timeout=1)
    ctx.__exit__(None, None, None)

    assert result == [False]


def test_disabled_gate_is_noop():
    gate = RealtimePriorityGate(enabled=False)
    with gate.realtime_section():
        gate.wait_realtime_clear(0.01)
    assert gate.realtime_active == 0
