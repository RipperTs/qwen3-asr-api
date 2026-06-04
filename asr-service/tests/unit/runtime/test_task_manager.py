"""app/runtime/task_manager.py 测试。

注入假 process_fn，不加载模型；超时用 monkeypatch 压缩。
行为依源码确认（task_manager.py:41/64/69/92/98/125/130/187）。
"""
import queue

import pytest

from tests.conftest import wait_for


# ─── submit / get_task ───

def test_submit_returns_id_and_pending(tm_factory):
    tm = tm_factory()
    tid = tm.submit("/tmp/a.wav", language="zh")
    assert isinstance(tid, str) and len(tid) > 0
    task = tm.get_task(tid)
    assert task["status"] == "pending"
    assert task["progress"] == 0.0
    assert task["file_path"] == "/tmp/a.wav"
    assert task["language"] == "zh"
    assert task["result"] is None


def test_get_task_unknown_returns_none(tm_factory):
    tm = tm_factory()
    assert tm.get_task("nope") is None


def test_submit_queue_full_raises(tm_factory):
    # 不启动 worker，队列填满后 put_nowait 抛 queue.Full
    tm = tm_factory(max_queue_size=2)
    tm.submit("/tmp/1.wav")
    tm.submit("/tmp/2.wav")
    with pytest.raises(queue.Full):
        tm.submit("/tmp/3.wav")


# ─── list_tasks ───

def _inject(tm, task_id, status, created_at, **extra):
    task = {
        "task_id": task_id,
        "status": status,
        "progress": 0.0,
        "file_path": f"/tmp/{task_id}.wav",
        "language": None,
        "result": {"full_text": "secret"},
        "error": None,
        "created_at": created_at,
        "finished_at": None,
    }
    task.update(extra)
    tm._tasks[task_id] = task


def test_list_tasks_summary_excludes_result_and_path(tm_factory):
    tm = tm_factory()
    _inject(tm, "a", "completed", "2026-06-03T10:00:00")
    items = tm.list_tasks()
    assert len(items) == 1
    item = items[0]
    assert "result" not in item
    assert "file_path" not in item
    assert set(item.keys()) == {
        "task_id", "status", "progress", "language", "created_at", "finished_at", "error",
    }


def test_list_tasks_sorted_desc_by_created_at(tm_factory):
    tm = tm_factory()
    _inject(tm, "old", "completed", "2026-06-03T10:00:00")
    _inject(tm, "new", "completed", "2026-06-03T12:00:00")
    items = tm.list_tasks()
    assert [i["task_id"] for i in items] == ["new", "old"]


def test_list_tasks_status_filter(tm_factory):
    tm = tm_factory()
    _inject(tm, "p", "pending", "2026-06-03T10:00:00")
    _inject(tm, "c", "completed", "2026-06-03T11:00:00")
    assert [i["task_id"] for i in tm.list_tasks(status="pending")] == ["p"]
    assert [i["task_id"] for i in tm.list_tasks(status="completed")] == ["c"]


# ─── update_progress ───

def test_update_progress(tm_factory):
    tm = tm_factory()
    tid = tm.submit("/tmp/a.wav")
    tm.update_progress(tid, 0.5)
    assert tm.get_task(tid)["progress"] == 0.5


def test_update_progress_unknown_noop(tm_factory):
    tm = tm_factory()
    tm.update_progress("nope", 0.5)  # 不抛异常即可


# ─── cancel_task / is_cancelled ───

def test_cancel_unknown_returns_none(tm_factory):
    tm = tm_factory()
    assert tm.cancel_task("nope") is None


def test_cancel_pending_marks_cancelled(tm_factory):
    tm = tm_factory()
    tid = tm.submit("/tmp/a.wav")
    prev = tm.cancel_task(tid)
    assert prev == "pending"
    task = tm.get_task(tid)
    assert task["status"] == "cancelled"
    assert task["error"] == "任务已取消"
    assert task["finished_at"] is not None
    assert tm.is_cancelled(tid) is True


def test_cancel_processing_keeps_status_sets_event(tm_factory):
    tm = tm_factory()
    tid = tm.submit("/tmp/a.wav")
    tm.get_task(tid)["status"] = "processing"
    prev = tm.cancel_task(tid)
    assert prev == "processing"
    assert tm.get_task(tid)["status"] == "processing"  # 等待 chunk 边界，不立即改
    assert tm.is_cancelled(tid) is True


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_cancel_terminal_noop(tm_factory, terminal):
    tm = tm_factory()
    tid = tm.submit("/tmp/a.wav")
    tm.get_task(tid)["status"] = terminal
    prev = tm.cancel_task(tid)
    assert prev == terminal
    assert tm.get_task(tid)["status"] == terminal


def test_is_cancelled_unknown_false(tm_factory):
    tm = tm_factory()
    assert tm.is_cancelled("nope") is False


# ─── is_stopping / shutdown ───

def test_is_stopping_toggles_on_shutdown(tm_factory):
    tm = tm_factory()
    assert tm.is_stopping is False
    tm.shutdown()
    assert tm.is_stopping is True


# ─── worker 集成（启动线程，process_fn 假实现）───

def test_worker_completes_task(tm_factory):
    tm = tm_factory(start=True, processor=lambda task: {"full_text": "hi"})
    tid = tm.submit("/tmp/a.wav")
    assert wait_for(lambda: tm.get_task(tid)["status"] == "completed")
    task = tm.get_task(tid)
    assert task["result"] == {"full_text": "hi"}
    assert task["progress"] == 1.0
    assert task["finished_at"] is not None


def test_worker_failure_sets_generic_error(tm_factory):
    def boom(task):
        raise ValueError("internal detail")

    tm = tm_factory(start=True, processor=boom)
    tid = tm.submit("/tmp/a.wav")
    assert wait_for(lambda: tm.get_task(tid)["status"] == "failed")
    task = tm.get_task(tid)
    # 错误信息脱敏，不泄露内部异常细节
    assert task["error"] == "内部处理错误，请检查服务日志"
    assert "internal detail" not in task["error"]


def test_worker_timeout(tm_factory, monkeypatch):
    monkeypatch.setattr("app.runtime.task_manager.TASK_TIMEOUT", 0.3)

    def slow(task):
        import time
        time.sleep(1.0)
        return {"x": 1}

    tm = tm_factory(start=True, processor=slow)
    tid = tm.submit("/tmp/a.wav")
    assert wait_for(lambda: tm.get_task(tid)["status"] == "failed", timeout=3.0)
    assert "超时" in tm.get_task(tid)["error"]


def test_worker_skips_cancelled_pending(tm_factory):
    # pending 阶段被取消的任务，worker 跳过处理（不会变 completed）
    processed = []
    tm = tm_factory(start=False, processor=lambda task: processed.append(task["task_id"]))
    tid = tm.submit("/tmp/a.wav")
    tm.cancel_task(tid)  # pending -> cancelled
    tm.start()
    import time
    time.sleep(0.3)
    assert tm.get_task(tid)["status"] == "cancelled"
    assert processed == []
