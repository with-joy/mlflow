import concurrent.futures.thread
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from unittest import mock

import pytest

from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task

from tests.tracing.helper import skip_when_testing_trace_sdk


def test_async_queue_handle_tasks():
    queue = AsyncTraceExportQueue()

    counter = 0

    def increment(delta):
        nonlocal counter
        counter += delta

    for _ in range(10):
        task = Task(handler=increment, args=(1,))
        queue.put(task)

    queue.flush(terminate=True)
    assert counter == 10


def exporter_process(counter):
    # This process exits before waiting for the tasks to finish

    queue = AsyncTraceExportQueue()

    def increment(counter):
        time.sleep(1)
        with counter.get_lock():
            counter.value += 1

    for _ in range(10):
        task = Task(handler=increment, args=(counter,))
        queue.put(task)


@skip_when_testing_trace_sdk
def test_async_queue_complete_task_process_finished():
    multiprocessing.set_start_method("spawn", force=True)
    counter = multiprocessing.Value("i", 0)
    process = multiprocessing.Process(target=exporter_process, args=(counter,))
    process.start()
    process.join(timeout=15)

    assert counter.value == 10


def test_async_queue_activate_thread_safe():
    with mock.patch("atexit.register") as mock_atexit:
        queue = AsyncTraceExportQueue()

        def count_threads():
            main_thread = threading.main_thread()
            return sum(
                t.is_alive()
                for t in threading.enumerate()
                if t is not main_thread and t.name.startswith("MLflowTraceLogging")
            )

        # 1. Validate activation
        with ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="test-async-export-queue-activate"
        ) as executor:
            for _ in range(10):
                executor.submit(queue.activate)
        assert count_threads() > 0  # Logging thread + max 5 worker threads
        mock_atexit.assert_called_once()
        mock_atexit.reset_mock()

        # 2. Validate flush (continue)
        queue.flush(terminate=False)
        assert queue.is_active()
        assert count_threads() > 0  # New threads should be created
        mock_atexit.assert_not_called()  # Exit callback should not be registered again

        # 3. Validate flush with termination
        with ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="test-async-export-queue-flush"
        ) as executor:
            for _ in range(10):
                executor.submit(queue.flush(terminate=True))
        assert count_threads() == 0


def test_put_after_terminate_executes_synchronously():
    queue = AsyncTraceExportQueue()

    calls = []
    queue.put(Task(handler=calls.append, args=(1,)))
    queue.flush(terminate=True)

    assert not queue.is_active()
    assert queue._stop_event.is_set()

    # Calling put() after termination must not deadlock; task must run synchronously.
    assert queue.put(Task(handler=calls.append, args=(2,)))

    assert calls == [1, 2]


def test_async_queue_drop_task_when_full(monkeypatch):
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE", "3")
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS", "1")

    queue = AsyncTraceExportQueue()

    processed_tasks = 0

    # Create a slow handler to keep tasks in the queue
    def slow_handler():
        time.sleep(0.5)

        nonlocal processed_tasks
        processed_tasks += 1

    queued = [queue.put(Task(handler=slow_handler, args=())) for _ in range(10)]

    queue.flush(terminate=True)

    # One more task than the queue size might be processed, because the first task
    # can be drained from the queue immediately, which creates a slot for another task
    assert processed_tasks <= 4
    # put() reports exactly the tasks it discarded
    assert queued.count(True) == processed_tasks


def _wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 10
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("Condition was not met within 10 seconds")
        time.sleep(0.01)


def _parallel_follow_ups(on_done: Callable[[], None]) -> list[Task]:
    # Passes only if both tasks run at the same time.
    barrier = threading.Barrier(2, timeout=5)

    def follow_up():
        barrier.wait()
        on_done()

    return [Task(handler=follow_up, args=()) for _ in range(2)]


@pytest.mark.parametrize("terminate", [False, True])
def test_flush_waits_for_follow_ups_run_in_parallel(terminate):
    queue = AsyncTraceExportQueue()
    done = []

    def on_done():
        # Finish only once flush() has started and the task that started this one is done, so
        # only the worker shutdown in flush() can wait for it.
        _wait_until(lambda: queue._stop_event.is_set() and queue._queue.unfinished_tasks == 0)
        done.append(threading.current_thread())

    queue.put(Task(handler=queue.run_in_parallel, args=(_parallel_follow_ups(on_done),)))
    queue.flush(terminate=terminate)

    assert len(done) == 2
    assert all(t.name.startswith("MlflowTraceLoggingWorker") for t in done)

    if not terminate:
        queue.flush(terminate=True)


def test_run_in_parallel_after_terminate_waits_for_tasks_on_new_threads():
    queue = AsyncTraceExportQueue()
    queue.activate()
    queue.flush(terminate=True)
    done = []

    queue.run_in_parallel(_parallel_follow_ups(lambda: done.append(threading.current_thread())))

    assert len(done) == 2
    assert threading.current_thread() not in done


def test_run_in_parallel_after_terminate_runs_at_most_max_workers_tasks_at_once(monkeypatch):
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS", "2")
    queue = AsyncTraceExportQueue()
    queue.activate()
    queue.flush(terminate=True)
    running = []
    release = threading.Event()

    def follow_up():
        running.append(threading.current_thread())
        release.wait(timeout=10)

    caller = threading.Thread(
        target=queue.run_in_parallel,
        args=([Task(handler=follow_up, args=())] * 3,),
        name="test-run-in-parallel",
    )
    caller.start()

    def follow_up_threads():
        return [t for t in threading.enumerate() if t.name == "MlflowTraceLoggingFollowUp"]

    _wait_until(lambda: len(follow_up_threads()) == 3 and len(running) >= 2)
    assert len(running) == 2
    release.set()
    caller.join(timeout=10)

    assert not caller.is_alive()
    assert len(set(running)) == 3


def exit_during_task_process(counter):
    # The process exits while the queued task runs. The task waits until concurrent.futures has
    # shut the worker pool down, then hands its follow-ups to run_in_parallel().
    queue = AsyncTraceExportQueue()

    def increment():
        with counter.get_lock():
            counter.value += 1

    def handler():
        _wait_until(lambda: concurrent.futures.thread._shutdown)
        queue.run_in_parallel(_parallel_follow_ups(increment))

    queue.put(Task(handler=handler, args=()))


@skip_when_testing_trace_sdk
def test_run_in_parallel_at_interpreter_exit():
    multiprocessing.set_start_method("spawn", force=True)
    counter = multiprocessing.Value("i", 0)
    process = multiprocessing.Process(target=exit_during_task_process, args=(counter,))
    process.start()
    process.join(timeout=15)

    assert counter.value == 2


@pytest.mark.parametrize("terminate", [False, True])
def test_concurrent_put_during_flush_does_not_deadlock(terminate):
    queue = AsyncTraceExportQueue()

    calls = []
    consumer_joined = threading.Event()
    allow_flush_to_continue = threading.Event()

    queue.activate()

    original_join = queue._consumer_thread.join

    def controlled_join(*args, **kwargs):
        result = original_join(*args, **kwargs)
        consumer_joined.set()
        allow_flush_to_continue.wait(timeout=5)
        return result

    queue._consumer_thread.join = controlled_join

    flush_thread = threading.Thread(
        target=queue.flush,
        kwargs={"terminate": terminate},
        name="test-concurrent-put-flush",
        daemon=True,
    )
    flush_thread.start()

    assert consumer_joined.wait(timeout=5)

    # The consumer has stopped while flush() is still in progress.
    # A concurrent put() must not enqueue work that no consumer can process.
    queue.put(Task(handler=calls.append, args=(1,)))

    allow_flush_to_continue.set()
    flush_thread.join(timeout=5)

    assert not flush_thread.is_alive()
    assert calls == [1]
    assert queue.is_active() is not terminate

    if not terminate:
        queue.flush(terminate=True)


def test_flush_does_not_wait_for_idle_consumer_poll(monkeypatch):
    queue = AsyncTraceExportQueue()
    polling = threading.Event()
    get = queue._queue.get

    def slow_poll(timeout):
        polling.set()
        # Much longer than the real 1 s poll, so a flush that waits for it stands out.
        return get(timeout=10)

    monkeypatch.setattr(queue._queue, "get", slow_poll)
    queue.activate()
    assert polling.wait(timeout=5)

    start = time.monotonic()
    queue.flush(terminate=True)

    assert time.monotonic() - start < 5


def test_flush_returns_when_consumer_exits_before_wake_up(monkeypatch):
    queue = AsyncTraceExportQueue()
    queue.activate()
    consumer = queue._consumer_thread
    set_stop_event = queue._stop_event.set

    def set_and_wait_for_consumer_exit():
        set_stop_event()
        # The flushing thread is descheduled right after setting the event, until the consumer
        # has seen it and exited.
        consumer.join(timeout=10)

    monkeypatch.setattr(queue._stop_event, "set", set_and_wait_for_consumer_exit)
    flush_thread = threading.Thread(
        target=queue.flush, kwargs={"terminate": True}, name="test-flush", daemon=True
    )
    flush_thread.start()
    flush_thread.join(timeout=10)

    assert not consumer.is_alive()
    assert not flush_thread.is_alive()


def test_flush_waits_for_in_progress_flush():
    queue = AsyncTraceExportQueue()
    release = threading.Event()
    completed = []

    def handler(value):
        assert release.wait(timeout=10)
        completed.append(value)

    for value in range(3):
        queue.put(Task(handler=handler, args=(value,)))

    first_flush_thread = threading.Thread(
        target=queue.flush,
        kwargs={"terminate": True},
        name="test-first-concurrent-flush",
        daemon=True,
    )
    first_flush_thread.start()

    deadline = time.monotonic() + 5
    while queue.is_active() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not queue.is_active()

    completed_when_second_flush_returns = []

    def second_flush():
        queue.flush(terminate=True)
        completed_when_second_flush_returns.append(len(completed))

    second_flush_thread = threading.Thread(
        target=second_flush,
        name="test-second-concurrent-flush",
        daemon=True,
    )
    second_flush_thread.start()

    second_flush_thread.join(timeout=0.1)

    release.set()
    first_flush_thread.join(timeout=5)
    second_flush_thread.join(timeout=5)

    assert not first_flush_thread.is_alive()
    assert not second_flush_thread.is_alive()
    assert completed_when_second_flush_returns == [3]
