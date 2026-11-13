"""Tests for tablesense.video."""

import queue
import threading
import time

import pytest

from tablesense.testing.synthetic import write_video
from tablesense.video import DropOldestQueue, FPSMeter, Frame, LatencyStats, ThreadedFrameSource


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    return write_video(tmp_path_factory.mktemp("video") / "clip.avi", n_frames=60)


# --- FPSMeter ---
def test_fps_meter():
    m = FPSMeter(window=30)
    assert m.fps == 0.0
    m.tick(0.0)
    assert m.fps == 0.0  # one tick is not enough
    for i in range(1, 10):
        m.tick(i * 0.1)
    assert m.fps == pytest.approx(10.0)


def test_fps_meter_sliding_window():
    m = FPSMeter(window=5)
    for i in range(10):  # slow: 2 FPS
        m.tick(i * 0.5)
    for i in range(10):  # then fast: 20 FPS
        m.tick(5.0 + i * 0.05)
    assert m.fps == pytest.approx(20.0)  # only the last 5 ticks count


def test_fps_meter_uses_clock_when_no_time_given():
    m = FPSMeter()
    m.tick()
    time.sleep(0.02)
    m.tick()
    assert 1 < m.fps < 1000


# --- LatencyStats ---
def test_latency_stats():
    s = LatencyStats(maxlen=1000)
    assert s.count == 0
    assert s.p50 == 0.0 and s.p95 == 0.0
    for ms in range(1, 101):  # 1..100 ms
        s.add(ms / 1000)
    assert s.count == 100
    assert s.p50 == pytest.approx(0.0505, abs=1e-3)
    assert s.p95 == pytest.approx(0.095, abs=2e-3)


def test_latency_stats_keeps_only_recent():
    s = LatencyStats(maxlen=10)
    for _ in range(100):
        s.add(1.0)
    for _ in range(10):
        s.add(0.01)
    assert s.count == 10
    assert s.p95 == pytest.approx(0.01)


# --- DropOldestQueue ---
def test_drop_oldest_queue():
    q = DropOldestQueue(maxsize=2)
    q.put(1)
    q.put(2)
    q.put(3)  # drops 1
    assert len(q) == 2
    assert q.dropped == 1
    assert q.get() == 2
    assert q.get() == 3
    with pytest.raises(queue.Empty):
        q.get(timeout=0.05)


def test_drop_oldest_queue_get_blocks_until_item():
    q = DropOldestQueue(maxsize=2)
    threading.Timer(0.05, q.put, args=("hello",)).start()
    assert q.get(timeout=2.0) == "hello"


def test_drop_oldest_queue_is_thread_safe():
    q = DropOldestQueue(maxsize=5)
    n_per_thread = 5000

    def produce():
        for i in range(n_per_thread):
            q.put(i)

    threads = [threading.Thread(target=produce) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(q) == 5
    assert q.dropped + len(q) == 4 * n_per_thread


# --- ThreadedFrameSource ---
def test_threaded_source_reads_every_frame_without_dropping(video_path):
    with ThreadedFrameSource(str(video_path), queue_size=2, drop_frames=False) as source:
        frames = list(source)
    assert len(frames) == 60
    assert all(isinstance(f, Frame) for f in frames)
    assert [f.index for f in frames] == list(range(60))
    assert frames[0].image.shape == (360, 640, 3)
    assert all(b.timestamp >= a.timestamp for a, b in zip(frames, frames[1:], strict=False))
    assert source.dropped == 0


def test_threaded_source_drops_frames_for_slow_consumer(video_path):
    received = []
    with ThreadedFrameSource(str(video_path), queue_size=2, drop_frames=True) as source:
        for frame in source:
            received.append(frame.index)
            time.sleep(0.03)  # a slow "model"
    indices = received
    assert 0 < len(indices) < 60
    assert indices == sorted(set(indices))  # strictly increasing, no repeats
    assert source.dropped > 0


def test_threaded_source_stop_mid_stream(video_path):
    source = ThreadedFrameSource(str(video_path), queue_size=2, drop_frames=False).start()
    for i, _ in enumerate(source):
        if i == 3:
            break
    source.stop()  # reader is likely blocked on a full queue; stop must still work
    assert not source.is_alive
    source.stop()  # idempotent


def test_threaded_source_bad_path_raises(tmp_path):
    with pytest.raises(OSError):
        ThreadedFrameSource(str(tmp_path / "does_not_exist.mp4")).start()
