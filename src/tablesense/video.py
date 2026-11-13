"""Video sources and real-time measurement.

The core idea: a background thread reads frames as fast as the source delivers them and puts
them in a small queue. The processing loop always takes the freshest frame. If processing is
slow, old frames are dropped, so results never lag behind reality.
"""

from __future__ import annotations

import queue  # noqa: F401
import threading  # noqa: F401
import time  # noqa: F401
from collections import deque  # noqa: F401
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import cv2  # noqa: F401
import numpy as np


@dataclass(frozen=True)
class Frame:
    """One captured frame."""

    index: int  # 0, 1, 2, ... in capture order (counts every frame read, including dropped ones)
    timestamp: float  # time.monotonic() when the frame was read
    image: np.ndarray  # HxWx3 uint8 BGR


class FPSMeter:
    """Frames per second over a sliding window of the most recent `window` ticks.

    >>> m = FPSMeter(window=30)
    >>> for t in [0.0, 0.1, 0.2, 0.3]: m.tick(t)
    >>> round(m.fps, 1)
    10.0

    - `tick(now=None)`: record one frame. If `now` is None, use time.monotonic().
    - `fps`: (number of intervals) / (time spanned) over the window; 0.0 with fewer than 2 ticks
      or zero elapsed time.
    """

    def __init__(self, window: int = 30) -> None:
        raise NotImplementedError

    def tick(self, now: float | None = None) -> None:
        raise NotImplementedError

    @property
    def fps(self) -> float:
        raise NotImplementedError


class LatencyStats:
    """Keeps the last `maxlen` latency samples (seconds) and reports percentiles.

    - `add(seconds)`
    - `count`: number of samples currently kept
    - `p50`, `p95`: percentiles of kept samples (0.0 when empty).
    """

    def __init__(self, maxlen: int = 1000) -> None:
        raise NotImplementedError

    def add(self, seconds: float) -> None:
        raise NotImplementedError

    @property
    def count(self) -> int:
        raise NotImplementedError

    @property
    def p50(self) -> float:
        raise NotImplementedError

    @property
    def p95(self) -> float:
        raise NotImplementedError


class DropOldestQueue:
    """A thread-safe bounded queue whose `put` never blocks: when full, the oldest item is dropped.

    - `put(item)`: add an item; if full, discard the oldest first and increment `dropped`.
    - `get(timeout=None)`: remove and return the oldest item. Block until one is available;
      if `timeout` seconds pass with nothing available, raise `queue.Empty`.
    - `dropped`: how many items were discarded so far.
    - `len(q)`: current number of items.
    """

    def __init__(self, maxsize: int) -> None:
        raise NotImplementedError

    def put(self, item: Any) -> None:
        raise NotImplementedError

    def get(self, timeout: float | None = None) -> Any:
        raise NotImplementedError

    @property
    def dropped(self) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class ThreadedFrameSource:
    """Reads frames from `source` in a background thread.

    `source` is a file path, a webcam index (0) or an RTSP URL: anything cv2.VideoCapture accepts.

    Usage:
        with ThreadedFrameSource("data/videos/clip.mp4") as frames:
            for frame in frames:
                ...

    Behavior:
    - `start()` opens the source and starts the thread. Raise `OSError` if it can't be opened.
    - Iterating yields `Frame` objects in increasing `index` order, and stops when the source
      ends (or after `stop()`).
    - `drop_frames=True` (live sources): use a DropOldestQueue, so the consumer always gets
      fresh frames and `dropped` counts what was skipped.
    - `drop_frames=False` (files, evaluation): never drop; the reader waits while the queue is
      full. Careful: `stop()` must still work even while the reader is waiting.
    - `stop()` signals the thread, waits for it to finish (with a timeout) and releases the capture.
      Calling `stop()` twice must be safe.
    - The consumer must be able to tell "end of stream" apart from "no frame yet".
    """

    def __init__(self, source: str | int, queue_size: int = 2, drop_frames: bool = True) -> None:
        raise NotImplementedError

    def start(self) -> ThreadedFrameSource:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    @property
    def dropped(self) -> int:
        raise NotImplementedError

    @property
    def is_alive(self) -> bool:
        """True while the background thread is running."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError

    def __enter__(self) -> ThreadedFrameSource:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
