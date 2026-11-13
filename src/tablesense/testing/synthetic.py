"""Deterministic synthetic restaurant video with known ground truth.

Why it exists: real video is big, licensed, and has no labels. Tests need small, repeatable
inputs where we *know* the right answer: where each person is, which table is occupied, and
when staff visit. Rendering the scene ourselves gives us exactly that.

The scene (640x360 @ 15 FPS, 300 frames = 20 s):

    +--------------------------------------------------------------+
    |      [ T1 ]                         [ T2 ]                  K|  K = kitchen door
    |   guest 1 sits here             guest 2 sits here,          K|  (staff enter here)
    |                                 staff visits ~7-11 s        K|
    |                    [ T3 ]  (always empty)                    |
    |  guest 3 walks across the bottom ------------------------->  |
    +--------------------------------------------------------------+

People are drawn as colored rectangles ("bodies") with a darker "head". Staff wear black.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2
Point = tuple[float, float]

FLOOR_BGR = (150, 160, 170)
TABLE_BGR = (40, 75, 120)


@dataclass(frozen=True)
class SyntheticPerson:
    person_id: int
    role: str  # "guest" or "staff"
    color: tuple[int, int, int]  # BGR
    # (frame_index, center_x, center_y). Linear interpolation between keyframes.
    # The person is visible from the first keyframe's frame to the last keyframe's frame.
    keyframes: tuple[tuple[int, float, float], ...]
    size: tuple[int, int] = (24, 48)  # width, height in pixels

    @property
    def first_frame(self) -> int:
        return self.keyframes[0][0]

    @property
    def last_frame(self) -> int:
        return self.keyframes[-1][0]

    def center_at(self, index: int) -> Point | None:
        """Center position at a frame, or None when the person is not in the scene."""
        if index < self.first_frame or index > self.last_frame:
            return None
        for (f0, x0, y0), (f1, x1, y1) in zip(self.keyframes, self.keyframes[1:], strict=False):
            if f0 <= index <= f1:
                a = 0.0 if f1 == f0 else (index - f0) / (f1 - f0)
                return (x0 + a * (x1 - x0), y0 + a * (y1 - y0))
        f, x, y = self.keyframes[-1]
        return (x, y)

    def box_at(self, index: int) -> Box | None:
        c = self.center_at(index)
        if c is None:
            return None
        w, h = self.size
        return (c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2)


@dataclass(frozen=True)
class GroundTruth:
    person_id: int
    role: str
    box: Box  # may extend outside the frame when entering/leaving


@dataclass(frozen=True)
class SyntheticScene:
    width: int = 640
    height: int = 360
    fps: int = 15
    n_frames: int = 300
    tables: dict[str, Box] = field(default_factory=dict)
    people: tuple[SyntheticPerson, ...] = ()

    def table_zone(self, table_id: str, margin: int = 35) -> list[tuple[int, int]]:
        """Polygon around a table where seated guests' bodies appear (table box + margin)."""
        x1, y1, x2, y2 = self.tables[table_id]
        return [
            (int(x1 - margin), int(y1 - margin)),
            (int(x2 + margin), int(y1 - margin)),
            (int(x2 + margin), int(y2 + margin)),
            (int(x1 - margin), int(y2 + margin)),
        ]

    def ground_truth(self, index: int) -> list[GroundTruth]:
        out = []
        for p in self.people:
            box = p.box_at(index)
            if box is not None:
                out.append(GroundTruth(p.person_id, p.role, box))
        return out


def default_scene() -> SyntheticScene:
    """The standard scene used by tests. Changing it will break tests, so make a new one instead."""
    return SyntheticScene(
        tables={
            "T1": (100.0, 70.0, 200.0, 130.0),
            "T2": (400.0, 70.0, 500.0, 130.0),
            "T3": (250.0, 220.0, 350.0, 280.0),
        },
        people=(
            # Guest 1: enters bottom-left, sits at T1 (~3 s), leaves at 16 s.
            SyntheticPerson(
                1,
                "guest",
                (60, 60, 220),
                ((0, 20.0, 330.0), (45, 150.0, 160.0), (240, 150.0, 160.0), (299, 20.0, 330.0)),
            ),
            # Guest 2: enters at 2 s, sits at T2 (~5.3 s), stays until the end.
            SyntheticPerson(
                2,
                "guest",
                (200, 120, 40),
                ((30, 20.0, 330.0), (80, 450.0, 160.0), (299, 450.0, 160.0)),
            ),
            # Staff: comes from the kitchen at 6.7 s, stands at T2 for 2 s, goes back.
            SyntheticPerson(
                3,
                "staff",
                (25, 25, 25),
                (
                    (100, 630.0, 200.0),
                    (140, 500.0, 175.0),
                    (170, 500.0, 175.0),
                    (220, 630.0, 200.0),
                ),
            ),
            # Guest 3: walks right-to-left along the bottom, never sits.
            SyntheticPerson(
                4,
                "guest",
                (60, 180, 60),
                ((60, 630.0, 325.0), (160, 10.0, 325.0)),
            ),
        ),
    )


def _background(scene: SyntheticScene) -> np.ndarray:
    """Static floor with a fixed texture plus the tables. Same every call."""
    rng = np.random.default_rng(12345)
    bg = np.empty((scene.height, scene.width, 3), dtype=np.uint8)
    bg[:] = FLOOR_BGR
    texture = rng.integers(-6, 7, size=(scene.height, scene.width, 1), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + texture, 0, 255).astype(np.uint8)
    for x1, y1, x2, y2 in scene.tables.values():
        cv2.rectangle(bg, (int(x1), int(y1)), (int(x2), int(y2)), TABLE_BGR, thickness=-1)
    return bg


def render_frame(
    scene: SyntheticScene, index: int, noise_sigma: float = 2.0, seed: int = 0
) -> tuple[np.ndarray, list[GroundTruth]]:
    """Render one BGR frame and its ground truth. Pure function of (scene, index, noise, seed)."""
    frame = _background(scene)
    gts = scene.ground_truth(index)
    for gt in gts:
        person = next(p for p in scene.people if p.person_id == gt.person_id)
        x1, y1, x2, y2 = (int(round(v)) for v in gt.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), person.color, thickness=-1)
        head_h = (y2 - y1) // 4
        head_color = tuple(max(0, c - 40) for c in person.color)
        cv2.rectangle(frame, (x1 + 4, y1), (x2 - 4, y1 + head_h), head_color, thickness=-1)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed * 100_003 + index)
        noise = rng.normal(0.0, noise_sigma, size=frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return frame, gts


def iter_frames(
    scene: SyntheticScene | None = None, noise_sigma: float = 2.0, seed: int = 0
) -> Iterator[tuple[int, np.ndarray, list[GroundTruth]]]:
    """Yield (index, frame, ground_truth) for every frame of the scene."""
    scene = scene or default_scene()
    for i in range(scene.n_frames):
        frame, gts = render_frame(scene, i, noise_sigma, seed)
        yield i, frame, gts


def write_video(
    path: str | Path,
    scene: SyntheticScene | None = None,
    n_frames: int | None = None,
    noise_sigma: float = 2.0,
) -> Path:
    """Write the scene to a video file (MJPG in .avi is the most portable OpenCV combination)."""
    scene = scene or default_scene()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), scene.fps, (scene.width, scene.height)
    )
    if not writer.isOpened():
        raise OSError(f"Could not open video writer for {path}")
    try:
        total = scene.n_frames if n_frames is None else min(n_frames, scene.n_frames)
        for i in range(total):
            frame, _ = render_frame(scene, i, noise_sigma)
            writer.write(frame)
    finally:
        writer.release()
    return path
