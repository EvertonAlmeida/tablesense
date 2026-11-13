"""Write the synthetic restaurant scene to data/videos/synthetic.avi (with zones drawn on a copy).

Run: uv run python scripts/make_synthetic_video.py [--annotated]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tablesense.testing.synthetic import default_scene, iter_frames, write_video

OUT_DIR = Path("data/videos")


def write_annotated(path: Path) -> None:
    """Same scene with zones and ground-truth boxes drawn, to see what the tests check."""
    scene = default_scene()
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), scene.fps, (scene.width, scene.height)
    )
    for i, frame, gts in iter_frames(scene):
        for table_id in scene.tables:
            pts = np.array(scene.table_zone(table_id), dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 200, 0), 1)
            cv2.putText(
                frame, table_id, tuple(pts[0] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1
            )
        for gt in gts:
            x1, y1, x2, y2 = (int(round(v)) for v in gt.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(
                frame,
                f"#{gt.person_id} {gt.role}",
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
        cv2.putText(
            frame,
            f"frame {i}  t={i / scene.fps:.1f}s",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated", action="store_true", help="also write an annotated copy")
    args = parser.parse_args()
    path = write_video(OUT_DIR / "synthetic.avi")
    print(f"wrote {path}")
    if args.annotated:
        ann = OUT_DIR / "synthetic_annotated.avi"
        write_annotated(ann)
        print(f"wrote {ann}")


if __name__ == "__main__":
    main()
