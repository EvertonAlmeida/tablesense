"""Check that the development environment is ready. Run: uv run python scripts/check_env.py"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys


def check(label: str, ok: bool, detail: str = "", optional: bool = False) -> bool:
    icon = "✅" if ok else ("⏭️ " if optional else "❌")
    print(f"{icon} {label}" + (f"  ({detail})" if detail else ""))
    return ok or optional


def try_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        return True, getattr(mod, "__version__", "")
    except Exception as e:  # noqa: BLE001
        return False, type(e).__name__


def main() -> int:
    print(f"Python {sys.version.split()[0]} on {platform.platform()}\n")
    all_ok = True

    print("Core:")
    all_ok &= check("Python >= 3.12", sys.version_info >= (3, 12), sys.version.split()[0])
    for name in ("numpy", "cv2", "yaml", "pytest"):
        ok, detail = try_import(name)
        all_ok &= check(name, ok, detail)
    all_ok &= check(
        "ffmpeg on PATH",
        shutil.which("ffmpeg") is not None,
        "brew install ffmpeg" if not shutil.which("ffmpeg") else "",
    )
    try:
        import cv2
        import numpy as np

        from tablesense.testing.synthetic import default_scene, render_frame

        frame, _ = render_frame(default_scene(), 50)
        ok = frame.shape == (360, 640, 3) and frame.dtype == np.uint8
        all_ok &= check("synthetic scene renders", ok)
        all_ok &= check("OpenCV MJPG writer", cv2.VideoWriter_fourcc(*"MJPG") != 0)
    except Exception as e:  # noqa: BLE001
        all_ok &= check("tablesense package importable", False, repr(e))

    print("\nOptional extras:")
    ok, detail = try_import("torch")
    check("torch  [uv sync --extra dl]", ok, detail, optional=True)
    if ok:
        import torch

        check("  Apple GPU (mps) available", torch.backends.mps.is_available(), optional=True)
    for name, extra in [
        ("ultralytics", "dl"),
        ("scipy", "track"),
        ("onnxruntime", "deploy"),
        ("fastapi", "deploy"),
        ("jax", "jax"),
    ]:
        ok, detail = try_import(name)
        check(f"{name}  [uv sync --extra {extra}]", ok, detail, optional=True)

    print("\n" + ("All core checks passed." if all_ok else "Fix the ❌ items above."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
