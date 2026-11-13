# TableSense: conventions for AI assistants

Real-time computer vision for restaurants (see `docs/ARCHITECTURE.md`).

- Python 3.12, managed with `uv`: `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`,
  extras with `uv sync --extra dl|track|deploy|jax`.
- Images are HxWx3 uint8 **BGR**; boxes are `(x1, y1, x2, y2)` floats; points are `(x, y)`;
  index arrays as `image[y, x]`. Ground position of a person or vehicle = bottom-center of its box.
- Pure functions where possible; never mutate inputs; frozen dataclasses for records.
- Tests must not need real video or network: use `tablesense.testing.synthetic`.
- Thresholds, zones and lines live in `configs/*.yaml`, not in code.
- Never commit videos, datasets or model weights (`data/`, `runs/`, `*.pt`, `*.onnx` are ignored).
  Record every video source in `data/videos/SOURCES.md`.
- No face recognition, license plate reading or identity storage.
- Commit messages: conventional commits scoped by component, e.g. `feat(geometry): ...`, `fix(track): ...`.
