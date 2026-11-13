"""Geometry primitives: boxes, polygons, masks and zone drawing.

Conventions used everywhere in TableSense:
- Images are NumPy arrays of shape (height, width, 3), dtype uint8, **BGR** channel order.
- Indexing is image[y, x]. OpenCV functions take points as (x, y).
- Boxes are (x1, y1, x2, y2) in pixels ("xyxy") unless a function name says otherwise.
- Polygons are sequences of (x, y) points.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2  # noqa: F401
import numpy as np

Box = tuple[float, float, float, float]
Point = tuple[float, float]
Polygon = Sequence[tuple[float, float]]
Color = tuple[int, int, int]


def box_xywh_to_xyxy(box: tuple[float, float, float, float]) -> Box:
    """(x, y, w, h) with (x, y) the top-left corner  ->  (x1, y1, x2, y2).

    >>> box_xywh_to_xyxy((10, 20, 30, 40))
    (10.0, 20.0, 40.0, 60.0)
    """
    raise NotImplementedError


def box_xyxy_to_cxcywh(box: Box) -> tuple[float, float, float, float]:
    """(x1, y1, x2, y2)  ->  (center_x, center_y, width, height). This is YOLO's box format.

    >>> box_xyxy_to_cxcywh((10, 20, 40, 60))
    (25.0, 40.0, 30.0, 40.0)
    """
    raise NotImplementedError


def box_bottom_center(box: Box) -> Point:
    """The point where a standing person touches the floor: middle of the bottom edge.

    We use this (not the box center) to decide which floor zone a person is in.
    """
    raise NotImplementedError


def clip_box(box: Box, width: int, height: int) -> Box:
    """Clamp a box so it lies inside an image of the given size.

    x values are limited to [0, width], y values to [0, height]. Returns floats.
    """
    raise NotImplementedError


def crop(image: np.ndarray, box: Box) -> np.ndarray:
    """Return the part of `image` inside `box` as a **new array** (not a view).

    - Coordinates are rounded to the nearest integer after clipping to the image.
    - A box fully outside the image returns an empty array (size 0); it must not raise.
    - Modifying the returned crop must not modify `image`.
    """
    raise NotImplementedError


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """True if `point` is strictly inside `polygon` (works for concave polygons too).

    Must agree with
    cv2.pointPolygonTest for points not on the boundary (boundary behavior is unspecified).
    """
    raise NotImplementedError


def polygon_mask(height: int, width: int, polygon: Polygon) -> np.ndarray:
    """A (height, width) uint8 mask: 255 inside the polygon, 0 elsewhere."""
    raise NotImplementedError


def draw_zones(
    image: np.ndarray,
    zones: Mapping[str, Polygon],
    colors: Mapping[str, Color] | None = None,
    alpha: float = 0.35,
) -> np.ndarray:
    """Draw each zone as a translucent filled polygon with an outline and its name.

    - Returns a **new** image; `image` must be left unchanged.
    - `colors` maps zone name -> BGR color; zones missing from it use green (0, 255, 0).
    - `alpha` is the fill opacity (0 = invisible, 1 = solid).
    - Each zone is labeled with its name near the polygon.
    """
    raise NotImplementedError
