"""Tests for tablesense.geometry."""

import numpy as np
import pytest

from tablesense.geometry import (
    box_bottom_center,
    box_xywh_to_xyxy,
    box_xyxy_to_cxcywh,
    clip_box,
    crop,
    draw_zones,
    point_in_polygon,
    polygon_mask,
)

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
# An "L" shape: the top-right quadrant (x in 5..10, y in 0..5) is NOT inside.
L_SHAPE = [(0, 0), (5, 0), (5, 5), (10, 5), (10, 10), (0, 10)]


# --- box format conversions ---
def test_box_conversions():
    assert box_xywh_to_xyxy((10, 20, 30, 40)) == pytest.approx((10, 20, 40, 60))
    assert box_xyxy_to_cxcywh((10, 20, 40, 60)) == pytest.approx((25, 40, 30, 40))
    # round trip: xywh -> xyxy -> cxcywh
    cx, cy, w, h = box_xyxy_to_cxcywh(box_xywh_to_xyxy((0, 0, 8, 4)))
    assert (cx, cy, w, h) == pytest.approx((4, 2, 8, 4))
    assert isinstance(box_xywh_to_xyxy((1, 2, 3, 4)), tuple)


# --- bottom center ---
def test_bottom_center():
    assert box_bottom_center((10, 20, 40, 60)) == pytest.approx((25, 60))
    assert box_bottom_center((0, 0, 1, 1)) == pytest.approx((0.5, 1))


# --- clip_box / crop ---
def test_clip_and_crop():
    assert clip_box((-10, -5, 700, 400), 640, 360) == pytest.approx((0, 0, 640, 360))
    assert clip_box((10, 20, 30, 40), 640, 360) == pytest.approx((10, 20, 30, 40))

    image = np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3).astype(np.uint8)
    c = crop(image, (10, 20, 30, 40))
    assert c.shape == (20, 20, 3)
    np.testing.assert_array_equal(c, image[20:40, 10:30])

    # fractional coordinates round to nearest integer
    assert crop(image, (10.2, 19.8, 29.9, 40.1)).shape == (20, 20, 3)

    # partially outside: clipped
    assert crop(image, (-50, -50, 10, 10)).shape == (10, 10, 3)

    # fully outside: empty, no exception
    assert crop(image, (500, 500, 600, 600)).size == 0

    # the crop is a copy, not a view
    before = image.copy()
    c[:] = 0
    np.testing.assert_array_equal(image, before)


# --- point_in_polygon ---
@pytest.mark.parametrize(
    ("point", "polygon", "expected"),
    [
        ((5, 5), SQUARE, True),
        ((15, 5), SQUARE, False),
        ((-1, -1), SQUARE, False),
        ((2, 8), L_SHAPE, True),
        ((8, 8), L_SHAPE, True),
        ((8, 2), L_SHAPE, False),  # in the notch of the L
        ((2, 2), L_SHAPE, True),
        ((5, 3), [(0, 0), (10, 0), (5, 10)], True),  # triangle
        ((1, 9), [(0, 0), (10, 0), (5, 10)], False),
    ],
)
def test_point_in_polygon(point, polygon, expected):
    assert point_in_polygon(point, polygon) is expected


def test_point_in_polygon_matches_opencv_on_random_points():
    import cv2

    rng = np.random.default_rng(0)
    poly = np.array([(50, 10), (90, 40), (70, 90), (30, 80), (10, 40), (40, 45)], dtype=np.int32)
    for x, y in rng.uniform(0, 100, size=(300, 2)):
        d = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(x), float(y)), measureDist=True)
        if abs(d) < 1e-6:
            continue  # skip points exactly on an edge
        assert point_in_polygon((x, y), [tuple(p) for p in poly]) is (d > 0)


# --- polygon_mask ---
def test_polygon_mask():
    mask = polygon_mask(30, 40, [(10, 10), (20, 10), (20, 20), (10, 20)])
    assert mask.shape == (30, 40)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert mask[15, 15] == 255
    assert mask[5, 5] == 0
    assert mask[15, 30] == 0  # remember: mask[y, x]
    assert 100 <= int((mask == 255).sum()) <= 121


# --- draw_zones ---
def test_draw_zones():
    image = np.full((100, 200, 3), 50, dtype=np.uint8)
    before = image.copy()
    zones = {"T1": [(10, 10), (60, 10), (60, 50), (10, 50)]}

    out = draw_zones(image, zones, colors={"T1": (0, 0, 255)})

    np.testing.assert_array_equal(image, before)  # input untouched
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out[30, 35], image[30, 35])  # inside the zone changed
    np.testing.assert_array_equal(out[95, 195], image[95, 195])  # far away unchanged
    # translucent red fill: red channel goes up inside the zone
    assert out[40, 50, 2] > image[40, 50, 2]


def test_draw_zones_default_color_and_multiple_zones():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    zones = {
        "A": [(10, 10), (40, 10), (40, 40), (10, 40)],
        "B": [(110, 50), (190, 50), (190, 90), (110, 90)],
    }
    out = draw_zones(image, zones)
    assert out[35, 35, 1] > 0  # green-ish in A
    assert out[85, 180, 1] > 0  # green-ish in B
