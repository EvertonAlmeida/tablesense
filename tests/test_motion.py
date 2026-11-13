"""Tests for tablesense.motion."""

import numpy as np
import pytest

from tablesense.motion import DebouncedSwitch, MotionDetector, clean_mask, zone_activity
from tablesense.testing.synthetic import default_scene, iter_frames


def _zone_mask(scene, table_id):
    import cv2

    mask = np.zeros((scene.height, scene.width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(scene.table_zone(table_id), dtype=np.int32)], 255)
    return mask


# --- clean_mask ---
def test_clean_mask():
    raw = np.zeros((40, 40), dtype=np.uint8)
    raw[5:25, 5:25] = 255  # a solid blob...
    raw[15, 15] = 0  # ...with a 1-pixel hole
    raw[35, 35] = 255  # isolated noise pixel
    raw[30:34, 2:6] = 127  # a shadow

    out = clean_mask(raw, kernel_size=3)

    assert out.shape == raw.shape
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 255}
    assert out[15, 15] == 255  # hole filled (closing)
    assert out[35, 35] == 0  # noise removed (opening)
    assert out[31:33, 3:5].max() == 0  # shadow removed
    assert out[10, 10] == 255  # blob kept


def test_clean_mask_does_not_modify_input():
    raw = np.zeros((10, 10), dtype=np.uint8)
    raw[5, 5] = 255
    before = raw.copy()
    clean_mask(raw)
    np.testing.assert_array_equal(raw, before)


# --- zone_activity ---
def test_zone_activity():
    mask = np.zeros((10, 10), dtype=np.uint8)
    zone = np.zeros((10, 10), dtype=np.uint8)
    zone[0:4, 0:5] = 255  # 20 pixels
    assert zone_activity(mask, zone) == 0.0
    mask[0:2, 0:5] = 255  # 10 of them on
    mask[8, 8] = 255  # outside the zone, ignored
    assert zone_activity(mask, zone) == pytest.approx(0.5)
    assert zone_activity(mask, np.zeros_like(zone)) == 0.0


# --- MotionDetector ---
def test_motion_detector_on_synthetic():
    scene = default_scene()
    t3_zone = _zone_mask(scene, "T3")  # T3 is always empty
    detector = MotionDetector(history=100, var_threshold=16)

    t3_activity_after_warmup = []
    walker_coverage = None
    for i, frame, gts in iter_frames(scene):
        mask = detector.apply(frame)
        assert mask.shape == (scene.height, scene.width)
        assert mask.dtype == np.uint8
        if i < 30:
            continue  # let the background model warm up
        # Nobody sits at T3, but guest 4 walks past its lower edge. Only measure frames where
        # no person overlaps the zone: then any activity is pure noise.
        zx1, zy1 = scene.table_zone("T3")[0]
        zx2, zy2 = scene.table_zone("T3")[2]
        someone_in_zone = any(
            g.box[0] < zx2 and g.box[2] > zx1 and g.box[1] < zy2 and g.box[3] > zy1 for g in gts
        )
        if not someone_in_zone:
            t3_activity_after_warmup.append(zone_activity(mask, t3_zone))
        if i == 55:
            # guest 2 is walking across the room at frame 55
            g2 = next(g for g in gts if g.person_id == 2)
            x1, y1, x2, y2 = (int(round(v)) for v in g2.box)
            walker_coverage = (mask[y1:y2, x1:x2] > 0).mean()

    assert walker_coverage is not None and walker_coverage > 0.3, "moving person not detected"
    assert max(t3_activity_after_warmup) < 0.02, "empty table shows motion (noise not cleaned?)"


# --- DebouncedSwitch ---
def test_debounced_switch_turns_on_after_min_duration():
    sw = DebouncedSwitch(on_threshold=0.3, off_threshold=0.1, min_on_seconds=2, min_off_seconds=3)
    assert sw.state is False
    assert sw.update(0.5, 0) is False
    assert sw.update(0.5, 1) is False
    assert sw.update(0.5, 2) is True
    assert sw.state is True


def test_debounced_switch_ignores_spikes():
    sw = DebouncedSwitch(0.3, 0.1, 2, 3)
    assert sw.update(0.5, 0) is False
    assert sw.update(0.0, 1) is False  # interruption resets the timer
    assert sw.update(0.5, 2) is False
    assert sw.update(0.5, 3) is False
    assert sw.update(0.5, 4) is True


def test_debounced_switch_hysteresis_and_turn_off():
    sw = DebouncedSwitch(0.3, 0.1, 2, 3, initial=True)
    for t in range(10):
        assert sw.update(0.2, t) is True  # between thresholds: stays on
    assert sw.update(0.05, 10) is True
    assert sw.update(0.05, 11) is True
    assert sw.update(0.2, 12) is True  # back above off threshold: resets off timer
    assert sw.update(0.05, 13) is True
    assert sw.update(0.05, 15) is True
    assert sw.update(0.05, 16) is False  # 16 - 13 >= 3


def test_debounced_switch_validates_thresholds():
    with pytest.raises(ValueError):
        DebouncedSwitch(on_threshold=0.1, off_threshold=0.3, min_on_seconds=1, min_off_seconds=1)
