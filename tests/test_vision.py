"""tests/test_vision.py — Unit tests for CV detection and workspace camera."""

import numpy as np
import pytest
from src.vision.workspace import WorkspaceCamera
from src.vision.detector import detect_objects, Detection


class TestWorkspaceCamera:
    def test_render_shape(self):
        cam = WorkspaceCamera(640, 480)
        frame = cam.render()
        assert frame.shape == (480, 640, 3)

    def test_render_dtype(self):
        cam = WorkspaceCamera()
        frame = cam.render()
        assert frame.dtype == np.uint8

    def test_add_object_stored(self):
        cam = WorkspaceCamera()
        cam.add_object("red", 0.1, 0.2)
        assert len(cam.objects) == 1

    def test_random_scene_count(self):
        cam = WorkspaceCamera()
        cam.random_scene(n_objects=4)
        assert len(cam.objects) == 4

    def test_pixel_to_world_origin(self):
        cam = WorkspaceCamera()
        world = cam.pixel_to_world(320, 240)
        assert np.allclose(world[:2], [0.0, 0.0], atol=0.01)

    def test_roundtrip_world_pixel(self):
        cam = WorkspaceCamera()
        # Internal method for testing consistency
        pos = np.array([0.2, -0.1, 0.0])
        px, py = cam._world_to_pixel(pos)
        recovered = cam.pixel_to_world(px, py)
        assert np.allclose(recovered[:2], pos[:2], atol=0.01)


class TestDetector:
    def _make_color_frame(self, bgr_color, size=(200, 200)):
        import numpy as np
        frame = np.zeros((*size, 3), dtype=np.uint8)
        # Draw a solid circle in the centre
        import cv2
        cv2.circle(frame, (100, 100), 40, bgr_color, -1)
        return frame

    def test_detects_red(self):
        frame = self._make_color_frame((0, 0, 220))  # BGR red
        dets = detect_objects(frame, colors=["red"])
        assert len(dets) >= 1
        assert dets[0].color == "red"

    def test_detects_green(self):
        frame = self._make_color_frame((0, 200, 0))
        dets = detect_objects(frame, colors=["green"])
        assert len(dets) >= 1

    def test_no_detection_on_black(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        import numpy as np
        dets = detect_objects(frame)
        assert len(dets) == 0

    def test_detection_has_centroid(self):
        frame = self._make_color_frame((0, 0, 220))
        dets = detect_objects(frame, colors=["red"])
        if dets:
            cx, cy = dets[0].centroid
            assert 60 < cx < 140
            assert 60 < cy < 140
