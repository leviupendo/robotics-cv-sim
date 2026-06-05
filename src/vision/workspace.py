"""
workspace.py
============
Simulates a top-down camera looking at the robot workspace.
Generates synthetic frames with coloured target objects.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict


# Camera intrinsics for a 640×480 simulated camera
CAMERA_FX = 600.0
CAMERA_FY = 600.0
CAMERA_CX = 320.0
CAMERA_CY = 240.0
CAMERA_HEIGHT = 1.2   # metres above workspace origin

# Workspace bounds in metres
WORKSPACE_X = (-0.5, 0.5)
WORKSPACE_Y = (-0.5, 0.5)


class WorkspaceCamera:
    """
    Simulates a top-down RGB camera over the robot workspace.
    Places coloured circular objects and returns BGR frames.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.objects: List[Dict] = []   # list of {color, pos_world}
        self._rng = np.random.default_rng(42)

    def add_object(self, color: str, x: float, y: float):
        """Add a coloured target object at world position (x, y)."""
        self.objects.append({"color": color, "pos": np.array([x, y, 0.0])})

    def random_scene(self, n_objects: int = 3):
        """Randomly place n_objects in the workspace."""
        self.objects.clear()
        colors = ["red", "green", "blue", "yellow"]
        for _ in range(n_objects):
            x = self._rng.uniform(*WORKSPACE_X)
            y = self._rng.uniform(*WORKSPACE_Y)
            c = colors[self._rng.integers(len(colors))]
            self.add_object(c, x, y)

    def render(self) -> np.ndarray:
        """Render the current scene to a BGR image."""
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 200

        # Grid overlay
        for gx in np.linspace(0, self.width,  11, dtype=int):
            cv2.line(frame, (gx, 0), (gx, self.height), (180, 180, 180), 1)
        for gy in np.linspace(0, self.height, 11, dtype=int):
            cv2.line(frame, (0, gy), (self.width, gy), (180, 180, 180), 1)

        # Origin cross
        ox, oy = int(CAMERA_CX), int(CAMERA_CY)
        cv2.line(frame, (ox - 10, oy), (ox + 10, oy), (80, 80, 80), 2)
        cv2.line(frame, (ox, oy - 10), (ox, oy + 10), (80, 80, 80), 2)

        COLOR_BGR = {
            "red":    (0, 0, 220),
            "green":  (0, 180, 0),
            "blue":   (200, 80, 0),
            "yellow": (0, 200, 220),
        }
        for obj in self.objects:
            px, py = self._world_to_pixel(obj["pos"])
            bgr = COLOR_BGR.get(obj["color"], (128, 128, 128))
            cv2.circle(frame, (px, py), 18, bgr, -1)
            cv2.circle(frame, (px, py), 18, (50, 50, 50), 2)

        return frame

    def pixel_to_world(self, px: int, py: int, z_world: float = 0.0) -> np.ndarray:
        """Back-project pixel to 3D world coordinates (assuming flat z=z_world plane)."""
        x = (px - CAMERA_CX) / CAMERA_FX * CAMERA_HEIGHT
        y = (py - CAMERA_CY) / CAMERA_FY * CAMERA_HEIGHT
        return np.array([x, y, z_world])

    def _world_to_pixel(self, pos: np.ndarray) -> Tuple[int, int]:
        px = int(pos[0] * CAMERA_FX / CAMERA_HEIGHT + CAMERA_CX)
        py = int(pos[1] * CAMERA_FY / CAMERA_HEIGHT + CAMERA_CY)
        return px, py
