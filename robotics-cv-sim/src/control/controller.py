"""
controller.py
=============
Vision-guided pick-and-place controller.

Orchestrates: camera → detection → IK → trajectory execution.
"""

import numpy as np
from ..robot.robot_arm import RobotArm
from ..vision.detector import detect_objects
from ..vision.workspace import WorkspaceCamera


class PickPlaceController:
    """
    High-level controller that:
      1. Renders a camera frame
      2. Detects coloured objects
      3. Back-projects detections to 3D positions
      4. Commands the robot arm to pick-and-place
    """

    PICK_HEIGHT  = 0.05   # metres above the object
    PLACE_TARGET = np.array([0.4, 0.0, 0.1])  # fixed drop zone

    def __init__(self):
        self.arm    = RobotArm()
        self.camera = WorkspaceCamera()
        self.camera.random_scene(n_objects=3)
        self.log: list = []

    # ── Main loop step ─────────────────────────────────────────────────────────

    def run_step(self, target_color: str = "red") -> dict:
        """
        Single perception → planning step.

        Returns a dict with detection results and planned target position.
        """
        frame   = self.camera.render()
        detects = detect_objects(frame, colors=[target_color])

        if not detects:
            self._log(f"No {target_color} object detected.")
            return {"success": False, "detections": [], "arm_state": self.arm.get_state()}

        # Pick the largest detection
        best = max(detects, key=lambda d: d.area)
        world_pos = self.camera.pixel_to_world(*best.centroid)
        best.position_3d = world_pos

        pick_pos = world_pos.copy()
        pick_pos[2] = self.PICK_HEIGHT

        success = self.arm.move_to_position(pick_pos)
        self._log(
            f"Detected {target_color} at {world_pos.round(3)} | "
            f"IK {'✓' if success else '✗'}"
        )

        return {
            "success": success,
            "detections": [
                {
                    "color":      best.color,
                    "centroid":   best.centroid,
                    "position3d": world_pos.tolist(),
                    "area":       best.area,
                }
            ],
            "pick_target":  pick_pos.tolist(),
            "place_target": self.PLACE_TARGET.tolist(),
            "arm_state":    self.arm.get_state(),
        }

    def step_arm(self) -> bool:
        """Advance the robot arm one trajectory step."""
        return self.arm.step_trajectory()

    def get_full_state(self) -> dict:
        return {
            "arm":    self.arm.get_state(),
            "scene":  [{"color": o["color"], "pos": o["pos"].tolist()}
                       for o in self.camera.objects],
            "log":    self.log[-20:],
        }

    def _log(self, msg: str):
        self.log.append(msg)
        print(f"[Controller] {msg}")
