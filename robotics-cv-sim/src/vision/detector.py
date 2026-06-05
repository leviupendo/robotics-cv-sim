"""
detector.py
===========
OpenCV-based object detection for vision-guided robot control.

Pipeline:
  RGB frame → HSV conversion → colour mask → morphological cleanup
  → contour detection → bounding box → 3D position estimate
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Colour profiles ────────────────────────────────────────────────────────────
COLOR_PROFILES = {
    "red": {
        "lower1": np.array([0,   120, 70]),
        "upper1": np.array([10,  255, 255]),
        "lower2": np.array([170, 120, 70]),
        "upper2": np.array([180, 255, 255]),
        "dual":   True,
    },
    "green": {
        "lower1": np.array([35, 80, 50]),
        "upper1": np.array([85, 255, 255]),
        "dual":   False,
    },
    "blue": {
        "lower1": np.array([100, 80, 50]),
        "upper1": np.array([130, 255, 255]),
        "dual":   False,
    },
    "yellow": {
        "lower1": np.array([20, 100, 100]),
        "upper1": np.array([35, 255, 255]),
        "dual":   False,
    },
}


@dataclass
class Detection:
    color: str
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in pixels
    centroid: Tuple[int, int]          # (cx, cy) in pixels
    area: float
    position_3d: Optional[np.ndarray] = None  # (x, y, z) in metres


def detect_objects(
    frame: np.ndarray,
    colors: List[str] = None,
    min_area: int = 500,
) -> List[Detection]:
    """
    Detect coloured objects in an RGB frame.

    Parameters
    ----------
    frame    : (H, W, 3) uint8 BGR image (OpenCV convention)
    colors   : list of colour names to detect; None = all
    min_area : minimum contour area in pixels²

    Returns
    -------
    List of Detection objects
    """
    if colors is None:
        colors = list(COLOR_PROFILES.keys())

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    detections: List[Detection] = []

    for color in colors:
        if color not in COLOR_PROFILES:
            continue
        prof = COLOR_PROFILES[color]

        mask = cv2.inRange(hsv, prof["lower1"], prof["upper1"])
        if prof.get("dual"):
            mask2 = cv2.inRange(hsv, prof["lower2"], prof["upper2"])
            mask = cv2.bitwise_or(mask, mask2)

        # Morphological cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            detections.append(Detection(
                color=color,
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                area=float(area),
            ))

    return detections


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the frame."""
    COLOR_BGR = {
        "red":    (0, 0, 220),
        "green":  (0, 200, 0),
        "blue":   (220, 80, 0),
        "yellow": (0, 200, 220),
    }
    out = frame.copy()
    for d in detections:
        x, y, w, h = d.bbox
        bgr = COLOR_BGR.get(d.color, (200, 200, 200))
        cv2.rectangle(out, (x, y), (x + w, y + h), bgr, 2)
        cv2.circle(out, d.centroid, 4, bgr, -1)
        label = d.color
        if d.position_3d is not None:
            p = d.position_3d
            label += f" ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"
        cv2.putText(out, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, bgr, 1, cv2.LINE_AA)
    return out
