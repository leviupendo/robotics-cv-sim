"""
demo.py
=======
Standalone CLI demo — no browser required.
Runs FK, IK, and a vision-guided pick sequence and prints results.
"""

import numpy as np
from src.robot.kinematics import (
    forward_kinematics, inverse_kinematics,
    get_ee_position, joint_angles_to_dict
)
from src.robot.trajectory import trapezoidal_profile
from src.robot.robot_arm import RobotArm
from src.vision.workspace import WorkspaceCamera
from src.vision.detector import detect_objects


def hr(title=""):
    print("\n" + "─" * 55)
    if title: print(f"  {title}")
    print("─" * 55)


def demo_fk():
    hr("Forward Kinematics Demo")
    q = np.radians([0, -45, 0, -90, 0, 45])
    T, transforms = forward_kinematics(q)
    print(f"  Joint config: {np.round(np.degrees(q), 1)} deg")
    print(f"  EE position : {T[:3, 3].round(4)} m")
    print(f"  EE rotation :\n{T[:3, :3].round(3)}")


def demo_ik():
    hr("Inverse Kinematics Demo")
    targets = [
        np.array([0.4,  0.0,  0.4]),
        np.array([0.3,  0.3,  0.2]),
        np.array([0.0,  0.5,  0.3]),
        np.array([-0.3, 0.2,  0.5]),
    ]
    for target in targets:
        q_sol, success, err = inverse_kinematics(target)
        ee = get_ee_position(q_sol)
        print(f"  Target {target} → {'✓' if success else '✗'} err={err:.5f}m | EE={ee.round(4)}")


def demo_trajectory():
    hr("Trajectory Planning Demo")
    q_start = np.zeros(6)
    q_end   = np.radians([30, -60, 45, -90, 60, -30])
    t, q_traj, qd_traj = trapezoidal_profile(q_start, q_end, duration=2.0, freq=50)
    print(f"  Trajectory: {len(t)} points over {t[-1]:.1f}s")
    print(f"  Start EE: {get_ee_position(q_start).round(3)} m")
    print(f"  End   EE: {get_ee_position(q_end).round(3)} m")
    print(f"  Peak joint vel: {np.abs(qd_traj).max(axis=0).round(3)} rad/s")


def demo_vision():
    hr("Computer Vision Detection Demo")
    cam = WorkspaceCamera()
    cam.random_scene(n_objects=4)
    frame = cam.render()
    print(f"  Frame size: {frame.shape}")
    detections = detect_objects(frame)
    print(f"  Detected {len(detections)} object(s):")
    for d in detections:
        world = cam.pixel_to_world(*d.centroid)
        print(f"    [{d.color}] centroid={d.centroid}  world={world.round(3)} m  area={d.area:.0f}px²")


def demo_pick_place():
    hr("Vision-Guided Pick & Place Demo")
    arm = RobotArm()
    cam = WorkspaceCamera()
    cam.add_object("red",   0.3,  0.1)
    cam.add_object("green", -0.2, 0.25)

    frame = cam.render()
    detections = detect_objects(frame, colors=["red"])

    if not detections:
        print("  No target found!")
        return

    best = max(detections, key=lambda d: d.area)
    world_pos = cam.pixel_to_world(*best.centroid)
    pick_pos  = world_pos.copy(); pick_pos[2] = 0.05

    print(f"  Detected {best.color} at pixel {best.centroid} → world {world_pos.round(3)}")
    ok = arm.move_to_position(pick_pos)
    print(f"  IK solved: {'✓' if ok else '✗'}")

    steps = 0
    while arm.step_trajectory():
        steps += 1
    print(f"  Trajectory executed: {steps} steps")
    print(f"  Final {arm}")


if __name__ == "__main__":
    print("\n🤖  Robotics CV Sim — CLI Demo")
    demo_fk()
    demo_ik()
    demo_trajectory()
    demo_vision()
    demo_pick_place()
    hr()
    print("  ✓ All demos complete. Run `python app.py` for the web dashboard.\n")
