 🤖 6-DOF Robotic Arm Simulation with Computer Vision

> **Course:** Robotics and Computer Integrated Manufacturing — Advanced  
> **Project #18** | Robotics Simulation & Vision-Guided Control

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-scientific-orange?logo=numpy)
![Flask](https://img.shields.io/badge/Flask-web--server-lightgrey?logo=flask)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 📌 Overview

A **real-time 6-DOF robotic arm simulator** with integrated computer vision for object detection and vision-guided pick-and-place operations. The system computes forward/inverse kinematics, plans collision-free paths, and uses OpenCV to detect and localize target objects in a simulated workspace.

### ✨ Key Features

| Feature | Description |
|---|---|
| 🦾 **6-DOF Kinematics** | Full forward & inverse kinematics using DH parameters |
| 👁️ **Computer Vision** | Object detection via OpenCV (color segmentation + contour detection) |
| 🧠 **IK Solver** | Jacobian pseudo-inverse iterative solver |
| 🛤️ **Path Planning** | Joint-space trajectory interpolation with velocity profiling |
| 🌐 **Web Dashboard** | Real-time 3D visualization via browser (Three.js) |
| 📡 **WebSocket Stream** | Live joint state + vision feed over WebSocket |
| 🎮 **Manual Control** | Slider-based joint control UI |

---

## 🗂️ Project Structure

```
robotics-cv-sim/
├── src/
│   ├── robot/
│   │   ├── __init__.py
│   │   ├── kinematics.py       # DH parameters, FK, IK solver
│   │   ├── trajectory.py       # Path planning & interpolation
│   │   └── robot_arm.py        # Main RobotArm class
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── detector.py         # OpenCV object detection
│   │   └── workspace.py        # Workspace camera simulation
│   └── control/
│       ├── __init__.py
│       └── controller.py       # Vision-guided pick-and-place
├── web/
│   ├── static/
│   │   ├── sim.js              # Three.js 3D visualizer
│   │   └── style.css
│   └── templates/
│       └── index.html          # Dashboard UI
├── tests/
│   ├── test_kinematics.py
│   └── test_vision.py
├── docs/
│   └── kinematics_derivation.md
├── app.py                      # Flask + WebSocket server
├── requirements.txt
├── demo.py                     # Standalone CLI demo
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/robotics-cv-sim.git
cd robotics-cv-sim

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Web Dashboard
```bash
python app.py
# Open http://localhost:5000
```

### Run CLI Demo
```bash
python demo.py
```

### Run Tests
```bash
pytest tests/ -v
```

---

## 🧮 Kinematics

The arm uses **Denavit-Hartenberg (DH) parameters** for a 6-DOF configuration:

| Joint | a (m) | d (m) | α (rad) | θ offset |
|---|---|---|---|---|
| 1 | 0 | 0.333 | 0 | 0 |
| 2 | 0 | 0 | -π/2 | 0 |
| 3 | 0 | 0.316 | π/2 | 0 |
| 4 | 0.0825 | 0 | π/2 | 0 |
| 5 | -0.0825 | 0.384 | -π/2 | 0 |
| 6 | 0 | 0 | π/2 | 0 |

**Inverse Kinematics** is solved iteratively using the Jacobian pseudo-inverse method.

---

## 👁️ Computer Vision Pipeline

```
Camera Frame → Preprocessing → Color Segmentation → Contour Detection
     → Bounding Box → 3D Position Estimate → IK Target → Motion Plan
```

Objects are detected by HSV color range. The 2D pixel coordinates are back-projected to 3D workspace coordinates using a known camera-to-robot transform.

---

## 📸 Screenshots

> Add screenshots of your dashboard and simulation here.

---

## 📚 References

- Denavit, J. & Hartenberg, R.S. (1955). *A kinematic notation for lower-pair mechanisms*
- Siciliano, B. et al. (2009). *Robotics: Modelling, Planning and Control*
- OpenCV Documentation: https://docs.opencv.org

---

## 👤 Author

LEVI UPENDO— Robotics and Computer Integrated Manufacturing, Advanced  
*Project #18*

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
