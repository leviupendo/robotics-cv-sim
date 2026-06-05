"""
app.py
======
Flask + Flask-SocketIO web server.
Streams robot state to the browser dashboard in real time.
"""

import time
import threading
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from src.control.controller import PickPlaceController
from src.robot.kinematics import inverse_kinematics

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
app.config["SECRET_KEY"] = "robotics-cv-sim-2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

controller = PickPlaceController()
_sim_running = False
_sim_thread = None


# ── HTTP Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    return jsonify(controller.get_full_state())


@app.route("/api/detect", methods=["POST"])
def detect():
    color = request.json.get("color", "red")
    result = controller.run_step(target_color=color)
    return jsonify(result)


@app.route("/api/joints", methods=["POST"])
def set_joints():
    """Directly set joint angles (degrees) from the manual control UI."""
    data = request.json
    q_deg = [data.get(f"joint_{i+1}", 0.0) for i in range(6)]
    q_rad = np.radians(q_deg)
    controller.arm.set_joints(q_rad)
    return jsonify(controller.arm.get_state())


@app.route("/api/home", methods=["POST"])
def go_home():
    controller.arm.go_home()
    return jsonify({"status": "homing"})


@app.route("/api/scene/randomize", methods=["POST"])
def randomize_scene():
    controller.camera.random_scene()
    return jsonify({"status": "ok", "scene": [
        {"color": o["color"], "pos": o["pos"].tolist()}
        for o in controller.camera.objects
    ]})


# ── WebSocket ──────────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    emit("state", controller.get_full_state())


@socketio.on("start_sim")
def start_sim(data):
    global _sim_running, _sim_thread
    if _sim_running:
        return
    _sim_running = True
    color = data.get("color", "red")
    controller.run_step(target_color=color)
    _sim_thread = threading.Thread(target=_sim_loop, daemon=True)
    _sim_thread.start()


@socketio.on("stop_sim")
def stop_sim():
    global _sim_running
    _sim_running = False


def _sim_loop():
    """Background thread: steps the trajectory and broadcasts state."""
    global _sim_running
    while _sim_running:
        has_more = controller.step_arm()
        state = controller.get_full_state()
        socketio.emit("state", state)
        if not has_more:
            _sim_running = False
        time.sleep(0.02)   # 50 Hz broadcast


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Robotics CV Sim — http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
