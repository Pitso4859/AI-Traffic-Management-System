"""
Flask dashboard for the Smart Traffic Management System.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Optional
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from utils.config import LANE_NAMES, SAMPLE_DIR, set_lane_image
from analytics.dataset_analyzer import get_dataset_analyzer

if TYPE_CHECKING:
    from detection.lane_detector import LaneDetector
    from logic.signal_controller import SignalController
    from chatbot.traffic_assistant import TrafficAssistant
    from analytics.predictor import TrafficPredictor, TrafficMetrics

app = Flask(__name__)

detectors: dict[str, LaneDetector] = {}
controller: SignalController | None = None
assistant: TrafficAssistant | None = None
predictor: TrafficPredictor | None = None
metrics: TrafficMetrics | None = None


def set_shared_state(
    det: dict[str, LaneDetector],
    ctrl: SignalController,
    asst: TrafficAssistant = None,
    pred: TrafficPredictor = None,
    met: TrafficMetrics = None,
) -> None:
    global detectors, controller, assistant, predictor, metrics
    detectors = det
    controller = ctrl
    assistant = asst
    predictor = pred
    metrics = met


def _generate_mjpeg(lane_name: str):
    while True:
        det = detectors.get(lane_name)
        if det is None:
            time.sleep(0.5)
            continue
        with det.lock:
            frame = det.latest_result.annotated_frame
        if frame is None:
            time.sleep(0.1)
            continue
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.05)


@app.route("/feed/<lane_name>")
def video_feed(lane_name: str):
    if lane_name not in LANE_NAMES:
        return "Lane not found", 404
    return Response(_generate_mjpeg(lane_name), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    if controller is None:
        return jsonify({"error": "Controller not initialised"}), 503
    return jsonify(controller.get_status())


@app.route("/api/chat", methods=["POST"])
def chat():
    if assistant is None:
        return jsonify({"error": "Chatbot not initialised"}), 503
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400
    try:
        response = assistant.process_query(data["message"].strip())
        return jsonify({"success": True, "response": response})
    except Exception as e:
        return jsonify({"error": str(e), "response": "Error processing request"}), 500


@app.route("/api/predict/<lane_name>")
def predict_lane(lane_name: str):
    if lane_name not in LANE_NAMES:
        return jsonify({"error": "Lane not found"}), 404
    if predictor is None:
        return jsonify({"error": "Predictor not initialised"}), 503
    pred_count, conf = predictor.predict_next_count(lane_name)
    pred_wait, wait_conf = predictor.predict_waiting_time(lane_name)
    trend = predictor.get_traffic_trend(lane_name)
    return jsonify({
        "lane": lane_name,
        "predicted_vehicles": pred_count,
        "prediction_confidence": round(conf, 2),
        "predicted_waiting_seconds": pred_wait,
        "waiting_confidence": round(wait_conf, 2),
        "trend": trend
    })


@app.route("/api/dataset-predict")
def dataset_predict():
    hour = request.args.get("hour", type=int, default=datetime.now().hour)
    weather = request.args.get("weather", type=str, default="Sunny")
    analyzer = get_dataset_analyzer()
    result = analyzer.predict_traffic_volume(hour, weather=weather)
    return jsonify({"hour": hour, "weather": weather, "prediction": result})


@app.route("/api/weather-impact")
def weather_impact():
    weather = request.args.get("weather", type=str, default="Sunny")
    analyzer = get_dataset_analyzer()
    impact = analyzer.get_weather_impact(weather)
    return jsonify({"weather": weather, "volume_factor": round(impact["volume_factor"], 2)})


@app.route("/api/metrics")
def get_metrics():
    if metrics is None:
        return jsonify({"error": "Metrics not initialised"}), 503
    return jsonify(metrics.get_metrics())


@app.route("/api/trend/<lane_name>")
def get_trend(lane_name: str):
    if lane_name not in LANE_NAMES:
        return jsonify({"error": "Lane not found"}), 404
    if predictor is None:
        return jsonify({"error": "Predictor not initialised"}), 503
    trend = predictor.get_traffic_trend(lane_name)
    return jsonify({"lane": lane_name, "trend": trend})


@app.route("/upload/<lane_name>", methods=["POST"])
def upload_image(lane_name: str):
    if lane_name not in LANE_NAMES:
        return jsonify({"error": "Lane not found"}), 404
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        safe_name = f"{lane_name}_{int(time.time())}_{filename}"
        filepath = os.path.join(SAMPLE_DIR, safe_name)
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        file.save(filepath)
        set_lane_image(lane_name, filepath)
        return jsonify({"success": True})
    return jsonify({"error": "Unknown error"}), 500


@app.route("/")
def index():
    return render_template("index.html")