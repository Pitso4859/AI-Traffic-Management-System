"""
Smart Traffic Management System — Entry Point.
"""

from __future__ import annotations

import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.lane_detector import LaneDetector
from logic.signal_controller import SignalController
from dashboard.app import app, set_shared_state
from utils.config import DETECTION_INTERVAL, FLASK_PORT, LANE_NAMES
from chatbot.traffic_assistant import TrafficAssistant
from analytics.predictor import TrafficPredictor, TrafficMetrics
from analytics.dataset_analyzer import get_dataset_analyzer


_controller: SignalController = None
_detectors: dict[str, LaneDetector] = {}
_predictor: TrafficPredictor = None
_metrics: TrafficMetrics = None
_assistant: TrafficAssistant = None


def detection_loop(detector: LaneDetector, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            detector.detect()
        except Exception as exc:
            print(f"[{detector.lane_name}] error: {exc} - main.py:35")
        time.sleep(DETECTION_INTERVAL)


def control_loop(controller: SignalController, detectors: dict, predictor: TrafficPredictor,
                 metrics: TrafficMetrics, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        counts = {}
        emergencies = {}
        for name, det in detectors.items():
            with det.lock:
                counts[name] = det.latest_result.vehicle_count
                emergencies[name] = det.latest_result.emergency_detected
        
        total = sum(counts.values())
        metrics.record_vehicles(total)
        if any(emergencies.values()):
            metrics.record_emergency()
        
        status = controller.get_status()
        for name in LANE_NAMES:
            with detectors[name].lock:
                wait = status.get("lanes", {}).get(name, {}).get("waiting_seconds", 0)
                signal = status.get("lanes", {}).get(name, {}).get("signal", "RED")
                predictor.add_sample(name, counts[name], signal, wait)
            if wait > 0:
                metrics.record_waiting_time(wait)
        
        metrics.update_throughput(total, DETECTION_INTERVAL)
        controller.update(counts, emergencies)
        
        new_status = controller.get_status()
        for name in LANE_NAMES:
            new_sig = new_status.get("lanes", {}).get(name, {}).get("signal", "RED")
            old_sig = status.get("lanes", {}).get(name, {}).get("signal", "RED")
            if new_sig != old_sig and new_sig == "GREEN":
                metrics.record_signal_change()
                break
        
        time.sleep(DETECTION_INTERVAL)


def main() -> None:
    global _controller, _detectors, _predictor, _metrics, _assistant
    
    print("= - main.py:80" * 60)
    print("Smart Traffic Management System - main.py:81")
    print("With AI Chatbot & Predictive Analytics - main.py:82")
    print("= - main.py:83" * 60)
    
    print("\n📊 Loading dataset... - main.py:85")
    get_dataset_analyzer()
    print("✓ Dataset loaded - main.py:87")

    _detectors = {}
    for name in LANE_NAMES:
        print(f"► Initialising {name} lane - main.py:91")
        _detectors[name] = LaneDetector(name)
    print()

    _controller = SignalController()
    _predictor = TrafficPredictor()
    _metrics = TrafficMetrics()
    _assistant = TrafficAssistant(_controller, _detectors)
    set_shared_state(_detectors, _controller, _assistant, _predictor, _metrics)

    stop_event = threading.Event()
    threads = []

    for name, det in _detectors.items():
        t = threading.Thread(target=detection_loop, args=(det, stop_event), daemon=True, name=f"detect-{name}")
        t.start()
        threads.append(t)
        print(f"✓ Detection started: {name} - main.py:108")

    sig_thread = threading.Thread(target=control_loop, args=(_controller, _detectors, _predictor, _metrics, stop_event), daemon=True)
    sig_thread.start()
    print("✓ Controller started - main.py:112")

    print(f"\n Dashboard: http://localhost:{FLASK_PORT}\n - main.py:114")
    
    try:
        app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down... - main.py:119")
        stop_event.set()
        for t in threads:
            t.join(timeout=2)
        sig_thread.join(timeout=2)
        print("Done. - main.py:124")


if __name__ == "__main__":
    main()