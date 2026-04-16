"""
Traffic prediction and analytics module.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import numpy as np

from analytics.dataset_analyzer import get_dataset_analyzer


@dataclass
class TrafficSample:
    timestamp: float
    lane_name: str
    vehicle_count: int
    signal_state: str
    waiting_seconds: int


class TrafficPredictor:
    def __init__(self, max_history: int = 100):
        self._lock = threading.Lock()
        self._history: Dict[str, deque] = {}
        self._max_history = max_history
        self._dataset_analyzer = get_dataset_analyzer()

    def add_sample(self, lane_name: str, vehicle_count: int, signal_state: str, waiting_seconds: int) -> None:
        with self._lock:
            if lane_name not in self._history:
                self._history[lane_name] = deque(maxlen=self._max_history)
            self._history[lane_name].append(TrafficSample(
                timestamp=time.time(), lane_name=lane_name,
                vehicle_count=vehicle_count, signal_state=signal_state, waiting_seconds=waiting_seconds
            ))

    def predict_next_count(self, lane_name: str) -> Tuple[int, float]:
        with self._lock:
            if lane_name not in self._history or len(self._history[lane_name]) < 3:
                return 0, 0.0
            samples = list(self._history[lane_name])
            counts = [s.vehicle_count for s in samples]
            weights = np.exp(np.linspace(0, 1, len(counts)))
            weights /= weights.sum()
            weighted_avg = np.average(counts, weights=weights)
            if len(counts) >= 5:
                recent_avg = np.mean(counts[-3:])
                older_avg = np.mean(counts[:-3])
                trend = recent_avg - older_avg
            else:
                trend = 0
            predicted = max(0, int(weighted_avg + trend * 0.5))
            std_dev = np.std(counts) if len(counts) > 1 else 0
            confidence = min(0.95, 0.7 / (std_dev + 0.5)) if std_dev > 0 else 0.5
            return predicted, confidence

    def predict_waiting_time(self, lane_name: str) -> Tuple[int, float]:
        with self._lock:
            if lane_name not in self._history or len(self._history[lane_name]) < 5:
                return 0, 0.0
            samples = list(self._history[lane_name])
            waiting_times = [s.waiting_seconds for s in samples]
            x = np.arange(len(waiting_times))
            if len(x) > 1:
                slope = np.polyfit(x, waiting_times, 1)[0]
                predicted = max(0, int(waiting_times[-1] + slope))
            else:
                predicted = waiting_times[-1] if waiting_times else 0
            confidence = min(0.9, 0.6 + abs(slope) / 10) if 'slope' in locals() else 0.5
            return predicted, confidence

    def get_traffic_trend(self, lane_name: str) -> Dict[str, any]:
        with self._lock:
            if lane_name not in self._history or len(self._history[lane_name]) < 3:
                return {"trend": "insufficient_data", "change_rate": 0}
            samples = list(self._history[lane_name])
            counts = [s.vehicle_count for s in samples]
            if len(counts) >= 5:
                recent_avg = np.mean(counts[-3:])
                older_avg = np.mean(counts[:-3])
                change = recent_avg - older_avg
                trend = "increasing" if change > 2 else ("decreasing" if change < -2 else "stable")
            else:
                trend = "stable"
                change = 0
            return {"trend": trend, "change_rate": round(change, 1), "current": counts[-1] if counts else 0}

    def get_dataset_prediction(self, hour: int, weather: str = "Sunny") -> Dict[str, Any]:
        return self._dataset_analyzer.predict_traffic_volume(hour=hour, weather=weather)

    def get_weather_impact(self, weather: str) -> Dict[str, float]:
        return self._dataset_analyzer.get_weather_impact(weather)


class TrafficMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics = {
            "total_vehicles_processed": 0,
            "emergency_responses": 0,
            "signal_changes": 0,
            "average_waiting_time": 0,
            "max_waiting_time": 0,
            "throughput": 0,
            "uptime_seconds": 0,
            "start_time": time.time()
        }

    def record_vehicles(self, count: int) -> None:
        with self._lock:
            self._metrics["total_vehicles_processed"] += count

    def record_emergency(self) -> None:
        with self._lock:
            self._metrics["emergency_responses"] += 1

    def record_signal_change(self) -> None:
        with self._lock:
            self._metrics["signal_changes"] += 1

    def record_waiting_time(self, seconds: int) -> None:
        with self._lock:
            total = self._metrics["average_waiting_time"] * max(1, self._metrics["signal_changes"])
            self._metrics["signal_changes"] = max(1, self._metrics["signal_changes"])
            self._metrics["average_waiting_time"] = (total + seconds) / self._metrics["signal_changes"]
            self._metrics["max_waiting_time"] = max(self._metrics["max_waiting_time"], seconds)

    def update_throughput(self, total_vehicles: int, time_elapsed: float) -> None:
        with self._lock:
            if time_elapsed > 0:
                self._metrics["throughput"] = total_vehicles / time_elapsed
            self._metrics["uptime_seconds"] = time.time() - self._metrics["start_time"]

    def get_metrics(self) -> Dict[str, any]:
        with self._lock:
            return self._metrics.copy()

    def get_performance_report(self) -> str:
        with self._lock:
            return f"""
📊 System Performance
🚗 Vehicles: {self._metrics['total_vehicles_processed']:,}
🚨 Emergencies: {self._metrics['emergency_responses']}
⏱️ Avg Wait: {self._metrics['average_waiting_time']:.1f}s
📈 Throughput: {self._metrics['throughput']:.2f} veh/s
Status: {'🟢 Optimal' if self._metrics['throughput'] > 0.5 else '🟡 Normal' if self._metrics['throughput'] > 0.2 else '🔴 Congested'}
"""