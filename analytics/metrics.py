"""
Performance metrics and analytics for the traffic management system.

Tracks system performance including:
- Vehicle throughput
- Emergency response times
- Signal change frequency
- Waiting time statistics
- System uptime
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Any, Optional
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PerformanceRecord:
    """Single performance record for historical tracking."""
    
    timestamp: float
    metric_name: str
    value: float
    lane_name: Optional[str] = None


class TrafficMetrics:
    """
    Comprehensive metrics tracking for the traffic system.
    
    Tracks real-time and historical performance metrics.
    Thread-safe for concurrent access from multiple threads.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics tracker.
        
        Args:
            max_history: Maximum number of historical records to keep
        """
        self._lock = threading.RLock()
        self._max_history = max_history
        
        # Current metrics
        self._metrics: Dict[str, Any] = {
            # Vehicle counts
            "total_vehicles_processed": 0,
            "vehicles_per_lane": {"north": 0, "south": 0, "east": 0, "west": 0},
            "peak_vehicle_count": 0,
            "peak_vehicle_lane": None,
            
            # Emergency metrics
            "emergency_responses": 0,
            "emergency_lanes": [],
            "last_emergency_time": None,
            "avg_emergency_response_time": 0,
            
            # Signal metrics
            "signal_changes": 0,
            "green_cycles": {"north": 0, "south": 0, "east": 0, "west": 0},
            "yellow_cycles": 0,
            
            # Timing metrics
            "average_waiting_time": 0,
            "max_waiting_time": 0,
            "total_waiting_time": 0,
            "waiting_samples": 0,
            
            # Throughput metrics
            "throughput": 0,
            "throughput_history": deque(maxlen=60),  # Last 60 seconds
            "peak_throughput": 0,
            
            # Uptime
            "start_time": time.time(),
            "uptime_seconds": 0,
            "last_update": time.time(),
            
            # Detection metrics
            "total_detection_cycles": 0,
            "avg_detection_time_ms": 0,
            "detection_errors": 0,
            
            # Queue metrics
            "max_queue_length": 0,
            "avg_queue_length": 0,
        }
        
        # Historical records
        self._history: deque = deque(maxlen=max_history)
        
        # Temporary accumulators
        self._temp_detection_times: deque = deque(maxlen=100)
        self._temp_waiting_times: deque = deque(maxlen=100)
    
    # ──────────────────────────────────────────────────────────────
    # Vehicle Metrics
    # ──────────────────────────────────────────────────────────────
    
    def record_vehicles(self, count: int, lane_name: str = None) -> None:
        """
        Record number of vehicles detected.
        
        Args:
            count: Number of vehicles detected
            lane_name: Optional lane name for per-lane tracking
        """
        with self._lock:
            self._metrics["total_vehicles_processed"] += count
            
            if lane_name and lane_name in self._metrics["vehicles_per_lane"]:
                self._metrics["vehicles_per_lane"][lane_name] += count
            
            if count > self._metrics["peak_vehicle_count"]:
                self._metrics["peak_vehicle_count"] = count
                self._metrics["peak_vehicle_lane"] = lane_name
            
            self._add_history("vehicles_detected", count, lane_name)
    
    def get_vehicle_stats(self) -> Dict[str, Any]:
        """Get vehicle-related statistics."""
        with self._lock:
            return {
                "total_vehicles": self._metrics["total_vehicles_processed"],
                "per_lane": self._metrics["vehicles_per_lane"].copy(),
                "peak_count": self._metrics["peak_vehicle_count"],
                "peak_lane": self._metrics["peak_vehicle_lane"],
            }
    
    # ──────────────────────────────────────────────────────────────
    # Emergency Metrics
    # ──────────────────────────────────────────────────────────────
    
    def record_emergency(self, lane_name: str = None) -> None:
        """
        Record an emergency detection event.
        
        Args:
            lane_name: Lane where emergency was detected
        """
        with self._lock:
            current_time = time.time()
            
            self._metrics["emergency_responses"] += 1
            
            if lane_name:
                self._metrics["emergency_lanes"].append(lane_name)
                # Keep only last 100 emergencies
                if len(self._metrics["emergency_lanes"]) > 100:
                    self._metrics["emergency_lanes"] = self._metrics["emergency_lanes"][-100:]
            
            if self._metrics["last_emergency_time"]:
                response_time = current_time - self._metrics["last_emergency_time"]
                total = self._metrics["avg_emergency_response_time"] * (self._metrics["emergency_responses"] - 1)
                self._metrics["avg_emergency_response_time"] = (total + response_time) / self._metrics["emergency_responses"]
            
            self._metrics["last_emergency_time"] = current_time
            self._add_history("emergency", 1, lane_name)
    
    def get_emergency_stats(self) -> Dict[str, Any]:
        """Get emergency-related statistics."""
        with self._lock:
            return {
                "total_emergencies": self._metrics["emergency_responses"],
                "emergency_lanes": self._metrics["emergency_lanes"][-10:],  # Last 10
                "avg_response_time": round(self._metrics["avg_emergency_response_time"], 2),
                "last_emergency": self._metrics["last_emergency_time"],
            }
    
    # ──────────────────────────────────────────────────────────────
    # Signal Metrics
    # ──────────────────────────────────────────────────────────────
    
    def record_signal_change(self, new_green_lane: str = None) -> None:
        """
        Record a signal state change.
        
        Args:
            new_green_lane: Lane that received GREEN signal
        """
        with self._lock:
            self._metrics["signal_changes"] += 1
            
            if new_green_lane and new_green_lane in self._metrics["green_cycles"]:
                self._metrics["green_cycles"][new_green_lane] += 1
            
            self._add_history("signal_change", 1, new_green_lane)
    
    def record_yellow_phase(self) -> None:
        """Record a YELLOW phase activation."""
        with self._lock:
            self._metrics["yellow_cycles"] += 1
            self._add_history("yellow_phase", 1)
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """Get signal-related statistics."""
        with self._lock:
            total_greens = sum(self._metrics["green_cycles"].values())
            return {
                "total_signal_changes": self._metrics["signal_changes"],
                "green_cycles": self._metrics["green_cycles"].copy(),
                "yellow_cycles": self._metrics["yellow_cycles"],
                "green_distribution": {
                    lane: round(count / total_greens * 100, 1) if total_greens > 0 else 0
                    for lane, count in self._metrics["green_cycles"].items()
                }
            }
    
    # ──────────────────────────────────────────────────────────────
    # Waiting Time Metrics
    # ──────────────────────────────────────────────────────────────
    
    def record_waiting_time(self, seconds: int, lane_name: str = None) -> None:
        """
        Record waiting time for a lane.
        
        Args:
            seconds: Waiting time in seconds
            lane_name: Optional lane name for per-lane tracking
        """
        with self._lock:
            self._temp_waiting_times.append(seconds)
            
            # Update running average
            total = self._metrics["average_waiting_time"] * self._metrics["waiting_samples"]
            self._metrics["waiting_samples"] += 1
            self._metrics["average_waiting_time"] = (total + seconds) / self._metrics["waiting_samples"]
            
            # Update max
            if seconds > self._metrics["max_waiting_time"]:
                self._metrics["max_waiting_time"] = seconds
            
            # Update total
            self._metrics["total_waiting_time"] += seconds
            
            self._add_history("waiting_time", seconds, lane_name)
    
    def get_waiting_stats(self) -> Dict[str, Any]:
        """Get waiting time statistics."""
        with self._lock:
            return {
                "average_waiting_seconds": round(self._metrics["average_waiting_time"], 1),
                "max_waiting_seconds": self._metrics["max_waiting_time"],
                "total_waiting_seconds": self._metrics["total_waiting_time"],
                "waiting_samples": self._metrics["waiting_samples"],
                "recent_waiting_times": list(self._temp_waiting_times)[-10:],
            }
    
    # ──────────────────────────────────────────────────────────────
    # Throughput Metrics
    # ──────────────────────────────────────────────────────────────
    
    def update_throughput(self, vehicles_in_interval: int, interval_seconds: float) -> None:
        """
        Update throughput metrics.
        
        Args:
            vehicles_in_interval: Number of vehicles detected in the interval
            interval_seconds: Length of the interval in seconds
        """
        with self._lock:
            if interval_seconds > 0:
                current_throughput = vehicles_in_interval / interval_seconds
                self._metrics["throughput_history"].append(current_throughput)
                
                # Calculate average throughput over last 60 seconds
                if len(self._metrics["throughput_history"]) > 0:
                    self._metrics["throughput"] = sum(self._metrics["throughput_history"]) / len(self._metrics["throughput_history"])
                
                # Update peak throughput
                if current_throughput > self._metrics["peak_throughput"]:
                    self._metrics["peak_throughput"] = current_throughput
    
    def get_throughput_stats(self) -> Dict[str, Any]:
        """Get throughput statistics."""
        with self._lock:
            recent = list(self._metrics["throughput_history"])[-10:] if self._metrics["throughput_history"] else []
            return {
                "current_throughput": round(self._metrics["throughput"], 2),
                "peak_throughput": round(self._metrics["peak_throughput"], 2),
                "recent_throughput": [round(t, 2) for t in recent],
                "trend": self._calculate_trend(list(self._metrics["throughput_history"])[-5:]) if self._metrics["throughput_history"] else "stable"
            }
    
    def _calculate_trend(self, values: list) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return "stable"
        if values[-1] > values[0] * 1.1:
            return "increasing"
        elif values[-1] < values[0] * 0.9:
            return "decreasing"
        return "stable"
    
    # ──────────────────────────────────────────────────────────────
    # Uptime Metrics
    # ──────────────────────────────────────────────────────────────
    
    def update_uptime(self) -> None:
        """Update system uptime."""
        with self._lock:
            self._metrics["uptime_seconds"] = time.time() - self._metrics["start_time"]
    
    def get_uptime_stats(self) -> Dict[str, Any]:
        """Get uptime statistics."""
        with self._lock:
            uptime_sec = time.time() - self._metrics["start_time"]
            return {
                "uptime_seconds": int(uptime_sec),
                "uptime_minutes": int(uptime_sec / 60),
                "uptime_hours": round(uptime_sec / 3600, 1),
                "start_time": self._metrics["start_time"],
            }
    
    # ──────────────────────────────────────────────────────────────
    # Detection Metrics
    # ──────────────────────────────────────────────────────────────
    
    def record_detection_cycle(self, detection_time_ms: float, success: bool = True) -> None:
        """
        Record a detection cycle.
        
        Args:
            detection_time_ms: Time taken for detection in milliseconds
            success: Whether detection was successful
        """
        with self._lock:
            self._metrics["total_detection_cycles"] += 1
            self._temp_detection_times.append(detection_time_ms)
            
            if not success:
                self._metrics["detection_errors"] += 1
            
            # Update average detection time
            if len(self._temp_detection_times) > 0:
                self._metrics["avg_detection_time_ms"] = sum(self._temp_detection_times) / len(self._temp_detection_times)
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        with self._lock:
            error_rate = 0
            if self._metrics["total_detection_cycles"] > 0:
                error_rate = self._metrics["detection_errors"] / self._metrics["total_detection_cycles"] * 100
            
            return {
                "total_cycles": self._metrics["total_detection_cycles"],
                "avg_time_ms": round(self._metrics["avg_detection_time_ms"], 2),
                "errors": self._metrics["detection_errors"],
                "error_rate_percent": round(error_rate, 2),
            }
    
    # ──────────────────────────────────────────────────────────────
    # Queue Metrics
    # ──────────────────────────────────────────────────────────────
    
    def update_queue_stats(self, queue_lengths: Dict[str, int]) -> None:
        """
        Update queue length statistics.
        
        Args:
            queue_lengths: Dictionary of lane names to queue lengths
        """
        with self._lock:
            avg_length = sum(queue_lengths.values()) / len(queue_lengths) if queue_lengths else 0
            max_length = max(queue_lengths.values()) if queue_lengths else 0
            
            # Update running average
            if self._metrics["avg_queue_length"] == 0:
                self._metrics["avg_queue_length"] = avg_length
            else:
                self._metrics["avg_queue_length"] = (self._metrics["avg_queue_length"] + avg_length) / 2
            
            if max_length > self._metrics["max_queue_length"]:
                self._metrics["max_queue_length"] = max_length
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                "max_queue_length": self._metrics["max_queue_length"],
                "average_queue_length": round(self._metrics["avg_queue_length"], 1),
            }
    
    # ──────────────────────────────────────────────────────────────
    # Helper Methods
    # ──────────────────────────────────────────────────────────────
    
    def _add_history(self, metric_name: str, value: float, lane_name: str = None) -> None:
        """Add a record to history."""
        self._history.append(PerformanceRecord(
            timestamp=time.time(),
            metric_name=metric_name,
            value=value,
            lane_name=lane_name
        ))
    
    def get_history(self, metric_name: str = None, limit: int = 100) -> list:
        """
        Get historical records.
        
        Args:
            metric_name: Optional filter by metric name
            limit: Maximum number of records to return
            
        Returns:
            List of historical records
        """
        with self._lock:
            records = list(self._history)[-limit:]
            if metric_name:
                records = [r for r in records if r.metric_name == metric_name]
            return records
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics in a single dictionary."""
        with self._lock:
            self.update_uptime()
            
            return {
                "vehicles": self.get_vehicle_stats(),
                "emergencies": self.get_emergency_stats(),
                "signals": self.get_signal_stats(),
                "waiting": self.get_waiting_stats(),
                "throughput": self.get_throughput_stats(),
                "uptime": self.get_uptime_stats(),
                "detection": self.get_detection_stats(),
                "queues": self.get_queue_stats(),
            }
    
    def get_performance_report(self) -> str:
        """
        Generate a human-readable performance report.
        
        Returns:
            Formatted report string
        """
        with self._lock:
            self.update_uptime()
            
            # Determine system status
            throughput = self._metrics["throughput"]
            if throughput > 0.8:
                status = "🟢 OPTIMAL"
            elif throughput > 0.4:
                status = "🟡 NORMAL"
            elif throughput > 0.1:
                status = "🟠 CONGESTED"
            else:
                status = "🔴 CRITICAL"
            
            report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    SYSTEM PERFORMANCE REPORT                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📊 VEHICLE STATISTICS                                           ║
║     Total Vehicles Processed: {self._metrics['total_vehicles_processed']:,}     ║
║     Peak Vehicle Count: {self._metrics['peak_vehicle_count']} ({self._metrics['peak_vehicle_lane'] or 'N/A'})     ║
║                                                                  ║
║  🚨 EMERGENCY STATISTICS                                         ║
║     Emergency Responses: {self._metrics['emergency_responses']}                      ║
║     Avg Response Time: {self._metrics['avg_emergency_response_time']:.1f}s                       ║
║                                                                  ║
║  🔄 SIGNAL STATISTICS                                            ║
║     Signal Changes: {self._metrics['signal_changes']}                          ║
║     Yellow Phases: {self._metrics['yellow_cycles']}                            ║
║     Green Distribution:                                           ║
║       North: {self._metrics['green_cycles']['north']} | South: {self._metrics['green_cycles']['south']} | East: {self._metrics['green_cycles']['east']} | West: {self._metrics['green_cycles']['west']}     ║
║                                                                  ║
║  ⏱️ WAITING TIME STATISTICS                                      ║
║     Average Wait: {self._metrics['average_waiting_time']:.1f}s                           ║
║     Maximum Wait: {self._metrics['max_waiting_time']}s                            ║
║                                                                  ║
║  📈 THROUGHPUT STATISTICS                                        ║
║     Current Throughput: {self._metrics['throughput']:.2f} veh/s                      ║
║     Peak Throughput: {self._metrics['peak_throughput']:.2f} veh/s                       ║
║                                                                  ║
║  ⏰ UPTIME                                                       ║
║     Uptime: {int(self._metrics['uptime_seconds'] / 3600)}h {int((self._metrics['uptime_seconds'] % 3600) / 60)}m {int(self._metrics['uptime_seconds'] % 60)}s                    ║
║                                                                  ║
║  🎯 DETECTION PERFORMANCE                                        ║
║     Avg Detection Time: {self._metrics['avg_detection_time_ms']:.1f}ms                       ║
║     Detection Error Rate: {self._calculate_error_rate()}%                      ║
║                                                                  ║
║  📊 SYSTEM STATUS: {status:<45} ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
            return report
    
    def _calculate_error_rate(self) -> float:
        """Calculate detection error rate percentage."""
        if self._metrics["total_detection_cycles"] == 0:
            return 0
        return round(self._metrics["detection_errors"] / self._metrics["total_detection_cycles"] * 100, 1)
    
    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self.__init__(self._max_history)
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all metrics as a JSON-serializable dictionary.
        
        Returns:
            Dictionary with all metrics
        """
        with self._lock:
            self.update_uptime()
            return {
                "timestamp": time.time(),
                "vehicles": self.get_vehicle_stats(),
                "emergencies": self.get_emergency_stats(),
                "signals": self.get_signal_stats(),
                "waiting": self.get_waiting_stats(),
                "throughput": self.get_throughput_stats(),
                "uptime": self.get_uptime_stats(),
                "detection": self.get_detection_stats(),
                "queues": self.get_queue_stats(),
            }


# Singleton instance for global access
_metrics_instance: Optional[TrafficMetrics] = None


def get_metrics() -> TrafficMetrics:
    """
    Get the singleton TrafficMetrics instance.
    
    Returns:
        Global TrafficMetrics instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = TrafficMetrics()
    return _metrics_instance


# Convenience functions for quick metric recording
def record_vehicles(count: int, lane_name: str = None) -> None:
    """Quick function to record vehicle count."""
    get_metrics().record_vehicles(count, lane_name)


def record_emergency(lane_name: str = None) -> None:
    """Quick function to record emergency."""
    get_metrics().record_emergency(lane_name)


def record_signal_change(new_green_lane: str = None) -> None:
    """Quick function to record signal change."""
    get_metrics().record_signal_change(new_green_lane)


def record_waiting_time(seconds: int, lane_name: str = None) -> None:
    """Quick function to record waiting time."""
    get_metrics().record_waiting_time(seconds, lane_name)


def get_performance_report() -> str:
    """Quick function to get performance report."""
    return get_metrics().get_performance_report()