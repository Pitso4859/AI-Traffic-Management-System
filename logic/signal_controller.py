"""
Centralized traffic signal controller.

Collects detection results from all four lanes and decides which lane
receives GREEN using a score-based algorithm with emergency override.
"""

from __future__ import annotations

import threading
from time import monotonic
from dataclasses import dataclass
from typing import Optional

from utils.config import (
    EMERGENCY_DURATION,
    GREEN_DURATION,
    LANE_NAMES,
    WAITING_WEIGHT,
    YELLOW_DURATION,
)


@dataclass
class LaneState:
    """Mutable state for a single lane."""

    name: str
    vehicle_count: int = 0
    score: float = 0.0
    signal: str = "RED"
    waiting_cycles: int = 0
    waiting_since: float = 0.0
    emergency_detected: bool = False


class SignalController:
    """
    Decision engine for the 4-way intersection.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lanes: dict[str, LaneState] = {
            name: LaneState(name=name) for name in LANE_NAMES
        }
        self.active_green: str = LANE_NAMES[0]
        self.emergency_active: bool = False
        self.emergency_lane: Optional[str] = None

        self._green_start: float = monotonic()
        self._yellow_active: bool = False
        self._yellow_start: float = 0.0
        self._emergency_start: float = 0.0

        now = monotonic()
        for ls in self.lanes.values():
            ls.waiting_since = now

        self.lanes[self.active_green].signal = "GREEN"
        self.lanes[self.active_green].waiting_since = 0.0

    def update(self, counts: dict[str, int], emergencies: dict[str, bool]) -> None:
        """Feed new detection results and advance the signal state machine."""
        with self.lock:
            for name in LANE_NAMES:
                ls = self.lanes[name]
                ls.vehicle_count = counts.get(name, 0)
                ls.emergency_detected = emergencies.get(name, False)

            if self._check_emergency():
                return

            if self._yellow_active:
                if monotonic() - self._yellow_start >= YELLOW_DURATION:
                    self._finish_yellow()
                return

            if monotonic() - self._green_start >= GREEN_DURATION:
                self._begin_switch()

            self._update_scores()

    def get_status(self) -> dict:
        """Build the status dict consumed by the /status endpoint."""
        with self.lock:
            now = monotonic()
            lanes_status = {}
            for name in LANE_NAMES:
                ls = self.lanes[name]
                waiting_seconds = self._get_waiting_seconds(ls, now)
                lanes_status[name] = {
                    "count": ls.vehicle_count,
                    "score": round(ls.score, 1),
                    "signal": ls.signal,
                    "waiting": waiting_seconds,
                    "waiting_seconds": waiting_seconds,
                    "waiting_turns": ls.waiting_cycles,
                }
            return {
                "lanes": lanes_status,
                "emergency_active": self.emergency_active,
                "emergency_lane": self.emergency_lane,
                "active_green": self.active_green,
                "phase": self._current_phase(),
                "phase_remaining_seconds": self._phase_remaining_seconds(now),
            }

    def _update_scores(self) -> None:
        for ls in self.lanes.values():
            ls.score = ls.vehicle_count + (ls.waiting_cycles * WAITING_WEIGHT)

    def _check_emergency(self) -> bool:
        if self.emergency_active:
            if monotonic() - self._emergency_start >= EMERGENCY_DURATION:
                self.emergency_active = False
                self.emergency_lane = None
                self._set_green(self.active_green)
                self._green_start = monotonic()
                return False
            return True

        emergency_lanes = [name for name in LANE_NAMES if self.lanes[name].emergency_detected]
        if not emergency_lanes:
            return False

        chosen = max(emergency_lanes, key=lambda n: self.lanes[n].vehicle_count)
        self.emergency_active = True
        self.emergency_lane = chosen
        self._emergency_start = monotonic()
        self._yellow_active = False
        self._set_green(chosen)
        self.active_green = chosen
        return True

    def _begin_switch(self) -> None:
        self.lanes[self.active_green].signal = "YELLOW"
        self._yellow_active = True
        self._yellow_start = monotonic()

    def _finish_yellow(self) -> None:
        self._yellow_active = False

        for ls in self.lanes.values():
            if ls.name != self.active_green:
                ls.waiting_cycles += 1

        self.lanes[self.active_green].waiting_cycles = 0
        self._update_scores()

        candidates = [ls for ls in self.lanes.values() if ls.name != self.active_green]
        if candidates:
            best = max(candidates, key=lambda ls: ls.score)
        else:
            best = self.lanes[self.active_green]

        self.active_green = best.name
        self._set_green(best.name)
        self._green_start = monotonic()

    def _set_green(self, lane_name: str) -> None:
        now = monotonic()
        for ls in self.lanes.values():
            if ls.name == lane_name:
                ls.signal = "GREEN"
                ls.waiting_since = 0.0
            else:
                if ls.signal != "RED":
                    ls.waiting_since = now
                ls.signal = "RED"

    def _get_waiting_seconds(self, lane: LaneState, now: float) -> int:
        if lane.signal != "RED" or lane.waiting_since <= 0:
            return 0
        return max(0, int(now - lane.waiting_since))

    def _current_phase(self) -> str:
        if self.emergency_active:
            return "EMERGENCY"
        if self._yellow_active:
            return "YELLOW"
        return "GREEN"

    def _phase_remaining_seconds(self, now: float) -> int:
        if self.emergency_active:
            remaining = EMERGENCY_DURATION - (now - self._emergency_start)
        elif self._yellow_active:
            remaining = YELLOW_DURATION - (now - self._yellow_start)
        else:
            remaining = GREEN_DURATION - (now - self._green_start)
        return max(0, int(remaining))