"""
Intent handler for executing detected intents and generating responses.
"""

from __future__ import annotations

import random
from typing import Dict, Any


class IntentHandler:
    """Handles intent execution with context awareness."""

    def __init__(self, controller, detectors):
        self.controller = controller
        self.detectors = detectors
        self.conversation_history = []

    def execute(self, intent: str, entities: Dict, query: str) -> str:
        if intent == "greeting":
            return self._handle_greeting()
        elif intent == "status":
            return self._handle_status()
        elif intent == "vehicle_count":
            return self._handle_vehicle_count(entities)
        elif intent == "emergency":
            return self._handle_emergency()
        elif intent == "lane_specific":
            return self._handle_lane_query(entities)
        elif intent == "prediction":
            return self._handle_prediction()
        elif intent == "thanks":
            return self._handle_thanks()
        elif intent == "farewell":
            return self._handle_farewell()
        elif intent == "help":
            return self._handle_help()
        else:
            return self._handle_unknown()

    def _handle_greeting(self) -> str:
        greetings = [
            "Hello! I'm your AI traffic assistant. How can I help you with traffic management today?",
            "Hi there! Ready to help you monitor and manage traffic flow. What would you like to know?",
            "Welcome! I can provide real-time traffic updates, vehicle counts, and signal status. Ask me anything!"
        ]
        return random.choice(greetings)

    def _handle_status(self) -> str:
        if not self.controller:
            return "Traffic controller not available."
        status = self.controller.get_status()
        lanes = status.get("lanes", {})
        response_parts = []
        for lane_name, lane_data in lanes.items():
            signal = lane_data.get("signal", "RED")
            count = lane_data.get("count", 0)
            response_parts.append(f"{lane_name.capitalize()}: {signal}, {count} vehicles")
        emergency = status.get("emergency_active", False)
        if emergency:
            emergency_lane = status.get("emergency_lane", "unknown")
            response_parts.insert(0, f"⚠️ EMERGENCY on {emergency_lane.capitalize()} lane! ⚠️")
        return " | ".join(response_parts)

    def _handle_vehicle_count(self, entities: Dict) -> str:
        if not self.controller:
            return "Traffic data not available."
        status = self.controller.get_status()
        lanes = status.get("lanes", {})

        if entities.get("lane"):
            lane = entities["lane"][0].lower()
            if lane in lanes:
                count = lanes[lane].get("count", 0)
                return f"{count} vehicle{'s' if count != 1 else ''} on {lane.capitalize()} lane."
            return f"No data for {lane} lane."

        counts = [f"{lane.capitalize()}: {data.get('count', 0)}" for lane, data in lanes.items()]
        return "Vehicle counts — " + ", ".join(counts)

    def _handle_emergency(self) -> str:
        if not self.controller:
            return "Emergency detection unavailable."
        status = self.controller.get_status()
        if status.get("emergency_active", False):
            lane = status.get("emergency_lane", "unknown")
            return f"⚠️ EMERGENCY on {lane.capitalize()} lane! Signal overridden to GREEN. Please give way! ⚠️"
        return "No emergency vehicles detected at this time."

    def _handle_lane_query(self, entities: Dict) -> str:
        if not self.controller:
            return "Traffic data unavailable."
        status = self.controller.get_status()
        lanes = status.get("lanes", {})

        if not entities.get("lane"):
            return "Which lane? (north, south, east, or west)"

        lane = entities["lane"][0].lower()
        if lane not in lanes:
            return f"Lane '{lane}' not found."

        data = lanes[lane]
        return f"{lane.capitalize()} Lane: {data.get('signal', 'RED')} signal, {data.get('count', 0)} vehicles, waiting {data.get('waiting_seconds', 0)}s"

    def _handle_prediction(self) -> str:
        if not self.controller:
            return "Prediction service unavailable."
        status = self.controller.get_status()
        phase = status.get("phase", "GREEN")
        remaining = status.get("phase_remaining_seconds", 0)

        if phase == "EMERGENCY":
            return f"Emergency mode active. Normal operation in ~{remaining}s."
        elif phase == "YELLOW":
            return f"Signal changing in {remaining}s."
        else:
            return f"Green phase ends in {remaining}s. Next lane determined by traffic density."

    def _handle_thanks(self) -> str:
        return random.choice(["You're welcome!", "My pleasure!", "Glad I could help!"])

    def _handle_farewell(self) -> str:
        return random.choice(["Goodbye! Drive safely!", "See you later!", "Farewell!"])

    def _handle_help(self) -> str:
        return """🤖 **Commands** 🤖
• "What's the traffic status?"
• "How many vehicles on north lane?"
• "Is there an emergency?"
• "When will the light change?"
• "Help" - This menu
• "Thanks" / "Bye" """

    def _handle_unknown(self) -> str:
        return "I didn't understand. Try 'help' for available commands."