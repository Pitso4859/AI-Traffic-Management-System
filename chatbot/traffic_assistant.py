"""
Main chatbot interface combining NLP and intent handling.
"""

from __future__ import annotations

from chatbot.nlp_processor import NLPProcessor
from chatbot.intent_handler import IntentHandler


class TrafficAssistant:
    """Main chatbot interface."""

    def __init__(self, controller, detectors):
        self.nlp = NLPProcessor()
        self.intent_handler = IntentHandler(controller, detectors)

    def process_query(self, user_input: str) -> str:
        if not user_input or not user_input.strip():
            return "Please type a question. Type 'help' for commands."

        cleaned = self.nlp.preprocess(user_input)
        intent, confidence = self.nlp.detect_intent(cleaned)
        entities = self.nlp.extract_entities(cleaned)

        response = self.intent_handler.execute(intent, entities, cleaned)
        return response