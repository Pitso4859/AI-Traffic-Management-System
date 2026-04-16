"""
Natural Language Processing module for traffic chatbot.
Handles text preprocessing, intent recognition, and entity extraction.
"""

from __future__ import annotations

import re
import threading
from typing import Tuple, List, Dict, Any

import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag


class NLPProcessor:
    """Process natural language queries about traffic conditions."""

    _instance = None
    _lock = threading.Lock()

    INTENT_PATTERNS = {
        "greeting": [r"\b(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b"],
        "status": [
            r"\b(what'?s|status|current|now|right now)\b.*\b(traffic|signal|lights?)\b",
            r"\b(traffic|signal|lights?)\b.*\b(status|now|current)\b"
        ],
        "vehicle_count": [
            r"\b(how many|count|number of)\b.*\b(vehicles|cars|trucks?|buses?)\b",
            r"\b(density|congestion)\b"
        ],
        "emergency": [
            r"\b(emergency|ambulance|fire truck|police|override)\b",
            r"\b(is there|any)\b.*\b(emergency|ambulance)\b"
        ],
        "lane_specific": [
            r"\b(north|south|east|west)\b.*\b(traffic|vehicles|count|status)\b"
        ],
        "prediction": [
            r"\b(when|how long|predict|forecast|will)\b.*\b(clear|green|change|switch)\b",
            r"\b(prediction|forecast|expect|estimate)\b"
        ],
        "help": [r"\b(help|commands|what can you do|capabilities|features)\b"],
        "thanks": [r"\b(thanks|thank you|appreciate|good|great)\b"],
        "farewell": [r"\b(bye|goodbye|exit|quit|see you|farewell)\b"]
    }

    ENTITY_PATTERNS = {
        "lane": r"\b(north|south|east|west)\b",
        "time": r"\b(\d+)\s*(minutes?|mins?|seconds?|secs?|hours?)\b",
    }

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._stopwords = set(stopwords.words('english'))

    def preprocess(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s\?\!\.\,\']', ' ', text)
        return ' '.join(text.split())

    def detect_intent(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        best_intent = "unknown"
        best_score = 0.0

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    match_len = len(str(matches[0])) if matches else 0
                    text_len = len(text_lower)
                    confidence = min(0.95, (match_len / max(text_len, 1)) * 2 + 0.3)
                    if confidence > best_score:
                        best_score = confidence
                        best_intent = intent

        return best_intent, best_score

    def extract_entities(self, text: str) -> Dict[str, Any]:
        entities = {}
        text_lower = text.lower()

        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))

        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            proper_nouns = [word for word, tag in pos_tags if tag.startswith('NNP')]
            for pn in proper_nouns:
                if pn.lower() in ['north', 'south', 'east', 'west']:
                    if 'lane' not in entities:
                        entities['lane'] = []
                    entities['lane'].append(pn.lower())
        except Exception:
            pass

        return entities

    def generate_response_template(self, intent: str) -> str:
        templates = {
            "greeting": "Hello! I'm your Traffic Management Assistant. I can help you with traffic status, vehicle counts, emergency alerts, and predictions. What would you like to know?",
            "thanks": "You're welcome! Anything else I can help you with?",
            "farewell": "Goodbye! Stay safe on the roads. I'll be here if you need me.",
            "help": """I can help you with:
• Current traffic signal status
• Vehicle counts per lane
• Emergency vehicle presence
• Traffic predictions
• Lane-specific information

Try asking:
- "What's the traffic status?"
- "How many vehicles on north lane?"
- "Is there an emergency?"
- "When will the light change?" """,
            "unknown": "I'm not sure I understood. You can ask about traffic status, vehicle counts, emergencies, or type 'help' for more options."
        }
        return templates.get(intent, templates["unknown"])