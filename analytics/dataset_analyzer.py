"""
Dataset analyzer for traffic pattern learning.
Uses historical data to predict traffic conditions.
"""

from __future__ import annotations

import os
import threading
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


class DatasetAnalyzer:
    """Analyzes traffic dataset to provide intelligent predictions."""

    _instance = None
    _lock = threading.Lock()

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
        self._data = None
        self._models = {}
        self._encoders = {}
        self._load_data()
        self._train_models()

    def _load_data(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(base_dir, "data", "smart_traffic_management_dataset.csv")

        if not os.path.exists(dataset_path):
            self._create_synthetic_data()
            return

        try:
            self._data = pd.read_csv(dataset_path)
            self._preprocess_data()
        except Exception as e:
            print(f"Error loading dataset: {e} - dataset_analyzer.py:54")
            self._create_synthetic_data()

    def _create_synthetic_data(self):
        np.random.seed(42)
        n_samples = 5000
        hours = np.random.randint(0, 24, n_samples)
        weather = np.random.choice(['Sunny', 'Cloudy', 'Rainy', 'Windy', 'Foggy'], n_samples)
        accidents = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])

        base_volume = 400 + 200 * np.sin(np.pi * (hours - 8) / 12)
        rush = (hours >= 7) & (hours <= 9) | (hours >= 17) & (hours <= 19)
        base_volume[rush] += 150
        base_volume[hours < 5] = 150

        weather_mult = {'Sunny': 1.0, 'Cloudy': 0.9, 'Rainy': 0.7, 'Windy': 0.85, 'Foggy': 0.6}
        volume = base_volume + np.random.normal(0, 50, n_samples)
        for w, mult in weather_mult.items():
            volume[np.array(weather) == w] *= mult

        self._data = pd.DataFrame({
            'hour': hours,
            'weather_condition': weather,
            'accident_reported': accidents,
            'traffic_volume': np.maximum(50, volume).astype(int),
        })
        self._preprocess_data()

    def _preprocess_data(self):
        if 'weather_condition' in self._data.columns:
            self._encoders['weather'] = LabelEncoder()
            self._data['weather_encoded'] = self._encoders['weather'].fit_transform(
                self._data['weather_condition'].fillna('Unknown')
            )

    def _train_models(self):
        if self._data is None or len(self._data) < 100:
            return

        feature_cols = ['hour', 'weather_encoded', 'accident_reported']
        available = [c for c in feature_cols if c in self._data.columns]

        if len(available) >= 2:
            X = self._data[available].fillna(0)
            y = self._data['traffic_volume'].fillna(0)
            try:
                self._models['volume'] = RandomForestRegressor(n_estimators=50, random_state=42)
                self._models['volume'].fit(X, y)
            except Exception:
                pass

    def predict_traffic_volume(self, hour: int, location_id: int = 1, weather: str = "Sunny") -> Dict[str, Any]:
        if 'volume' not in self._models or self._models['volume'] is None:
            base = 400
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                base = 600
            if hour < 5 or hour > 22:
                base = 200
            weather_factors = {'Sunny': 1.0, 'Cloudy': 0.9, 'Rainy': 0.7, 'Windy': 0.85, 'Foggy': 0.6}
            base *= weather_factors.get(weather, 0.8)
            predicted = int(base)
            confidence = 0.6
        else:
            try:
                weather_enc = self._encoders.get('weather', LabelEncoder())
                if hasattr(weather_enc, 'classes_') and weather in weather_enc.classes_:
                    w_val = weather_enc.transform([weather])[0]
                else:
                    w_val = 0
                pred = self._models['volume'].predict([[hour, w_val, 0]])[0]
                predicted = int(max(0, pred))
                confidence = 0.8
            except Exception:
                predicted = 400
                confidence = 0.5

        if predicted > 700:
            congestion = "high"
        elif predicted > 350:
            congestion = "medium"
        else:
            congestion = "low"

        return {'predicted_volume': predicted, 'confidence': confidence, 'congestion_level': congestion}

    def get_weather_impact(self, weather: str) -> Dict[str, float]:
        impacts = {
            'Sunny': {'volume_factor': 1.0, 'speed_factor': 1.0},
            'Cloudy': {'volume_factor': 0.95, 'speed_factor': 0.98},
            'Rainy': {'volume_factor': 0.85, 'speed_factor': 0.85},
            'Windy': {'volume_factor': 0.9, 'speed_factor': 0.92},
            'Foggy': {'volume_factor': 0.7, 'speed_factor': 0.75}
        }
        return impacts.get(weather, {'volume_factor': 0.9, 'speed_factor': 0.9})

    def get_traffic_summary(self, hour: int, location_id: int = None) -> Dict[str, Any]:
        is_rush = (7 <= hour <= 9) or (17 <= hour <= 19)
        is_night = hour < 5 or hour > 22
        return {
            'hour': hour,
            'typical_volume': 600 if is_rush else (200 if is_night else 400),
            'congestion': 'high' if is_rush else ('low' if is_night else 'medium'),
            'peak_status': 'Rush Hour' if is_rush else ('Night' if is_night else 'Normal')
        }


_dataset_analyzer = None

def get_dataset_analyzer() -> DatasetAnalyzer:
    global _dataset_analyzer
    if _dataset_analyzer is None:
        _dataset_analyzer = DatasetAnalyzer()
    return _dataset_analyzer