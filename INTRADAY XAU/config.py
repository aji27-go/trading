import json
import os

class Config:

    SYMBOL = None
    prediction_threshold = None
    adx_threshold = None
    risk_percent = None
    atr_multiplier = None

    @classmethod
    def load_config(cls, file_path="config.json"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        cls.SYMBOL = data.get("SYMBOL", "BTC")  # Default fallback
        cls.prediction_threshold = data.get("prediction_threshold", 0.02)
        cls.adx_threshold = data.get("adx_threshold", 25)
        cls.risk_percent = data.get("risk_percent", 1.0)
        cls.atr_multiplier = data.get("atr_multiplier", 1.5)
