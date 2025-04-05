import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

# Memuat file .env
load_dotenv("D:/INTRADAY XAU/XAU.env")

class Config:
    """Konfigurasi global bot trading"""
    # Simbol trading
    SYMBOL = "XAUUSD"
    
    # Timeframe yang digunakan
    TIMEFRAMES = {
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1
    }

    # Risiko per trading (% dari balance)
    RISK_PERCENT = 1.0

    # Panjang input model LSTM (jumlah candlestick)
    MODEL_INPUT_LENGTH = 60

    # API untuk filter berita
    NEWS_API_URL = "https://newsapi.org/v2/everything"
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")

    # Maksimum trading per hari
    MAX_DAILY_TRADES = 5

    # Cooldown antar trade (detik)
    TRADE_COOLDOWN = 900  # 15 menit

    # File model LSTM
    MODEL_PATH = "lstm_model.keras"

    # Path untuk menyimpan grafik training loss
    LOSS_PLOT_PATH = "training_loss.png"

    # Magic number untuk identifikasi order
    MAGIC_NUMBER = 2024
def load_config(config_path: str = "config.ini") -> Config:
    return Config(config_path)
