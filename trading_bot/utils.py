from datetime import datetime
import pytz
import pandas as pd
import MetaTrader5 as mt5
from config_loader import Config
import csv
import os
from datetime import datetime

def get_timezone():
    """Dapatkan timezone sesuai config"""
    return pytz.timezone(Config.TIMEZONE)

def utc_to_local(utc_dt):
    """Konversi waktu UTC ke lokal sesuai timezone"""
    local_tz = get_timezone()
    return utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)

def print_trade_log(message):
    """Log trading dengan timestamp"""
    now = datetime.now(get_timezone()).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{now}] {message}")

def fetch_price_data(symbol, timeframe, bars=1000):
    """Ambil data harga historis dari MT5"""
    if not mt5.initialize():
        mt5.initialize()
    utc_from = datetime.utcnow()
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, bars)
    df = pd.DataFrame(rates)
    if df.empty:
        return pd.DataFrame()
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

LOG_FILE = "trade_logs.csv"

def log_trade(symbol: str, order_type: str, lot: float, price: float, result: str):
    """
    Mencatat trade ke file CSV.
    """
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['timestamp', 'symbol', 'order_type', 'lot', 'price', 'result'])

        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            symbol,
            order_type,
            lot,
            price,
            result
        ])
