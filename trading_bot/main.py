from mt5_connector import mt5_connect
from model_handler import ModelHandler
from news_filter import NewsFilter
from trading_strategy import Strategy
from trading_executor import TradeExecutor
from config_loader import load_config
import MetaTrader5 as mt5
import time
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv("D:/trading_bot/.env")

account = os.getenv("MT5_ACCOUNT")
password = os.getenv("MT5_PASSWORD")

def run_bot():
    print("\n" + "="*50)
    print(f"🤖 AI GOLD TRADING BOT (XAU/USD)")
    print(f"📅 Versi: {datetime.now().strftime('%Y-%m-%d')}")
    print("="*50)

    # Inisialisasi koneksi ke MT5
    mt5_connect()

    # Inisialisasi modul utama
    model_handler = ModelHandler()
    news_filter = NewsFilter()
    strategy = Strategy(model_handler, news_filter)
    trading_executor = TradingExecutor()

    try:
        while True:
            signal = strategy.generate_signal()
            if signal:
                print(f"\n🎯 Signal {signal} terdeteksi!")
                trading_executor.execute_trade(signal)
            time.sleep(300)  # 5 menit jeda antar evaluasi sinyal

    except KeyboardInterrupt:
        print("\n🛑 Bot dihentikan oleh pengguna")

    finally:
        mt5.shutdown()
        print("\n🔌 Koneksi MT5 ditutup")
def main():
    # Inisialisasi koneksi ke MT5
    if not mt5_connect():
        print("Gagal koneksi ke MT5")
        return

    # Inisialisasi handler model dan strategi
    model_handler = ModelHandler()
    strategy = Strategy()
    executor = TradeExecutor()

    # Ambil simbol yang digunakan
    symbol = "EURUSD"

    # Dapatkan data dari model
    data = model_handler.get_latest_data(symbol)

    # Generate sinyal trading
    signal = strategy.generate_signal(data)

    # Eksekusi trade jika ada sinyal
    if signal:
        executor.execute_trade(signal)
    else:
        print("Tidak ada sinyal trading.")

if __name__ == "__main__":
    main()

