import MetaTrader5 as mt5
import time
import threading
from utils import fetch_price_data, print_trade_log
from config_loader import Config
import pandas as pd

def start_trailing_stop(order_ticket: int, symbol: str, order_type: int, price_open: float):
    """Memulai trailing stop untuk posisi tertentu"""
    thread = threading.Thread(
        target=_trailing_worker,
        args=(order_ticket, symbol, order_type, price_open),
        daemon=True
    )
    thread.start()

def _trailing_worker(order_ticket, symbol, order_type, price_open):
    """Fungsi inti trailing stop (berjalan di thread terpisah)"""
    try:
        while True:
            time.sleep(150)  # Update setiap 2.5 menit

            if not mt5.initialize():
                print_trade_log("⛔ MT5 tidak terhubung, mencoba ulang...")
                continue

            # Ambil posisi berdasarkan ticket
            positions = mt5.positions_get(ticket=order_ticket)
            if not positions or len(positions) == 0:
                print_trade_log(f"ℹ️ Posisi dengan ticket {order_ticket} sudah ditutup")
                break

            position = positions[0]

            # Ambil data terbaru
            df = fetch_price_data(symbol, Config.TIMEFRAMES['M15'], bars=30)
            if df.empty or 'close' not in df.columns:
                print_trade_log("⚠️ Data historis kosong")
                continue

            df['atr'] = pd.Series(mt5.terminal_info()).astype(float)  # Placeholder jika ATR error
            df['atr'] = df['close'].rolling(14).std() * 1.5  # Fallback ATR manual

            atr = df['atr'].iloc[-1]
            tick = mt5.symbol_info_tick(symbol)

            if not tick:
                print_trade_log("❌ Gagal mendapatkan tick info")
                continue

            new_sl = None
            if order_type == mt5.ORDER_TYPE_BUY:
                new_sl = price_open + (atr * 1.2)
                if new_sl > position.sl + 0.5:
                    result = mt5.order_modify(
                        ticket=order_ticket,
                        price=0,
                        sl=new_sl,
                        tp=position.tp,
                        deviation=20
                    )
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print_trade_log(f"🔄 Trailing SL BUY diperbarui ke {new_sl:.2f}")
            elif order_type == mt5.ORDER_TYPE_SELL:
                new_sl = price_open - (atr * 1.2)
                if new_sl < position.sl - 0.5:
                    result = mt5.order_modify(
                        ticket=order_ticket,
                        price=0,
                        sl=new_sl,
                        tp=position.tp,
                        deviation=20
                    )
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print_trade_log(f"🔄 Trailing SL SELL diperbarui ke {new_sl:.2f}")

    except Exception as e:
        print_trade_log(f"❌ Error di trailing stop: {e}")
    finally:
        mt5.shutdown()
