# trading_executor.py

import logging
from mt5_connector import send_order
from trading_strategy import Signal  # Pastikan Signal ada di trading_strategy.py

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TradeExecutor:
    def __init__(self):
        pass

    def execute_trade(self, signal: Signal):
        """
        Eksekusi order berdasarkan sinyal.
        """
        if not signal or not signal.action:
            logger.warning("No valid signal received.")
            return None

        symbol = signal.symbol
        volume = signal.volume
        order_type = signal.action

        try:
            result = send_order(symbol=symbol, volume=volume, order_type=order_type)
            logger.info(f"Order sent: {order_type} {symbol} {volume}")
            return result
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return None

# Fungsi agar bisa langsung diimport ke main.py
def execute_trade(signal: Signal):
    executor = TradeExecutor()
    return executor.execute_trade(signal)
