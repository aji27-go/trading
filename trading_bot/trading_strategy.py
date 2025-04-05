# trading_strategy.py

from dataclasses import dataclass

@dataclass
class Signal:
    symbol: str
    action: str  # 'buy' atau 'sell'
    volume: float

class Strategy:
    def __init__(self):
        pass

    def generate_signal(self, data):
        """
        Contoh strategi sederhana: buy jika close > open, sell jika sebaliknya
        """
        if not data or 'open' not in data or 'close' not in data:
            return None

        if data['close'] > data['open']:
            return Signal(symbol=data['symbol'], action='buy', volume=0.1)
        elif data['close'] < data['open']:
            return Signal(symbol=data['symbol'], action='sell', volume=0.1)
        else:
            return None
