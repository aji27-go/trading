import talib as ta
import pandas as pd

def calculate_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung indikator dasar untuk dataframe harga"""
    try:
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], _ = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = ta.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0
        )
        df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        print(f"❌ Gagal menghitung indikator: {e}")
        return pd.DataFrame()

def calculate_ema_trend(df: pd.DataFrame) -> str:
    """Analisa trend berdasarkan EMA 50 & 200"""
    try:
        df['ema50'] = ta.EMA(df['close'], timeperiod=50)
        df['ema200'] = ta.EMA(df['close'], timeperiod=200)
        if df['ema50'].iloc[-1] > df['ema200'].iloc[-1]:
            return 'BULLISH'
        elif df['ema50'].iloc[-1] < df['ema200'].iloc[-1]:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    except Exception as e:
        print(f"❌ Gagal menghitung trend EMA: {e}")
        return 'NEUTRAL'
