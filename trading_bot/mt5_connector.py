import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from config_loader import Config

config = Config()

def get_historical_data(symbol: str, timeframe: int, bars: int = 500) -> pd.DataFrame:
    """
    Mengambil data historis dari MT5.
    """
    utc_from = datetime.now() - pd.Timedelta(minutes=bars)
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, bars)

    if rates is None or len(rates) == 0:
        raise ValueError(f"Gagal mengambil data historis untuk {symbol}.")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def mt5_connect():
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return False
    logger.info("MT5 initialized")
    return True

def mt5_shutdown():
    mt5.shutdown()
    logger.info("MT5 shutdown")

def send_order(symbol: str, order_type: str, lot: float) -> bool:
    price = mt5.symbol_info_tick(symbol).ask if order_type == "BUY" else mt5.symbol_info_tick(symbol).bid
    order_type_enum = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type_enum,
        "price": price,
        "deviation": int(config.get("TRADING", "deviation", fallback=20)),
        "magic": int(config.get("TRADING", "magic_number", fallback=123456)),
        "comment": f"Bot trade {order_type}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order failed: {result.retcode} - {result.comment}")
        return False

    logger.info(f"Order successful: {order_type} {lot} lots at {price}")
    return True

