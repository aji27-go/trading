import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from datetime import datetime, timedelta
from config import Config
import requests
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import threading
import time
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import joblib
import csv
import warnings

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,  # Ubah ke DEBUG kalau mau detail
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Untuk tampilkan di konsol
    ]
)
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# Load environment variables
load_dotenv("D:/INTRADAY XAU/XAU.env")
Config.load_config()
def mt5_connect():
    """Enhanced MT5 connection handler with retry logic"""
    while True:
        try:
            if not mt5.initialize():
                raise ConnectionError("MT5 initialization failed")
            
            authorized = mt5.login(
                login=int(os.getenv('MT5_ACCOUNT')),
                password=os.getenv('MT5_PASSWORD'),
                server=os.getenv('MT5_SERVER')
            )
            
            if authorized:
                logging(f"\n✅ Connected to account {os.getenv('MT5_ACCOUNT')}")
                logging(f"🖥  Server: {os.getenv('MT5_SERVER')}")
                return
            else:
                raise ConnectionError(f"❌ Login failed: {mt5.last_error()}")
                
        except Exception as e:
            logging(f"⚠️  Connection error: {e}, retrying in 10s...")
            time.sleep(10)

class Config:
    MT5_SERVER = "Headway-Demo"  # 🛠 TODO: Ganti dengan server MT5 sebenarnya jika berbeda
    MT5_LOGIN = "1814645"
    MT5_PASSWORD = "Js9IV$4D"  # 🛠 TODO: Ganti dengan password asli
    SYMBOL = "BTC"
    TIMEFRAMES = {
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1
    }
    RISK_PERCENT = 1.0
    MODEL_INPUT_LENGTH = 60
    NEWS_API_URL = "https://newsapi.org/v2/everything"
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    MAX_DAILY_TRADES = 5
    TRADE_COOLDOWN = 900  # 15 minutes
    ATR_DEFAULT = 3.0  # Default ATR value if calculation fails
    
    @staticmethod
    def load_config():
        with open("config.json", "r") as f:
            return json.load(f)
    
    config = load_config()

class NewsFilter:
    def check_high_impact(self):
        """Advanced news filtering with sentiment analysis"""
        try:
            now = datetime.utcnow()
            params = {
                'q': 'gold OR BTC OR Fed OR inflation',
                'from': (now - timedelta(hours=24)).strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'apiKey': Config.NEWS_API_KEY,
                'language': 'en',
                'pageSize': 20
            }
            
            response = requests.get(Config.NEWS_API_URL, params=params, timeout=10)
            news_data = response.json()
            
            keywords = {
                'positive': ['bullish', 'raise', 'strong', 'growth'],
                'negative': ['bearish', 'cut', 'weak', 'recession']
            }
            
            for article in news_data.get('articles', []):
                title = article.get('title', '').lower()
                if any(kw in title for kw in ['gold', 'btc', 'fed']):
                    pub_time = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                    if (now - pub_time).total_seconds() < 3600 * 3:
                        sentiment_score = sum(1 for kw in keywords['positive'] if kw in title) - \
                                       sum(1 for kw in keywords['negative'] if kw in title)
                        if abs(sentiment_score) >= 2:
                            logging(f"\n⚠️  Significant news: {article['title']} (Sentiment: {'+' if sentiment_score >0 else '-'})")
                            return True
            return False
            
        except Exception as e:
            logging(f"\n❌ News filter error: {str(e)}")
            return False

class ForexTradingBot:
    def __init__(self):
        self.symbol = Config.SYMBOL
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.news_filter = NewsFilter()
        self.trade_count = 0
        self.last_trade_time = None
        self.lock = threading.Lock()
        mt5_connect()
        self.check_symbol()
        self.init_model()
        self.start_system_monitor()
    def fetch_market_data(self):
        rates = mt5.copy_rates_from_pos(Config.SYMBOL, mt5.TIMEFRAME_M5, 0, 200)
        if rates is None or len(rates) == 0:
            logging.warning("⚠️ Gagal mengambil data pasar dari MT5")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df


    def start_system_monitor(self):
        """Comprehensive system monitoring thread"""
        def monitor():
            while True:
                time.sleep(3600)  # Check every hour
                self.check_system_health()
                
        threading.Thread(target=monitor, daemon=True).start()

    def check_system_health(self):
        """System health diagnostics"""
        checks = {
            'model_loaded': self.model is not None,
            'feature_scaler': self.feature_scaler is not None,
            'target_scaler': self.target_scaler is not None,
            'connection': mt5.terminal_info() is not None,
            'data_quality': len(self.get_recent_data()) > 50
        }
        
        if not all(checks.values()):
            logging("\n⚠️  System Health Check Failed!")
            for k, v in checks.items():
                logging(f"{k.upper()}: {'OK' if v else 'FAIL'}")
            self.recover_system()

    def recover_system(self):
        """Automated system recovery"""
        logging("\n🛠  Attempting system recovery...")
        mt5.shutdown()
        time.sleep(5)
        mt5_connect()
        self.init_model()
        
    def init_model(self):
        """Advanced model initialization with fallback"""
        try:
            model_path = "models/lstm_model.keras"
            if all([
                os.path.exists("lstm_model.keras"),
                os.path.exists("feature_scaler.pkl"),
                os.path.exists("target_scaler.pkl")
            ]):
                self.model = load_model("lstm_model.keras")
                self.feature_scaler = joblib.load("feature_scaler.pkl")
                self.target_scaler = joblib.load("target_scaler.pkl")
                logging("\n✅ Model and scalers loaded successfully")
                return
                
            logging("\n⏳ Training new model...")
            self.train_model()
            
        except Exception as e:
            logging(f"\n⚠️  Model initialization failed: {e}")
            self.train_model()

    def build_lstm(self):
        """Enhanced LSTM architecture"""
        model = Sequential([
            Input(shape=(Config.MODEL_INPUT_LENGTH, 4)),
            LSTM(128, return_sequences=True, recurrent_dropout=0.2),
            Dropout(0.3),
            LSTM(64, recurrent_dropout=0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        return model

    def train_model(self):
        """Robust training process with enhanced validation"""
        try:
            data = self.get_historical_data(2)  # 2 years data
            if data.empty or len(data) < 2000:
                raise ValueError("\n❌ Insufficient historical data")
                
            # Advanced preprocessing
            features = data[['open', 'high', 'low', 'close']]
            features = self.clean_data(features)
            target = data['close'].shift(-1).ffill().bfill()
            
            # Initialize scalers
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
            
            scaled_features = self.feature_scaler.fit_transform(features)
            scaled_target = self.target_scaler.fit_transform(target.values.reshape(-1, 1))
            
            # Create sequences with lookback
            X, y = [], []
            for i in range(len(scaled_features)-Config.MODEL_INPUT_LENGTH-1):
                X.append(scaled_features[i:i+Config.MODEL_INPUT_LENGTH])
                y.append(scaled_target[i+Config.MODEL_INPUT_LENGTH])
                
            X, y = np.array(X), np.array(y)
            
            # Train model with validation split
            self.model = self.build_lstm()
            history = self.model.fit(
                X, 
                y,
                epochs=50,
                batch_size=256,
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=5),
                ],
                verbose=1
            )
            
            # Save artifacts
            self.model.save("lstm_model.keras")
            joblib.dump(self.feature_scaler, 'feature_scaler.pkl')
            joblib.dump(self.target_scaler, 'target_scaler.pkl')
            
            # Plot training results
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Training Progress')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.savefig('training_progress.png')
            plt.close()
            
            logging(f"\n✅ Training completed | Final Validation Loss: {history.history['val_loss'][-1]:.6f}")

        except Exception as e:
            logging(f"\n❌ Training failed: {e}")
            self.feature_scaler = None
            self.target_scaler = None

    def clean_data(self, df):
        """Advanced data cleaning pipeline"""
        try:
            # Remove outliers
            df = df[(df > 1000).all(axis=1) & (df < 10000).all(axis=1)]
            
            # Handle missing values
            df = df.ffill().bfill()
            
            # Remove duplicate bars
            df = df[~df.index.duplicated(keep='last')]
            
            return df
        except Exception as e:
            logging(f"\n❌ Data cleaning failed: {e}")
            return df

    def get_historical_data(self, years=2, timeframe=mt5.TIMEFRAME_M15):
        """Fetch historical data from MT5 for the given number of years."""
        try:
            if not mt5.initialize():
                raise RuntimeError("Failed to initialize MT5")

            end = datetime.now()
            start = end - timedelta(days=365 * years)

            rates = mt5.copy_rates_range(
                self.symbol,  # <- pastikan self.symbol sudah diset (misal: "BTCUSD")
                timeframe,
                start,
                end
            )

            if rates is None or len(rates) == 0:
                raise ValueError("No historical data fetched")

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            return df

        except Exception as e:
            logging(f"❌ Error fetching historical data: {e}")
            return pd.DataFrame()

    def get_recent_data(self, timeframe='M15', bars=100):
        """Robust data fetcher with validation"""
        try:
            rates = mt5.copy_rates_from_pos(
                Config.SYMBOL, 
                Config.TIMEFRAMES[timeframe], 
                0, 
                bars
            )
            
            if rates is None or len(rates) < 20:
                logging(f"⚠️  Insufficient {timeframe} data")
                return pd.DataFrame()
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Validate price data
            price_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in price_cols):
                logging(f"⚠️  Missing price columns")
                return pd.DataFrame()
                
            # Filter valid prices
            df = df[(df[price_cols] > 1000).all(axis=1) & (df[price_cols] < 10000).all(axis=1)]
            return df.set_index('time') if len(df) > 10 else pd.DataFrame()
            
        except Exception as e:
            logging(f"\n❌ Data fetch error: {e}")
            return pd.DataFrame()
    def check_symbol(self):
        try:
            if not mt5.symbol_select(Config.SYMBOL, True):
                raise ValueError(f"Simbol '{Config.SYMBOL}' tidak bisa diaktifkan.")
        except Exception as e:
            logging(f"\n⚠️  Gagal mengaktifkan simbol: {Config.SYMBOL}")
            logging("🔍 Mencoba cari simbol alternatif...\n")

            # 🔍 Cari simbol alternatif (kripto)
            fallback_symbols = [s.name for s in mt5.symbols_get() if "btc" in s.name.lower()]
            
            if fallback_symbols:
                for alt_symbol in fallback_symbols:
                    if mt5.symbol_select(alt_symbol, True):
                        logging(f"✅ Simbol otomatis diganti ke: {alt_symbol}")
                        Config.SYMBOL = alt_symbol  # 🚨 Overwrite simbol yang digunakan
                        return
                logging("❌ Tidak ada simbol kripto yang bisa diaktifkan.")
            else:
                logging("❌ Tidak ditemukan simbol kripto yang tersedia di akun ini.")
            
            self.show_symbol_activation_guide()
            sys.exit("Fatal Symbol Error")


    def show_symbol_activation_guide(self):
        """Interactive troubleshooting guide"""
        logging("\n🛠  Panduan Aktivasi Simbol:")
        logging("1. Pastikan MT5 terupdate ke versi terbaru")
        logging("2. Cek di MT5: Tools > Options > Server")
        logging("   Pastikan server terhubung:", Config.MT5_SERVER)
        logging("3. Jika pakai VPN, matikan")
        logging("4. Hubungi broker untuk konfirmasi:")
        logging("   'Apakah BTC tersedia di akun demo/server ini?'")
    def calculate_indicators(self, df):
            """Enhanced indicator calculation with fallback"""
            try:
                if df.empty:
                    return pd.DataFrame()
                    
                # Calculate ATR with multiple methods
                df['atr_ta'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
                df['atr_manual'] = df['high'] - df['low']
                df['atr'] = df[['atr_ta', 'atr_manual']].mean(axis=1)
                
                # Filter and smooth ATR
                df['atr'] = df['atr'].apply(lambda x: x if 0.5 < x < 50 else np.nan)
                df['atr'] = df['atr'].interpolate().ffill().bfill()
                
                # Additional indicators
                df['macd'], df['signal'], _ = ta.MACD(df['close'])
                df['rsi'] = ta.RSI(df['close'])
                df['ema50'] = ta.EMA(df['close'], timeperiod=50)
                df['ema200'] = ta.EMA(df['close'], timeperiod=200)
                df['adx'] = ta.ADX(df['high'], df['low'], df['close'])
                return df.dropna().reset_index(drop=True)
                
            except Exception as e:
                logging(f"\n❌ Indicator error: {e}")
                return pd.DataFrame()

    def trend_analysis(self, df):
        """Advanced trend detection"""
        try:
            if len(df) < 100:
                return 'NEUTRAL'
                
            ema50 = df['ema50'].iloc[-1]
            ema200 = df['ema200'].iloc[-1]
            price_ratio = df['close'].iloc[-1] / df['close'].iloc[-100]
            
            if ema50 > ema200 and price_ratio > 1.02:
                return 'STRONG_BULLISH'
            elif ema50 > ema200:
                return 'BULLISH'
            elif ema50 < ema200 and price_ratio < 0.98:
                return 'STRONG_BEARISH'
            elif ema50 < ema200:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
        except:
            return 'NEUTRAL'

    def generate_signal(self):
        """Advanced signal generation with multiple confirmations"""
        try:
            if self.news_filter.check_high_impact():
                logging("\n🔇 Trading paused due to news")
                return None
                
            m15_data = self.get_recent_data('M15', 200)
            if m15_data.empty:
                return None
                
            m15_data = self.calculate_indicators(m15_data)
            
            # Market condition filter
            if m15_data['atr'].iloc[-1] < 1.5:
                logging("\n🔇 Low volatility market")
                return None
                
            if m15_data['adx'].iloc[-1] < 25:
                logging("\n🔇 Sideways market")
                return None
            
            # Prepare prediction data
            raw_data = m15_data[['open','high','low','close']].tail(60).values
            
            # Validate scaler
            if (self.feature_scaler is None or 
                raw_data.shape[1] != self.feature_scaler.n_features_in_):
                raise ValueError("Scaler mismatch")
                
            scaled_data = self.feature_scaler.transform(raw_data)
            prediction = self.model.predict(np.array([scaled_data]), verbose=0)
            predicted_price = self.target_scaler.inverse_transform(prediction)[0][0]
            
            # Get trend confirmation
            h1_data = self.get_recent_data('H1', 200)
            trend = self.trend_analysis(h1_data) if not h1_data.empty else 'NEUTRAL'
            
            # Generate signal
            current_price = m15_data['close'].iloc[-1]
            price_diff = abs(predicted_price - current_price)
            
            logging(f"\n📊 Prediction: {predicted_price:.2f} | Current: {current_price:.2f}")
            logging(f"📈 Trend: {trend} | ATR: {m15_data['atr'].iloc[-1]:.2f}")
            
            if trend in ['STRONG_BULLISH', 'BULLISH'] and predicted_price > current_price + max(0.5, m15_data['atr'].iloc[-1]*0.3):
                logging("\n🚀 STRONG BUY SIGNAL")
                return 'BUY'
                
            if trend in ['STRONG_BEARISH', 'BEARISH'] and predicted_price < current_price - max(0.5, m15_data['atr'].iloc[-1]*0.3):
                logging("\n🔻 STRONG SELL SIGNAL")
                return 'SELL'
                
            logging("\n🔍 No valid signal")
            return None
            
        except Exception as e:
            logging(f"\n❌ Signal error: {e}")
            self.recover_system()
            return None

    def execute_trade(self, signal):
        """Enhanced trade execution with ATR protection"""
        with self.lock:
            try:
                # Connection check
                if not mt5.initialize():
                    self.recover_system()
                    time.sleep(5)
                
                # Trade limits
                if self.trade_count >= Config.MAX_DAILY_TRADES:
                    logging("\n⏳ Daily trade limit reached")
                    return
                    
                if self.last_trade_time and (datetime.now() - self.last_trade_time).seconds < Config.TRADE_COOLDOWN:
                    return
                
                # Get market data
                df = self.get_recent_data('M15', 100)
                df = self.calculate_indicators(df)
                
                # ATR validation
                if df.empty or 'atr' not in df.columns:
                    atr = Config.ATR_DEFAULT
                    logging(f"⚠️  Using default ATR: {atr}")
                else:
                    atr = df['atr'].iloc[-1]
                                   # Validasi akhir ATR
                if np.isnan(atr) or atr <= 0:
                    logging(f"⚠️  ATR invalid: {atr}, menggunakan default {Config.ATR_DEFAULT}")
                    atr = Config.ATR_DEFAULT

                # Kalkulasi parameter trading
                tick = mt5.symbol_info_tick(Config.SYMBOL)
                current_price = tick.ask if signal == 'BUY' else tick.bid
                balance = mt5.account_info().balance
                h1_data = self.get_recent_data('H1', 200)
                trend = self.trend_analysis(h1_data) 
                # Manajemen risiko dinamis
                risk_amount = balance * (Config.RISK_PERCENT / 100) * (0.5 if trend.startswith('STRONG') else 1.0)
                lot_size = round(risk_amount / (atr * 10), 2)
                lot_size = max(0.01, min(lot_size, 20))  # MT5 batasan lot

                # Setup order dengan proteksi spread
                spread = tick.ask - tick.bid
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": Config.SYMBOL,
                    "volume": lot_size,
                    "type": mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL,
                    "price": current_price,
                    "sl": current_price - (atr * 1.5 + spread) if signal == 'BUY' 
                            else current_price + (atr * 1.5 + spread),
                    "tp": current_price + (atr * 3 - spread) if signal == 'BUY' 
                            else current_price - (atr * 3 - spread),
                    "deviation": 20,
                    "magic": 2024,
                    "comment": "AI-BTC-TRADE",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC
                }

                # Eksekusi order
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logging(f"\n❌ Gagal order: {result.comment}")
                    self.log_trade_error(signal, current_price, result.comment)
                else:
                    self.trade_count += 1
                    self.last_trade_time = datetime.now()
                    logging(f"\n✅ Order sukses | Lot: {lot_size:.2f}")
                    logging(f"   SL: {request['sl']:.2f} | TP: {request['tp']:.2f}")
                    self.start_trailing_stop(result.order)
                    self.log_trade(signal, current_price, 'SUCCESS', '')

            except Exception as e:
                logging(f"\n❌ Critical trade error: {e}")
                self.log_trade_error(signal, current_price, str(e))
                self.recover_system()

    def log_trade(self, signal, price, result, reason):
        """Enhanced logging dengan struktur JSON"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': Config.SYMBOL,
            'signal': signal,
            'price': price,
            'result': result,
            'reason': reason,
            'balance': mt5.account_info().balance,
            'equity': mt5.account_info().equity
        }
        
        with open('trading_log.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def start_trailing_stop(self, order_id):
        def trail():
            try:
                while True:
                    positions = mt5.positions_get(ticket=order_id)
                    if not positions:
                        break
                        
                    position = positions[0]
                    df = self.get_recent_data('M15', 20)
                    df = self.calculate_indicators(df)
                    
                    # Update ATR secara real-time
                    current_atr = df['atr'].iloc[-1] if not df.empty else Config.ATR_DEFAULT
                    current_price = mt5.symbol_info_tick(Config.SYMBOL).ask
                    
                    # Hitung SL dinamis
                    if position.type == mt5.ORDER_TYPE_BUY:
                        new_sl = current_price - (current_atr * 1.2)
                        new_sl = max(new_sl, position.sl)  # Hanya geser ke atas
                    else:
                        new_sl = current_price + (current_atr * 1.2)
                        new_sl = min(new_sl, position.sl)  # Hanya geser ke bawah
                    
                    # Update jika perubahan signifikan
                    if abs(new_sl - position.sl) > 0.5:
                        result = mt5.order_modify(position.ticket, sl=new_sl)
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"🔄 Trailing SL updated: {new_sl:.2f}")
                            self.log_trade('SL_UPDATE', new_sl, 'SUCCESS', '')
                            
                    time.sleep(60)  # Update setiap 1 menit
                    
            except Exception as e:
                print(f"\n❌ Trailing error: {e}")
                
        threading.Thread(target=trail, daemon=True).start()

    # 2. Perbaikan error logging
    def log_trade_error(self, signal, price, error_msg):
        logging.error(f"❌ Critical trade error on {signal.upper()} at {price}: {error_msg}")

    # 3. Perbaikan trend detection
    def detect_trend(df):
        """
        Deteksi tren sederhana berdasarkan perbandingan Moving Average
        """
        df['ma_fast'] = df['close'].rolling(window=10).mean()
        df['ma_slow'] = df['close'].rolling(window=50).mean()

        if df['ma_fast'].iloc[-1] > df['ma_slow'].iloc[-1]:
            return "UPTREND"
        elif df['ma_fast'].iloc[-1] < df['ma_slow'].iloc[-1]:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
            
    def log_trade_error(self, signal, price, error_msg):
        logging.error(f"❌ Critical trade error on {signal.upper()} at {price}: {error_msg}")

def detect_trend(df):
    """
    Deteksi tren sederhana berdasarkan perbandingan Moving Average
    """
    df['ma_fast'] = df['close'].rolling(window=10).mean()
    df['ma_slow'] = df['close'].rolling(window=50).mean()

    if df['ma_fast'].iloc[-1] > df['ma_slow'].iloc[-1]:
        return "UPTREND"
    elif df['ma_fast'].iloc[-1] < df['ma_slow'].iloc[-1]:
        return "DOWNTREND"
    else:
        return "SIDEWAYS"

def signal_generator(df, model=None, scaler=None, confidence_threshold=0.6):
    """
    Generate trading signal based on technical indicators and AI prediction if available.
    """
    # --- 1. Hitung indikator teknikal dasar
    df['ema'] = ta.ema(df['close'], length=50)
    df['ma'] = ta.sma(df['close'], length=200)
    df['rsi'] = ta.rsi(df['close'], length=14)

    # --- 2. Tentukan tren berdasarkan posisi EMA vs MA
    trend = "Naik" if df['ema'].iloc[-1] > df['ma'].iloc[-1] else "Turun"
    logging.info(f"🔁 Tren saat ini: {trend}")

    # --- 3. Aturan sinyal berdasarkan RSI klasik
    rsi = df['rsi'].iloc[-1]
    signal = None
    if rsi < 30:
        signal = "BUY"
    elif rsi > 70:
        signal = "SELL"

    # --- 4. AI prediction (jika model & scaler disediakan)
    if model and scaler:
        try:
            # Gunakan fitur terakhir sebagai input model
            features = df[['ema', 'ma', 'rsi', 'close']].dropna().tail(1)
            scaled_features = scaler.transform(features)

            prediction = model.predict(scaled_features)[0][0]
            confidence = 1.0 - (abs(prediction - df['close'].iloc[-1])/df['close'].iloc[-1])
            
            if confidence >= confidence_threshold:
                signal = "BUY" if prediction > df['close'].iloc[-1] else "SELL"

            # Logging prediksi
            if confidence >= confidence_threshold:
                signal = prediction
                logging.info(f"🤖 Prediksi AI: {prediction} (confidence: {confidence:.2f})")
            else:
                logging.info(f"🤖 Prediksi AI: ⏳ Tidak ada sinyal (confidence rendah: {confidence:.2f})")

        except Exception as e:
            logging.warning(f"❌ AI prediction error: {e}")
    else:
        # Jika tidak ada model, log sinyal teknikal saja
        logging.info(f"🤖 Prediksi AI: {signal if signal else '⏳ Tidak ada sinyal'}")

    return signal

def run():

    bot = ForexTradingBot()
    confidence_threshold = 0.6
    logging.info("🚀 Memulai AI GOLD TRADING BOT PRO BTC")
    logging.info(f"🔧 Versi: {datetime.now().strftime('%Y-%m-%d')}")
    # ✅ Tambahan: Load model & scaler
    try:
        model = joblib.load("models/model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        confidence_threshold = 0.6  # bisa kamu ubah sesuai keperluan
        logging.info("✅ Model and scalers loaded successfully")
    except Exception as e:
        model, scaler = None, None
        logging.warning(f"⚠️  Model initialization failed: {e}")

    try:
        while True:
            logging.info(f"🕒 Menjalankan bot pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            symbol_info = mt5.symbol_info_tick(Config.SYMBOL)
            if symbol_info:
                logging.info(f"📊 {Config.SYMBOL} Market:")
                logging.info(f"   🔹 Bid: {symbol_info.bid} | Ask: {symbol_info.ask} | Spread: {round(symbol_info.ask - symbol_info.bid, 5)} | Time: {datetime.fromtimestamp(symbol_info.time)}")

            df = bot.fetch_market_data()
            signal = signal_generator(df, model, scaler, confidence_threshold)

            if signal:
                bot.execute_trade(signal)

            if bot.trade_count > 0 and (datetime.now() - bot.last_trade_time).seconds > 7200:
                logging.warning("⚠️  System reset setelah 2 jam tidak aktif")
                bot.recover_system()

            time.sleep(300)

    except KeyboardInterrupt:
        logging("\n🛑 Bot dihentikan oleh pengguna")
    finally:
        mt5.shutdown()
        logging.info("🔌 Semua koneksi MT5 ditutup")
        logging.info("📁 Log aktivitas tersedia di 'bot_activity.log'")


if __name__ == "__main__":
    run()

