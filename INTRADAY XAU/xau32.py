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
                logging.info(f"\n✅ Connected to account {os.getenv('MT5_ACCOUNT')}")
                logging.info(f"🖥  Server: {os.getenv('MT5_SERVER')}")
                return
            else:
                raise ConnectionError(f"❌ Login failed: {mt5.last_error()}")
                
        except Exception as e:
            logging.error(f"⚠️  Connection error: {e}, retrying in 10s...")
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
                            logging.info(f"\n⚠️  Significant news: {article['title']} (Sentiment: {'+' if sentiment_score >0 else '-'})")
                            return True
            return False
            
        except Exception as e:
            logging.error(f"\n❌ News filter error: {str(e)}")
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
        self.active_trades = {} 
        self.trailing_active = False
        self.start_trailing_thread()
        
    def fetch_market_data(self):
        rates = mt5.copy_rates_from_pos(Config.SYMBOL, mt5.TIMEFRAME_M5, 0, 200)
        if rates is None or len(rates) == 0:
            logging.warning("⚠️ Gagal mengambil data pasar dari MT5")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        logging.info(f"Fetched market data: {len(df)} records")
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
            logging.error("\n⚠️  System Health Check Failed!")
            for k, v in checks.items():
                logging.error(f"{k.upper()}: {'OK' if v else 'FAIL'}")
            self.recover_system()

    def recover_system(self):
        """Automated system recovery"""
        logging.info("\n🛠  Attempting system recovery...")
        mt5.shutdown()
        time.sleep(5)
        mt5_connect()
        self.init_model()
        
    def init_model(self):
        """Inisialisasi model yang canggih dengan fallback"""
        try:
            model_path = "lstm_model.keras"
            if all([
                os.path.exists("lstm_model.keras"),
                os.path.exists("feature_scaler.pkl"),
                os.path.exists("target_scaler.pkl")
            ]):
                self.model = load_model("lstm_model.keras")
                self.feature_scaler = joblib.load("feature_scaler.pkl")
                self.target_scaler = joblib.load("target_scaler.pkl")
                logging.info("\n✅ Model dan scaler berhasil dimuat")
                return
                
            logging.info("\n⏳ Melatih model baru...")
            self.train_model()
            
        except Exception as e:
            logging.error(f"\n⚠️  Inisialisasi model gagal: {e}")
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
            logging.info(f"Data fetched: {len(data)} records")
            if data.empty or len(data) < 2000:
                raise ValueError("\n❌ Insufficient historical data")
                
            # Advanced preprocessing
            features = data[['open', 'high', 'low', 'close']]
            features = self.clean_data(features)
            logging.info(f"Features after cleaning: {len(features)} records")
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
            logging.info(f"Training data prepared: {len(X)} sequences")
            
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
            
            logging.info(f"\n✅ Training completed | Final Validation Loss: {history.history['val_loss'][-1]:.6f}")

        except Exception as e:
            logging.error(f"\n❌ Training failed: {e}")
            self.feature_scaler = None
            self.target_scaler = None
    def clean_data(self, df):
        """Advanced data cleaning pipeline"""
        try:
            logging.info(f"Original data: {len(df)} records")
            
            # Remove outliers
            df = df[(df['open'] > 0) & (df['open'] < 100000) &  # Adjusted range
                    (df['high'] > 0) & (df['high'] < 100000) &  # Adjusted range
                    (df['low'] > 0) & (df['low'] < 100000) &    # Adjusted range
                    (df['close'] > 0) & (df['close'] < 100000)] # Adjusted range
            logging.info(f"After removing outliers: {len(df)} records")
            
            # Handle missing values
            df = df.ffill().bfill()
            logging.info(f"After handling missing values: {len(df)} records")
            
            # Remove duplicate bars
            df = df[~df.index.duplicated(keep='last')]
            logging.info(f"After removing duplicates: {len(df)} records")
            
            return df
        except Exception as e:
            logging.error(f"\n❌ Data cleaning failed: {e}")
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
            logging.error(f"❌ Error fetching historical data: {e}")
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
                logging.error(f"⚠️  Insufficient {timeframe} data")
                return pd.DataFrame()
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Validate price data
            price_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in price_cols):
                logging.error(f"⚠️  Missing price columns")
                return pd.DataFrame()
                
            # Filter valid prices
            df = df[(df[price_cols] > 1000).all(axis=1) & (df[price_cols] < 10000).all(axis=1)]
            return df.set_index('time') if len(df) > 10 else pd.DataFrame()
            
        except Exception as e:
            logging.error(f"\n❌ Data fetch error: {e}")
            return pd.DataFrame()
    def check_symbol(self):
        try:
            if not mt5.symbol_select(Config.SYMBOL, True):
                raise ValueError(f"Simbol '{Config.SYMBOL}' tidak bisa diaktifkan.")
        except Exception as e:
            logging.error(f"\n⚠️  Gagal mengaktifkan simbol: {Config.SYMBOL}")
            logging.error("🔍 Mencoba cari simbol alternatif...\n")

            # 🔍 Cari simbol alternatif (kripto)
            fallback_symbols = [s.name for s in mt5.symbols_get() if "btc" in s.name.lower()]
            
            if fallback_symbols:
                for alt_symbol in fallback_symbols:
                    if mt5.symbol_select(alt_symbol, True):
                        logging.info(f"✅ Simbol otomatis diganti ke: {alt_symbol}")
                        Config.SYMBOL = alt_symbol  # 🚨 Overwrite simbol yang digunakan
                        return
                logging.error("❌ Tidak ada simbol kripto yang bisa diaktifkan.")
            else:
                logging.error("❌ Tidak ditemukan simbol kripto yang tersedia di akun ini.")
            
            self.show_symbol_activation_guide()
            sys.exit("Fatal Symbol Error")


    def show_symbol_activation_guide(self):
        """Interactive troubleshooting guide"""
        logging.info("\n🛠  Panduan Aktivasi Simbol:")
        logging.info("1. Pastikan MT5 terupdate ke versi terbaru")
        logging.info("2. Cek di MT5: Tools > Options > Server")
        logging.info("   Pastikan server terhubung:", Config.MT5_SERVER)
        logging.info("3. Jika pakai VPN, matikan")
        logging.info("4. Hubungi broker untuk konfirmasi:")
        logging.info("   'Apakah BTC tersedia di akun demo/server ini?'")
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
                logging.error(f"\n❌ Indicator error: {e}")
                return pd.DataFrame()
    def calculate_atr(self, periods=14):
        """Hitung Average True Range (ATR)"""
        try:
            df = self.get_recent_data(bars=periods + 1)
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=periods)
            return df['atr'].iloc[-1] if not df.empty else Config.ATR_DEFAULT
        except:
            return Config.ATR_DEFAULT
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
    def execute_trade(self, signal):
        """Eksekusi order trading dengan TP/SL dan trailing stop"""
        try:
            if self.news_filter.check_high_impact():
                logging.warning("⛔ Trade dibatalkan: Ada berita high-impact")
                return

            # Hitung level TP/SL berdasarkan ATR
            atr = self.calculate_atr()
            current_price = mt5.symbol_info_tick(Config.SYMBOL).ask if signal == 'BUY' else mt5.symbol_info_tick(Config.SYMBOL).bid
            
            # Atur risk:reward ratio 1:2
            sl_distance = atr * 2
            tp_distance = atr * 4

            if signal == 'BUY':
                sl_price = current_price - sl_distance
                tp_price = current_price + tp_distance
            else:
                sl_price = current_price + sl_distance
                tp_price = current_price - tp_distance

            request = {
                # ... parameter sebelumnya
                "sl": sl_price,
                "tp": tp_price,
                # ... sisa parameter
            }

            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Simpan info trade untuk trailing
                self.active_trades[result.order] = {
                    'type': signal,
                    'sl': sl_price,
                    'current_sl': sl_price,
                    'tp': tp_price,
                    'entry': current_price
                }
                logging.info(f"⚡ SL: {sl_price:.2f} | TP: {tp_price:.2f}")

        except Exception as e:
            logging.error(f"🔥 Error eksekusi trade: {str(e)}")

    def start_trailing_thread(self):
        """Thread untuk trailing stop otomatis"""
        def trailing_task():
            while True:
                if self.trailing_active and self.active_trades:
                    for order_id, trade in self.active_trades.items():
                        tick = mt5.symbol_info_tick(Config.SYMBOL)
                        
                        if trade['type'] == 'BUY':
                            # Update SL jika harga naik 0.5x ATR
                            new_sl = tick.bid - (self.calculate_atr() * 0.5)
                            if new_sl > trade['current_sl']:
                                self.modify_sl(order_id, new_sl)
                        else:
                            # Update SL jika harga turun 0.5x ATR
                            new_sl = tick.ask + (self.calculate_atr() * 0.5)
                            if new_sl < trade['current_sl']:
                                self.modify_sl(order_id, new_sl)
                        
                    time.sleep(10)  # Cek setiap 10 detik

        threading.Thread(target=trailing_task, daemon=True).start()

    def modify_sl(self, order_id, new_sl):
        """Modifikasi SL untuk posisi terbuka"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": order_id,
            "sl": new_sl,
            "magic": 20240406
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.active_trades[order_id]['current_sl'] = new_sl
            logging.info(f"🔄 Trailing Stop Updated | New SL: {new_sl:.2f}")
def signal_generator(df, model, feature_scaler, target_scaler, confidence_threshold):
    """Generate trading signal based on model prediction and confidence threshold."""
    try:
        if df is None or df.empty:
            logging.warning("⚠️  No market data available for signal generation")
            return None
        
        # Prepare data for prediction (4 features)
        features = df[['open', 'high', 'low', 'close']].tail(60)
        
        # Gunakan FEATURE_SCALER untuk input (4 fitur)
        scaled_features = feature_scaler.transform(features)
        prediction = model.predict(scaled_features.reshape(1, 60, 4), verbose=0)
        
        # Gunakan TARGET_SCALER untuk inverse transform hasil prediksi
        predicted_price = target_scaler.inverse_transform(prediction)[0][0]  # <-- Diubah di sini
        
        current_price = df['close'].iloc[-1]
        confidence = abs(predicted_price - current_price) / current_price
        
        logging.info(f"📈 Predicted Price: {predicted_price:.2f} | Current Price: {current_price:.2f} | Confidence: {confidence:.2f}")
        
        if confidence > confidence_threshold:
            return 'BUY' if predicted_price > current_price else 'SELL'
        return None
    
    except Exception as e:
        logging.error(f"❌ Signal generation failed: {e}")
        return None
def run():
    bot = ForexTradingBot()
    logging.info("🚀 Memulai AI GOLD TRADING BOT PRO BTC")

    # ✅ Tambahan: Load model & scaler
    model, scaler = None, None
    confidence_threshold = 0.02  # bisa kamu ubah sesuai keperluan
    try:
        model = load_model("lstm_model.keras")
        scaler = joblib.load("target_scaler.pkl")
        logging.info("✅ Model and scalers loaded successfully")
    except Exception as e:
        logging.warning(f"⚠️  Model initialization failed: {e}")

    try:
        while True:
            logging.info(f"🕒 Menjalankan bot pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            symbol_info = mt5.symbol_info_tick(Config.SYMBOL)
            if symbol_info:
                logging.info(f"📊 {Config.SYMBOL} Market:")
                logging.info(f" 🔹 Bid: {symbol_info.bid} | Ask: {symbol_info.ask} | Spread: {round(symbol_info.ask - symbol_info.bid, 5)} | Time: {datetime.fromtimestamp(symbol_info.time)}")

            df = bot.fetch_market_data()
            
            if model is not None and bot.feature_scaler is not None and bot.target_scaler is not None:
                signal = signal_generator(df, model, bot.feature_scaler, bot.target_scaler, confidence_threshold)
                if signal:
                    bot.execute_trade(signal)
            else:
                logging.error("❌ Model atau scaler tidak tersedia, tidak dapat menghasilkan sinyal") 

            if bot.trade_count > 0 and (datetime.now() - bot.last_trade_time).seconds > 7200:
                logging.warning("⚠️  System reset setelah 2 jam tidak aktif")
                bot.recover_system()

            time.sleep(300)  # Jeda 5 menit sebelum mengevaluasi sinyal lagi

    except KeyboardInterrupt:
        logging.info("\n🛑 Bot dihentikan oleh pengguna")
    finally:
        mt5.shutdown()
        logging.info("🔌 Semua koneksi MT5 ditutup")
        logging.info("📁 Log aktivitas tersedia di 'bot_activity.log'")

if __name__ == "__main__":
    run()
