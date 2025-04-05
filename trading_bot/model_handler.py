import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from config_loader import Config
from mt5_connector import get_historical_data

class ModelHandler:
    def __init__(self):
        self.model = None
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_path = "lstm_model.keras"
        self.init_model()

    def init_model(self):
        """Load model dari file atau latih baru jika tidak ada"""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print("\n✅ Model LSTM berhasil dimuat")
        else:
            print("\n⏳ Melatih model baru...")
            self.train_model()

    def build_lstm(self):
        """Bangun arsitektur LSTM"""
        model = Sequential([
            Input(shape=(Config.MODEL_INPUT_LENGTH, 4)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
        return model

    def train_model(self):
        """Latih model LSTM menggunakan data historis"""
        data = get_historical_data(years=1)
        if data.empty or len(data) < 1000:
            raise ValueError("❌ Data historis tidak cukup untuk pelatihan")

        features = data[['open', 'high', 'low', 'close']].ffill().bfill()
        target = data['close'].shift(-1).ffill().bfill()

        # Normalisasi
        scaled_features = self.feature_scaler.fit_transform(features)
        scaled_target = self.target_scaler.fit_transform(target.values.reshape(-1, 1))

        X, y = [], []
        for i in range(len(scaled_features) - Config.MODEL_INPUT_LENGTH - 1):
            X.append(scaled_features[i:i+Config.MODEL_INPUT_LENGTH])
            y.append(scaled_target[i+Config.MODEL_INPUT_LENGTH])
        X, y = np.array(X), np.array(y)

        self.model = self.build_lstm()
        history = self.model.fit(
            X, y,
            epochs=30,
            batch_size=128,
            callbacks=[EarlyStopping(monitor='loss', patience=3)],
            verbose=1
        )

        self.model.save(self.model_path)
        self.plot_training_loss(history.history['loss'])

        print(f"\n✅ Model disimpan | Final loss: {history.history['loss'][-1]:.6f}")

    def predict_price(self, raw_data):
        """Prediksi harga berdasarkan data input (60x4 array)"""
        scaled_input = self.feature_scaler.transform(raw_data)
        prediction = self.model.predict(np.array([scaled_input]), verbose=0)
        return self.target_scaler.inverse_transform(prediction)[0][0]

    def plot_training_loss(self, loss_history):
        """Plot dan simpan grafik training loss"""
        plt.plot(loss_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig("training_loss.png")
        plt.close()
