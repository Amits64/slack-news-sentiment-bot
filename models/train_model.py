import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization,
                                     Bidirectional, LSTM, Dense, Dropout,
                                     Attention, Concatenate)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas_ta as ta
from matplotlib import pyplot as plt
from config import Config
import keras_tuner as kt
import yfinance as yf

# ---------------------------
# ✅ Get macroeconomic data
# ---------------------------
def fetch_macro_indicators(start_date="2020-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    tickers = {
        "DXY": "DX-Y.NYB",  # US Dollar Index
        "SP500": "^GSPC",  # S&P 500
        "Gold": "GC=F",  # Gold Futures
        "Oil": "CL=F",  # Crude Oil Futures
        "US10Y": "^TNX",  # 10-Year Treasury Yield
        "CPI_ETF": "TIP"  # TIPS ETF as CPI proxy
    }

    macro_df = pd.DataFrame()

    for name, symbol in tickers.items():
        print(f"Fetching {name} ({symbol})...")
        data = yf.download(symbol, start=start_date, end=end_date, interval="1d", progress=False)

        # Use Close instead of Adj Close (since auto_adjust=True by default now)
        if "Close" in data.columns:
            series = data["Close"].copy()
        else:
            print(f"⚠️ Warning: 'Close' column not found for {symbol}. Skipping...")
            continue

        series.name = name
        macro_df = pd.concat([macro_df, series], axis=1)

    macro_df.index.name = "Date"
    macro_df = macro_df.ffill().dropna()
    print("✅ Macro indicators fetched and processed")
    return macro_df


# ---------------------------
# ✅ Get top 20 crypto symbols by 24h USDT trading volume
# ---------------------------
def get_top_20_symbols(client):
    """Get top 20 crypto symbols by 24h USDT trading volume"""
    # Use get_ticker() for older versions of python-binance
    tickers = client.get_ticker()  # works as an alias in some versions

    # Filter only symbols that end with USDT and have volume
    usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT') and 'quoteVolume' in t]
    sorted_symbols = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    top_20 = [s['symbol'] for s in sorted_symbols[:20]]
    return top_20


# ---------------------------
# ⚙️ Advanced Crypto Predictor Class
# ---------------------------
class CryptoPredictor:
    def __init__(self, interval=Client.KLINE_INTERVAL_1DAY, lookback="3650 days ago UTC",
                 seq_length=60, future_offset=5):
        """Initialize predictor and set parameters."""
        self.api_key = Config.BINANCE_API_KEY
        self.api_secret = Config.BINANCE_API_SECRET
        self.client = Client(self.api_key, self.api_secret)

        self.interval = interval
        self.lookback = lookback
        self.seq_length = seq_length
        self.future_offset = future_offset

        self.top_symbols = []
        self.all_data = {}  # Dictionary to store data for all symbols
        self.scaler = MinMaxScaler()
        self.model_a = None
        self.model_b = None
        self.input_shape = None

        # Define the list of features we'll use for modeling
        self.features = [
            'Open_norm', 'High_norm', 'Low_norm', 'Close_norm',
            'SMA_20', 'EMA_20', 'RSI_14', 'MACD',
            'BBL', 'BBM', 'BBU', 'ADX',
            'CCI', 'OBV', 'Pct_Change_1d', 'Pct_Change_7d',
            'Volatility'
        ]

    # ---------------------------
    # 📊 Fetch Data for Top 20 Coins
    # ---------------------------
    def fetch_data(self):
        """Fetch historical data for top 20 coins by volume from Binance."""
        try:
            # Get top 20 symbols by trading volume
            self.top_symbols = get_top_20_symbols(self.client)
            print(f"✅ Top 20 symbols by volume: {self.top_symbols}")

            # Fetch data for each symbol
            for symbol in self.top_symbols:
                print(f"Fetching data for {symbol}...")
                klines = self.client.get_historical_klines(symbol, self.interval, self.lookback)

                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                    'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
                ])

                df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('Date', inplace=True)
                float_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                df[float_cols] = df[float_cols].astype(float)

                # Store the dataframe in the dictionary
                self.all_data[symbol] = df

            total_rows = sum(len(df) for df in self.all_data.values())
            print(f"✅ Data fetched for {len(self.all_data)} symbols, total {total_rows} rows")
        except Exception as e:
            print(f"⚠️ Error fetching data: {e}")
            exit()

    # ---------------------------
    # 📊 Fetch Macroeconomic Data
    # ---------------------------
    def integrate_macro_data(self, macro_df):
        """Add macroeconomic indicators to each coin's dataset."""
        if not self.all_data:
            print("⚠️ No coin data to integrate with macro indicators.")
            return

        for symbol, df in self.all_data.items():
            df = df.copy()
            df = df.join(macro_df, how='left')  # Align by date
            df = df.ffill().dropna()  # Fill missing macro rows
            self.all_data[symbol] = df

        # Add macro feature names to the feature list if not already present
        new_macros = [col for col in macro_df.columns if col not in self.features]
        self.features.extend(new_macros)
        print(f"✅ Integrated macro indicators: {new_macros}")

    # ---------------------------
    # 📈 Add Technical Indicators
    # ---------------------------
    def add_indicators(self):
        """Add technical indicators to enrich the feature set for all symbols."""
        if not self.all_data:
            print("⚠️ No data available. Fetch data first.")
            return

        for symbol, df in self.all_data.items():
            if df.empty:
                continue

            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['EMA_20'] = df['Close'].ewm(span=20).mean()
            df['RSI_14'] = ta.rsi(df['Close'], length=14)
            macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            df['MACD'] = macd['MACDh_12_26_9']
            bb = ta.bbands(df['Close'], length=20, std=2)
            df['BBL'] = bb['BBL_20_2.0']
            df['BBM'] = bb['BBM_20_2.0']
            df['BBU'] = bb['BBU_20_2.0']
            df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
            df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
            df['OBV'] = ta.obv(df['Close'], df['Volume'])

            # Add symbol-specific normalization: divide by the median price to make different coins comparable
            median_price = df['Close'].median()
            df['Open_norm'] = df['Open'] / median_price
            df['High_norm'] = df['High'] / median_price
            df['Low_norm'] = df['Low'] / median_price
            df['Close_norm'] = df['Close'] / median_price

            # Add percent change features
            df['Pct_Change_1d'] = df['Close'].pct_change(1)
            df['Pct_Change_7d'] = df['Close'].pct_change(7)

            # Add Market Cap proxy (price * volume)
            df['Market_Cap_Proxy'] = df['Close'] * df['Volume']

            # Add volatility indicator
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()

            # Drop rows with NaN values
            self.all_data[symbol] = df.dropna()

        print("✅ Indicators added successfully for all symbols!")

    # ---------------------------
    # 🔥 Create Sequences for Training (Combined Approach)
    # ---------------------------
    def create_sequences(self):
        """Create sequences from all symbols for a unified model."""
        if not self.all_data:
            print("⚠️ No data available. Fetch and process data first.")
            return None, None, None, None

        # Use the features defined in __init__
        # Combine all data for scaling
        all_feature_data = []
        for symbol, df in self.all_data.items():
            valid_features = [col for col in self.features if col in df.columns]
            if len(valid_features) == len(self.features):  # Only use if all features present
                all_feature_data.append(df[valid_features])

        if not all_feature_data:
            print("⚠️ No valid feature data found.")
            return None, None, None, None

        combined_data = pd.concat(all_feature_data)

        # Train-test split on combined data
        train_size = int(0.8 * len(combined_data))
        train_data = combined_data.iloc[:train_size]
        test_data = combined_data.iloc[train_size:]

        # Scale the data
        self.scaler.fit(train_data)
        train_scaled = self.scaler.transform(train_data)
        test_scaled = self.scaler.transform(test_data)

        # Target will be the normalized close price at future_offset
        def create(X, y, offset):
            sequences, targets = [], []
            for i in range(len(X) - self.seq_length - offset):
                seq = X[i:i + self.seq_length]
                # Use Close_norm column (index 3) as target
                target = y[i + self.seq_length + offset - 1, 3]
                sequences.append(seq)
                targets.append(target)
            return np.array(sequences), np.array(targets)

        X_train, y_train = create(train_scaled, train_scaled, self.future_offset)
        X_test, y_test = create(test_scaled, test_scaled, self.future_offset)

        if X_train.size == 0 or X_test.size == 0:
            print("⚠️ Could not create sequences, not enough data after filtering.")
            return None, None, None, None

        print(f"✅ Sequences created: {X_train.shape}, {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ---------------------------
    # 🔥 Define Model Architectures with Hyperparameter Search
    # ---------------------------
    def build_model_a(self, hp):
        """Advanced model: CNN + Bidirectional LSTM + Attention."""
        inputs = Input(shape=self.input_shape)
        cnn_filters = hp.Int("cnn_filters", 16, 64, step=16, default=32)
        x = Conv1D(filters=cnn_filters, kernel_size=hp.Choice("kernel_size", [3, 5]), activation='relu',
                   padding='same')(inputs)
        x = BatchNormalization()(x)
        lstm_units = hp.Int("lstm_units", 32, 128, step=32, default=64)
        x = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(0.01)))(x)
        x = Dropout(hp.Float("dropout_lstm", 0.2, 0.5, step=0.1, default=0.3))(x)
        attention = Attention()([x, x])
        attention = BatchNormalization()(attention)
        combined = Concatenate()([x, attention])
        lstm_units2 = hp.Int("lstm_units2", 16, 64, step=16, default=32)
        x = LSTM(lstm_units2, return_sequences=False)(combined)
        x = Dropout(hp.Float("dropout_dense", 0.2, 0.5, step=0.1, default=0.3))(x)
        dense_units = hp.Int("dense_units", 16, 64, step=16, default=32)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(hp.Float("dropout_dense2", 0.2, 0.5, step=0.1, default=0.3))(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        lr = hp.Float("lr", 1e-4, 1e-2, sampling="log", default=1e-3)
        model.compile(optimizer=AdamW(learning_rate=lr), loss='mse')
        return model

    def build_model_b(self, hp):
        """Simpler model: CNN + LSTM."""
        inputs = Input(shape=self.input_shape)
        cnn_filters = hp.Int("cnn_filters", 16, 64, step=16, default=32)
        x = Conv1D(filters=cnn_filters, kernel_size=hp.Choice("kernel_size", [3, 5]), activation='relu',
                   padding='same')(inputs)
        x = BatchNormalization()(x)
        lstm_units = hp.Int("lstm_units", 32, 128, step=32, default=64)
        x = LSTM(lstm_units, return_sequences=False, kernel_regularizer=l2(0.01))(x)
        x = Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1, default=0.3))(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        lr = hp.Float("lr", 1e-4, 1e-2, sampling="log", default=1e-3)
        model.compile(optimizer=AdamW(learning_rate=lr), loss='mse')
        return model

    # ---------------------------
    # 🔍 Hyperparameter Tuning Utility
    # ---------------------------
    def tune_model(self, build_fn, X_train, y_train, X_val, y_val, max_trials=10, executions_per_trial=1,
                   project_name="tuner"):
        tuner = kt.RandomSearch(
            build_fn,
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory="tuner_dir",
            project_name=project_name,
            overwrite=True
        )
        tuner.search(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=50,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                     verbose=1)
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("✅ Best hyperparameters for", project_name, ":", best_hp.values)
        return tuner.hypermodel.build(best_hp)

    # ---------------------------
    # ✅ Train, Tune, and Evaluate with Ensemble
    # ---------------------------
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        # Split training data for tuning
        split_idx = int(0.9 * len(X_train))
        X_tr, y_tr = X_train[:split_idx], y_train[:split_idx]
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]
        self.input_shape = X_train.shape[1:]

        print("Tuning Model A (Advanced)...")
        model_a = self.tune_model(self.build_model_a, X_tr, y_tr, X_val, y_val, project_name="model_a")
        print("Tuning Model B (Simpler)...")
        model_b = self.tune_model(self.build_model_b, X_tr, y_tr, X_val, y_val, project_name="model_b")

        # Callbacks for full training
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_a = ModelCheckpoint(f'model/model_a_{timestamp}.keras', save_best_only=True)
        checkpoint_b = ModelCheckpoint(f'model/model_b_{timestamp}.keras', save_best_only=True)

        print("Training Model A on full training data...")
        history_a = model_a.fit(X_train, y_train,
                                validation_data=(X_test, y_test),
                                epochs=100,
                                batch_size=32,
                                callbacks=[early_stopping, lr_scheduler, checkpoint_a],
                                verbose=1)
        print("Training Model B on full training data...")
        history_b = model_b.fit(X_train, y_train,
                                validation_data=(X_test, y_test),
                                epochs=100,
                                batch_size=32,
                                callbacks=[early_stopping, lr_scheduler, checkpoint_b],
                                verbose=1)

        # Ensemble prediction: average predictions from both models
        print("Ensembling predictions...")
        y_pred_a = model_a.predict(X_test)
        y_pred_b = model_b.predict(X_test)
        y_pred_a = y_pred_a.reshape(-1)
        y_pred_b = y_pred_b.reshape(-1)
        ensemble_pred = (y_pred_a + y_pred_b) / 2.0
        y_test_flat = y_test.reshape(-1)

        # For evaluation, we need to create a full feature vector to inverse transform
        # Use the class property self.features to determine the length and positions
        dummy_features = np.zeros((len(ensemble_pred), len(self.features)))
        dummy_features[:, 3] = ensemble_pred  # Close_norm is at index 3

        # Same for test values
        dummy_test = np.zeros((len(y_test_flat), len(self.features)))
        dummy_test[:, 3] = y_test_flat  # Close_norm is at index 3

        # Inverse transform
        y_pred_inv = self.scaler.inverse_transform(dummy_features)[:, 3]  # Get Close_norm column
        y_test_inv = self.scaler.inverse_transform(dummy_test)[:, 3]  # Get Close_norm column

        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)

        print("\n📊 Ensemble Model Evaluation Metrics:")
        print(f"✅ MSE: {mse:.4f}")
        print(f"✅ RMSE: {rmse:.4f}")
        print(f"✅ MAE: {mae:.4f}")
        print(f"✅ R² Score: {r2:.4f}")

        plt.figure(figsize=(14, 7))
        plt.plot(y_test_inv, label="Actual Normalized Price", color="blue")
        plt.plot(y_pred_inv, label="Ensemble Predicted Normalized Price", color="orange")
        plt.title("Actual vs Ensemble Predicted Price (Normalized)")
        plt.xlabel("Time")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Save the trained models
        model_a.save(f'model/top20_model_a_{timestamp}.keras')
        model_b.save(f'model/top20_model_b_{timestamp}.keras')
        print(f"✅ Models saved with timestamp {timestamp}")

        self.model_a = model_a
        self.model_b = model_b

        return model_a, model_b

    # ---------------------------
    # 🔮 Make predictions for new data
    # ---------------------------
    def predict_future(self, days=30):
        """Make future predictions for all top symbols."""
        if self.model_a is None or self.model_b is None:
            print("⚠️ Models not trained. Train models first.")
            return

        results = {}

        for symbol in self.top_symbols:
            if symbol not in self.all_data:
                continue

            df = self.all_data[symbol]
            if len(df) < self.seq_length:
                continue

            # Get the latest data for prediction
            valid_features = [col for col in self.features if col in df.columns]

            latest_data = df[valid_features].iloc[-self.seq_length:].values
            latest_scaled = self.scaler.transform(latest_data)
            latest_seq = latest_scaled.reshape(1, self.seq_length, len(valid_features))

            # Get median price for denormalization
            median_price = df['Close'].median()

            # Make predictions for each future day
            future_preds = []
            current_seq = latest_seq.copy()

            for i in range(days):
                # Get ensemble prediction
                pred_a = self.model_a.predict(current_seq)[0][0]
                pred_b = self.model_b.predict(current_seq)[0][0]
                ensemble_pred = (pred_a + pred_b) / 2.0

                # Denormalize the prediction
                dummy_features = np.zeros((1, len(self.features)))
                dummy_features[0, 3] = ensemble_pred  # Close_norm is at index 3
                denorm_pred = self.scaler.inverse_transform(dummy_features)[0, 3] * median_price

                future_preds.append(denorm_pred)

                # Update the sequence for the next prediction (roll forward)
                # Create a dummy row for the new prediction
                new_row = current_seq[0, -1:].copy()
                new_row[0, 3] = ensemble_pred  # Set the Close_norm value

                # Roll the window forward (drop oldest, add newest)
                current_seq[0, :-1] = current_seq[0, 1:]
                current_seq[0, -1:] = new_row

            results[symbol] = future_preds

        return results


# ---------------------------
# 🚀 Main Execution
# ---------------------------
if __name__ == "__main__":
    predictor = CryptoPredictor()
    predictor.fetch_data()
    predictor.add_indicators()

    # ⬇️ Fetch and merge macro data
    macro_df = fetch_macro_indicators(start_date="2021-01-01")
    predictor.integrate_macro_data(macro_df)

    X_train, X_test, y_train, y_test = predictor.create_sequences()

    if X_train is not None and X_train.size > 0:
        model_a, model_b = predictor.train_and_evaluate(X_train, y_train, X_test, y_test)

        # Make future predictions
        future_predictions = predictor.predict_future(days=30)

        # Plot future predictions for top 5 coins
        plt.figure(figsize=(15, 10))
        for i, symbol in enumerate(list(future_predictions.keys())[:5]):
            plt.subplot(2, 3, i + 1)
            plt.plot(future_predictions[symbol], marker='o')
            plt.title(f"30-Day Price Prediction for {symbol}")
            plt.xlabel("Days")
            plt.ylabel("Predicted Price")
            plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("❌ No valid data. Exiting.")