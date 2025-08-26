import os
import pickle
import yfinance as yf
import pandas as pd
import xgboost as xgb
from indicators import add_indicators

MODELS_DIR = "models"

def train_model(symbol, model_file):
    data = yf.download(symbol, interval="1m", period="7d")
    data = add_indicators(data)
    data.dropna(inplace=True)

    # Target: 1 ако цената след 3 минути е по-висока, -1 ако е по-ниска, 0 ако е почти същата
    future_close = data["Close"].shift(-3)
    data["Target"] = 0
    data.loc[future_close > data["Close"], "Target"] = 1
    data.loc[future_close < data["Close"], "Target"] = -1

    X = data.drop(columns=["Target", "Open", "High", "Low", "Adj Close", "Volume"])
    y = data["Target"]

    train_size = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    model = xgb.XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
    model.fit(X_train, y_train)

    with open(model_file, "wb") as f:
        pickle.dump(model, f)

def load_model(asset):
    model_file = os.path.join(MODELS_DIR, f"{asset.replace('/', '')}_model.pkl")
    if not os.path.exists(model_file):
        print(f"Training model for {asset}...")
        symbol = {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X"}[asset]
        train_model(symbol, model_file)
    with open(model_file, "rb") as f:
        return pickle.load(f)

def get_ai_signal(symbol):
    asset = None
    for k, v in {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X"}.items():
        if v == symbol:
            asset = k
    model = load_model(asset)
    data = yf.download(symbol, interval="1m", period="1d")
    data = add_indicators(data)
    data.dropna(inplace=True)
    X_live = data.drop(columns=["Open", "High", "Low", "Adj Close", "Volume"])
    latest = X_live.iloc[[-1]]
    pred = model.predict(latest)[0]
    proba = model.predict_proba(latest).max()
    if pred == 1:
        return "BUY", proba
    elif pred == -1:
        return "SELL", proba
    else:
        return "NONE", proba
