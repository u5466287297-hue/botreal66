import pandas as pd
import numpy as np

def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    # SMA 20 и SMA 50
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["SMA50"] = data["Close"].rolling(window=50).mean()

    # RSI (14)
    delta = data["Close"].diff()
    gain = np.where(delta > 0, delta, 0).flatten()
    loss = np.where(delta < 0, -delta, 0).flatten()

    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()

    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = data["Close"].ewm(span=12, adjust=False).mean()
    exp2 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = exp1 - exp2
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    return data
