"""Unit III - feature engineering (technical indicators) + targets."""
import numpy as np
import pandas as pd

FEATURES = ["ret_1", "lag_ret_1", "lag_ret_2", "lag_ret_3", "ret_5",
            "sma5_ratio", "sma10_ratio", "sma20_ratio", "sma50_ratio",
            "macd", "macd_signal", "rsi_14", "bb_pctb",
            "volatility_10", "volatility_20", "vol_ratio", "hl_range", "gap"]


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def add_features(df):
    d = df.copy()
    c = d["Close"]
    d["ret_1"] = c.pct_change()
    for l in (1, 2, 3):
        d[f"lag_ret_{l}"] = d["ret_1"].shift(l)
    d["ret_5"] = c.pct_change(5)
    for w in (5, 10, 20, 50):
        d[f"sma{w}_ratio"] = c / c.rolling(w).mean() - 1
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["macd"] = (ema12 - ema26) / c
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["rsi_14"] = rsi(c)
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    d["bb_pctb"] = (c - (ma20 - 2 * sd20)) / (4 * sd20 + 1e-12)
    d["volatility_10"] = d["ret_1"].rolling(10).std()
    d["volatility_20"] = d["ret_1"].rolling(20).std()
    d["vol_ratio"] = d["Volume"] / d["Volume"].rolling(20).mean()
    d["hl_range"] = (d["High"] - d["Low"]) / c
    d["gap"] = d["Open"] / c.shift(1) - 1
    # targets = NEXT day (what we want to predict)
    d["target_ret"] = c.pct_change().shift(-1)
    d["target_dir"] = (d["target_ret"] > 0).astype(int)
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    return d


def chrono_split(df, test_size=0.2):
    """Time series must NOT be shuffled: train on the past, test on the future."""
    k = int(len(df) * (1 - test_size))
    return df.iloc[:k], df.iloc[k:]
