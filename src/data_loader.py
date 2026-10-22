"""Unit I - data collection.
Downloads prices with yfinance. If there is no internet / yfinance, or --offline
is used, it falls back to realistic SYNTHETIC data (GARCH-style) so the whole
project still runs end to end. You can also pass your own CSV
(columns: Date, Open, High, Low, Close, Volume)."""
import numpy as np
import pandas as pd

COLS = ["Open", "High", "Low", "Close", "Volume"]


def _synthetic(n=1500, seed=42, mu=0.0004, sigma=0.015, beta=1.0, market=None, start="2018-01-01"):
    rng = np.random.default_rng(seed)
    omega = sigma ** 2 * 0.02
    r = np.zeros(n)
    var = sigma ** 2
    for t in range(1, n):
        var = omega + 0.08 * r[t - 1] ** 2 + 0.90 * var          # GARCH(1,1) volatility clustering
        r[t] = mu + 0.04 * r[t - 1] + np.sqrt(var) * rng.standard_normal()
    if market is not None:
        r = r + beta * market[:n]
    close = 100 * np.exp(np.cumsum(r))
    openp = np.r_[close[0], close[:-1]] * (1 + 0.002 * rng.standard_normal(n))
    high = np.maximum(openp, close) * (1 + np.abs(0.004 * rng.standard_normal(n)))
    low = np.minimum(openp, close) * (1 - np.abs(0.004 * rng.standard_normal(n)))
    vol = rng.lognormal(15, 0.25, n) * (1 + 25 * np.abs(r))
    idx = pd.bdate_range(start=start, periods=n)
    df = pd.DataFrame({"Open": openp, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)
    df.index.name = "Date"
    return df


def load_stock(ticker="AAPL", start="2018-01-01", end=None, csv=None, offline=False, **syn):
    df, source = None, None
    if csv:
        df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date").sort_index()
        source = f"CSV file {csv}"
    elif not offline:
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                raise ValueError("empty download")
            source = f"Yahoo Finance ({ticker})"
        except Exception as e:
            print(f"[data] Could not download {ticker} ({e}). Using synthetic data.")
            df = None
    if df is None:
        df = _synthetic(start=start, **syn)
        source = "SYNTHETIC data (offline fallback)"
    df = df[COLS].dropna()
    df.attrs["source"] = source
    return df


def load_universe(tickers, start="2018-01-01", offline=False, n=1500):
    """Many stocks for the clustering step. Synthetic fallback creates 3 'types'
    (high-growth/high-vol, balanced, defensive) sharing one market factor."""
    rng = np.random.default_rng(0)
    market = 0.008 * rng.standard_normal(n)
    types = [(0.0008, 0.025, 1.3), (0.0004, 0.015, 1.0), (0.0002, 0.010, 0.5)]
    out = {}
    for i, t in enumerate(tickers):
        mu, sg, bt = types[i % 3]
        out[t] = load_stock(t, start=start, offline=offline, seed=100 + i, mu=mu, sigma=sg,
                            beta=bt, market=market, n=n)
    return out
