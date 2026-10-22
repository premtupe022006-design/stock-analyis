"""Turn the best classifier into a long/flat strategy and compare with buy & hold."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .features import FEATURES, chrono_split
from .utils import save_fig, save_table, header


def _stats(ret):
    eq = (1 + ret).cumprod()
    return {"Total_Return": eq.iloc[-1] - 1,
            "Ann_Return": ret.mean() * 252, "Ann_Vol": ret.std() * np.sqrt(252),
            "Sharpe": ret.mean() / (ret.std() + 1e-12) * np.sqrt(252),
            "Max_Drawdown": (eq / eq.cummax() - 1).min()}


def run(df, model, feats=FEATURES, cost=0.001):
    header("BACKTEST - ML SIGNAL vs BUY & HOLD (test period only)")
    _, test = chrono_split(df)
    sig = pd.Series(model.predict(test[feats]), index=test.index)      # 1 = long, 0 = cash
    trades = sig.diff().abs().fillna(sig.abs())
    strat = sig * test["target_ret"] - trades * cost
    res = pd.DataFrame({"ML strategy": _stats(strat), "Buy & Hold": _stats(test["target_ret"])}).T
    print(res.round(4)); save_table(res, "backtest")
    (1 + strat).cumprod().plot(label="ML strategy", figsize=(9, 4))
    (1 + test["target_ret"]).cumprod().plot(label="Buy & Hold")
    plt.legend(); plt.title(f"Equity curve (cost {cost:.1%} per trade)"); save_fig("14_backtest")
