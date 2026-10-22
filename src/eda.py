"""Unit I (exploring data) + Unit III (probability, CLT, hypothesis testing)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from .utils import save_fig, save_table, header


def run(raw, ticker):
    header(f"UNIT I/III - EXPLORATORY DATA ANALYSIS  [{raw.attrs.get('source')}]")
    print(raw.describe().T.round(3))
    print("\nMissing values:\n", raw.isna().sum().to_string())
    print(f"Rows: {len(raw)}   Range: {raw.index[0].date()} -> {raw.index[-1].date()}")
    r = raw["Close"].pct_change().dropna()

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    raw["Close"].plot(ax=ax[0], label="Close")
    raw["Close"].rolling(50).mean().plot(ax=ax[0], label="SMA50")
    raw["Close"].rolling(200).mean().plot(ax=ax[0], label="SMA200")
    ax[0].set_title(f"{ticker} price"); ax[0].legend()
    ax[1].hist(r, bins=60, density=True, alpha=.7)
    xs = np.linspace(r.min(), r.max(), 200)
    ax[1].plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), "r", label="Normal fit")
    ax[1].set_title("Daily returns vs Normal"); ax[1].legend()
    stats.probplot(r, dist="norm", plot=ax[2]); ax[2].set_title("Q-Q plot (fat tails?)")
    save_fig("01_eda_price_returns")

    jb = stats.jarque_bera(r)
    print(f"\nSkew={r.skew():.3f}  Excess kurtosis={r.kurtosis():.3f}  "
          f"Jarque-Bera p={jb.pvalue:.4g}  ->  {'NOT normal' if jb.pvalue < .05 else 'looks normal'}")

    # Central limit theorem: means of samples of size 30 are ~Normal even if returns are not
    rng = np.random.default_rng(0)
    means = [rng.choice(r.values, 30).mean() for _ in range(3000)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(r, bins=50); ax[0].set_title("Population: daily returns")
    ax[1].hist(means, bins=50, density=True)
    xs = np.linspace(min(means), max(means), 200)
    ax[1].plot(xs, stats.norm.pdf(xs, np.mean(means), np.std(means)), "r")
    ax[1].set_title("CLT: distribution of sample means (n=30)")
    save_fig("02_central_limit_theorem")
    print(f"CLT: std of sample means={np.std(means):.5f}  vs sigma/sqrt(30)={r.std()/np.sqrt(30):.5f}")

    # Hypothesis tests
    t1 = stats.ttest_1samp(r, 0)
    print(f"\nH0: mean daily return = 0  ->  t={t1.statistic:.3f}, p={t1.pvalue:.4f}")
    mon = r[r.index.dayofweek == 0]; other = r[r.index.dayofweek != 0]
    t2 = stats.ttest_ind(mon, other, equal_var=False)
    print(f"H0: Monday returns = other days (Welch) -> t={t2.statistic:.3f}, p={t2.pvalue:.4f}")
    summary = pd.DataFrame({"test": ["mean return = 0", "Monday = other days"],
                            "t_stat": [t1.statistic, t2.statistic], "p_value": [t1.pvalue, t2.pvalue]})
    save_table(summary, "hypothesis_tests")


def corr_heatmap(df, feats):
    c = df[feats + ["target_ret"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=90)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns)
    plt.colorbar(im); ax.set_title("Feature correlation heatmap")
    save_fig("03_correlation_heatmap")
