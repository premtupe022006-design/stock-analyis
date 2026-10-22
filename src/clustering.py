"""Unit V - unsupervised learning: k-Means, k-Medoids, hierarchical, DBSCAN,
and association rules. Stocks are grouped by risk/return profile."""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
from .data_loader import load_universe
from .utils import save_fig, save_table, header

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC",
            "XOM", "CVX", "JNJ", "PFE", "KO", "PEP", "WMT"]


def kmedoids(X, k, max_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    D = pairwise_distances(X)
    med = rng.choice(len(X), k, replace=False)
    for _ in range(max_iter):
        labels = D[:, med].argmin(1)
        new = med.copy()
        for j in range(k):
            mem = np.where(labels == j)[0]
            if len(mem): new[j] = mem[D[np.ix_(mem, mem)].sum(1).argmin()]
        if set(new) == set(med): break
        med = new
    return D[:, med].argmin(1), med


def run_clustering(tickers=UNIVERSE, start="2018-01-01", offline=False):
    header("UNIT V - CLUSTERING STOCKS BY RISK/RETURN PROFILE")
    data = load_universe(tickers, start, offline)
    print("Data source example:", next(iter(data.values())).attrs["source"])
    rets = pd.DataFrame({t: d["Close"].pct_change() for t, d in data.items()}).dropna()
    mkt = rets.mean(axis=1)
    prof = {}
    for t in rets:
        r = rets[t]; px = data[t]["Close"].loc[rets.index]
        prof[t] = {"ann_return": r.mean() * 252, "ann_vol": r.std() * np.sqrt(252),
                   "sharpe": r.mean() / r.std() * np.sqrt(252), "skew": r.skew(), "kurtosis": r.kurtosis(),
                   "max_drawdown": (px / px.cummax() - 1).min(), "beta": np.cov(r, mkt)[0, 1] / mkt.var()}
    P = pd.DataFrame(prof).T
    X = StandardScaler().fit_transform(P)

    sil = {k: silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X)) for k in range(2, min(7, len(P)))}
    k = max(sil, key=sil.get)
    print("Silhouette by k:", {a: round(b, 3) for a, b in sil.items()}, "-> k =", k)
    P["KMeans"] = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    P["KMedoids"], med = kmedoids(X, k)
    P["Hierarchical"] = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    P["DBSCAN"] = DBSCAN(eps=1.8, min_samples=2).fit_predict(X)      # -1 = noise/outlier
    print(P.round(3).to_string())
    print("\nCluster profiles (KMeans mean):\n",
          P.groupby("KMeans")[["ann_return", "ann_vol", "sharpe", "beta", "max_drawdown"]].mean().round(3))
    print("k-Medoids representative stocks:", list(P.index[med]))
    save_table(P, "stock_clusters")

    Z = PCA(2).fit_transform(X)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].scatter(Z[:, 0], Z[:, 1], c=P["KMeans"], cmap="viridis", s=80)
    for i, t in enumerate(P.index): ax[0].annotate(t, Z[i], fontsize=8)
    ax[0].set_title("K-Means clusters (PCA 2-D)")
    dendrogram(linkage(X, "ward"), labels=list(P.index), ax=ax[1]); ax[1].set_title("Hierarchical dendrogram (Ward)")
    save_fig("09_clustering")


def association_rules(df, min_support=0.05, top=10):
    """Market-pattern mining: antecedent (1-2 conditions today) -> next-day Up/Down.
    support, confidence and lift computed with plain pandas (same maths as Apriori)."""
    header("UNIT V - ASSOCIATION RULES")
    q = lambda c, p: df[c].quantile(p)
    T = pd.DataFrame({
        "RSI_low": df["rsi_14"] <= q("rsi_14", .3), "RSI_high": df["rsi_14"] >= q("rsi_14", .7),
        "Today_down": df["ret_1"] < 0, "Today_up": df["ret_1"] >= 0,
        "Volume_spike": df["vol_ratio"] > 1.3, "High_volatility": df["volatility_10"] >= q("volatility_10", .7),
        "Above_SMA50": df["sma50_ratio"] > 0, "MACD_positive": df["macd"] > 0,
        "Gap_down": df["gap"] <= q("gap", .2)})
    cons = {"NEXT_UP": df["target_dir"] == 1, "NEXT_DOWN": df["target_dir"] == 0}
    rows = []
    for size in (1, 2):
        for combo in itertools.combinations(T.columns, size):
            a = T[list(combo)].all(axis=1)
            if a.mean() < min_support: continue
            for cn, c in cons.items():
                sup = (a & c).mean()
                if sup < min_support: continue
                conf = sup / a.mean()
                rows.append({"antecedent": " & ".join(combo), "consequent": cn, "support": sup,
                             "confidence": conf, "lift": conf / c.mean()})
    R = pd.DataFrame(rows).sort_values("lift", ascending=False)
    print(R.head(top).round(3).to_string(index=False))
    print("\nLift ~ 1.0 means the pattern gives almost no edge (markets are hard to predict).")
    save_table(R, "association_rules")
