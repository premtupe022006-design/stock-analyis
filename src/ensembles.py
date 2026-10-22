"""Unit VI - ensembles, regularization, active learning, generative model,
reinforcement learning (Q-learning trader)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from .features import FEATURES, chrono_split
from .utils import save_fig, save_table, header


def ensembles(train, test, feats):
    header("UNIT VI - ENSEMBLE LEARNING")
    Xtr, ytr, Xte, yte = train[feats], train["target_dir"], test[feats], test["target_dir"]
    models = {
        "Single Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=0),
        "Bagging": BaggingClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=100, random_state=0),
        "Random Forest": RandomForestClassifier(200, max_depth=6, min_samples_leaf=10, random_state=0),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=.05, random_state=0),
        "Voting (soft)": VotingClassifier([
            ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
            ("rf", RandomForestClassifier(150, max_depth=6, min_samples_leaf=10, random_state=0)),
            ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=.05, random_state=0))],
            voting="soft"),
    }
    rows = {}
    for n, m in models.items():
        cv = cross_val_score(m, Xtr, ytr, cv=TimeSeriesSplit(5)).mean()
        rows[n] = {"CV_Acc": cv, "Test_Acc": m.fit(Xtr, ytr).score(Xte, yte)}
    res = pd.DataFrame(rows).T; print(res.round(4)); save_table(res, "ensemble_results")


def regularization(train, feats):
    header("UNIT VI - REGULARIZATION (Ridge vs Lasso paths)")
    X = StandardScaler().fit_transform(train[feats]); y = train["target_ret"].values
    alphas = np.logspace(-6, -2, 30)
    lasso = np.array([Lasso(alpha=a, max_iter=10000).fit(X, y).coef_ for a in alphas])
    ridge = np.array([Ridge(alpha=a * 1e5).fit(X, y).coef_ for a in alphas])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(alphas, ridge); ax[0].set_xscale("log"); ax[0].set_title("Ridge: coefficients shrink")
    ax[1].plot(alphas, lasso); ax[1].set_xscale("log"); ax[1].set_title("Lasso: coefficients hit exactly 0")
    save_fig("11_regularization_paths")
    nz = (np.abs(lasso) > 1e-12).sum(1)
    print(f"Lasso non-zero coefficients: {nz[0]} at smallest alpha -> {nz[-1]} at largest alpha")


def active_learning(train, test, feats, rounds=15, batch=20):
    header("UNIT VI - ACTIVE LEARNING (uncertainty sampling vs random)")
    sc = StandardScaler().fit(train[feats])
    X, y = sc.transform(train[feats]), train["target_dir"].values
    Xt, yt = sc.transform(test[feats]), test["target_dir"].values
    rng = np.random.default_rng(0)
    curves = {}
    for strategy in ("uncertainty", "random"):
        lab = list(rng.choice(len(X), 30, replace=False)); acc = []
        for _ in range(rounds):
            m = LogisticRegression(max_iter=1000).fit(X[lab], y[lab]); acc.append(m.score(Xt, yt))
            pool = np.setdiff1d(np.arange(len(X)), lab)
            if strategy == "uncertainty":
                pick = pool[np.argsort(np.abs(m.predict_proba(X[pool])[:, 1] - .5))[:batch]]
            else:
                pick = rng.choice(pool, batch, replace=False)
            lab += list(pick)
        curves[strategy] = acc
    for k, v in curves.items(): plt.plot(range(len(v)), v, label=k)
    plt.xlabel("query round"); plt.ylabel("test accuracy"); plt.legend(); plt.title("Active learning")
    save_fig("12_active_learning")
    print({k: round(v[-1], 4) for k, v in curves.items()}, "(final accuracy after labelling "
          f"{30 + rounds * batch} samples)")


def generative(train):
    header("UNIT VI - GENERATIVE MODEL (Gaussian Mixture for returns)")
    r = train["ret_1"].values.reshape(-1, 1)
    gmm = GaussianMixture(3, random_state=0).fit(r)
    s, _ = gmm.sample(len(r))
    from scipy.stats import kurtosis
    print(f"Real: std={r.std():.4f} kurt={kurtosis(r.ravel()):.2f} | Synthetic: std={s.std():.4f} kurt={kurtosis(s.ravel()):.2f}")
    plt.hist(r, 60, density=True, alpha=.5, label="real"); plt.hist(s, 60, density=True, alpha=.5, label="GMM samples")
    plt.legend(); plt.title("Generative model of daily returns"); save_fig("13_gmm_generative")


def reinforcement(train, test, episodes=30, cost=0.0005):
    header("UNIT VI - REINFORCEMENT LEARNING (tabular Q-learning, long/flat)")
    def states(d):
        rb = np.digitize(d["rsi_14"], [40, 60]); tr = (d["sma20_ratio"] > 0).astype(int)
        return (rb * 2 + tr).values
    S_tr, R_tr, S_te, R_te = states(train), train["target_ret"].values, states(test), test["target_ret"].values
    Q = np.zeros((6, 2, 2))      # [state, prev_position, action]
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        eps, pos = max(.05, 1 - ep / (episodes * .7)), 0
        for t in range(len(S_tr)):
            a = rng.integers(2) if rng.random() < eps else int(Q[S_tr[t], pos].argmax())
            reward = a * R_tr[t] - cost * abs(a - pos) * 1.0
            nxt = Q[S_tr[min(t + 1, len(S_tr) - 1)], a].max()
            Q[S_tr[t], pos, a] += .1 * (reward * 100 + .9 * nxt - Q[S_tr[t], pos, a])
            pos = a
    pos, eq = 0, [1.0]
    for t in range(len(S_te)):
        a = int(Q[S_te[t], pos].argmax()); eq.append(eq[-1] * (1 + a * R_te[t] - cost * abs(a - pos))); pos = a
    bh = np.cumprod(1 + R_te)
    print(f"Q-learning agent final equity={eq[-1]:.3f}  vs buy&hold={bh[-1]:.3f}  (test period)")
    print("Educational demo only - not trading advice.")


def run(df, feats=FEATURES, skip_rl=False):
    train, test = chrono_split(df)
    ensembles(train, test, feats); regularization(train, feats)
    active_learning(train, test, feats); generative(train)
    if not skip_rl: reinforcement(train, test)
