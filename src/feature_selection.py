"""Unit III - feature transformation, subset selection, curse of dimensionality."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from .utils import save_fig, save_table, header


def run(train, test, feats):
    header("UNIT III - FEATURE SELECTION, PCA, CURSE OF DIMENSIONALITY")
    X, y = train[feats], train["target_dir"]
    Xs = StandardScaler().fit_transform(X)
    f, p = f_classif(Xs, y)
    mi = mutual_info_classif(Xs, y, random_state=0)
    chi, _ = chi2(MinMaxScaler().fit_transform(X), y)
    rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=8).fit(Xs, y)
    tab = pd.DataFrame({"F_score": f, "F_pvalue": p, "MutualInfo": mi, "Chi2": chi,
                        "RFE_selected": rfe.support_}, index=feats).sort_values("MutualInfo", ascending=False)
    print(tab.round(4))
    save_table(tab, "feature_selection")
    selected = [fe for fe, s in zip(feats, rfe.support_) if s]
    print("\nRFE-selected features:", selected)

    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    print(f"PCA: {np.argmax(cum >= .95) + 1} of {len(feats)} components explain 95% variance")
    plt.figure(figsize=(6, 4)); plt.plot(range(1, len(cum) + 1), cum, "o-"); plt.axhline(.95, ls="--", c="r")
    plt.xlabel("components"); plt.ylabel("cumulative variance"); plt.title("PCA explained variance")
    save_fig("04_pca_variance")

    # Curse of dimensionality: add pure-noise features, watch k-NN degrade
    sc = StandardScaler().fit(X)
    Xtr, Xte = sc.transform(X), sc.transform(test[feats])
    rng = np.random.default_rng(0)
    rows = []
    for k in [0, 10, 50, 100, 200]:
        a = np.hstack([Xtr, rng.standard_normal((len(Xtr), k))])
        b = np.hstack([Xte, rng.standard_normal((len(Xte), k))])
        acc = KNeighborsClassifier(15).fit(a, y).score(b, test["target_dir"])
        rows.append((k, a.shape[1], acc))
    cd = pd.DataFrame(rows, columns=["noise_features", "total_dims", "kNN_test_acc"])
    print("\nCurse of dimensionality (k-NN):\n", cd.round(4).to_string(index=False))
    plt.figure(figsize=(6, 4)); plt.plot(cd.total_dims, cd.kNN_test_acc, "o-")
    plt.xlabel("dimensions"); plt.ylabel("k-NN accuracy"); plt.title("Curse of dimensionality")
    save_fig("05_curse_of_dimensionality")
    return selected
