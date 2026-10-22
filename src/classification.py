"""Units II & IV - classification: will tomorrow close UP (1) or DOWN (0)?
Holdout, time-series CV, bootstrap, lazy vs eager, Naive Bayes (Bayesian),
k-NN, Decision Tree, Random Forest, SVM, Logistic Regression (MLE)."""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from .features import FEATURES, chrono_split
from .utils import save_fig, save_table, header


def get_models():
    return {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression (MLE)": LogisticRegression(max_iter=1000),
        "k-NN (lazy learner)": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree (eager)": DecisionTreeClassifier(max_depth=4, random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10,
                                                random_state=0, n_jobs=-1),
        "SVM (RBF)": SVC(probability=True, random_state=0),
    }


def run(df, feats=FEATURES):
    header("UNITS II & IV - CLASSIFICATION (UP/DOWN)")
    train, test = chrono_split(df)                       # holdout: first 80% train, last 20% test
    Xtr, ytr, Xte, yte = train[feats], train["target_dir"], test[feats], test["target_dir"]
    print(f"Holdout split -> train={len(train)}  test={len(test)}  (class balance train: {ytr.mean():.2%} up)")
    tscv = TimeSeriesSplit(n_splits=5)
    rows, fitted, probas = [], {}, {}
    for name, m in get_models().items():
        p = make_pipeline(StandardScaler(), m)
        cv = cross_val_score(p, Xtr, ytr, cv=tscv, scoring="accuracy")
        t0 = time.perf_counter(); p.fit(Xtr, ytr); fit_t = time.perf_counter() - t0
        t0 = time.perf_counter(); pred = p.predict(Xte); pred_t = time.perf_counter() - t0
        proba = p.predict_proba(Xte)[:, 1]
        fitted[name], probas[name] = p, proba
        rows.append({"Model": name, "CV_Acc": cv.mean(), "CV_Std": cv.std(),
                     "Test_Acc": accuracy_score(yte, pred),
                     "Precision": precision_score(yte, pred, zero_division=0),
                     "Recall": recall_score(yte, pred, zero_division=0),
                     "F1": f1_score(yte, pred, zero_division=0),
                     "ROC_AUC": roc_auc_score(yte, proba),
                     "Fit_sec": fit_t, "Predict_sec": pred_t})
    res = pd.DataFrame(rows).set_index("Model")
    print(res.round(4))
    save_table(res, "classification_results")
    print("\nLazy vs eager: k-NN fits instantly but predicts slowly; tree/SVM spend time fitting, predict fast.")

    cand = res.drop("Baseline (majority)")
    best = cand["CV_Acc"].idxmax()                       # choose by CV (not by test) to avoid leakage
    print(f"\nBest model by time-series CV accuracy: {best}")

    # ROC curves + confusion matrix of best
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for n, pr in probas.items():
        if n.startswith("Baseline"): continue
        fpr, tpr, _ = roc_curve(yte, pr); ax[0].plot(fpr, tpr, label=f"{n} ({roc_auc_score(yte, pr):.2f})")
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_title("ROC curves (test)"); ax[0].legend(fontsize=7)
    ConfusionMatrixDisplay(confusion_matrix(yte, fitted[best].predict(Xte)),
                           display_labels=["Down", "Up"]).plot(ax=ax[1], colorbar=False)
    ax[1].set_title(f"Confusion matrix - {best}")
    save_fig("06_roc_confusion")

    # Standard shuffled k-fold vs time-series CV (shows look-ahead leakage)
    p = make_pipeline(StandardScaler(), RandomForestClassifier(100, max_depth=6, random_state=0, n_jobs=-1))
    kf = cross_val_score(p, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=0)).mean()
    ts = cross_val_score(p, Xtr, ytr, cv=tscv).mean()
    print(f"\nk-fold (shuffled) acc={kf:.4f}  vs  TimeSeriesSplit acc={ts:.4f}  "
          "(shuffled k-fold peeks into the future -> optimistic)")

    # Bootstrap sampling with out-of-bag evaluation
    rng = np.random.default_rng(0); idx = np.arange(len(Xtr)); accs = []
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    for _ in range(100):
        b = rng.choice(idx, len(idx)); oob = np.setdiff1d(idx, b)
        lr.fit(Xtr.iloc[b], ytr.iloc[b]); accs.append(lr.score(Xtr.iloc[oob], ytr.iloc[oob]))
    lo, hi = np.percentile(accs, [2.5, 97.5])
    print(f"Bootstrap (100 resamples, OOB) logistic acc = {np.mean(accs):.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    # Hyper-parameter tuning
    grid = GridSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1),
                        {"n_estimators": [100, 200], "max_depth": [3, 6, 10], "min_samples_leaf": [5, 20]},
                        cv=TimeSeriesSplit(3), scoring="accuracy").fit(Xtr, ytr)
    print(f"GridSearchCV best RF params: {grid.best_params_}  CV acc={grid.best_score_:.4f}  "
          f"Test acc={grid.score(Xte, yte):.4f}")

    imp = pd.Series(grid.best_estimator_.feature_importances_, index=feats).sort_values()
    imp.plot.barh(figsize=(7, 6), title="Random Forest feature importance"); save_fig("07_feature_importance")
    return {"results": res, "best_name": best, "best_model": fitted[best]}
