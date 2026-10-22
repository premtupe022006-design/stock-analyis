"""Unit V - neural networks: MLP (backpropagation), activation functions,
Rosenblatt perceptron and ADALINE (from scratch)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from .features import FEATURES, chrono_split
from .utils import save_fig, save_table, header


class Adaline:
    """ADAptive LInear NEuron: identity activation, batch gradient descent on squared error."""
    def __init__(self, lr=0.01, epochs=100):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y01):
        y = np.where(y01 == 1, 1, -1)
        self.w, self.b, self.cost = np.zeros(X.shape[1]), 0.0, []
        for _ in range(self.epochs):
            err = y - (X @ self.w + self.b)
            self.w += self.lr * X.T @ err / len(y); self.b += self.lr * err.mean()
            self.cost.append((err ** 2).mean() / 2)
        return self

    def predict(self, X):
        return (X @ self.w + self.b >= 0).astype(int)


def run(df, feats=FEATURES):
    header("UNIT V - NEURAL NETWORKS")
    train, test = chrono_split(df)
    sc = StandardScaler().fit(train[feats])
    Xtr, Xte = sc.transform(train[feats]), sc.transform(test[feats])
    ytr, yte = train["target_dir"].values, test["target_dir"].values

    rows = {}
    mlp = MLPClassifier((32, 16), activation="relu", solver="adam", alpha=1e-2,
                        max_iter=200, random_state=0).fit(Xtr, ytr)
    rows["MLP (32-16, ReLU)"] = mlp.score(Xte, yte)
    for act in ["identity", "logistic", "tanh"]:
        m = MLPClassifier((32, 16), activation=act, alpha=1e-2, max_iter=200, random_state=0).fit(Xtr, ytr)
        rows[f"MLP (32-16, {act})"] = m.score(Xte, yte)
    rows["Perceptron (Rosenblatt)"] = Perceptron(random_state=0).fit(Xtr, ytr).score(Xte, yte)
    ada = Adaline().fit(Xtr, ytr); rows["ADALINE"] = (ada.predict(Xte) == yte).mean()
    rows["Majority baseline"] = max(yte.mean(), 1 - yte.mean())
    res = pd.Series(rows, name="Test_Acc").to_frame()
    print(res.round(4)); save_table(res, "neural_net_results")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(mlp.loss_curve_); ax[0].set_title("MLP training loss (backpropagation)")
    ax[0].set_xlabel("epoch")
    ax[1].plot(ada.cost); ax[1].set_title("ADALINE cost per epoch"); ax[1].set_xlabel("epoch")
    save_fig("10_neural_network_training")
