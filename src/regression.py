"""Unit IV - regression: predict NEXT-DAY RETURN (then convert to price).
Simple / multiple / polynomial / Ridge / Lasso / ElasticNet + assumption checks."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .features import FEATURES, chrono_split
from .utils import save_fig, save_table, header


def vif_table(X):
    Xs = (X - X.mean()) / X.std()
    out = {}
    for c in X.columns:
        o = Xs.drop(columns=c)
        r2 = LinearRegression().fit(o, Xs[c]).score(o, Xs[c])
        out[c] = 1 / (1 - r2 + 1e-12)
    return pd.Series(out, name="VIF").sort_values(ascending=False)


def run(df, feats=FEATURES):
    header("UNIT IV - REGRESSION (next-day return)")
    train, test = chrono_split(df)
    ytr, yte = train["target_ret"], test["target_ret"]
    poly_cols = ["ret_1", "rsi_14", "sma20_ratio", "macd", "volatility_10"]
    models = {
        "Baseline (predict 0)": (feats, DummyRegressor(strategy="constant", constant=0.0)),
        "Simple LR (rsi_14)": (["rsi_14"], make_pipeline(StandardScaler(), LinearRegression())),
        "Multiple LR": (feats, make_pipeline(StandardScaler(), LinearRegression())),
        "Polynomial (deg 2)": (poly_cols, make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression())),
        "Ridge (L2)": (feats, make_pipeline(StandardScaler(), Ridge(alpha=200))),
        "Lasso (L1)": (feats, make_pipeline(StandardScaler(), Lasso(alpha=2e-4))),
        "ElasticNet": (feats, make_pipeline(StandardScaler(), ElasticNet(alpha=2e-4, l1_ratio=.5))),
        "Random Forest Reg": (feats, RandomForestRegressor(200, max_depth=5, min_samples_leaf=20,
                                                            random_state=0, n_jobs=-1)),
    }
    actual_price = test["Close"] * (1 + yte)
    rows = []
    for n, (cols, m) in models.items():
        m.fit(train[cols], ytr); pred = m.predict(test[cols])
        rows.append({"Model": n, "MSE": mean_squared_error(yte, pred),
                     "RMSE": np.sqrt(mean_squared_error(yte, pred)), "MAE": mean_absolute_error(yte, pred),
                     "R2": r2_score(yte, pred),
                     "Price_RMSE": np.sqrt(mean_squared_error(actual_price, test["Close"] * (1 + pred))),
                     "Dir_Acc": (np.mean(np.sign(pred) == np.sign(yte)) if np.any(pred != 0) else np.nan)})
    res = pd.DataFrame(rows).set_index("Model")
    print(res.round(6))
    print("\nNote: R2 near/below 0 is NORMAL - daily returns are very close to random noise.")
    save_table(res, "regression_results")

    # Assumption checks on multiple linear regression
    ols = make_pipeline(StandardScaler(), LinearRegression()).fit(train[feats], ytr)
    fitted_, resid = ols.predict(train[feats]), ytr - ols.predict(train[feats])
    dw = np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)
    sh = stats.shapiro(resid.sample(min(len(resid), 4000), random_state=0))
    vif = vif_table(train[feats])
    print(f"\nAssumptions: Durbin-Watson={dw:.3f} (2 = no autocorrelation) | Shapiro p={sh.pvalue:.3g} "
          f"({'residuals not normal' if sh.pvalue < .05 else 'normal'})")
    print("VIF (>10 = multicollinearity):\n", vif.round(2).head(8).to_string())
    save_table(vif.to_frame(), "vif")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].scatter(fitted_, resid, s=6, alpha=.5); ax[0].axhline(0, c="r"); ax[0].set_title("Residuals vs fitted")
    stats.probplot(resid, plot=ax[1]); ax[1].set_title("Residual Q-Q")
    pred = ols.predict(test[feats]); ax[2].plot(yte.values[:120], label="actual"); ax[2].plot(pred[:120], label="pred")
    ax[2].legend(); ax[2].set_title("Next-day return: actual vs predicted (first 120 test days)")
    save_fig("08_regression_diagnostics")
