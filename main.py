"""Stock Market Analysis & Prediction - ML syllabus project.
Run everything:   python main.py
Real data:        python main.py --ticker TSLA --start 2019-01-01
No internet:      python main.py --offline
Your own CSV:     python main.py --csv my_prices.csv
Some steps only:  python main.py --only eda classification backtest"""
import argparse
import warnings
warnings.filterwarnings("ignore")
from src import (data_loader, features, eda, feature_selection, classification, regression,
                 clustering, neural_net, ensembles, backtest)
from src.utils import header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--offline", action="store_true", help="use synthetic data")
    ap.add_argument("--skip-rl", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["eda", "features", "classification", "regression", "clustering",
                             "rules", "neural", "ensembles", "backtest"])
    a = ap.parse_args()
    want = lambda s: a.only is None or s in a.only

    raw = data_loader.load_stock(a.ticker, a.start, a.end, a.csv, a.offline)
    df = features.add_features(raw)
    F = features.FEATURES
    train, test = features.chrono_split(df)
    clf = None

    if want("eda"): eda.run(raw, a.ticker); eda.corr_heatmap(df, F)
    if want("features"): feature_selection.run(train, test, F)
    if want("classification") or want("backtest"): clf = classification.run(df, F)
    if want("regression"): regression.run(df, F)
    if want("clustering"): clustering.run_clustering(start=a.start, offline=a.offline)
    if want("rules"): clustering.association_rules(df)
    if want("neural"): neural_net.run(df, F)
    if want("ensembles"): ensembles.run(df, F, skip_rl=a.skip_rl)
    if want("backtest") and clf: backtest.run(df, clf["best_model"], F)
    header("DONE - see outputs/figures and outputs/tables")


if __name__ == "__main__":
    main()
