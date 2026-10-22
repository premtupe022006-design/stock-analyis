# Stock Market Analysis & Prediction (ML Syllabus Project)

Predict next-day stock direction (classification) and return (regression), group stocks
by risk profile (clustering), mine market patterns (association rules), and test the
signals with a backtest. Every syllabus unit is covered.

## Quick start
```bash
pip install -r requirements.txt
python main.py                      # AAPL from Yahoo Finance (needs internet)
python main.py --ticker TSLA --start 2019-01-01
python main.py --offline            # synthetic data, no internet needed
python main.py --csv prices.csv     # your own CSV (Date,Open,High,Low,Close,Volume)
python main.py --only eda classification backtest
streamlit run app.py                # optional dashboard
```
Outputs: `outputs/figures/*.png` (14 plots) and `outputs/tables/*.csv`.

## Syllabus mapping
| Unit | Topic | File / function |
|---|---|---|
| I | Data types, structure, missing values, preprocessing, scaling | `data_loader.py`, `eda.py`, StandardScaler in pipelines |
| II | Holdout, k-fold, TimeSeriesSplit, bootstrap, lazy vs eager, metrics, tuning | `classification.py` |
| III | Feature engineering (SMA, EMA/MACD, RSI, Bollinger...), RFE/chi2/MI, PCA, curse of dimensionality | `features.py`, `feature_selection.py` |
| III | Probability, distributions, CLT, hypothesis tests (t-test, Jarque-Bera) | `eda.py` |
| IV | Naive Bayes, k-NN, Decision Tree, Random Forest, SVM, Logistic Regression (MLE) | `classification.py` |
| IV | Simple/multiple/polynomial regression, VIF, residual & Durbin-Watson checks | `regression.py` |
| V | k-Means, k-Medoids (from scratch), hierarchical + dendrogram, DBSCAN | `clustering.py` |
| V | Association rules (support, confidence, lift) | `clustering.py::association_rules` |
| V | MLP + backpropagation, activation functions, Perceptron, ADALINE (from scratch) | `neural_net.py` |
| VI | Bagging, AdaBoost, Gradient Boosting, Voting | `ensembles.py` |
| VI | Regularization (Ridge/Lasso paths), active learning, generative model (GMM), reinforcement learning (Q-learning) | `ensembles.py` |
| - | Strategy backtest vs buy & hold | `backtest.py` |

## Important notes for your report
* **No shuffling:** time series are split chronologically; `TimeSeriesSplit` replaces plain k-fold
  (the code shows shuffled k-fold giving optimistic scores = data leakage).
* **Expect ~50-55% accuracy.** Daily returns are close to random; R2 near 0 and lift near 1 are
  genuine findings, not bugs. Discuss this honestly in your conclusion.
* The synthetic fallback data is almost pure noise by design, so results on it hover near 50%.
  Run with real data for your final report.
* Educational project - not financial advice.

## Suggested report outline
Objective, Data, EDA, Feature engineering, Models and validation, Results table, Backtest,
Limitations (efficient markets, transaction costs, regime change), Conclusion.
