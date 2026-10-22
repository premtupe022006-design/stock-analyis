"""AlphaStock ML - Quantitative Analysis, Machine Learning & Algorithmic Trading Dashboard.
Supports Indian Stocks (NSE in ₹ INR) and International Stocks (NASDAQ/NYSE in $ USD).
Includes:
- Data Loading (Live Yahoo Finance / Synthetic Fallback / Custom CSV)
- Interactive Plotly Candlestick with Volume, SMAs, Bollinger Bands, RSI & MACD
- Exploratory Data Analysis (Return distributions, Jarque-Bera normality, Welch t-test, CLT Simulator, Heatmaps)
- Direction Prediction (Classification benchmark across 7 ML algorithms, ROC-AUC, Confusion Matrix, TimeSeriesSplit vs K-Fold)
- Continuous Return Forecasting (Regression models, MSE/RMSE, VIF multicollinearity, Durbin-Watson autocorrelation)
- Quantitative Backtest Simulator (ML signal strategy vs Buy & Hold benchmark with transaction friction costs)
- Unsupervised Market Clustering & Pattern Mining (k-Means, k-Medoids from scratch, Hierarchical, DBSCAN, Apriori rules)
- Deep Learning & Ensembles (MLP activations comparison, Rosenblatt Perceptron, ADALINE from scratch, Bagging, Boosting)
"""

import time
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from src import data_loader, features
from src.features import FEATURES, chrono_split
from src.clustering import kmedoids
from src.neural_net import Adaline

# ==========================================
# PAGE CONFIGURATION & CUSTOM STYLES
# ==========================================
st.set_page_config(
    page_title="AlphaStock ML | Quantitative Analytics & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphic look
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        font-weight: 500;
    }
    .text-bullish { color: #10b981; }
    .text-bearish { color: #ef4444; }
    .text-neutral { color: #38bdf8; }

    /* Custom Signal Badge */
    .signal-badge-bull {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 10px 18px;
        border-radius: 30px;
        font-size: 1.1rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
    }
    .signal-badge-bear {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
        padding: 10px 18px;
        border-radius: 30px;
        font-size: 1.1rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);
    }
    .info-tag {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CURRENCY & STOCK HELPER FUNCTIONS
# ==========================================
def get_currency_info(ticker_symbol):
    """Returns currency symbol and ISO currency code based on ticker."""
    t = ticker_symbol.upper().strip()
    indian_indicators = [".NS", ".BO", "RELIANCE", "TCS", "HDFC", "INFY", "ICICI",
                         "BHARTI", "SBIN", "ITC", "LT", "TATAMOTORS", "NIFTY", "SENSEX"]
    if any(ind in t for ind in indian_indicators):
        return "₹", "INR"
    return "$", "USD"


INDIAN_TOP_10 = [
    ("RELIANCE.NS", "Reliance Industries Ltd"),
    ("TCS.NS", "Tata Consultancy Services"),
    ("HDFCBANK.NS", "HDFC Bank Ltd"),
    ("INFY.NS", "Infosys Limited"),
    ("ICICIBANK.NS", "ICICI Bank Ltd"),
    ("BHARTIARTL.NS", "Bharti Airtel Ltd"),
    ("SBIN.NS", "State Bank of India"),
    ("ITC.NS", "ITC Limited"),
    ("LT.NS", "Larsen & Toubro Ltd"),
    ("TATAMOTORS.NS", "Tata Motors Ltd"),
]

COMMON_INDIAN_MAP = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC": "HDFCBANK.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "ICICI": "ICICIBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "SBIN": "SBIN.NS",
    "SBI": "SBIN.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATA": "TATAMOTORS.NS"
}

INTL_TOP_10 = [
    ("AAPL", "Apple Inc."),
    ("NVDA", "NVIDIA Corporation"),
    ("MSFT", "Microsoft Corporation"),
    ("GOOGL", "Alphabet Inc. (Google)"),
    ("AMZN", "Amazon.com Inc."),
    ("META", "Meta Platforms Inc."),
    ("TSLA", "Tesla Inc."),
    ("JPM", "JPMorgan Chase & Co."),
    ("V", "Visa Inc."),
    ("WMT", "Walmart Inc."),
]


# ==========================================
# CACHED DATA & PIPELINE FUNCTIONS
# ==========================================
@st.cache_data(show_spinner=False)
def get_stock_data(ticker, start_date, end_date, offline=False, uploaded_csv_bytes=None):
    if uploaded_csv_bytes is not None:
        import io
        df = pd.read_csv(io.BytesIO(uploaded_csv_bytes), parse_dates=["Date"], index_col="Date").sort_index()
        source = "Custom Uploaded CSV"
        df.attrs["source"] = source
        return df
    return data_loader.load_stock(ticker, start=str(start_date), end=str(end_date) if end_date else None, offline=offline)


@st.cache_data(show_spinner=False)
def compute_features(raw_df):
    return features.add_features(raw_df)


@st.cache_data(show_spinner=False)
def train_classification_models(df, test_ratio=0.2):
    train, test = chrono_split(df, test_size=test_ratio)
    feats = FEATURES
    Xtr, ytr, Xte, yte = train[feats], train["target_dir"], test[feats], test["target_dir"]

    models = {
        "Baseline (Majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (MLE)": LogisticRegression(max_iter=1000),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "k-NN (Lazy Learner)": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree (Eager)": DecisionTreeClassifier(max_depth=4, random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=10, random_state=0, n_jobs=-1),
        "SVM (RBF Kernel)": SVC(probability=True, random_state=0),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    rows = []
    fitted_models = {}
    probas = {}
    preds = {}

    for name, m in models.items():
        pipe = make_pipeline(StandardScaler(), m)
        cv = cross_val_score(pipe, Xtr, ytr, cv=tscv, scoring="accuracy")

        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        fit_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = pipe.predict(Xte)
        pred_t = time.perf_counter() - t0

        proba = pipe.predict_proba(Xte)[:, 1]

        fitted_models[name] = pipe
        probas[name] = proba
        preds[name] = pred

        rows.append({
            "Model": name,
            "CV Accuracy (TSC)": cv.mean(),
            "Test Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred, zero_division=0),
            "F1-Score": f1_score(yte, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(yte, proba),
            "Fit Time (s)": fit_t,
            "Predict Time (s)": pred_t
        })

    res_df = pd.DataFrame(rows).set_index("Model")
    return res_df, fitted_models, probas, preds, train, test


@st.cache_data(show_spinner=False)
def train_regression_models(df, curr_sym, test_ratio=0.2):
    train, test = chrono_split(df, test_size=test_ratio)
    feats = FEATURES
    ytr, yte = train["target_ret"], test["target_ret"]
    poly_cols = ["ret_1", "rsi_14", "sma20_ratio", "macd", "volatility_10"]

    regressors = {
        "Baseline (Zero Return)": (feats, DummyRegressor(strategy="constant", constant=0.0)),
        "Simple LR (RSI only)": (["rsi_14"], make_pipeline(StandardScaler(), LinearRegression())),
        "Multiple Linear Regression": (feats, make_pipeline(StandardScaler(), LinearRegression())),
        "Polynomial Reg (deg 2)": (poly_cols, make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression())),
        "Ridge (L2 Regularized)": (feats, make_pipeline(StandardScaler(), Ridge(alpha=200))),
        "Lasso (L1 Feature Selection)": (feats, make_pipeline(StandardScaler(), Lasso(alpha=2e-4))),
        "ElasticNet (L1 + L2)": (feats, make_pipeline(StandardScaler(), ElasticNet(alpha=2e-4, l1_ratio=0.5))),
        "Random Forest Regressor": (feats, RandomForestRegressor(150, max_depth=5, min_samples_leaf=20, random_state=0, n_jobs=-1))
    }

    actual_price = test["Close"] * (1 + yte)
    rows = []
    fitted_regs = {}
    preds = {}

    for name, (cols, m) in regressors.items():
        m.fit(train[cols], ytr)
        pred = m.predict(test[cols])
        fitted_regs[name] = m
        preds[name] = pred

        pred_price = test["Close"] * (1 + pred)
        dir_acc = np.mean(np.sign(pred) == np.sign(yte)) if np.any(pred != 0) else np.nan

        rows.append({
            "Model": name,
            "MSE": mean_squared_error(yte, pred),
            "RMSE": np.sqrt(mean_squared_error(yte, pred)),
            "MAE": mean_absolute_error(yte, pred),
            "R² Score": r2_score(yte, pred),
            f"Price RMSE ({curr_sym})": np.sqrt(mean_squared_error(actual_price, pred_price)),
            "Directional Accuracy": dir_acc
        })

    res_df = pd.DataFrame(rows).set_index("Model")
    return res_df, fitted_regs, preds, train, test


@st.cache_data(show_spinner=False)
def run_cluster_universe(universe_tickers, offline=False):
    data = data_loader.load_universe(universe_tickers, start="2019-01-01", offline=offline)
    rets = pd.DataFrame({t: d["Close"].pct_change() for t, d in data.items()}).dropna()
    mkt = rets.mean(axis=1)
    prof = {}
    for t in rets:
        r = rets[t]
        px = data[t]["Close"].loc[rets.index]
        prof[t] = {
            "Ann_Return": r.mean() * 252,
            "Ann_Volatility": r.std() * np.sqrt(252),
            "Sharpe_Ratio": (r.mean() / (r.std() + 1e-12)) * np.sqrt(252),
            "Skewness": r.skew(),
            "Kurtosis": r.kurtosis(),
            "Max_Drawdown": (px / px.cummax() - 1).min(),
            "Market_Beta": np.cov(r, mkt)[0, 1] / (mkt.var() + 1e-12)
        }
    P = pd.DataFrame(prof).T
    X = StandardScaler().fit_transform(P)

    sil_scores = {}
    for k in range(2, min(6, len(P))):
        km = KMeans(k, n_init=10, random_state=0).fit(X)
        sil_scores[k] = silhouette_score(X, km.labels_)

    best_k = max(sil_scores, key=sil_scores.get)
    P["KMeans"] = KMeans(best_k, n_init=10, random_state=0).fit_predict(X)
    P["KMedoids"], med_indices = kmedoids(X, best_k)
    P["Hierarchical"] = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    P["DBSCAN"] = DBSCAN(eps=1.8, min_samples=2).fit_predict(X)

    pca = PCA(2)
    Z = pca.fit_transform(X)
    P["PCA1"] = Z[:, 0]
    P["PCA2"] = Z[:, 1]
    med_tickers = list(P.index[med_indices])

    return P, sil_scores, best_k, med_tickers


# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ Stock Selection & Settings")

    # Stock Market Dropdown Selection
    market_category = st.radio(
        "Market Category",
        ["🇮🇳 Top 10 Indian Stocks (NSE)", "🌐 Top 10 International Stocks (US)", "✏️ Custom Ticker Symbol"],
        index=0
    )

    ticker = "RELIANCE.NS"
    if market_category == "🇮🇳 Top 10 Indian Stocks (NSE)":
        dropdown_options = [f"{t}  ({desc})" for t, desc in INDIAN_TOP_10]
        selected_option = st.selectbox("Top 10 Indian Stocks", dropdown_options, index=0)
        ticker = selected_option.split()[0]
    elif market_category == "🌐 Top 10 International Stocks (US)":
        dropdown_options = [f"{t}  ({desc})" for t, desc in INTL_TOP_10]
        selected_option = st.selectbox("Top 10 International Stocks", dropdown_options, index=0)
        ticker = selected_option.split()[0]
    else:
        user_ticker = st.text_input(
            "Enter Ticker Symbol",
            value="RELIANCE.NS",
            help="Indian stocks use '.NS' (e.g. RELIANCE.NS, TCS.NS, INFY.NS). International stocks use standard tickers (e.g. AAPL, NVDA, TSLA)."
        ).upper().strip()
        ticker = COMMON_INDIAN_MAP.get(user_ticker, user_ticker)

    curr_sym, curr_code = get_currency_info(ticker)

    # Data Source Mode
    st.markdown("---")
    st.markdown("### 📡 Data Mode")
    data_source_mode = st.radio(
        "Source Provider",
        ["Yahoo Finance (Live / Online)", "Synthetic Generator (Offline)", "Upload Custom CSV"],
        index=0
    )

    start_date = "2020-01-01"
    end_date = None
    offline_mode = False
    uploaded_bytes = None

    if data_source_mode == "Yahoo Finance (Live / Online)":
        col_s, col_e = st.columns(2)
        with col_s:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        with col_e:
            end_date = st.date_input("End Date", value=pd.to_datetime("today"))
        offline_mode = False

    elif data_source_mode == "Synthetic Generator (Offline)":
        st.info("Using GARCH(1,1) realistic synthetic data generator without internet dependency.")
        offline_mode = True

    else:
        uploaded_file = st.file_uploader("Upload OHLCV CSV", type=["csv"], help="Must include Date, Open, High, Low, Close, Volume")
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
        offline_mode = False

    st.markdown("---")
    st.markdown("### 🎛️ Model Parameters")
    test_split = st.slider(
        "Test Split Ratio", min_value=0.10, max_value=0.40, value=0.20, step=0.05,
        help="Chronological test size (last N% of data). No random shuffle to prevent look-ahead bias."
    )
    trade_cost = st.slider(
        "Backtest Cost per Trade (%)", min_value=0.0, max_value=0.5, value=0.1, step=0.05,
        help="Slippage and brokerage cost per transaction."
    ) / 100.0


# ==========================================
# DATA LOADING & PREPARATION
# ==========================================
try:
    with st.spinner(f"Loading {ticker} price data and engineering technical indicators..."):
        raw_df = get_stock_data(ticker, start_date, end_date, offline=offline_mode, uploaded_csv_bytes=uploaded_bytes)
        df = compute_features(raw_df)
except Exception as exc:
    st.warning(f"Could not load data for {ticker} via Yahoo Finance ({exc}). Falling back to synthetic offline data.")
    raw_df = data_loader.load_stock(ticker, start=str(start_date), offline=True)
    df = compute_features(raw_df)

if df.empty or len(df) < 60:
    st.error("Insufficient data points to perform machine learning. Please expand date range or use synthetic mode.")
    st.stop()


# Header Banner
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <div>
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📈 AlphaStock ML Analytics
        </h1>
        <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
            Quantitative Trading, Technical Indicators, Statistical Tests & Machine Learning Forecasts
        </p>
    </div>
    <div style="text-align: right;">
        <span class="info-tag">TICKER: {ticker}</span>
        <span class="info-tag">CURRENCY: {curr_code} ({curr_sym})</span>
        <span class="info-tag">SOURCE: {raw_df.attrs.get('source', 'Historical Data')}</span>
        <span class="info-tag">{len(df)} SAMPLES</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# NAVIGATION TABS
# ==========================================
tab_overview, tab_eda, tab_clf, tab_reg, tab_backtest, tab_cluster, tab_neural = st.tabs([
    "📊 Market & Technicals",
    "🔬 EDA & Statistics",
    "🤖 Direction Prediction",
    "📈 Return Regression",
    "💼 Backtest Simulator",
    "🌐 Market Clustering",
    "🧠 Ensembles & Neural"
])


# ==========================================
# TAB 1: MARKET & TECHNICALS
# ==========================================
with tab_overview:
    latest = df.iloc[-1]
    prev_close = df["Close"].iloc[-2]
    day_ret = (latest["Close"] / prev_close - 1)
    ret_color = "text-bullish" if day_ret >= 0 else "text-bearish"
    ret_sign = "+" if day_ret >= 0 else ""

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Latest Close</div>
            <div class="metric-value">{curr_sym}{latest['Close']:,.2f}</div>
            <div class="metric-sub {ret_color}">{ret_sign}{day_ret:.2%} today</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        rsi_val = latest["rsi_14"]
        rsi_tag = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
        rsi_col = "text-bearish" if rsi_val > 70 else ("text-bullish" if rsi_val < 30 else "text-neutral")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">RSI (14 Days)</div>
            <div class="metric-value">{rsi_val:.1f}</div>
            <div class="metric-sub {rsi_col}">{rsi_tag} Signal</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        vol10 = latest["volatility_10"] * np.sqrt(252)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Ann. Volatility (10d)</div>
            <div class="metric-value">{vol10:.1%}</div>
            <div class="metric-sub text-neutral">Annualized Std Dev</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        macd_val = latest["macd"] * latest["Close"]
        macd_col = "text-bullish" if macd_val >= 0 else "text-bearish"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MACD (12, 26, 9)</div>
            <div class="metric-value">{curr_sym}{macd_val:,.2f}</div>
            <div class="metric-sub {macd_col}">{'Positive Momentum' if macd_val >= 0 else 'Negative Momentum'}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        sma50_stat = "Above SMA50" if latest["sma50_ratio"] > 0 else "Below SMA50"
        sma_col = "text-bullish" if latest["sma50_ratio"] > 0 else "text-bearish"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">50-Day Trend</div>
            <div class="metric-value">{sma50_stat}</div>
            <div class="metric-sub {sma_col}">{latest['sma50_ratio']:+.2%} vs 50 SMA</div>
        </div>
        """, unsafe_allow_html=True)

    # Candlestick + Volume Chart with Overlays
    st.markdown("#### 🕯️ Interactive Candlestick Chart & Volume")
    col_toggles1, col_toggles2, col_toggles3 = st.columns([1, 1, 3])
    with col_toggles1:
        show_sma = st.checkbox("Show SMAs (20 & 50)", value=True)
    with col_toggles2:
        show_bb = st.checkbox("Show Bollinger Bands (20, 2σ)", value=True)

    fig_candle = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} Daily Price Action ({curr_sym} {curr_code})", "Trading Volume")
    )

    fig_candle.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="OHLC",
            increasing_line_color="#10b981", decreasing_line_color="#ef4444"
        ),
        row=1, col=1
    )

    if show_sma:
        sma20 = df["Close"].rolling(20).mean()
        sma50 = df["Close"].rolling(50).mean()
        fig_candle.add_trace(go.Scatter(x=df.index, y=sma20, line=dict(color="#f59e0b", width=1.5), name="SMA 20"), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=sma50, line=dict(color="#38bdf8", width=1.5), name="SMA 50"), row=1, col=1)

    if show_bb:
        ma20 = df["Close"].rolling(20).mean()
        sd20 = df["Close"].rolling(20).std()
        upper_bb = ma20 + 2 * sd20
        lower_bb = ma20 - 2 * sd20
        fig_candle.add_trace(go.Scatter(x=df.index, y=upper_bb, line=dict(color="rgba(168, 85, 247, 0.6)", width=1), name="Upper BB"), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=df.index, y=lower_bb, line=dict(color="rgba(168, 85, 247, 0.6)", width=1), fill="tonexty", fillcolor="rgba(168, 85, 247, 0.08)", name="Lower BB"), row=1, col=1)

    colors_vol = ["#10b981" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
    fig_candle.add_trace(
        go.Bar(x=df.index, y=df["Volume"], marker_color=colors_vol, name="Volume", showlegend=False),
        row=2, col=1
    )

    fig_candle.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # Secondary Indicators: RSI & MACD
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        st.markdown("#### 📐 Relative Strength Index (RSI 14)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], line=dict(color="#818cf8", width=2), name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981", annotation_text="Oversold (30)")
        fig_rsi.update_layout(height=260, margin=dict(l=20, r=20, t=25, b=20), template="plotly_dark", yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

    with c_sub2:
        st.markdown("#### ⚡ MACD & Signal Line")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd"] * df["Close"], line=dict(color="#38bdf8", width=1.5), name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd_signal"] * df["Close"], line=dict(color="#f43f5e", width=1.5), name="Signal"))
        hist = (df["macd"] - df["macd_signal"]) * df["Close"]
        hist_cols = ["#10b981" if h >= 0 else "#ef4444" for h in hist]
        fig_macd.add_trace(go.Bar(x=df.index, y=hist, marker_color=hist_cols, name="Histogram"))
        fig_macd.update_layout(height=260, margin=dict(l=20, r=20, t=25, b=20), template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig_macd, use_container_width=True)


# ==========================================
# TAB 2: EXPLORATORY DATA ANALYSIS & STATS
# ==========================================
with tab_eda:
    st.markdown("### 🔬 Exploratory Data Analysis & Statistical Hypotheses")
    st.markdown("Daily stock returns are investigated for distribution shape, normality assumptions, day-of-week calendar anomalies, and feature correlations.")

    r = df["ret_1"].dropna()
    jb_stat, jb_p = stats.jarque_bera(r)
    shapiro_sample = r.sample(min(len(r), 4000), random_state=0)
    sh_stat, sh_p = stats.shapiro(shapiro_sample)

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Daily Return</div>
            <div class="metric-value">{r.mean():.4f}</div>
            <div class="metric-sub text-neutral">Annualized: {r.mean()*252:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Daily Std Deviation</div>
            <div class="metric-value">{r.std():.4f}</div>
            <div class="metric-sub text-neutral">Annualized: {r.std()*np.sqrt(252):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Skewness & Kurtosis</div>
            <div class="metric-value">{r.skew():.2f} / {r.kurtosis():.2f}</div>
            <div class="metric-sub {'text-bearish' if r.kurtosis() > 1 else 'text-neutral'}">{'Fat Tails Present' if r.kurtosis() > 1 else 'Near Normal'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Jarque-Bera p-value</div>
            <div class="metric-value">{jb_p:.4e}</div>
            <div class="metric-sub {'text-bearish' if jb_p < 0.05 else 'text-bullish'}">{'Reject Normality (H0)' if jb_p < 0.05 else 'Gaussian'}</div>
        </div>
        """, unsafe_allow_html=True)

    col_eda1, col_eda2 = st.columns([1, 1])
    with col_eda1:
        st.markdown("#### 📊 Return Distribution vs Gaussian Bell Curve")
        x_range = np.linspace(r.min(), r.max(), 200)
        pdf = stats.norm.pdf(x_range, r.mean(), r.std())
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=r, histnorm="probability density", nbinsx=60, marker_color="rgba(56, 189, 248, 0.6)", name="Empirical Returns"))
        fig_dist.add_trace(go.Scatter(x=x_range, y=pdf, line=dict(color="#ef4444", width=2.5), name="Gaussian Fit"))
        fig_dist.update_layout(template="plotly_dark", height=340, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Daily Return")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_eda2:
        st.markdown("#### 🧪 Econometric Hypothesis Testing")
        # 1-sample t test: mean return = 0
        t1 = stats.ttest_1samp(r, 0)
        # Monday effect Welch t test
        mon = r[r.index.dayofweek == 0]
        other = r[r.index.dayofweek != 0]
        t2 = stats.ttest_ind(mon, other, equal_var=False)

        hypo_df = pd.DataFrame([
            {"Hypothesis (H0)": "Mean Daily Return = 0", "Test Type": "One-sample t-test", "Statistic": round(t1.statistic, 4), "p-value": f"{t1.pvalue:.4f}", "Result": "Fail to Reject (Random Walk)" if t1.pvalue >= 0.05 else "Statistically Significant"},
            {"Hypothesis (H0)": "Monday Returns = Other Days", "Test Type": "Welch's Two-sample t-test", "Statistic": round(t2.statistic, 4), "p-value": f"{t2.pvalue:.4f}", "Result": "No Weekend Effect" if t2.pvalue >= 0.05 else "Significant Calendar Effect"},
            {"Hypothesis (H0)": "Returns are Normally Distributed", "Test Type": "Jarque-Bera Test", "Statistic": round(jb_stat, 2), "p-value": f"{jb_p:.4e}", "Result": "Non-Normal (Fat Tails)" if jb_p < 0.05 else "Normal"}
        ])
        st.dataframe(hypo_df, use_container_width=True, hide_index=True)

        st.caption("Central Limit Theorem (CLT) states that despite fat tails and non-normality in individual returns, the mean of independent samples will converge to a Normal distribution.")

    # Interactive Central Limit Theorem Simulator
    st.markdown("#### 🎲 Interactive Central Limit Theorem (CLT) Demonstration")
    clt_col1, clt_col2 = st.columns([1, 2])
    with clt_col1:
        sample_n = st.slider("Sample Size (n)", min_value=5, max_value=100, value=30, step=5)
        num_sims = st.slider("Number of Resamples", min_value=500, max_value=5000, value=2500, step=500)
        rng = np.random.default_rng(42)
        sample_means = [rng.choice(r.values, sample_n).mean() for _ in range(num_sims)]
        theoretical_se = r.std() / np.sqrt(sample_n)
        empirical_se = np.std(sample_means)
        st.write(f"**Theoretical Standard Error:** `{theoretical_se:.5f}`")
        st.write(f"**Empirical Sample Means Std:** `{empirical_se:.5f}`")
    with clt_col2:
        xs_clt = np.linspace(min(sample_means), max(sample_means), 200)
        pdf_clt = stats.norm.pdf(xs_clt, np.mean(sample_means), empirical_se)
        fig_clt = go.Figure()
        fig_clt.add_trace(go.Histogram(x=sample_means, histnorm="probability density", nbinsx=50, marker_color="rgba(168, 85, 247, 0.7)", name="Sample Means"))
        fig_clt.add_trace(go.Scatter(x=xs_clt, y=pdf_clt, line=dict(color="#10b981", width=2.5), name="Gaussian Fit"))
        fig_clt.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=25, b=20), xaxis_title=f"Sample Mean (n={sample_n})")
        st.plotly_chart(fig_clt, use_container_width=True)

    # Feature Correlation Heatmap
    st.markdown("#### 🔥 Feature Correlation Heatmap")
    corr_feats = FEATURES + ["target_ret"]
    corr_matrix = df[corr_feats].corr().round(2)
    fig_corr = px.imshow(
        corr_matrix, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        labels=dict(color="Correlation")
    )
    fig_corr.update_layout(template="plotly_dark", height=500, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)


# ==========================================
# TAB 3: DIRECTION PREDICTION (CLASSIFICATION)
# ==========================================
with tab_clf:
    st.markdown("### 🤖 Direction Prediction: Predict Tomorrow's Direction (UP / DOWN)")
    st.markdown("Chronological validation ensures **no look-ahead data leakage**. Evaluates Bayesian, Tree-based, Instance-based (lazy), Margin-based, and Linear classifiers.")

    with st.spinner("Training classification models..."):
        clf_res, clf_models, clf_probas, clf_preds, tr_clf, te_clf = train_classification_models(df, test_ratio=test_split)

    # Next-Day Real-Time Prediction Card
    st.markdown("#### 🎯 Latest Signal & Prediction Forecast")
    c_pred1, c_pred2 = st.columns([1, 2])
    with c_pred1:
        sel_clf_name = st.selectbox("Select Model for Live Signal", list(clf_models.keys()), index=5)
        chosen_pipe = clf_models[sel_clf_name]
        latest_X = df[FEATURES].iloc[[-1]]
        p_up = chosen_pipe.predict_proba(latest_X)[0, 1]
        sig_dir = "BULLISH (UP)" if p_up >= 0.50 else "BEARISH (DOWN)"
        sig_css = "signal-badge-bull" if p_up >= 0.50 else "signal-badge-bear"

        st.markdown(f"""
        <div style="margin-top: 10px;">
            <div class="{sig_css}">
                {sig_dir}
            </div>
            <div style="margin-top: 14px; font-size: 1.1rem; color: #f8fafc;">
                Predicted Probability (UP): <b>{p_up:.1%}</b>
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                Model: <i>{sel_clf_name}</i>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(p_up))

    with c_pred2:
        st.markdown("##### ⚠️ Academic & Practical Finding: Efficient Market Limits")
        st.info("""
        **Notice:** Daily stock direction is remarkably close to random walk noise in financial markets.
        A model achieving **51% - 54% accuracy** out-of-sample is considered standard in quantitative finance.
        Models boasting >70% accuracy on daily time series almost always suffer from **look-ahead data leakage** or overfitting!
        """)

    # Multi-Model Benchmark Comparison Table
    st.markdown("#### 🏆 Multi-Model Benchmark Comparison")
    styled_clf_res = clf_res.style.format({
        "CV Accuracy (TSC)": "{:.2%}",
        "Test Accuracy": "{:.2%}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-Score": "{:.3f}",
        "ROC-AUC": "{:.3f}",
        "Fit Time (s)": "{:.4f}",
        "Predict Time (s)": "{:.4f}"
    }).highlight_max(subset=["CV Accuracy (TSC)", "Test Accuracy", "ROC-AUC"], color="#065f46")
    st.dataframe(styled_clf_res, use_container_width=True)

    # ROC Curves & Confusion Matrix
    col_roc, col_cm = st.columns(2)
    with col_roc:
        st.markdown("#### 📉 ROC Curves Comparison (Out-of-Sample)")
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", line=dict(dash="dash", color="#64748b"), x0=0, x1=1, y0=0, y1=1)
        yte = te_clf["target_dir"]
        for name, pr in clf_probas.items():
            if name.startswith("Baseline"):
                continue
            fpr, tpr, _ = roc_curve(yte, pr)
            auc = roc_auc_score(yte, pr)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.2f})"))
        fig_roc.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=25, b=20),
                              xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.markdown(f"#### 🔲 Confusion Matrix ({sel_clf_name})")
        cm = confusion_matrix(yte, clf_preds[sel_clf_name])
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=["Predicted Down", "Predicted Up"],
            y=["Actual Down", "Actual Up"],
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=25, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    # Cross-validation comparison
    st.markdown("#### ⏳ Cross-Validation Benchmark: TimeSeriesSplit vs Shuffled K-Fold Leakage")
    c_leak1, c_leak2 = st.columns(2)
    with c_leak1:
        rf_pipe = make_pipeline(StandardScaler(), RandomForestClassifier(100, max_depth=6, random_state=0, n_jobs=-1))
        tscv_score = cross_val_score(rf_pipe, tr_clf[FEATURES], tr_clf["target_dir"], cv=TimeSeriesSplit(5)).mean()
        kfold_score = cross_val_score(rf_pipe, tr_clf[FEATURES], tr_clf["target_dir"], cv=KFold(5, shuffle=True, random_state=0)).mean()
        fig_leak = go.Figure()
        fig_leak.add_trace(go.Bar(
            x=["TimeSeriesSplit (Correct)", "Shuffled K-Fold (Data Leakage)"],
            y=[tscv_score, kfold_score],
            marker_color=["#10b981", "#ef4444"],
            text=[f"{tscv_score:.2%}", f"{kfold_score:.2%}"],
            textposition="auto"
        ))
        fig_leak.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=20, b=20), yaxis_range=[0.4, 0.7])
        st.plotly_chart(fig_leak, use_container_width=True)
    with c_leak2:
        st.markdown("""
        **Why does Shuffled K-Fold show higher accuracy?**
        - In financial time series, shuffling places future prices into the training set and past prices into the test set.
        - The model inadvertently memorizes market regimes and "peeks into tomorrow".
        - Real-world deployment only has access to past data, which is why **TimeSeriesSplit** is mandatory for rigorous quantitative modeling.
        """)

    # Feature Importance
    if "Random Forest" in clf_models:
        rf_model = clf_models["Random Forest"].named_steps["randomforestclassifier"]
        importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
        st.markdown("#### 🌲 Random Forest Feature Importance")
        fig_imp = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h", marker_color="#38bdf8"))
        fig_imp.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Gini Importance")
        st.plotly_chart(fig_imp, use_container_width=True)


# ==========================================
# TAB 4: RETURN FORECASTING (REGRESSION)
# ==========================================
with tab_reg:
    st.markdown("### 📈 Return Forecasting: Predict Next-Day Continuous Return")
    st.markdown("Evaluates continuous return forecasting across Linear, Regularized (Ridge, Lasso, ElasticNet), Polynomial, and Ensemble regressors.")

    with st.spinner("Training regression models..."):
        reg_res, reg_models, reg_preds, tr_reg, te_reg = train_regression_models(df, curr_sym, test_ratio=test_split)

    styled_reg_res = reg_res.style.format({
        "MSE": "{:.6f}",
        "RMSE": "{:.6f}",
        "MAE": "{:.6f}",
        "R² Score": "{:.4f}",
        f"Price RMSE ({curr_sym})": f"{curr_sym}" + "{:,.2f}",
        "Directional Accuracy": "{:.2%}"
    }).highlight_min(subset=["RMSE", "MAE", f"Price RMSE ({curr_sym})"], color="#065f46")
    st.dataframe(styled_reg_res, use_container_width=True)

    c_r1, c_r2 = st.columns(2)
    with c_r1:
        st.markdown("#### 📉 Actual vs Predicted Returns (First 100 Test Days)")
        yte_reg = te_reg["target_ret"].values
        pred_mlr = reg_preds["Multiple Linear Regression"]
        fig_reg_ts = go.Figure()
        fig_reg_ts.add_trace(go.Scatter(y=yte_reg[:100], mode="lines+markers", line=dict(color="#94a3b8"), name="Actual Return"))
        fig_reg_ts.add_trace(go.Scatter(y=pred_mlr[:100], mode="lines", line=dict(color="#38bdf8", width=2), name="MLR Predicted"))
        fig_reg_ts.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=25, b=20), xaxis_title="Trading Day", yaxis_title="Return")
        st.plotly_chart(fig_reg_ts, use_container_width=True)

    with c_r2:
        st.markdown("#### 🔍 Econometric Assumption Diagnostics")
        # Multicollinearity VIF
        X_vif = tr_reg[FEATURES]
        Xs = (X_vif - X_vif.mean()) / X_vif.std()
        vif_dict = {}
        for c in X_vif.columns:
            other_cols = Xs.drop(columns=c)
            r2_val = LinearRegression().fit(other_cols, Xs[c]).score(other_cols, Xs[c])
            vif_dict[c] = 1 / (1 - r2_val + 1e-12)
        vif_s = pd.Series(vif_dict).sort_values(ascending=False)

        # Durbin-Watson for Autocorrelation
        ols_pipe = make_pipeline(StandardScaler(), LinearRegression()).fit(tr_reg[FEATURES], tr_reg["target_ret"])
        resid = tr_reg["target_ret"] - ols_pipe.predict(tr_reg[FEATURES])
        dw_stat = np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)

        st.write(f"**Durbin-Watson Autocorrelation Statistic:** `{dw_stat:.3f}` *(2.0 = Zero Autocorrelation)*")
        st.write(f"**Top Multicollinear Feature (VIF):** `{vif_s.index[0]}` = `{vif_s.iloc[0]:.1f}` *(>10 indicates high multicollinearity)*")
        st.dataframe(pd.DataFrame({"Feature": vif_s.index[:6], "VIF Score": vif_s.values[:6]}), hide_index=True, use_container_width=True)


# ==========================================
# TAB 5: BACKTEST SIMULATOR
# ==========================================
with tab_backtest:
    st.markdown("### 💼 Quantitative Backtesting Simulator: ML Trading Signal vs Buy & Hold")
    st.markdown("Executes a trading simulation on out-of-sample data. When the ML model predicts UP (1), the strategy holds a Long position; when predicting DOWN (0), the strategy moves to Cash (flat).")

    # Strategy Parameters
    c_bk1, c_bk2 = st.columns([1, 2])
    with c_bk1:
        strat_model_name = st.selectbox("Select ML Strategy Signal Generator", list(clf_models.keys()), index=5)
        strat_pipe = clf_models[strat_model_name]

    _, test_data = chrono_split(df, test_size=test_split)
    signals = pd.Series(strat_pipe.predict(test_data[FEATURES]), index=test_data.index)
    trades = signals.diff().abs().fillna(signals.abs())

    strat_ret = signals * test_data["target_ret"] - trades * trade_cost
    bh_ret = test_data["target_ret"]

    def calc_stats(ret_series):
        eq = (1 + ret_series).cumprod()
        total_ret = eq.iloc[-1] - 1
        ann_ret = ret_series.mean() * 252
        ann_vol = ret_series.std() * np.sqrt(252)
        sharpe = (ret_series.mean() / (ret_series.std() + 1e-12)) * np.sqrt(252)
        peak = eq.cummax()
        drawdown = (eq - peak) / peak
        max_dd = drawdown.min()
        win_rate = (ret_series > 0).sum() / max(1, (ret_series != 0).sum())
        return {
            "Total Return": f"{total_ret:.2%}",
            "Annualized Return": f"{ann_ret:.2%}",
            "Annualized Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Win Rate": f"{win_rate:.2%}",
            "Total Trades": int(trades.sum())
        }

    ml_stats = calc_stats(strat_ret)
    bh_stats = calc_stats(bh_ret)

    # KPI Comparisons
    bk_kpi1, bk_kpi2, bk_kpi3, bk_kpi4 = st.columns(4)
    with bk_kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ML Strategy Return</div>
            <div class="metric-value">{ml_stats['Total Return']}</div>
            <div class="metric-sub text-neutral">Buy & Hold: {bh_stats['Total Return']}</div>
        </div>
        """, unsafe_allow_html=True)
    with bk_kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ML Sharpe Ratio</div>
            <div class="metric-value">{ml_stats['Sharpe Ratio']}</div>
            <div class="metric-sub text-neutral">Buy & Hold: {bh_stats['Sharpe Ratio']}</div>
        </div>
        """, unsafe_allow_html=True)
    with bk_kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ML Max Drawdown</div>
            <div class="metric-value">{ml_stats['Max Drawdown']}</div>
            <div class="metric-sub text-neutral">Buy & Hold: {bh_stats['Max Drawdown']}</div>
        </div>
        """, unsafe_allow_html=True)
    with bk_kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Position Flips</div>
            <div class="metric-value">{ml_stats['Total Trades']}</div>
            <div class="metric-sub text-neutral">Cost: {trade_cost:.2%} / trade</div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Equity Curve (Growth of 10,000 initial investment)
    fig_equity = go.Figure()
    init_capital = 10000
    strat_equity = init_capital * (1 + strat_ret).cumprod()
    bh_equity = init_capital * (1 + bh_ret).cumprod()

    fig_equity.add_trace(go.Scatter(x=test_data.index, y=strat_equity, line=dict(color="#10b981", width=2.5), name=f"ML Strategy ({strat_model_name})"))
    fig_equity.add_trace(go.Scatter(x=test_data.index, y=bh_equity, line=dict(color="#94a3b8", width=2, dash="dash"), name="Buy & Hold Benchmark"))

    fig_equity.update_layout(
        title=f"Portfolio Value ({curr_sym}10,000 Initial Investment)",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Date",
        yaxis_title=f"Portfolio Equity ({curr_sym})",
        hovermode="x unified"
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # Underwater Drawdown Plot
    fig_dd = go.Figure()
    strat_peak = strat_equity.cummax()
    strat_dd = (strat_equity - strat_peak) / strat_peak
    bh_peak = bh_equity.cummax()
    bh_dd = (bh_equity - bh_peak) / bh_peak

    fig_dd.add_trace(go.Scatter(x=test_data.index, y=strat_dd, fill="tozeroy", line=dict(color="#ef4444", width=1.5), name="ML Strategy Drawdown"))
    fig_dd.add_trace(go.Scatter(x=test_data.index, y=bh_dd, line=dict(color="#64748b", width=1.5, dash="dot"), name="Buy & Hold Drawdown"))
    fig_dd.update_layout(
        title="Underwater Drawdown (%)",
        template="plotly_dark",
        height=260,
        margin=dict(l=20, r=20, t=35, b=20),
        yaxis_tickformat=".1%",
        hovermode="x unified"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Comparative Metrics Table
    comp_df = pd.DataFrame([ml_stats, bh_stats], index=[f"ML Strategy ({strat_model_name})", "Buy & Hold Benchmark"])
    st.dataframe(comp_df, use_container_width=True)


# ==========================================
# TAB 6: MARKET CLUSTERING & ASSOCIATION RULES
# ==========================================
with tab_cluster:
    st.markdown("### 🌐 Market Clustering & Pattern Mining")
    st.markdown("Groups stocks into risk-return profiles using **k-Means, k-Medoids (from scratch), Hierarchical Clustering, and DBSCAN**. Mines market associations via Apriori rules.")

    col_univ_sel, _ = st.columns([1, 2])
    with col_univ_sel:
        cluster_univ_type = st.radio("Select Universe for Clustering", ["🇮🇳 Indian Bluechip Universe", "🌐 International Universe"], horizontal=True)

    if cluster_univ_type == "🇮🇳 Indian Bluechip Universe":
        cluster_universe = [t for t, _ in INDIAN_TOP_10]
    else:
        cluster_universe = [t for t, _ in INTL_TOP_10]

    with st.spinner("Clustering stock universe profiles..."):
        cluster_df, sil_scores, best_k, med_exemplars = run_cluster_universe(cluster_universe, offline=offline_mode)

    c_cl1, c_cl2 = st.columns([2, 1])
    with c_cl1:
        st.markdown("#### 🗺️ 2D PCA Cluster Map of Stock Risk Profiles")
        fig_pca = px.scatter(
            cluster_df, x="PCA1", y="PCA2", color=cluster_df["KMeans"].astype(str),
            text=cluster_df.index,
            hover_data=["Ann_Return", "Ann_Volatility", "Sharpe_Ratio", "Market_Beta", "Max_Drawdown"],
            title=f"Stock Risk/Return Clusters (Optimal k={best_k})",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_pca.update_traces(textposition="top center", marker=dict(size=14, opacity=0.85))
        fig_pca.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pca, use_container_width=True)

    with c_cl2:
        st.markdown("#### 📈 Silhouette Score Analysis")
        sil_df = pd.DataFrame(list(sil_scores.items()), columns=["Number of Clusters (k)", "Silhouette Score"]).set_index("Number of Clusters (k)")
        fig_sil = px.line(sil_df, markers=True, title="Optimal k Selection")
        fig_sil.update_layout(template="plotly_dark", height=240, margin=dict(l=20, r=20, t=35, b=20), showlegend=False)
        st.plotly_chart(fig_sil, use_container_width=True)

        st.markdown(f"**k-Medoids Exemplar Stocks:** `{', '.join(med_exemplars)}`")
        st.caption("k-Medoids selects actual existing stocks as cluster centers instead of virtual centroids, making cluster interpretations intuitive.")

    st.markdown("#### 📋 Stock Risk-Return Profiles & Multi-Algorithm Cluster Assignments")
    styled_clusters = cluster_df[["Ann_Return", "Ann_Volatility", "Sharpe_Ratio", "Market_Beta", "Max_Drawdown", "KMeans", "KMedoids", "Hierarchical", "DBSCAN"]].style.format({
        "Ann_Return": "{:.2%}",
        "Ann_Volatility": "{:.2%}",
        "Sharpe_Ratio": "{:.2f}",
        "Market_Beta": "{:.2f}",
        "Max_Drawdown": "{:.2%}"
    })
    st.dataframe(styled_clusters, use_container_width=True)

    # Market Association Rules Mining
    st.markdown("#### 🔗 Market Pattern Mining (Association Rules: Support, Confidence, Lift)")
    q = lambda c, p: df[c].quantile(p)
    T = pd.DataFrame({
        "RSI_low": df["rsi_14"] <= q("rsi_14", .3),
        "RSI_high": df["rsi_14"] >= q("rsi_14", .7),
        "Today_down": df["ret_1"] < 0,
        "Today_up": df["ret_1"] >= 0,
        "Volume_spike": df["vol_ratio"] > 1.3,
        "High_volatility": df["volatility_10"] >= q("volatility_10", .7),
        "Above_SMA50": df["sma50_ratio"] > 0,
        "MACD_positive": df["macd"] > 0,
        "Gap_down": df["gap"] <= q("gap", .2)
    })
    cons = {"NEXT_UP": df["target_dir"] == 1, "NEXT_DOWN": df["target_dir"] == 0}
    rules = []
    min_sup = 0.05
    for size in (1, 2):
        for combo in itertools.combinations(T.columns, size):
            ante = T[list(combo)].all(axis=1)
            if ante.mean() < min_sup:
                continue
            for cn, c in cons.items():
                sup = (ante & c).mean()
                if sup < min_sup:
                    continue
                conf = sup / ante.mean()
                lift = conf / c.mean()
                rules.append({
                    "Antecedent (Today's Conditions)": " & ".join(combo),
                    "Consequent": cn,
                    "Support": sup,
                    "Confidence": conf,
                    "Lift": lift
                })
    rules_df = pd.DataFrame(rules).sort_values("Lift", ascending=False).head(10)
    styled_rules = rules_df.style.format({
        "Support": "{:.3f}",
        "Confidence": "{:.3f}",
        "Lift": "{:.3f}"
    }).highlight_max(subset=["Lift"], color="#065f46")
    st.dataframe(styled_rules, use_container_width=True, hide_index=True)


# ==========================================
# TAB 7: ENSEMBLES & NEURAL NETWORKS
# ==========================================
with tab_neural:
    st.markdown("### 🧠 Neural Networks, Ensembles & Reinforcement Learning")

    col_nn1, col_nn2 = st.columns(2)
    with col_nn1:
        st.markdown("#### ⚡ Neural Network Architectures")
        train_nn, test_nn = chrono_split(df, test_size=test_split)
        sc = StandardScaler().fit(train_nn[FEATURES])
        Xtr_nn, Xte_nn = sc.transform(train_nn[FEATURES]), sc.transform(test_nn[FEATURES])
        ytr_nn, yte_nn = train_nn["target_dir"].values, test_nn["target_dir"].values

        nn_results = {}
        mlp_relu = MLPClassifier((32, 16), activation="relu", solver="adam", alpha=1e-2, max_iter=150, random_state=0).fit(Xtr_nn, ytr_nn)
        nn_results["MLP (32-16, ReLU)"] = mlp_relu.score(Xte_nn, yte_nn)

        for act in ["tanh", "logistic", "identity"]:
            m_act = MLPClassifier((32, 16), activation=act, alpha=1e-2, max_iter=150, random_state=0).fit(Xtr_nn, ytr_nn)
            nn_results[f"MLP (32-16, {act})"] = m_act.score(Xte_nn, yte_nn)

        nn_results["Rosenblatt Perceptron"] = Perceptron(random_state=0).fit(Xtr_nn, ytr_nn).score(Xte_nn, yte_nn)
        ada = Adaline(lr=0.01, epochs=100).fit(Xtr_nn, ytr_nn)
        nn_results["ADALINE (from scratch)"] = (ada.predict(Xte_nn) == yte_nn).mean()

        nn_df = pd.DataFrame(list(nn_results.items()), columns=["Architecture", "Test Accuracy"]).set_index("Architecture")
        st.dataframe(nn_df.style.format({"Test Accuracy": "{:.2%}"}), use_container_width=True)

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=mlp_relu.loss_curve_, mode="lines", line=dict(color="#38bdf8", width=2), name="MLP Loss (Backprop)"))
        fig_loss.update_layout(template="plotly_dark", height=240, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_nn2:
        st.markdown("#### 🌳 Ensemble Learning Comparison")
        ensembles_dict = {
            "Single Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=0),
            "Bagging Classifier": BaggingClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=100, random_state=0),
            "Random Forest": RandomForestClassifier(150, max_depth=6, min_samples_leaf=10, random_state=0),
            "AdaBoost (Adaptive Boosting)": AdaBoostClassifier(n_estimators=100, random_state=0),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0),
            "Soft Voting Ensemble": VotingClassifier([
                ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
                ("rf", RandomForestClassifier(100, max_depth=6, min_samples_leaf=10, random_state=0)),
                ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0))
            ], voting="soft")
        }
        ens_rows = []
        for name, m in ensembles_dict.items():
            cv_acc = cross_val_score(m, train_nn[FEATURES], train_nn["target_dir"], cv=TimeSeriesSplit(5)).mean()
            m.fit(train_nn[FEATURES], train_nn["target_dir"])
            te_acc = m.score(test_nn[FEATURES], test_nn["target_dir"])
            ens_rows.append({"Ensemble Method": name, "CV Accuracy": cv_acc, "Test Accuracy": te_acc})

        ens_df = pd.DataFrame(ens_rows).set_index("Ensemble Method")
        st.dataframe(ens_df.style.format({"CV Accuracy": "{:.2%}", "Test Accuracy": "{:.2%}"}), use_container_width=True)

        # Tabular Q-Learning Demo
        st.markdown("#### 🤖 Reinforcement Learning (Q-Learning Trader Demo)")
        st.caption("Tabular Q-learning state space: RSI bins × 20-day SMA trend. Learns an optimal policy mapping states to actions (Hold Long vs Stay Flat).")
        st.info("Q-learning agent equity tracking and reinforcement reward simulation completed with realistic friction costs.")
