import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import ta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import traceback

st.set_page_config(layout="wide", page_title="Krypto Dashboard (robust)")

WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]
CACHE_SHORT, CACHE_MED, CACHE_LONG = 300, 900, 3600
MIN_DATA_FOR_MODEL = 30

# --------------------- Hilfsfunktionen ---------------------
def _ensure_series_1d(s):
    if s is None:
        return pd.Series(dtype=float)
    if isinstance(s, pd.Series):
        return s.astype(float)
    if isinstance(s, pd.DataFrame):
        for c in s.columns:
            if pd.api.types.is_numeric_dtype(s[c]):
                return s[c].astype(float)
        return s.iloc[:, 0].astype(float)
    arr = np.asarray(s)
    if arr.ndim > 1:
        arr = arr.ravel()
    return pd.Series(arr).astype(float)

@st.cache_data(ttl=CACHE_SHORT)
def fetch_ohlc(symbol: str, months=12, interval="1d"):
    try:
        df = yf.download(symbol, period=f"{months}mo", interval=interval, progress=False)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

# --------------------- Technische Indikatoren ---------------------
def compute_indicators(df):
    if df is None or df.empty:
        return df
    close = _ensure_series_1d(df["Close"])
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close, 14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"], df["MACD_SIGNAL"] = macd.macd(), macd.macd_signal()
    df["MACD_DIFF"] = df["MACD"] - df["MACD_SIGNAL"]
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], close, 14).average_true_range()
    return df

def detect_swing_signals(df):
    if df is None or len(df) < 2:
        return []
    prev, cur = df.iloc[-2], df.iloc[-1]
    alerts = []
    if cur["EMA20"] > cur["EMA50"] and prev["EMA20"] <= prev["EMA50"]:
        alerts.append("🚀 EMA20 über EMA50 → Bullish")
    if cur["EMA20"] < cur["EMA50"] and prev["EMA20"] >= prev["EMA50"]:
        alerts.append("🔻 EMA20 unter EMA50 → Bearish")
    if cur["RSI"] > 70:
        alerts.append(f"⚠️ RSI überkauft ({cur['RSI']:.1f})")
    if cur["RSI"] < 30:
        alerts.append(f"⚠️ RSI überverkauft ({cur['RSI']:.1f})")
    if cur["MACD"] > cur["MACD_SIGNAL"] and prev["MACD"] <= prev["MACD_SIGNAL"]:
        alerts.append("📈 MACD-Bullish-Crossover")
    if cur["MACD"] < cur["MACD_SIGNAL"] and prev["MACD"] >= prev["MACD_SIGNAL"]:
        alerts.append("📉 MACD-Bearish-Crossover")
    return alerts

# --------------------- Makrodaten & Prognose ---------------------
@st.cache_data(ttl=CACHE_LONG)
def fetch_macro_daily():
    try:
        btc = yf.download("BTC-USD", period="2y", interval="1d", progress=False)["Close"]
        dxy = yf.download("DX-Y.NYB", period="2y", interval="1d", progress=False)["Close"]
        vix = yf.download("^VIX", period="2y", interval="1d", progress=False)["Close"]
        return pd.concat({"BTC": btc, "DXY": dxy, "VIX": vix}, axis=1).dropna()
    except Exception:
        return pd.DataFrame()

def prepare_features_daily(symbol):
    try:
        price = yf.download(symbol, period="1y", interval="1d", progress=False)["Close"]
    except Exception:
        return pd.DataFrame()
    if not isinstance(price, pd.Series) or price.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"Close": price})
    df["RET1"] = df["Close"].pct_change(1)
    df["RET7"] = df["Close"].pct_change(7)
    df["VOL14"] = df["RET1"].rolling(14).std()
    df = df.dropna()
    macro = fetch_macro_daily()
    if not macro.empty:
        df = df.join(macro[["DXY", "VIX"]], how="left").fillna(method="ffill")
    return df.dropna()

def train_and_predict(symbol, horizon=30):
    df = prepare_features_daily(symbol)
    if df.empty or len(df) < MIN_DATA_FOR_MODEL:
        return None, None
    df["FUT"] = df["Close"].shift(-horizon)
    df["RET_FUT"] = df["FUT"] / df["Close"] - 1
    df = df.dropna()
    X = df[["RET1", "RET7", "VOL14", "DXY", "VIX"]].values
    y = df["RET_FUT"].values
    if len(X) < 20:
        return None, None
    try:
        scaler = StandardScaler().fit(X)
        model = LinearRegression().fit(scaler.transform(X), y)
        pred = model.predict(scaler.transform([X[-1]]))[0]
        r2 = r2_score(y, model.predict(scaler.transform(X)))
        return pred, r2
    except Exception:
        return None, None

# --------------------- Plotting ---------------------
def plot_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Preis"))
    for p in (20, 50, 200):
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA{p}"], name=f"EMA{p}", line=dict(width=2)))
    fig.update_layout(title=f"{ticker} Chart", height=400)
    st.plotly_chart(fig, use_container_width=True)
    if "RSI" in df.columns:
        st.line_chart(df["RSI"], height=150)

# --------------------- UI ---------------------
def main_app():
    st.title("📊 Krypto-Dashboard (BTC / ETH / SOL)")
    st.caption("Robust, kombiniert technische & makroökonomische Analyse")

    tabs = st.tabs(["Übersicht", "Indikatoren", "Makroanalyse"])

    # Übersicht
    with tabs[0]:
        st.header("Übersicht & Prognosen")
        for sym in WATCHLIST:
            pred, r2 = train_and_predict(sym, 30)
            if pred is None:
                st.warning(f"{sym}: Keine Prognose möglich")
                continue
            trend = "Bullish" if pred > 0.02 else ("Bearish" if pred < -0.02 else "Neutral")
            emoji = "🟢" if trend == "Bullish" else ("🔴" if trend == "Bearish" else "⚪")
            st.write(f"{emoji} {sym}: {trend} ({pred*100:.2f}% | R²={r2:.2f})")

    # Indikatoren
    with tabs[1]:
        st.header("Technische Indikatoren")
        for sym in WATCHLIST:
            df = fetch_ohlc(sym)
            if df is None:
                continue
            df = compute_indicators(df)
            plot_chart(df, sym)
            for a in detect_swing_signals(df):
                st.warning(a)

    # Makroanalyse
    with tabs[2]:
        st.header("Makroanalyse (DXY, VIX)")
        macro = fetch_macro_daily()
        if macro.empty:
            st.info("Keine Makrodaten verfügbar.")
        else:
            st.dataframe(macro.tail(20).round(3))

if __name__ == "__main__":
    try:
        main_app()
    except Exception as e:
        st.error("Unerwarteter Fehler: " + str(e))
        st.text(traceback.format_exc())
