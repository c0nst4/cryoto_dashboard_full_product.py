# crypto_dashboard_full_product.py
"""
Krypto Profi-Dashboard (BTC / ETH / SOL)
----------------------------------------
Enthält:
- SMA- und EMA-Trends (getrennt)
- RSI, MACD, ATR, Swing-Signale
- Fear & Greed Index
- Top-News aus 4 Quellen
- Makroanalyse (BTC vs DXY & VIX)
"""

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

# ----------- App Config -----------
st.set_page_config(layout="wide", page_title="Krypto Dashboard")

WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]

# ----------- Hilfsfunktionen -----------

@st.cache_data(ttl=600)
def fetch_ohlc(symbol, months=12):
    """Holt OHLC-Daten (yfinance)"""
    try:
        df = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

def safe_get(df, col):
    """Sichere Spaltenauswahl"""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    try:
        if col in df.columns:
            return df[col].astype(float)
        else:
            m = [c for c in df.columns if col.lower() in c.lower()]
            return df[m[0]].astype(float) if m else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

def safe_val(latest, col):
    """Sichere Einzelwert-Ausgabe"""
    try:
        val = latest.get(col, np.nan)
        if isinstance(val, (list, np.ndarray, pd.Series)):
            val = val[-1]
        return float(val) if not pd.isna(val) else np.nan
    except Exception:
        return np.nan

# ----------- Indikatoren -----------

def compute_indicators(df):
    close = safe_get(df, "Close")
    high = safe_get(df, "High")
    low = safe_get(df, "Low")

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    df["EMA20"] = close.ewm(span=20).mean()
    df["EMA50"] = close.ewm(span=50).mean()
    df["EMA200"] = close.ewm(span=200).mean()

    df["RSI"] = ta.momentum.RSIIndicator(close, 14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["ATR"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

    return df

# ----------- Swing-Signale -----------

def detect_swing_signals(df):
    alerts = []
    if len(df) < 3:
        return alerts

    prev, cur = df.iloc[-2], df.iloc[-1]
    # EMA-Cross
    if prev["EMA20"] < prev["EMA50"] and cur["EMA20"] > cur["EMA50"]:
        alerts.append("🚀 EMA20 über EMA50 → Bullish Signal")
    if prev["EMA20"] > prev["EMA50"] and cur["EMA20"] < cur["EMA50"]:
        alerts.append("🔻 EMA20 unter EMA50 → Bearish Signal")
    # RSI
    if cur["RSI"] > 70:
        alerts.append(f"⚠️ RSI überkauft ({cur['RSI']:.1f})")
    elif cur["RSI"] < 30:
        alerts.append(f"⚠️ RSI überverkauft ({cur['RSI']:.1f})")
    return alerts

# ----------- Fear & Greed -----------

@st.cache_data(ttl=3600)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        js = r.json()
        return int(js["data"][0]["value"]), js["data"][0]["value_classification"]
    except Exception:
        return None, None

# ----------- News -----------

@st.cache_data(ttl=900)
def fetch_news(limit=10):
    sources = [
        "https://cointelegraph.com/rss",
        "https://www.investing.com/rss/news_25.rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://news.bitcoin.com/feed/"
    ]
    news = []
    for src in sources:
        try:
            r = requests.get(src, timeout=20)
            soup = BeautifulSoup(r.content, "xml")
            for item in soup.find_all("item")[:limit // 4]:
                news.append({
                    "title": item.title.text.strip(),
                    "link": item.link.text.strip(),
                    "source": src.split("//")[1].split("/")[0]
                })
        except Exception:
            continue
    return news

# ----------- Makro-Analyse -----------

@st.cache_data(ttl=3600)
def fetch_macro_data():
    try:
        vix = yf.download("^VIX", "2y", "1d")["Close"].resample("M").last().pct_change()
        dxy = yf.download("DX-Y.NYB", "2y", "1d")["Close"].resample("M").last().pct_change()
        btc = yf.download("BTC-USD", "2y", "1d")["Close"].resample("M").last().pct_change()
        df = pd.concat({"BTC": btc, "DXY": dxy, "VIX": vix}, axis=1).dropna()
        df.columns = ["BTC", "DXY", "VIX"]
        return df
    except Exception:
        return pd.DataFrame()

def predict_macro_trend():
    df = fetch_macro_data()
    if df.empty:
        return None, None
    X = df[["DXY", "VIX"]]
    y = df["BTC"]
    model = LinearRegression().fit(StandardScaler().fit_transform(X), y)
    pred = model.predict([StandardScaler().fit(X).transform([X.iloc[-1]])[0]])[0]
    r2 = r2_score(y, model.predict(StandardScaler().fit_transform(X)))
    return pred, r2

# ----------- UI -----------

st.title("📊 Profi Krypto-Dashboard (BTC / ETH / SOL)")
tabs = st.tabs(["Übersicht", "Charts & Indikatoren", "Makroanalyse"])

# -------- Übersicht --------
with tabs[0]:
    st.header("Schnellübersicht")
    fgi, ftxt = fetch_fear_greed()
    st.metric("Fear & Greed Index", fgi if fgi else "n/a", ftxt or "")
    st.subheader("📰 Aktuelle Krypto-News")
    news = fetch_news()
    if news:
        for n in news:
            st.markdown(f"- **[{n['source']}]** [{n['title']}]({n['link']})")
    else:
        st.warning("Keine News gefunden – evtl. RSS-Feed offline.")

# -------- Charts & Indikatoren --------
with tabs[1]:
    subtabs = st.tabs(["SMA-Trends", "EMA-Trends"])

    for sub, mode in zip(subtabs, ["SMA", "EMA"]):
        with sub:
            for ticker in WATCHLIST:
                st.subheader(f"{ticker} ({mode})")
                df = fetch_ohlc(ticker)
                if df is None or df.empty:
                    st.warning("Keine Kursdaten verfügbar.")
                    continue

                df = compute_indicators(df)
                latest = df.iloc[-1]

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Preis"
                ))

                if mode == "SMA":
                    for p in [20, 50, 200]:
                        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{p}"], name=f"SMA{p}"))
                else:
                    for p in [20, 50, 200]:
                        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA{p}"], name=f"EMA{p}", line=dict(dash="dot")))

                fig.update_layout(height=400, xaxis_title="Datum", yaxis_title="Preis (USD)")
                st.plotly_chart(fig, use_container_width=True)

                if mode == "EMA":
                    alerts = detect_swing_signals(df)
                    if alerts:
                        for a in alerts:
                            st.warning(a)
                    else:
                        st.info("Keine akuten Swing-Signale aktuell (EMA/RSI).")

# -------- Makroanalyse --------
with tabs[2]:
    st.header("🌍 Makroanalyse & Prognose")
    pred, r2 = predict_macro_trend()
    if pred is None:
        st.warning("Makroprognose nicht möglich (zu wenige Daten).")
    else:
        trend = "Bullish" if pred > 0.02 else "Bearish" if pred < -0.02 else "Neutral"
        symbol = "🟢" if trend == "Bullish" else "🔴" if trend == "Bearish" else "⚪"
        st.write(f"{symbol} Prognose nächster Monat: **{trend}** (R²={r2:.2f})")

    dfm = fetch_macro_data()
    if not dfm.empty:
        st.subheader("Monatliche Makrodaten (%)")
        st.dataframe(dfm.tail(8).round(3))

st.markdown("---")
st.caption("⚠️ Keine Finanzberatung. Nur zu Analysezwecken.")
