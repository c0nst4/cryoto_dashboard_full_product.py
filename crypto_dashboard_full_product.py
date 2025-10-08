# crypto_dashboard_full_product.py
"""
Robustes Profi Krypto-Dashboard (BTC / ETH / SOL)
Fehlerfix: sichere Extraktion von 'latest' Werten (keine ambiguous truth errors)
Enthält:
 - Kursdaten (yfinance)
 - Indikatoren: SMA20/50/200, EMA20/50/200, RSI, MACD, ATR
 - Swing-Signale: EMA-Cross + RSI Alerts
 - Fear & Greed, News, Makro-Analyse
"""

import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import ta
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="Krypto Profi-Dashboard (robust)")

# ---------------- Settings ----------------
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]

# ---------------- Helpers ----------------
@st.cache_data(ttl=600)
def fetch_ohlc(symbol, months=12):
    """Hole OHLC Daten via yfinance (robust)."""
    try:
        df = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

def safe_get_column_series(df, col):
    """
    Robust: gibt eine 1D pd.Series für Spalte 'col' zurück.
    Versucht MultiIndex / tolerante Spaltenauswahl.
    """
    try:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        # direkte Spalte
        if col in df.columns:
            s = df[col]
        else:
            # MultiIndex oder ähnliche Namen versuchen
            cols = list(df.columns)
            if isinstance(df.columns, pd.MultiIndex):
                # suche nach erstem Match
                for c in df.columns:
                    if c[-1] == col or c[0] == col or col.lower() in str(c).lower():
                        s = df[c]
                        break
                else:
                    # keine passende Spalte
                    return pd.Series(dtype=float, index=df.index)
            else:
                match = [c for c in cols if col.lower() in str(c).lower()]
                if match:
                    s = df[match[0]]
                else:
                    return pd.Series(dtype=float, index=df.index)
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.Series(s.values, index=df.index).astype(float)
    except Exception:
        try:
            return pd.Series(dtype=float, index=df.index)
        except Exception:
            return pd.Series(dtype=float)

def safe_get_latest(latest, col):
    """
    Extrahiert sicher einen numerischen Wert aus 'latest' (eine Pandas Series / Zeile).
    Rückgabe: float Wert oder np.nan.
    Vermeidet Pandas-Truth-Value-Fehler.
    """
    try:
        if latest is None:
            return np.nan
        # latest is expected to be a pd.Series (row)
        if isinstance(latest, pd.Series):
            # use .get to avoid KeyError
            val = latest.get(col, np.nan)
        else:
            # fallback - try dict-like
            try:
                val = latest[col]
            except Exception:
                return np.nan
        # if val is Series/array, take last element
        if isinstance(val, (pd.Series, np.ndarray, list)):
            if len(val) == 0:
                return np.nan
            try:
                val = val[-1]
            except Exception:
                val = val.item() if hasattr(val, "item") else val
        # final checks
        if pd.isna(val):
            return np.nan
        return float(val)
    except Exception:
        return np.nan

# ---------------- Technical indicators ----------------
def compute_indicators(df):
    """Berechnet SMA, EMA, RSI, MACD, ATR (so gut möglich)."""
    if df is None or df.empty:
        return df

    close = safe_get_column_series(df, "Close")
    high = safe_get_column_series(df, "High")
    low = safe_get_column_series(df, "Low")

    # SMA
    df["SMA20"] = close.rolling(20, min_periods=1).mean()
    df["SMA50"] = close.rolling(50, min_periods=1).mean()
    df["SMA200"] = close.rolling(200, min_periods=1).mean()

    # EMA
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # RSI
    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    except Exception:
        # simpler fallback
        delta = close.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rs = up / down.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    try:
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
    except Exception:
        df["MACD"], df["MACD_SIGNAL"] = np.nan, np.nan

    # ATR
    try:
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
        df["ATR"] = atr.average_true_range()
    except Exception:
        df["ATR"] = close.pct_change().rolling(14).std() * close

    return df

# ---------------- Swing-Signale ----------------
def detect_swing_signals(df):
    alerts = []
    if df is None or df.empty or len(df) < 3:
        return alerts
    # require columns
    for c in ["EMA20","EMA50","EMA200","RSI"]:
        if c not in df.columns:
            return alerts
    try:
        recent = df[["EMA20","EMA50","EMA200","RSI"]].dropna().tail(5)
        if recent.shape[0] >= 2:
            prev = recent.iloc[-2]
            cur = recent.iloc[-1]
            # cross EMA20/EMA50
            if float(prev["EMA20"]) < float(prev["EMA50"]) and float(cur["EMA20"]) > float(cur["EMA50"]):
                alerts.append("EMA20 kreuzt über EMA50 → möglicher Bull-Start")
            if float(prev["EMA20"]) > float(prev["EMA50"]) and float(cur["EMA20"]) < float(cur["EMA50"]):
                alerts.append("EMA20 kreuzt unter EMA50 → möglicher Bear-Start")
            # EMA50 / EMA200
            if float(prev["EMA50"]) < float(prev["EMA200"]) and float(cur["EMA50"]) > float(cur["EMA200"]):
                alerts.append("EMA50 kreuzt über EMA200 → mittelfristig Bullish")
            if float(prev["EMA50"]) > float(prev["EMA200"]) and float(cur["EMA50"]) < float(cur["EMA200"]):
                alerts.append("EMA50 kreuzt unter EMA200 → mittelfristig Bearish")
        # RSI thresholds
        rsi_val = recent["RSI"].iloc[-1]
        # rsi_val should be scalar
        try:
            rsi_num = float(rsi_val)
            if not pd.isna(rsi_num):
                if rsi_num > 70:
                    alerts.append(f"RSI überkauft ({rsi_num:.1f})")
                elif rsi_num < 30:
                    alerts.append(f"RSI überverkauft ({rsi_num:.1f})")
        except Exception:
            pass
    except Exception:
        pass
    return alerts

# ---------------- Fear & Greed ----------------
@st.cache_data(ttl=3600)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        js = r.json()
        return int(js["data"][0]["value"]), js["data"][0]["value_classification"]
    except Exception:
        return None, None

# ---------------- News ----------------
@st.cache_data(ttl=900)
def fetch_news(limit=8):
    items = []
    try:
        r = requests.get("https://cointelegraph.com/rss", timeout=8)
        soup = BeautifulSoup(r.content, "xml")
        for it in soup.find_all("item")[: limit//2]:
            items.append({"source":"CoinTelegraph", "title": it.title.text if it.title else "", "link": it.link.text if it.link else ""})
    except Exception:
        pass
    try:
        r = requests.get("https://www.investing.com/rss/news_25.rss", timeout=8)
        soup = BeautifulSoup(r.content, "xml")
        for it in soup.find_all("item")[: limit//2]:
            items.append({"source":"Investing.com", "title": it.title.text if it.title else "", "link": it.link.text if it.link else ""})
    except Exception:
        pass
    return items

# ---------------- Macro data & forecast ----------------
@st.cache_data(ttl=3600)
def fetch_macro_data():
    try:
        vix = yf.download("^VIX", period="2y", interval="1d", progress=False)["Close"].resample("M").last()
    except Exception:
        vix = pd.Series(dtype=float)
    dxy = pd.Series(dtype=float)
    for t in ["DX-Y.NYB", "DXY", "USDX"]:
        try:
            tmp = yf.download(t, period="2y", interval="1d", progress=False)["Close"].resample("M").last()
            if tmp is not None and not tmp.empty:
                dxy = tmp
                break
        except Exception:
            continue
    try:
        btc = yf.download("BTC-USD", period="2y", interval="1d", progress=False)["Close"].resample("M").last()
    except Exception:
        btc = pd.Series(dtype=float)

    try:
        df = pd.concat({"BTC": btc.pct_change(), "DXY": dxy.pct_change(), "VIX": vix.pct_change()}, axis=1).dropna()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def predict_macro_trend():
    df = fetch_macro_data()
    if df is None or df.empty or len(df) < 6:
        return None, None
    try:
        X = df[["DXY","VIX"]].values
        y = df["BTC"].values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        model = LinearRegression().fit(Xs, y)
        r2 = float(r2_score(y, model.predict(Xs)))
        lastX = Xs[-1].reshape(1, -1)
        pred = float(model.predict(lastX)[0])
        return pred, r2
    except Exception:
        return None, None

# ---------------- UI ----------------
st.title("📊 Profi Krypto-Dashboard (BTC / ETH / SOL)")
tabs = st.tabs(["Übersicht", "Charts & Indikatoren", "Makro-Analyse"])

# --- Übersicht ---
with tabs[0]:
    st.header("Schnellübersicht")
    fgi, fgi_text = fetch_fear_greed()
    st.metric("Fear & Greed", fgi if fgi is not None else "n/a", fgi_text if fgi_text else "")

    st.subheader("Top News")
    news = fetch_news(8)
    if news:
        for n in news:
            title = n.get("title","")[:180]
            src = n.get("source","")
            link = n.get("link","")
            if link:
                st.markdown(f"- **[{src}]** [{title}]({link})")
            else:
                st.markdown(f"- **[{src}]** {title}")
    else:
        st.write("Keine News verfügbar.")

# --- Charts & Indikatoren ---
with tabs[1]:
    st.header("Charts & Indikatoren (SMA & EMA + Signale)")
    for ticker in WATCHLIST:
        st.subheader(ticker)
        df = fetch_ohlc(ticker, months=14)
        if df is None or df.empty:
            st.warning(f"Keine Preisdaten für {ticker} verfügbar.")
            continue

        df = compute_indicators(df)
        if df is None or df.empty:
            st.warning("Indikatoren konnten nicht berechnet werden.")
            continue

        # sichere Extraktion der neuesten Werte
        try:
            latest = df.dropna(subset=["Close"]).iloc[-1]
        except Exception:
            latest = df.iloc[-1] if not df.empty else None

        latest_close = safe_get_latest(latest, "Close")
        latest_rsi = safe_get_latest(latest, "RSI")
        latest_ema20 = safe_get_latest(latest, "EMA20")
        latest_atr = safe_get_latest(latest, "ATR")

        cols = st.columns(4)
        cols[0].metric("Close (USD)", f"{latest_close:.2f}" if not np.isnan(latest_close) else "n/a")
        cols[1].metric("RSI", f"{latest_rsi:.1f}" if not np.isnan(latest_rsi) else "n/a")
        cols[2].metric("EMA20", f"{latest_ema20:.2f}" if not np.isnan(latest_ema20) else "n/a")
        cols[3].metric("ATR (14)", f"{latest_atr:.6f}" if not np.isnan(latest_atr) else "n/a")

        # Chart
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Candles")])
        if "SMA20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(width=1)))
        if "SMA50" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(width=1)))
        if "SMA200" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200", line=dict(width=1)))
        if "EMA20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(dash="dash")))
        if "EMA50" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(dash="dash")))
        if "EMA200" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200", line=dict(dash="dash")))

        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Signals
        alerts = detect_swing_signals(df)
        if alerts:
            for a in alerts:
                st.warning(a)
        else:
            st.success("Keine akuten Swing-Signale (EMA/RSI)")

        # show last rows
        show_cols = [c for c in ["Close","EMA20","EMA50","EMA200","SMA20","SMA50","SMA200","RSI","MACD","ATR"] if c in df.columns]
        try:
            st.dataframe(df[show_cols].tail(7).round(6))
        except Exception:
            pass

# --- Makro-Analyse ---
with tabs[2]:
    st.header("Makro-Analyse & Prognose")
    pred, r2 = predict_macro_trend()
    if pred is None:
        st.info("Makro-Prognose nicht möglich (zu wenige Makrodaten).")
    else:
        trend = "Bullish" if pred > 0.02 else ("Bearish" if pred < -0.02 else "Neutral")
        emoji = "🟢" if trend == "Bullish" else ("🔴" if trend == "Bearish" else "⚪")
        st.write(f"{emoji} Prognose nächster Monat: {trend} (geschätzte Rendite: {pred*100:.2f}% , R²={r2:.3f})")

    dfm = fetch_macro_data()
    if dfm is not None and not dfm.empty:
        st.subheader("Makrodaten (letzte Monate):")
        try:
            st.dataframe(dfm.tail(8).round(6))
        except Exception:
            st.write(dfm.tail(8).round(6))

# Footer
st.markdown("---")
st.write("⚠️ Hinweis: Das Dashboard liefert Indikationen, keine Anlageberatung.")
