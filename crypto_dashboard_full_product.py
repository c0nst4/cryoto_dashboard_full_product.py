# ============================================================
# crypto_dashboard_full_product.py
# Vollständiges Krypto-Dashboard – BTC / ETH / SOL
# ============================================================

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
from datetime import datetime, timedelta

# ------------------------------------------------------------
# Streamlit-Setup
# ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Krypto Profi-Dashboard")
st.title("📊 Krypto Profi-Dashboard – BTC / ETH / SOL")
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]

CACHE_TTL_SHORT = 600
CACHE_TTL_LONG = 3600

# ------------------------------------------------------------
# Sicherheitsfunktionen für Daten
# ------------------------------------------------------------
def safe_series(df, col):
    """Extrahiert eine Spalte als 1D-Serie, egal wie das DF aussieht."""
    if df is None or col not in df.columns:
        return pd.Series(dtype=float)
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    arr = np.asarray(s)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return pd.Series(arr, index=df.index)

def safe_value(row, col):
    """Sicherer Zugriff auf einzelne Werte."""
    try:
        val = row.get(col, np.nan)
        if isinstance(val, (list, np.ndarray, pd.Series)):
            val = val[-1]
        return float(val)
    except Exception:
        return np.nan

# ------------------------------------------------------------
# Kursdaten abrufen
# ------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_SHORT)
def fetch_ohlc(symbol, months=12):
    try:
        df = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

# ------------------------------------------------------------
# Indikatoren berechnen (SMA, EMA, RSI, MACD, ATR)
# ------------------------------------------------------------
def compute_indicators(df):
    if df is None or df.empty:
        return df

    close = safe_series(df, "Close")
    high = safe_series(df, "High")
    low = safe_series(df, "Low")

    # SMA / EMA
    for p in (20, 50, 200):
        df[f"SMA{p}"] = close.rolling(p).mean()
        df[f"EMA{p}"] = close.ewm(span=p, adjust=False).mean()

    # RSI
    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, 14).rsi()
    except Exception:
        df["RSI"] = np.nan

    # MACD
    try:
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
        df["MACD_DIFF"] = macd.macd_diff()
    except Exception:
        df["MACD"] = df["MACD_SIGNAL"] = df["MACD_DIFF"] = np.nan

    # ATR
    try:
        atr = ta.volatility.AverageTrueRange(high, low, close)
        df["ATR"] = atr.average_true_range()
    except Exception:
        df["ATR"] = np.nan

    return df

# ------------------------------------------------------------
# Swing-Signale (EMA, RSI, MACD)
# ------------------------------------------------------------
def detect_swing_signals(df):
    if df is None or df.empty:
        return []
    alerts = []
    if len(df) < 2:
        return alerts

    prev, cur = df.iloc[-2], df.iloc[-1]
    try:
        # EMA-Cross
        if cur["EMA20"] > cur["EMA50"] and prev["EMA20"] <= prev["EMA50"]:
            alerts.append("🚀 EMA20 kreuzt über EMA50 → Bullish Signal")
        if cur["EMA20"] < cur["EMA50"] and prev["EMA20"] >= prev["EMA50"]:
            alerts.append("🔻 EMA20 kreuzt unter EMA50 → Bearish Signal")

        # RSI
        if cur["RSI"] > 70:
            alerts.append(f"⚠️ RSI überkauft ({cur['RSI']:.1f})")
        elif cur["RSI"] < 30:
            alerts.append(f"⚠️ RSI überverkauft ({cur['RSI']:.1f})")

        # MACD Cross
        if cur["MACD"] > cur["MACD_SIGNAL"] and prev["MACD"] <= prev["MACD_SIGNAL"]:
            alerts.append("📈 MACD kreuzt über Signallinie → Bullish")
        if cur["MACD"] < cur["MACD_SIGNAL"] and prev["MACD"] >= prev["MACD_SIGNAL"]:
            alerts.append("📉 MACD kreuzt unter Signallinie → Bearish")
    except Exception:
        pass
    return alerts


# ------------------------------------------------------------
# Fear & Greed Index
# ------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_LONG)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        if r.status_code == 200:
            js = r.json()
            val = int(js["data"][0]["value"])
            text = js["data"][0]["value_classification"]
            return val, text
    except Exception:
        pass
    return None, None


# ------------------------------------------------------------
# News aus mehreren Quellen (robust)
# ------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_LONG)
def fetch_news(limit=10):
    feeds = [
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("Investing", "https://www.investing.com/rss/news_25.rss"),
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Bitcoin.com", "https://news.bitcoin.com/feed/"),
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    out = []
    for src, url in feeds:
        try:
            r = requests.get(url, timeout=10, headers=headers)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.content, "xml")
            for it in soup.find_all("item")[: max(1, limit // len(feeds))]:
                out.append({
                    "source": src,
                    "title": it.title.text.strip() if it.title else "",
                    "link": it.link.text.strip() if it.link else ""
                })
        except Exception:
            continue
    if not out:
        out.append({
            "source": "System",
            "title": "Keine News gefunden – evtl. RSS-Feeds offline.",
            "link": ""
        })
    return out[:limit]


# ------------------------------------------------------------
# Wirtschaftskalender (TradingEconomics → Fallback Investing)
# ------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_LONG)
def fetch_economic_calendar(limit=10):
    try:
        r = requests.get("https://api.tradingeconomics.com/calendar?c=guest:guest", timeout=10)
        if r.status_code == 200:
            js = r.json()
            out = []
            for e in js[:limit]:
                out.append({
                    "Date": e.get("Date", "")[:10],
                    "Country": e.get("Country", ""),
                    "Event": e.get("Event", ""),
                    "Impact": e.get("Impact", "")
                })
            return out
    except Exception:
        pass
    return [{"Date": "-", "Country": "-", "Event": "Keine High-Impact Events gefunden", "Impact": "-"}]


# ------------------------------------------------------------
# Makrodaten (monatliche Returns)
# ------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_LONG)
def fetch_macro_monthly_returns():
    try:
        btc = yf.download("BTC-USD", period="2y", interval="1d", progress=False)["Close"].resample("M").last()
        vix = yf.download("^VIX", period="2y", interval="1d", progress=False)["Close"].resample("M").last()
        dxy = None
        for t in ("DX-Y.NYB", "DXY", "USDX"):
            try:
                dxy = yf.download(t, period="2y", interval="1d", progress=False)["Close"].resample("M").last()
                if not dxy.empty:
                    break
            except Exception:
                continue
        if dxy is None or dxy.empty:
            dxy = pd.Series(dtype=float)
        df = pd.concat({"BTC": btc.pct_change(), "VIX": vix.pct_change(), "DXY": dxy.pct_change()}, axis=1).dropna()
        return df
    except Exception:
        return pd.DataFrame()


# ------------------------------------------------------------
# Kombinierte Prognose (Makro + Technik)
# ------------------------------------------------------------
def hybrid_forecast(symbol):
    df_macro = fetch_macro_monthly_returns()
    df_price = fetch_ohlc(symbol, 12)
    if df_macro is None or df_macro.empty or df_price is None or df_price.empty:
        return None, None, "Keine Daten"
    df_price = compute_indicators(df_price)
    try:
        tech = df_price[["EMA20", "EMA50", "RSI", "MACD", "MACD_SIGNAL"]].dropna().resample("M").last().pct_change()
        df = pd.concat([tech, df_macro], axis=1).dropna()
        if len(df) < 6:
            return None, None, "Zu wenige Daten"
        X = df.drop(columns=["EMA20"], errors="ignore").fillna(0).values
        y = df["BTC"].values if "BTC" in df.columns else df.iloc[:, 0].values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        model = LinearRegression().fit(Xs, y)
        r2 = r2_score(y, model.predict(Xs))
        pred = model.predict(Xs[-1].reshape(1, -1))[0]
        return pred, r2, "OK"
    except Exception as e:
        return None, None, f"Fehler: {e}"


# ------------------------------------------------------------
# Plot-Funktion für SMA / EMA + RSI + MACD
# ------------------------------------------------------------
def plot_with_indicators(df, ticker, mode="EMA"):
    if df is None or df.empty:
        st.warning(f"Keine Daten für {ticker}")
        return
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Preis", increasing_line_color="green", decreasing_line_color="red"
    ))
    if mode == "SMA":
        for p in (20, 50, 200):
            if f"SMA{p}" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{p}"], name=f"SMA{p}", line=dict(width=2)))
    else:
        colors = {20: "blue", 50: "orange", 200: "purple"}
        for p in (20, 50, 200):
            if f"EMA{p}" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[f"EMA{p}"], name=f"EMA{p}",
                    line=dict(width=2, color=colors.get(p, "gray"))
                ))
    fig.update_layout(height=400, title=f"{ticker} — {mode}-Chart", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # RSI-Plot
    if "RSI" in df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="gold", width=2)))
        fig_rsi.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2, line_width=0)
        fig_rsi.update_layout(height=200, title="RSI", showlegend=False)
        st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD-Plot
    if "MACD" in df.columns:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="green", width=2)))
        if "MACD_SIGNAL" in df.columns:
            fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="red", width=2)))
        fig_macd.add_trace(go.Bar(x=df.index, y=df.get("MACD_DIFF", 0), name="Diff", opacity=0.3))
        fig_macd.update_layout(height=200, title="MACD", showlegend=True)
        st.plotly_chart(fig_macd, use_container_width=True)

# ------------------------------------------------------------
# DASHBOARD-UI
# ------------------------------------------------------------

tabs = st.tabs([
    "Übersicht",
    "Indikatoren",
    "Makroanalyse",
    "Wirtschaftskalender",
    "Swing-Signale"
])

# ---------------- Übersicht ----------------
with tabs[0]:
    st.header("🧭 Marktübersicht")

    # Fear & Greed Index
    fgi_val, fgi_text = fetch_fear_greed()
    st.metric("Fear & Greed Index", fgi_val if fgi_val else "n/a", fgi_text or "")

    # News
    st.subheader("📰 Wichtige Krypto-News")
    news = fetch_news(12)
    if not news or news[0].get("title", "").startswith("Keine"):
        st.warning("Keine aktuellen News verfügbar.")
    else:
        for n in news:
            title = n.get("title", "")
            src = n.get("source", "")
            link = n.get("link", "")
            st.markdown(f"- **[{src}]** [{title}]({link})")

# ---------------- Indikatoren ----------------
with tabs[1]:
    st.header("📈 Technische Indikatoren")
    sub_tabs = st.tabs(["SMA-Charts", "EMA-Charts"])
    for tab_mode in sub_tabs:
        mode = "SMA" if "SMA" in tab_mode.title() else "EMA"
        with tab_mode:
            for ticker in WATCHLIST:
                st.subheader(f"{ticker} ({mode})")
                df = fetch_ohlc(ticker, 12)
                if df is None or df.empty:
                    st.warning("Keine Kursdaten.")
                    continue
                df = compute_indicators(df)
                plot_with_indicators(df, ticker, mode=mode)
                alerts = detect_swing_signals(df)
                if alerts:
                    for a in alerts:
                        st.warning(a)
                else:
                    st.info("Keine aktuellen Swing-Signale erkannt.")

# ---------------- Makroanalyse ----------------
with tabs[2]:
    st.header("🌍 Kombinierte Makro- und Technik-Analyse")
    for ticker in WATCHLIST:
        st.subheader(f"{ticker}")
        pred, r2, status = hybrid_forecast(ticker)
        if pred is None:
            st.warning(f"Keine Prognose für {ticker} ({status}).")
        else:
            trend = "Bullish" if pred > 0.02 else ("Bearish" if pred < -0.02 else "Neutral")
            emoji = "🟢" if trend == "Bullish" else ("🔴" if trend == "Bearish" else "⚪")
            st.write(f"{emoji} Erwartete Entwicklung: **{trend}** — geschätzte Monatsrendite {pred*100:.2f}% (R²={r2:.2f})")

    st.markdown("### 📊 Makro-Daten (monatliche Returns)")
    dfm = fetch_macro_monthly_returns()
    if dfm is not None and not dfm.empty:
        st.dataframe(dfm.tail(12).round(4))
    else:
        st.info("Keine Makro-Daten verfügbar.")

# ---------------- Wirtschaftskalender ----------------
with tabs[3]:
    st.header("📅 Wichtige Wirtschaftstermine")
    cal = fetch_economic_calendar(15)
    if not cal:
        st.warning("Keine Ereignisse gefunden.")
    else:
        st.dataframe(pd.DataFrame(cal))

# ---------------- Swing-Signale ----------------
with tabs[4]:
    st.header("⚡ Trend- und Swing-Signale (Daily Daten)")
    for ticker in WATCHLIST:
        st.subheader(ticker)
        df = fetch_ohlc(ticker, 6)
        if df is None or df.empty:
            continue
        df = compute_indicators(df)
        alerts = detect_swing_signals(df)
        if alerts:
            for a in alerts:
                st.warning(a)
        else:
            st.info("Keine aktuellen Trendwechsel erkannt.")

st.markdown("---")
st.caption("⚠️ Indikative Signale – keine Anlageberatung. Alle Daten ohne Gewähr.")
