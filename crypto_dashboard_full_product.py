# crypto_dashboard_full_product.py
# Teil 1/3
"""
Krypto Dashboard — Pro-Version (robust + RSI/MACD + verbesserte Prognosen + News/Calendar Fallback)
Instruktion: Teile 1,2,3 nacheinander in crypto_dashboard_full_product.py einfügen.
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
from datetime import datetime, timedelta
import time
import math
import typing
import traceback

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="Krypto Hybrid-Pro Dashboard")

# ---------------- Settings ----------------
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]
TIMEFRAMES = {
    "4h": {"interval": "4h", "period_days": 90},
    "D":  {"interval": "1d", "period_days": 365},
    "W":  {"interval": "1wk", "period_days": 730}
}
MIN_DATA_POINTS_FORECAST = 30   # abgesenkt für Robustheit
CACHE_TTL_SHORT = 300
CACHE_TTL_MED = 900
CACHE_TTL_LONG = 3600

# ---------------- Helpers ----------------
def _first_numeric_column_from_df(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            return s
    try:
        return df.iloc[:, 0].astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

def safe_series_from_df(df: pd.DataFrame, col_name: str) -> pd.Series:
    if df is None:
        return pd.Series(dtype=float)
    try:
        if col_name in df.columns:
            s = df[col_name]
        else:
            cols = df.columns
            if isinstance(cols, pd.MultiIndex):
                match = None
                for c in cols:
                    if col_name == c[-1] or col_name == c[0] or col_name.lower() in str(c[-1]).lower():
                        match = c
                        break
                s = df[match] if match else pd.Series(dtype=float, index=df.index)
            else:
                match = [c for c in cols if col_name.lower() in str(c).lower()]
                s = df[match[0]] if match else pd.Series(dtype=float, index=df.index)
        if isinstance(s, pd.DataFrame):
            s = _first_numeric_column_from_df(s)
        arr = np.asarray(s)
        if arr.ndim > 1:
            arr = arr[:, 0]
        return pd.Series(arr, index=df.index).astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

def safe_get_latest_value(latest_row, col: str) -> float:
    try:
        if latest_row is None:
            return np.nan
        if isinstance(latest_row, pd.Series):
            val = latest_row.get(col, np.nan)
        elif isinstance(latest_row, dict):
            val = latest_row.get(col, np.nan)
        else:
            return np.nan
        if isinstance(val, (list, np.ndarray, pd.Series)):
            val = np.array(val).flatten()[-1] if len(val) > 0 else np.nan
        return float(val) if not pd.isna(val) else np.nan
    except Exception:
        return np.nan

# ---------------- OHLC Fetching ----------------
@st.cache_data(ttl=CACHE_TTL_SHORT)
def fetch_ohlc_yf(symbol: str, interval: str, period_days: int) -> typing.Optional[pd.DataFrame]:
    try:
        if interval == "4h":
            per = f"{max(30, min(period_days, 120))}d"
            df = yf.download(symbol, period=per, interval="60m", progress=False)
            if df is None or df.empty:
                return None
            try:
                df_4h = df.resample("4H").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
                df_4h.dropna(how="any", inplace=True)
                return df_4h
            except Exception:
                return df
        elif interval in ("1d","1wk","1w"):
            if interval in ("1wk","1w"):
                df = yf.download(symbol, period=f"{period_days*2}d", interval="1wk", progress=False)
            else:
                df = yf.download(symbol, period=f"{period_days}d", interval="1d", progress=False)
            if df is None or df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            return df
        else:
            df = yf.download(symbol, period=f"{period_days}d", interval=interval, progress=False)
            if df is None or df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            return df
    except Exception:
        return None

# ---------------- Indicators ----------------
def compute_indicators_for_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    close = safe_series_from_df(df, "Close")
    high = safe_series_from_df(df, "High")
    low = safe_series_from_df(df, "Low")
    if close.empty:
        return df

    df["SMA20"] = close.rolling(20, min_periods=1).mean()
    df["SMA50"] = close.rolling(50, min_periods=1).mean()
    df["SMA200"] = close.rolling(200, min_periods=1).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    except Exception:
        delta = close.diff()
        up = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        down = -delta.clip(upper=0).rolling(14, min_periods=1).mean()
        rs = up / down.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

    try:
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
        df["MACD_DIFF"] = df["MACD"] - df["MACD_SIGNAL"]
    except Exception:
        df["MACD"] = df["MACD_SIGNAL"] = df["MACD_DIFF"] = np.nan

    try:
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
        df["ATR"] = atr.average_true_range()
    except Exception:
        df["ATR"] = close.pct_change().rolling(14).std() * close

    return df

@st.cache_data(ttl=CACHE_TTL_SHORT)
def get_tf_indicators(symbol: str) -> dict:
    out = {}
    for tf, meta in TIMEFRAMES.items():
        df = fetch_ohlc_yf(symbol, meta["interval"], meta["period_days"])
        df = compute_indicators_for_df(df) if df is not None else None
        out[tf] = df
    return out

# ---------------- Signals ----------------
def detect_signals_multi_tf(tf_dfs: dict) -> dict:
    res = {}
    for tf, df in tf_dfs.items():
        arr = []
        if df is None or df.empty:
            res[tf] = arr; continue
        need = ["EMA20","EMA50","EMA200","SMA50","SMA200","MACD","MACD_SIGNAL","RSI"]
        sub = df[need].dropna(how="any")
        if sub.shape[0] < 2:
            res[tf] = arr; continue
        prev, cur = sub.iloc[-2], sub.iloc[-1]
        try:
            if prev["EMA20"] < prev["EMA50"] and cur["EMA20"] > cur["EMA50"]:
                arr.append("EMA20↑ über EMA50 (bullish)")
            if prev["EMA20"] > prev["EMA50"] and cur["EMA20"] < cur["EMA50"]:
                arr.append("EMA20↓ unter EMA50 (bearish)")
            if prev["SMA50"] < prev["SMA200"] and cur["SMA50"] > cur["SMA200"]:
                arr.append("SMA50↑ über SMA200 (Golden Cross möglich)")
            if prev["SMA50"] > prev["SMA200"] and cur["SMA50"] < cur["SMA200"]:
                arr.append("SMA50↓ unter SMA200 (Death Cross möglich)")
            if prev["MACD"] < prev["MACD_SIGNAL"] and cur["MACD"] > cur["MACD_SIGNAL"]:
                arr.append("MACD kreuzt über Signal (bullish momentum)")
            if prev["MACD"] > prev["MACD_SIGNAL"] and cur["MACD"] < cur["MACD_SIGNAL"]:
                arr.append("MACD kreuzt unter Signal (bearish momentum)")
            if cur["RSI"] > 70:
                arr.append(f"RSI überkauft ({cur['RSI']:.1f})")
            if cur["RSI"] < 30:
                arr.append(f"RSI überverkauft ({cur['RSI']:.1f})")
        except Exception:
            pass
        res[tf] = arr
    return res

# ---------------- Improved Macro Data & Forecast helpers ----------------
@st.cache_data(ttl=CACHE_TTL_LONG)
def fetch_macro_features_daily() -> pd.DataFrame:
    try:
        btc = yf.download("BTC-USD", period="3y", interval="1d", progress=False)["Close"]
    except Exception:
        btc = pd.Series(dtype=float)
    dxy = pd.Series(dtype=float)
    for t in ("DX-Y.NYB","DXY","USDX"):
        try:
            tmp = yf.download(t, period="3y", interval="1d", progress=False)["Close"]
            if tmp is not None and not tmp.empty:
                dxy = tmp; break
        except Exception:
            continue
    if dxy is None or dxy.empty:
        try:
            eurusd = yf.download("EURUSD=X", period="3y", interval="1d", progress=False)["Close"]
            dxy = 100 / eurusd if eurusd is not None and not eurusd.empty else pd.Series(dtype=float)
        except Exception:
            dxy = pd.Series(dtype=float)
    try:
        vix = yf.download("^VIX", period="3y", interval="1d", progress=False)["Close"]
    except Exception:
        vix = pd.Series(dtype=float)
    try:
        df = pd.concat({"BTC": btc, "DXY": dxy, "VIX": vix}, axis=1)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL_LONG)
def fetch_macro_monthly_returns():
    try:
        df_daily = fetch_macro_features_daily()
        if df_daily is None or df_daily.empty:
            return pd.DataFrame()
        df_month = df_daily.resample("M").last()
        df_month = df_month.pct_change().dropna()
        return df_month
    except Exception:
        return pd.DataFrame()

# -----------------------------------------------------------
# --- Teil 2/3: Forecasts, Makro-Modell & Chart Plotting ----
# -----------------------------------------------------------

def hybrid_forecast(symbol: str) -> tuple[float | None, float | None, str]:
    """
    Kombiniert Makro-Faktoren (DXY, VIX) und technische Indikatoren (EMA, RSI, MACD)
    zur groben Richtungsprognose.
    Rückgabe: (pred_return, r2_score, status_text)
    """
    df_macro = fetch_macro_monthly_returns()
    if df_macro is None or df_macro.empty:
        return None, None, "Keine Makro-Daten"

    df_tf = get_tf_indicators(symbol)
    if "D" not in df_tf or df_tf["D"] is None or df_tf["D"].empty:
        return None, None, "Keine technischen Daten"

    df_daily = df_tf["D"].copy()
    df_daily = df_daily[["Close", "EMA20", "EMA50", "RSI", "MACD", "MACD_SIGNAL"]].dropna()
    if df_daily.empty:
        return None, None, "Zu wenige technische Daten"

    try:
        tech = df_daily.resample("M").last().pct_change().dropna()
        macro = df_macro.copy().loc[tech.index.intersection(df_macro.index)]
        df = pd.concat([tech, macro], axis=1).dropna()
        if df.shape[0] < 6:
            return None, None, "Zu wenige gemeinsame Datenpunkte"

        y = df["Close"]
        X = df.drop(columns=["Close"])
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        model = LinearRegression().fit(Xs, y)
        r2 = r2_score(y, model.predict(Xs))
        pred = model.predict(Xs[-1].reshape(1, -1))[0]
        return float(pred), float(r2), f"{len(df)} Datenpunkte"
    except Exception as e:
        return None, None, f"Fehler: {e}"

# -------------------- Wirtschaftskalender --------------------
@st.cache_data(ttl=CACHE_TTL_LONG)
def fetch_economic_calendar(limit: int = 10):
    """
    Holt die wichtigsten kommenden Ereignisse aus TradingEconomics oder Investing-Fallback.
    """
    out = []
    try:
        r = requests.get("https://api.tradingeconomics.com/calendar", timeout=10)
        if r.status_code == 200:
            js = r.json()
            for e in js[:limit]:
                if "Country" in e and "Event" in e:
                    out.append({
                        "Country": e.get("Country", ""),
                        "Event": e.get("Event", ""),
                        "Impact": e.get("Impact", ""),
                        "Date": e.get("Date", "")[:10]
                    })
            return out
    except Exception:
        pass
    # fallback Investing.com
    try:
        r = requests.get("https://www.investing.com/economic-calendar/", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            rows = soup.find_all("tr", {"id": lambda x: x and x.startswith("eventRowId")})
            for r_ in rows[:limit]:
                event = r_.get("event_attr_name", "")
                country = r_.get("event_country", "")
                out.append({"Country": country, "Event": event, "Impact": "", "Date": ""})
    except Exception:
        pass
    return out or [{"Country": "-", "Event": "Keine High-Impact Events gefunden (API-Limit oder keine Daten).", "Impact": "", "Date": ""}]

# -------------------- Plot Helpers --------------------
def plot_with_indicators(df: pd.DataFrame, ticker: str, mode: str = "EMA") -> None:
    """
    mode = "SMA" oder "EMA" — zeigt Candlestick, Indikatoren und Subplots (RSI & MACD)
    """
    if df is None or df.empty:
        st.warning(f"Keine Daten für {ticker}")
        return

    # Preis-Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Kurs", increasing_line_color="green", decreasing_line_color="red"
    ))

    # Linien
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

    fig.update_layout(title=f"{ticker} — {mode}-Chart", height=400, showlegend=True)

    # RSI-Subplot
    if "RSI" in df.columns:
        rsi = go.Figure()
        rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="gold", width=2)))
        rsi.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2, line_width=0)
        rsi.update_layout(title="RSI", height=200, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(rsi, use_container_width=True)
    else:
        st.plotly_chart(fig, use_container_width=True)

    # MACD-Subplot
    if "MACD" in df.columns:
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="green", width=2)))
        if "MACD_SIGNAL" in df.columns:
            macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="red", width=2)))
        macd_fig.add_trace(go.Bar(x=df.index, y=df.get("MACD_DIFF", 0), name="Diff", opacity=0.3))
        macd_fig.update_layout(title="MACD", height=200, showlegend=True)
        st.plotly_chart(macd_fig, use_container_width=True)

# -----------------------------------------------------------
# --- Teil 3/3: Dashboard Layout & UI ------------------------
# -----------------------------------------------------------

st.title("📊 Krypto Profi-Dashboard – BTC / ETH / SOL")

tabs = st.tabs(["Übersicht", "Indikatoren", "Makroanalyse", "Wirtschaftskalender", "Swing-Signale"])

# ---------- Übersicht ----------
with tabs[0]:
    st.header("🧭 Marktübersicht")

    # Fear & Greed Index
    fgi_val, fgi_text = fetch_fear_greed()
    st.metric("Fear & Greed Index", fgi_val if fgi_val else "n/a", fgi_text or "")

    # News
    st.subheader("📰 Wichtige Krypto-News")
    news = fetch_news(12)
    if not news or news[0].get("title","").startswith("Keine"):
        st.warning("Keine aktuellen News verfügbar.")
    else:
        for n in news:
            title = n.get("title","")
            src = n.get("source","")
            link = n.get("link","")
            st.markdown(f"- **[{src}]** [{title}]({link})")

# ---------- Indikatoren ----------
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
                    st.info("Keine akuten Swing-Signale erkannt.")

# ---------- Makroanalyse ----------
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

# ---------- Wirtschaftskalender ----------
with tabs[3]:
    st.header("📅 Wichtige Wirtschaftstermine")
    cal = fetch_economic_calendar(15)
    if not cal:
        st.warning("Keine Ereignisse gefunden.")
    else:
        st.dataframe(pd.DataFrame(cal))

# ---------- Swing-Signale ----------
with tabs[4]:
    st.header("⚡ Trend- und Swing-Signale (4h, D, W)")
    for ticker in WATCHLIST:
        st.subheader(ticker)
        df_tfs = get_tf_indicators(ticker)
        found = False
        for tf, df in df_tfs.items():
            if df is None or df.empty:
                continue
            alerts = detect_swing_signals(df)
            if alerts:
                found = True
                st.markdown(f"**{tf}-Chart**:")
                for a in alerts:
                    st.warning(a)
        if not found:
            st.info("Keine aktuellen Trendwechsel gefunden.")

st.markdown("---")
st.caption("⚠️ Indikative Signale – keine Anlageberatung. Daten & Prognosen ohne Gewähr.")
