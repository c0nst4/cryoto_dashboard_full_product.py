# crypto_dashboard_full_product.py
# Fertige Streamlit Crypto Dashboard App
# - robust gegen MultiIndex / 2D-Close (fix für RSI-Fehler)
# - zeigt Candlesticks, SMA20/50/200, RSI, Alerts, CSV-Download

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# ---------- Defaults ----------
DEFAULT_WATCHLIST = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]
INTERVAL_OPTIONS = ["1d", "1wk", "1mo"]  # daily/weekly/monthly

# ---------- Helpers ----------

def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Wenn df MultiIndex-Spalten hat, dann flachen wir sie zu einfachen Strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def ensure_1d_series(series_like, index=None):
    """
    Stellt sicher, dass das Objekt eine 1D pd.Series ist.
    - Wenn DataFrame mit einer Spalte -> diese Spalte zurückgeben.
    - Wenn numpy array (n,1) -> ravel + pd.Series mit index.
    """
    if isinstance(series_like, pd.Series):
        return series_like
    if isinstance(series_like, pd.DataFrame):
        if series_like.shape[1] == 1:
            return series_like.iloc[:, 0]
        # falls mehrere Spalten (unexpected), nehme die erste
        return series_like.iloc[:, 0]
    # numpy array oder ähnliches
    arr = np.asarray(series_like)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    return pd.Series(arr, index=index)

@st.cache_data(show_spinner=False)
def download_ohlc(ticker: str, period_months: int = 6, interval: str = "1d") -> pd.DataFrame:
    """
    Laden von yfinance, flache MultiIndex-Spalten und return DataFrame.
    auto_adjust=True damit Split/Dividenden etc berücksichtigt sind.
    """
    period_str = f"{period_months}mo"
    try:
        df = yf.download(ticker, period=period_str, interval=interval, progress=False, auto_adjust=True)
    except Exception as e:
        raise RuntimeError(f"Fehler beim Laden von {ticker}: {e}")

    if df is None or df.empty:
        return pd.DataFrame()

    # MultiIndex-Spalten abflachen und sicherstellen, dass Index datetime ist
    df = _flatten_multiindex_columns(df)
    df.index = pd.to_datetime(df.index)
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """SMA20/50/200 und RSI hinzufügen. Sorgt dafür, dass Close 1D ist."""
    if df is None or df.empty:
        return df

    # Stelle sicher, dass wir eine gültige Close-Serie haben
    if "Close" not in df.columns:
        raise KeyError("DataFrame enthält keine 'Close'-Spalte.")

    close_raw = df["Close"]
    close = ensure_1d_series(close_raw, index=df.index).astype(float)

    # SMAs
    df["SMA20"] = close.rolling(window=20, min_periods=1).mean()
    df["SMA50"] = close.rolling(window=50, min_periods=1).mean()
    df["SMA200"] = close.rolling(window=200, min_periods=1).mean()

    # RSI: ta erwartet 1D Series -> wir geben close (Serie) weiter
    try:
        rsi = ta.momentum.RSIIndicator(close, window=14, fillna=False).rsi()
    except Exception as e:
        # Falls irgendwas schiefgeht, erzeugen wir eine NaN-Serie mit gleichem Index
        rsi = pd.Series([np.nan] * len(df), index=df.index)
    df["RSI"] = rsi

    return df

def compute_alerts(latest_row: pd.Series) -> list:
    alerts = []
    try:
        rsi = float(latest_row.get("RSI", np.nan))
        if not np.isnan(rsi):
            if rsi > 70:
                alerts.append("⚠️ RSI über 70 — mögliches Überkauft-Signal")
            elif rsi < 30:
                alerts.append("📉 RSI unter 30 — mögliches Überverkauft-Signal")
    except Exception:
        pass

    try:
        sma20 = latest_row.get("SMA20", np.nan)
        sma50 = latest_row.get("SMA50", np.nan)
        if not (np.isnan(sma20) or np.isnan(sma50)):
            if sma20 > sma50:
                alerts.append("✅ Kurzfristiger Aufwärtstrend (SMA20 > SMA50)")
            else:
                alerts.append("❌ Kurzfristiger Abwärtstrend (SMA20 <= SMA50)")
    except Exception:
        pass

    return alerts

def plot_price_and_rsi(df: pd.DataFrame, ticker: str):
    """Erstellt ein Plotly-Subplot mit Candlestick oben und RSI unten."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08, row_heights=[0.72, 0.28])

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Preis"
        ),
        row=1, col=1
    )

    # SMAs
    if "SMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", mode="lines"), row=1, col=1)
    if "SMA50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", mode="lines"), row=1, col=1)
    if "SMA200" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200", mode="lines"), row=1, col=1)

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", mode="lines"), row=2, col=1)
        # horizontale Linien 70/30
        fig.add_hline(y=70, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Preis & RSI",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Preis", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, rangemode="tozero")

    return fig

# ---------- App Layout ----------
st.title("🚀 Crypto Dashboard (fixed)")
st.markdown("Robust gegen MultiIndex / 2D-Close (RSI-Fehler behoben).")

# Sidebar - Konfiguration
with st.sidebar:
    st.header("Einstellungen")
    tickers = st.multiselect("Watchlist (Tickers)", options=DEFAULT_WATCHLIST, default=DEFAULT_WATCHLIST)
    if not tickers:
        st.warning("Bitte mindestens einen Ticker auswählen.")
    months = st.slider("Zeitraum (Monate)", min_value=1, max_value=24, value=6, step=1)
    interval = st.selectbox("Intervall", INTERVAL_OPTIONS, index=0)
    st.caption("Datenquelle: yfinance")

# Main
if not tickers:
    st.stop()

cols = st.columns(1)
for ticker in tickers:
    st.header(f"{ticker}")

    try:
        df = download_ohlc(ticker, period_months=months, interval=interval)
    except Exception as e:
        st.error(f"Fehler beim Laden von {ticker}: {e}")
        continue

    if df is None or df.empty:
        st.warning(f"Keine Daten für {ticker} (möglicherweise ungültiger Ticker oder kein Daten-Zeitraum).")
        continue

    # Indikatoren hinzufügen (sorgt auch dafür, dass Close 1D ist)
    df = add_technical_indicators(df)

    # latest row
    latest = df.iloc[-1]

    # Anzeigen grundlegender Metriken
    col1, col2, col3, col4 = st.columns(4)
    try:
        col1.metric("Close", f"{latest['Close']:.2f}")
    except Exception:
        col1.metric("Close", str(latest.get("Close", "n/a")))
    try:
        col2.metric("RSI (14)", f"{latest['RSI']:.2f}")
    except Exception:
        col2.metric("RSI (14)", str(latest.get("RSI", "n/a")))
    try:
        col3.metric("SMA20", f"{latest['SMA20']:.2f}")
    except Exception:
        col3.metric("SMA20", str(latest.get("SMA20", "n/a")))
    try:
        col4.metric("SMA50", f"{latest['SMA50']:.2f}")
    except Exception:
        col4.metric("SMA50", str(latest.get("SMA50", "n/a")))

    # Alerts
    alerts = compute_alerts(latest)
    if alerts:
        for a in alerts:
            st.info(a)
    else:
        st.write("Keine besonderen Signale.")

    # Chart
    fig = plot_price_and_rsi(df, ticker)
    st.plotly_chart(fig, use_container_width=True)

    # Daten-Tabelle (letzte 50 Zeilen) + Download
    st.subheader("Rohdaten (letzte 50 Zeilen)")
    st.dataframe(df.tail(50))

    csv = df.to_csv(index=True).encode("utf-8")
    st.download_button(label="C
