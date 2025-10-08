# crypto_dashboard_full_product.py
"""
Robustes Profi Krypto-Dashboard (BTC / ETH / SOL)
Fixes:
 - Sichere Extraktion von Spalten / 1D-Serien (vermeidet ta-Fehler)
 - Robustere News-Abfrage (mehr Feeds, längere Timeouts, Fallback)
 - Getrennte SMA- & EMA-Ansichten (Tabs)
 - Swing-Signale (EMA-Cross, RSI)
 - Makro-Analyse (DXY, VIX vs BTC)
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

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="Krypto Profi-Dashboard (robust)")

# ---------------- Settings ----------------
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]

# ---------------- Helpers: sichere Series-Extraktion ----------------
def _first_numeric_column_from_df(df):
    """Wenn df eine DataFrame mit mehreren Spalten ist, gib erste numerische Spalte zurück."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            return s
    # fallback: take first column cast to numeric
    try:
        return df.iloc[:, 0].astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

def safe_series_from_df(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Robust: Extrahiere eine 1D pandas Series für 'col_name' aus DataFrame df.
    - Handhabt MultiIndex-Spalten, leichte Namensabweichungen, DataFrame statt Series, numpy-Arrays.
    - Gibt eine pd.Series mit dem gleichen Index wie df zurück (oder leere Series).
    """
    if df is None:
        return pd.Series(dtype=float)
    try:
        # direct match
        if col_name in df.columns:
            s = df[col_name]
        else:
            cols = df.columns
            # multiindex handling
            if isinstance(cols, pd.MultiIndex):
                # try to match any level equal or containing col_name
                match = None
                for c in cols:
                    if col_name == c[-1] or col_name == c[0] or col_name.lower() in str(c[-1]).lower() or col_name.lower() in str(c[0]).lower():
                        match = c
                        break
                if match is not None:
                    s = df[match]
                else:
                    # try substring on flattened names
                    flat = [str(c) for c in cols]
                    m = [cols[i] for i, name in enumerate(flat) if col_name.lower() in name.lower()]
                    s = df[m[0]] if m else pd.Series(dtype=float, index=df.index)
            else:
                # non-multiindex: substring match
                match = [c for c in cols if col_name.lower() in str(c).lower()]
                s = df[match[0]] if match else pd.Series(dtype=float, index=df.index)

        # if result is dataframe, choose first numeric column
        if isinstance(s, pd.DataFrame):
            s = _first_numeric_column_from_df(s)

        # coerce to numpy and ensure 1D
        arr = np.asarray(s)
        if arr.ndim > 1:
            # shape could be (n,1) or (n,m)
            if arr.shape[1] == 1:
                arr = arr.ravel()
            else:
                arr = arr[:, 0]
        # build pandas Series with same index as df (if possible)
        try:
            s2 = pd.Series(arr, index=df.index).astype(float)
        except Exception:
            # fallback: ignore index
            s2 = pd.Series(arr.flatten()).astype(float)
        return s2
    except Exception:
        # safe fallback
        try:
            return pd.Series(dtype=float, index=df.index)
        except Exception:
            return pd.Series(dtype=float)

def safe_get_latest_value(latest_row, col):
    """
    Sicherer Zugriff auf einen Wert in einer pd.Series (latest_row).
    Gibt float oder np.nan zurück.
    Vermeidet Pandas 'ambiguous truth value' Fehler.
    """
    try:
        if latest_row is None:
            return np.nan
        # latest_row may be pd.Series
        if isinstance(latest_row, pd.Series):
            val = latest_row.get(col, np.nan)
        else:
            # dict-like fallback
            val = latest_row[col] if (isinstance(latest_row, dict) and col in latest_row) else np.nan
        # if val is array-like -> take last element
        if isinstance(val, (list, np.ndarray, pd.Series)):
            if len(val) == 0:
                return np.nan
            try:
                val = val[-1]
            except Exception:
                try:
                    val = np.array(val).flatten()[-1]
                except Exception:
                    return np.nan
        if pd.isna(val):
            return np.nan
        return float(val)
    except Exception:
        return np.nan

# ---------------- Fetch OHLC ----------------
@st.cache_data(ttl=600)
def fetch_ohlc(symbol: str, months: int = 12) -> pd.DataFrame | None:
    """Hole OHLC Daten via yfinance. Gibt None bei Fehlern/leer."""
    try:
        df = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

# ---------------- Compute Indicators (robust) ----------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet SMA/EMA/RSI/MACD/ATR und schreibt die Spalten in df.
    Die Eingabe df bleibt inhaltlich erhalten; es werden Spalten hinzugefügt.
    """
    if df is None or df.empty:
        return df

    # sichere Series-Extraktion (1D)
    close = safe_series_from_df(df, "Close")
    high = safe_series_from_df(df, "High")
    low = safe_series_from_df(df, "Low")

    # If series lengths mismatch or are empty — return df early
    if close is None or close.size == 0:
        return df

    # simple moving averages (SMA)
    try:
        df["SMA20"] = close.rolling(window=20, min_periods=1).mean()
        df["SMA50"] = close.rolling(window=50, min_periods=1).mean()
        df["SMA200"] = close.rolling(window=200, min_periods=1).mean()
    except Exception:
        df["SMA20"] = np.nan
        df["SMA50"] = np.nan
        df["SMA200"] = np.nan

    # exponential moving averages (EMA)
    try:
        df["EMA20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA50"] = close.ewm(span=50, adjust=False).mean()
        df["EMA200"] = close.ewm(span=200, adjust=False).mean()
    except Exception:
        df["EMA20"] = np.nan
        df["EMA50"] = np.nan
        df["EMA200"] = np.nan

    # RSI (ta) with fallback
    try:
        # ta expects pd.Series (1D) — we pass close which is 1D
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    except Exception:
        # fallback approximate RSI
        try:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
            loss = -delta.clip(upper=0).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            df["RSI"] = 100 - (100 / (1 + rs))
        except Exception:
            df["RSI"] = np.nan

    # MACD (ta) with fallback
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
        try:
            df["ATR"] = close.pct_change().rolling(14, min_periods=1).std() * close
        except Exception:
            df["ATR"] = np.nan

    return df

# ---------------- Swing Signals ----------------
def detect_swing_signals(df: pd.DataFrame) -> list:
    """
    Ermittelt EMA-Crosses (20/50, 50/200) und RSI Extremwerte.
    Liefert Liste von Strings (Alerts). Sehr robust gegenüber NaNs.
    """
    alerts = []
    if df is None or df.empty:
        return alerts

    # ensure necessary columns exist
    for col in ["EMA20", "EMA50", "EMA200", "RSI"]:
        if col not in df.columns:
            return alerts

    # take last two valid rows (non-NaN for these columns)
    subset = df[["EMA20", "EMA50", "EMA200", "RSI"]].dropna()
    if subset.shape[0] < 2:
        return alerts
    prev = subset.iloc[-2]
    cur = subset.iloc[-1]

    try:
        p_e20 = float(prev["EMA20"]); p_e50 = float(prev["EMA50"])
        c_e20 = float(cur["EMA20"]); c_e50 = float(cur["EMA50"])
        p_e50 = float(prev["EMA50"]); p_e200 = float(prev["EMA200"])
        c_e50 = float(cur["EMA50"]); c_e200 = float(cur["EMA200"])
    except Exception:
        return alerts

    # EMA20 x EMA50
    if p_e20 < p_e50 and c_e20 > c_e50:
        alerts.append("🚀 EMA20 kreuzt über EMA50 → mögliches bullishes Signal")
    if p_e20 > p_e50 and c_e20 < c_e50:
        alerts.append("🔻 EMA20 kreuzt unter EMA50 → mögliches bearishes Signal")

    # EMA50 x EMA200
    try:
        if p_e50 < p_e200 and c_e50 > c_e200:
            alerts.append("📈 EMA50 kreuzt über EMA200 → mittelfristig Bullish")
        if p_e50 > p_e200 and c_e50 < c_e200:
            alerts.append("📉 EMA50 kreuzt unter EMA200 → mittelfristig Bearish")
    except Exception:
        pass

    # RSI thresholds
    try:
        rsi_val = float(cur["RSI"])
        if rsi_val > 70:
            alerts.append(f"⚠️ RSI überkauft ({rsi_val:.1f})")
        elif rsi_val < 30:
            alerts.append(f"⚠️ RSI überverkauft ({rsi_val:.1f})")
    except Exception:
        pass

    return alerts

# ---------------- Fear & Greed ----------------
@st.cache_data(ttl=3600)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=12)
        r.raise_for_status()
        js = r.json()
        val = int(js["data"][0]["value"])
        label = js["data"][0]["value_classification"]
        return val, label
    except Exception:
        return None, None

# ---------------- News Fetch (robust) ----------------
@st.cache_data(ttl=900)
def fetch_news(limit: int = 10) -> list:
    """
    Versucht mehrere RSS/News-Feeds (CoinTelegraph, Investing, CoinDesk, Bitcoin.com, Google News).
    Liefert Liste von {source,title,link,date} (wenn möglich). Falls nichts gefunden wird,
    wird eine erklärende Fallback-Nachricht zurückgegeben.
    """
    feeds = [
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("Investing", "https://www.investing.com/rss/news_25.rss"),
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Bitcoin.com", "https://news.bitcoin.com/feed/"),
        ("GoogleCrypto", "https://news.google.com/rss/search?q=crypto OR bitcoin OR ethereum&hl=en-US&gl=US&ceid=US:en")
    ]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; my-app/1.0)"}
    out = []
    for name, url in feeds:
        try:
            r = requests.get(url, timeout=15, headers=headers)
            if r.status_code != 200:
                continue
            # try xml parsing
            soup = BeautifulSoup(r.content, "xml")
            items = soup.find_all("item")
            for it in items[: max(1, limit // len(feeds))]:
                title = it.title.text if it.title else ""
                link = it.link.text if it.link else (it.find("guid").text if it.find("guid") else "")
                pub = it.pubDate.text if it.pubDate else ""
                out.append({"source": name, "title": title.strip(), "link": link.strip(), "date": pub})
            # stop early if we already have enough
            if len(out) >= limit:
                break
        except Exception:
            continue

    # deduplicate by title
    seen = set()
    dedup = []
    for item in out:
        key = (item.get("title",""))[:200]
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    if not dedup:
        # fallback message telling why
        return [{"source": "system", "title": "Keine News verfügbar — prüfe Internetzugang oder RSS-Feeds", "link": "", "date": ""}]
    return dedup[:limit]

# ---------------- Macro Data & Forecast ----------------
@st.cache_data(ttl=3600)
def fetch_macro_data():
    """
    Holt monatliche Werte / %Änderungen für VIX, DXY, BTC (resampled monthly).
    Liefert DataFrame mit Spalten ['BTC','DXY','VIX'] (monatliche returns) oder leeres DF.
    """
    try:
        vix = yf.download("^VIX", period="2y", interval="1d", progress=False)["Close"].resample("M").last()
    except Exception:
        vix = pd.Series(dtype=float)
    # try multiple tickers for DXY
    dxy = pd.Series(dtype=float)
    for t in ("DX-Y.NYB", "DXY", "USDX"):
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
        df = pd.concat({"BTC": btc.pct_change(), "DXY": dxy.pct_change(), "VIX": vix.pct_change()}, axis=1)
        # flatten possible multiindex names to simple names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna()
        return df
    except Exception:
        return pd.DataFrame()

def predict_macro_trend():
    """
    Train a small linear model: BTC_monthly_return ~ DXY + VIX
    returns pred (float) and r2 (float) or (None,None)
    """
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
        lastX = Xs[-1].reshape(1,-1)
        pred = float(model.predict(lastX)[0])
        return pred, r2
    except Exception:
        return None, None

# ---------------- UI ----------------
st.title("📊 Profi Krypto-Dashboard (robust)")

tabs = st.tabs(["Übersicht", "Charts & Indikatoren", "Makro-Analyse"])

# ------------- ÜBERSICHT -------------
with tabs[0]:
    st.header("Schnellübersicht")
    fgi_val, fgi_text = fetch_fear_greed()
    st.metric("Fear & Greed Index", fgi_val if fgi_val is not None else "n/a", fgi_text if fgi_text else "")

    st.subheader("Top News")
    news_list = fetch_news(10)
    if not news_list:
        st.warning("Keine News gefunden.")
    else:
        for n in news_list:
            title = n.get("title","")[:200]
            src = n.get("source","")
            link = n.get("link","")
            if link:
                st.markdown(f"- **[{src}]** [{title}]({link})")
            else:
                st.markdown(f"- **[{src}]** {title}")

# ------------- CHARTS & INDIKATOREN -------------
with tabs[1]:
    st.header("Charts & Indikatoren")
    sub_tabs = st.tabs(["SMA-Trends", "EMA-Trends"])
    # SMA Tab
    with sub_tabs[0]:
        for ticker in WATCHLIST:
            st.subheader(f"{ticker} — SMA")
            df = fetch_ohlc(ticker, months=14)
            if df is None:
                st.warning(f"Keine Daten für {ticker}.")
                continue
            df = compute_indicators(df)
            if df is None or df.empty:
                st.warning("Indikatoren konnten nicht berechnet werden.")
                continue
            # Get latest safely
            latest = None
            try:
                latest = df.dropna(subset=["Close"]).iloc[-1]
            except Exception:
                latest = df.iloc[-1] if not df.empty else None

            close_v = safe_get_latest_value(latest, "Close")
            rsi_v = safe_get_latest_value(latest, "RSI")
            st.write(f"Close: {close_v:.2f} USD" if not np.isnan(close_v) else "Close: n/a")
            # plot candle + SMA
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")])
            for p in (20,50,200):
                if f"SMA{p}" in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{p}"], name=f"SMA{p}", line=dict(width=1)))
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    # EMA Tab
    with sub_tabs[1]:
        for ticker in WATCHLIST:
            st.subheader(f"{ticker} — EMA")
            df = fetch_ohlc(ticker, months=14)
            if df is None:
                st.warning(f"Keine Daten für {ticker}.")
                continue
            df = compute_indicators(df)
            if df is None or df.empty:
                st.warning("Indikatoren konnten nicht berechnet werden.")
                continue

            latest = None
            try:
                latest = df.dropna(subset=["Close"]).iloc[-1]
            except Exception:
                latest = df.iloc[-1] if not df.empty else None

            close_v = safe_get_latest_value(latest, "Close")
            ema20_v = safe_get_latest_value(latest, "EMA20")
            st.write(f"Close: {close_v:.2f} USD" if not np.isnan(close_v) else "Close: n/a")
            # plot candle + EMA
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")])
            for p in (20,50,200):
                if f"EMA{p}" in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA{p}"], name=f"EMA{p}", line=dict(dash="dash")))
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

            # signals
            alerts = detect_swing_signals(df)
            if alerts:
                for a in alerts:
                    st.warning(a)
            else:
                st.info("Keine akuten Swing-Signale (EMA/RSI)")

            # small table last 7 rows
            cols_show = [c for c in ["Close","EMA20","EMA50","EMA200","RSI","ATR"] if c in df.columns]
            try:
                st.dataframe(df[cols_show].tail(7).round(6))
            except Exception:
                pass

# ------------- MAKRO-ANALYSE -------------
with tabs[2]:
    st.header("Makro-Analyse & Prognose")
    pred, r2 = predict_macro_trend()
    if pred is None:
        st.info("Makro-Prognose nicht möglich (zu wenige Makro-Daten).")
    else:
        trend = "Bullish" if pred > 0.02 else ("Bearish" if pred < -0.02 else "Neutral")
        emoji = "🟢" if trend == "Bullish" else ("🔴" if trend == "Bearish" else "⚪")
        st.write(f"{emoji} Prognose nächster Monat: **{trend}** (geschätzte Rendite: {pred*100:.2f}% , R²={r2:.3f})")

    dfm = fetch_macro_data()
    if dfm is None or dfm.empty:
        st.write("Makrodaten nicht verfügbar oder zu wenige Daten.")
    else:
        st.subheader("Monatliche Makrodaten (Returns)")
        try:
            st.dataframe(dfm.tail(12).round(4))
        except Exception:
            st.write(dfm.tail(12).round(4))

st.markdown("---")
st.caption("⚠️ Hinweis: Indikative Signale — keine Anlageberatung.")
