import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import traceback

st.set_page_config(layout="wide", page_title="Krypto-Dashboard (stabil)")

WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]
CACHE_SHORT, CACHE_LONG = 600, 3600

# ---------- Hilfsfunktionen ----------
def to_series(x):
    """Sichert, dass Eingabe 1D-NumPy/Pandas-Serie ist"""
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if "Close" in x.columns:
            return x["Close"]
        for c in x.columns:
            if pd.api.types.is_numeric_dtype(x[c]):
                return x[c]
        return pd.Series(dtype=float)
    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = arr.ravel()
    return pd.Series(arr, dtype=float)

@st.cache_data(ttl=CACHE_SHORT)
def fetch_ohlc(symbol, months=12, interval="1d"):
    try:
        df = yf.download(symbol, period=f"{months}mo", interval=interval, progress=False)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

# ---------- Technische Indikatoren ----------
def compute_indicators(df):
    if df is None or df.empty:
        return df
    close, high, low = to_series(df["Close"]), to_series(df["High"]), to_series(df["Low"])

    # SMA / EMA
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["EMA20"] = close.ewm(span=20).mean()
    df["EMA50"] = close.ewm(span=50).mean()
    df["EMA200"] = close.ewm(span=200).mean()

    # RSI
    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, 14).rsi()
    except Exception:
        df["RSI"] = close.diff().rolling(14).mean()

    # MACD
    try:
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
        df["MACD_DIFF"] = df["MACD"] - df["MACD_SIGNAL"]
    except Exception:
        df["MACD"], df["MACD_SIGNAL"], df["MACD_DIFF"] = np.nan, np.nan, np.nan

    # ATR (robust)
    try:
        atr = ta.volatility.AverageTrueRange(high, low, close, 14)
        df["ATR"] = atr.average_true_range()
    except Exception:
        # Fallback: einfache Volatilität
        df["ATR"] = (high - low).rolling(14).mean()

    return df

# ---------- Prognose ----------
def fetch_macro_daily():
    try:
        btc = yf.download("BTC-USD", period="2y", interval="1d", progress=False)["Close"]
        dxy = yf.download("DX-Y.NYB", period="2y", interval="1d", progress=False)["Close"]
        vix = yf.download("^VIX", period="2y", interval="1d", progress=False)["Close"]
        return pd.concat({"BTC": btc, "DXY": dxy, "VIX": vix}, axis=1).dropna()
    except Exception:
        return pd.DataFrame()

def prepare_features(symbol):
    try:
        price = yf.download(symbol, period="1y", interval="1d", progress=False)["Close"]
        if not isinstance(price, pd.Series) or price.empty:
            return pd.DataFrame()
        df = pd.DataFrame({"Close": price})
        df["RET1"] = df["Close"].pct_change(1)
        df["RET7"] = df["Close"].pct_change(7)
        df["VOL14"] = df["RET1"].rolling(14).std()
        macro = fetch_macro_daily()
        if not macro.empty:
            df = df.join(macro[["DXY", "VIX"]], how="left").fillna(method="ffill")
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def train_and_predict(symbol, horizon=30):
    df = prepare_features(symbol)
    if df.empty or len(df) < 60:
        return None, None
    df["FUT"] = df["Close"].shift(-horizon)
    df["RET_FUT"] = df["FUT"] / df["Close"] - 1
    df = df.dropna()
    if len(df) < 30:
        return None, None
    X = df[["RET1", "RET7", "VOL14", "DXY", "VIX"]].values
    y = df["RET_FUT"].values
    try:
        scaler = StandardScaler().fit(X)
        model = LinearRegression().fit(scaler.transform(X), y)
        pred = model.predict(scaler.transform([X[-1]]))[0]
        r2 = r2_score(y, model.predict(scaler.transform(X)))
        return pred, r2
    except Exception:
        return None, None

# ---------- UI ----------
def plot_chart(df, sym):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Preis"))
    for p in (20, 50, 200):
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA{p}"], name=f"EMA{p}", line=dict(width=2)))
    fig.update_layout(title=f"{sym}", height=400)
    st.plotly_chart(fig, use_container_width=True)
    if "RSI" in df.columns:
        st.line_chart(df["RSI"], height=150)

def main_app():
    st.title("📊 Krypto-Dashboard (BTC, ETH, SOL)")
    tabs = st.tabs(["Übersicht", "Indikatoren", "Makroanalyse"])

    with tabs[0]:
        st.header("Prognosen")
        for sym in WATCHLIST:
            pred, r2 = train_and_predict(sym, 30)
            if pred is None:
                st.info(f"{sym}: Prognose nicht möglich (zu wenig Daten).")
                continue
            trend = "Bullish" if pred > 0.02 else ("Bearish" if pred < -0.02 else "Neutral")
            emoji = "🟢" if trend == "Bullish" else ("🔴" if trend == "Bearish" else "⚪")
            st.write(f"{emoji} {sym}: {trend} ({pred*100:.2f}% | R²={r2:.2f})")

    with tabs[1]:
        st.header("Technische Indikatoren")
        for sym in WATCHLIST:
            df = fetch_ohlc(sym)
            if df is None:
                st.warning(f"Keine Daten für {sym}.")
                continue
            df = compute_indicators(df)
            plot_chart(df, sym)

    with tabs[2]:
        st.header("Makrodaten")
        macro = fetch_macro_daily()
        if macro.empty:
            st.info("Keine Makrodaten verfügbar.")
        else:
            st.dataframe(macro.tail(20).round(3))

if __name__ == "__main__":
    try:
        main_app()
    except Exception as e:
        st.error(f"Unerwarteter Fehler: {e}")
        st.text(traceback.format_exc())
