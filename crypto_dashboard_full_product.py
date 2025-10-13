# crypto_forecast_pro_multisource.py
"""
Krypto-Prognose-App (einseitig)
- Multi-News (Investing.com, CoinDesk, CoinTelegraph, Bitcoin.com, Google News)
- Makro + Event-Analyse
- Technische Indikatoren: SMA/EMA(20/50/200), RSI(14), MACD, ATR
- ML: GradientBoostingRegressor (Fallback: LinearRegression -> heuristisch)
- Vorhersagen: Tag/Woche/Monat (aktuell & nächste)
- Keine Indikator-Übersicht unten
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import ta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import warnings
import math
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide", page_title="Krypto Prognosen (Multi-Source, ML)")
st.title("🔮 Krypto-Prognosen – Tag / Woche / Monat (ML + Makro + News)")
ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]

# Anforderungen (gelockert)
MIN_ROWS_GB = 100
MIN_ROWS_LR = 40
MIN_ROWS_HEUR = 5

GB_PARAMS = {"n_estimators":120, "max_depth":3, "learning_rate":0.05, "random_state":42}

# ---------------- NEWS SOURCES ----------------
@st.cache_data(ttl=900)
def fetch_news_multi(limit=20):
    feeds = [
        ("CoinTelegraph","https://cointelegraph.com/rss"),
        ("CoinDesk","https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Bitcoin.com","https://news.bitcoin.com/feed/"),
        ("Investing","https://www.investing.com/rss/news_25.rss"),
        ("GoogleNews","https://news.google.com/rss/search?q=bitcoin+OR+crypto+OR+ethereum&hl=en-US&gl=US&ceid=US:en")
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    news = []
    for src, url in feeds:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.content, "xml")
            for item in soup.find_all("item")[: max(1, limit // len(feeds))]:
                news.append({
                    "source": src,
                    "title": item.title.text if item.title else "",
                    "link": item.link.text if item.link else "",
                    "date": item.pubDate.text if item.pubDate else ""
                })
        except Exception:
            continue
    seen = set(); dedup = []
    for n in news:
        t = n.get("title","")[:200]
        if not t or t in seen: continue
        seen.add(t); dedup.append(n)
    return dedup[:limit] if dedup else [{"source":"System","title":"Keine News gefunden","link":"","date":""}]

# ---------------- MACRO ----------------
@st.cache_data(ttl=3600)
def fetch_macro_timeseries(days=365*5):
    btc = yf.download("BTC-USD", period=f"{days}d", interval="1d", progress=False)
    btc_c = btc["Close"] if not btc.empty else pd.Series(dtype=float)

    try:
        vix = yf.download("^VIX", period=f"{days}d", interval="1d", progress=False)["Close"]
    except Exception:
        vix = pd.Series(dtype=float)

    dxy = pd.Series(dtype=float)
    for t in ("DX-Y.NYB","DXY","USDX"):
        try:
            d = yf.download(t, period=f"{days}d", interval="1d", progress=False)
            if not d.empty:
                dxy = d["Close"]; break
        except Exception:
            continue
    if dxy.empty:
        try:
            eur = yf.download("EURUSD=X", period=f"{days}d", interval="1d", progress=False)
            if not eur.empty:
                dxy = 100 / eur["Close"]
        except Exception:
            pass

    df = pd.concat({"BTC":btc_c, "DXY":dxy, "VIX":vix}, axis=1).fillna(0)
    df = df.dropna(how="all")
    return df

@st.cache_data(ttl=1800)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        if r.status_code == 200:
            j = r.json()
            return int(j["data"][0]["value"]), j["data"][0]["value_classification"]
    except Exception:
        pass
    return None, None

# ---------------- TECHNICAL INDICATORS ----------------
def compute_technical(df):
    if df is None or df.empty: return df
    close = df["Close"]
    for p in (20,50,200):
        df[f"SMA{p}"] = close.rolling(p, min_periods=1).mean()
        df[f"EMA{p}"] = close.ewm(span=p, adjust=False).mean()
    # RSI may fail if not enough data -> wrap
    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, 14).rsi()
    except Exception:
        df["RSI"] = np.nan
    try:
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
        df["MACD_DIFF"] = df["MACD"] - df["MACD_SIGNAL"]
    except Exception:
        df["MACD"] = df["MACD_SIGNAL"] = df["MACD_DIFF"] = np.nan
    return df.fillna(0)

# ---------------- FEATURE BUILD ----------------
@st.cache_data(ttl=1800)
def build_features(asset):
    try:
        df = yf.download(asset, period="max", interval="1d", progress=False)
        if df is None or df.empty: return pd.DataFrame()
        df = compute_technical(df)
        df["Ret1"] = df["Close"].pct_change(1)
        df["Vol14"] = df["Ret1"].rolling(14).std().fillna(0)
        macro = fetch_macro_timeseries()
        if not macro.empty:
            macro = macro.reindex(df.index).fillna(method="ffill").fillna(method="bfill").fillna(0)
            df["DXY"] = macro["DXY"]
            df["VIX"] = macro["VIX"]
        else:
            df["DXY"] = 0
            df["VIX"] = 0
        fgi, lbl = fetch_fear_greed()
        df["FearGreed"] = fgi if fgi else 50
        return df.fillna(0)
    except Exception:
        return pd.DataFrame()

# ---------------- ML MODEL ----------------
def train_predict(df, horizon):
    df = df.copy()
    df["target"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df = df.dropna()
    n = len(df)
    if n < MIN_ROWS_HEUR:
        return None, {"status":"not_enough_rows", "n":n}
    feats = [c for c in ["RSI","MACD_DIFF","EMA20","EMA50","EMA200","SMA20","SMA50","SMA200","Vol14","DXY","VIX","FearGreed"] if c in df.columns]
    if not feats:
        return None, {"status":"no_features"}
    X, y = df[feats].values, df["target"].values
    if n >= MIN_ROWS_GB:
        try:
            Xtr, ytr = X[:-60], y[:-60]
            Xte, yte = X[-60:], y[-60:]
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            model = GradientBoostingRegressor(**GB_PARAMS).fit(Xtr, ytr)
            pred = float(model.predict(sc.transform(df[feats].tail(1)))[0])
            r2 = float(r2_score(yte, model.predict(Xte)))
            return pred, {"model":"GB","r2":r2,"n":n}
        except Exception:
            pass
    if n >= MIN_ROWS_LR:
        try:
            lr = LinearRegression().fit(X, y)
            pred = float(lr.predict(df[feats].tail(1).values)[0])
            return pred, {"model":"LR","n":n}
        except Exception:
            pass
    # heuristic fallback
    if "Ret1" in df.columns:
        recent = df["Ret1"].tail(60).mean()
        return recent, {"model":"heuristic","n":n}
    return None, {"status":"no_fallback"}

# =========================================
# 🧠 MULTI-TIMEFRAME FEATURE ENGINEERING + ML PROGNOSE
# =========================================

def build_multi_tf_features(asset):
    try:
        df_4h = yf.download(asset, period="360d", interval="4h", progress=False)
        if df_4h is not None and not df_4h.empty:
            df_4h = compute_technical(df_4h)
            df_4h = df_4h.add_suffix("_4h")
        else:
            df_4h = pd.DataFrame()

        df_d = yf.download(asset, period="max", interval="1d", progress=False)
        if df_d is not None and not df_d.empty:
            df_d = compute_technical(df_d)
            df_d = df_d.add_suffix("_1d")
        else:
            df_d = pd.DataFrame()

        df_w = yf.download(asset, period="5y", interval="1wk", progress=False)
        if df_w is not None and not df_w.empty:
            df_w = compute_technical(df_w)
            df_w = df_w.add_suffix("_1w")
        else:
            df_w = pd.DataFrame()

        dfs = [df for df in [df_4h, df_d, df_w] if not df.empty]
        if not dfs:
            return pd.DataFrame()

        for i, df in enumerate(dfs):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
            dfs[i] = df

        df_all = pd.concat(dfs, axis=1)
        df_all = df_all[~df_all.index.duplicated(keep="last")]
        df_all = df_all.sort_index()

        close_col = [c for c in df_all.columns if "Close_1d" in c]
        if close_col:
            df_all["Close"] = df_all[close_col[0]]
        else:
            df_all["Close"] = df_all.iloc[:, 0]

        df_all["Ret1"] = df_all["Close"].pct_change(1)

        macro = fetch_macro_timeseries()
        if not macro.empty:
            macro = macro.reindex(df_all.index, method="ffill").fillna(0)
            df_all["VIX"] = macro.get("VIX", 0)
            df_all["DXY"] = macro.get("DXY", 0)
        else:
            df_all["VIX"] = 0
            df_all["DXY"] = 0

        return df_all.fillna(0)

    except Exception as e:
        print(f"⚠️ Fehler in build_multi_tf_features({asset}): {e}")
        return pd.DataFrame()


def train_predict_multi_tf(asset, horizon_days=7):
    try:
        df = build_multi_tf_features(asset)
        if df is None or df.empty:
            return None, {"status": "no_data"}

        df["target"] = df["Close"].shift(-horizon_days) / df["Close"] - 1
        df = df.dropna()
        n = len(df)
        if n < 60:
            return None, {"status": "not_enough_rows", "n": n}

        feat_cols = [c for c in df.columns if any(x in c for x in ["RSI", "MACD_DIFF", "EMA", "SMA", "Vol", "VIX", "DXY"])]

        if not feat_cols:
            return None, {"status": "no_features"}

        X = df[feat_cols].values
        y = df["target"].values

        if n > 120:
            sc = StandardScaler().fit(X)
            X_scaled = sc.transform(X)
            Xtr, Xte = X_scaled[:-60], X_scaled[-60:]
            ytr, yte = y[:-60], y[-60:]
            model = GradientBoostingRegressor(
                n_estimators=180, learning_rate=0.05, max_depth=3, random_state=42
            )
            model.fit(Xtr, ytr)
            pred = float(model.predict(sc.transform(df[feat_cols].tail(1)))[0])
            r2 = float(r2_score(yte, model.predict(Xte)))
            return pred, {"model": "GradientBoosting", "r2": r2, "n": n}

        else:
            lr = LinearRegression().fit(X, y)
            pred = float(lr.predict(df[feat_cols].tail(1).values)[0])
            return pred, {"model": "LinearRegression", "n": n}

    except Exception as e:
        print(f"⚠️ Fehler in train_predict_multi_tf({asset}): {e}")
        return None, {"status": "error", "msg": str(e)}

# ---------------- Utility: safe number extractor ----------------
def safe_num(val):
    """
    Extract a scalar float from possible Series/ndarray/list/scalar.
    Returns np.nan if cannot convert.
    """
    try:
        if isinstance(val, (pd.Series, np.ndarray, list)):
            arr = np.asarray(val)
            if arr.size == 0:
                return float("nan")
            return float(arr.flatten()[-1])
        if val is None:
            return float("nan")
        return float(val)
    except Exception:
        return float("nan")

# ---------------- MAIN PAGE ----------------
def main():
    st.header("💡 Kombinierte ML-Prognosen (Technik + Makro + Nachrichten)")

    # Makro Snapshot (robust)
    st.subheader("🌍 Makro-Umfeld")
    macro = fetch_macro_timeseries()
    if macro is None or macro.empty:
        st.info("Makro-Daten nicht verfügbar.")
    else:
        # safe extraction of the last row values
        last_row = macro.tail(1)
        try:
            # last_row may be a DataFrame with single row; use .iloc[0] to get Series if possible
            series_last = last_row.iloc[0] if isinstance(last_row, pd.DataFrame) and not last_row.empty else None
        except Exception:
            series_last = None
        if series_last is None:
            st.info("Makro-Daten nicht ausreichend.")
        else:
            dxy_val = safe_num(series_last.get("DXY", np.nan))
            vix_val = safe_num(series_last.get("VIX", np.nan))
            if math.isnan(dxy_val) and math.isnan(vix_val):
                st.info("Makro-Daten nicht verfügbar.")
            else:
                st.write(f"DXY: {dxy_val:.2f} | VIX: {vix_val:.2f}")

    # News
    st.subheader("📰 Relevante Krypto- und Makro-News")
    news = fetch_news_multi(12)
    for n in news[:10]:
        src, title, link = n["source"], n["title"], n["link"]
        if link:
            st.markdown(f"- **[{src}]** [{title}]({link})")
        else:
            st.markdown(f"- **[{src}]** {title}")

    # Prognosen
    st.subheader("📈 Preisprognosen (Tag / Woche / Monat)")
    horizons = {"Day (now)":1,"Day (next)":2,"Week (now)":7,"Week (next)":14,"Month (now)":30,"Month (next)":60}
    for a in ASSETS:
        st.markdown(f"## {a}")
        df = build_features(a)
        if df.empty:
            st.warning(f"Keine Daten für {a}.")
            continue
        rows=[]
        for label, h in horizons.items():
            p, info = train_predict(df, h)
            if p is None:
                rows.append({"Zeitraum":label,"Prognose":"n/a","Info":info.get("status","")})
            else:
                trend = "🟢 Bullish" if p>0.01 else ("🔴 Bearish" if p<-0.01 else "⚪ Neutral")
                rows.append({"Zeitraum":label,"Prognose":f"{p*100:.2f}%","Trend":trend,"Modell":info.get("model",""),"n":info.get("n","")})
        st.table(pd.DataFrame(rows))


    



# ---------------- SWING SIGNALS (robust) ----------------
def detect_signals(df):
    """Analysiert RSI, MACD, EMA, SMA und gibt bullishe/bearishe Textsignale zurück (robust gegen Spaltennamen-Fehler)."""
    signals = []
    if df is None or df.empty:
        return ["Keine Daten"]

    # Versuche, Close-Spalte zu finden (robust)
    close_col = None
    for c in df.columns:
        if "close" in str(c).lower():
            close_col = c
            break

    if close_col is None:
        return ["❌ Keine 'Close'-Spalte gefunden (keine Signalanalyse möglich)."]

    # sichere Kopie letzter Zeilen
    try:
        df = df.dropna(subset=[close_col]).tail(3).copy()
    except Exception:
        df = df.tail(3).copy()

    if df.shape[0] < 3:
        return ["⚪ Zu wenige Kursdaten für Signale"]

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # === RSI Signale ===
    if "RSI" in df.columns:
        try:
            if latest["RSI"] > 70:
                signals.append("🔻 RSI überkauft (>70) → mögliches bearishes Signal")
            elif latest["RSI"] < 30:
                signals.append("🟢 RSI überverkauft (<30) → mögliches bullishes Signal")
            elif prev["RSI"] < 50 and latest["RSI"] > 50:
                signals.append("🟢 RSI steigt über 50 → Momentum dreht bullisch")
            elif prev["RSI"] > 50 and latest["RSI"] < 50:
                signals.append("🔻 RSI fällt unter 50 → Momentum dreht bärisch")
        except Exception:
            pass

    # === MACD Signale ===
    if "MACD" in df.columns and "MACD_SIGNAL" in df.columns:
        try:
            if prev["MACD"] < prev["MACD_SIGNAL"] and latest["MACD"] > latest["MACD_SIGNAL"]:
                signals.append("🟢 MACD kreuzt über Signallinie → bullishes Signal")
            elif prev["MACD"] > prev["MACD_SIGNAL"] and latest["MACD"] < latest["MACD_SIGNAL"]:
                signals.append("🔻 MACD kreuzt unter Signallinie → bearishes Signal")
            elif "MACD_DIFF" in df.columns:
                if latest["MACD_DIFF"] > 0:
                    signals.append("🟢 MACD positiv → bullische Tendenz")
                elif latest["MACD_DIFF"] < 0:
                    signals.append("🔻 MACD negativ → bärische Tendenz")
        except Exception:
            pass

    # === EMA Signale (Crossovers) ===
    if all(c in df.columns for c in ["EMA20", "EMA50", "EMA200"]):
        try:
            if prev["EMA20"] < prev["EMA50"] and latest["EMA20"] > latest["EMA50"]:
                signals.append("🟢 EMA20 kreuzt über EMA50 → kurzfristig bullisch")
            elif prev["EMA20"] > prev["EMA50"] and latest["EMA20"] < latest["EMA50"]:
                signals.append("🔻 EMA20 kreuzt unter EMA50 → kurzfristig bärisch")

            if prev["EMA50"] < prev["EMA200"] and latest["EMA50"] > latest["EMA200"]:
                signals.append("🟢 EMA50 kreuzt über EMA200 → mittelfristig bullisch")
            elif prev["EMA50"] > prev["EMA200"] and latest["EMA50"] < latest["EMA200"]:
                signals.append("🔻 EMA50 kreuzt unter EMA200 → mittelfristig bärisch")
        except Exception:
            pass

    # === SMA Signale (Crossovers) ===
    if all(c in df.columns for c in ["SMA20", "SMA50", "SMA200"]):
        try:
            if prev["SMA20"] < prev["SMA50"] and latest["SMA20"] > latest["SMA50"]:
                signals.append("🟢 SMA20 kreuzt über SMA50 → kurzfristig bullisch")
            elif prev["SMA20"] > prev["SMA50"] and latest["SMA20"] < latest["SMA50"]:
                signals.append("🔻 SMA20 kreuzt unter SMA50 → kurzfristig bärisch")

            if prev["SMA50"] < prev["SMA200"] and latest["SMA50"] > latest["SMA200"]:
                signals.append("🟢 SMA50 kreuzt über SMA200 → mittelfristig bullisch")
            elif prev["SMA50"] > prev["SMA200"] and latest["SMA50"] < latest["SMA200"]:
                signals.append("🔻 SMA50 kreuzt unter SMA200 → mittelfristig bärisch")
        except Exception:
            pass

    if not signals:
        signals.append("⚪ Keine akuten Swing-Signale erkannt")
    return signals


def show_signals_section():
    """Zeigt nur einmal die technischen Swing-Signale an."""
    st.subheader("📊 Technische Swing-Signale")
    for asset in ASSETS:
        st.markdown(f"### {asset}")
        df = build_features(asset)
        if df.empty:
            st.warning(f"Keine Daten für {asset}.")
            continue
        sigs = detect_signals(df)
        for s in sigs:
            if "🟢" in s:
                st.success(s)
            elif "🔻" in s:
                st.error(s)
            else:
                st.info(s)
        st.markdown("---")
    st.caption("⚠️ Prognosen basieren auf historischen Mustern. Keine Anlageberatung.")


def main():
    st.header("💡 Kombinierte ML-Prognosen (Technik + Makro + Nachrichten)")

    # 🌍 Makro-Analyse
    st.subheader("🌍 Makro-Umfeld")
    macro = fetch_macro_timeseries()
    if macro is not None and not macro.empty:
        last = macro.iloc[-1]
        st.write(f"DXY: {safe_num(last.get('DXY')):.2f} | VIX: {safe_num(last.get('VIX')):.2f}")
    else:
        st.info("Makro-Daten nicht verfügbar.")

    # 📰 News
    st.subheader("📰 Relevante Krypto- und Makro-News")
    for n in fetch_news_multi(12):
        src, title, link = n["source"], n["title"], n["link"]
        st.markdown(f"- **[{src}]** [{title}]({link})" if link else f"- **[{src}]** {title}")

    # 📈 Preisprognosen
    st.subheader("📈 Preisprognosen (Tag / Woche / Monat)")
    horizons = {"Day (now)":1,"Day (next)":2,"Week (now)":7,"Week (next)":14,"Month (now)":30,"Month (next)":60}
    for a in ASSETS:
        st.markdown(f"## {a}")
        df = build_features(a)
        if df.empty:
            st.warning(f"Keine Daten für {a}.")
            continue
        rows = []
        for label, h in horizons.items():
            p, info = train_predict(df, h)
            if p is None:
                rows.append({"Zeitraum":label,"Prognose":"n/a","Info":info.get("status","")})
            else:
                trend = "🟢 Bullish" if p>0.01 else ("🔻 Bearish" if p<-0.01 else "⚪ Neutral")
                rows.append({"Zeitraum":label,"Prognose":f"{p*100:.2f}%","Trend":trend,"Modell":info.get("model",""),"n":info.get("n","")})
        st.table(pd.DataFrame(rows))

    # 🧠 Multi-Timeframe Prognose
    st.subheader("🧠 Multi-Timeframe-Prognose (4h + 1d + 1w kombiniert)")
    for asset in ASSETS:
        st.markdown(f"### {asset}")
        pred, info = train_predict_multi_tf(asset, horizon_days=7)
        if pred is None:
            st.warning(f"{asset}: Keine Prognose (Grund: {info.get('status','unbekannt')})")
        else:
            trend = "🟢 Bullish" if pred > 0.01 else ("🔻 Bearish" if pred < -0.01 else "⚪ Neutral")
            st.write(f"{trend} Erwartete 7-Tage-Rendite: **{pred*100:.2f}%** — Modell: {info.get('model','?')} — R²={info.get('r2','n/a')}")

    # 📊 Signale
    show_signals_section()


if __name__ == "__main__":
    main()


# ---------------- SWING SIGNALS (robust) ----------------
def detect_signals(df):
    """Analysiert RSI, MACD, EMA, SMA und gibt bullishe/bearishe Textsignale zurück (robust gegen Spaltennamen-Fehler)."""
    signals = []
    if df is None or df.empty:
        return ["Keine Daten"]

    # Versuche, Close-Spalte zu finden (robust)
    close_col = None
    for c in df.columns:
        if "close" in str(c).lower():
            close_col = c
            break

    if close_col is None:
        return ["❌ Keine 'Close'-Spalte gefunden (keine Signalanalyse möglich)."]

    # sichere Kopie letzter Zeilen
    try:
        df = df.dropna(subset=[close_col]).tail(3).copy()
    except Exception:
        df = df.tail(3).copy()

    if df.shape[0] < 3:
        return ["⚪ Zu wenige Kursdaten für Signale"]

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # === RSI Signale ===
    if "RSI" in df.columns:
        try:
            if latest["RSI"] > 70:
                signals.append("🔻 RSI überkauft (>70) → mögliches bearishes Signal")
            elif latest["RSI"] < 30:
                signals.append("🟢 RSI überverkauft (<30) → mögliches bullishes Signal")
            elif prev["RSI"] < 50 and latest["RSI"] > 50:
                signals.append("🟢 RSI steigt über 50 → Momentum dreht bullisch")
            elif prev["RSI"] > 50 and latest["RSI"] < 50:
                signals.append("🔻 RSI fällt unter 50 → Momentum dreht bärisch")
        except Exception:
            pass

    # === MACD Signale ===
    if "MACD" in df.columns and "MACD_SIGNAL" in df.columns:
        try:
            if prev["MACD"] < prev["MACD_SIGNAL"] and latest["MACD"] > latest["MACD_SIGNAL"]:
                signals.append("🟢 MACD kreuzt über Signallinie → bullishes Signal")
            elif prev["MACD"] > prev["MACD_SIGNAL"] and latest["MACD"] < latest["MACD_SIGNAL"]:
                signals.append("🔻 MACD kreuzt unter Signallinie → bearishes Signal")
            elif "MACD_DIFF" in df.columns:
                if latest["MACD_DIFF"] > 0:
                    signals.append("🟢 MACD positiv → bullische Tendenz")
                elif latest["MACD_DIFF"] < 0:
                    signals.append("🔻 MACD negativ → bärische Tendenz")
        except Exception:
            pass

    # === EMA Signale (Crossovers) ===
    if all(c in df.columns for c in ["EMA20", "EMA50", "EMA200"]):
        try:
            if prev["EMA20"] < prev["EMA50"] and latest["EMA20"] > latest["EMA50"]:
                signals.append("🟢 EMA20 kreuzt über EMA50 → kurzfristig bullisch")
            elif prev["EMA20"] > prev["EMA50"] and latest["EMA20"] < latest["EMA50"]:
                signals.append("🔻 EMA20 kreuzt unter EMA50 → kurzfristig bärisch")

            if prev["EMA50"] < prev["EMA200"] and latest["EMA50"] > latest["EMA200"]:
                signals.append("🟢 EMA50 kreuzt über EMA200 → mittelfristig bullisch")
            elif prev["EMA50"] > prev["EMA200"] and latest["EMA50"] < latest["EMA200"]:
                signals.append("🔻 EMA50 kreuzt unter EMA200 → mittelfristig bärisch")
        except Exception:
            pass

    # === SMA Signale (Crossovers) ===
    if all(c in df.columns for c in ["SMA20", "SMA50", "SMA200"]):
        try:
            if prev["SMA20"] < prev["SMA50"] and latest["SMA20"] > latest["SMA50"]:
                signals.append("🟢 SMA20 kreuzt über SMA50 → kurzfristig bullisch")
            elif prev["SMA20"] > prev["SMA50"] and latest["SMA20"] < latest["SMA50"]:
                signals.append("🔻 SMA20 kreuzt unter SMA50 → kurzfristig bärisch")

            if prev["SMA50"] < prev["SMA200"] and latest["SMA50"] > latest["SMA200"]:
                signals.append("🟢 SMA50 kreuzt über SMA200 → mittelfristig bullisch")
            elif prev["SMA50"] > prev["SMA200"] and latest["SMA50"] < latest["SMA200"]:
                signals.append("🔻 SMA50 kreuzt unter SMA200 → mittelfristig bärisch")
        except Exception:
            pass

    if not signals:
        signals.append("⚪ Keine akuten Swing-Signale erkannt")
    return signals


# ---------------- SIGNAL-AUSGABE ----------------
st.subheader("📊 Technische Swing-Signale")

for asset in ASSETS:
    st.markdown(f"### {asset}")
    df = build_features(asset)
    if df.empty:
        st.warning(f"Keine Daten für {asset}.")
        continue
    sigs = detect_signals(df)
    for s in sigs:
        if "🟢" in s:
            st.success(s)
        elif "🔻" in s:
            st.error(s)
        else:
            st.info(s)
    
    st.markdown("---")
    st.caption("⚠️ Prognosen basieren auf historischen Mustern. Keine Anlageberatung.")

if __name__ == "__main__":
    main()

