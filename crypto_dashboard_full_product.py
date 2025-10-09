# crypto_forecast_pro.py
"""
Einseitige Prognose-App (Technik + Makro + Ereignis-Analyse + ML)
- Vorhersagen: Tag (heute, morgen), Woche (aktuell, nächste), Monat (aktuell, nächste)
- Assets: BTC-USD, ETH-USD, SOL-USD
- Makro: DXY (oder EURUSD fallback), VIX, Fear&Greed
- Economic Calendar: Investing.com scraping, fallback FRED (wenn pandas_datareader & API-Key vorhanden)
- Modell: GradientBoostingRegressor (sklearn)
- Hinweis: Indikative Prognosen, keine Anlageberatung.
"""

import os
import time
from datetime import datetime, timedelta
import math
import traceback

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import ta
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Optional: pandas_datareader for FRED (fallback)
try:
    import pandas_datareader.data as pdr
    PDR_AVAILABLE = True
except Exception:
    PDR_AVAILABLE = False

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="Krypto Prognose (ML + Makro + Events)")
st.title("🔮 Krypto-Prognosen (Technisch + Makro + Ereignis-Analyse)")

# ---------------- Settings ----------------
ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]
NOW = datetime.utcnow()
YEARS_EVENT_HISTORY = 5
MIN_ROWS_TRAIN = 200   # minimum rows for ML training (daily)
HORIZONS = {
    "day": 1,
    "week": 7,
    "month": 30
}
# model parameters
GB_PARAMS = {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05, "random_state": 42}

# cache TTLs
TTL_SHORT = 300
TTL_MED = 900
TTL_LONG = 3600

# ---------------- Helpers / Robust utils ----------------
def log_exc(e):
    st.error("Fehler (siehe Konsole).")
    print("EXC:", e)
    traceback.print_exc()

def safe_series(df, col):
    """Return 1D series for col robustly."""
    if df is None:
        return pd.Series(dtype=float)
    if col in df.columns:
        s = df[col]
    else:
        # try partial match
        for c in df.columns:
            if col.lower() in str(c).lower():
                s = df[c]; break
        else:
            return pd.Series(dtype=float, index=df.index)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    arr = np.asarray(s)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return pd.Series(arr, index=df.index).astype(float)

def resample_to_daily(df):
    """Ensure df is daily; if intraday provided, resample to daily last."""
    if df is None:
        return None
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.freq is None:
            # make sure daily
            try:
                return df.resample("D").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna(how="all")
            except Exception:
                return df
        else:
            return df
    return df

# ---------------- Data fetching ----------------
@st.cache_data(ttl=TTL_LONG)
def fetch_price(symbol, period_days=365*3, interval="1d"):
    """Fetch price series via yfinance; return DataFrame with OHLCV indexed by date."""
    try:
        per = f"{period_days}d"
        df = yf.download(symbol, period=per, interval=interval, progress=False)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print("fetch_price error:", e)
        return None

@st.cache_data(ttl=TTL_LONG)
def fetch_macro_timeseries():
    """
    Fetch macro proxies:
    - DXY (try multiple tickers), fallback 100/EURUSD
    - VIX (^VIX)
    - BTC price
    Return daily DataFrame with columns DXY, VIX, BTC_close
    """
    try:
        # BTC daily close (for correlation)
        btc = fetch_price("BTC-USD", period_days=365*5, interval="1d")
        btc_close = btc["Close"] if btc is not None else pd.Series(dtype=float)

        # VIX
        try:
            vix = fetch_price("^VIX", period_days=365*5, interval="1d")["Close"]
        except Exception:
            vix = pd.Series(dtype=float)

        # try DXY tickers
        dxy = pd.Series(dtype=float)
        for t in ("DX-Y.NYB", "DXY", "USDX"):
            try:
                tmp = fetch_price(t, period_days=365*5, interval="1d")
                if tmp is not None and not tmp.empty:
                    dxy = tmp["Close"]; break
            except Exception:
                continue
        if dxy is None or dxy.empty:
            # fallback to EURUSD inverse
            try:
                eur = fetch_price("EURUSD=X", period_days=365*5, interval="1d")["Close"]
                if eur is not None and not eur.empty:
                    dxy = 100 / eur
            except Exception:
                dxy = pd.Series(dtype=float)

        # build DF
        df = pd.concat({"BTC": btc_close, "DXY": dxy, "VIX": vix}, axis=1)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna()
        return df
    except Exception as e:
        print("fetch_macro_timeseries error:", e)
        return pd.DataFrame()

# ---------------- Fear & Greed ----------------
@st.cache_data(ttl=TTL_MED)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        js = r.json()
        val = int(js["data"][0]["value"])
        label = js["data"][0]["value_classification"]
        return val, label
    except Exception:
        return None, None

# ---------------- Economic calendar: Investing.com scraping ----------------
@st.cache_data(ttl=TTL_MED)
def fetch_investing_calendar(days_ahead=30):
    """
    Scrape Investing.com economic calendar for the next `days_ahead` days.
    Returns list of events with: date (YYYY-MM-DD), country, event, impact (low/med/high)
    """
    out = []
    try:
        base = "https://www.investing.com/economic-calendar/"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(base, headers=headers, timeout=12)
        if r.status_code != 200:
            return out
        soup = BeautifulSoup(r.text, "html.parser")
        # Investing's calendar is JS heavy; they embed rows with ids eventRowId{N}
        rows = soup.find_all("tr", id=lambda x: x and x.startswith("eventRowId"))
        for r0 in rows:
            try:
                date_attr = r0.get("data-event-datetime") or r0.get("data-event-datetime2") or ""
                if date_attr:
                    dt = pd.to_datetime(date_attr).date()
                else:
                    dt = None
                event = r0.get("event_attr_name") or r0.get("data-event-name") or r0.get("aria-label") or ""
                country = r0.get("data-event-country") or ""
                impact_tag = r0.find("td", {"class": "sentiment"})
                impact = ""
                if impact_tag:
                    # count filled stars/dots
                    dots = impact_tag.find_all("i", {"class": lambda x: x and "low" in x or "medium" in x or "high" in x})
                    # fallback: look for title
                    impact = impact_tag.get("title","")
                # normalize impact if possible
                if impact:
                    im = impact.lower()
                else:
                    im = ""
                out.append({"date": str(dt) if dt else "", "country": country.strip(), "event": event.strip(), "impact": im})
            except Exception:
                continue
        # dedupe
        ded = []
        seen = set()
        for e in out:
            key = (e.get("date",""), e.get("event",""))[:2]
            if key in seen: continue
            seen.add(key); ded.append(e)
        return ded
    except Exception as e:
        print("fetch_investing_calendar error:", e)
        return []

# ---------------- FRED fallback (optional) ----------------
def fetch_fred_series(series_id, start, end):
    """
    Attempt to fetch a FRED series using pandas_datareader (requires user to set FRED_API_KEY env var if needed).
    Returns Series or empty Series.
    """
    if not PDR_AVAILABLE:
        return pd.Series(dtype=float)
    try:
        api_key = os.environ.get("FRED_API_KEY", None)
        if api_key:
            os.environ["FRED_API_KEY"] = api_key
        s = pdr.DataReader(series_id, "fred", start, end)
        return s[series_id] if series_id in s else s.squeeze()
    except Exception as e:
        print("fetch_fred_series error:", e)
        return pd.Series(dtype=float)

# ---------------- Historical event analysis ----------------
@st.cache_data(ttl=TTL_LONG)
def analyze_past_event_reactions(asset_symbol, event_name_keyword, window_days=7, years=YEARS_EVENT_HISTORY):
    """
    For a given asset and an event keyword (e.g. 'CPI', 'Fed', 'Interest Rate'), find event dates in Investing calendar
    for the past 'years' years and compute average post-event returns (1d, 3d, 7d).
    Returns dict with mean returns and sample size.
    """
    try:
        # get historical calendar pages? Investing's site is JS; so easier: we will analyze using TradingEconomics if available via their calendar API (guest)
        # fallback: use investing.com search for event text with year - but scraping historical investing pages is brittle.
        # Practical approach: check recent N years by scanning dates on Investing current+past via their 'calendar' with query param year? Not reliable.
        # Simpler robust approach: use known dates for major events via TradingEconomics API (guest), then match event names containing keyword.
        hits = []
        try:
            base = "https://api.tradingeconomics.com/calendar"
            start = (datetime.utcnow().replace(year=datetime.utcnow().year - years)).strftime("%Y-%m-%d")
            end = datetime.utcnow().strftime("%Y-%m-%d")
            params = {"c":"guest:guest","from":start,"to":end}
            r = requests.get(base, params=params, timeout=12)
            if r.status_code == 200:
                arr = r.json()
                for ev in arr:
                    if event_name_keyword.lower() in str(ev.get("event","")).lower():
                        hits.append(ev)
        except Exception:
            pass

        # if no hits, return empty stats
        if not hits:
            return {"n": 0, "mean_1d": None, "mean_3d": None, "mean_7d": None}

        # For each hit, compute asset returns after event using yfinance
        prices = fetch_price(asset_symbol, period_days=365*years+60, interval="1d")
        if prices is None or prices.empty:
            return {"n": 0, "mean_1d": None, "mean_3d": None, "mean_7d": None}
        close = prices["Close"]
        rets_1d, rets_3d, rets_7d = [], [], []
        for ev in hits:
            date_str = ev.get("date","")[:10]
            try:
                ev_date = pd.to_datetime(date_str).date()
                # find nearest available trading date >= ev_date
                cand = close.index.normalize()
                # get index location
                idx = None
                try:
                    idx = cand.get_loc(pd.Timestamp(ev_date))
                except Exception:
                    # find next valid date
                    try:
                        idx = np.searchsorted(cand, pd.Timestamp(ev_date))
                        if idx >= len(cand): continue
                    except Exception:
                        continue
                i = int(idx)
                p0 = close.iloc[i]
                # compute returns
                if i+1 < len(close):
                    rets_1d.append((close.iloc[i+1] / p0) - 1)
                if i+3 < len(close):
                    rets_3d.append((close.iloc[i+3] / p0) - 1)
                if i+7 < len(close):
                    rets_7d.append((close.iloc[i+7] / p0) - 1)
            except Exception:
                continue
        out = {
            "n": len(hits),
            "mean_1d": np.mean(rets_1d) if rets_1d else None,
            "mean_3d": np.mean(rets_3d) if rets_3d else None,
            "mean_7d": np.mean(rets_7d) if rets_7d else None
        }
        return out
    except Exception as e:
        print("analyze_past_event_reactions error:", e)
        return {"n": 0, "mean_1d": None, "mean_3d": None, "mean_7d": None}

# ---------------- Feature engineering (daily) ----------------
@st.cache_data(ttl=TTL_MED)
def build_features(asset_symbol):
    """
    Build daily features DataFrame for ML:
    - price returns 1/7/30
    - SMA/EMA differences and slopes
    - RSI, MACD_DIFF, ATR
    - macro: DXY, VIX, Fear&Greed
    - event-based features: for upcoming calendar events, include impact and historical mean reaction
    """
    try:
        # prices
        prices = fetch_price(asset_symbol, period_days=365*6, interval="1d")
        if prices is None or prices.empty:
            return pd.DataFrame()
        df = pd.DataFrame(index=prices.index)
        df["Close"] = prices["Close"]
        df["Ret1"] = df["Close"].pct_change(1)
        df["Ret7"] = df["Close"].pct_change(7)
        df["Ret30"] = df["Close"].pct_change(30)
        df["Vol14"] = df["Ret1"].rolling(14).std()

        # compute indicators (we reuse earlier compute)
        tmp = prices.copy()
        tmp = tmp.assign(Open=tmp["Open"], High=tmp["High"], Low=tmp["Low"], Close=tmp["Close"], Volume=tmp.get("Volume",0))
        tmp = tmp.fillna(method="ffill").fillna(method="bfill")
        tmp = tmp.apply(pd.to_numeric, errors="coerce")
        tmp = tmp.dropna(how="all")
        tmp = tmp.copy()
        tmp = tmp.reset_index().set_index("Date") if "Date" in tmp.columns else tmp
        tmp = compute_technical_indicators_for_features(tmp)

        # merge selected technical features
        for col in ["EMA20","EMA50","EMA200","SMA20","SMA50","SMA200","RSI","MACD","MACD_SIGNAL","ATR"]:
            if col in tmp.columns:
                df[col] = tmp[col].reindex(df.index)

        # macro features
        macro = fetch_macro_timeseries()
        if macro is not None and not macro.empty:
            macro_r = macro.reindex(df.index).ffill().bfill()
            df["DXY"] = macro_r["DXY"]
            df["VIX"] = macro_r["VIX"]
            df["BTC_macro"] = macro_r["BTC"]
        else:
            df["DXY"] = np.nan; df["VIX"] = np.nan; df["BTC_macro"] = np.nan

        # fear & greed
        fgi_val, _ = fetch_fear_greed()
        df["FearGreed"] = fgi_val if fgi_val is not None else np.nan

        # event-related features: upcoming events in investing calendar (we mark days with events)
        events = fetch_investing_calendar(days_ahead=60)
        # build an event matrix: for each event date, mark its impact (0/1/2 high)
        event_map = {}
        for ev in events:
            try:
                d = ev.get("date","")
                if not d: continue
                d0 = pd.to_datetime(d).date()
                impact_text = ev.get("impact","").lower()
                impact_score = 2 if "high" in impact_text else (1 if "medium" in impact_text or "med" in impact_text else 0)
                event_map.setdefault(d0,[]).append({"event": ev.get("event",""), "impact": impact_score})
            except Exception:
                continue
        # For each row date, attach: max_event_impact_today and count_events_today
        df["event_impact_today"] = 0
        df["event_count_today"] = 0
        df["event_historical_mean_1d"] = 0.0  # avg reaction from past similar events
        for dt in df.index:
            ddate = dt.date()
            arr = event_map.get(ddate, [])
            if arr:
                imp = max([a["impact"] for a in arr])
                df.at[dt, "event_impact_today"] = imp
                df.at[dt, "event_count_today"] = len(arr)
                # compute historical reaction for top keyword (first event name)
                try:
                    kw = arr[0]["event"].split()[0] if arr[0]["event"] else ""
                    stats = analyze_past_event_reactions(asset_symbol, kw, window_days=7, years=YEARS_EVENT_HISTORY)
                    if stats and stats.get("mean_1d") is not None:
                        df.at[dt, "event_historical_mean_1d"] = stats["mean_1d"]
                except Exception:
                    pass

        # final cleaning
        df = df.dropna(how="all")
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df
    except Exception as e:
        print("build_features error:", e)
        traceback.print_exc()
        return pd.DataFrame()

# need a local robust technical indicators builder used above
def compute_technical_indicators_for_features(df):
    """Compute RSI/MACD/EMA/SMA/ATR on df expecting columns Open/High/Low/Close."""
    try:
        df = df.copy()
        close = df["Close"]
        high = df["High"] if "High" in df.columns else close
        low = df["Low"] if "Low" in df.columns else close
        for p in (20,50,200):
            df[f"SMA{p}"] = close.rolling(window=p, min_periods=1).mean()
            df[f"EMA{p}"] = close.ewm(span=p, adjust=False).mean()
        try:
            df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        except Exception:
            df["RSI"] = np.nan
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
    except Exception as e:
        print("compute_technical_indicators_for_features error:", e)
        return df

# ---------------- ML training & prediction per horizon ----------------
def train_and_predict_for_horizon(feature_df, horizon_days):
    """
    Train GradientBoostingRegressor to predict horizon_days return.
    Returns predicted_return (float) and metrics dict (r2, mse) or (None,None) if cannot train.
    """
    try:
        df = feature_df.copy()
        # target: forward return after horizon_days
        df["future_price"] = df["Close"].shift(-horizon_days)
        df["target_ret"] = df["future_price"] / df["Close"] - 1
        df = df.dropna()
        if df.shape[0] < MIN_ROWS_TRAIN:
            return None, {"status": "not_enough_rows", "n_rows": df.shape[0]}
        # features list
        features = ["Ret1","Ret7","Ret30","Vol14",
                    "EMA20","EMA50","EMA200","SMA20","SMA50","SMA200",
                    "RSI","MACD_DIFF","ATR",
                    "DXY","VIX","FearGreed",
                    "event_impact_today","event_count_today","event_historical_mean_1d"]
        features = [f for f in features if f in df.columns]
        if not features:
            return None, {"status":"no_features"}
        X = df[features].values
        y = df["target_ret"].values
        # train/test split: last 20% as test
        split = max(int(len(X)*0.7), len(X)-int(0.2*len(X)))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        if len(X_train) < 50 or len(X_test) < 10:
            return None, {"status":"insufficient_train_test", "n_train":len(X_train),"n_test":len(X_test)}
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = GradientBoostingRegressor(**GB_PARAMS)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        # predict latest
        last_feat = scaler.transform(df[features].tail(1).values)
        pred = float(model.predict(last_feat)[0])
        return pred, {"r2": float(r2), "mse": float(mse), "n_train": len(X_train), "n_test": len(X_test)}
    except Exception as e:
        print("train_and_predict_for_horizon error:", e)
        traceback.print_exc()
        return None, {"status":"error","error":str(e)}

# ---------------- Wrapper to predict day/week/month now & next ----------------
def predict_all_horizons(asset_symbol):
    """
    Returns dict with:
      day_now, day_next, week_now, week_next, month_now, month_next
    Each is (pred_return, metrics dict)
    """
    out = {}
    feats = build_features(asset_symbol)
    if feats is None or feats.empty:
        # fill n/a results
        for k in ["day_now","day_next","week_now","week_next","month_now","month_next"]:
            out[k] = (None, {"status":"no_features"})
        return out
    # day now (1), day next (2)
    pred1, m1 = train_and_predict_for_horizon(feats, 1)
    pred2, m2 = train_and_predict_for_horizon(feats, 2)
    pred7, m7 = train_and_predict_for_horizon(feats, 7)
    pred14, m14 = train_and_predict_for_horizon(feats, 14)
    pred30, m30 = train_and_predict_for_horizon(feats, 30)
    pred60, m60 = train_and_predict_for_horizon(feats, 60)
    out["day_now"], out["day_now_metrics"] = (pred1, m1)
    out["day_next"], out["day_next_metrics"] = (pred2, m2)
    out["week_now"], out["week_now_metrics"] = (pred7, m7)
    out["week_next"], out["week_next_metrics"] = (pred14, m14)
    out["month_now"], out["month_now_metrics"] = (pred30, m30)
    out["month_next"], out["month_next_metrics"] = (pred60, m60)
    return out

# ---------------- UI Rendering (single page) ----------------
def main_page():
    st.header("Kompakte Prognoseseite — Tag / Woche / Monat (Heute + Next)")
    st.markdown("Die Vorhersagen kombinieren technische Indikatoren, makroökonomische Faktoren und historische Reaktionen auf ähnliche Ereignisse. Modelle = Gradient Boosting Regressors (indikativ).")

    # show macro snapshot
    st.subheader("Makro-Snapshot")
    macro = fetch_macro_timeseries()
    if macro is None or macro.empty:
        st.info("Makro-Daten nicht verfügbar.")
    else:
        last = macro.tail(1)
        try:
            st.write(f"DXY: {float(last['DXY']):.4f} | VIX: {float(last['VIX']):.2f} | BTC: {float(last['BTC']):.2f}")
        except Exception:
            st.write(last.tail(1).to_dict())

    # upcoming economic events
    st.subheader("Anstehende Wirtschaftsereignisse (Investing.com)")
    events = fetch_investing_calendar(days_ahead=60)
    if not events:
        st.info("Keine Events gefunden (Investing.com-Parsing evtl. geblockt).")
    else:
        # show first 8 events
        df_ev = pd.DataFrame(events[:12])
        st.dataframe(df_ev)

    # predictions per asset
    st.subheader("Preisprognosen (heute / morgen, diese Woche / nächste Woche, dieser Monat / nächster Monat)")
    results = {}
    for asset in ASSETS:
        st.markdown(f"### {asset}")
        with st.spinner(f"Berechne Features & ML-Prognosen für {asset} (kann ~10-30s dauern)..."):
            preds = predict_all_horizons(asset)
        # display nicely
        def fmt_cell(item):
            val, meta = item
            if val is None:
                info = meta.get("status") if isinstance(meta, dict) else ""
                return f"n/a ({info})"
            try:
                r2 = meta.get("r2") if isinstance(meta, dict) else None
                if r2 is None:
                    return f"{val*100:.2f}%"
                else:
                    return f"{val*100:.2f}% (R²={r2:.2f})"
            except Exception:
                return f"{val*100:.2f}%"
        row = {
            "Period": ["Day (now)", "Day (next)", "Week (now)", "Week (next)", "Month (now)", "Month (next)"],
            "Prediction": [
                fmt_cell((preds.get("day_now"), preds.get("day_now_metrics"))),
                fmt_cell((preds.get("day_next"), preds.get("day_next_metrics"))),
                fmt_cell((preds.get("week_now"), preds.get("week_now_metrics"))),
                fmt_cell((preds.get("week_next"), preds.get("week_next_metrics"))),
                fmt_cell((preds.get("month_now"), preds.get("month_now_metrics"))),
                fmt_cell((preds.get("month_next"), preds.get("month_next_metrics"))),
            ]
        }
        st.table(pd.DataFrame(row))
        # also show model diagnostics for month_now if available
        mm = preds.get("month_now_metrics", {})
        if isinstance(mm, dict) and mm.get("r2") is not None:
            st.write(f"Model: Month-now — R²={mm.get('r2'):.3f}, MSE={mm.get('mse'):.6f}, rows used={mm.get('n_train')+mm.get('n_test',0)}")
        elif isinstance(mm, dict) and mm.get("status"):
            st.write(f"Model Status: {mm.get('status')}")
        results[asset] = preds

    st.markdown("---")
    st.caption("Hinweis: Modelle sind datengetrieben und indikativ. Prüfe Datenqualität und nutze eigene Risikokontrolle.")

# ---------------- Run app ----------------
if __name__ == "__main__":
    main_page()
