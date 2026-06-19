"""
Fintech-style stock analytics dashboard.
Updated version with:
- Live stock quote panel and auto-refresh
- Dark multi-color UI inspired by modern investing apps
- Extra backend APIs for quotes, chart history, watchlist, and recent searches
- Simple in-memory caching for faster repeated requests
"""
import json
import math
import os
import threading
import time
import warnings
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, Response, render_template_string, request

warnings.filterwarnings("ignore")

try:
    from curl_cffi.requests.exceptions import HTTPError as CurlHTTPError
except ImportError:
    CurlHTTPError = OSError

from advanced_distributions.davies_distribution import DaviesDistribution
from advanced_distributions.fractal_distribution import FractalDistribution
from advanced_distributions.fractional_distribution import FractionalDistribution
from advanced_distributions.quantile_distribution import QuantileDistribution
from advanced_distributions.sinh_arcsinh import SinhArcsinhDistribution
from advanced_distributions.slash_distribution import SlashDistribution

app = Flask(__name__)


def sf(x, n=6):
    try:
        v = float(x)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else round(v, n)
    except Exception:
        return 0.0


def _clean(obj):
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.floating):
        v = float(obj)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def safe_json(obj, status=200):
    return Response(
        json.dumps(_clean(obj), allow_nan=False),
        status=status,
        mimetype="application/json",
    )


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_number(value, default=0.0):
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def to_mapping(value):
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        return {}


CORRECTIONS = {
    "APPL": "AAPL",
    "AMZON": "AMZN",
    "AMAZN": "AMZN",
    "MICROSFT": "MSFT",
    "MICROSFOT": "MSFT",
    "NETFLX": "NFLX",
    "TESTA": "TSLA",
    "RELINCE": "RELIANCE.NS",
    "RELIACE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "SBIN": "SBIN.NS",
    "WIPRO": "WIPRO.NS",
    "INFY": "INFY.NS",
}

DEFAULT_WATCHLIST = [
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "RELIANCE.NS",
    "INFY.NS",
]


def resolve(ticker):
    u = (ticker or "").upper().strip()
    if u in CORRECTIONS:
        c = CORRECTIONS[u]
        return c, f"'{u}' auto-corrected to '{c}'"
    return u, None


CACHE_LOCK = threading.Lock()
CACHE = {"quote": {}, "analysis": {}, "history": {}}
RECENT_SEARCHES = deque(maxlen=8)


def cache_get(bucket, key):
    with CACHE_LOCK:
        entry = CACHE.get(bucket, {}).get(key)
        if not entry:
            return None
        if entry["expires_at"] <= time.time():
            CACHE[bucket].pop(key, None)
            return None
        return entry["value"]


def cache_set(bucket, key, value, ttl=30):
    with CACHE_LOCK:
        CACHE.setdefault(bucket, {})[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
        }
    return value


def remember_search(ticker):
    if not ticker:
        return
    with CACHE_LOCK:
        existing = [x for x in RECENT_SEARCHES if x["ticker"] != ticker]
        RECENT_SEARCHES.clear()
        RECENT_SEARCHES.appendleft({"ticker": ticker, "searched_at": utc_now()})
        for item in existing[: RECENT_SEARCHES.maxlen - 1]:
            RECENT_SEARCHES.append(item)


def fetch(ticker, period="3y"):
    df = pd.DataFrame()
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception:
        pass
    if df is None or df.empty:
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        except Exception:
            pass
    if df is None or df.empty:
        t = ticker.upper()
        hint = (
            f" Try '{t}.NS' for Indian stocks."
            if (not t.endswith(".NS") and t.isalpha() and len(t) <= 6)
            else ""
        )
        raise ValueError(f"No data found for '{ticker}'.{hint}")
    df.columns = (
        [str(c[0]).lower() for c in df.columns]
        if isinstance(df.columns, pd.MultiIndex)
        else [str(c).lower() for c in df.columns]
    )
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'")
    return df


def clean(df):
    df = df.copy().dropna()
    df = df[~df.index.duplicated(keep="first")]
    mask = (df.high >= df.low) & (df.high >= df.close) & (df.low <= df.close) & (df.volume > 0)
    df = df[mask]
    lr = np.log(df.close / df.close.shift(1)).dropna()
    z = (lr - lr.mean()) / lr.std()
    return df.loc[z[np.abs(z) <= 5].index].sort_index()


def features(df, w=20):
    df = df.copy()
    df["lr"] = np.log(df.close / df.close.shift(1))
    df["vol"] = df.lr.rolling(w).std() * np.sqrt(252)
    rm, rs = df.close.rolling(w).mean(), df.close.rolling(w).std()
    df["mom"] = (df.close - rm) / (rs + 1e-10)
    d = df.close.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / (l + 1e-10))
    df.dropna(inplace=True)
    return df


def run_fractional(r):
    pos = np.abs(r) + 1e-8
    d = FractionalDistribution.fit(pos)
    h = float(np.clip(0.5 + (d.alpha - 1.5) * 0.1, 0.3, 0.8))
    mem = float(np.clip((h - 0.5) * 2, 0, 1))
    reg = "trending" if h > 0.55 else "mean-reverting" if h < 0.45 else "random walk"
    bull_signal = float(np.clip((h - 0.5) * 4, 0, 1))
    bear_signal = float(np.clip((0.5 - h) * 4, 0, 1))
    return {
        "alpha": sf(d.alpha, 4),
        "beta": sf(d.beta, 4),
        "gamma": sf(d.gamma, 4),
        "mean": sf(d.mean()),
        "std": sf(d.std()),
        "skewness": sf(d.skewness(), 4),
        "kurtosis": sf(d.kurtosis(), 4),
        "hurst_proxy": sf(h, 4),
        "memory_score": sf(mem, 4),
        "regime": reg,
        "_bull": bull_signal,
        "_bear": bear_signal,
    }


def run_fractal(r):
    pos = np.abs(r) + 1e-8
    p = FractalDistribution.fit(pos)
    d_value, lam = sf(p[0], 4), sf(p[1], 4)
    d = FractalDistribution(D=d_value, lambda_=lam)
    comp = float(np.clip(d_value - 1.0, 0, 1))
    s = "chaotic" if d_value > 1.7 else "complex" if d_value > 1.4 else "structured"
    bull_signal = 1.0 - comp
    bear_signal = comp
    return {
        "D": d_value,
        "lambda": lam,
        "mean": sf(d.mean()),
        "std": sf(d.std()),
        "complexity_score": sf(comp, 4),
        "structure": s,
        "_bull": bull_signal,
        "_bear": bear_signal,
    }


def run_sinh(r):
    d = SinhArcsinhDistribution.fit(r)
    sk = "left-skewed" if d.epsilon < -0.1 else "right-skewed" if d.epsilon > 0.1 else "symmetric"
    skew_strength = float(np.tanh(d.epsilon * 2))
    bull_signal = float(np.clip(skew_strength, 0, 1))
    bear_signal = float(np.clip(-skew_strength, 0, 1))
    tail_penalty = float(np.clip((d.delta - 1.0) * 0.3, 0, 0.3))
    return {
        "epsilon": sf(d.epsilon, 4),
        "delta": sf(d.delta, 4),
        "mu": sf(d.mu),
        "sigma": sf(d.sigma),
        "skewness": sf(d.skewness(), 4),
        "kurtosis": sf(d.kurtosis(), 4),
        "skewness_direction": sk,
        "skew_score": sf(np.tanh(d.epsilon), 4),
        "_bull": max(0, bull_signal - tail_penalty),
        "_bear": max(0, bear_signal + tail_penalty * 0.5),
    }


def run_slash(r):
    d = SlashDistribution.fit(r)
    s = d.rvs(size=5000, random_state=42)
    ep = sf(np.mean(np.abs(s - d.mu) > 3.0 * d.sigma), 4)
    cr = "high" if ep > 0.05 else "medium" if ep > 0.02 else "low"
    ent = sf(d.entropy(), 4)
    crash_bull = float(np.clip(1.0 - ep * 15, 0, 1))
    crash_bear = float(np.clip(ep * 15, 0, 1))
    ent_penalty = float(np.clip((ent - 2.0) * 0.1, 0, 0.3)) if ent > 2.0 else 0.0
    return {
        "mu": sf(d.mu),
        "sigma": sf(d.sigma),
        "extreme_event_prob": ep,
        "crash_risk": cr,
        "median": sf(d.median()),
        "entropy": ent,
        "_bull": max(0, crash_bull - ent_penalty),
        "_bear": min(1, crash_bear + ent_penalty),
    }


def run_spline(r):
    from scipy.stats import johnsonsu

    try:
        a, b, loc, scale = johnsonsu.fit(r)
        ql = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        qv = johnsonsu.ppf(ql, a, b, loc, scale)
        q = {f"q{int(p * 100):02d}": sf(v) for p, v in zip(ql, qv)}
        mn = sf(johnsonsu.mean(a, b, loc, scale))
        std = sf(johnsonsu.std(a, b, loc, scale))
        skw = sf(johnsonsu.stats(a, b, loc, scale, moments="s"), 4)
        krt = sf(johnsonsu.stats(a, b, loc, scale, moments="k"), 4)
        ent = sf(johnsonsu.entropy(a, b, loc, scale), 4)
        unc = sf(np.clip(abs(q["q95"] - q["q05"]) * 50, 0, 1), 4)
        upside = float(q["q75"]) - float(q["q50"])
        downside = float(q["q50"]) - float(q["q25"])
        ratio = upside / (abs(downside) + 1e-8)
        bull_signal = float(np.clip(ratio * 0.4, 0, 1))
        bear_signal = float(np.clip((1 / ratio) * 0.4, 0, 1)) if ratio > 0 else 0.5
    except Exception:
        q = {
            f"q{int(p * 100):02d}": sf(np.percentile(r, p * 100))
            for p in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        }
        mn = sf(np.mean(r))
        std = sf(np.std(r))
        skw = krt = ent = unc = 0.0
        bull_signal = bear_signal = 0.3
    return {
        **q,
        "mean": mn,
        "std": std,
        "skewness": skw,
        "kurtosis": krt,
        "entropy": ent,
        "uncertainty": unc,
        "_bull": bull_signal,
        "_bear": bear_signal,
    }


def run_quantile(r):
    d = QuantileDistribution.fit(r)
    v95 = sf(d.ppf(0.05))
    v99 = sf(d.ppf(0.01))
    s = d.rvs(size=10000, random_state=42)
    cv = sf(np.mean(s[s <= v95]))
    tr = sf(np.clip(-cv * 10, 0, 1), 4)
    vbr = sf(np.mean(r < v95), 4)
    bull_signal = float(np.clip(1.0 - tr - vbr * 2, 0, 1))
    bear_signal = float(np.clip(tr + vbr * 2, 0, 1))
    return {
        "mu": sf(d.mu),
        "sigma": sf(d.sigma),
        "alpha_shape": sf(d.alpha, 4),
        "beta_shape": sf(d.beta, 4),
        "VaR_95": v95,
        "VaR_99": v99,
        "CVaR_95": cv,
        "tail_risk_score": tr,
        "var_breach_rate": vbr,
        "skewness": sf(d.skewness(), 4),
        "kurtosis": sf(d.kurtosis(), 4),
        "_bull": bull_signal,
        "_bear": bear_signal,
    }


def run_davies(r, w=20):
    pos = np.abs(r) + 1e-8
    p = DaviesDistribution.fit(pos)
    a, b, th, k = [sf(x, 4) for x in p]
    d = DaviesDistribution(alpha=a, beta=b, theta=th, k=k)
    rv = float(np.mean(np.abs(r[-w:])))
    ov = float(np.mean(pos))
    vr = sf(rv / (ov + 1e-10), 4)
    try:
        sc = float(np.clip(d.cdf(np.array([rv]))[0], 0, 1))
    except Exception:
        sc = 0.5
    reg = "crisis" if sc > 0.85 else "stress" if sc > 0.65 else "caution" if sc > 0.40 else "normal"
    stability = float(np.clip(1.0 - abs(vr - 1.0), 0, 1))
    bull_signal = float(np.clip((1.0 - sc) * 0.8 + stability * 0.2, 0, 1))
    bear_signal = float(np.clip(sc * 0.8 + (1 - stability) * 0.2, 0, 1))
    return {
        "alpha": a,
        "beta": b,
        "theta": th,
        "k": k,
        "mean": sf(d.mean()),
        "std": sf(d.std()),
        "stress_score": sf(sc, 4),
        "vol_ratio": vr,
        "regime": reg,
        "_bull": bull_signal,
        "_bear": bear_signal,
    }


def make_decision(frac, fractal, sinh, slash, spline, quant, davies):
    weights = {
        "davies": 0.25,
        "slash": 0.20,
        "spline": 0.20,
        "quant": 0.18,
        "frac": 0.10,
        "fractal": 0.07,
    }
    models = {
        "davies": davies,
        "slash": slash,
        "spline": spline,
        "quant": quant,
        "frac": frac,
        "fractal": fractal,
    }
    skew_boost = sinh["_bull"] - sinh["_bear"]
    bull_score = sum(models[k]["_bull"] * weights[k] for k in weights)
    bear_score = sum(models[k]["_bear"] * weights[k] for k in weights)
    bull_score = float(np.clip(bull_score + skew_boost * 0.10, 0, 1))
    bear_score = float(np.clip(bear_score - skew_boost * 0.10, 0, 1))
    net = bull_score - bear_score
    raw_ratio = abs(net) / (bull_score + bear_score + 1e-6)
    confidence = 1.0 / (1.0 + math.exp(-8.0 * (raw_ratio - 0.25)))
    decision = "BUY" if net > 0.08 else "SELL" if net < -0.08 else "HOLD"
    breakdown = {}
    for k, m in models.items():
        breakdown[k] = {"bull": sf(m["_bull"], 3), "bear": sf(m["_bear"], 3), "weight": weights[k]}
    breakdown["sinh"] = {
        "bull": sf(sinh["_bull"], 3),
        "bear": sf(sinh["_bear"], 3),
        "weight": 0.0,
        "note": "skew modifier",
    }
    return {
        "decision": decision,
        "bull_score": sf(bull_score, 4),
        "bear_score": sf(bear_score, 4),
        "net_score": sf(net, 4),
        "confidence": sf(confidence, 4),
        "breakdown": breakdown,
    }


def fetch_live_quote(ticker):
    ticker, suggestion = resolve(ticker)
    cached = cache_get("quote", ticker)
    if cached:
        return cached

    stock = yf.Ticker(ticker)
    info = {}
    fast_info = {}

    try:
        fast_info = to_mapping(stock.fast_info)
    except Exception:
        fast_info = {}
    try:
        info = stock.info or {}
    except Exception:
        info = {}

    intraday = pd.DataFrame()
    daily = pd.DataFrame()
    try:
        intraday = stock.history(period="1d", interval="1m", auto_adjust=False, prepost=False)
    except Exception:
        intraday = pd.DataFrame()
    try:
        daily = stock.history(period="5d", interval="1d", auto_adjust=False, prepost=False)
    except Exception:
        daily = pd.DataFrame()

    if intraday is None or intraday.empty:
        intraday = pd.DataFrame()
    if daily is None or daily.empty:
        daily = pd.DataFrame()

    price = None
    previous_close = None
    day_open = None
    day_high = None
    day_low = None
    volume = None
    updated_at = utc_now()

    if not intraday.empty:
        last_row = intraday.iloc[-1]
        price = to_number(last_row.get("Close"))
        day_open = to_number(intraday.iloc[0].get("Open"))
        day_high = to_number(intraday["High"].max())
        day_low = to_number(intraday["Low"].min())
        volume = to_int(intraday["Volume"].sum())
        try:
            updated_at = intraday.index[-1].to_pydatetime().astimezone(timezone.utc).replace(microsecond=0).isoformat()
        except Exception:
            updated_at = utc_now()

    if not daily.empty:
        if previous_close is None or previous_close == 0:
            if len(daily) >= 2:
                previous_close = to_number(daily["Close"].iloc[-2])
            else:
                previous_close = to_number(daily["Open"].iloc[-1])
        if not price:
            price = to_number(daily["Close"].iloc[-1])
        if not day_open:
            day_open = to_number(daily["Open"].iloc[-1])
        if not day_high:
            day_high = to_number(daily["High"].iloc[-1])
        if not day_low:
            day_low = to_number(daily["Low"].iloc[-1])
        if not volume:
            volume = to_int(daily["Volume"].iloc[-1])

    if not price:
        price = to_number(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or fast_info.get("lastPrice")
            or fast_info.get("regularMarketPrice")
        )
    if not previous_close:
        previous_close = to_number(
            info.get("previousClose")
            or info.get("regularMarketPreviousClose")
            or fast_info.get("previousClose")
        )
    if not day_open:
        day_open = to_number(info.get("open") or fast_info.get("open"))
    if not day_high:
        day_high = to_number(info.get("dayHigh") or fast_info.get("dayHigh"))
    if not day_low:
        day_low = to_number(info.get("dayLow") or fast_info.get("dayLow"))
    if not volume:
        volume = to_int(info.get("volume") or fast_info.get("lastVolume"))

    if not price:
        raise ValueError(f"No live quote found for '{ticker}'")

    change = price - previous_close if previous_close else 0.0
    change_percent = (change / previous_close * 100.0) if previous_close else 0.0
    market_cap = to_number(info.get("marketCap") or fast_info.get("marketCap"))
    avg_volume = to_int(info.get("averageVolume") or fast_info.get("tenDayAverageVolume"))
    currency = info.get("currency") or fast_info.get("currency") or "USD"
    exchange = info.get("exchange") or info.get("fullExchangeName") or ""
    name = (
        info.get("shortName")
        or info.get("longName")
        or info.get("displayName")
        or ticker
    )

    payload = {
        "ticker": ticker,
        "suggestion": suggestion,
        "name": name,
        "price": sf(price, 2),
        "change": sf(change, 2),
        "change_percent": sf(change_percent, 2),
        "previous_close": sf(previous_close, 2),
        "open": sf(day_open, 2),
        "day_high": sf(day_high, 2),
        "day_low": sf(day_low, 2),
        "volume": volume,
        "avg_volume": avg_volume,
        "market_cap": market_cap,
        "currency": currency,
        "exchange": exchange,
        "updated_at": updated_at,
        "is_market_open": bool(info.get("marketState") == "REGULAR" or info.get("regularMarketOpen", False)),
    }
    return cache_set("quote", ticker, payload, ttl=20)


def fetch_chart_history(ticker, range_key="1d"):
    ticker, _ = resolve(ticker)
    allowed = {
        "1d": ("1d", "5m"),
        "5d": ("5d", "30m"),
        "1mo": ("1mo", "1d"),
        "3mo": ("3mo", "1d"),
        "1y": ("1y", "1d"),
    }
    if range_key not in allowed:
        raise ValueError("Invalid range. Use 1d, 5d, 1mo, 3mo, or 1y.")

    cache_key = f"{ticker}:{range_key}"
    cached = cache_get("history", cache_key)
    if cached:
        return cached

    period, interval = allowed[range_key]
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval, auto_adjust=False, prepost=False)
    if df is None or df.empty:
        raise ValueError(f"No chart data found for '{ticker}'")

    points = []
    for idx, row in df.tail(200).iterrows():
        try:
            dt = idx.to_pydatetime().astimezone(timezone.utc).replace(microsecond=0).isoformat()
        except Exception:
            dt = utc_now()
        points.append(
            {
                "time": dt,
                "close": sf(row.get("Close"), 3),
                "open": sf(row.get("Open"), 3),
                "high": sf(row.get("High"), 3),
                "low": sf(row.get("Low"), 3),
                "volume": to_int(row.get("Volume")),
            }
        )

    payload = {
        "ticker": ticker,
        "range": range_key,
        "interval": interval,
        "points": points,
        "updated_at": utc_now(),
    }
    return cache_set("history", cache_key, payload, ttl=60)


def get_watchlist_quotes(tickers):
    rows = []
    for ticker in tickers:
        try:
            q = fetch_live_quote(ticker)
            rows.append(
                {
                    "ticker": q["ticker"],
                    "price": q["price"],
                    "change": q["change"],
                    "change_percent": q["change_percent"],
                    "currency": q["currency"],
                }
            )
        except Exception:
            rows.append(
                {
                    "ticker": ticker,
                    "price": 0,
                    "change": 0,
                    "change_percent": 0,
                    "currency": "",
                    "error": True,
                }
            )
    return rows


def analyze_ticker(ticker):
    ticker, suggestion = resolve(ticker)
    cached = cache_get("analysis", ticker)
    if cached:
        remember_search(ticker)
        return cached

    started = time.time()
    df = features(clean(fetch(ticker)))
    r = df["lr"].values
    px = df["close"].values
    frac = run_fractional(r)
    fractal = run_fractal(r)
    sinh = run_sinh(r)
    slash = run_slash(r)
    spline = run_spline(r)
    quant = run_quantile(r)
    davies = run_davies(r)
    decision = make_decision(frac, fractal, sinh, slash, spline, quant, davies)
    quote = fetch_live_quote(ticker)

    momentum_20d = sf(((df["close"].iloc[-1] / df["close"].iloc[-20]) - 1) * 100, 2) if len(df) >= 20 else 0
    annualized_vol = sf(df["vol"].iloc[-1] * 100, 2) if "vol" in df.columns and len(df) else 0
    rsi_14 = sf(df["rsi"].iloc[-1], 2) if "rsi" in df.columns and len(df) else 0

    def pub(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    payload = _clean(
        {
            "ticker": ticker,
            "suggestion": suggestion,
            "n_sessions": len(df),
            "mean_return": sf(r.mean()),
            "std_return": sf(r.std()),
            "momentum_20d": momentum_20d,
            "annualized_volatility": annualized_vol,
            "rsi_14": rsi_14,
            "fractional": pub(frac),
            "fractal": pub(fractal),
            "sinh_arcsinh": pub(sinh),
            "slash": pub(slash),
            "neural_spline": pub(spline),
            "quantile": pub(quant),
            "davies": pub(davies),
            "decision": decision,
            "live_quote": quote,
            "_prices": px[-160:].tolist(),
            "_returns": r[-320:].tolist(),
            "meta": {
                "generated_at": utc_now(),
                "analysis_ms": int((time.time() - started) * 1000),
            },
        }
    )
    remember_search(ticker)
    return cache_set("analysis", ticker, payload, ttl=300)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PulseVest Market Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#081018;
      --bg2:#0c1624;
      --panel:rgba(11,22,34,.78);
      --panel-2:rgba(16,32,48,.88);
      --border:rgba(121,156,191,.16);
      --text:#ebf6f4;
      --muted:#91a8b7;
      --green:#00d09c;
      --green-soft:rgba(0,208,156,.16);
      --red:#ff6b6b;
      --amber:#ffbf5f;
      --blue:#63b3ff;
      --violet:#9f7aea;
      --shadow:0 20px 60px rgba(0,0,0,.35);
    }
    body{
      min-height:100vh;
      font-family:Inter,Segoe UI,system-ui,-apple-system,sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(0,208,156,.18), transparent 30%),
        radial-gradient(circle at top right, rgba(99,179,255,.18), transparent 28%),
        radial-gradient(circle at bottom center, rgba(159,122,234,.16), transparent 32%),
        linear-gradient(180deg, #071019 0%, #0a1320 35%, #071018 100%);
    }
    .shell{max-width:1320px;margin:0 auto;padding:22px 18px 48px}
    .topbar{
      display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;
      margin-bottom:20px
    }
    .brand{display:flex;align-items:center;gap:14px}
    .logo{
      width:44px;height:44px;border-radius:14px;
      background:linear-gradient(135deg,var(--green),#0aa5ff 70%,var(--violet));
      box-shadow:0 16px 34px rgba(0,208,156,.22);
      display:grid;place-items:center;color:#f5fffd;font-weight:900;font-size:20px
    }
    .title{font-size:1.1rem;font-weight:800;letter-spacing:-.02em}
    .subtitle{font-size:.84rem;color:var(--muted);margin-top:2px}
    .live-pill{
      display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:999px;
      border:1px solid var(--border);background:rgba(9,18,27,.64);color:var(--muted);font-size:.84rem
    }
    .dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 0 6px rgba(0,208,156,.12)}
    .hero{
      display:grid;grid-template-columns:1.5fr .95fr;gap:18px;margin-bottom:18px
    }
    @media(max-width:1000px){.hero{grid-template-columns:1fr}}
    .panel{
      background:var(--panel);border:1px solid var(--border);border-radius:28px;
      box-shadow:var(--shadow);backdrop-filter:blur(14px)
    }
    .search-panel{padding:24px;position:relative;overflow:hidden}
    .search-panel::after{
      content:'';position:absolute;right:-80px;top:-90px;width:250px;height:250px;border-radius:50%;
      background:radial-gradient(circle, rgba(0,208,156,.18), transparent 65%);pointer-events:none
    }
    .search-copy{max-width:620px;position:relative;z-index:1}
    .eyebrow{font-size:.78rem;text-transform:uppercase;letter-spacing:.18em;color:#86ffe1;margin-bottom:10px}
    h1{font-size:2.1rem;line-height:1.08;letter-spacing:-.04em;margin-bottom:10px}
    .lead{color:var(--muted);font-size:.96rem;line-height:1.65;margin-bottom:20px}
    .search-row{display:flex;gap:12px;flex-wrap:wrap}
    .search-box{
      flex:1;min-width:240px;display:flex;align-items:center;gap:10px;padding:14px 16px;border-radius:18px;
      background:rgba(5,12,19,.52);border:1px solid rgba(160,190,220,.12)
    }
    .search-box input{
      width:100%;background:none;border:none;outline:none;color:var(--text);font-size:1rem
    }
    .search-box input::placeholder{color:#6e8596}
    .btn{
      border:none;border-radius:18px;padding:14px 18px;font-weight:800;cursor:pointer;
      transition:transform .18s ease, opacity .18s ease
    }
    .btn:active{transform:scale(.98)}
    .btn-primary{background:linear-gradient(135deg,var(--green),#00b386);color:#f6fffd}
    .btn-secondary{background:rgba(99,179,255,.14);color:#c7e2ff;border:1px solid rgba(99,179,255,.18)}
    .quick{
      display:flex;gap:10px;flex-wrap:wrap;margin-top:18px;position:relative;z-index:1
    }
    .chip{
      border:none;padding:10px 14px;border-radius:999px;background:rgba(17,31,44,.92);color:#cfe3f0;
      border:1px solid rgba(145,168,183,.14);cursor:pointer;font-size:.85rem
    }
    .chip.active,.chip:hover{border-color:rgba(0,208,156,.35);color:#eafff9}
    .status-card{padding:22px;display:flex;flex-direction:column;justify-content:space-between}
    .status-top{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:14px}
    .status-title{font-size:.9rem;color:var(--muted)}
    .status-price{font-size:2.3rem;font-weight:900;letter-spacing:-.04em;margin-top:12px}
    .change{
      display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border-radius:999px;font-weight:700;font-size:.92rem
    }
    .pos{background:rgba(0,208,156,.14);color:#77ffd9}
    .neg{background:rgba(255,107,107,.14);color:#ffb5b5}
    .flat{background:rgba(255,191,95,.14);color:#ffd58d}
    .mini-grid{
      display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:18px
    }
    .metric{
      padding:14px;border-radius:20px;background:rgba(7,15,23,.55);border:1px solid rgba(145,168,183,.1)
    }
    .metric .label{font-size:.76rem;color:var(--muted);margin-bottom:7px;text-transform:uppercase;letter-spacing:.1em}
    .metric .value{font-size:1.02rem;font-weight:800}
    .grid{
      display:grid;grid-template-columns:1.15fr .85fr;gap:18px;margin-bottom:18px
    }
    @media(max-width:1100px){.grid{grid-template-columns:1fr}}
    .stack{display:flex;flex-direction:column;gap:18px}
    .card{padding:20px}
    .card-head{
      display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px
    }
    .card-title{font-size:.9rem;text-transform:uppercase;letter-spacing:.14em;color:#98f1da}
    .card-sub{font-size:.8rem;color:var(--muted);margin-top:5px}
    .decision-pill{
      display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:999px;font-size:.92rem;font-weight:800
    }
    .BUY{background:rgba(0,208,156,.14);color:#6dffd7}
    .SELL{background:rgba(255,107,107,.14);color:#ffb7b7}
    .HOLD{background:rgba(255,191,95,.14);color:#ffd58d}
    .overview{
      display:grid;grid-template-columns:repeat(2,1fr);gap:12px
    }
    .stat{
      padding:14px;border-radius:20px;background:rgba(7,15,23,.55);border:1px solid rgba(145,168,183,.1)
    }
    .stat .k{font-size:.78rem;color:var(--muted);margin-bottom:8px}
    .stat .v{font-size:1.15rem;font-weight:800}
    .conf{
      margin-top:10px;padding:14px;border-radius:20px;background:linear-gradient(180deg, rgba(10,18,27,.72), rgba(14,27,40,.55));
      border:1px solid rgba(145,168,183,.1)
    }
    .bar{height:10px;border-radius:999px;background:rgba(255,255,255,.06);overflow:hidden;margin-top:10px}
    .bar > span{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,var(--green),#0aa5ff)}
    .watchlist-row,.recent-row{
      display:flex;gap:10px;flex-wrap:wrap
    }
    .watch-item,.recent-item{
      padding:12px 14px;border-radius:18px;background:rgba(7,15,23,.52);border:1px solid rgba(145,168,183,.1);
      min-width:132px;color:#eef9ff;cursor:pointer;text-align:left
    }
    .watch-item .sym,.recent-item .sym{font-weight:800;font-size:.9rem;color:#eef9ff}
    .watch-item .price{font-size:1rem;font-weight:800;margin-top:8px;color:#f4fcff}
    .watch-item .pct{font-size:.85rem;margin-top:4px}
    .chart-card{padding:20px}
    .small-chart{height:240px}
    .split{display:grid;grid-template-columns:1fr 1fr;gap:18px}
    @media(max-width:900px){.split{grid-template-columns:1fr}}
    .recommend-card{
      padding:22px;border-radius:26px;
      background:
        radial-gradient(circle at top left, rgba(0,208,156,.16), transparent 34%),
        radial-gradient(circle at bottom right, rgba(99,179,255,.18), transparent 32%),
        linear-gradient(145deg, rgba(10,20,30,.94), rgba(16,31,45,.92));
      border:1px solid rgba(145,168,183,.12)
    }
    .recommend-top{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;margin-bottom:16px}
    .recommend-label{font-size:.82rem;text-transform:uppercase;letter-spacing:.16em;color:#98f1da}
    .recommend-title{font-size:2.3rem;line-height:1;font-weight:900;letter-spacing:-.05em;margin-top:8px}
    .recommend-text{font-size:.95rem;color:#cddde6;line-height:1.7;max-width:700px}
    .recommend-meta{
      display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px
    }
    @media(max-width:700px){.recommend-meta{grid-template-columns:1fr}}
    .recommend-box{
      padding:14px 16px;border-radius:18px;background:rgba(8,16,25,.58);border:1px solid rgba(145,168,183,.1)
    }
    .recommend-box .k{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px}
    .recommend-box .v{font-size:1.05rem;font-weight:800;color:#f1fbff}
    .breakdown{display:flex;flex-direction:column;gap:12px}
    .row{
      display:grid;grid-template-columns:100px 1fr 58px;align-items:center;gap:10px
    }
    .row .name{font-size:.82rem;color:#dcebf2}
    .bars{display:flex;flex-direction:column;gap:6px}
    .track{height:8px;border-radius:999px;background:rgba(255,255,255,.06);overflow:hidden}
    .fill-green{height:100%;background:linear-gradient(90deg,var(--green),#11b7ff)}
    .fill-red{height:100%;background:linear-gradient(90deg,#ff7a7a,#ff4d6d)}
    .weight{font-size:.78rem;color:var(--muted);text-align:right}
    .models{display:flex;flex-direction:column;gap:12px}
    .model{
      border:1px solid rgba(145,168,183,.1);background:rgba(7,15,23,.48);border-radius:20px;overflow:hidden
    }
    .model-hd{
      padding:16px 18px;display:flex;justify-content:space-between;gap:16px;align-items:center;cursor:pointer
    }
    .model-left{display:flex;gap:14px;align-items:center}
    .icon{
      width:40px;height:40px;border-radius:14px;background:rgba(0,208,156,.12);display:grid;place-items:center;font-size:1.1rem
    }
    .model-name{font-weight:800}
    .model-tag{font-size:.8rem;color:var(--muted);margin-top:3px}
    .pills{display:flex;gap:8px;flex-wrap:wrap}
    .pill{
      padding:6px 10px;border-radius:999px;font-size:.76rem;font-weight:700
    }
    .pill.bull{background:rgba(0,208,156,.12);color:#74ffd9}
    .pill.bear{background:rgba(255,107,107,.12);color:#ffb7b7}
    .model-body{display:none;padding:0 18px 18px}
    .model.open .model-body{display:block}
    .kv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
    .kv{
      padding:12px;border-radius:16px;background:rgba(12,23,34,.8);border:1px solid rgba(145,168,183,.08)
    }
    .kv .k{font-size:.76rem;color:var(--muted);margin-bottom:6px}
    .kv .v{font-size:.95rem;font-weight:800}
    .note{
      font-size:.88rem;color:var(--muted);line-height:1.6
    }
    .banner{
      margin-bottom:16px;padding:14px 16px;border-radius:18px;font-size:.92rem;display:none
    }
    .error{background:rgba(255,107,107,.12);border:1px solid rgba(255,107,107,.24);color:#ffcbcb}
    .info{background:rgba(99,179,255,.12);border:1px solid rgba(99,179,255,.18);color:#cfe6ff}
    .footer-note{margin-top:14px;color:var(--muted);font-size:.82rem}
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div class="brand">
        <div class="logo">P</div>
        <div>
          <div class="title">PulseVest Market Dashboard</div>
          <div class="subtitle">Live price tracking and model-based risk analytics in one screen.</div>
        </div>
      </div>
      <div class="live-pill"><span class="dot"></span><span id="liveMeta">Waiting for first quote…</span></div>
    </div>

    <div class="hero">
      <div class="panel search-panel">
        <div class="search-copy">
          <div class="eyebrow">Smart stock terminal</div>
          <h1>Track live stock price, market mood, and a clear model-based action in one place.</h1>
          <div class="lead">Enter any supported ticker like <b>AAPL</b>, <b>MSFT</b>, <b>NVDA</b>, or Indian symbols like <b>RELIANCE.NS</b> and the app will load a live quote plus a full model-based recommendation.</div>
          <div class="search-row">
            <div class="search-box">
              <svg width="18" height="18" fill="none" stroke="#86a2b5" stroke-width="2" viewBox="0 0 24 24">
                <circle cx="11" cy="11" r="8"></circle>
                <path d="m21 21-4.35-4.35"></path>
              </svg>
              <input id="tickerInput" value="AAPL" placeholder="Search ticker symbol" />
            </div>
            <button class="btn btn-primary" onclick="analyze()">Analyze</button>
            <button class="btn btn-secondary" onclick="refreshQuote()">Refresh live</button>
          </div>
          <div class="quick" id="quickChips"></div>
        </div>
      </div>

      <div class="panel status-card">
        <div>
          <div class="status-top">
            <div>
              <div class="status-title">Live quote</div>
              <div id="quoteName" class="subtitle">Choose a ticker</div>
            </div>
            <div id="marketBadge" class="change flat">Market snapshot</div>
          </div>
          <div id="quoteTicker" class="subtitle">--</div>
          <div id="quotePrice" class="status-price">--</div>
          <div id="quoteChange" class="change flat" style="margin-top:12px">--</div>
          <div class="mini-grid">
            <div class="metric"><div class="label">Open</div><div class="value" id="metricOpen">--</div></div>
            <div class="metric"><div class="label">Prev Close</div><div class="value" id="metricPrev">--</div></div>
            <div class="metric"><div class="label">Day Range</div><div class="value" id="metricRange">--</div></div>
            <div class="metric"><div class="label">Volume</div><div class="value" id="metricVolume">--</div></div>
          </div>
        </div>
        <div class="footer-note" id="quoteUpdated">Waiting for live data…</div>
      </div>
    </div>

    <div id="errorBanner" class="banner error"></div>
    <div id="infoBanner" class="banner info"></div>

    <div class="grid">
      <div class="stack">
        <div class="panel recommend-card">
          <div class="recommend-top">
            <div>
              <div class="recommend-label">Model recommendation</div>
              <div id="recommendTitle" class="recommend-title">HOLD</div>
            </div>
            <div id="decisionPill" class="decision-pill HOLD">HOLD</div>
          </div>
          <div id="recommendText" class="recommend-text">Run an analysis to get a stock action from the fitted models.</div>
          <div class="recommend-meta">
            <div class="recommend-box"><div class="k">Confidence</div><div class="v" id="confidenceValue">--</div></div>
            <div class="recommend-box"><div class="k">Bull score</div><div class="v" id="bullValue">--</div></div>
            <div class="recommend-box"><div class="k">Bear score</div><div class="v" id="bearValue">--</div></div>
          </div>
        </div>

        <div class="split">
          <div class="panel chart-card">
            <div class="card-head">
              <div>
                <div class="card-title">Pipeline price view</div>
                <div class="card-sub">Longer look from the full analytics run.</div>
              </div>
            </div>
            <div id="priceChart" class="small-chart"></div>
          </div>
          <div class="panel chart-card">
            <div class="card-head">
              <div>
                <div class="card-title">Return distribution</div>
                <div class="card-sub">How recent returns are spread out.</div>
              </div>
            </div>
            <div id="returnChart" class="small-chart"></div>
          </div>
        </div>

        <div class="panel card">
          <div class="card-head">
            <div>
              <div class="card-title">Signal breakdown</div>
              <div class="card-sub">Bull vs bear weight from each model.</div>
            </div>
          </div>
          <div id="breakdown" class="breakdown"></div>
        </div>

        <div class="panel card">
          <div class="card-head">
            <div>
              <div class="card-title">Model details</div>
              <div class="card-sub">Tap a model to inspect its values.</div>
            </div>
          </div>
          <div id="models" class="models"></div>
        </div>
      </div>

      <div class="stack">
        <div class="panel card">
          <div class="card-head">
            <div>
              <div class="card-title">Decision summary</div>
              <div class="card-sub">Output from the backend risk engine.</div>
            </div>
          </div>
          <div class="overview">
            <div class="stat"><div class="k">Stress regime</div><div class="v" id="regimeValue">--</div></div>
            <div class="stat"><div class="k">20D momentum</div><div class="v" id="momentumValue">--</div></div>
            <div class="stat"><div class="k">RSI 14</div><div class="v" id="rsiValue">--</div></div>
            <div class="stat"><div class="k">Sessions used</div><div class="v" id="sessionsValue">--</div></div>
            <div class="stat"><div class="k">Analysis time</div><div class="v" id="analysisTimeValue">--</div></div>
          </div>
          <div class="conf">
            <div class="k">Confidence bar</div>
            <div class="bar"><span id="confidenceBar" style="width:0%"></span></div>
            <div class="footer-note" id="analysisMeta">No analysis yet.</div>
          </div>
        </div>

        <div class="panel card">
          <div class="card-head">
            <div>
              <div class="card-title">Watchlist</div>
              <div class="card-sub">Live backend quote snapshots for popular tickers.</div>
            </div>
          </div>
          <div id="watchlist" class="watchlist-row"></div>
        </div>

        <div class="panel card">
          <div class="card-head">
            <div>
              <div class="card-title">Recent searches</div>
              <div class="card-sub">Stored in backend memory during this session.</div>
            </div>
          </div>
          <div id="recentSearches" class="recent-row"></div>
        </div>

        <div class="panel card">
          <div class="card-head">
            <div>
              <div class="card-title">Backend features added</div>
              <div class="card-sub">What changed beyond the UI.</div>
            </div>
          </div>
          <div class="note">
            <p>Live quote API with auto-refresh, in-memory recent search tracking, watchlist endpoint, short-term caching for faster repeat requests, and a more prominent model-based stock action are now built into the app.</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const $ = id => document.getElementById(id);
    const COLORS = {
      green:'#00d09c',
      blue:'#63b3ff',
      red:'#ff6b6b',
      amber:'#ffbf5f',
      grid:'rgba(145,168,183,.12)',
      text:'#dbe8ef'
    };
    const chartLayout = {
      paper_bgcolor:'rgba(0,0,0,0)',
      plot_bgcolor:'rgba(0,0,0,0)',
      font:{color:'#9bb0be'},
      margin:{t:10,r:10,b:40,l:48},
      xaxis:{gridcolor:COLORS.grid, zerolinecolor:COLORS.grid},
      yaxis:{gridcolor:COLORS.grid, zerolinecolor:COLORS.grid}
    };
    const chartConfig = {displayModeBar:false,responsive:true};
    const state = {
      ticker:'AAPL',
      quoteTimer:null
    };

    const MODEL_META = {
      frac:{icon:'〜',name:'Fractional Distribution',tag:'Long-memory and regime signal'},
      fractal:{icon:'❄',name:'Fractal Distribution',tag:'Market structure and complexity'},
      sinh:{icon:'⟛',name:'Sinh-Arcsinh',tag:'Skew and tail shape'},
      slash:{icon:'⚡',name:'Slash Distribution',tag:'Crash and extreme event risk'},
      spline:{icon:'📈',name:'JohnsonSU / Neural Spline',tag:'Quantile forecast spread'},
      quant:{icon:'📉',name:'Quantile Distribution',tag:'VaR and CVaR risk'},
      davies:{icon:'🧭',name:'Davies Distribution',tag:'Stress regime classifier'}
    };

    function formatMoney(v, currency='USD'){
      if(v === null || v === undefined || Number.isNaN(Number(v))) return '--';
      return new Intl.NumberFormat('en-US',{style:'currency',currency:currency,maximumFractionDigits:2}).format(Number(v));
    }

    function formatCompact(v){
      if(v === null || v === undefined || Number.isNaN(Number(v))) return '--';
      return new Intl.NumberFormat('en-US',{notation:'compact',maximumFractionDigits:2}).format(Number(v));
    }

    function formatSignedPercent(v){
      if(v === null || v === undefined || Number.isNaN(Number(v))) return '--';
      const n = Number(v);
      const sign = n > 0 ? '+' : '';
      return `${sign}${n.toFixed(2)}%`;
    }

    function formatSignedValue(v, currency='USD'){
      if(v === null || v === undefined || Number.isNaN(Number(v))) return '--';
      const n = Number(v);
      const sign = n > 0 ? '+' : '';
      return `${sign}${formatMoney(n, currency)}`;
    }

    function banner(id, text){
      const el = $(id);
      if(!text){
        el.style.display = 'none';
        el.textContent = '';
        return;
      }
      el.style.display = 'block';
      el.textContent = text;
    }

    function setQuickChips(){
      $('quickChips').innerHTML = ['AAPL','MSFT','NVDA','TSLA','RELIANCE.NS','INFY.NS','HDFCBANK.NS']
        .map(t => `<button class="chip ${state.ticker===t?'active':''}" onclick="pickTicker('${t}')">${t}</button>`)
        .join('');
    }

    function pickTicker(ticker){
      $('tickerInput').value = ticker;
      state.ticker = ticker;
      setQuickChips();
      analyze();
    }

    function renderQuote(q){
      const cls = q.change > 0 ? 'pos' : q.change < 0 ? 'neg' : 'flat';
      $('quoteName').textContent = `${q.name || q.ticker}${q.exchange ? ' · '+q.exchange : ''}`;
      $('quoteTicker').textContent = q.ticker;
      $('quotePrice').textContent = formatMoney(q.price, q.currency || 'USD');
      $('quoteChange').className = `change ${cls}`;
      $('quoteChange').textContent = `${formatSignedValue(q.change, q.currency || 'USD')} (${formatSignedPercent(q.change_percent)})`;
      $('metricOpen').textContent = formatMoney(q.open, q.currency || 'USD');
      $('metricPrev').textContent = formatMoney(q.previous_close, q.currency || 'USD');
      $('metricRange').textContent = `${formatMoney(q.day_low, q.currency || 'USD')} - ${formatMoney(q.day_high, q.currency || 'USD')}`;
      $('metricVolume').textContent = formatCompact(q.volume);
      $('quoteUpdated').textContent = `Updated ${new Date(q.updated_at).toLocaleString()} · Market cap ${formatCompact(q.market_cap)}`;
      $('marketBadge').className = `change ${cls}`;
      $('marketBadge').textContent = q.is_market_open ? 'Live market' : 'Latest close';
      $('liveMeta').textContent = `${q.ticker} updated ${new Date(q.updated_at).toLocaleTimeString()}`;
    }

    function renderAnalysis(d){
      const dec = d.decision || {};
      $('decisionPill').className = `decision-pill ${dec.decision || 'HOLD'}`;
      $('decisionPill').textContent = dec.decision || 'HOLD';
      $('recommendTitle').textContent = dec.decision || 'HOLD';
      const confidencePct = Math.round((dec.confidence || 0) * 100);
      $('confidenceValue').textContent = `${confidencePct}%`;
      $('regimeValue').textContent = d.davies?.regime || '--';
      $('bullValue').textContent = dec.bull_score ?? '--';
      $('bearValue').textContent = dec.bear_score ?? '--';
      $('momentumValue').textContent = `${Number(d.momentum_20d || 0).toFixed(2)}%`;
      $('rsiValue').textContent = Number(d.rsi_14 || 0).toFixed(2);
      $('sessionsValue').textContent = d.n_sessions || '--';
      $('analysisTimeValue').textContent = `${d.meta?.analysis_ms || 0} ms`;
      const actionText = dec.decision === 'BUY'
        ? 'Models currently lean bullish, with stronger upside probability than downside risk.'
        : dec.decision === 'SELL'
          ? 'Models currently lean defensive, with downside risk stronger than upside probability.'
          : 'Models are mixed right now, so the signal stays neutral until a clearer edge appears.';
      $('recommendText').textContent = `${actionText} Confidence is ${confidencePct}% with ${d.davies?.regime || 'unknown'} market stress regime.`;
      $('confidenceBar').style.width = `${confidencePct}%`;
      $('analysisMeta').textContent = `${d.ticker} · ${d.n_sessions} sessions · analysis ${d.meta?.analysis_ms || 0} ms`;
      renderBreakdown(dec.breakdown || {});
      renderModelCards(d);
      renderPipelineCharts(d);
    }

    function renderBreakdown(bd){
      const order = ['davies','slash','spline','quant','frac','fractal','sinh'];
      const labels = {davies:'Davies',slash:'Slash',spline:'Spline',quant:'Quantile',frac:'Fractional',fractal:'Fractal',sinh:'Sinh'};
      $('breakdown').innerHTML = order.map(key => {
        const b = bd[key] || {};
        const bull = Number(b.bull || 0);
        const bear = Number(b.bear || 0);
        const weight = b.weight !== undefined ? Math.round(Number(b.weight)*100) : 0;
        return `
          <div class="row">
            <div class="name">${labels[key]}</div>
            <div class="bars">
              <div class="track"><div class="fill-green" style="width:${Math.round(bull*100)}%"></div></div>
              <div class="track"><div class="fill-red" style="width:${Math.round(bear*100)}%"></div></div>
            </div>
            <div class="weight">${weight}%</div>
          </div>
        `;
      }).join('');
    }

    function modelKv(data, badgeKeys=[]){
      return `
        <div class="kv-grid">
          ${Object.entries(data || {}).map(([k,v]) => `
            <div class="kv">
              <div class="k">${k}</div>
              <div class="v">${badgeKeys.includes(k) ? String(v).toUpperCase() : v}</div>
            </div>
          `).join('')}
        </div>
      `;
    }

    function renderModelCards(d){
      const maps = [
        {key:'frac',data:d.fractional,badges:['regime']},
        {key:'fractal',data:d.fractal,badges:['structure']},
        {key:'sinh',data:d.sinh_arcsinh,badges:['skewness_direction']},
        {key:'slash',data:d.slash,badges:['crash_risk']},
        {key:'spline',data:d.neural_spline,badges:[]},
        {key:'quant',data:d.quantile,badges:[]},
        {key:'davies',data:d.davies,badges:['regime']}
      ];
      const bd = d.decision?.breakdown || {};
      $('models').innerHTML = maps.map(({key,data,badges}) => {
        const meta = MODEL_META[key];
        const bull = Number(bd[key]?.bull || 0).toFixed(2);
        const bear = Number(bd[key]?.bear || 0).toFixed(2);
        return `
          <div class="model" id="model_${key}">
            <div class="model-hd" onclick="toggleModel('${key}')">
              <div class="model-left">
                <div class="icon">${meta.icon}</div>
                <div>
                  <div class="model-name">${meta.name}</div>
                  <div class="model-tag">${meta.tag}</div>
                </div>
              </div>
              <div class="pills">
                <span class="pill bull">Bull ${bull}</span>
                <span class="pill bear">Bear ${bear}</span>
              </div>
            </div>
            <div class="model-body">${modelKv(data, badges)}</div>
          </div>
        `;
      }).join('');
    }

    function toggleModel(key){
      document.getElementById(`model_${key}`).classList.toggle('open');
    }

    function renderPipelineCharts(d){
      Plotly.newPlot('priceChart', [{
        x:d._prices.map((_,i)=>i+1),
        y:d._prices,
        type:'scatter',
        mode:'lines',
        line:{color:COLORS.blue,width:2},
        fill:'tozeroy',
        fillcolor:'rgba(99,179,255,.08)'
      }], {...chartLayout, margin:{t:10,r:10,b:34,l:50}}, chartConfig);

      Plotly.newPlot('returnChart', [{
        x:d._returns,
        type:'histogram',
        nbinsx:48,
        marker:{color:'rgba(0,208,156,.75)'}
      }], {...chartLayout, margin:{t:10,r:10,b:34,l:44}}, chartConfig);
    }

    async function refreshQuote(){
      try{
        banner('errorBanner', '');
        const res = await fetch(`/api/quote?ticker=${encodeURIComponent(state.ticker)}`);
        const d = await res.json();
        if(!res.ok) throw new Error(d.error || 'Unable to load quote');
        if(d.suggestion) banner('infoBanner', d.suggestion); else banner('infoBanner', '');
        renderQuote(d);
      }catch(err){
        banner('errorBanner', err.message);
      }
    }

    async function loadWatchlist(){
      try{
        const res = await fetch('/api/watchlist');
        const d = await res.json();
        if(!res.ok) throw new Error(d.error || 'Unable to load watchlist');
        $('watchlist').innerHTML = (d.items || []).map(item => {
          const cls = item.change > 0 ? 'pos' : item.change < 0 ? 'neg' : 'flat';
          return `
            <button class="watch-item" onclick="pickTicker('${item.ticker}')">
              <div class="sym">${item.ticker}</div>
              <div class="price">${item.currency ? formatMoney(item.price, item.currency) : item.price}</div>
              <div class="pct ${cls.includes('pos') ? '' : ''}" style="color:${item.change > 0 ? '#77ffd9' : item.change < 0 ? '#ffb5b5' : '#ffd58d'}">
                ${formatSignedPercent(item.change_percent)}
              </div>
            </button>
          `;
        }).join('');
      }catch(err){
        $('watchlist').innerHTML = `<div class="note">${err.message}</div>`;
      }
    }

    async function loadRecent(){
      try{
        const res = await fetch('/api/recent');
        const d = await res.json();
        if(!res.ok) throw new Error(d.error || 'Unable to load recent searches');
        const items = d.items || [];
        $('recentSearches').innerHTML = items.length
          ? items.map(item => `
              <button class="recent-item" onclick="pickTicker('${item.ticker}')">
                <div class="sym">${item.ticker}</div>
                <div class="footer-note">${new Date(item.searched_at).toLocaleTimeString()}</div>
              </button>
            `).join('')
          : `<div class="note">Recent searches will appear here after you run an analysis.</div>`;
      }catch(err){
        $('recentSearches').innerHTML = `<div class="note">${err.message}</div>`;
      }
    }

    async function analyze(){
      try{
        state.ticker = $('tickerInput').value.trim().toUpperCase();
        if(!state.ticker) return;
        setQuickChips();
        banner('errorBanner','');
        banner('infoBanner',`Running analytics for ${state.ticker}…`);
        const res = await fetch(`/api/analyze?ticker=${encodeURIComponent(state.ticker)}`);
        const d = await res.json();
        if(!res.ok) throw new Error(d.error || 'Analysis failed');
        if(d.suggestion) banner('infoBanner', d.suggestion); else banner('infoBanner', `${d.ticker} analysis loaded successfully.`);
        renderQuote(d.live_quote);
        renderAnalysis(d);
        await Promise.all([loadWatchlist(), loadRecent()]);
        if(state.quoteTimer) clearInterval(state.quoteTimer);
        state.quoteTimer = setInterval(async () => {
          await refreshQuote();
          await loadWatchlist();
        }, 15000);
      }catch(err){
        banner('errorBanner', err.message);
        banner('infoBanner', '');
      }
    }

    $('tickerInput').addEventListener('keydown', e => {
      if(e.key === 'Enter') analyze();
    });

    setQuickChips();
    analyze();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/analyze")
def api_analyze():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return safe_json({"error": "ticker parameter required"}, 400)
    try:
        return safe_json(analyze_ticker(ticker))
    except ValueError as e:
        return safe_json({"error": str(e)}, 404)
    except CurlHTTPError as e:
        return safe_json({"error": f"Data fetch failed: {e}"}, 503)
    except Exception as e:
        return safe_json({"error": f"Pipeline error: {e}"}, 500)


@app.route("/api/quote")
def api_quote():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return safe_json({"error": "ticker parameter required"}, 400)
    try:
        return safe_json(fetch_live_quote(ticker))
    except ValueError as e:
        return safe_json({"error": str(e)}, 404)
    except Exception as e:
        return safe_json({"error": f"Quote error: {e}"}, 500)


@app.route("/api/history")
def api_history():
    ticker = request.args.get("ticker", "").strip().upper()
    range_key = request.args.get("range", "1d").strip().lower()
    if not ticker:
        return safe_json({"error": "ticker parameter required"}, 400)
    try:
        return safe_json(fetch_chart_history(ticker, range_key))
    except ValueError as e:
        return safe_json({"error": str(e)}, 400)
    except Exception as e:
        return safe_json({"error": f"History error: {e}"}, 500)


@app.route("/api/recent")
def api_recent():
    with CACHE_LOCK:
        items = list(RECENT_SEARCHES)
    return safe_json({"items": items, "count": len(items)})


@app.route("/api/watchlist")
def api_watchlist():
    raw = request.args.get("tickers", "").strip()
    tickers = [x.strip().upper() for x in raw.split(",") if x.strip()] if raw else DEFAULT_WATCHLIST
    tickers = tickers[:8]
    return safe_json({"items": get_watchlist_quotes(tickers), "updated_at": utc_now()})


@app.route("/health")
def health():
    return safe_json(
        {
            "status": "ok",
            "watchlist_size": len(DEFAULT_WATCHLIST),
            "recent_searches": len(RECENT_SEARCHES),
            "cache_buckets": {k: len(v) for k, v in CACHE.items()},
            "timestamp": utc_now(),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
