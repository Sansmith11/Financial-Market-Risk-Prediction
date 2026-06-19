"""
Financial Market Risk Pipeline — Flask Web App
Render deployment (Python 3.11) — v14 GROWW-STYLE UI + Current Price
Improvements: Groww-inspired modern interface, current price display, better UX
"""
import os, math, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, Response, render_template_string

warnings.filterwarnings("ignore")

try:
    from curl_cffi.requests.exceptions import HTTPError as CurlHTTPError
except ImportError:
    CurlHTTPError = OSError

from advanced_distributions.fractional_distribution import FractionalDistribution
from advanced_distributions.fractal_distribution   import FractalDistribution
from advanced_distributions.sinh_arcsinh           import SinhArcsinhDistribution
from advanced_distributions.slash_distribution     import SlashDistribution
from advanced_distributions.quantile_distribution  import QuantileDistribution
from advanced_distributions.davies_distribution    import DaviesDistribution

app = Flask(__name__)

# ── Safe float & JSON ─────────────────────────────────────────────────────────
def sf(x, n=6):
    try:
        v = float(x)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else round(v, n)
    except Exception:
        return 0.0

def _clean(obj):
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_clean(v) for v in obj]
    return obj

def safe_json(obj, status=200):
    return Response(json.dumps(_clean(obj), allow_nan=False),
                    status=status, mimetype="application/json")

# ── Ticker helpers ──────────────────────────────────────────────────────────
CORRECTIONS = {
    "APPL":"AAPL","AMZON":"AMZN","AMAZN":"AMZN","MICROSFT":"MSFT","MICROSFOT":"MSFT",
    "NETFLX":"NFLX","TESTA":"TSLA","RELINCE":"RELIANCE.NS","RELIACE":"RELIANCE.NS",
    "HDFCBANK":"HDFCBANK.NS","ICICIBANK":"ICICIBANK.NS","BAJFINANCE":"BAJFINANCE.NS",
    "SBIN":"SBIN.NS","WIPRO":"WIPRO.NS","INFY":"INFY.NS",
}
def resolve(ticker):
    u = ticker.upper().strip()
    if u in CORRECTIONS:
        c = CORRECTIONS[u]; return c, f"'{u}' auto-corrected to '{c}'"
    return u, None

# ── Data pipeline ───────────────────────────────────────────────────────────
def fetch(ticker, period="3y"):
    df = pd.DataFrame()
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception: pass
    if df is None or df.empty:
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        except Exception: pass
    if df is None or df.empty:
        t = ticker.upper()
        hint = f" Try '{t}.NS' for Indian stocks." if (not t.endswith(".NS") and t.isalpha() and len(t)<=6) else ""
        raise ValueError(f"No data found for '{ticker}'.{hint}")
    df.columns = ([str(c[0]).lower() for c in df.columns]
                  if isinstance(df.columns, pd.MultiIndex)
                  else [str(c).lower() for c in df.columns])
    for col in ["open","high","low","close","volume"]:
        if col not in df.columns: raise ValueError(f"Missing column '{col}'")
    return df

def get_current_price(ticker):
    """Fetch current stock price and basic info"""
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="1d")
        if data is None or data.empty:
            return None
        current_price = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-1]) if len(data) == 1 else float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
        price_change = current_price - prev_close
        pct_change = (price_change / prev_close * 100) if prev_close != 0 else 0
        return {
            "current": sf(current_price, 4),
            "change": sf(price_change, 4),
            "pct_change": sf(pct_change, 2),
            "direction": "up" if price_change >= 0 else "down"
        }
    except Exception:
        return None

def clean(df):
    df = df.copy().dropna()
    df = df[~df.index.duplicated(keep="first")]
    m = (df.high>=df.low)&(df.high>=df.close)&(df.low<=df.close)&(df.volume>0)
    df = df[m]
    lr = np.log(df.close/df.close.shift(1)).dropna()
    z  = (lr-lr.mean())/lr.std()
    return df.loc[z[np.abs(z)<=5].index].sort_index()

def features(df, w=20):
    df = df.copy()
    df["lr"]  = np.log(df.close/df.close.shift(1))
    df["vol"] = df.lr.rolling(w).std()*np.sqrt(252)
    rm, rs    = df.close.rolling(w).mean(), df.close.rolling(w).std()
    df["mom"] = (df.close-rm)/(rs+1e-10)
    d = df.close.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100-100/(1+g/(l+1e-10))
    df.dropna(inplace=True)
    return df

# ── Distribution models ───────────────────────────────────────────────────────
def run_fractional(r):
    pos = np.abs(r)+1e-8
    d   = FractionalDistribution.fit(pos)
    h   = float(np.clip(0.5+(d.alpha-1.5)*0.1, 0.3, 0.8))
    mem = float(np.clip((h-0.5)*2, 0, 1))
    reg = "trending" if h>0.55 else "mean-reverting" if h<0.45 else "random walk"
    bull_signal = float(np.clip((h-0.5)*4, 0, 1))
    bear_signal = float(np.clip((0.5-h)*4, 0, 1))
    return {"alpha":sf(d.alpha,4),"beta":sf(d.beta,4),"gamma":sf(d.gamma,4),
            "mean":sf(d.mean()),"std":sf(d.std()),
            "skewness":sf(d.skewness(),4),"kurtosis":sf(d.kurtosis(),4),
            "hurst_proxy":sf(h,4),"memory_score":sf(mem,4),"regime":reg,
            "_bull":bull_signal,"_bear":bear_signal}

def run_fractal(r):
    pos    = np.abs(r)+1e-8
    p      = FractalDistribution.fit(pos)
    D, lam = sf(p[0],4), sf(p[1],4)
    d      = FractalDistribution(D=D, lambda_=lam)
    comp   = float(np.clip(D-1.0, 0, 1))
    s      = "chaotic" if D>1.7 else "complex" if D>1.4 else "structured"
    bull_signal = 1.0 - comp
    bear_signal = comp
    return {"D":D,"lambda":lam,"mean":sf(d.mean()),"std":sf(d.std()),
            "complexity_score":sf(comp,4),"structure":s,
            "_bull":bull_signal,"_bear":bear_signal}

def run_sinh(r):
    d  = SinhArcsinhDistribution.fit(r)
    sk = "left-skewed" if d.epsilon<-0.1 else "right-skewed" if d.epsilon>0.1 else "symmetric"
    skew_strength = float(np.tanh(d.epsilon * 2))
    bull_signal = float(np.clip(skew_strength, 0, 1))
    bear_signal = float(np.clip(-skew_strength, 0, 1))
    tail_penalty = float(np.clip((d.delta - 1.0) * 0.3, 0, 0.3))
    return {"epsilon":sf(d.epsilon,4),"delta":sf(d.delta,4),
            "mu":sf(d.mu),"sigma":sf(d.sigma),
            "skewness":sf(d.skewness(),4),"kurtosis":sf(d.kurtosis(),4),
            "skewness_direction":sk,"skew_score":sf(np.tanh(d.epsilon),4),
            "_bull":max(0, bull_signal - tail_penalty),
            "_bear":max(0, bear_signal + tail_penalty*0.5)}

def run_slash(r):
    d   = SlashDistribution.fit(r)
    s   = d.rvs(size=5000, random_state=42)
    ep  = sf(np.mean(np.abs(s-d.mu)>3.0*d.sigma), 4)
    cr  = "high" if ep>0.05 else "medium" if ep>0.02 else "low"
    ent = sf(d.entropy(), 4)
    crash_bull = float(np.clip(1.0 - ep*15, 0, 1))
    crash_bear = float(np.clip(ep*15, 0, 1))
    ent_penalty = float(np.clip((ent - 2.0) * 0.1, 0, 0.3)) if ent > 2.0 else 0.0
    return {"mu":sf(d.mu),"sigma":sf(d.sigma),"extreme_event_prob":ep,
            "crash_risk":cr,"median":sf(d.median()),"entropy":ent,
            "_bull":max(0, crash_bull - ent_penalty),
            "_bear":min(1, crash_bear + ent_penalty)}

def run_spline(r):
    from scipy.stats import johnsonsu
    try:
        a, b, loc, scale = johnsonsu.fit(r)
        ql  = [0.05,0.10,0.25,0.50,0.75,0.90,0.95]
        qv  = johnsonsu.ppf(ql, a, b, loc, scale)
        q   = {f"q{int(p*100):02d}": sf(v) for p,v in zip(ql,qv)}
        mn  = sf(johnsonsu.mean(a,b,loc,scale))
        std = sf(johnsonsu.std(a,b,loc,scale))
        skw = sf(johnsonsu.stats(a,b,loc,scale,moments='s'),4)
        krt = sf(johnsonsu.stats(a,b,loc,scale,moments='k'),4)
        ent = sf(johnsonsu.entropy(a,b,loc,scale),4)
        unc = sf(np.clip(abs(q["q95"]-q["q05"])*50,0,1),4)
        upside   = float(q["q75"]) - float(q["q50"])
        downside = float(q["q50"]) - float(q["q25"])
        ratio    = upside/(abs(downside)+1e-8)
        bull_signal = float(np.clip(ratio*0.4, 0, 1))
        bear_signal = float(np.clip((1/ratio)*0.4, 0, 1)) if ratio>0 else 0.5
    except Exception:
        q   = {f"q{int(p*100):02d}": sf(np.percentile(r,p*100)) for p in [0.05,0.10,0.25,0.50,0.75,0.90,0.95]}
        mn  = sf(np.mean(r)); std = sf(np.std(r))
        skw = krt = ent = unc = 0.0
        bull_signal = bear_signal = 0.3
    return {**q,"mean":mn,"std":std,"skewness":skw,"kurtosis":krt,
            "entropy":ent,"uncertainty":unc,
            "_bull":bull_signal,"_bear":bear_signal}

def run_quantile(r):
    d   = QuantileDistribution.fit(r)
    v95 = sf(d.ppf(0.05)); v99 = sf(d.ppf(0.01))
    s   = d.rvs(size=10000, random_state=42)
    cv  = sf(np.mean(s[s<=v95])); tr = sf(np.clip(-cv*10,0,1),4)
    vbr = sf(np.mean(r<v95),4)
    bull_signal = float(np.clip(1.0 - tr - vbr*2, 0, 1))
    bear_signal = float(np.clip(tr + vbr*2, 0, 1))
    return {"mu":sf(d.mu),"sigma":sf(d.sigma),
            "alpha_shape":sf(d.alpha,4),"beta_shape":sf(d.beta,4),
            "VaR_95":v95,"VaR_99":v99,"CVaR_95":cv,
            "tail_risk_score":tr,"var_breach_rate":vbr,
            "skewness":sf(d.skewness(),4),"kurtosis":sf(d.kurtosis(),4),
            "_bull":bull_signal,"_bear":bear_signal}

def run_davies(r, w=20):
    pos = np.abs(r)+1e-8
    p   = DaviesDistribution.fit(pos)
    a,b,th,k = [sf(x,4) for x in p]
    d   = DaviesDistribution(alpha=a,beta=b,theta=th,k=k)
    rv  = float(np.mean(np.abs(r[-w:]))); ov = float(np.mean(pos))
    vr  = sf(rv/(ov+1e-10),4)
    try:   sc = float(np.clip(d.cdf(np.array([rv]))[0], 0, 1))
    except: sc = 0.5
    reg = "crisis" if sc>0.85 else "stress" if sc>0.65 else "caution" if sc>0.40 else "normal"
    stability = float(np.clip(1.0 - abs(vr - 1.0), 0, 1))
    bull_signal = float(np.clip((1.0-sc)*0.8 + stability*0.2, 0, 1))
    bear_signal = float(np.clip(sc*0.8 + (1-stability)*0.2, 0, 1))
    return {"alpha":a,"beta":b,"theta":th,"k":k,
            "mean":sf(d.mean()),"std":sf(d.std()),
            "stress_score":sf(sc,4),"vol_ratio":vr,"regime":reg,
            "_bull":bull_signal,"_bear":bear_signal}

# ── Decision engine ────────────────────────────────────────────────────────
def make_decision(frac, fractal, sinh, slash, spline, quant, davies):
    W = {
        "davies":   0.25,
        "slash":    0.20,
        "spline":   0.20,
        "quant":    0.18,
        "frac":     0.10,
        "fractal":  0.07,
    }
    models = {
        "davies":  davies,
        "slash":   slash,
        "spline":  spline,
        "quant":   quant,
        "frac":    frac,
        "fractal": fractal,
    }
    skew_boost = sinh["_bull"] - sinh["_bear"]

    bs  = sum(models[k]["_bull"] * W[k] for k in W)
    brs = sum(models[k]["_bear"] * W[k] for k in W)

    bs  = float(np.clip(bs  + skew_boost * 0.10, 0, 1))
    brs = float(np.clip(brs - skew_boost * 0.10, 0, 1))

    net = bs - brs

    raw_ratio = abs(net) / (bs + brs + 1e-6)
    confidence = 1.0 / (1.0 + math.exp(-8.0 * (raw_ratio - 0.25)))

    dec = "BUY" if net > 0.08 else "SELL" if net < -0.08 else "HOLD"

    breakdown = {}
    for k, m in models.items():
        breakdown[k] = {"bull": sf(m["_bull"],3), "bear": sf(m["_bear"],3),
                        "weight": W[k]}
    breakdown["sinh"] = {"bull": sf(sinh["_bull"],3), "bear": sf(sinh["_bear"],3),
                         "weight": 0.0, "note":"skew modifier"}

    return {"decision":dec,"bull_score":sf(bs,4),"bear_score":sf(brs,4),
            "net_score":sf(net,4),"confidence":sf(confidence,4),
            "breakdown":breakdown}

def pipeline(ticker):
    ticker, suggestion = resolve(ticker)
    df  = features(clean(fetch(ticker)))
    r   = df["lr"].values; px = df["close"].values
    current_price_data = get_current_price(ticker)
    frac    = run_fractional(r)
    fractal = run_fractal(r)
    sinh    = run_sinh(r)
    slash   = run_slash(r)
    spline  = run_spline(r)
    quant   = run_quantile(r)
    davies  = run_davies(r)
    dec     = make_decision(frac,fractal,sinh,slash,spline,quant,davies)
    def pub(d):
        return {k:v for k,v in d.items() if not k.startswith("_")}
    return _clean({
        "ticker":ticker,"suggestion":suggestion,"n_sessions":len(df),
        "mean_return":sf(r.mean()),"std_return":sf(r.std()),
        "current_price":current_price_data,
        "fractional":pub(frac),"fractal":pub(fractal),"sinh_arcsinh":pub(sinh),
        "slash":pub(slash),"neural_spline":pub(spline),"quantile":pub(quant),
        "davies":pub(davies),"decision":dec,
        "_prices":px[-120:].tolist(),"_returns":r[-300:].tolist()
    })

# ── Dashboard HTML (Groww-style) ────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Risk Predictor</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #ffffff;
  --bg-secondary: #f5f5f5;
  --card: #ffffff;
  --text: #1a1a1a;
  --text-secondary: #666666;
  --border: #e5e5e5;
  --accent: #7c3aed;
  --success: #10b981;
  --danger: #ef4444;
  --warning: #f59e0b;
  --muted: #999999;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f0f0f;
    --bg-secondary: #1a1a1a;
    --card: #1f1f1f;
    --text: #ffffff;
    --text-secondary: #b0b0b0;
    --border: #333333;
    --muted: #666666;
  }
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  line-height: 1.6;
}

/* HEADER */
.navbar {
  background: var(--card);
  border-bottom: 1px solid var(--border);
  padding: 1rem 1.5rem;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.navbar-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}

.logo {
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--accent);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.search-box {
  display: flex;
  align-items: center;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.6rem 1rem;
  flex: 1;
  max-width: 300px;
  gap: 0.5rem;
}

.search-box input {
  background: none;
  border: none;
  outline: none;
  color: var(--text);
  font-size: 0.95rem;
  width: 100%;
}

.search-box input::placeholder {
  color: var(--text-secondary);
}

.btn {
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 8px;
  padding: 0.65rem 1.5rem;
  font-weight: 600;
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.3s ease;
  white-space: nowrap;
}

.btn:hover {
  background: #6d28d9;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
}

.btn:active {
  transform: translateY(0);
}

/* MAIN CONTAINER */
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
}

/* ALERT */
.alert {
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin-bottom: 1.5rem;
  display: none;
  line-height: 1.5;
  animation: slideDown 0.3s ease;
}

@keyframes slideDown {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

.alert.error {
  background: #fee;
  border: 1px solid #fcc;
  color: #c33;
}

.alert.warning {
  background: #fef3c7;
  border: 1px solid #fcd34d;
  color: #92400e;
}

.alert-tip {
  font-size: 0.85rem;
  margin-top: 0.5rem;
  opacity: 0.9;
}

/* LOADING & STATUS */
.status {
  font-size: 0.9rem;
  color: var(--text-secondary);
  margin-bottom: 1.5rem;
  min-height: 1.2rem;
}

/* QUICK TICKER PILLS */
.quick-access {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin-bottom: 1.5rem;
  align-items: center;
}

.quick-label {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-secondary);
  margin-right: 0.5rem;
}

.ticker-pill {
  padding: 0.5rem 1rem;
  border: 1px solid var(--border);
  border-radius: 20px;
  background: var(--bg-secondary);
  color: var(--text);
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 500;
  transition: all 0.2s ease;
}

.ticker-pill:hover {
  border-color: var(--accent);
  background: var(--accent);
  color: white;
}

.ticker-pill.active {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
}

/* RESULT CONTAINER */
.results {
  display: none;
}

.results.show {
  display: block;
  animation: fadeIn 0.4s ease;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* HEADER CARD */
.header-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 1.5rem;
}

.ticker-info {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
}

.ticker-name {
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--text);
}

.ticker-price-section {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.current-price {
  font-size: 2.2rem;
  font-weight: 700;
  color: var(--accent);
}

.price-change {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  font-size: 1rem;
}

.price-change.up {
  color: var(--success);
}

.price-change.down {
  color: var(--danger);
}

.ticker-stats {
  font-size: 0.9rem;
  color: var(--text-secondary);
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.decision-badge {
  font-size: 1rem;
  font-weight: 700;
  padding: 0.8rem 1.5rem;
  border-radius: 8px;
  display: inline-block;
  text-align: center;
}

.decision-BUY {
  background: #d1fae5;
  color: #065f46;
  border: 1px solid #a7f3d0;
}

.decision-SELL {
  background: #fee;
  color: #7f1d1d;
  border: 1px solid #fca5a5;
}

.decision-HOLD {
  background: #fef3c7;
  color: #78350f;
  border: 1px solid #fcd34d;
}

/* SIGNAL CARDS */
.signal-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.signal-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem;
  text-align: center;
}

.signal-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary);
  margin-bottom: 0.5rem;
  font-weight: 600;
}

.signal-value {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
}

.confidence-bar {
  background: var(--bg-secondary);
  border-radius: 4px;
  height: 6px;
  margin-top: 0.5rem;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.8s ease;
}

/* CHARTS */
.chart-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem;
  margin-bottom: 1.5rem;
}

.chart-title {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary);
  font-weight: 700;
  margin-bottom: 1rem;
}

.chart-container {
  height: 300px;
}

.chart-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  margin-bottom: 1.5rem;
}

@media (max-width: 768px) {
  .chart-row {
    grid-template-columns: 1fr;
  }
}

/* MODEL CARDS */
.model-section {
  margin-bottom: 1.5rem;
}

.section-title {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--accent);
  font-weight: 700;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--accent);
}

.models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.model-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.model-card:hover {
  border-color: var(--accent);
  box-shadow: 0 4px 12px rgba(124, 58, 237, 0.1);
  transform: translateY(-2px);
}

.model-header {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  margin-bottom: 0.8rem;
}

.model-icon {
  font-size: 1.5rem;
}

.model-name {
  font-weight: 600;
  font-size: 0.95rem;
}

.model-tag {
  font-size: 0.75rem;
  color: var(--text-secondary);
  margin-top: 0.2rem;
}

.model-signals {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 0.8rem;
}

.signal-pill {
  flex: 1;
  padding: 0.4rem 0.6rem;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 600;
  text-align: center;
}

.pill-bull {
  background: #d1fae5;
  color: #065f46;
  border: 1px solid #a7f3d0;
}

.pill-bear {
  background: #fee;
  color: #7f1d1d;
  border: 1px solid #fca5a5;
}

.model-details {
  display: none;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  font-size: 0.8rem;
}

.model-details.show {
  display: block;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  padding: 0.3rem 0;
  color: var(--text-secondary);
}

.detail-value {
  color: var(--text);
  font-weight: 600;
}

/* BREAKDOWN */
.breakdown-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.breakdown-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}

.breakdown-row:last-child {
  margin-bottom: 0;
  padding-bottom: 0;
  border-bottom: none;
}

.breakdown-name {
  min-width: 100px;
  font-weight: 600;
  font-size: 0.9rem;
}

.breakdown-bars {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.bar {
  height: 6px;
  background: var(--bg-secondary);
  border-radius: 3px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.6s ease;
}

.bar-bull { background: var(--success); }
.bar-bear { background: var(--danger); }

.breakdown-values {
  display: flex;
  gap: 0.5rem;
  min-width: 80px;
  justify-content: flex-end;
  font-size: 0.85rem;
  font-weight: 600;
}

.val-bull { color: var(--success); }
.val-bear { color: var(--danger); }

.weight {
  min-width: 45px;
  text-align: right;
  font-size: 0.8rem;
  color: var(--text-secondary);
}

/* HIDDEN */
.hidden { display: none; }

/* RESPONSIVE */
@media (max-width: 640px) {
  .container {
    padding: 1rem;
  }
  
  .navbar-content {
    flex-direction: column;
    gap: 0.8rem;
  }
  
  .search-box {
    max-width: 100%;
  }
  
  .header-card {
    flex-direction: column;
    gap: 1rem;
  }
  
  .signal-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .models-grid {
    grid-template-columns: 1fr;
  }
  
  .ticker-info {
    flex-direction: column;
    width: 100%;
  }
}
</style>
</head>
<body>

<!-- NAVBAR -->
<div class="navbar">
  <div class="navbar-content">
    <div class="logo">Financial Market Risk Prediction</div>
    <div class="search-box">
      <svg width="16" height="16" fill="none" stroke="var(--text-secondary)" stroke-width="2" viewBox="0 0 24 24">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
      </svg>
      <input id="ticker-input" placeholder="AAPL, RELIANCE.NS, MSFT..." value="RELIANCE.NS">
    </div>
    <button class="btn" onclick="analyze()">Analyze</button>
  </div>
</div>

<!-- MAIN -->
<div class="container">
  <!-- Quick Access -->
  <div class="quick-access">
    <span class="quick-label">Popular:</span>
    <span class="ticker-pill" onclick="go('RELIANCE.NS')">RELIANCE.NS</span>
    <span class="ticker-pill" onclick="go('TCS.NS')">TCS.NS</span>
    <span class="ticker-pill" onclick="go('INFY.NS')">INFY.NS</span>
    <span class="ticker-pill" onclick="go('AAPL')">AAPL</span>
    <span class="ticker-pill" onclick="go('MSFT')">MSFT</span>
    <span class="ticker-pill" onclick="go('TSLA')">TSLA</span>
  </div>

  <!-- Alerts -->
  <div id="error-alert" class="alert error">
    <span id="error-msg"></span>
    <div id="error-tip" class="alert-tip" style="display:none;"></div>
  </div>
  <div id="warning-alert" class="alert warning"></div>

  <!-- Status -->
  <div class="status" id="status"></div>

  <!-- Results -->
  <div class="results" id="results">
    <!-- Header with ticker & decision -->
    <div class="header-card">
      <div class="ticker-info">
        <div>
          <div class="ticker-name" id="ticker-display">—</div>
          <div class="ticker-stats">
            <span id="sessions-stat">—</span>
            <span id="return-stat">—</span>
          </div>
        </div>
        <div class="ticker-price-section">
          <div class="current-price" id="current-price">—</div>
          <div class="price-change" id="price-change">—</div>
        </div>
      </div>
      <div id="decision-badge" class="decision-badge">—</div>
    </div>

    <!-- Signal Cards -->
    <div class="signal-grid">
      <div class="signal-card">
        <div class="signal-label">Bull Score</div>
        <div class="signal-value" style="color: var(--success);" id="bull-score">—</div>
      </div>
      <div class="signal-card">
        <div class="signal-label">Bear Score</div>
        <div class="signal-value" style="color: var(--danger);" id="bear-score">—</div>
      </div>
      <div class="signal-card">
        <div class="signal-label">Confidence</div>
        <div class="signal-value" id="confidence-value">—</div>
        <div class="confidence-bar">
          <div class="confidence-fill" id="confidence-bar"></div>
        </div>
      </div>
      <div class="signal-card">
        <div class="signal-label">Net Score</div>
        <div class="signal-value" id="net-score">—</div>
      </div>
    </div>

    <!-- Charts -->
    <div class="chart-row">
      <div class="chart-card">
        <div class="chart-title" id="price-title">Price Chart</div>
        <div class="chart-container" id="price-chart"></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Return Distribution</div>
        <div class="chart-container" id="return-chart"></div>
      </div>
    </div>

    <!-- Stress Gauge -->
    <div class="chart-card">
      <div class="chart-title">Market Stress Indicator</div>
      <div class="chart-container" id="stress-chart"></div>
    </div>

    <!-- Breakdown -->
    <div class="breakdown-card">
      <h3 class="section-title">Signal Breakdown</h3>
      <div id="breakdown"></div>
    </div>

    <!-- Model Cards -->
    <div class="model-section">
      <h3 class="section-title">Model Details</h3>
      <div class="models-grid" id="models"></div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const P = {responsive: true, displayModeBar: false};
const L = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: {color: 'var(--text-secondary)', size: 11},
  xaxis: {gridcolor: 'var(--border)', zerolinecolor: 'var(--border)'},
  yaxis: {gridcolor: 'var(--border)', zerolinecolor: 'var(--border)'},
  margin: {t: 20, r: 20, b: 30, l: 40},
};

function go(ticker) {
  document.querySelectorAll('.ticker-pill').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  $('ticker-input').value = ticker;
  analyze();
}

function showError(msg, tip = '') {
  $('error-alert').style.display = 'block';
  $('error-msg').textContent = msg;
  $('error-tip').textContent = tip;
  $('error-tip').style.display = tip ? 'block' : 'none';
}

function hideError() {
  $('error-alert').style.display = 'none';
}

async function analyze() {
  const ticker = $('ticker-input').value.trim().toUpperCase();
  if (!ticker) return;

  hideError();
  $('warning-alert').style.display = 'none';
  $('results').classList.remove('show');
  $('status').textContent = '⏳ Analyzing ' + ticker + '...';

  try {
    const res = await fetch('/api/analyze?ticker=' + encodeURIComponent(ticker));
    const contentType = res.headers.get('content-type') || '';
    
    if (!contentType.includes('application/json')) {
      $('status').textContent = '';
      showError('⚠️ Server is starting up', 'Please wait 30 seconds and try again.');
      return;
    }

    const data = await res.json();
    
    if (!res.ok) {
      $('status').textContent = '';
      let tip = '';
      if (data.error && data.error.includes('No data')) {
        if (!ticker.endsWith('.NS') && /^[A-Z]+$/.test(ticker)) {
          tip = 'Try ' + ticker + '.NS for Indian stocks';
        }
      }
      showError('❌ ' + (data.error || 'Error'), tip);
      return;
    }

    // Show warning if ticker was corrected
    if (data.suggestion) {
      $('warning-alert').textContent = 'ℹ️ ' + data.suggestion;
      $('warning-alert').style.display = 'block';
      $('ticker-input').value = data.ticker;
    }

    $('status').textContent = '✅ Loaded ' + data.ticker + ' · ' + data.n_sessions + ' sessions';
    renderResults(data);
    $('results').classList.add('show');

  } catch (e) {
    $('status').textContent = '';
    showError('❌ ' + e.message, 'Check your connection and try again.');
  }
}

function renderResults(data) {
  const dec = data.decision;
  
  // Header
  $('ticker-display').textContent = data.ticker;
  $('sessions-stat').textContent = data.n_sessions + ' trading sessions';
  $('return-stat').textContent = 'Avg return: ' + (data.mean_return * 100).toFixed(3) + '%';
  
  // Current Price
  if (data.current_price) {
    const cp = data.current_price;
    $('current-price').textContent = '₹' + cp.current.toFixed(2);
    const changeClass = cp.direction === 'up' ? 'up' : 'down';
    const changeSymbol = cp.direction === 'up' ? '▲' : '▼';
    $('price-change').className = 'price-change ' + changeClass;
    $('price-change').textContent = changeSymbol + ' ' + Math.abs(cp.change).toFixed(2) + ' (' + cp.pct_change + '%)';
  }
  
  // Decision
  const badgeClass = 'decision-' + dec.decision;
  $('decision-badge').className = 'decision-badge ' + badgeClass;
  $('decision-badge').textContent = dec.decision + ' (Confidence: ' + Math.round(dec.confidence * 100) + '%)';
  
  // Scores
  $('bull-score').textContent = dec.bull_score.toFixed(3);
  $('bear-score').textContent = dec.bear_score.toFixed(3);
  const confPct = Math.round(dec.confidence * 100);
  $('confidence-value').textContent = confPct + '%';
  const confColor = confPct >= 70 ? 'var(--success)' : confPct >= 45 ? 'var(--warning)' : 'var(--danger)';
  $('confidence-bar').style.cssText = 'width: ' + confPct + '%; background: ' + confColor + ';';
  $('net-score').textContent = dec.net_score.toFixed(3);
  
  // Price chart
  const n = data._prices.length;
  Plotly.newPlot('price-chart', [{
    x: Array.from({length: n}, (_, i) => i - n + 1),
    y: data._prices,
    type: 'scatter',
    mode: 'lines',
    line: {color: 'var(--accent)', width: 2},
    fill: 'tozeroy',
    fillcolor: 'rgba(124, 58, 237, 0.1)',
  }], {...L}, P);
  
  $('price-title').textContent = data.ticker + ' — Price (Last 120 Sessions)';
  
  // Return histogram
  Plotly.newPlot('return-chart', [{
    x: data._returns,
    type: 'histogram',
    nbinsx: 50,
    marker: {color: 'var(--accent)', opacity: 0.7},
  }], {...L}, P);
  
  // Stress gauge
  const sc = data.davies.stress_score;
  const gaugeColor = sc > 0.65 ? '#ef4444' : sc > 0.4 ? '#f59e0b' : '#10b981';
  Plotly.newPlot('stress-chart', [{
    type: 'indicator',
    mode: 'gauge+number',
    value: Math.round(sc * 100),
    gauge: {
      axis: {range: [0, 100], tickcolor: 'var(--text-secondary)'},
      bar: {color: gaugeColor, thickness: 0.25},
      bgcolor: 'var(--bg-secondary)',
      bordercolor: 'var(--border)',
      steps: [
        {range: [0, 40], color: 'rgba(16, 185, 129, 0.2)'},
        {range: [40, 65], color: 'rgba(245, 158, 11, 0.2)'},
        {range: [65, 100], color: 'rgba(239, 68, 68, 0.2)'},
      ],
    },
    number: {suffix: '%', font: {color: gaugeColor, size: 24}},
    title: {text: 'Regime: ' + data.davies.regime, font: {color: gaugeColor, size: 12}},
  }], {...L, margin: {t: 30, r: 20, b: 20, l: 20}}, P);
  
  // Breakdown
  renderBreakdown(dec);
  
  // Models
  renderModels(data, dec);
}

function renderBreakdown(dec) {
  const order = ['davies', 'slash', 'spline', 'quant', 'frac', 'fractal'];
  const names = {
    davies: 'Davies',
    slash: 'Slash',
    spline: 'JohnsonSU',
    quant: 'Quantile',
    frac: 'Fractional',
    fractal: 'Fractal',
  };
  
  const html = order.map(k => {
    const b = dec.breakdown[k] || {};
    const bullVal = (b.bull || 0).toFixed(2);
    const bearVal = (b.bear || 0).toFixed(2);
    const weight = Math.round((b.weight || 0) * 100);
    
    return `
      <div class="breakdown-row">
        <div class="breakdown-name">${names[k]}</div>
        <div class="breakdown-bars">
          <div class="bar"><div class="bar-fill bar-bull" style="width: ${bullVal * 100}%"></div></div>
          <div class="bar"><div class="bar-fill bar-bear" style="width: ${bearVal * 100}%"></div></div>
        </div>
        <div class="breakdown-values">
          <span class="val-bull">▲ ${bullVal}</span>
          <span class="val-bear">▼ ${bearVal}</span>
        </div>
        <div class="weight">${weight}%</div>
      </div>
    `;
  }).join('');
  
  $('breakdown').innerHTML = html;
}

const MODEL_META = {
  frac: {icon: '〜', name: 'Fractional', tag: 'Hurst proxy & memory'},
  fractal: {icon: '❄', name: 'Fractal', tag: 'Market structure'},
  sinh: {icon: '⟛', name: 'Sinh-Arcsinh', tag: 'Skewness & tails'},
  slash: {icon: '⚡', name: 'Slash', tag: 'Crash risk'},
  spline: {icon: '📈', name: 'JohnsonSU', tag: 'Quantile forecast'},
  quant: {icon: '📉', name: 'Quantile', tag: 'VaR & tail risk'},
};

function renderModels(data, dec) {
  const models = [
    {key: 'frac', data: data.fractional},
    {key: 'fractal', data: data.fractal},
    {key: 'sinh', data: data.sinh_arcsinh},
    {key: 'slash', data: data.slash},
    {key: 'spline', data: data.neural_spline},
    {key: 'quant', data: data.quantile},
    {key: 'davies', data: data.davies},
  ];
  
  const html = models.map(({key, data: mdata}) => {
    const m = MODEL_META[key];
    const b = dec.breakdown[key] || {};
    const bullVal = (b.bull || 0).toFixed(2);
    const bearVal = (b.bear || 0).toFixed(2);
    
    const details = Object.entries(mdata)
      .map(([k, v]) => `<div class="detail-item"><span>${k}</span><span class="detail-value">${typeof v === 'number' ? v.toFixed(4) : v}</span></div>`)
      .join('');
    
    return `
      <div class="model-card" onclick="toggleModel(event)">
        <div class="model-header">
          <div class="model-icon">${m.icon}</div>
          <div>
            <div class="model-name">${m.name}</div>
            <div class="model-tag">${m.tag}</div>
          </div>
        </div>
        <div class="model-signals">
          <div class="signal-pill pill-bull">▲ ${bullVal}</div>
          <div class="signal-pill pill-bear">▼ ${bearVal}</div>
        </div>
        <div class="model-details">${details}</div>
      </div>
    `;
  }).join('');
  
  $('models').innerHTML = html;
}

function toggleModel(event) {
  const card = event.currentTarget;
  const details = card.querySelector('.model-details');
  details.classList.toggle('show');
}

$('ticker-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') analyze();
});
</script>

</body>
</html>"""

# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/analyze")
def api_analyze():
    ticker = request.args.get("ticker","").strip().upper()
    if not ticker:
        return safe_json({"error":"ticker parameter required"}, 400)
    try:
        return safe_json(pipeline(ticker))
    except ValueError as e:
        return safe_json({"error":str(e)}, 404)
    except CurlHTTPError as e:
        return safe_json({"error":f"Data fetch failed: {e}"}, 503)
    except Exception as e:
        return safe_json({"error":f"Pipeline error: {e}"}, 500)

@app.route("/health")
def health():
    return safe_json({"status":"ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
