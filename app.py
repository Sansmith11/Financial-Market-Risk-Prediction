"""
Financial Market Risk Pipeline — Flask Web App
Render deployment (Python 3.11) — v13 GROWW UI
Improvements: Groww-inspired mobile-first design with bottom navigation
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

# ── Safe float & JSON ────────────────────────────────────────────────────────
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

# ── Ticker helpers ─────────────────────────────────────────────────────────
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

# ── Data pipeline ─────────────────────────────────────────────────────────
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
        "fractional":pub(frac),"fractal":pub(fractal),"sinh_arcsinh":pub(sinh),
        "slash":pub(slash),"neural_spline":pub(spline),"quantile":pub(quant),
        "davies":pub(davies),"decision":dec,
        "_prices":px[-120:].tolist(),"_returns":r[-300:].tolist()
    })

# ── Dashboard HTML (Groww-Style UI) ────��───────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>StockInsight - Risk Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--primary:#00D084;--primary-dark:#00A366;--bg:#FFFFFF;--surface:#F8F9FA;
      --card:#FFFFFF;--border:#E8EAED;--text:#1A1A1A;--text-secondary:#666666;
      --danger:#FF4444;--warning:#FFA500;--success:#00D084;--info:#2563eb}
body{background:var(--surface);color:var(--text);font-family:'Inter','Segoe UI',sans-serif;
     min-height:100vh;padding-bottom:80px}

/* HEADER */
.header{background:var(--bg);border-bottom:1px solid var(--border);
        padding:1rem 1rem;position:sticky;top:0;z-index:100;box-shadow:0 1px 4px rgba(0,0,0,0.05)}
.header-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem}
.header-title{font-size:1.2rem;font-weight:700}
.header-subtitle{font-size:.75rem;color:var(--text-secondary)}
.search-box{display:flex;align-items:center;gap:.5rem;background:var(--surface);
           border:1px solid var(--border);border-radius:20px;padding:.5rem 1rem;flex:1}
.search-box input{background:none;border:none;outline:none;color:var(--text);
                  font-size:.9rem;width:100%;flex:1}
.search-box input::placeholder{color:var(--text-secondary)}
.btn-search{background:var(--primary);color:#fff;border:none;border-radius:20px;
           padding:.5rem 1.2rem;cursor:pointer;font-weight:600;font-size:.85rem;transition:.2s}
.btn-search:active{background:var(--primary-dark)}

/* MAIN CONTENT */
.main{max-width:600px;margin:0 auto;padding:1rem}

/* QUICK TICKERS CAROUSEL */
.ticker-carousel{display:flex;gap:.7rem;margin-bottom:1.5rem;overflow-x:auto;
                padding-bottom:.5rem;-webkit-overflow-scrolling:touch}
.ticker-carousel::-webkit-scrollbar{height:3px}
.ticker-carousel::-webkit-scrollbar-track{background:var(--surface)}
.ticker-carousel::-webkit-scrollbar-thumb{background:var(--border)}
.ticker-chip{padding:.6rem 1rem;background:var(--card);border:1px solid var(--border);
            border-radius:20px;cursor:pointer;white-space:nowrap;transition:.2s;
            font-size:.85rem;font-weight:500;flex-shrink:0}
.ticker-chip:active{background:var(--primary);color:#fff;border-color:var(--primary)}

/* ALERTS */
.alert{border-radius:10px;padding:.8rem 1rem;margin-bottom:1rem;font-size:.9rem;display:none}
.alert-error{background:#FFE8E8;border:1px solid #FF4444;color:#990000}
.alert-warning{background:#FFF3E0;border:1px solid #FFA500;color:#994400}
.alert-info{background:#E8F5E9;border:1px solid #00D084;color:#004400}

/* RISK SCORE CARD (Main Hero) */
.hero-card{background:linear-gradient(135deg, var(--primary) 0%, #00A366 100%);
          color:#fff;border-radius:16px;padding:1.5rem;margin-bottom:1.5rem;
          box-shadow:0 4px 12px rgba(0,208,132,0.25)}
.risk-score{font-size:3rem;font-weight:700;line-height:1}
.risk-label{font-size:.85rem;opacity:.9;margin-top:.5rem}
.risk-status{display:flex;gap:1rem;margin-top:1.2rem;font-size:.9rem}
.status-item{display:flex;align-items:center;gap:.4rem}
.status-badge{width:8px;height:8px;border-radius:50%;background:#fff}

/* SIGNAL CARDS GRID */
.signals-grid{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-bottom:1.5rem}
.signal-card{background:var(--card);border:1px solid var(--border);border-radius:12px;
            padding:1rem;text-align:center;transition:.2s}
.signal-card.active{border-color:var(--primary);background:#F0FAF7}
.signal-label{font-size:.75rem;color:var(--text-secondary);text-transform:uppercase;
             letter-spacing:.05em;margin-bottom:.5rem}
.signal-value{font-size:1.8rem;font-weight:700;margin:.3rem 0}
.signal-indicator{font-size:.85rem;font-weight:600}
.signal-indicator.buy{color:var(--success)}
.signal-indicator.sell{color:var(--danger)}
.signal-indicator.hold{color:var(--warning)}

/* CHARTS SECTION */
.chart-section{background:var(--card);border:1px solid var(--border);border-radius:12px;
              padding:1rem;margin-bottom:1.5rem}
.chart-title{font-size:.8rem;text-transform:uppercase;color:var(--text-secondary);
            margin-bottom:1rem;font-weight:600;letter-spacing:.05em}
.chart-container{height:200px;border-radius:8px;background:var(--surface)}

/* MODEL CARDS */
.models-section{margin-bottom:1.5rem}
.models-title{font-size:.8rem;text-transform:uppercase;color:var(--text-secondary);
             margin-bottom:1rem;font-weight:600;letter-spacing:.05em}
.model-card{background:var(--card);border:1px solid var(--border);border-radius:12px;
           margin-bottom:.8rem;overflow:hidden;transition:.2s}
.model-header{padding:.9rem 1rem;cursor:pointer;display:flex;justify-content:space-between;
             align-items:center;user-select:none}
.model-header:active{background:var(--surface)}
.model-name{font-size:.9rem;font-weight:600}
.model-subtitle{font-size:.75rem;color:var(--text-secondary);margin-top:.2rem}
.model-signals{display:flex;gap:.4rem}
.badge{font-size:.7rem;padding:.25rem .6rem;border-radius:12px;font-weight:600;
      display:inline-flex;align-items:center;gap:.3rem}
.badge-bull{background:#E8F5E9;color:#00A366}
.badge-bear{background:#FFE8E8;color:#DD0000}
.chevron{transition:transform .3s;color:var(--text-secondary)}
.model-card.open .chevron{transform:rotate(180deg)}
.model-body{display:none;border-top:1px solid var(--border);padding:1rem;
           background:var(--surface);font-size:.85rem}
.model-card.open .model-body{display:block}
.metrics-grid{display:grid;grid-template-columns:1fr 1fr;gap:.8rem}
.metric{text-align:center}
.metric-label{font-size:.7rem;color:var(--text-secondary);margin-bottom:.3rem}
.metric-value{font-size:1rem;font-weight:700}

/* RISK BREAKDOWN */
.breakdown-section{background:var(--card);border:1px solid var(--border);border-radius:12px;
                  padding:1rem;margin-bottom:1.5rem}
.breakdown-title{font-size:.8rem;text-transform:uppercase;color:var(--text-secondary);
                margin-bottom:1rem;font-weight:600}
.breakdown-item{display:flex;align-items:center;margin-bottom:.8rem;gap:.8rem}
.breakdown-item:last-child{margin-bottom:0}
.breakdown-label{font-size:.9rem;width:80px;flex-shrink:0}
.breakdown-bar{flex:1;background:var(--surface);height:8px;border-radius:4px;
              overflow:hidden}
.breakdown-fill{height:100%;border-radius:4px;transition:width .4s ease}
.breakdown-fill-bull{background:var(--success)}
.breakdown-fill-bear{background:var(--danger)}
.breakdown-value{font-size:.85rem;font-weight:600;width:50px;text-align:right}

/* BOTTOM NAVIGATION */
.bottom-nav{position:fixed;bottom:0;left:0;right:0;background:var(--card);
           border-top:1px solid var(--border);display:flex;justify-content:space-around;
           align-items:center;height:70px;z-index:1000;box-shadow:0 -2px 8px rgba(0,0,0,0.05)}
.nav-item{display:flex;flex-direction:column;align-items:center;gap:.3rem;cursor:pointer;
         color:var(--text-secondary);padding:.5rem;transition:.2s;font-size:.75rem}
.nav-item.active{color:var(--primary)}
.nav-icon{font-size:1.3rem}

/* LOADING STATE */
.loading{display:none;text-align:center;padding:2rem;color:var(--text-secondary)}
.spinner{display:inline-block;width:30px;height:30px;border:3px solid var(--border);
        border-top-color:var(--primary);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

.hidden{display:none}
</style>
</head>
<body>
<div class="header">
  <div class="header-top">
    <div>
      <div class="header-title">📈 StockInsight</div>
      <div class="header-subtitle">AI-Powered Risk Analysis</div>
    </div>
  </div>
  <div class="search-box">
    <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
      <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
    </svg>
    <input id="ti" placeholder="Search ticker…" value="RELIANCE.NS"/>
    <button class="btn-search" onclick="analyze()">Go</button>
  </div>
</div>

<div class="main">
  <!-- Quick tickers carousel -->
  <div class="ticker-carousel">
    <span class="ticker-chip" onclick="go('RELIANCE.NS')">RELIANCE</span>
    <span class="ticker-chip" onclick="go('TCS.NS')">TCS</span>
    <span class="ticker-chip" onclick="go('INFY.NS')">INFY</span>
    <span class="ticker-chip" onclick="go('HDFCBANK.NS')">HDFCBANK</span>
    <span class="ticker-chip" onclick="go('AAPL')">AAPL</span>
    <span class="ticker-chip" onclick="go('MSFT')">MSFT</span>
    <span class="ticker-chip" onclick="go('TSLA')">TSLA</span>
  </div>

  <!-- Alerts -->
  <div id="alert-error" class="alert alert-error"></div>
  <div id="alert-warning" class="alert alert-warning"></div>
  <div id="alert-info" class="alert alert-info"></div>

  <!-- Loading state -->
  <div id="loading" class="loading">
    <div class="spinner"></div>
    <p style="margin-top:1rem">Analyzing market data...</p>
  </div>

  <!-- Results section -->
  <div id="results" class="hidden">
    <!-- Hero Card -->
    <div class="hero-card" id="hero">
      <div style="font-size:.9rem;opacity:.9;margin-bottom:.5rem" id="ticker-display">—</div>
      <div class="risk-score" id="signal-display">—</div>
      <div class="risk-label">Overall Signal</div>
      <div class="risk-status">
        <div class="status-item">
          <span class="status-badge"></span>
          <span>Bull: <strong id="bull-score">—</strong></span>
        </div>
        <div class="status-item">
          <span class="status-badge"></span>
          <span>Bear: <strong id="bear-score">—</strong></span>
        </div>
      </div>
    </div>

    <!-- Signal Cards -->
    <div class="signals-grid">
      <div class="signal-card">
        <div class="signal-label">Confidence</div>
        <div class="signal-value" id="confidence">—</div>
        <div class="signal-indicator">Reliability</div>
      </div>
      <div class="signal-card">
        <div class="signal-label">Net Score</div>
        <div class="signal-value" id="net-score">—</div>
        <div class="signal-indicator">Momentum</div>
      </div>
    </div>

    <!-- Price Chart -->
    <div class="chart-section">
      <div class="chart-title">Price Trend (120 Days)</div>
      <div id="chart-price" class="chart-container"></div>
    </div>

    <!-- Analysis Charts -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-bottom:1.5rem">
      <div class="chart-section" style="margin-bottom:0">
        <div class="chart-title">Returns</div>
        <div id="chart-returns" style="height:150px"></div>
      </div>
      <div class="chart-section" style="margin-bottom:0">
        <div class="chart-title">Stress Level</div>
        <div id="chart-stress" style="height:150px"></div>
      </div>
    </div>

    <!-- Risk Breakdown -->
    <div class="breakdown-section">
      <div class="breakdown-title">Model Consensus</div>
      <div id="breakdown"></div>
    </div>

    <!-- Model Details -->
    <div class="models-section">
      <div class="models-title">Advanced Metrics</div>
      <div id="models"></div>
    </div>
  </div>
</div>

<!-- Bottom Navigation -->
<div class="bottom-nav">
  <div class="nav-item active">
    <div class="nav-icon">📊</div>
    <span>Analyze</span>
  </div>
  <div class="nav-item" onclick="showWatchlist()">
    <div class="nav-icon">⭐</div>
    <span>Watchlist</span>
  </div>
  <div class="nav-item" onclick="showPortfolio()">
    <div class="nav-icon">💼</div>
    <span>Portfolio</span>
  </div>
  <div class="nav-item" onclick="showSettings()">
    <div class="nav-icon">⚙️</div>
    <span>Settings</span>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);

const CHART_LAYOUT={
  paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:'#666666',size:10,family:'Inter, sans-serif'},
  xaxis:{gridcolor:'#E8EAED',zerolinecolor:'#E8EAED'},
  yaxis:{gridcolor:'#E8EAED',zerolinecolor:'#E8EAED'},
  margin:{t:5,r:5,b:25,l:40}
};
const CHART_CONFIG={responsive:true,displayModeBar:false};

function go(t){
  $('ti').value=t;analyze();
}

function showAlert(type,msg){
  const el=$('alert-'+type);
  el.textContent=msg;
  el.style.display='block';
  setTimeout(()=>{el.style.display='none'},5000);
}

async function analyze(){
  const t=$('ti').value.trim().toUpperCase();
  if(!t)return;
  
  $('alert-error').style.display='none';
  $('alert-warning').style.display='none';
  $('results').classList.add('hidden');
  $('loading').style.display='block';
  
  try{
    const res=await fetch('/api/analyze?ticker='+encodeURIComponent(t));
    const d=await res.json();
    
    if(!res.ok){
      $('loading').style.display='none';
      const tip=d.error&&d.error.includes('No data')?(!t.endsWith('.NS')?'Try '+t+'.NS for Indian stocks':'Check ticker at finance.yahoo.com'):'';
      showAlert('error',d.error||'Error analyzing ticker');
      if(tip)showAlert('warning',tip);
      return;
    }
    
    if(d.suggestion){
      showAlert('info',d.suggestion);
      $('ti').value=d.ticker;
    }
    
    // Populate hero card
    $('ticker-display').textContent=d.ticker;
    const dec=d.decision;
    $('signal-display').textContent=dec.decision;
    $('bull-score').textContent=dec.bull_score;
    $('bear-score').textContent=dec.bear_score;
    
    // Populate signals
    $('confidence').textContent=Math.round(dec.confidence*100)+'%';
    $('net-score').textContent=dec.net_score;
    
    // Charts
    const n=d._prices.length;
    Plotly.newPlot('chart-price',[{
      x:[...Array(n).keys()].map(i=>i-n+1),
      y:d._prices,
      type:'scatter',mode:'lines',
      line:{color:'#00D084',width:2},
      fill:'tozeroy',fillcolor:'rgba(0,208,132,0.1)'
    }],CHART_LAYOUT,CHART_CONFIG);
    
    Plotly.newPlot('chart-returns',[{
      x:d._returns,type:'histogram',nbinsx:40,
      marker:{color:'#2563eb',opacity:.7}
    }],CHART_LAYOUT,CHART_CONFIG);
    
    const sc=d.davies.stress_score;
    const gc=sc>0.65?'#FF4444':sc>0.4?'#FFA500':'#00D084';
    Plotly.newPlot('chart-stress',[{
      type:'indicator',mode:'gauge+number',value:Math.round(sc*100),
      gauge:{
        axis:{range:[0,100]},
        bar:{color:gc},
        bgcolor:'#F0FAF7',
        steps:[{range:[0,40],color:'#E8F5E9'},{range:[40,65],color:'#FFF3E0'},{range:[65,100],color:'#FFE8E8'}]
      },
      number:{suffix:'%',font:{color:gc,size:18}},
      title:{text:'Risk Level',font:{color:'#1A1A1A',size:10}}
    }],{...CHART_LAYOUT,margin:{t:20,r:15,b:10,l:15}},CHART_CONFIG);
    
    // Breakdown
    const bd=dec.breakdown;
    const order=['davies','slash','spline','quant','frac','fractal'];
    const names={davies:'Davies',slash:'Slash',spline:'Spline',quant:'Quantile',frac:'Fractional',fractal:'Fractal'};
    $('breakdown').innerHTML=order.map(k=>{
      const b=bd[k]||{};const bv=+(b.bull||0).toFixed(2);const brv=+(b.bear||0).toFixed(2);
      return `<div class="breakdown-item">
        <div class="breakdown-label">${names[k]}</div>
        <div style="flex:1;display:flex;gap:2px">
          <div class="breakdown-bar" style="flex:${bv}">
            <div class="breakdown-fill breakdown-fill-bull" style="width:100%"></div>
          </div>
          <div class="breakdown-bar" style="flex:${brv}">
            <div class="breakdown-fill breakdown-fill-bear" style="width:100%"></div>
          </div>
        </div>
        <div class="breakdown-value">${bv.toFixed(2)}/${brv.toFixed(2)}</div>
      </div>`;
    }).join('');
    
    // Models
    const models_data=[
      {key:'frac',data:d.fractional,name:'Fractional',icon:'〜'},
      {key:'fractal',data:d.fractal,name:'Fractal',icon:'❄'},
      {key:'sinh',data:d.sinh_arcsinh,name:'Sinh-Arcsinh',icon:'⟛'},
      {key:'slash',data:d.slash,name:'Slash',icon:'⚡'},
      {key:'spline',data:d.neural_spline,name:'Spline',icon:'📈'},
      {key:'quant',data:d.quantile,name:'Quantile',icon:'📉'},
    ];
    $('models').innerHTML=models_data.map(({key,data,name,icon})=>{
      const b=bd[key]||{};
      return `<div class="model-card" id="model-${key}">
        <div class="model-header" onclick="toggle('${key}')">
          <div>
            <div class="model-name">${icon} ${name}</div>
            <div class="model-subtitle">${Object.keys(data).slice(0,2).join(', ')}</div>
          </div>
          <div class="model-signals">
            <span class="badge badge-bull">▲${(b.bull||0).toFixed(2)}</span>
            <span class="badge badge-bear">▼${(b.bear||0).toFixed(2)}</span>
            <span class="chevron">▼</span>
          </div>
        </div>
        <div class="model-body">
          <div class="metrics-grid">
            ${Object.entries(data).map(([k,v])=>`
            <div class="metric">
              <div class="metric-label">${k}</div>
              <div class="metric-value">${typeof v==='number'?v.toFixed(2):v}</div>
            </div>`).join('')}
          </div>
        </div>
      </div>`;
    }).join('');
    
    $('loading').style.display='none';
    $('results').classList.remove('hidden');
  }catch(e){
    $('loading').style.display='none';
    showAlert('error','Connection error: '+e.message);
  }
}

function toggle(key){
  const el=$('model-'+key);
  el.classList.toggle('open');
}

function showWatchlist(){showAlert('info','Watchlist coming soon');}
function showPortfolio(){showAlert('info','Portfolio coming soon');}
function showSettings(){showAlert('info','Settings coming soon');}

$('ti').addEventListener('keydown',e=>{if(e.key==='Enter')analyze();});
</script>
</body>
</html>"""

# ── Routes ───────────────────────────────────────────────────────────────
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
