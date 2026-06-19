"""
Financial Market Risk Pipeline — Flask Web App
Render deployment (Python 3.11) — v12 FINAL
Improvements: better confidence scoring + richer signals per distribution
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

# ── Ticker helpers ────────────────────────────────────────────────────────────
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

# ── Data pipeline ─────────────────────────────────────────────────────────────
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
    # Signal: hurst above 0.5 = momentum/bull, below = mean-revert/bear
    bull_signal = float(np.clip((h-0.5)*4, 0, 1))   # 0→1 as h goes 0.5→0.75
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
    # Low complexity = structured market = bull signal
    bull_signal = 1.0 - comp
    bear_signal = comp
    return {"D":D,"lambda":lam,"mean":sf(d.mean()),"std":sf(d.std()),
            "complexity_score":sf(comp,4),"structure":s,
            "_bull":bull_signal,"_bear":bear_signal}

def run_sinh(r):
    d  = SinhArcsinhDistribution.fit(r)
    sk = "left-skewed" if d.epsilon<-0.1 else "right-skewed" if d.epsilon>0.1 else "symmetric"
    # Positive epsilon = right skew = more upside = bull
    skew_strength = float(np.tanh(d.epsilon * 2))  # amplified tanh
    bull_signal = float(np.clip(skew_strength, 0, 1))
    bear_signal = float(np.clip(-skew_strength, 0, 1))
    # Heavy tails (high delta) = uncertainty
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
    # Low extreme event prob + low entropy = structured = bull
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
        # Upside = q75-q50, Downside = q50-q25
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
    # Low tail risk + low breach rate = bull
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
    # vol_ratio close to 1 = stable = slightly bull
    stability = float(np.clip(1.0 - abs(vr - 1.0), 0, 1))
    bull_signal = float(np.clip((1.0-sc)*0.8 + stability*0.2, 0, 1))
    bear_signal = float(np.clip(sc*0.8 + (1-stability)*0.2, 0, 1))
    return {"alpha":a,"beta":b,"theta":th,"k":k,
            "mean":sf(d.mean()),"std":sf(d.std()),
            "stress_score":sf(sc,4),"vol_ratio":vr,"regime":reg,
            "_bull":bull_signal,"_bear":bear_signal}

# ── Decision engine (improved confidence) ────────────────────────────────────
def make_decision(frac, fractal, sinh, slash, spline, quant, davies):
    # Weighted signal aggregation — weights tuned by reliability
    W = {
        "davies":   0.25,   # Most reliable regime indicator
        "slash":    0.20,   # Tail/crash risk — critical
        "spline":   0.20,   # Forward-looking quantile spread
        "quant":    0.18,   # VaR-adjusted risk
        "frac":     0.10,   # Trend memory
        "fractal":  0.07,   # Market structure
    }
    models = {
        "davies":  davies,
        "slash":   slash,
        "spline":  spline,
        "quant":   quant,
        "frac":    frac,
        "fractal": fractal,
    }
    # Sinh adds a skew modifier on top
    skew_boost = sinh["_bull"] - sinh["_bear"]

    bs  = sum(models[k]["_bull"] * W[k] for k in W)
    brs = sum(models[k]["_bear"] * W[k] for k in W)

    # Apply skew modifier (±10% max)
    bs  = float(np.clip(bs  + skew_boost * 0.10, 0, 1))
    brs = float(np.clip(brs - skew_boost * 0.10, 0, 1))

    net = bs - brs

    # Sigmoid-scaled confidence — maps raw ratio to meaningful percentage
    raw_ratio = abs(net) / (bs + brs + 1e-6)
    confidence = 1.0 / (1.0 + math.exp(-8.0 * (raw_ratio - 0.25)))

    dec = "BUY" if net > 0.08 else "SELL" if net < -0.08 else "HOLD"

    # Build per-model signal breakdown for UI
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
    # Strip internal _bull/_bear keys from output
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

# ── Dashboard HTML ────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Financial Risk Pipeline</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0b0f19;--sur:#111827;--card:#161e2e;--bdr:#1f2d45;--acc:#3b82f6;
      --grn:#22c55e;--red:#ef4444;--yel:#f59e0b;--txt:#e2e8f0;--mut:#64748b;
      --tag:#0c1a35;--card2:#1a2437}
body{background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}

/* HEADER */
.hdr{background:var(--sur);border-bottom:1px solid var(--bdr);padding:.8rem 1.5rem;
     display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem;
     position:sticky;top:0;z-index:100;backdrop-filter:blur(8px)}
.htitle{font-size:1.05rem;font-weight:700;letter-spacing:-.01em}
.hsub{font-size:.68rem;color:var(--mut);margin-top:.1rem}
.sw{display:flex;align-items:center;gap:.4rem;background:var(--bg);
    border:1px solid var(--bdr);border-radius:8px;padding:.38rem .65rem;min-width:220px;
    transition:border-color .2s}
.sw:focus-within{border-color:var(--acc)}
.sw input{background:none;border:none;outline:none;color:var(--txt);font-size:.85rem;width:100%}
.sw input::placeholder{color:var(--mut)}
.btn{background:var(--acc);color:#fff;font-weight:600;border:none;border-radius:8px;
     padding:.42rem 1.1rem;cursor:pointer;font-size:.85rem;letter-spacing:.01em;
     transition:background .2s}
.btn:hover{background:#2563eb}

/* MAIN */
.main{max-width:960px;margin:0 auto;padding:1.25rem 1rem}

/* QUICK TAGS */
.quick{display:flex;flex-wrap:wrap;gap:.35rem;margin-bottom:1rem;align-items:center}
.ql{font-size:.7rem;color:var(--mut);margin-right:.2rem}
.qt{font-size:.7rem;font-weight:600;padding:.2rem .6rem;border:1px solid var(--bdr);
    border-radius:20px;color:var(--acc);cursor:pointer;background:var(--tag);transition:.15s}
.qt:hover,.qt.active{border-color:var(--acc);background:#0c1f45}

/* ALERTS */
.ale{border-radius:10px;padding:.7rem 1rem;font-size:.83rem;margin-bottom:1rem;display:none;line-height:1.5}
.ae{background:#1a0707;border:1px solid #7f1d1d;color:#fca5a5}
.aw{background:#1a1207;border:1px solid #78350f;color:#fcd34d}
.atip{margin-top:.3rem;font-size:.77rem;opacity:.85}
#st{font-size:.78rem;color:var(--mut);margin-bottom:1rem;min-height:1rem;padding-left:.1rem}

/* SIGNAL CARDS */
.sr{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin-bottom:1rem}
@media(max-width:640px){.sr{grid-template-columns:repeat(2,1fr)}}
.sc{background:var(--card);border:1px solid var(--bdr);border-radius:12px;padding:.9rem 1rem;
    position:relative;overflow:hidden}
.sc::before{content:'';position:absolute;inset:0;opacity:.04;pointer-events:none}
.sc .lb{font-size:.6rem;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin-bottom:.4rem}
.sc .vl{font-size:1.4rem;font-weight:700;line-height:1}
.bdg{display:inline-flex;align-items:center;padding:.28rem .75rem;border-radius:8px;
     font-size:.9rem;font-weight:700;letter-spacing:.02em}
.BUY{background:#052e16;color:var(--grn);border:1px solid #166534}
.SELL{background:#1c0505;color:var(--red);border:1px solid #991b1b}
.HOLD{background:#1c1205;color:var(--yel);border:1px solid #92400e}

/* CONFIDENCE BAR */
.conf-wrap{margin-top:.4rem}
.conf-track{background:#1e2d40;border-radius:4px;height:5px;margin-top:.35rem}
.conf-fill{height:100%;border-radius:4px;transition:width .8s ease}

/* CHARTS */
.cc{background:var(--card);border:1px solid var(--bdr);border-radius:12px;padding:.9rem;margin-bottom:.85rem}
.cr{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-bottom:.85rem}
@media(max-width:600px){.cr{grid-template-columns:1fr}}
.cl{font-size:.62rem;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);margin-bottom:.4rem}

/* SCORE BREAKDOWN */
.bdc{background:var(--card);border:1px solid var(--bdr);border-radius:12px;
     padding:.9rem 1.1rem;margin-bottom:.85rem}
.bdt{font-size:.62rem;text-transform:uppercase;letter-spacing:.07em;color:var(--acc);
     font-weight:700;margin-bottom:.75rem}
.br{display:flex;align-items:center;gap:.65rem;margin-bottom:.5rem}
.br:last-child{margin-bottom:0}
.bl{font-size:.71rem;color:var(--mut);width:95px;text-align:right;flex-shrink:0}
.bt{flex:1;background:#1a2535;border-radius:4px;height:8px;position:relative}
.bf{height:100%;border-radius:4px;transition:width .6s ease}
.bv{font-size:.71rem;font-weight:600;width:42px;text-align:right;flex-shrink:0}
.bw{font-size:.62rem;color:var(--mut);width:32px;text-align:right;flex-shrink:0}

/* SECTION TITLE */
.stl{font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;
     color:var(--acc);font-weight:700;margin:1rem 0 .6rem;
     display:flex;align-items:center;gap:.5rem}
.stl::after{content:'';flex:1;height:1px;background:var(--bdr)}

/* MODEL ACCORDION */
.models{display:flex;flex-direction:column;gap:.5rem;margin-bottom:.85rem}
.mod{background:var(--card);border:1px solid var(--bdr);border-radius:12px;overflow:hidden}
.mod-hdr{display:flex;align-items:center;justify-content:space-between;
          padding:.8rem 1rem;cursor:pointer;user-select:none;transition:background .15s}
.mod-hdr:hover{background:var(--card2)}
.mod-left{display:flex;align-items:center;gap:.65rem}
.mod-icon{font-size:1rem;width:1.4rem;text-align:center}
.mod-name{font-size:.82rem;font-weight:600}
.mod-tag{font-size:.67rem;color:var(--mut)}
.mod-right{display:flex;align-items:center;gap:.6rem}
.sig-pills{display:flex;gap:.3rem}
.pill{font-size:.65rem;font-weight:600;padding:.15rem .45rem;border-radius:20px}
.pill-b{background:#052e16;color:var(--grn);border:1px solid #166534}
.pill-s{background:#1c0505;color:var(--red);border:1px solid #991b1b}
.chevron{color:var(--mut);font-size:.75rem;transition:transform .25s}
.mod.open .chevron{transform:rotate(180deg)}
.mod-body{display:none;border-top:1px solid var(--bdr);padding:.85rem 1rem}
.mod.open .mod-body{display:block}
.kv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:.3rem}
.kv{display:flex;justify-content:space-between;align-items:center;
    padding:.28rem .5rem;border-radius:6px;font-size:.78rem;background:var(--bg)}
.kv .k{color:var(--mut)}.kv .v{font-weight:600;color:var(--txt)}
.mb{display:inline-block;padding:.1rem .45rem;border-radius:5px;font-size:.7rem;font-weight:600}
.trending,.structured,.low,.normal,.right-skewed{background:#052e16;color:var(--grn)}
.mean-reverting,.chaotic,.high,.crisis,.left-skewed{background:#1c0505;color:var(--red)}
.random-walk,.complex,.medium,.stress,.caution,.symmetric{background:#1c1205;color:var(--yel)}

/* DAVIES BOTTOM */
.dvc{background:var(--card);border:1px solid var(--bdr);border-radius:12px;padding:.85rem 1rem;margin-bottom:.85rem}
.dg{display:grid;grid-template-columns:repeat(4,1fr);gap:.4rem}
@media(max-width:500px){.dg{grid-template-columns:repeat(2,1fr)}}

.hidden{display:none}
</style>
</head>
<body>
<div class="hdr">
  <div style="display:flex;align-items:center;gap:.7rem">
    <span style="font-size:1.2rem">📊</span>
    <div>
      <div class="htitle">Financial Risk Pipeline</div>
      <div class="hsub">7 distribution models · advanced-distributions v0.2.1</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:.5rem">
    <div class="sw">
      <svg width="13" height="13" fill="none" stroke="#64748b" stroke-width="2" viewBox="0 0 24 24">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
      <input id="ti" placeholder="RELIANCE.NS, AAPL, MSFT…" value="RELIANCE.NS"/>
    </div>
    <button class="btn" onclick="analyze()">Analyze</button>
  </div>
</div>

<div class="main">
  <!-- Quick tickers -->
  <div class="quick">
    <span class="ql">Try:</span>
    <span class="qt" onclick="go('RELIANCE.NS')">RELIANCE.NS</span>
    <span class="qt" onclick="go('TCS.NS')">TCS.NS</span>
    <span class="qt" onclick="go('INFY.NS')">INFY.NS</span>
    <span class="qt" onclick="go('HDFCBANK.NS')">HDFCBANK.NS</span>
    <span class="qt" onclick="go('AAPL')">AAPL</span>
    <span class="qt" onclick="go('MSFT')">MSFT</span>
    <span class="qt" onclick="go('TSLA')">TSLA</span>
    <span class="qt" onclick="go('AMZN')">AMZN</span>
  </div>

  <div id="eb" class="ale ae"><span id="et"></span><div id="ep" class="atip"></div></div>
  <div id="wb" class="ale aw"></div>
  <div id="st"></div>

  <div id="res" class="hidden">
    <!-- Signal cards -->
    <div class="sr">
      <div class="sc">
        <div class="lb">Signal</div>
        <div class="vl" id="vs">—</div>
      </div>
      <div class="sc">
        <div class="lb">Confidence</div>
        <div class="vl" id="vc">—</div>
        <div class="conf-wrap">
          <div class="conf-track"><div class="conf-fill" id="cf"></div></div>
        </div>
      </div>
      <div class="sc">
        <div class="lb">Bull Score</div>
        <div class="vl" style="color:var(--grn)" id="vb">—</div>
      </div>
      <div class="sc">
        <div class="lb">Bear Score</div>
        <div class="vl" style="color:var(--red)" id="vr">—</div>
      </div>
    </div>

    <!-- Price chart -->
    <div class="cc">
      <div class="cl" id="pl">PRICE — LAST 120 SESSIONS</div>
      <div id="cp" style="height:210px"></div>
    </div>

    <!-- Return dist + Stress gauge -->
    <div class="cr">
      <div class="cc" style="margin-bottom:0">
        <div class="cl">RETURN DISTRIBUTION</div>
        <div id="ch" style="height:155px"></div>
      </div>
      <div class="cc" style="margin-bottom:0">
        <div class="cl">STRESS GAUGE · DAVIES</div>
        <div id="cg" style="height:155px"></div>
      </div>
    </div>

    <!-- Score breakdown -->
    <div class="bdc">
      <div class="bdt">SIGNAL BREAKDOWN — BULL VS BEAR PER MODEL</div>
      <div id="sb"></div>
    </div>

    <!-- Model accordion -->
    <div class="stl">MODEL DETAILS — CLICK TO EXPAND</div>
    <div class="models" id="models"></div>

    <!-- Davies summary -->
    <div class="dvc">
      <div class="cl">DAVIES DISTRIBUTION — STRESS REGIME</div>
      <div id="md"></div>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const L={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
         font:{color:'#64748b',size:10},
         xaxis:{gridcolor:'#1a2535',zerolinecolor:'#1a2535'},
         yaxis:{gridcolor:'#1a2535',zerolinecolor:'#1a2535'},
         margin:{t:8,r:8,b:28,l:42}};
const C={responsive:true,displayModeBar:false};

function go(t){
  document.querySelectorAll('.qt').forEach(x=>x.classList.remove('active'));
  event.target.classList.add('active');
  $('ti').value=t; analyze();
}

function kv(obj,bk=[]){
  return '<div class="kv-grid">'+Object.entries(obj).map(([k,v])=>{
    let d=v;
    if(bk.includes(k)){const c=(v||'').toString().toLowerCase().replace(/ /g,'-');
      d=`<span class="mb ${c}">${v}</span>`;}
    return `<div class="kv"><span class="k">${k}</span><span class="v">${d}</span></div>`;
  }).join('')+'</div>';
}

const MODEL_META = {
  frac:    {icon:'〜', name:'Fractional Distribution',    tag:'Long-memory · Hurst proxy'},
  fractal: {icon:'❄', name:'Fractal Distribution',       tag:'Complexity · Market structure'},
  sinh:    {icon:'⟛', name:'Sinh-Arcsinh Distribution',  tag:'Skewness · Tail shape'},
  slash:   {icon:'⚡', name:'Slash Distribution',         tag:'Extreme events · Crash risk'},
  spline:  {icon:'📈', name:'JohnsonSU / Neural Spline',  tag:'Quantile forecast · Uncertainty'},
  quant:   {icon:'📉', name:'Quantile Distribution',      tag:'VaR · CVaR · Tail risk'},
};

function buildModels(d){
  const dec = d.decision;
  const bd  = dec.breakdown;
  const maps = [
    {key:'frac',    data:d.fractional,   bk:['regime']},
    {key:'fractal', data:d.fractal,      bk:['structure']},
    {key:'sinh',    data:d.sinh_arcsinh, bk:['skewness_direction']},
    {key:'slash',   data:d.slash,        bk:['crash_risk']},
    {key:'spline',  data:d.neural_spline,bk:[]},
    {key:'quant',   data:d.quantile,     bk:[]},
  ];
  $('models').innerHTML = maps.map(({key,data,bk})=>{
    const m   = MODEL_META[key];
    const b   = bd[key] || {};
    const bs  = (b.bull||0).toFixed(2);
    const brs = (b.bear||0).toFixed(2);
    return `<div class="mod" id="mod_${key}">
      <div class="mod-hdr" onclick="toggle('${key}')">
        <div class="mod-left">
          <div class="mod-icon">${m.icon}</div>
          <div>
            <div class="mod-name">${m.name}</div>
            <div class="mod-tag">${m.tag}</div>
          </div>
        </div>
        <div class="mod-right">
          <div class="sig-pills">
            <span class="pill pill-b">▲ ${bs}</span>
            <span class="pill pill-s">▼ ${brs}</span>
          </div>
          <span class="chevron">▼</span>
        </div>
      </div>
      <div class="mod-body">${kv(data,bk)}</div>
    </div>`;
  }).join('');
}

function toggle(key){
  const el = $('mod_'+key);
  el.classList.toggle('open');
}

function breakdown(dec){
  const bd = dec.breakdown;
  const order = ['davies','slash','spline','quant','frac','fractal'];
  const names = {davies:'Davies',slash:'Slash',spline:'JohnsonSU',
                 quant:'Quantile',frac:'Fractional',fractal:'Fractal'};
  const rows = order.map(k=>{
    const b=bd[k]||{}; const w=b.weight||0;
    const bv=+(b.bull||0).toFixed(2); const brv=+(b.bear||0).toFixed(2);
    return `<div class="br">
      <div class="bl">${names[k]}</div>
      <div style="flex:1;display:flex;flex-direction:column;gap:3px">
        <div class="bt" style="height:5px"><div class="bf" style="width:${Math.round(bv*100)}%;background:#22c55e"></div></div>
        <div class="bt" style="height:5px"><div class="bf" style="width:${Math.round(brv*100)}%;background:#ef4444"></div></div>
      </div>
      <div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px;width:80px">
        <span class="bv" style="color:#22c55e">▲ ${bv.toFixed(2)}</span>
        <span class="bv" style="color:#ef4444">▼ ${brv.toFixed(2)}</span>
      </div>
      <div class="bw">${Math.round(w*100)}%</div>
    </div>`;
  });
  $('sb').innerHTML = rows.join('');
}

function showErr(msg,tip=''){
  $('eb').style.display='block';$('et').textContent=msg;
  $('ep').textContent=tip;$('ep').style.display=tip?'block':'none';
}

async function analyze(){
  const t=$('ti').value.trim().toUpperCase();
  if(!t)return;
  $('eb').style.display='none';$('wb').style.display='none';
  $('res').classList.add('hidden');
  $('st').textContent='⏳ Analyzing '+t+'… (~30s)';
  try{
    const res=await fetch('/api/analyze?ticker='+encodeURIComponent(t));
    const ct=res.headers.get('content-type')||'';
    if(!ct.includes('application/json')){
      $('st').textContent='';
      showErr('⚠️ Server starting up — wait 30s and retry.',
              'Render free tier sleeps after inactivity.');return;
    }
    const d=await res.json();
    if(!res.ok){
      $('st').textContent='';
      const tip=d.error&&d.error.includes('No data')
        ?(!t.endsWith('.NS')&&/^[A-Z]+$/.test(t)?'💡 Try '+t+'.NS for Indian stocks'
          :'💡 Verify at finance.yahoo.com'):'';
      showErr('❌ '+(d.error||'Error'),tip);return;
    }
    if(d.suggestion){$('wb').textContent='ℹ️ '+d.suggestion;$('wb').style.display='block';$('ti').value=d.ticker;}
    $('st').textContent='✅ '+d.ticker+' · '+d.n_sessions+' sessions · mean return '+(d.mean_return*100).toFixed(3)+'%';

    // Signal cards
    const dec=d.decision;
    $('vs').innerHTML='<span class="bdg '+dec.decision+'">'+dec.decision+'</span>';
    const confPct=Math.round(dec.confidence*100);
    $('vc').textContent=confPct+'%';
    const cColor=confPct>=70?'#22c55e':confPct>=45?'#f59e0b':'#ef4444';
    $('cf').style.cssText='width:'+confPct+'%;background:'+cColor;
    $('vb').textContent=dec.bull_score;
    $('vr').textContent=dec.bear_score;

    // Price chart
    $('pl').textContent=d.ticker+' — PRICE (LAST 120 SESSIONS)';
    const n=d._prices.length;
    Plotly.newPlot('cp',[{x:[...Array(n).keys()].map(i=>i-n+1),y:d._prices,
      type:'scatter',mode:'lines',line:{color:'#3b82f6',width:1.6},
      fill:'tozeroy',fillcolor:'rgba(59,130,246,0.05)'}],
      {...L,margin:{t:5,r:8,b:28,l:52}},C);

    // Histogram
    Plotly.newPlot('ch',[{x:d._returns,type:'histogram',nbinsx:50,
      marker:{color:'#3b82f6',opacity:.75}}],
      {...L,margin:{t:5,r:8,b:28,l:42}},C);

    // Gauge
    const sc=d.davies.stress_score;
    const gc=sc>0.65?'#ef4444':sc>0.4?'#f59e0b':'#22c55e';
    Plotly.newPlot('cg',[{type:'indicator',mode:'gauge+number',value:Math.round(sc*100),
      gauge:{axis:{range:[0,100],tickcolor:'#64748b'},bar:{color:gc,thickness:.22},
             bgcolor:'#161e2e',bordercolor:'#1f2d45',
             steps:[{range:[0,40],color:'#052e16'},{range:[40,65],color:'#1c1205'},
                    {range:[65,100],color:'#1c0505'}]},
      number:{suffix:'%',font:{color:gc,size:22}},
      title:{text:d.davies.regime,font:{color:gc,size:11}}}],
      {...L,margin:{t:18,r:18,b:10,l:18}},C);

    breakdown(dec);
    buildModels(d);

    // Davies bottom bar
    const dv=d.davies;
    $('md').innerHTML=`<div class="dg">
      <div class="kv"><span class="k">stress score</span><span class="v">${dv.stress_score}</span></div>
      <div class="kv"><span class="k">regime</span><span class="v"><span class="mb ${dv.regime}">${dv.regime}</span></span></div>
      <div class="kv"><span class="k">vol ratio</span><span class="v">${dv.vol_ratio}</span></div>
      <div class="kv"><span class="k">alpha</span><span class="v">${dv.alpha}</span></div></div>`;

    $('res').classList.remove('hidden');
  }catch(e){
    $('st').textContent='';
    showErr('❌ '+e.message,'Check connection and retry.');
  }
}
$('ti').addEventListener('keydown',e=>{if(e.key==='Enter')analyze();});
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────
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
