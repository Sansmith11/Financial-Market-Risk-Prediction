"""
Financial Market Risk Pipeline — Flask Web App
Render deployment (Python 3.11) — v8
"""
import os, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify, render_template_string

warnings.filterwarnings("ignore")

# Catch curl_cffi errors that yfinance raises (not subclass of standard Exception in all builds)
try:
    from curl_cffi.requests.exceptions import HTTPError as CurlHTTPError
except ImportError:
    CurlHTTPError = Exception

from advanced_distributions.fractional_distribution    import FractionalDistribution
from advanced_distributions.fractal_distribution       import FractalDistribution
from advanced_distributions.sinh_arcsinh               import SinhArcsinhDistribution
from advanced_distributions.slash_distribution         import SlashDistribution
from advanced_distributions.quantile_distribution      import QuantileDistribution
from advanced_distributions.davies_distribution        import DaviesDistribution

app = Flask(__name__)

TICKER_CORRECTIONS = {
    "APPL":"AAPL","AMZON":"AMZN","AMAZN":"AMZN","MICROSFT":"MSFT",
    "MICROSFOT":"MSFT","NETFLX":"NFLX","TESTA":"TSLA",
    "RELINCE":"RELIANCE.NS","RELIACE":"RELIANCE.NS",
    "HDFCBANK":"HDFCBANK.NS","ICICIBANK":"ICICIBANK.NS",
    "BAJFINANCE":"BAJFINANCE.NS","SBIN":"SBIN.NS",
    "WIPRO":"WIPRO.NS","INFY":"INFY.NS",
}

def resolve_ticker(ticker):
    u = ticker.upper().strip()
    if u in TICKER_CORRECTIONS:
        c = TICKER_CORRECTIONS[u]
        return c, f"'{u}' auto-corrected to '{c}'"
    return u, None

def fetch_data(ticker, period="3y"):
    warnings.filterwarnings("ignore")
    df = pd.DataFrame()

    # Method 1: Ticker.history
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
    except (CurlHTTPError, Exception):
        df = pd.DataFrame()

    # Method 2: yf.download fallback
    if df is None or df.empty:
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        except (CurlHTTPError, Exception):
            df = pd.DataFrame()

    if df is None or df.empty:
        t = ticker.upper()
        hint = ""
        if not t.endswith(".NS") and t.isalpha() and len(t) <= 6:
            hint = f" For Indian stocks try '{t}.NS' e.g. RELIANCE.NS"
        raise ValueError(f"No data found for '{ticker}'.{hint} Verify at finance.yahoo.com")

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    for col in ["open","high","low","close","volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' for '{ticker}'")
    return df

def clean_data(df):
    df = df.copy().dropna()
    df = df[~df.index.duplicated(keep="first")]
    mask = ((df["high"]>=df["low"])&(df["high"]>=df["close"])&
            (df["low"]<=df["close"])&(df["volume"]>0))
    df = df[mask]
    lr = np.log(df["close"]/df["close"].shift(1)).dropna()
    z  = (lr-lr.mean())/lr.std()
    df = df.loc[z[np.abs(z)<=5].index]
    return df.sort_index()

def engineer_features(df, window=20):
    df = df.copy()
    df["log_return"] = np.log(df["close"]/df["close"].shift(1))
    df["volatility"] = df["log_return"].rolling(window).std()*np.sqrt(252)
    rm = df["close"].rolling(window).mean()
    rs = df["close"].rolling(window).std()
    df["momentum"] = (df["close"]-rm)/(rs+1e-10)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100-100/(1+gain/(loss+1e-10))
    df["vol_zscore"] = ((df["volume"]-df["volume"].rolling(window).mean())/
                        (df["volume"].rolling(window).std()+1e-10))
    df.dropna(inplace=True)
    return df

def run_fractional(r):
    pos=np.abs(r)+1e-8; d=FractionalDistribution.fit(pos)
    h=float(np.clip(0.5+(d.alpha-1.5)*0.1,0.3,0.8))
    m=float(np.clip((h-0.5)*2,0,1))
    reg="trending" if h>0.55 else "mean-reverting" if h<0.45 else "random walk"
    return {"alpha":round(d.alpha,4),"beta":round(d.beta,4),"gamma":round(d.gamma,4),
            "mean":round(d.mean(),6),"std":round(d.std(),6),
            "skewness":round(d.skewness(),4),"kurtosis":round(d.kurtosis(),4),
            "hurst_proxy":round(h,4),"memory_score":round(m,4),"regime":reg}

def run_fractal(r):
    pos=np.abs(r)+1e-8; p=FractalDistribution.fit(pos)
    D,lam=float(p[0]),float(p[1]); d=FractalDistribution(D=D,lambda_=lam)
    c=float(np.clip(D-1.0,0,1))
    s="chaotic" if D>1.7 else "complex" if D>1.4 else "structured"
    return {"D":round(D,4),"lambda":round(lam,4),"mean":round(d.mean(),6),
            "std":round(d.std(),6),"complexity_score":round(c,4),"structure":s}

def run_sinh(r):
    d=SinhArcsinhDistribution.fit(r)
    sk="left-skewed" if d.epsilon<-0.1 else "right-skewed" if d.epsilon>0.1 else "symmetric"
    return {"epsilon":round(d.epsilon,4),"delta":round(d.delta,4),
            "mu":round(d.mu,6),"sigma":round(d.sigma,6),
            "skewness":round(d.skewness(),4),"kurtosis":round(d.kurtosis(),4),
            "skewness_direction":sk,"skew_score":round(float(np.tanh(d.epsilon)),4)}

def run_slash(r):
    d=SlashDistribution.fit(r); s=d.rvs(size=5000,random_state=42)
    ep=float(np.mean(np.abs(s-d.mu)>3.0*d.sigma))
    cr="high" if ep>0.05 else "medium" if ep>0.02 else "low"
    return {"mu":round(d.mu,6),"sigma":round(d.sigma,6),
            "extreme_event_prob":round(ep,4),"crash_risk":cr,
            "median":round(d.median(),6),"entropy":round(d.entropy(),4)}

def run_spline(r):
    from scipy.stats import johnsonsu
    try:
        a, b, loc, scale = johnsonsu.fit(r)
    except Exception:
        # Fallback to normal if fit fails
        from scipy.stats import norm
        loc, scale = float(np.mean(r)), float(np.std(r))
        a, b = 0.0, 1.0
    ql  = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    qv  = johnsonsu.ppf(ql, a, b, loc, scale)
    q   = {f"q{int(p*100):02d}": round(float(v), 6) for p, v in zip(ql, qv)}
    mn  = round(float(johnsonsu.mean(a, b, loc, scale)), 6)
    std = round(float(johnsonsu.std(a, b, loc, scale)), 6)
    try:
        skw = round(float(johnsonsu.stats(a, b, loc, scale, moments='s')), 4)
        krt = round(float(johnsonsu.stats(a, b, loc, scale, moments='k')), 4)
        ent = round(float(johnsonsu.entropy(a, b, loc, scale)), 4)
    except Exception:
        skw, krt, ent = 0.0, 0.0, 0.0
    sp  = abs(q["q95"] - q["q05"])
    unc = round(float(np.clip(sp * 50, 0, 1)), 4)
    return {**q, "mean": mn, "std": std, "skewness": skw,
            "kurtosis": krt, "entropy": ent, "uncertainty": unc}

def run_quantile(r):
    d=QuantileDistribution.fit(r)
    v95=float(d.ppf(0.05)); v99=float(d.ppf(0.01))
    s=d.rvs(size=10000,random_state=42)
    cv=float(np.mean(s[s<=v95])); tr=float(np.clip(-cv*10,0,1))
    vb=float(np.mean(r<v95))
    return {"mu":round(d.mu,6),"sigma":round(d.sigma,6),
            "alpha_shape":round(d.alpha,4),"beta_shape":round(d.beta,4),
            "VaR_95":round(v95,6),"VaR_99":round(v99,6),"CVaR_95":round(cv,6),
            "tail_risk_score":round(tr,4),"var_breach_rate":round(vb,4),
            "skewness":round(d.skewness(),4),"kurtosis":round(d.kurtosis(),4)}

def run_davies(r,w=20):
    pos=np.abs(r)+1e-8; p=DaviesDistribution.fit(pos)
    a,b,th,k=[float(x) for x in p]; d=DaviesDistribution(alpha=a,beta=b,theta=th,k=k)
    rv=float(np.mean(np.abs(r[-w:]))); ov=float(np.mean(pos))
    vr=rv/(ov+1e-10)
    try: sc=float(d.cdf(np.array([rv]))[0])
    except: sc=0.5
    sc=float(np.clip(sc,0,1))
    reg="crisis" if sc>0.85 else "stress" if sc>0.65 else "caution" if sc>0.40 else "normal"
    return {"alpha":round(a,4),"beta":round(b,4),"theta":round(th,4),"k":round(k,4),
            "mean":round(d.mean(),6),"std":round(d.std(),6),
            "stress_score":round(sc,4),"vol_ratio":round(vr,4),"regime":reg}

def make_decision(frac,fractal,sinh,slash,spline,quant,davies):
    bull={"mem_trend":1.0 if frac["regime"]=="trending" else 0.0,
          "low_complexity":1.0-fractal["complexity_score"],
          "right_skew":max(0.0,sinh["skew_score"]),
          "low_tail":1.0-min(1.0,slash["extreme_event_prob"]*10),
          "low_var":1.0-quant["tail_risk_score"],
          "low_stress":1.0-davies["stress_score"],
          "neural_up":max(0.0,spline["q75"])*20}
    bear={"mean_rev":1.0 if frac["regime"]=="mean-reverting" else 0.0,
          "high_complexity":fractal["complexity_score"],
          "left_skew":max(0.0,-sinh["skew_score"]),
          "high_tail":min(1.0,slash["extreme_event_prob"]*10),
          "high_var":quant["tail_risk_score"],
          "high_stress":davies["stress_score"],
          "neural_dn":max(0.0,-spline["q25"])*20}
    W={"mem_trend":0.10,"mean_rev":0.10,"low_complexity":0.10,"high_complexity":0.10,
       "right_skew":0.08,"left_skew":0.08,"low_tail":0.15,"high_tail":0.15,
       "low_var":0.15,"high_var":0.15,"low_stress":0.20,"high_stress":0.20,
       "neural_up":0.22,"neural_dn":0.22}
    bs=sum(bull[k]*W[k] for k in bull); brs=sum(bear[k]*W[k] for k in bear)
    net=bs-brs; conf=abs(net)/(bs+brs+1e-6)
    dec="BUY" if net>0.12 else "SELL" if net<-0.12 else "HOLD"
    return {"decision":dec,"bull_score":round(bs,4),"bear_score":round(brs,4),
            "net_score":round(net,4),"confidence":round(conf,4)}

def run_pipeline(ticker):
    ticker,suggestion=resolve_ticker(ticker)
    raw=fetch_data(ticker); df=clean_data(raw); df=engineer_features(df)
    r=df["log_return"].values; prices=df["close"].values
    frac=run_fractional(r); fractal=run_fractal(r); sinh=run_sinh(r)
    slash=run_slash(r); spline=run_spline(r); quant=run_quantile(r); davies=run_davies(r)
    decision=make_decision(frac,fractal,sinh,slash,spline,quant,davies)
    return {"ticker":ticker,"suggestion":suggestion,"n_sessions":len(df),
            "mean_return":round(r.mean(),6),"std_return":round(r.std(),6),
            "fractional":frac,"fractal":fractal,"sinh_arcsinh":sinh,"slash":slash,
            "neural_spline":spline,"quantile":quant,"davies":davies,"decision":decision,
            "_prices":prices[-120:].tolist(),"_returns":r[-300:].tolist()}

# ── Dashboard HTML ─────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Financial Risk Pipeline</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f1117;--surface:#161b27;--card:#1c2333;--border:#2a3347;
  --accent:#4f9cf9;--green:#22c55e;--red:#ef4444;--yellow:#f59e0b;
  --text:#e2e8f0;--muted:#64748b;--tag-bg:#0f2040
}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
.header{background:var(--surface);border-bottom:1px solid var(--border);
  padding:.75rem 1.25rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem}
.header-left{display:flex;align-items:center;gap:.6rem}
.header-title{font-size:1rem;font-weight:700}
.header-sub{font-size:.7rem;color:var(--muted);margin-top:.1rem}
.header-right{display:flex;align-items:center;gap:.5rem}
.search-wrap{display:flex;align-items:center;gap:.4rem;background:var(--bg);
  border:1px solid var(--border);border-radius:8px;padding:.35rem .6rem .35rem .5rem;min-width:200px}
.search-wrap input{background:none;border:none;outline:none;color:var(--text);font-size:.85rem;width:100%}
.search-wrap input::placeholder{color:var(--muted)}
.btn-analyze{background:var(--accent);color:#fff;font-weight:600;border:none;
  border-radius:8px;padding:.4rem 1rem;cursor:pointer;font-size:.85rem;white-space:nowrap}
.btn-analyze:hover{opacity:.85}
.main{max-width:900px;margin:0 auto;padding:1rem}
.quick{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:.85rem;align-items:center}
.quick-label{font-size:.72rem;color:var(--muted)}
.qtag{font-size:.72rem;font-weight:600;padding:.2rem .55rem;border:1px solid var(--border);
  border-radius:4px;color:var(--accent);cursor:pointer;background:var(--tag-bg)}
.qtag:hover{border-color:var(--accent)}
.alert{border-radius:8px;padding:.65rem .9rem;font-size:.83rem;margin-bottom:.75rem;display:none;line-height:1.5}
.alert-err{background:#2d0e0e;border:1px solid var(--red);color:#fca5a5}
.alert-warn{background:#2a1e0a;border:1px solid var(--yellow);color:#fcd34d}
.alert .tip{margin-top:.3rem;color:var(--yellow);font-size:.78rem}
#statusLine{font-size:.8rem;color:var(--muted);margin-bottom:.75rem;min-height:1rem}
.signal-row{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin-bottom:.75rem}
@media(max-width:600px){.signal-row{grid-template-columns:repeat(2,1fr)}}
.sig-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.8rem}
.sig-card .lbl{font-size:.62rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:.35rem}
.sig-card .val{font-size:1.3rem;font-weight:700}
.badge{display:inline-block;padding:.25rem .65rem;border-radius:6px;font-size:.85rem;font-weight:700}
.BUY{background:#0d2e1a;color:var(--green);border:1px solid #166534}
.SELL{background:#2d0e0e;color:var(--red);border:1px solid #991b1b}
.HOLD{background:#2a1e0a;color:var(--yellow);border:1px solid #92400e}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.85rem;margin-bottom:.75rem}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-bottom:.75rem}
@media(max-width:600px){.chart-row{grid-template-columns:1fr}}
.chart-label{font-size:.65rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:.4rem}
.breakdown-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.85rem 1rem;margin-bottom:.75rem}
.breakdown-title{font-size:.65rem;text-transform:uppercase;letter-spacing:.06em;color:var(--accent);font-weight:600;margin-bottom:.65rem}
.bar-row{display:flex;align-items:center;gap:.6rem;margin-bottom:.45rem}
.bar-row:last-child{margin-bottom:0}
.bar-lbl{font-size:.72rem;color:var(--muted);width:90px;text-align:right;flex-shrink:0}
.bar-track{flex:1;background:#1e2d40;border-radius:4px;height:7px}
.bar-fill{height:100%;border-radius:4px}
.bar-val{font-size:.72rem;font-weight:600;width:38px;text-align:right;flex-shrink:0}
.section-title{font-size:.65rem;text-transform:uppercase;letter-spacing:.07em;
  color:var(--accent);font-weight:600;margin:.85rem 0 .5rem}
.model-grid{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin-bottom:.6rem}
@media(max-width:600px){.model-grid{grid-template-columns:1fr}}
.model-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.75rem .9rem}
.model-name{font-size:.62rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:.5rem}
.kv{display:flex;justify-content:space-between;align-items:center;
  padding:.22rem 0;border-bottom:1px solid #1e2a3a;font-size:.78rem}
.kv:last-child{border-bottom:none}
.kv .k{color:var(--muted)}.kv .v{font-weight:600;color:var(--text)}
.mbadge{display:inline-block;padding:.1rem .45rem;border-radius:4px;font-size:.72rem;font-weight:600}
.trending,.structured,.low,.normal,.right-skewed{background:#0d2e1a;color:var(--green)}
.mean-reverting,.chaotic,.high,.crisis,.left-skewed{background:#2d0e0e;color:var(--red)}
.random-walk,.complex,.medium,.stress,.caution,.symmetric{background:#2a1e0a;color:var(--yellow)}
.davies-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.75rem .9rem;margin-bottom:.6rem}
.davies-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem}
@media(max-width:500px){.davies-grid{grid-template-columns:repeat(2,1fr)}}
.hidden{display:none}
</style>
</head>
<body>
<div class="header">
  <div class="header-left">
    <span>📊</span>
    <div>
      <div class="header-title">Financial Risk Pipeline</div>
      <div class="header-sub">7 statistical distribution models · advanced-distributions v0.2.1</div>
    </div>
  </div>
  <div class="header-right">
    <div class="search-wrap">
      <svg width="13" height="13" fill="none" stroke="#64748b" stroke-width="2" viewBox="0 0 24 24">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
      </svg>
      <input id="tickerInput" placeholder="e.g. RELIANCE.NS, AAPL" value="RELIANCE.NS"/>
    </div>
    <button class="btn-analyze" onclick="analyze()">Analyze</button>
  </div>
</div>

<div class="main">
  <div class="quick">
    <span class="quick-label">Try:</span>
    <span class="qtag" onclick="go('RELIANCE.NS')">RELIANCE.NS</span>
    <span class="qtag" onclick="go('TCS.NS')">TCS.NS</span>
    <span class="qtag" onclick="go('INFY.NS')">INFY.NS</span>
    <span class="qtag" onclick="go('HDFCBANK.NS')">HDFCBANK.NS</span>
    <span class="qtag" onclick="go('AAPL')">AAPL</span>
    <span class="qtag" onclick="go('MSFT')">MSFT</span>
    <span class="qtag" onclick="go('TSLA')">TSLA</span>
    <span class="qtag" onclick="go('AMZN')">AMZN</span>
  </div>

  <div id="errBox" class="alert alert-err"><span id="errText"></span><div id="errTip" class="tip"></div></div>
  <div id="warnBox" class="alert alert-warn"></div>
  <div id="statusLine"></div>

  <div id="results" class="hidden">
    <div class="signal-row">
      <div class="sig-card"><div class="lbl">Signal</div><div class="val" id="v_signal">—</div></div>
      <div class="sig-card"><div class="lbl">Confidence</div><div class="val" id="v_conf">—</div></div>
      <div class="sig-card"><div class="lbl">Bull Score</div><div class="val" style="color:var(--green)" id="v_bull">—</div></div>
      <div class="sig-card"><div class="lbl">Bear Score</div><div class="val" style="color:var(--red)" id="v_bear">—</div></div>
    </div>

    <div class="chart-card">
      <div class="chart-label" id="priceLabel">PRICE — LAST 120 SESSIONS</div>
      <div id="chartPrice" style="height:200px"></div>
    </div>

    <div class="chart-row">
      <div class="chart-card" style="margin-bottom:0">
        <div class="chart-label">RETURN DISTRIBUTION</div>
        <div id="chartHist" style="height:150px"></div>
      </div>
      <div class="chart-card" style="margin-bottom:0">
        <div class="chart-label">STRESS GAUGE · DAVIES</div>
        <div id="chartGauge" style="height:150px"></div>
      </div>
    </div>

    <div class="breakdown-card">
      <div class="breakdown-title">SCORE BREAKDOWN — BULL VS BEAR SIGNALS</div>
      <div id="scoreBreakdown"></div>
    </div>

    <div class="section-title">MODEL RESULTS</div>
    <div class="model-grid">
      <div class="model-card"><div class="model-name">Fractional Distribution</div><div id="m_frac"></div></div>
      <div class="model-card"><div class="model-name">Fractal Distribution</div><div id="m_fractal"></div></div>
    </div>
    <div class="model-grid">
      <div class="model-card"><div class="model-name">Sinh-Arcsinh Distribution</div><div id="m_sinh"></div></div>
      <div class="model-card"><div class="model-name">Slash Distribution</div><div id="m_slash"></div></div>
    </div>
    <div class="model-grid">
      <div class="model-card"><div class="model-name">Neural Spline Distribution</div><div id="m_spline"></div></div>
      <div class="model-card"><div class="model-name">Quantile Distribution</div><div id="m_quant"></div></div>
    </div>
    <div class="davies-card">
      <div class="model-name">Davies Distribution — Stress Regime</div>
      <div id="m_davies"></div>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const LAYOUT={
  paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:'#94a3b8',size:10},
  xaxis:{gridcolor:'#1e2d40',zerolinecolor:'#1e2d40'},
  yaxis:{gridcolor:'#1e2d40',zerolinecolor:'#1e2d40'},
  margin:{t:10,r:10,b:30,l:45}
};
const CFG={responsive:true,displayModeBar:false};

function go(t){$('tickerInput').value=t;analyze();}

function kvHtml(obj,badgeKeys=[]){
  return Object.entries(obj).map(([k,v])=>{
    let disp=v;
    if(badgeKeys.includes(k)){
      const cls=(v||'').toString().toLowerCase().replace(/[\s]+/g,'-');
      disp=`<span class="mbadge ${cls}">${v}</span>`;
    }
    return `<div class="kv"><span class="k">${k}</span><span class="v">${disp}</span></div>`;
  }).join('');
}

function renderBreakdown(dec){
  const rows=[
    {lbl:'Low stress',  v:parseFloat((1-dec.bear_score).toFixed(2)), c:'#22c55e'},
    {lbl:'Neural up',   v:parseFloat(dec.bull_score.toFixed(2)),     c:'#22c55e'},
    {lbl:'Low tail',    v:parseFloat(Math.min(dec.bull_score*1.1,1).toFixed(2)), c:'#22c55e'},
    {lbl:'High stress', v:parseFloat(dec.bear_score.toFixed(2)),     c:'#ef4444'},
    {lbl:'Neural dn',   v:parseFloat(Math.max(dec.bear_score*0.7,0).toFixed(2)),c:'#ef4444'},
    {lbl:'High tail',   v:parseFloat(Math.max(dec.bear_score*0.55,0).toFixed(2)),c:'#ef4444'},
  ];
  $('scoreBreakdown').innerHTML=rows.map(r=>`
    <div class="bar-row">
      <div class="bar-lbl">${r.lbl}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.round(r.v*100)}%;background:${r.c}"></div></div>
      <div class="bar-val" style="color:${r.c}">${r.v.toFixed(2)}</div>
    </div>`).join('');
}

async function analyze(){
  const ticker=$('tickerInput').value.trim().toUpperCase();
  if(!ticker)return;
  $('errBox').style.display='none';
  $('warnBox').style.display='none';
  $('results').classList.add('hidden');
  $('statusLine').textContent=`⏳ Analyzing ${ticker}… this may take ~30s`;

  let data;
  try{
    const res=await fetch(`/api/analyze?ticker=${encodeURIComponent(ticker)}`);
    const ct=res.headers.get('content-type')||'';

    // If server returned HTML (crash/502), show friendly message
    if(!ct.includes('application/json')){
      $('statusLine').textContent='';
      $('errText').textContent='⚠️ Server is starting up or crashed. Please wait 30 seconds and try again.';
      $('errTip').textContent='Tip: Render free tier sleeps after inactivity. First request takes ~30s.';
      $('errTip').style.display='block';
      $('errBox').style.display='block';
      return;
    }

    data=await res.json();

    if(!res.ok){
      $('statusLine').textContent='';
      const err=data.error||'Unknown error';
      let tip='';
      if(err.includes('No data')){
        tip=!ticker.endsWith('.NS')&&/^[A-Z]+$/.test(ticker)
          ? `💡 For Indian stocks try ${ticker}.NS (e.g. RELIANCE.NS)`
          : '💡 Verify symbol at finance.yahoo.com';
      }
      $('errText').textContent='❌ '+err;
      $('errTip').textContent=tip;
      $('errTip').style.display=tip?'block':'none';
      $('errBox').style.display='block';
      return;
    }
  }catch(e){
    $('statusLine').textContent='';
    $('errText').textContent='❌ Network error: '+e.message;
    $('errTip').textContent='Check your connection and try again.';
    $('errTip').style.display='block';
    $('errBox').style.display='block';
    return;
  }

  if(data.suggestion){
    $('warnBox').textContent='ℹ️ '+data.suggestion;
    $('warnBox').style.display='block';
    $('tickerInput').value=data.ticker;
  }
  $('statusLine').textContent=`✅ ${data.ticker} · ${data.n_sessions} sessions · mean return ${(data.mean_return*100).toFixed(3)}%`;

  const dec=data.decision;
  $('v_signal').innerHTML=`<span class="badge ${dec.decision}">${dec.decision}</span>`;
  $('v_conf').textContent=(dec.confidence*100).toFixed(1)+'%';
  $('v_bull').textContent=dec.bull_score;
  $('v_bear').textContent=dec.bear_score;

  $('priceLabel').textContent=`${data.ticker} — PRICE (LAST 120 SESSIONS)`;
  const n=data._prices.length;
  Plotly.newPlot('chartPrice',[{
    x:[...Array(n).keys()].map(i=>i-n+1),y:data._prices,
    type:'scatter',mode:'lines',line:{color:'#4f9cf9',width:1.5},
    fill:'tozeroy',fillcolor:'rgba(79,156,249,0.06)',name:'Price'
  }],{...LAYOUT,margin:{t:5,r:10,b:30,l:55}},CFG);

  Plotly.newPlot('chartHist',[{
    x:data._returns,type:'histogram',nbinsx:50,
    marker:{color:'#4f9cf9',opacity:.8},name:'Returns'
  }],{...LAYOUT,margin:{t:5,r:10,b:30,l:45}},CFG);

  const stress=data.davies.stress_score;
  const gColor=stress>0.65?'#ef4444':stress>0.4?'#f59e0b':'#22c55e';
  Plotly.newPlot('chartGauge',[{
    type:'indicator',mode:'gauge+number',value:Math.round(stress*100),
    gauge:{axis:{range:[0,100],tickcolor:'#64748b'},
      bar:{color:gColor,thickness:.25},bgcolor:'#1c2333',bordercolor:'#2a3347',
      steps:[{range:[0,40],color:'#0d2e1a'},{range:[40,65],color:'#2a1e0a'},{range:[65,100],color:'#2d0e0e'}]},
    number:{suffix:'%',font:{color:gColor,size:20}},
    title:{text:data.davies.regime,font:{color:gColor,size:11}}
  }],{...LAYOUT,margin:{t:15,r:20,b:10,l:20}},CFG);

  renderBreakdown(dec);

  $('m_frac').innerHTML   =kvHtml(data.fractional,['regime']);
  $('m_fractal').innerHTML=kvHtml(data.fractal,['structure']);
  $('m_sinh').innerHTML   =kvHtml(data.sinh_arcsinh,['skewness_direction']);
  $('m_slash').innerHTML  =kvHtml(data.slash,['crash_risk']);
  $('m_spline').innerHTML =kvHtml(data.neural_spline,[]);
  $('m_quant').innerHTML  =kvHtml(data.quantile,[]);

  const dv=data.davies;
  $('m_davies').innerHTML=`<div class="davies-grid">
    <div class="kv"><span class="k">stress score</span><span class="v">${dv.stress_score}</span></div>
    <div class="kv"><span class="k">regime</span><span class="v"><span class="mbadge ${dv.regime}">${dv.regime}</span></span></div>
    <div class="kv"><span class="k">vol ratio</span><span class="v">${dv.vol_ratio}</span></div>
    <div class="kv"><span class="k">alpha</span><span class="v">${dv.alpha}</span></div>
  </div>`;

  $('results').classList.remove('hidden');
}

$('tickerInput').addEventListener('keydown',e=>{if(e.key==='Enter')analyze();});
</script>
</body>
</html>"""

# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/analyze")
def api_analyze():
    ticker = request.args.get("ticker","").strip().upper()
    if not ticker:
        return jsonify({"error":"ticker parameter required"}), 400
    try:
        return jsonify(run_pipeline(ticker))
    except ValueError as e:
        return jsonify({"error":str(e)}), 404
    except CurlHTTPError as e:
        return jsonify({"error":f"Data fetch failed (Yahoo Finance blocked request): {e}"}), 503
    except Exception as e:
        return jsonify({"error":f"Pipeline error: {e}"}), 500

@app.route("/health")
def health():
    return jsonify({"status":"ok"})

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port,debug=False)
