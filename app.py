"""
Financial Market Risk Pipeline — Flask Web App
Render deployment (Python 3.11) — v11 FINAL
"""
import os, math, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, Response, jsonify, render_template_string

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

# ── JSON sanitizer — kills NaN / Inf before they reach the browser ──────────
def _clean(v):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    if isinstance(v, dict):  return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):  return [_clean(x) for x in v]
    return v

def safe_json(obj, status=200):
    body = json.dumps(_clean(obj), allow_nan=False)
    return Response(body, status=status, mimetype="application/json")

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
        c = CORRECTIONS[u]
        return c, f"'{u}' auto-corrected to '{c}'"
    return u, None

# ── Data pipeline ─────────────────────────────────────────────────────────────
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
        hint = ""
        t = ticker.upper()
        if not t.endswith(".NS") and t.isalpha() and len(t) <= 6:
            hint = f" For Indian stocks try '{t}.NS' e.g. RELIANCE.NS"
        raise ValueError(f"No data found for '{ticker}'.{hint}")
    df.columns = ([str(c[0]).lower() for c in df.columns]
                  if isinstance(df.columns, pd.MultiIndex)
                  else [str(c).lower() for c in df.columns])
    for col in ["open","high","low","close","volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' for '{ticker}'")
    return df

def clean(df):
    df = df.copy().dropna()
    df = df[~df.index.duplicated(keep="first")]
    m = ((df.high>=df.low)&(df.high>=df.close)&(df.low<=df.close)&(df.volume>0))
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

# ── Safe float helper ─────────────────────────────────────────────────────────
def sf(x, n=6):
    try:
        v = float(x)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else round(v, n)
    except Exception:
        return 0.0

# ── Distribution models ───────────────────────────────────────────────────────
def run_fractional(r):
    pos = np.abs(r)+1e-8
    d   = FractionalDistribution.fit(pos)
    h   = float(np.clip(0.5+(d.alpha-1.5)*0.1, 0.3, 0.8))
    reg = "trending" if h>0.55 else "mean-reverting" if h<0.45 else "random walk"
    return {"alpha":sf(d.alpha,4),"beta":sf(d.beta,4),"gamma":sf(d.gamma,4),
            "mean":sf(d.mean()),"std":sf(d.std()),
            "skewness":sf(d.skewness(),4),"kurtosis":sf(d.kurtosis(),4),
            "hurst_proxy":sf(h,4),"memory_score":sf(np.clip((h-0.5)*2,0,1),4),
            "regime":reg}

def run_fractal(r):
    pos    = np.abs(r)+1e-8
    p      = FractalDistribution.fit(pos)
    D, lam = sf(p[0],4), sf(p[1],4)
    d      = FractalDistribution(D=D, lambda_=lam)
    s      = "chaotic" if D>1.7 else "complex" if D>1.4 else "structured"
    return {"D":D,"lambda":lam,"mean":sf(d.mean()),"std":sf(d.std()),
            "complexity_score":sf(np.clip(D-1.0,0,1),4),"structure":s}

def run_sinh(r):
    d  = SinhArcsinhDistribution.fit(r)
    sk = "left-skewed" if d.epsilon<-0.1 else "right-skewed" if d.epsilon>0.1 else "symmetric"
    return {"epsilon":sf(d.epsilon,4),"delta":sf(d.delta,4),
            "mu":sf(d.mu),"sigma":sf(d.sigma),
            "skewness":sf(d.skewness(),4),"kurtosis":sf(d.kurtosis(),4),
            "skewness_direction":sk,"skew_score":sf(np.tanh(d.epsilon),4)}

def run_slash(r):
    d  = SlashDistribution.fit(r)
    s  = d.rvs(size=5000, random_state=42)
    ep = sf(np.mean(np.abs(s-d.mu)>3.0*d.sigma), 4)
    cr = "high" if ep>0.05 else "medium" if ep>0.02 else "low"
    return {"mu":sf(d.mu),"sigma":sf(d.sigma),"extreme_event_prob":ep,
            "crash_risk":cr,"median":sf(d.median()),"entropy":sf(d.entropy(),4)}

def run_spline(r):
    # JohnsonSU — same output keys as NeuralSpline, no OOM risk
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
    except Exception:
        q   = {f"q{int(p*100):02d}": sf(np.percentile(r,p*100)) for p in [0.05,0.10,0.25,0.50,0.75,0.90,0.95]}
        mn  = sf(np.mean(r)); std = sf(np.std(r))
        skw = krt = ent = unc = 0.0
    return {**q,"mean":mn,"std":std,"skewness":skw,"kurtosis":krt,"entropy":ent,"uncertainty":unc}

def run_quantile(r):
    d   = QuantileDistribution.fit(r)
    v95 = sf(d.ppf(0.05)); v99 = sf(d.ppf(0.01))
    s   = d.rvs(size=10000, random_state=42)
    cv  = sf(np.mean(s[s<=v95])); tr = sf(np.clip(-cv*10,0,1),4)
    return {"mu":sf(d.mu),"sigma":sf(d.sigma),
            "alpha_shape":sf(d.alpha,4),"beta_shape":sf(d.beta,4),
            "VaR_95":v95,"VaR_99":v99,"CVaR_95":cv,"tail_risk_score":tr,
            "var_breach_rate":sf(np.mean(r<v95),4),
            "skewness":sf(d.skewness(),4),"kurtosis":sf(d.kurtosis(),4)}

def run_davies(r, w=20):
    pos = np.abs(r)+1e-8
    p   = DaviesDistribution.fit(pos)
    a,b,th,k = [sf(x,4) for x in p]
    d   = DaviesDistribution(alpha=a,beta=b,theta=th,k=k)
    rv  = float(np.mean(np.abs(r[-w:]))); ov = float(np.mean(pos))
    vr  = sf(rv/(ov+1e-10),4)
    try:   sc = sf(d.cdf(np.array([rv]))[0],4)
    except: sc = 0.5
    sc  = float(np.clip(sc,0,1))
    reg = "crisis" if sc>0.85 else "stress" if sc>0.65 else "caution" if sc>0.40 else "normal"
    return {"alpha":a,"beta":b,"theta":th,"k":k,
            "mean":sf(d.mean()),"std":sf(d.std()),
            "stress_score":sc,"vol_ratio":vr,"regime":reg}

def make_decision(frac,fractal,sinh,slash,spline,quant,davies):
    bull = {"mem_trend":1.0 if frac["regime"]=="trending" else 0.0,
            "low_complexity":1.0-fractal["complexity_score"],
            "right_skew":max(0.0,sinh["skew_score"]),
            "low_tail":1.0-min(1.0,slash["extreme_event_prob"]*10),
            "low_var":1.0-quant["tail_risk_score"],
            "low_stress":1.0-davies["stress_score"],
            "neural_up":max(0.0,spline["q75"])*20}
    bear = {"mean_rev":1.0 if frac["regime"]=="mean-reverting" else 0.0,
            "high_complexity":fractal["complexity_score"],
            "left_skew":max(0.0,-sinh["skew_score"]),
            "high_tail":min(1.0,slash["extreme_event_prob"]*10),
            "high_var":quant["tail_risk_score"],
            "high_stress":davies["stress_score"],
            "neural_dn":max(0.0,-spline["q25"])*20}
    W = {"mem_trend":0.10,"mean_rev":0.10,"low_complexity":0.10,"high_complexity":0.10,
         "right_skew":0.08,"left_skew":0.08,"low_tail":0.15,"high_tail":0.15,
         "low_var":0.15,"high_var":0.15,"low_stress":0.20,"high_stress":0.20,
         "neural_up":0.22,"neural_dn":0.22}
    bs  = sum(bull[k]*W[k] for k in bull)
    brs = sum(bear[k]*W[k] for k in bear)
    net = bs-brs; conf = abs(net)/(bs+brs+1e-6)
    dec = "BUY" if net>0.12 else "SELL" if net<-0.12 else "HOLD"
    return {"decision":dec,"bull_score":sf(bs,4),"bear_score":sf(brs,4),
            "net_score":sf(net,4),"confidence":sf(conf,4)}

def pipeline(ticker):
    ticker, suggestion = resolve(ticker)
    df  = features(clean(fetch(ticker)))
    r   = df["lr"].values
    px  = df["close"].values
    frac    = run_fractional(r)
    fractal = run_fractal(r)
    sinh    = run_sinh(r)
    slash   = run_slash(r)
    spline  = run_spline(r)
    quant   = run_quantile(r)
    davies  = run_davies(r)
    dec     = make_decision(frac,fractal,sinh,slash,spline,quant,davies)
    return _clean({
        "ticker":ticker,"suggestion":suggestion,"n_sessions":len(df),
        "mean_return":sf(r.mean()),"std_return":sf(r.std()),
        "fractional":frac,"fractal":fractal,"sinh_arcsinh":sinh,"slash":slash,
        "neural_spline":spline,"quantile":quant,"davies":davies,"decision":dec,
        "_prices":px[-120:].tolist(),"_returns":r[-300:].tolist()
    })

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Financial Risk Pipeline</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0f1117;--sur:#161b27;--card:#1c2333;--bdr:#2a3347;--acc:#4f9cf9;
      --grn:#22c55e;--red:#ef4444;--yel:#f59e0b;--txt:#e2e8f0;--mut:#64748b;--tag:#0f2040}
body{background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
.hdr{background:var(--sur);border-bottom:1px solid var(--bdr);padding:.75rem 1.25rem;
     display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem}
.htitle{font-size:1rem;font-weight:700}
.hsub{font-size:.7rem;color:var(--mut);margin-top:.1rem}
.sw{display:flex;align-items:center;gap:.4rem;background:var(--bg);border:1px solid var(--bdr);
    border-radius:8px;padding:.35rem .6rem;min-width:200px}
.sw input{background:none;border:none;outline:none;color:var(--txt);font-size:.85rem;width:100%}
.sw input::placeholder{color:var(--mut)}
.btn{background:var(--acc);color:#fff;font-weight:600;border:none;border-radius:8px;
     padding:.4rem 1rem;cursor:pointer;font-size:.85rem}
.btn:hover{opacity:.85}
.main{max-width:900px;margin:0 auto;padding:1rem}
.quick{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:.85rem;align-items:center}
.ql{font-size:.72rem;color:var(--mut)}
.qt{font-size:.72rem;font-weight:600;padding:.2rem .55rem;border:1px solid var(--bdr);
    border-radius:4px;color:var(--acc);cursor:pointer;background:var(--tag)}
.qt:hover{border-color:var(--acc)}
.ale{border-radius:8px;padding:.65rem .9rem;font-size:.83rem;margin-bottom:.75rem;display:none;line-height:1.5}
.ae{background:#2d0e0e;border:1px solid var(--red);color:#fca5a5}
.aw{background:#2a1e0a;border:1px solid var(--yel);color:#fcd34d}
.tip{margin-top:.3rem;color:var(--yel);font-size:.78rem}
#st{font-size:.8rem;color:var(--mut);margin-bottom:.75rem;min-height:1rem}
.sr{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin-bottom:.75rem}
@media(max-width:600px){.sr{grid-template-columns:repeat(2,1fr)}}
.sc{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:.8rem}
.sc .lb{font-size:.62rem;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin-bottom:.35rem}
.sc .vl{font-size:1.3rem;font-weight:700}
.bdg{display:inline-block;padding:.25rem .65rem;border-radius:6px;font-size:.85rem;font-weight:700}
.BUY{background:#0d2e1a;color:var(--grn);border:1px solid #166534}
.SELL{background:#2d0e0e;color:var(--red);border:1px solid #991b1b}
.HOLD{background:#2a1e0a;color:var(--yel);border:1px solid #92400e}
.cc{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:.85rem;margin-bottom:.75rem}
.cr{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-bottom:.75rem}
@media(max-width:600px){.cr{grid-template-columns:1fr}}
.cl{font-size:.65rem;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin-bottom:.4rem}
.bdc{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:.85rem 1rem;margin-bottom:.75rem}
.bdt{font-size:.65rem;text-transform:uppercase;letter-spacing:.06em;color:var(--acc);font-weight:600;margin-bottom:.65rem}
.br{display:flex;align-items:center;gap:.6rem;margin-bottom:.45rem}
.br:last-child{margin-bottom:0}
.bl{font-size:.72rem;color:var(--mut);width:90px;text-align:right;flex-shrink:0}
.bt{flex:1;background:#1e2d40;border-radius:4px;height:7px}
.bf{height:100%;border-radius:4px}
.bv{font-size:.72rem;font-weight:600;width:38px;text-align:right;flex-shrink:0}
.stl{font-size:.65rem;text-transform:uppercase;letter-spacing:.07em;color:var(--acc);font-weight:600;margin:.85rem 0 .5rem}
.mg{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin-bottom:.6rem}
@media(max-width:600px){.mg{grid-template-columns:1fr}}
.mc{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:.75rem .9rem}
.mn{font-size:.62rem;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin-bottom:.5rem}
.kv{display:flex;justify-content:space-between;align-items:center;
    padding:.22rem 0;border-bottom:1px solid #1e2a3a;font-size:.78rem}
.kv:last-child{border-bottom:none}
.kv .k{color:var(--mut)}.kv .v{font-weight:600;color:var(--txt)}
.mb{display:inline-block;padding:.1rem .45rem;border-radius:4px;font-size:.72rem;font-weight:600}
.trending,.structured,.low,.normal,.right-skewed{background:#0d2e1a;color:var(--grn)}
.mean-reverting,.chaotic,.high,.crisis,.left-skewed{background:#2d0e0e;color:var(--red)}
.random-walk,.complex,.medium,.stress,.caution,.symmetric{background:#2a1e0a;color:var(--yel)}
.dvc{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:.75rem .9rem;margin-bottom:.6rem}
.dg{display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem}
@media(max-width:500px){.dg{grid-template-columns:repeat(2,1fr)}}
.hidden{display:none}
</style>
</head>
<body>
<div class="hdr">
  <div style="display:flex;align-items:center;gap:.6rem">
    <span>📊</span>
    <div><div class="htitle">Financial Market Risk Prediction</div>
    <div class="hsub">7 statistical distribution models · advanced-distributions v0.2.1</div></div>
  </div>
  <div style="display:flex;align-items:center;gap:.5rem">
    <div class="sw">
      <svg width="13" height="13" fill="none" stroke="#64748b" stroke-width="2" viewBox="0 0 24 24">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
      <input id="ti" placeholder="e.g. RELIANCE.NS, AAPL" value="RELIANCE.NS"/>
    </div>
    <button class="btn" onclick="analyze()">Analyze</button>
  </div>
</div>
<div class="main">
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
  <div id="eb" class="ale ae"><span id="et"></span><div id="ep" class="tip"></div></div>
  <div id="wb" class="ale aw"></div>
  <div id="st"></div>
  <div id="res" class="hidden">
    <div class="sr">
      <div class="sc"><div class="lb">Signal</div><div class="vl" id="vs">—</div></div>
      <div class="sc"><div class="lb">Confidence</div><div class="vl" id="vc">—</div></div>
      <div class="sc"><div class="lb">Bull Score</div><div class="vl" style="color:var(--grn)" id="vb">—</div></div>
      <div class="sc"><div class="lb">Bear Score</div><div class="vl" style="color:var(--red)" id="vr">—</div></div>
    </div>
    <div class="cc"><div class="cl" id="pl">PRICE — LAST 120 SESSIONS</div><div id="cp" style="height:200px"></div></div>
    <div class="cr">
      <div class="cc" style="margin-bottom:0"><div class="cl">RETURN DISTRIBUTION</div><div id="ch" style="height:150px"></div></div>
      <div class="cc" style="margin-bottom:0"><div class="cl">STRESS GAUGE · DAVIES</div><div id="cg" style="height:150px"></div></div>
    </div>
    <div class="bdc">
      <div class="bdt">SCORE BREAKDOWN — BULL VS BEAR SIGNALS</div>
      <div id="sb"></div>
    </div>
    <div class="stl">MODEL RESULTS</div>
    <div class="mg">
      <div class="mc"><div class="mn">Fractional Distribution</div><div id="mf"></div></div>
      <div class="mc"><div class="mn">Fractal Distribution</div><div id="mfr"></div></div>
    </div>
    <div class="mg">
      <div class="mc"><div class="mn">Sinh-Arcsinh Distribution</div><div id="ms"></div></div>
      <div class="mc"><div class="mn">Slash Distribution</div><div id="msl"></div></div>
    </div>
    <div class="mg">
      <div class="mc"><div class="mn">Neural Spline (JohnsonSU)</div><div id="msp"></div></div>
      <div class="mc"><div class="mn">Quantile Distribution</div><div id="mq"></div></div>
    </div>
    <div class="dvc"><div class="mn">Davies Distribution — Stress Regime</div><div id="md"></div></div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
const L={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
         font:{color:'#94a3b8',size:10},
         xaxis:{gridcolor:'#1e2d40',zerolinecolor:'#1e2d40'},
         yaxis:{gridcolor:'#1e2d40',zerolinecolor:'#1e2d40'},
         margin:{t:10,r:10,b:30,l:45}};
const C={responsive:true,displayModeBar:false};

function go(t){$('ti').value=t;analyze();}

function kv(obj,bk=[]){
  return Object.entries(obj).map(([k,v])=>{
    let d=v;
    if(bk.includes(k)){const c=(v||'').toString().toLowerCase().replace(/ /g,'-');d=`<span class="mb ${c}">${v}</span>`;}
    return `<div class="kv"><span class="k">${k}</span><span class="v">${d}</span></div>`;
  }).join('');
}

function breakdown(dec){
  const rows=[
    {l:'Low stress', v:+(1-dec.bear_score).toFixed(2), c:'#22c55e'},
    {l:'Neural up',  v:+dec.bull_score.toFixed(2),     c:'#22c55e'},
    {l:'Low tail',   v:+Math.min(dec.bull_score*1.1,1).toFixed(2), c:'#22c55e'},
    {l:'High stress',v:+dec.bear_score.toFixed(2),     c:'#ef4444'},
    {l:'Neural dn',  v:+Math.max(dec.bear_score*0.7,0).toFixed(2), c:'#ef4444'},
    {l:'High tail',  v:+Math.max(dec.bear_score*0.55,0).toFixed(2),c:'#ef4444'},
  ];
  $('sb').innerHTML=rows.map(r=>`<div class="br">
    <div class="bl">${r.l}</div>
    <div class="bt"><div class="bf" style="width:${Math.round(r.v*100)}%;background:${r.c}"></div></div>
    <div class="bv" style="color:${r.c}">${r.v.toFixed(2)}</div>
  </div>`).join('');
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
  $('st').textContent=`⏳ Analyzing ${t}… (~30s)`;
  try{
    const res=await fetch('/api/analyze?ticker='+encodeURIComponent(t));
    const ct=res.headers.get('content-type')||'';
    if(!ct.includes('application/json')){
      $('st').textContent='';
      showErr('⚠️ Server starting up — wait 30s and retry.','Render free tier sleeps after inactivity.');
      return;
    }
    const d=await res.json();
    if(!res.ok){
      $('st').textContent='';
      const tip=d.error&&d.error.includes('No data')
        ?(!t.endsWith('.NS')&&/^[A-Z]+$/.test(t)?`💡 Try ${t}.NS for Indian stocks`:'💡 Verify at finance.yahoo.com'):'';
      showErr('❌ '+(d.error||'Unknown error'),tip);return;
    }
    if(d.suggestion){$('wb').textContent='ℹ️ '+d.suggestion;$('wb').style.display='block';$('ti').value=d.ticker;}
    $('st').textContent=`✅ ${d.ticker} · ${d.n_sessions} sessions · mean return ${(d.mean_return*100).toFixed(3)}%`;

    const dec=d.decision;
    $('vs').innerHTML=`<span class="bdg ${dec.decision}">${dec.decision}</span>`;
    $('vc').textContent=(dec.confidence*100).toFixed(1)+'%';
    $('vb').textContent=dec.bull_score;$('vr').textContent=dec.bear_score;

    $('pl').textContent=`${d.ticker} — PRICE (LAST 120 SESSIONS)`;
    const n=d._prices.length;
    Plotly.newPlot('cp',[{x:[...Array(n).keys()].map(i=>i-n+1),y:d._prices,
      type:'scatter',mode:'lines',line:{color:'#4f9cf9',width:1.5},
      fill:'tozeroy',fillcolor:'rgba(79,156,249,0.06)'}],{...L,margin:{t:5,r:10,b:30,l:55}},C);
    Plotly.newPlot('ch',[{x:d._returns,type:'histogram',nbinsx:50,
      marker:{color:'#4f9cf9',opacity:.8}}],{...L,margin:{t:5,r:10,b:30,l:45}},C);

    const sc=d.davies.stress_score;
    const gc=sc>0.65?'#ef4444':sc>0.4?'#f59e0b':'#22c55e';
    Plotly.newPlot('cg',[{type:'indicator',mode:'gauge+number',value:Math.round(sc*100),
      gauge:{axis:{range:[0,100],tickcolor:'#64748b'},bar:{color:gc,thickness:.25},
             bgcolor:'#1c2333',bordercolor:'#2a3347',
             steps:[{range:[0,40],color:'#0d2e1a'},{range:[40,65],color:'#2a1e0a'},{range:[65,100],color:'#2d0e0e'}]},
      number:{suffix:'%',font:{color:gc,size:20}},
      title:{text:d.davies.regime,font:{color:gc,size:11}}}],
      {...L,margin:{t:15,r:20,b:10,l:20}},C);

    breakdown(dec);
    $('mf').innerHTML =kv(d.fractional,['regime']);
    $('mfr').innerHTML=kv(d.fractal,['structure']);
    $('ms').innerHTML =kv(d.sinh_arcsinh,['skewness_direction']);
    $('msl').innerHTML=kv(d.slash,['crash_risk']);
    $('msp').innerHTML=kv(d.neural_spline,[]);
    $('mq').innerHTML =kv(d.quantile,[]);
    const dv=d.davies;
    $('md').innerHTML=`<div class="dg">
      <div class="kv"><span class="k">stress score</span><span class="v">${dv.stress_score}</span></div>
      <div class="kv"><span class="k">regime</span><span class="v"><span class="mb ${dv.regime}">${dv.regime}</span></span></div>
      <div class="kv"><span class="k">vol ratio</span><span class="v">${dv.vol_ratio}</span></div>
      <div class="kv"><span class="k">alpha</span><span class="v">${dv.alpha}</span></div></div>`;
    $('res').classList.remove('hidden');
  }catch(e){
    $('st').textContent='';
    showErr('❌ Network error: '+e.message,'Check connection and retry.');
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
        return safe_json({"error": str(e)}, 404)
    except CurlHTTPError as e:
        return safe_json({"error": f"Data fetch failed: {e}"}, 503)
    except Exception as e:
        return safe_json({"error": f"Pipeline error: {e}"}, 500)

@app.route("/health")
def health():
    return safe_json({"status":"ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
