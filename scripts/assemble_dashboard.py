"""
Assemble premium RoadSight Rwanda dashboard HTML.
Dark theme, glassmorphism cards, animated KPIs, particle header, 3D surface.
"""
import json
from pathlib import Path

BASE = Path('.')
CHARTS_FILE = Path('/tmp/dashboard_charts.json')
OUT = BASE / 'dashboard' / 'accident_risk_dashboard.html'

with open(CHARTS_FILE) as f:
    charts = json.load(f)

def plotly_div(chart_json, div_id, height='100%'):
    return f'''<div id="{div_id}" style="height:{height};width:100%;"></div>
<script>
(function(){{
  var fig = {chart_json};
  Plotly.newPlot('{div_id}', fig.data, fig.layout, {{responsive:true, displayModeBar:false}});
}})();
</script>'''

html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RoadSight Rwanda — Road Accident Risk Analysis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
/* ── Reset & base ─────────────────────────────────────────────────── */
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:       #080c16;
  --bg2:      #0d1220;
  --glass:    rgba(255,255,255,0.04);
  --border:   rgba(255,255,255,0.08);
  --cyan:     #00d4ff;
  --green:    #2ed573;
  --amber:    #ffa502;
  --red:      #ff4757;
  --orange:   #ff6b35;
  --text:     #e0e6f0;
  --muted:    #8899aa;
  --card-r:   16px;
}
html{scroll-behavior:smooth}
body{
  background:var(--bg);
  color:var(--text);
  font-family:'Inter',sans-serif;
  font-size:15px;
  line-height:1.6;
  overflow-x:hidden;
}

/* ── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:var(--bg2)}
::-webkit-scrollbar-thumb{background:#1e3a4a;border-radius:3px}

/* ── Side nav ─────────────────────────────────────────────────────── */
#sidenav{
  position:fixed;right:20px;top:50%;transform:translateY(-50%);
  z-index:999;display:flex;flex-direction:column;gap:10px;
}
.nav-dot{
  width:10px;height:10px;border-radius:50%;
  background:rgba(255,255,255,0.15);
  border:1px solid rgba(255,255,255,0.2);
  cursor:pointer;transition:all .3s;
}
.nav-dot.active,.nav-dot:hover{
  background:var(--cyan);
  box-shadow:0 0 10px var(--cyan);
  transform:scale(1.3);
}

/* ── Hero header ──────────────────────────────────────────────────── */
#hero{
  position:relative;overflow:hidden;
  min-height:340px;
  display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  padding:60px 40px 50px;
  background:linear-gradient(160deg,#040812 0%,#061428 50%,#030b1c 100%);
}
#hero::before{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(0,180,255,0.07),transparent);
  pointer-events:none;
}
#particle-canvas{
  position:absolute;inset:0;width:100%;height:100%;opacity:.6;
}
.hero-content{position:relative;z-index:2;text-align:center}
.hero-eyebrow{
  font-family:'Inter',sans-serif;font-size:.78rem;font-weight:500;
  letter-spacing:.2em;text-transform:uppercase;
  color:var(--cyan);opacity:.8;margin-bottom:16px;
}
.hero-title{
  font-family:'Orbitron',sans-serif;
  font-size:clamp(2rem,5vw,3.4rem);
  font-weight:900;letter-spacing:.04em;
  background:linear-gradient(90deg,#ffffff 0%,var(--cyan) 60%,#7dd3fc 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
  margin-bottom:14px;
  opacity:0;animation:fadeSlideUp .9s ease .2s forwards;
}
.hero-sub{
  font-size:.95rem;color:var(--muted);max-width:600px;margin:0 auto 28px;
  opacity:0;animation:fadeSlideUp .9s ease .5s forwards;
}
.hero-badges{
  display:flex;flex-wrap:wrap;gap:10px;justify-content:center;
  opacity:0;animation:fadeSlideUp .9s ease .8s forwards;
}
.badge{
  font-size:.75rem;font-weight:500;letter-spacing:.05em;
  padding:5px 14px;border-radius:30px;
  border:1px solid rgba(0,212,255,0.25);
  background:rgba(0,212,255,0.07);
  color:var(--cyan);
}
.badge.warn{border-color:rgba(255,165,2,0.3);background:rgba(255,165,2,0.07);color:var(--amber)}

/* ── KPI row ──────────────────────────────────────────────────────── */
#kpi-row{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:16px;padding:28px 32px 16px;
}
.kpi-card{
  background:var(--glass);
  border:1px solid var(--border);
  border-radius:var(--card-r);
  padding:22px 20px;
  text-align:center;
  backdrop-filter:blur(12px);
  -webkit-backdrop-filter:blur(12px);
  position:relative;overflow:hidden;
  transition:transform .25s,box-shadow .25s;
  opacity:0;transform:translateY(20px);
}
.kpi-card.visible{opacity:1;transform:translateY(0);transition:opacity .6s,transform .6s}
.kpi-card:hover{transform:translateY(-4px);box-shadow:0 12px 40px rgba(0,0,0,.4)}
.kpi-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  border-radius:var(--card-r) var(--card-r) 0 0;
}
.kpi-card.c-cyan::before{background:var(--cyan);box-shadow:0 0 12px var(--cyan)}
.kpi-card.c-red::before {background:var(--red); box-shadow:0 0 12px var(--red)}
.kpi-card.c-orange::before{background:var(--orange);box-shadow:0 0 12px var(--orange)}
.kpi-card.c-amber::before{background:var(--amber);box-shadow:0 0 12px var(--amber)}
.kpi-card.c-green::before{background:var(--green);box-shadow:0 0 12px var(--green)}
.kpi-val{
  font-family:'Orbitron',sans-serif;
  font-size:2.1rem;font-weight:700;
  display:block;margin-bottom:6px;line-height:1;
}
.kpi-card.c-cyan .kpi-val{color:var(--cyan);text-shadow:0 0 20px rgba(0,212,255,.5)}
.kpi-card.c-red  .kpi-val{color:var(--red); text-shadow:0 0 20px rgba(255,71,87,.5)}
.kpi-card.c-orange .kpi-val{color:var(--orange);text-shadow:0 0 20px rgba(255,107,53,.5)}
.kpi-card.c-amber .kpi-val{color:var(--amber);text-shadow:0 0 20px rgba(255,165,2,.5)}
.kpi-card.c-green .kpi-val{color:var(--green);text-shadow:0 0 20px rgba(46,213,115,.5)}
.kpi-label{font-size:.8rem;color:var(--muted);letter-spacing:.04em;text-transform:uppercase}

/* ── Data notice ──────────────────────────────────────────────────── */
.notice{
  margin:8px 32px 20px;
  padding:11px 20px;border-radius:10px;
  background:rgba(255,165,2,.06);
  border:1px solid rgba(255,165,2,.2);
  font-size:.82rem;color:#cc9922;
  display:flex;align-items:center;gap:10px;
}

/* ── Sections ─────────────────────────────────────────────────────── */
.section{padding:12px 32px 24px}
.section-header{
  display:flex;align-items:center;gap:14px;
  margin-bottom:20px;padding-bottom:12px;
  border-bottom:1px solid var(--border);
}
.section-num{
  font-family:'Orbitron',sans-serif;font-size:.7rem;font-weight:700;
  color:var(--cyan);letter-spacing:.15em;opacity:.7;
  min-width:28px;
}
.section-title-text{
  font-size:1.05rem;font-weight:600;color:var(--text);letter-spacing:.02em;
}
.section-line{flex:1;height:1px;background:linear-gradient(90deg,var(--cyan),transparent);opacity:.3}

/* ── Chart cards ──────────────────────────────────────────────────── */
.chart-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(520px,1fr));
  gap:20px;
}
.chart-card{
  background:var(--glass);
  border:1px solid var(--border);
  border-radius:var(--card-r);
  padding:20px;
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  overflow:hidden;
  transition:border-color .3s,box-shadow .3s;
  opacity:0;transform:translateY(24px);
}
.chart-card.visible{opacity:1;transform:translateY(0);transition:opacity .7s,transform .7s}
.chart-card:hover{
  border-color:rgba(0,212,255,.2);
  box-shadow:0 8px 40px rgba(0,212,255,.05);
}
.chart-card-full{
  background:var(--glass);
  border:1px solid var(--border);
  border-radius:var(--card-r);
  padding:20px;
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  overflow:hidden;margin-bottom:20px;
  transition:border-color .3s,box-shadow .3s;
  opacity:0;transform:translateY(24px);
}
.chart-card-full.visible{opacity:1;transform:translateY(0);transition:opacity .7s,transform .7s}
.chart-card-full:hover{
  border-color:rgba(0,212,255,.2);
  box-shadow:0 8px 40px rgba(0,212,255,.05);
}

/* ── Insight callouts ─────────────────────────────────────────────── */
.insight-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:20px}
.insight{
  background:rgba(0,212,255,.04);
  border:1px solid rgba(0,212,255,.12);
  border-radius:10px;padding:16px 18px;
  font-size:.85rem;color:var(--muted);
  opacity:0;transform:translateY(16px);
}
.insight.visible{opacity:1;transform:translateY(0);transition:opacity .6s,transform .6s}
.insight strong{color:var(--text);display:block;margin-bottom:4px;font-size:.9rem}

/* ── Footer ───────────────────────────────────────────────────────── */
footer{
  text-align:center;padding:40px 20px;
  color:var(--muted);font-size:.8rem;
  border-top:1px solid var(--border);
  margin-top:20px;
}
.footer-logo{
  font-family:'Orbitron',sans-serif;font-size:1.1rem;font-weight:700;
  color:var(--cyan);letter-spacing:.1em;margin-bottom:10px;
}
.footer-rwanda{
  width:120px;height:60px;opacity:.12;margin:12px auto;
  fill:var(--cyan);
}

/* ── Animations ───────────────────────────────────────────────────── */
@keyframes fadeSlideUp{
  from{opacity:0;transform:translateY(20px)}
  to  {opacity:1;transform:translateY(0)}
}
@keyframes glow-pulse{
  0%,100%{text-shadow:0 0 20px rgba(0,212,255,.4)}
  50%     {text-shadow:0 0 40px rgba(0,212,255,.8)}
}

/* ── Responsive ───────────────────────────────────────────────────── */
@media(max-width:768px){
  #kpi-row{grid-template-columns:repeat(2,1fr);padding:16px}
  .section{padding:10px 16px 20px}
  .chart-grid{grid-template-columns:1fr}
  #sidenav{display:none}
  .notice{margin:8px 16px 16px}
}
</style>
</head>
<body>

<!-- Side navigation -->
<nav id="sidenav">
  <div class="nav-dot active" data-target="hero"      title="Overview"></div>
  <div class="nav-dot"        data-target="s-temporal" title="Temporal"></div>
  <div class="nav-dot"        data-target="s-geo"      title="Geographic"></div>
  <div class="nav-dot"        data-target="s-risk"     title="Risk Factors"></div>
  <div class="nav-dot"        data-target="s-model"    title="ML Model"></div>
</nav>

<!-- ── Hero ──────────────────────────────────────────────────────── -->
<section id="hero">
  <canvas id="particle-canvas"></canvas>
  <div class="hero-content">
    <p class="hero-eyebrow">Group D · Big Data Project 2026</p>
    <h1 class="hero-title">RoadSight Rwanda</h1>
    <p class="hero-sub">Road Accident Risk Analysis — 3,007 incidents · 5 provinces · 2022–2025</p>
    <div class="hero-badges">
      <span class="badge">📅 2022–2025</span>
      <span class="badge">📍 5 Provinces</span>
      <span class="badge">🤖 Random Forest + XGBoost</span>
      <span class="badge warn">⚠ Synthetic dataset — not real police records</span>
    </div>
  </div>
</section>

<!-- ── KPI Row ─────────────────────────────────────────────────── -->
<div id="kpi-row">
  <div class="kpi-card c-cyan"><span class="kpi-val" data-count="3007">0</span><span class="kpi-label">Total Accidents</span></div>
  <div class="kpi-card c-red"> <span class="kpi-val" data-count="206">0</span><span class="kpi-label">Fatal Accidents</span></div>
  <div class="kpi-card c-orange"><span class="kpi-val" data-count="400">0</span><span class="kpi-label">Serious / Grievous</span></div>
  <div class="kpi-card c-red"><span class="kpi-val" data-count-dec="6.9" data-suffix="%">0</span><span class="kpi-label">Fatal Rate</span></div>
  <div class="kpi-card c-amber"><span class="kpi-val" data-count-dec="18.0" data-suffix="%">0</span><span class="kpi-label">Speeding Suspected</span></div>
  <div class="kpi-card c-amber"><span class="kpi-val" data-count-dec="5.9" data-suffix="%">0</span><span class="kpi-label">Alcohol Suspected</span></div>
</div>

<div class="notice">
  ⚠ <strong>Data notice:</strong> This dataset is synthetic/simulated, generated for educational purposes only.
  Findings are illustrative and must not inform real policy without validation against official Rwanda National Police or RURA records.
</div>

<!-- ── Section 1: Severity ───────────────────────────────────── -->
<section class="section" id="s-severity">
  <div class="section-header">
    <span class="section-num">01</span>
    <span class="section-title-text">Severity Overview</span>
    <div class="section-line"></div>
  </div>

  <div class="insight-row">
    <div class="insight"><strong>44.9% — Property damage only</strong>Nearly half of all accidents result in no injury, but 1 in 15 is fatal.</div>
    <div class="insight"><strong>6.9% fatal rate</strong>206 of 3,007 accidents result in death — the primary target for prevention.</div>
    <div class="insight"><strong>13.3% serious/grievous</strong>These 400 cases represent severe injury that may cause permanent disability.</div>
  </div>

  <div class="chart-grid">
    <div class="chart-card" style="min-height:400px">
"""
html += plotly_div(charts['donut'], 'chart-donut', '380px')
html += r"""    </div>
    <div class="chart-card" style="min-height:400px">
"""
html += plotly_div(charts['hourly'], 'chart-hourly', '360px')
html += r"""    </div>
  </div>
</section>

<!-- ── Section 2: Temporal ──────────────────────────────────── -->
<section class="section" id="s-temporal">
  <div class="section-header">
    <span class="section-num">02</span>
    <span class="section-title-text">When Do Accidents Happen?</span>
    <div class="section-line"></div>
  </div>

  <div class="insight-row">
    <div class="insight"><strong>Peak hour: 17:00</strong>Evening rush sees the highest accident count — fatigue and traffic volume combine.</div>
    <div class="insight"><strong>3D surface is rotatable</strong>Drag the chart below to explore the full day × hour accident landscape.</div>
    <div class="insight"><strong>Fatal trend: 2022–2025</strong>No clear seasonal pattern — fatalities are distributed throughout the year.</div>
  </div>

  <div class="chart-card-full" style="min-height:520px">
"""
html += plotly_div(charts['surface3d'], 'chart-surface3d', '500px')
html += r"""  </div>

  <div class="chart-card-full" style="min-height:360px">
"""
html += plotly_div(charts['monthly'], 'chart-monthly', '340px')
html += r"""  </div>
</section>

<!-- ── Section 3: Geographic ────────────────────────────────── -->
<section class="section" id="s-geo">
  <div class="section-header">
    <span class="section-num">03</span>
    <span class="section-title-text">Where Do Accidents Happen?</span>
    <div class="section-line"></div>
  </div>

  <div class="insight-row">
    <div class="insight"><strong>63.1% — Kigali City</strong>The capital dominates the accident count due to traffic density and road network complexity.</div>
    <div class="insight"><strong>All 5 provinces affected</strong>Fatal accidents occur nationwide — rural roads carry their own risk despite lower volumes.</div>
    <div class="insight"><strong>Map tip</strong>Purple dots = fatal. Zoom and hover for district, vehicle, and weather detail.</div>
  </div>

  <div class="chart-card-full" style="min-height:560px">
"""
html += plotly_div(charts['map'], 'chart-map', '530px')
html += r"""  </div>

  <div class="chart-card-full" style="min-height:400px">
"""
html += plotly_div(charts['province'], 'chart-province', '380px')
html += r"""  </div>
</section>

<!-- ── Section 4: Risk Factors ──────────────────────────────── -->
<section class="section" id="s-risk">
  <div class="section-header">
    <span class="section-num">04</span>
    <span class="section-title-text">Risk Factors &amp; Vehicle Analysis</span>
    <div class="section-line"></div>
  </div>

  <div class="insight-row">
    <div class="insight"><strong>No helmet = highest risk</strong>Motorcycle riders without helmets have a mean severity score of 1.25 — 52% above baseline.</div>
    <div class="insight"><strong>Speeding + alcohol</strong>Both behavioural factors push severity well above baseline and are the most actionable targets for enforcement.</div>
    <div class="insight"><strong>Trucks: 10.1% fatal rate</strong>Though fewer in count, truck-involved accidents are twice as likely to be fatal as car accidents.</div>
  </div>

  <div class="chart-grid">
    <div class="chart-card" style="min-height:400px">
"""
html += plotly_div(charts['risk'], 'chart-risk', '380px')
html += r"""    </div>
    <div class="chart-card" style="min-height:380px">
"""
html += plotly_div(charts['vehicle'], 'chart-vehicle', '360px')
html += r"""    </div>
  </div>
</section>

<!-- ── Section 5: ML Model ───────────────────────────────────── -->
<section class="section" id="s-model">
  <div class="section-header">
    <span class="section-num">05</span>
    <span class="section-title-text">Machine Learning Model Performance</span>
    <div class="section-line"></div>
  </div>

  <div class="insight-row">
    <div class="insight"><strong>Random Forest wins</strong>Macro F1 = 0.320 vs XGBoost 0.265 — RF handles the class imbalance better across all severity levels.</div>
    <div class="insight"><strong>Fatal class is hardest</strong>With only 206 fatal cases (6.9%), both models struggle — F1 of 0.15 for Fatal. SMOTE helps but doesn't fully resolve rare-class learning.</div>
    <div class="insight"><strong>Top features</strong>Driver age, crash hour, and speed limit dominate feature importance — time and demographics matter most.</div>
  </div>

  <div class="chart-card-full" style="min-height:400px">
"""
html += plotly_div(charts['model'], 'chart-model', '380px')
html += r"""  </div>
</section>

<!-- ── Footer ────────────────────────────────────────────────── -->
<footer>
  <div class="footer-logo">ROADSIGHT RWANDA</div>
  <!-- Rwanda SVG silhouette (simplified outline) -->
  <svg class="footer-rwanda" viewBox="0 0 200 160" xmlns="http://www.w3.org/2000/svg">
    <path d="M60,10 L80,8 L110,15 L140,20 L165,35 L175,55 L170,80 L155,100
             L140,120 L120,140 L100,150 L80,145 L60,130 L45,110 L35,88
             L30,65 L35,42 L48,22 Z" fill="currentColor" opacity="0.6"/>
  </svg>
  <p>Group D · Big Data Project 2026 · Rwanda</p>
  <p style="margin-top:6px;opacity:.5">Built with Python · Plotly · scikit-learn · XGBoost &nbsp;|&nbsp; Dataset: Synthetic — not real police data</p>
</footer>

<!-- ── Scripts ───────────────────────────────────────────────── -->
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
// ── Particle canvas ────────────────────────────────────────────────
(function(){
  var canvas = document.getElementById('particle-canvas');
  var ctx    = canvas.getContext('2d');
  var W, H, particles = [];
  var PARTICLE_COUNT = 60;
  var MAX_DIST = 130;

  function resize(){
    W = canvas.width  = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  function Particle(){
    this.x  = Math.random()*W;
    this.y  = Math.random()*H;
    this.vx = (Math.random()-.5)*.4;
    this.vy = (Math.random()-.5)*.4;
    this.r  = Math.random()*1.8+.8;
  }
  for(var i=0;i<PARTICLE_COUNT;i++) particles.push(new Particle());

  function draw(){
    ctx.clearRect(0,0,W,H);
    for(var i=0;i<particles.length;i++){
      var p = particles[i];
      p.x += p.vx; p.y += p.vy;
      if(p.x<0||p.x>W) p.vx*=-1;
      if(p.y<0||p.y>H) p.vy*=-1;
      ctx.beginPath();
      ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle='rgba(0,212,255,.55)';
      ctx.fill();
      for(var j=i+1;j<particles.length;j++){
        var q   = particles[j];
        var dx  = p.x-q.x, dy = p.y-q.y;
        var dist= Math.sqrt(dx*dx+dy*dy);
        if(dist<MAX_DIST){
          ctx.beginPath();
          ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y);
          ctx.strokeStyle='rgba(0,212,255,'+(1-dist/MAX_DIST)*.18+')';
          ctx.lineWidth=.6;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
})();

// ── KPI counter animation ──────────────────────────────────────────
function animateCount(el, target, isDecimal, suffix, duration){
  var start  = 0;
  var step   = target / (duration/16);
  var timer  = setInterval(function(){
    start += step;
    if(start >= target){ start = target; clearInterval(timer); }
    el.textContent = isDecimal
      ? start.toFixed(1) + (suffix||'')
      : Math.floor(start).toLocaleString() + (suffix||'');
  }, 16);
}

// ── Intersection Observer for reveal + KPI trigger ─────────────────
var kpiTriggered = false;
var io = new IntersectionObserver(function(entries){
  entries.forEach(function(e){
    if(e.isIntersecting){
      e.target.classList.add('visible');
      // Trigger KPI counts when first KPI card appears
      if(!kpiTriggered && e.target.closest('#kpi-row')){
        kpiTriggered = true;
        document.querySelectorAll('[data-count]').forEach(function(el){
          animateCount(el, parseInt(el.dataset.count), false, el.dataset.suffix||'', 1600);
        });
        document.querySelectorAll('[data-count-dec]').forEach(function(el){
          animateCount(el, parseFloat(el.dataset.countDec), true, el.dataset.suffix||'', 1600);
        });
      }
    }
  });
},{threshold:0.1});

document.querySelectorAll(
  '.kpi-card,.chart-card,.chart-card-full,.insight'
).forEach(function(el){ io.observe(el); });

// ── Side nav active dot ─────────────────────────────────────────────
var sections = ['hero','s-temporal','s-geo','s-risk','s-model'];
var dots     = document.querySelectorAll('.nav-dot');
var sio = new IntersectionObserver(function(entries){
  entries.forEach(function(e){
    if(e.isIntersecting){
      var id = e.target.id;
      dots.forEach(function(d){ d.classList.remove('active'); });
      var dot = document.querySelector('[data-target="'+id+'"]');
      if(dot) dot.classList.add('active');
    }
  });
},{threshold:0.4});
sections.forEach(function(id){
  var el = document.getElementById(id);
  if(el) sio.observe(el);
});
dots.forEach(function(d){
  d.addEventListener('click',function(){
    var el = document.getElementById(d.dataset.target);
    if(el) el.scrollIntoView({behavior:'smooth'});
  });
});
</script>
</body>
</html>
"""

OUT.write_text(html, encoding='utf-8')
size_kb = OUT.stat().st_size / 1024
print(f"Dashboard saved: {OUT}  ({size_kb:.0f} KB)")
