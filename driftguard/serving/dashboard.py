"""Built-in monitoring dashboard (zero-build React via CDN)."""

from __future__ import annotations

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>DriftGuard — Model Health</title>
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<style>
  :root{--bg:#0a0e18;--panel:#121a2c;--panel2:#1b2540;--text:#e8eefb;--muted:#8fa0c0;
        --accent:#5ac8fa;--good:#37d67a;--warn:#ffb020;--bad:#ff5c72;--line:#222f4a;}
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       background:radial-gradient(1100px 500px at 75% -10%,#132244,#0a0e18);color:var(--text)}
  header{padding:18px 26px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px}
  header h1{font-size:18px;margin:0}
  .badge{font-size:11px;color:var(--muted);border:1px solid var(--line);padding:3px 8px;border-radius:20px}
  .btn{margin-left:auto;background:var(--accent);color:#04101f;border:none;font-weight:700;
       padding:8px 14px;border-radius:9px;cursor:pointer}
  .btn.alt{background:var(--panel2);color:var(--text);margin-left:8px;border:1px solid var(--line)}
  .wrap{padding:22px 26px;max-width:1180px;margin:0 auto}
  .kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:20px}
  .kpi{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}
  .kpi .n{font-size:22px;font-weight:700}.kpi .l{font-size:11px;color:var(--muted);margin-top:3px}
  .grid{display:grid;grid-template-columns:1.3fr 1fr;gap:18px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:18px}
  .card h2{font-size:13px;margin:0 0 12px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
  .row{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--line);font-size:13px}
  .row:last-child{border-bottom:none}
  .grow{flex:1;min-width:0}
  .bar{height:8px;border-radius:5px;background:var(--panel2);overflow:hidden}
  .bar>i{display:block;height:100%}
  .pill{font-size:10px;padding:2px 7px;border-radius:6px}
  .drift{background:#3a1520;color:var(--bad)} .ok{background:#12321f;color:var(--good)}
  .ver{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
  .muted{color:var(--muted)}
  @media(max-width:900px){.grid{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div id="root"></div>
<script>
const {useState,useEffect}=React;const e=React.createElement;
const api=(p,o)=>fetch(p,o).then(r=>r.json());
function Kpi({n,l}){return e('div',{className:'kpi'},e('div',{className:'n'},n),e('div',{className:'l'},l));}
function App(){
  const [st,setSt]=useState(null);const [rep,setRep]=useState(null);const [busy,setBusy]=useState(false);
  async function load(){setSt(await api('/status'));}
  useEffect(()=>{load();},[]);
  async function runMonitor(drift){setBusy(true);
    const d=await api('/monitor',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({rows:1000,drift})});
    setRep(d);await load();setBusy(false);}
  const m=st?st.monitor:{};
  const latest=st&&st.versions.length?st.versions[st.versions.length-1]:null;
  return e('div',null,
    e('header',null,e('h1',null,'DriftGuard'),e('span',{className:'badge'},'PSI drift · self-retraining'),
      e('button',{className:'btn',onClick:()=>runMonitor(true),disabled:busy},busy?'Running…':'Simulate drift'),
      e('button',{className:'btn alt',onClick:()=>runMonitor(false),disabled:busy},'Normal batch')),
    e('div',{className:'wrap'},
      e('div',{className:'kpis'},
        e(Kpi,{n:st?('v'+st.model_version):'—',l:'live model version'}),
        e(Kpi,{n:st?st.total_versions:'—',l:'registered versions'}),
        e(Kpi,{n:latest?latest.metrics.f1:'—',l:'model F1'}),
        e(Kpi,{n:st?st.monitor.drift_events:'—',l:'drift events'}),
        e(Kpi,{n:st?((st.monitor.last_drift_share*100).toFixed(0)+'%'):'—',l:'last drift share'})),
      e('div',{className:'grid'},
        e('div',null,
          e('div',{className:'card'},e('h2',null,'Per-feature PSI (latest check)'),
            rep&&rep.report? rep.report.features.slice().sort((a,b)=>b.psi-a.psi).map((f,i)=>{
              const w=Math.min(100,f.psi/0.5*100);
              const col=f.drifted?'var(--bad)':(f.psi>=0.1?'var(--warn)':'var(--good)');
              return e('div',{className:'row',key:i},
                e('div',{style:{width:150}},f.feature),
                e('div',{className:'grow bar'},e('i',{style:{width:w+'%',background:col}})),
                e('div',{style:{width:60,textAlign:'right'}},f.psi.toFixed(3)),
                e('span',{className:'pill '+(f.drifted?'drift':'ok')},f.drifted?'DRIFT':'ok'));
            }):e('div',{className:'muted'},'Click “Simulate drift” to run a PSI check.')),
          rep&&rep.retrained? e('div',{className:'card'},e('h2',null,'Auto-retrain'),
            e('div',null,'Drift alert fired → retrained to ',e('b',null,'v'+rep.new_version),'. ',
              'F1 on drifted data recovered ',e('b',null,rep.f1_before),' → ',e('b',null,rep.f1_after),'.')):null),
        e('div',null,
          e('div',{className:'card'},e('h2',null,'Model registry'),
            st? st.versions.slice().reverse().map((v,i)=>e('div',{className:'row',key:i},
              e('span',{className:'ver'},'v'+v.version),
              e('div',{className:'grow'},e('span',{className:'muted'},v.reason),' · F1 '+v.metrics.f1+' · AUC '+v.metrics.auc)
            )):null),
          rep? e('div',{className:'card'},e('h2',null,'Last verdict'),
            e('div',null,'Dataset drift: ',e('b',{style:{color:rep.report.dataset_drift?'var(--bad)':'var(--good)'}},
              String(rep.report.dataset_drift)),' — ',(rep.report.drift_share*100).toFixed(0),
              '% of features shifted over ',rep.report.n_samples,' samples.')):null)
      )));
}
ReactDOM.createRoot(document.getElementById('root')).render(e(App));
</script>
</body>
</html>
"""
