"""Self-contained linked HTML workspace for scientific refinement review."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from .core import BraggCalculator
from .diagnostics import compare_calculators
from .structural_diagnostics import peak_group_attribution


WORKSPACE_SCHEMA = "braggcalculator.workspace/v1"


def _downsample_indices(length: int, maximum: int = 2400) -> np.ndarray:
    return np.linspace(0, length - 1, min(length, maximum)).astype(int)


def _parameter_rows(candidate):
    policy = candidate.provenance["policy"]
    bounds = {
        "scale": "> 0",
        "background_coefficients": "unbounded coefficients; negative profile penalized",
        "zero_shift": "unbounded local shift",
        "lattice": "positive-definite, symmetry-restricted metric",
        "coordinate_displacements": "periodic fractional coordinates",
        "occupancy_groups": "simplex: each fraction >= 0; site total <= 1",
        "isotropic_displacement_groups": "Biso >= 0",
        "anisotropic_displacement_groups": "positive-semidefinite U",
        "rigid_body_groups": "declared Cartesian rigid transform",
    }
    release = {
        "scale": "active",
        "background_coefficients": "active",
        "zero_shift": "active",
        "lattice": "active" if policy["refine_lattice"] else "fixed",
        "coordinate_displacements": "active" if policy["refine_coordinates"] else "fixed",
        "occupancy_groups": "active" if policy["occupancy_mode"] != "fixed" else "fixed",
        "isotropic_displacement_groups": "active" if policy["refine_b_iso"] else "fixed",
        "anisotropic_displacement_groups": "active" if policy["refine_u_aniso"] else "fixed",
        "rigid_body_groups": "active" if policy["rigid_bodies"] else "fixed",
    }
    restraint = {
        "coordinate_displacements": policy["coordinate_restraint"],
        "occupancy_groups": policy["occupancy_restraint"],
        "isotropic_displacement_groups": policy["b_iso_restraint"],
        "anisotropic_displacement_groups": policy["u_aniso_restraint"],
        "rigid_body_groups": policy["rigid_body_restraint"],
    }
    rows = []
    for name, value in candidate.physical_parameters.items():
        if name in {
            "lattice_parameterization", "structural_restraint_contributions",
            "structural_restraint_mean_chi_squared", "adaptive_release",
        }:
            continue
        rows.append(
            {
                "path": name,
                "value": value,
                "release": release.get(name, "derived"),
                "bounds": bounds.get(name, "see parameterization/provenance"),
                "restraint_weight": restraint.get(name),
            }
        )
    return rows


def _candidate_state(result, candidate):
    indices = _downsample_indices(len(result.dataset.coordinate))
    structure = candidate.structure
    calculator = BraggCalculator(
        mode=result.dataset.radiation,
        wavelength=result.dataset.wavelength,
        two_theta_range=(
            float(result.dataset.coordinate[0]), float(result.dataset.coordinate[-1])
        ),
        primitive=False,
    ).load(structure)
    table = calculator.reflection_table(domain="two_theta")
    intensity = np.asarray(table.intensity, dtype=float)
    strongest = np.argsort(intensity)[-min(250, len(intensity)) :][::-1]
    peak_groups = peak_group_attribution(calculator, fwhm_q=0.08, maximum_groups=40)
    sites = [
        {
            "index": index,
            "label": site.label or f"site {index}",
            "species": site.species_string,
            "fractional": np.asarray(site.frac_coords).tolist(),
        }
        for index, site in enumerate(structure)
    ]
    return {
        "name": candidate.name,
        "formula": structure.composition.reduced_formula,
        "r_wp": candidate.r_wp,
        "chi_squared": candidate.chi_squared,
        "held_out_r_wp": candidate.held_out_r_wp,
        "coordinate": result.dataset.coordinate[indices].tolist(),
        "observed": result.dataset.intensity[indices].tolist(),
        "calculated": candidate.calculated[indices].tolist(),
        "residual": candidate.residual[indices].tolist(),
        "sigma": result.dataset.sigma[indices].tolist(),
        "loss": candidate.loss_history.tolist(),
        "stages": list(candidate.stage_history),
        "sites": sites,
        "reflections": [
            {
                "hkl": np.asarray(table.hkl[index], dtype=int).tolist(),
                "two_theta": float(np.asarray(table.two_theta)[index]),
                "q": float(np.asarray(table.q)[index]),
                "intensity": float(intensity[index]),
            }
            for index in strongest
        ],
        "peak_groups": [
            {
                "q_center": group.q_center,
                "q_min": group.q_min,
                "q_max": group.q_max,
                "two_theta": float(
                    np.degrees(2 * np.arcsin(group.q_center * result.dataset.wavelength / (4 * np.pi)))
                ),
                "integrated_intensity": group.integrated_intensity,
                "effective_reflections": group.effective_reflections,
                "hkl": group.hkl.tolist(),
                "reflection_intensity": group.reflection_intensity.tolist(),
                "site_effects": group.site_effects,
            }
            for group in peak_groups
            if group.q_center * result.dataset.wavelength / (4 * np.pi) <= 1
        ],
        "peak_group_assumption": "Groups use a fixed 0.08 A^-1 FWHM resolution proxy",
        "parameters": _parameter_rows(candidate),
        "informative_regions": list(candidate.informative_regions),
        "identifiability": candidate.identifiability,
        "recommendation": candidate.recommendation,
        "warnings": list(candidate.warnings),
        "provenance": candidate.provenance,
    }


def _mismatch_state(result):
    if len(result.candidates) < 2:
        return {"available": False, "reason": "A second candidate is required."}
    left, right = result.candidates[:2]
    settings = {
        "mode": result.dataset.radiation,
        "wavelength": result.dataset.wavelength,
        "two_theta_range": (
            float(result.dataset.coordinate[0]), float(result.dataset.coordinate[-1])
        ),
        "primitive": False,
    }
    calculator_a = BraggCalculator(**settings).load(left.structure)
    calculator_b = BraggCalculator(**settings).load(right.structure)
    try:
        mismatch = compare_calculators(
            calculator_a, calculator_b, domain="two_theta", optimize_origin=True
        )
    except (ValueError, NotImplementedError) as error:
        return {
            "available": False,
            "reason": str(error),
            "scope": "starting structural models",
        }
    table = calculator_a.reflection_table(domain="two_theta")
    indices = mismatch.match.indices_a
    return {
        "available": True,
        "scope": "origin-aligned starting structural models",
        "left": left.name,
        "right": right.name,
        "d_sf": mismatch.d_sf,
        "d_amplitude": mismatch.d_amplitude,
        "d_phase": mismatch.d_phase,
        "points": [
            {
                "x": float(mismatch.x[position]),
                "y": float(mismatch.y[position]),
                "radius": float(mismatch.radius[position]),
                "hkl": mismatch.match.hkl[position].tolist(),
                "q": float(np.asarray(table.q)[indices[position]]),
                "two_theta": float(np.asarray(table.two_theta)[indices[position]]),
            }
            for position in np.argsort(mismatch.radius)[::-1][:500]
        ],
    }


def write_session_workspace(result, path, *, project=None, run_id=None) -> Path:
    """Write an offline HTML application with linked scientific views."""
    output = Path(path)
    state = {
        "schema": WORKSPACE_SCHEMA,
        "title": (project or {}).get("title", "BraggCalculator workspace"),
        "run_id": run_id,
        "conclusion": result.conclusion,
        "ranking": list(result.ranking),
        "pairwise_discrimination": result.pairwise_discrimination,
        "dataset": {
            "domain": result.dataset.domain,
            "radiation": result.dataset.radiation,
            "wavelength_angstrom": result.dataset.wavelength,
            "source": (
                project["dataset"]["path"]
                if project is not None and "dataset" in project
                else result.dataset.source
            ),
            "source_sha256": result.dataset.source_sha256,
        },
        "candidates": [_candidate_state(result, candidate) for candidate in result.candidates],
        "mismatch": _mismatch_state(result),
        "project": project or {},
    }
    encoded = json.dumps(state, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(state["title"])
    output.write_text(_workspace_html(title, encoded), encoding="utf-8")
    return output


def _workspace_html(title: str, encoded: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--ink:#17202a;--muted:#68737d;--paper:#f7f8fa;--card:#fff;--blue:#1261a0;
--orange:#d56b1f;--green:#16866a;--red:#b42318;--line:#d8dee5}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);
font:14px/1.45 system-ui,sans-serif}}header{{background:#102a43;color:white;padding:1rem 1.4rem;
display:flex;justify-content:space-between;align-items:center}}header h1{{font-size:1.15rem;margin:0}}
.layout{{display:grid;grid-template-columns:260px minmax(500px,1fr) 360px;gap:12px;padding:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}}
h2{{font-size:1rem;margin:.2rem 0 .7rem}}h3{{font-size:.9rem}}button,select{{font:inherit}}
button{{border:1px solid #9aabb9;background:white;border-radius:5px;padding:.35rem .55rem;cursor:pointer}}
button.active{{background:var(--blue);color:white;border-color:var(--blue)}}.toolbar{{display:flex;gap:.4rem;flex-wrap:wrap}}
svg{{width:100%;height:auto;border:1px solid var(--line);background:white}}.muted{{color:var(--muted)}}
.warning{{border-left:4px solid var(--orange);padding:.35rem .6rem;background:#fff7ed;margin:.35rem 0}}
.site{{cursor:pointer;stroke:white;stroke-width:1.5}}.site.selected{{stroke:#111;stroke-width:3}}
.tick{{stroke:#536777;cursor:pointer}}.mismatch{{cursor:pointer;opacity:.72}}.mismatch:hover{{opacity:1;stroke:#111}}
.cursor{{stroke:var(--red);stroke-width:1;stroke-dasharray:4 3}}table{{width:100%;border-collapse:collapse;font-size:.78rem}}
th,td{{border-bottom:1px solid var(--line);padding:.35rem;text-align:left;vertical-align:top}}
td.value{{max-width:190px;overflow-wrap:anywhere}}.active-param{{color:var(--green);font-weight:650}}
.fixed-param{{color:var(--muted)}}#detail{{min-height:95px;background:#f3f6f9;padding:.6rem;border-radius:5px}}
.wide{{grid-column:2/4}}@media(max-width:1100px){{.layout{{grid-template-columns:220px 1fr}}.right{{grid-column:1/3}}}}
@media(max-width:720px){{.layout{{display:block}}.card{{margin-bottom:10px}}}}
</style></head><body><header><h1>{title}</h1><div id="run"></div></header>
<main class="layout"><aside class="card"><h2>Candidate</h2><select id="candidate"></select>
<p id="score"></p><h2>Structure</h2><svg id="structure" viewBox="0 0 230 230"></svg>
<p class="muted">Fractional a-b projection. Select a site to inspect it.</p><div id="siteDetail"></div>
<h2>Warnings</h2><div id="warnings"></div></aside>
<section class="card"><div class="toolbar"><button class="active" data-view="profile">Profile</button>
<button data-view="mismatch">Mismatch disk</button><button data-view="trace">Optimization trace</button></div>
<h2 id="viewTitle">Observed, calculated and residual</h2><svg id="mainPlot" viewBox="0 0 900 430"></svg>
<div id="detail">Select a profile position, reflection tick, or mismatch point.</div>
<h2>Recommendation</h2><p id="recommendation"></p></section>
<aside class="card right"><h2>Parameters, constraints and release state</h2>
<div style="max-height:610px;overflow:auto"><table><thead><tr><th>Path</th><th>State</th><th>Value / bounds</th></tr></thead>
<tbody id="parameters"></tbody></table></div><h2>Provenance</h2><p id="provenance" class="muted"></p>
<h2>Exports</h2><ul><li><a href="profiles.csv">profile table</a></li><li><a href="parameters.csv">parameter table</a></li>
<li><a href="result.json">structured result</a></li></ul></aside></main>
<script id="state" type="application/json">{encoded}</script><script>
const S=JSON.parse(document.getElementById('state').textContent), NS='http://www.w3.org/2000/svg';
let ci=0,view='profile',cursor=null; const $=id=>document.getElementById(id);
function E(name,a={{}}){{const e=document.createElementNS(NS,name);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);return e}}
function extent(a){{let lo=Infinity,hi=-Infinity;for(const v of a){{lo=Math.min(lo,v);hi=Math.max(hi,v)}}return[lo,hi]}}
function scale(v,a,b,c,d){{return c+(v-a)*(d-c)/(b-a||1)}}
function path(x,y,X,Y){{return x.map((v,i)=>(i?'L':'M')+X(v).toFixed(1)+','+Y(y[i]).toFixed(1)).join(' ')}}
function candidate(){{return S.candidates[ci]}}
function drawStructure(){{const svg=$('structure');svg.replaceChildren();const c=candidate();
for(const site of c.sites){{const [x,y]=site.fractional;const hue=[...site.species].reduce((a,ch)=>a+ch.charCodeAt(0),0)%360;
const dot=E('circle',{{cx:20+190*x,cy:210-190*y,r:8,fill:`hsl(${{hue}} 58% 48%)`,class:'site'}});
dot.onclick=()=>{{svg.querySelectorAll('.site').forEach(x=>x.classList.remove('selected'));dot.classList.add('selected');
$('siteDetail').innerHTML=`<b>${{site.species}}</b><br>site ${{site.index}}<br><code>${{site.fractional.map(x=>x.toFixed(4)).join(', ')}}</code>`}};svg.append(dot)}}}}
function axes(svg){{svg.append(E('line',{{x1:55,y1:365,x2:865,y2:365,stroke:'#68737d'}}));svg.append(E('line',{{x1:55,y1:25,x2:55,y2:365,stroke:'#68737d'}}))}}
function drawProfile(){{const svg=$('mainPlot');svg.replaceChildren();axes(svg);const c=candidate(),x=c.coordinate;
const all=[...c.observed,...c.calculated,...c.residual], [xmin,xmax]=extent(x),[ymin,ymax]=extent(all);
const X=v=>scale(v,xmin,xmax,55,865),Y=v=>scale(v,ymin,ymax,365,25);
for(const [data,color] of [[c.observed,'#17202a'],[c.calculated,'#1261a0'],[c.residual,'#d56b1f']])
svg.append(E('path',{{d:path(x,data,X,Y),fill:'none',stroke:color,'stroke-width':1}}));
const maxI=Math.max(...c.peak_groups.map(r=>r.integrated_intensity),1);for(const g of c.peak_groups){{const xx=X(g.two_theta);
const tick=E('line',{{x1:xx,x2:xx,y1:395,y2:395-22*g.integrated_intensity/maxI,class:'tick'}});tick.onclick=()=>selectGroup(g,xx);svg.append(tick)}}
svg.onclick=e=>{{if(e.target!==svg)return;const box=svg.getBoundingClientRect(),xx=(e.clientX-box.left)*900/box.width;
const value=scale(xx,55,865,xmin,xmax);selectCoordinate(value,X(value))}};if(cursor!==null)selectCoordinate(cursor,X(cursor),false);
$('viewTitle').textContent='Observed (black), calculated (blue), residual (orange)';}}
function selectCoordinate(value,xpixel,update=true){{if(update)cursor=value;const svg=$('mainPlot');svg.querySelectorAll('.cursor').forEach(x=>x.remove());
svg.append(E('line',{{x1:xpixel,x2:xpixel,y1:25,y2:365,class:'cursor'}}));const c=candidate();let best=c.reflections[0];
for(const r of c.reflections)if(Math.abs(r.two_theta-value)<Math.abs(best.two_theta-value))best=r;
$('detail').innerHTML=`<b>${{value.toFixed(4)}} degrees</b><br>Nearest strong reflection: (${{best.hkl.join(' ')}}), Q=${{best.q.toFixed(4)}} A<sup>-1</sup>`}}
function selectReflection(r,xpixel){{cursor=r.two_theta;selectCoordinate(r.two_theta,xpixel);$('detail').innerHTML=`<b>Reflection (${{r.hkl.join(' ')}})</b><br>2theta=${{r.two_theta.toFixed(4)}} degrees; Q=${{r.q.toFixed(4)}} A<sup>-1</sup>; relative line value=${{r.intensity.toPrecision(4)}}`}}
function selectGroup(g,xpixel){{cursor=g.two_theta;selectCoordinate(g.two_theta,xpixel);const top=g.hkl.slice(0,6).map(x=>'('+x.join(' ')+')').join(', ');
$('detail').innerHTML=`<b>Resolution-defined peak group</b><br>2theta=${{g.two_theta.toFixed(4)}} degrees; Q=${{g.q_center.toFixed(4)}} A<sup>-1</sup>; effective reflections=${{g.effective_reflections.toFixed(2)}}<br>${{top}}<br><span class="muted">${{candidate().peak_group_assumption}}</span>`}}
function drawMismatch(){{const svg=$('mainPlot');svg.replaceChildren();const m=S.mismatch;$('viewTitle').textContent='Complex structure-factor mismatch';
if(!m.available){{$('detail').textContent=m.reason;return}}const circle=E('circle',{{cx:450,cy:210,r:170,fill:'none',stroke:'#68737d'}});svg.append(circle);
svg.append(E('line',{{x1:280,y1:210,x2:620,y2:210,stroke:'#d8dee5'}}));svg.append(E('line',{{x1:450,y1:40,x2:450,y2:380,stroke:'#d8dee5'}}));
for(const p of m.points){{const dot=E('circle',{{cx:450+170*p.x,cy:210-170*p.y,r:2+5*p.radius,fill:'#1261a0',class:'mismatch'}});
dot.onclick=()=>{{$('detail').innerHTML=`<b>(${{p.hkl.join(' ')}})</b> radius=${{p.radius.toFixed(4)}}<br>Q=${{p.q.toFixed(4)}} A<sup>-1</sup>; 2theta=${{p.two_theta.toFixed(4)}} degrees`;cursor=p.two_theta}};svg.append(dot)}}
$('detail').innerHTML=`${{m.scope}}<br>D<sub>SF</sub>=${{m.d_sf.toFixed(4)}}; amplitude=${{m.d_amplitude.toFixed(4)}}; phase=${{m.d_phase.toFixed(4)}}`;}}
function drawTrace(){{const svg=$('mainPlot');svg.replaceChildren();axes(svg);const y=candidate().loss,x=y.map((_,i)=>i),[ymin,ymax]=extent(y.map(v=>Math.log10(Math.max(v,1e-30))));
const X=v=>scale(v,0,Math.max(x.length-1,1),55,865),Y=v=>scale(Math.log10(Math.max(v,1e-30)),ymin,ymax,365,25);
svg.append(E('path',{{d:path(x,y,X,Y),fill:'none',stroke:'#16866a','stroke-width':2}}));$('viewTitle').textContent='Optimization loss (log scale)';
$('detail').innerHTML=`${{y.length}} recorded steps; ${{y[0].toPrecision(4)}} to ${{y[y.length-1].toPrecision(4)}}`;}}
function drawMain(){{if(view==='profile')drawProfile();else if(view==='mismatch')drawMismatch();else drawTrace()}}
function render(){{const c=candidate();$('score').innerHTML=`<b>Rwp ${{c.r_wp.toFixed(5)}}</b><br>chi squared ${{c.chi_squared.toFixed(3)}}`;
$('recommendation').textContent=c.recommendation;$('warnings').innerHTML=c.warnings.map(x=>`<div class="warning">${{x}}</div>`).join('')||'<span class="muted">None</span>';
$('parameters').innerHTML=c.parameters.map(p=>`<tr><td><code>${{p.path}}</code></td><td class="${{p.release==='active'?'active-param':'fixed-param'}}">${{p.release}}</td><td class="value"><code>${{JSON.stringify(p.value)}}</code><br><span class="muted">${{p.bounds}}${{p.restraint_weight===null||p.restraint_weight===undefined?'':'; restraint '+p.restraint_weight}}</span></td></tr>`).join('');
$('provenance').textContent=`dataset ${{S.dataset.source_sha256||'in-memory'}}; checkpoint ${{c.provenance.checkpoint.format}}`;drawStructure();drawMain()}}
const candidateSelect=$('candidate');for(const [i,c]of S.candidates.entries()){{const o=document.createElement('option');o.value=i;o.textContent=`${{c.name}} — ${{c.formula}}`;candidateSelect.append(o)}}
candidateSelect.onchange=e=>{{ci=+e.target.value;cursor=null;render()}};document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-view]').forEach(x=>x.classList.remove('active'));b.classList.add('active');view=b.dataset.view;drawMain()}});
$('run').textContent=`${{S.run_id||''}} · ${{S.dataset.radiation}} · ${{S.dataset.wavelength_angstrom}} A`;render();
</script></body></html>"""
