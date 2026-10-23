#!/usr/bin/env python3
"""
Build the ThingsBoard widget for the cabin twin from the viewer's own 3-D code, so the page and
the widget never drift apart.

  python3 build_tb_widget.py            -> thingsboard/cabin_twin_widget.json (+ .html/.css/.js copies)

Import in ThingsBoard: Widgets library -> + -> Import widget -> cabin_twin_widget.json.
The widget reads ONE telemetry key, `twin` (a JSON object per frame, as sent by twin_publish.py
or the bus unit).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "thingsboard")
src = open(os.path.join(HERE, "cabin_twin_template.html")).read()

a = src.index("  // ---------- three.js scene ----------")
b = src.index("  // ---------- playback ----------")
core = src[a:b]
# --- adapt the page code to a widget container ---
core = core.replace('const view = $("view3d");', "const view = viewEl;")
core = core.replace('document.querySelectorAll(".camset button")', 'root.querySelectorAll(".camset button")')
t0 = core.index("  function applyTheme() {")
t1 = core.index("  // ---------- pose application ----------")
core = core[:t0] + '  scene.background = new THREE.Color(0xdfe3e1);\n\n' + core[t1:]
core = core.replace("    drawTimeline();\n", "")
core = core.replace("  new ResizeObserver(resize).observe(view);", "  const ro = new ResizeObserver(resize); ro.observe(view);")
assert "drawTimeline" not in core and "$(" not in core, "page-only code left in widget core"

TEMPLATE_HTML = """<div class="tw">
  <div class="tw-view">
    <div class="tw-top">
      <span class="tw-driver">Waiting for telemetry…</span>
      <div class="camset">
        <button type="button" data-cam="dms" class="on">DMS camera</button>
        <button type="button" data-cam="side">Side</button>
        <button type="button" data-cam="free">Orbit</button>
      </div>
    </div>
    <div class="tw-chips"></div>
    <div class="tw-alert" hidden></div>
    <div class="tw-stale" hidden>No telemetry for <b>0</b> s</div>
  </div>
</div>"""

TEMPLATE_CSS = """.tw { position: absolute; inset: 0; font-family: Roboto, system-ui, sans-serif; }
.tw-view { position: absolute; inset: 0; background: #dfe3e1; overflow: hidden; }
.tw-view canvas { display: block; width: 100%; height: 100%; touch-action: none; }
.tw-top { position: absolute; left: 10px; right: 8px; top: 8px; display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; pointer-events: none; }
.tw-driver { font-weight: 700; font-size: 16px; color: #1b2124; background: rgba(248,249,247,.85); padding: 3px 8px; border-radius: 4px; }
.camset { display: flex; gap: 4px; pointer-events: auto; }
.camset button { font: 500 12px Roboto, system-ui, sans-serif; padding: 4px 8px; border: 1px solid #c9cfcc; background: #f8f9f7; color: #1b2124; border-radius: 4px; cursor: pointer; }
.camset button.on { background: #c27a00; border-color: #c27a00; color: #1b1408; }
.tw-chips { position: absolute; left: 10px; top: 44px; display: grid; gap: 4px; pointer-events: none; }
.tw-chip { font: 600 11px Roboto, system-ui, sans-serif; letter-spacing: .04em; text-transform: uppercase; padding: 2px 7px; border-radius: 99px; background: rgba(248,249,247,.9); border: 1px solid currentColor; justify-self: start; }
.tw-chip.ok { color: #2f7d4f; } .tw-chip.warn { color: #b5671a; } .tw-chip.bad { color: #c4362c; } .tw-chip.na { color: #5e676b; }
.tw-alert { position: absolute; left: 0; right: 0; bottom: 0; background: #c4362c; color: #fff; font-weight: 700; font-size: 18px; padding: 7px 12px; }
.tw-stale { position: absolute; right: 10px; bottom: 10px; font-size: 12px; color: #5e676b; background: rgba(248,249,247,.9); padding: 3px 8px; border-radius: 4px; }
[hidden] { display: none !important; }"""

CONTROLLER = """// Cabin twin widget: renders a generic driver from the `twin` telemetry key (one JSON object per frame).
let twinInst = null;

self.onInit = function () {
  const root = self.ctx.$container[0];
  twinInst = makeCabinTwin(root);
  self.onDataUpdated();
};

self.onDataUpdated = function () {
  if (!twinInst) return;
  for (const d of self.ctx.data || []) {
    if (!d.dataKey || d.dataKey.name !== "twin" || !d.data || !d.data.length) continue;
    const [ts, v] = d.data[d.data.length - 1];
    let f = v;
    if (typeof v === "string") { try { f = JSON.parse(v); } catch (e) { continue; } }
    if (f && typeof f === "object") twinInst.push(f, ts);
  }
};

self.onResize = function () { if (twinInst) twinInst.resize(); };
self.onDestroy = function () { if (twinInst) twinInst.destroy(); twinInst = null; };

self.typeParameters = function () {
  return { maxDatasources: 1, maxDataKeys: 1, dataKeysOptional: false, singleEntity: true };
};

function makeCabinTwin(root) {
  const viewEl = root.querySelector(".tw-view");
  const NEUTRAL = { yaw: 0, pitch: 0, roll: 0, eyeL: 1, eyeR: 1, jaw: 0, lean: [0, 0], lE: [0.75, -0.95, 0.45], lW: [0.45, -1.15, 1.1],
    rE: [-0.75, -0.95, 0.45], rW: [-0.45, -1.15, 1.1], armOk: 0, face: 1, facing: 1, closed: 0, hidden: 0, phone: 0, mouth: 0, ear: 0,
    belt: -1, perclos: 0, hand: "", alert: "", ev: [], driver: "" };
  let cur = Object.assign({}, NEUTRAL), lastTs = 0, lastRecv = 0, raf = 0, alive = true;

/*CORE*/

  const chips = root.querySelector(".tw-chips"), drv = root.querySelector(".tw-driver"),
        alertEl = root.querySelector(".tw-alert"), stale = root.querySelector(".tw-stale");
  function chip(text, cls) { return `<span class="tw-chip ${cls}">${text}</span>`; }
  function panel(f) {
    drv.textContent = f.driver || (f.face ? "Identifying driver…" : "No driver");
    const eyes = !f.face ? chip("Eyes n/a", "na") : f.hidden ? chip("Eyes hidden", "na") : !f.facing ? chip("Head turned", "warn")
               : f.closed ? chip("Eyes closed", "bad") : chip("Eyes open", "ok");
    chips.innerHTML = eyes
      + (f.ear ? chip("Hand at ear", "warn") : f.mouth ? chip("Hand at mouth", "warn") : "")
      + (f.phone ? chip("Phone", "bad") : "")
      + (f.belt === 1 ? chip("Belt on", "ok") : f.belt === 0 ? chip("Belt off", "bad") : chip("Belt checking", "na"))
      + chip("PERCLOS " + Math.round((f.perclos || 0) * 100) + "%", f.perclos > 0.25 ? "bad" : f.perclos > 0.15 ? "warn" : "ok");
    alertEl.hidden = !f.alert; alertEl.textContent = f.alert || "";
  }
  panel(cur);
  let prev = performance.now();
  function loop(now) {
    if (!alive) return;
    const dt = Math.min(0.1, (now - prev) / 1000); prev = now;
    apply(cur, dt);
    const age = lastRecv ? (Date.now() - lastRecv) / 1000 : 0;
    stale.hidden = !lastRecv || age < 3; if (!stale.hidden) stale.querySelector("b").textContent = Math.round(age);
    controls.update(); renderer.render(scene, cam);
    raf = requestAnimationFrame(loop);
  }
  raf = requestAnimationFrame(loop);
  return {
    push(f, ts) {
      if (ts && ts < lastTs) return;               // ignore out-of-order frames
      lastTs = ts || lastTs; lastRecv = Date.now();
      cur = Object.assign({}, NEUTRAL, f);
      for (const k of ["lean", "lE", "lW", "rE", "rW"]) if (typeof cur[k] === "string") { try { cur[k] = JSON.parse(cur[k]); } catch (e) { cur[k] = NEUTRAL[k]; } }
      if (!Array.isArray(cur.ev)) cur.ev = [];
      panel(cur);
    },
    resize,
    destroy() { alive = false; cancelAnimationFrame(raf); ro.disconnect(); controls.dispose(); renderer.dispose(); renderer.domElement.remove(); },
  };
}
""".replace("/*CORE*/", core)

widget = {
    "fqn": "cabin_driver_twin",
    "name": "Cabin driver twin",
    "deprecated": False,
    "image": None,
    "description": "3-D driver in a bus cabin, driven by the DMS 'twin' telemetry key (head, eyes, mouth, arms, phone, seat belt, alerts).",
    "descriptor": {
        "type": "latest",
        "sizeX": 8,
        "sizeY": 7,
        "resources": [
            {"url": "https://cdn.jsdelivr.net/npm/three@0.147.0/build/three.min.js"},
            {"url": "https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/controls/OrbitControls.js"},
        ],
        "templateHtml": TEMPLATE_HTML,
        "templateCss": TEMPLATE_CSS,
        "controllerScript": CONTROLLER,
        "settingsSchema": "",
        "dataKeySettingsSchema": "",
        "defaultConfig": json.dumps({
            "datasources": [{"type": "function", "name": "function",
                             "dataKeys": [{"name": "twin", "type": "timeseries", "label": "twin", "color": "#c27a00",
                                           "settings": {}, "_hash": 0.5}]}],
            "timewindow": {"realtime": {"timewindowMs": 60000}},
            "showTitle": True, "backgroundColor": "#ffffff", "color": "rgba(0, 0, 0, 0.87)",
            "padding": "0px", "settings": {}, "title": "Driver twin", "dropShadow": True,
        }),
    },
    "tags": ["dms", "driver", "3d", "fleet"],
}

os.makedirs(OUT, exist_ok=True)
json.dump(widget, open(os.path.join(OUT, "cabin_twin_widget.json"), "w"), indent=2)
open(os.path.join(OUT, "widget.html"), "w").write(TEMPLATE_HTML)
open(os.path.join(OUT, "widget.css"), "w").write(TEMPLATE_CSS)
open(os.path.join(OUT, "widget.js"), "w").write(CONTROLLER)
print("wrote", OUT)
