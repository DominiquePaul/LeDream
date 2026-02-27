#!/usr/bin/env python3
"""Web-based camera tuner — adjust V4L2 settings, take snapshots, compare."""

import argparse
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, request

# ---------------------------------------------------------------------------
# V4L2 helpers
# ---------------------------------------------------------------------------

def set_v4l2(device: str, control: str, value: int):
    subprocess.run(
        ["v4l2-ctl", "-d", device, f"--set-ctrl={control}={value}"],
        capture_output=True, timeout=2,
    )


def gray_world(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    b, g, r = cv2.split(f)
    avg = (r.mean() + g.mean() + b.mean()) / 3
    return np.clip(
        cv2.merge([b * avg / b.mean(), g * avg / g.mean(), r * avg / r.mean()]),
        0, 255,
    ).astype(np.uint8)


def swap_crcb(img: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 1], ycrcb[:, :, 2] = ycrcb[:, :, 2].copy(), ycrcb[:, :, 1].copy()
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

CONTROLS = [
    ("Brightness",  -64,   64,    0, "brightness"),
    ("Contrast",      0,   64,   32, "contrast"),
    ("Saturation",    0,  100,   40, "saturation"),
    ("Hue",         -40,   40,   40, "hue"),
    ("Gamma",       100,  500,  100, "gamma"),
    ("WB Temp",    2800, 6500, 4600, "white_balance_temperature"),
    ("Gain",          0,  100,    0, "gain"),
    ("Sharpness",     0,    6,    3, "sharpness"),
    ("Exposure",      1,  500,  157, "exposure_time_absolute"),
]

state = {
    "gray_world": True,
    "swap_crcb": {},        # per-camera: name -> bool
    "values": {v4l2: default for _, _, _, default, v4l2 in CONTROLS},
}
state_lock = threading.Lock()

cameras: dict[str, str] = {}
resolved: dict[str, str] = {}
caps: dict[str, cv2.VideoCapture] = {}
swap_crcb_defaults: set[str] = set()

_apply_queue: queue.Queue = queue.Queue()

# Snapshot storage
_snap_lock = threading.Lock()
_snap_counter = 0
_snap_images: dict[int, bytes] = {}
_snap_meta: list[dict] = []
_capture_lock = threading.Lock()

# ---------------------------------------------------------------------------
# V4L2 apply worker — serialises all v4l2-ctl writes so every camera sees
# the same (latest) value for each control.
# ---------------------------------------------------------------------------

_RESET_SENTINEL = "__reset__"

def _apply_worker():
    while True:
        item = _apply_queue.get()
        pending: dict[str, int] = {}
        reset = False

        if item == _RESET_SENTINEL:
            reset = True
        else:
            pending[item[0]] = item[1]

        while not _apply_queue.empty():
            try:
                nxt = _apply_queue.get_nowait()
            except queue.Empty:
                break
            if nxt == _RESET_SENTINEL:
                reset = True
                pending.clear()
            else:
                pending[nxt[0]] = nxt[1]

        if reset:
            for dev in resolved.values():
                set_v4l2(dev, "white_balance_automatic", 0)
                set_v4l2(dev, "auto_exposure", 3)
            for _, _mn, _mx, default, v4l2 in CONTROLS:
                for dev in resolved.values():
                    set_v4l2(dev, v4l2, default)
        else:
            for control, value in pending.items():
                for dev in resolved.values():
                    if control == "exposure_time_absolute":
                        set_v4l2(dev, "auto_exposure", 1)
                    if control == "white_balance_temperature":
                        set_v4l2(dev, "white_balance_automatic", 0)
                    set_v4l2(dev, control, value)

# ---------------------------------------------------------------------------
# Buffer drain — keeps camera buffers fresh in the background so snapshots
# are not stale.
# ---------------------------------------------------------------------------

def _drain_loop():
    while True:
        with _capture_lock:
            for cap in caps.values():
                cap.grab()
        time.sleep(0.05)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Camera Tuner</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,system-ui,sans-serif;background:#1a1a2e;color:#eee;padding:16px}

  .top-bar{display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap}
  .cam-btns{display:flex;gap:6px}
  .btn{padding:7px 18px;border:1px solid #555;border-radius:6px;
       background:#2a2a4a;cursor:pointer;font-size:13px;color:#ccc;white-space:nowrap}
  .btn:hover{border-color:#888}
  .btn.active{background:#4a3f8a;border-color:#7b68ee;color:#fff}
  .snap-btn{padding:9px 26px;border:none;border-radius:8px;background:#7b68ee;
            color:#fff;font-size:14px;font-weight:600;cursor:pointer;margin-left:auto}
  .snap-btn:hover{background:#6a5acd}
  .snap-btn:active{transform:scale(.97)}

  .main-view{text-align:center;margin-bottom:14px}
  .main-view img{max-width:640px;width:100%;border-radius:8px;display:block;margin:0 auto}
  .placeholder{width:640px;max-width:100%;aspect-ratio:4/3;background:#2a2a4a;border-radius:8px;
               display:flex;align-items:center;justify-content:center;color:#555;font-size:16px;margin:0 auto}
  .stats-line{font-family:monospace;font-size:13px;color:#999;text-align:center;margin-top:6px}

  .toggles{display:flex;gap:10px;justify-content:center;margin-bottom:10px}
  .controls{max-width:700px;margin:0 auto 14px}
  .slider-row{display:flex;align-items:center;gap:10px;margin:5px 0}
  .slider-row label{width:90px;text-align:right;font-size:13px;flex-shrink:0}
  .slider-row input[type=range]{flex:1;accent-color:#7b68ee}
  .slider-row .val{width:50px;text-align:right;font-family:monospace;font-size:13px}

  .history-label{font-size:13px;color:#666;margin-bottom:6px}
  .history{display:flex;gap:8px;overflow-x:auto;padding-bottom:8px}
  .history-item{flex-shrink:0;cursor:pointer;position:relative;border-radius:6px;
                border:2px solid transparent;transition:border-color .15s}
  .history-item:hover{border-color:#7b68ee}
  .history-item.sel{border-color:#7b68ee}
  .history-item img{width:180px;border-radius:4px;display:block}
  .history-item .tag{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.7);
                     padding:2px 6px;border-radius:0 0 4px 4px;font-size:11px;font-family:monospace}
  .history-item .del{position:absolute;top:2px;right:4px;background:rgba(0,0,0,.6);
                     border:none;color:#aaa;font-size:14px;cursor:pointer;border-radius:3px;
                     padding:0 4px;display:none}
  .history-item:hover .del{display:block}
  .history-item .del:hover{color:#ff6b6b}
</style>
</head><body>

<div class="top-bar">
  <div class="cam-btns" id="cam-btns"></div>
  <div class="toggles">
    <button class="btn active" id="btn-gw" onclick="toggle('gray_world')">Gray World</button>
    <button class="btn" id="btn-swap" onclick="toggle('swap_crcb')">Swap Cr/Cb</button>
    <button class="btn" onclick="resetAll()">Reset</button>
  </div>
  <button class="snap-btn" onclick="takeSnap()">&#128247; Take Picture</button>
</div>

<div class="main-view">
  <div class="placeholder" id="ph">Take a picture to start</div>
  <img id="main-img" style="display:none" alt="snapshot">
  <div class="stats-line" id="main-stats"></div>
</div>

<div class="controls" id="controls"></div>

<div class="history-label" id="hist-label" style="display:none">Previous snapshots</div>
<div class="history" id="history"></div>

<script>
const CAMERAS = CAMERA_NAMES_JSON;
const CONTROLS = CONTROLS_JSON;
let selectedCam = CAMERAS[0];
let snaps = [];
let selId = null;
let swapState = SWAP_DEFAULTS_JSON;

const camBtns = document.getElementById('cam-btns');
CAMERAS.forEach(name => {
  const b = document.createElement('button');
  b.className = 'btn' + (name === selectedCam ? ' active' : '');
  b.textContent = name;
  b.onclick = () => { selectedCam = name;
    camBtns.querySelectorAll('.btn').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    updateSwapBtn(); };
  camBtns.appendChild(b);
});

function updateSwapBtn() {
  document.getElementById('btn-swap').classList.toggle('active', !!swapState[selectedCam]);
}

const ctrlEl = document.getElementById('controls');
CONTROLS.forEach(([label, mn, mx, def, v4l2]) => {
  const row = document.createElement('div');
  row.className = 'slider-row';
  row.innerHTML = `<label>${label}</label>
    <input type="range" min="${mn}" max="${mx}" value="${def}" step="1"
           id="sl-${v4l2}" oninput="sliderChange('${v4l2}',this.value)">
    <span class="val" id="val-${v4l2}">${def}</span>`;
  ctrlEl.appendChild(row);
});

function sliderChange(v, val) {
  document.getElementById('val-'+v).textContent = val;
  fetch('/set?control='+v+'&value='+val);
}
function toggle(w) {
  const url = w==='swap_crcb' ? '/toggle/'+w+'?cam='+selectedCam : '/toggle/'+w;
  fetch(url).then(r=>r.json()).then(d=>{
    document.getElementById('btn-gw').classList.toggle('active',d.gray_world);
    swapState = d.swap_crcb;
    updateSwapBtn();
  });
}
function resetAll() {
  fetch('/reset').then(r=>r.json()).then(d=>{
    CONTROLS.forEach(([,,, def, v4l2])=>{
      document.getElementById('sl-'+v4l2).value=def;
      document.getElementById('val-'+v4l2).textContent=def;
    });
    swapState = d.swap_crcb;
    updateSwapBtn();
  });
}

function takeSnap() {
  fetch('/snap?cam='+selectedCam).then(r=>r.json()).then(m=>{
    snaps.unshift(m);
    show(m);
    renderHist();
  });
}

function show(m) {
  selId = m.id;
  const img = document.getElementById('main-img');
  img.src = '/snap/'+m.id+'.jpg?'+Date.now();
  img.style.display = 'block';
  document.getElementById('ph').style.display = 'none';
  const s = m.stats;
  document.getElementById('main-stats').textContent =
    m.cam+' @ '+m.time+'  \u2014  R'+s.r+'  G'+s.g+'  B'+s.b;
  document.querySelectorAll('.history-item').forEach(el=>{
    el.classList.toggle('sel', +el.dataset.id===m.id);
  });
}

function deleteSnap(id, ev) {
  ev.stopPropagation();
  fetch('/snap/'+id, {method:'DELETE'});
  snaps = snaps.filter(s=>s.id!==id);
  if (selId===id) {
    if (snaps.length) show(snaps[0]);
    else { document.getElementById('main-img').style.display='none';
           document.getElementById('ph').style.display='flex';
           document.getElementById('main-stats').textContent=''; selId=null; }
  }
  renderHist();
}

function renderHist() {
  const el = document.getElementById('history');
  document.getElementById('hist-label').style.display = snaps.length ? '' : 'none';
  el.innerHTML = '';
  snaps.forEach(m=>{
    const d = document.createElement('div');
    d.className = 'history-item'+(m.id===selId?' sel':'');
    d.dataset.id = m.id;
    d.onclick = ()=>show(m);
    d.innerHTML = `<img src="/snap/${m.id}.jpg" alt="#${m.id}">
      <div class="tag">${m.cam} ${m.time}</div>
      <button class="del" onclick="deleteSnap(${m.id},event)">&times;</button>`;
    el.appendChild(d);
  });
}
updateSwapBtn();
</script>
</body></html>"""


@app.route("/")
def index():
    cam_names = json.dumps(list(cameras.keys()))
    ctrl_json = json.dumps(CONTROLS)
    with state_lock:
        swap_json = json.dumps(state["swap_crcb"])
    html = (HTML
            .replace("CAMERA_NAMES_JSON", cam_names)
            .replace("CONTROLS_JSON", ctrl_json)
            .replace("SWAP_DEFAULTS_JSON", swap_json))
    return html


@app.route("/snap")
def take_snap():
    global _snap_counter
    cam_name = request.args.get("cam", list(cameras.keys())[0])
    cap = caps.get(cam_name)
    if not cap:
        return json.dumps({"error": "unknown camera"}), 404

    with _capture_lock:
        for _ in range(3):
            cap.grab()
        ret, frame = cap.read()

    if not ret:
        return json.dumps({"error": "capture failed"}), 500

    with state_lock:
        do_gw = state["gray_world"]
        do_swap = state["swap_crcb"].get(cam_name, False)
        current_settings = dict(state["values"])

    if do_swap:
        frame = swap_crcb(frame)
    if do_gw:
        frame = gray_world(frame)

    b_avg = frame[:, :, 0].astype(float).mean()
    g_avg = frame[:, :, 1].astype(float).mean()
    r_avg = frame[:, :, 2].astype(float).mean()

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

    with _snap_lock:
        snap_id = _snap_counter
        _snap_counter += 1
        meta = {
            "id": snap_id,
            "cam": cam_name,
            "time": time.strftime("%H:%M:%S"),
            "settings": current_settings,
            "gray_world": do_gw,
            "swap_crcb": do_swap,
            "stats": {"r": round(r_avg, 1), "g": round(g_avg, 1), "b": round(b_avg, 1)},
        }
        _snap_meta.append(meta)
        _snap_images[snap_id] = jpeg.tobytes()

    return json.dumps(meta)


@app.route("/snap/<int:snap_id>.jpg")
def get_snap_image(snap_id):
    img = _snap_images.get(snap_id)
    if img is None:
        return "not found", 404
    return Response(img, mimetype="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.route("/snap/<int:snap_id>", methods=["DELETE"])
def delete_snap(snap_id):
    with _snap_lock:
        _snap_images.pop(snap_id, None)
        _snap_meta[:] = [m for m in _snap_meta if m["id"] != snap_id]
    return "ok"


@app.route("/set")
def set_control():
    control = request.args.get("control")
    value = int(request.args.get("value"))
    with state_lock:
        state["values"][control] = value
    _apply_queue.put((control, value))
    return "ok"


@app.route("/toggle/<what>")
def toggle(what):
    cam = request.args.get("cam")
    with state_lock:
        if what == "swap_crcb" and cam:
            state["swap_crcb"][cam] = not state["swap_crcb"].get(cam, False)
        elif what == "gray_world":
            state["gray_world"] = not state["gray_world"]
        return json.dumps({"gray_world": state["gray_world"],
                           "swap_crcb": state["swap_crcb"]})


@app.route("/reset")
def reset():
    with state_lock:
        for _, mn, mx, default, v4l2 in CONTROLS:
            state["values"][v4l2] = default
        state["swap_crcb"] = {n: (n in swap_crcb_defaults) for n in cameras}
    _apply_queue.put(_RESET_SENTINEL)
    return json.dumps({"swap_crcb": state["swap_crcb"]})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global cameras, resolved, caps, swap_crcb_defaults

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", required=True, help="Comma-separated name=path pairs")
    parser.add_argument("--swap-crcb", default="",
                        help="Comma-separated camera names that need Cr/Cb swap (e.g. 'top')")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    for pair in args.cameras.split(","):
        name, path = pair.split("=", 1)
        cameras[name.strip()] = path.strip()

    if args.swap_crcb:
        swap_crcb_defaults = {n.strip() for n in args.swap_crcb.split(",")}
    state["swap_crcb"] = {n: (n in swap_crcb_defaults) for n in cameras}

    resolved = {n: str(Path(p).resolve()) for n, p in cameras.items()}

    for dev in resolved.values():
        set_v4l2(dev, "white_balance_automatic", 0)
        set_v4l2(dev, "auto_exposure", 3)

    for name, path in cameras.items():
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        caps[name] = cap

    threading.Thread(target=_drain_loop, daemon=True).start()
    threading.Thread(target=_apply_worker, daemon=True).start()

    print(f"\n  Camera tuner running at  http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
