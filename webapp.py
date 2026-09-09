"""
webapp.py

Riposte's local web page: pick a clip, scrub to the action, drag a box
round each fencer, press Run, watch the result.

    python3 webapp.py

Then open http://127.0.0.1:8765 in your browser.

WHAT THIS IS AND IS NOT
This is a small server running on your own computer. Nothing is uploaded
anywhere, there is no account, no cloud and no database. It binds to
127.0.0.1, which
means only this computer can reach it, not other machines on the wifi.

The browser cannot do the tracking itself, because the tracker is Python
and OpenCV. So the page is a front end and the work happens in exactly
the same `main.py` you run from Terminal. That is deliberate: the page
and the command line run the SAME code with the SAME defaults, so a
result you get here is a result you can reproduce there, and improving
one improves both.

NO NEW DEPENDENCIES
Python's own http.server, nothing else. The project still installs with
two pip packages.

TWO WAYS TO LOAD A CLIP
* Drop a file on the page, which copies it into a working folder.
* Or paste the file's path, which copies nothing at all.
Use the path for big clips. Competition footage runs to 300MB and
copying it achieves nothing when it is already sitting on this disk.
"""

import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(HERE, "webruns")
PYTHON = sys.executable
HOST, PORT = "127.0.0.1", 8765

# job id -> dict describing an upload and any run against it
JOBS = {}
JOBS_LOCK = threading.Lock()


# ----------------------------------------------------------------------
# Running the tracker
# ----------------------------------------------------------------------

def start_run(job_id, box_a, box_b, start_frame, max_frames, use_detect,
              use_pose):
    """
    Launch main.py as a separate process and follow its progress.

    Shelling out rather than importing keeps ONE implementation of the
    tracking loop. The alternative, a second code path for the web,
    is how a project ends up with a page and a terminal that quietly
    disagree about what the answer is.
    """
    job = JOBS[job_id]
    out_dir = os.path.join(job["dir"], "output")
    os.makedirs(out_dir, exist_ok=True)

    command = [
        PYTHON, os.path.join(HERE, "main.py"), job["video"],
        "--box-a", ",".join(str(int(v)) for v in box_a),
        "--box-b", ",".join(str(int(v)) for v in box_b),
        "--out-dir", out_dir,
        "--no-display", "--progress",
        "--start-frame", str(int(start_frame)),
    ]
    if max_frames:
        command += ["--max-frames", str(int(max_frames))]
    if use_detect:
        command.append("--detect")
    if use_pose:
        command.append("--pose")

    job.update(state="running", done=0, total=max_frames or 0,
               log=[], events=[], error=None,
               video_out=None, csv_out=None, started=time.time())

    def worker():
        try:
            process = subprocess.Popen(
                command, cwd=HERE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                line = line.rstrip()
                if line.startswith("PROGRESS "):
                    _, done, total = line.split()
                    with JOBS_LOCK:
                        job["done"], job["total"] = int(done), int(total)
                    continue
                # Keep the messages worth reading, drop the codec noise.
                if line.startswith("[ INFO") or "x265" in line or \
                        line.startswith("[ WARN") or not line.strip():
                    continue
                with JOBS_LOCK:
                    job["log"].append(line)
                    if "LOST" in line or "WARNING" in line or \
                            "re-attached" in line:
                        job["events"].append(line)
            process.wait()

            base = os.path.splitext(os.path.basename(job["video"]))[0]
            video_out = os.path.join(out_dir, f"{base}_tracked.mp4")
            csv_out = os.path.join(out_dir, f"{base}_tracking.csv")
            with JOBS_LOCK:
                if process.returncode != 0:
                    job["state"] = "error"
                    job["error"] = "\n".join(job["log"][-8:]) or \
                        f"main.py exited with code {process.returncode}"
                elif not os.path.exists(video_out):
                    job["state"] = "error"
                    job["error"] = "No output video was produced."
                else:
                    job["state"] = "done"
                    job["video_out"] = video_out
                    job["csv_out"] = csv_out if os.path.exists(csv_out) else None
                    job["seconds"] = time.time() - job["started"]
        except Exception as exc:                       # noqa: BLE001
            with JOBS_LOCK:
                job["state"] = "error"
                job["error"] = f"{type(exc).__name__}: {exc}"

    threading.Thread(target=worker, daemon=True).start()


# ----------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "Riposte/1.0"

    def log_message(self, fmt, *args):
        pass                                   # keep the terminal readable

    def handle_one_request(self):
        """Same reason as the catch in send_file_ranged: a browser that
        walks away mid-request is normal, not an error worth printing."""
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    # -- helpers -------------------------------------------------------

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, data, content_type, status=200, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def send_file_ranged(self, path, content_type, download_name=None):
        """
        Serve a file, honouring Range requests.

        Video needs this: without it the browser will play the clip but
        refuse to let you scrub, because it cannot ask for the middle of
        the file.
        """
        size = os.path.getsize(path)
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                end = min(end, size - 1)
                status = 206

        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download_name:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{download_name}"')
        self.end_headers()
        try:
            with open(path, "rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # The browser hung up mid-stream. This is completely normal:
            # it happens every time you scrub the result video, because
            # the player abandons the current range request and asks for
            # a different byte range instead. Without this catch the
            # terminal fills with tracebacks that look like crashes and
            # are not.
            pass

    def job_from_query(self, query):
        job_id = (query.get("job") or [""])[0]
        return JOBS.get(job_id)

    # -- GET -----------------------------------------------------------

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        route = parsed.path

        if route == "/":
            self.send_bytes(PAGE.encode(), "text/html; charset=utf-8")
            return

        if route == "/api/frame":
            job = self.job_from_query(query)
            if not job:
                self.send_json({"error": "unknown job"}, 404)
                return
            index = int((query.get("index") or ["0"])[0])
            from video_io import read_frame_at
            frame = read_frame_at(job["video"], index)
            if frame is None:
                self.send_json({"error": "no frame there"}, 404)
                return
            ok, buffer = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                self.send_json({"error": "could not encode frame"}, 500)
                return
            self.send_bytes(buffer.tobytes(), "image/jpeg",
                            extra={"Cache-Control": "no-store"})
            return

        if route == "/api/status":
            job = self.job_from_query(query)
            if not job:
                self.send_json({"error": "unknown job"}, 404)
                return
            with JOBS_LOCK:
                self.send_json({
                    "state": job.get("state", "ready"),
                    "done": job.get("done", 0),
                    "total": job.get("total", 0),
                    "events": job.get("events", [])[-12:],
                    "log": job.get("log", [])[-14:],
                    "error": job.get("error"),
                    "seconds": round(job.get("seconds", 0), 1),
                    "has_csv": bool(job.get("csv_out")),
                })
            return

        if route in ("/api/video", "/api/csv"):
            job = self.job_from_query(query)
            if not job:
                self.send_json({"error": "unknown job"}, 404)
                return
            if route == "/api/video":
                path = job.get("video_out")
                if not path or not os.path.exists(path):
                    self.send_json({"error": "no video yet"}, 404)
                    return
                self.send_file_ranged(path, "video/mp4")
            else:
                path = job.get("csv_out")
                if not path or not os.path.exists(path):
                    self.send_json({"error": "no csv"}, 404)
                    return
                self.send_file_ranged(path, "text/csv",
                                      download_name=os.path.basename(path))
            return

        self.send_json({"error": "not found"}, 404)

    # -- POST ----------------------------------------------------------

    def do_POST(self):
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/upload":
            query = parse_qs(parsed.query)
            name = (query.get("name") or ["clip.mov"])[0]
            name = os.path.basename(name) or "clip.mov"
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                self.send_json({"error": "empty upload"}, 400)
                return
            job_id = uuid.uuid4().hex[:12]
            job_dir = os.path.join(WORK_DIR, job_id)
            os.makedirs(job_dir, exist_ok=True)
            target = os.path.join(job_dir, name)
            remaining = length
            with open(target, "wb") as handle:
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    handle.write(chunk)
                    remaining -= len(chunk)
            self.finish_job(job_id, job_dir, target)
            return

        if route == "/api/use-path":
            body = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length", 0))) or b"{}")
            raw = (body.get("path") or "").strip()
            # People paste paths with quotes round them, because that is
            # what Terminal needs. Strip them rather than failing.
            raw = raw.strip('"').strip("'")
            path = os.path.abspath(os.path.expanduser(raw))
            if not os.path.isfile(path):
                self.send_json({"error": f"No file at {path}"}, 400)
                return
            job_id = uuid.uuid4().hex[:12]
            job_dir = os.path.join(WORK_DIR, job_id)
            os.makedirs(job_dir, exist_ok=True)
            self.finish_job(job_id, job_dir, path)
            return

        if route == "/api/run":
            body = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length", 0))) or b"{}")
            job = JOBS.get(body.get("job"))
            if not job:
                self.send_json({"error": "unknown job"}, 404)
                return
            try:
                start_run(job["id"], body["boxA"], body["boxB"],
                          body.get("startFrame", 0), body.get("maxFrames", 0),
                          bool(body.get("detect", True)),
                          bool(body.get("pose", False)))
            except Exception as exc:                   # noqa: BLE001
                self.send_json({"error": str(exc)}, 500)
                return
            self.send_json({"ok": True})
            return

        self.send_json({"error": "not found"}, 404)

    def finish_job(self, job_id, job_dir, video_path):
        """Read the clip's basics and register the job."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError("OpenCV could not open that video.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        except Exception as exc:                       # noqa: BLE001
            self.send_json({"error": str(exc)}, 400)
            return

        JOBS[job_id] = {
            "id": job_id, "dir": job_dir, "video": video_path,
            "state": "ready", "log": [], "events": [],
        }
        self.send_json({
            "job": job_id,
            "name": os.path.basename(video_path),
            "width": width, "height": height,
            # The header frame count is often wrong (measured: one clip
            # claims 127 and really has 120). Good enough for a slider.
            "frames": max(frames, 1),
            "fps": round(fps, 2),
        })


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Riposte</title>
<style>
:root{--bg:#12141a;--panel:#1b1e26;--line:#2c313d;--ink:#e7e9ee;--dim:#9aa1b1;
      --a:#ff8c00;--b:#0a5aff;--ok:#35c26b;--bad:#ff5a5a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:17px;font-weight:600}
header p{margin:4px 0 0;color:var(--dim);font-size:13px}
main{max-width:1180px;margin:0 auto;padding:24px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      padding:18px;margin-bottom:18px}
.card h2{margin:0 0 12px;font-size:13px;text-transform:uppercase;
         letter-spacing:.08em;color:var(--dim);font-weight:600}
.drop{border:2px dashed var(--line);border-radius:10px;padding:34px;
      text-align:center;color:var(--dim);cursor:pointer}
.drop.over{border-color:var(--a);color:var(--ink)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input[type=text]{flex:1;min-width:280px;background:#0e1016;color:var(--ink);
   border:1px solid var(--line);border-radius:7px;padding:9px 11px;font-size:13px}
button{background:#2a2f3a;color:var(--ink);border:1px solid var(--line);
   border-radius:7px;padding:9px 15px;font-size:13px;cursor:pointer}
button:hover:not(:disabled){background:#343a48}
button.primary{background:var(--ok);border-color:var(--ok);color:#06240f;font-weight:600}
button.primary:disabled{opacity:.4;cursor:not-allowed}
button.sel{border-color:var(--a);color:var(--a)}
button.sel.b{border-color:#6aa1ff;color:#6aa1ff}
canvas{max-width:100%;border-radius:8px;display:block;cursor:crosshair;
       background:#000}
.hint{color:var(--dim);font-size:13px;margin:10px 0 0}
.chip{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;
      border:1px solid var(--line);color:var(--dim)}
.chip.set{border-color:var(--ok);color:var(--ok)}
.bar{height:8px;background:#0e1016;border-radius:999px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--ok);width:0;transition:width .2s}
pre{background:#0e1016;border:1px solid var(--line);border-radius:7px;
    padding:11px;font-size:12px;max-height:190px;overflow:auto;margin:10px 0 0;
    white-space:pre-wrap}
.ev{color:#ffcf70}
.err{color:var(--bad)}
label.opt{display:inline-flex;gap:7px;align-items:center;cursor:pointer;
          margin-right:18px}
video{width:100%;border-radius:8px;background:#000}
.hidden{display:none}
input[type=range]{flex:1;min-width:220px}
</style></head><body>
<header>
  <h1>Riposte</h1>
  <p>Runs on your own computer. Nothing is uploaded anywhere. Same code as the terminal.</p>
</header>
<main>

<div class="card" id="loadCard">
  <h2>1 &nbsp;Choose a clip</h2>
  <div class="drop" id="drop">Drop a video here, or click to pick one</div>
  <input type="file" id="file" accept="video/*" class="hidden">
  <p class="hint">Big clip already on this Mac? Paste its path instead.
     Nothing gets copied. In Finder, right&#8209;click the file and hold
     Option, then &ldquo;Copy as Pathname&rdquo;.</p>
  <div class="row">
    <input type="text" id="path" placeholder="/Users/you/Movies/bout.MOV">
    <button id="usePath">Use this path</button>
  </div>
  <p class="hint err hidden" id="loadErr"></p>
</div>

<div class="card hidden" id="pickCard">
  <h2>2 &nbsp;Find the action, then box each fencer</h2>
  <div class="row" style="margin-bottom:10px">
    <button id="back30">&minus;30</button>
    <button id="back5">&minus;5</button>
    <input type="range" id="scrub" min="0" value="0">
    <button id="fwd5">+5</button>
    <button id="fwd30">+30</button>
    <span class="chip" id="frameLabel">frame 0</span>
  </div>
  <canvas id="cv"></canvas>
  <p class="hint">
    Drag a box around each fencer&rsquo;s <b>mask and torso only</b>, not
    the legs. Measured on your own footage: a full&#8209;body box drifted onto
    the referee after 49 frames; a torso box followed the fencer correctly.
  </p>
  <div class="row" style="margin-top:10px">
    <button id="drawA" class="sel">Draw Fencer A</button>
    <button id="drawB" class="sel b">Draw Fencer B</button>
    <span class="chip" id="chipA">A not set</span>
    <span class="chip" id="chipB">B not set</span>
    <button id="clearBoxes">Clear</button>
  </div>
  <div class="row" style="margin-top:14px">
    <label class="opt"><input type="checkbox" id="optDetect" checked>
      Person detection <span class="chip">recommended</span></label>
    <label class="opt"><input type="checkbox" id="optPose"> Body landmarks</label>
    <label class="opt">Frames to process
      <input type="text" id="optMax" value="150" style="width:80px;min-width:0"></label>
  </div>
  <div class="row" style="margin-top:14px">
    <button class="primary" id="run" disabled>Run tracking</button>
    <span class="hint" id="runHint">Box both fencers first.</span>
  </div>
</div>

<div class="card hidden" id="runCard">
  <h2>3 &nbsp;Running</h2>
  <div class="bar"><i id="barFill"></i></div>
  <p class="hint" id="progText">Starting&hellip;</p>
  <pre id="events" class="hidden"></pre>
</div>

<div class="card hidden" id="doneCard">
  <h2>4 &nbsp;Result</h2>
  <video id="out" controls playsinline></video>
  <div class="row" style="margin-top:12px">
    <a id="dlVideo" download><button>Download video</button></a>
    <a id="dlCsv" download><button>Download CSV</button></a>
    <button id="again">Track another</button>
  </div>
  <pre id="finalLog"></pre>
</div>

</main>
<script>
const $ = id => document.getElementById(id);
let job=null, meta=null, frame=0, img=new Image();
let boxes={A:null,B:null}, drawing=null, startPt=null, curBox=null, scale=1;

function show(id,on){ $(id).classList.toggle('hidden',!on); }

/* ---------- loading a clip ---------- */
$('drop').onclick = ()=> $('file').click();
$('drop').ondragover = e=>{e.preventDefault(); $('drop').classList.add('over');};
$('drop').ondragleave = ()=> $('drop').classList.remove('over');
$('drop').ondrop = e=>{
  e.preventDefault(); $('drop').classList.remove('over');
  if(e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
};
$('file').onchange = e=>{ if(e.target.files[0]) upload(e.target.files[0]); };

async function upload(file){
  $('drop').textContent = 'Copying '+file.name+' …';
  try{
    const r = await fetch('/api/upload?name='+encodeURIComponent(file.name),
                          {method:'POST', body:file});
    const d = await r.json();
    if(d.error) throw new Error(d.error);
    loaded(d);
  }catch(err){ loadError(err.message); $('drop').textContent='Drop a video here, or click to pick one'; }
}
$('usePath').onclick = async ()=>{
  const p = $('path').value.trim();
  if(!p) return;
  try{
    const r = await fetch('/api/use-path',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({path:p})});
    const d = await r.json();
    if(d.error) throw new Error(d.error);
    loaded(d);
  }catch(err){ loadError(err.message); }
};
function loadError(msg){ $('loadErr').textContent = msg; show('loadErr',true); }

function loaded(d){
  job=d.job; meta=d; show('loadErr',false);
  $('scrub').max = Math.max(0, d.frames-1);
  $('scrub').value = 0; frame = 0;
  show('pickCard',true); show('runCard',false); show('doneCard',false);
  $('loadCard').querySelector('h2').textContent =
     '1  Clip: '+d.name+'  ('+d.width+'×'+d.height+', ~'+d.frames+' frames)';
  loadFrame();
}

/* ---------- scrubbing ---------- */
function setFrame(n){
  frame = Math.max(0, Math.min(parseInt($('scrub').max), n));
  $('scrub').value = frame; loadFrame();
}
$('scrub').oninput = e=> setFrame(parseInt(e.target.value));
$('back30').onclick=()=>setFrame(frame-30); $('back5').onclick=()=>setFrame(frame-5);
$('fwd5').onclick=()=>setFrame(frame+5);   $('fwd30').onclick=()=>setFrame(frame+30);

function loadFrame(){
  $('frameLabel').textContent='frame '+frame;
  img = new Image();
  img.onload = ()=>{
    const cv=$('cv'), maxW=Math.min(1080, document.querySelector('main').clientWidth-40);
    scale = Math.min(1, maxW/img.width);
    cv.width = Math.round(img.width*scale);
    cv.height = Math.round(img.height*scale);
    redraw();
  };
  img.src='/api/frame?job='+job+'&index='+frame+'&t='+Date.now();
}

/* ---------- drawing boxes ---------- */
function redraw(){
  const cv=$('cv'), g=cv.getContext('2d');
  g.clearRect(0,0,cv.width,cv.height);
  if(img.complete) g.drawImage(img,0,0,cv.width,cv.height);
  const paint=(b,col,lab)=>{
    if(!b) return;
    g.lineWidth=3; g.strokeStyle=col;
    g.strokeRect(b.x*scale,b.y*scale,b.w*scale,b.h*scale);
    g.fillStyle=col; g.font='bold 15px sans-serif';
    g.fillText(lab, b.x*scale+3, Math.max(14,b.y*scale-5));
  };
  paint(boxes.A,'#ff8c00','Fencer A'); paint(boxes.B,'#4d90ff','Fencer B');
  if(curBox) paint(curBox, drawing==='A'?'#ff8c00':'#4d90ff', drawing);
}
$('drawA').onclick=()=>{drawing='A'; hintDraw();};
$('drawB').onclick=()=>{drawing='B'; hintDraw();};
function hintDraw(){ $('runHint').textContent =
  drawing? ('Drag a box around Fencer '+drawing+"'s mask and torso.") : ''; }
$('clearBoxes').onclick=()=>{boxes={A:null,B:null}; curBox=null; refreshChips(); redraw();};

const pos = e=>{ const r=$('cv').getBoundingClientRect();
  return {x:(e.clientX-r.left)/scale, y:(e.clientY-r.top)/scale}; };
$('cv').onmousedown = e=>{ if(!drawing) drawing='A';
  startPt=pos(e); curBox={x:startPt.x,y:startPt.y,w:0,h:0}; };
$('cv').onmousemove = e=>{ if(!startPt) return;
  const p=pos(e);
  curBox={x:Math.min(p.x,startPt.x), y:Math.min(p.y,startPt.y),
          w:Math.abs(p.x-startPt.x), h:Math.abs(p.y-startPt.y)};
  redraw(); };
window.onmouseup = ()=>{
  if(!startPt) return;
  if(curBox && curBox.w>12 && curBox.h>12){
    boxes[drawing]={x:Math.round(curBox.x),y:Math.round(curBox.y),
                    w:Math.round(curBox.w),h:Math.round(curBox.h)};
    drawing = boxes.A && !boxes.B ? 'B' : (boxes.B && !boxes.A ? 'A' : null);
  }
  startPt=null; curBox=null; refreshChips(); redraw(); hintDraw();
};

function refreshChips(){
  const f=(b,el,n)=>{ const c=$(el);
    c.textContent = b? (n+' '+b.w+'×'+b.h) : (n+' not set');
    c.classList.toggle('set',!!b); };
  f(boxes.A,'chipA','A'); f(boxes.B,'chipB','B');
  const ready = boxes.A && boxes.B;
  $('run').disabled = !ready;
  if(ready) $('runHint').textContent='Ready.';
  else if(!drawing) $('runHint').textContent='Box both fencers first.';
}

/* ---------- running ---------- */
$('run').onclick = async ()=>{
  const body={job, boxA:[boxes.A.x,boxes.A.y,boxes.A.w,boxes.A.h],
              boxB:[boxes.B.x,boxes.B.y,boxes.B.w,boxes.B.h],
              startFrame:frame, maxFrames:parseInt($('optMax').value)||0,
              detect:$('optDetect').checked, pose:$('optPose').checked};
  show('runCard',true); show('doneCard',false);
  $('barFill').style.width='0%'; $('progText').textContent='Starting…';
  $('events').textContent=''; show('events',false);
  $('run').disabled=true;
  const r=await fetch('/api/run',{method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const d=await r.json();
  if(d.error){ $('progText').innerHTML='<span class="err">'+d.error+'</span>';
               $('run').disabled=false; return; }
  poll();
};

async function poll(){
  const r=await fetch('/api/status?job='+job); const s=await r.json();
  if(s.total>0){
    const pct=Math.round(100*s.done/s.total);
    $('barFill').style.width=pct+'%';
    $('progText').textContent='Frame '+s.done+' of '+s.total+'  ('+pct+'%)';
  }
  if(s.events && s.events.length){
    show('events',true);
    $('events').innerHTML = s.events.map(e=>'<span class="ev">'+
      e.replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))+'</span>').join('\n');
  }
  if(s.state==='done'){
    $('barFill').style.width='100%';
    $('progText').textContent='Done in '+s.seconds+'s.';
    $('out').src='/api/video?job='+job+'&t='+Date.now();
    $('dlVideo').href='/api/video?job='+job;
    $('dlCsv').href='/api/csv?job='+job;
    $('dlCsv').parentElement.style.display = s.has_csv? '' : 'none';
    $('finalLog').textContent=(s.log||[]).join('\n');
    show('doneCard',true); $('run').disabled=false;
    document.getElementById('doneCard').scrollIntoView({behavior:'smooth'});
    return;
  }
  if(s.state==='error'){
    $('progText').innerHTML='<span class="err">'+
      (s.error||'Something went wrong').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))+'</span>';
    $('run').disabled=false; return;
  }
  setTimeout(poll, 400);
}
$('again').onclick=()=>{ show('doneCard',false); window.scrollTo({top:0,behavior:'smooth'}); };
refreshChips();
</script></body></html>
"""


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    mimetypes.add_type("video/mp4", ".mp4")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Riposte is running.\n")
    print(f"    Open  http://{HOST}:{PORT}\n")
    print("Only this Mac can reach it. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
