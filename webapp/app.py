# -*- coding: utf-8 -*-
"""드론 VRA 처방맵 웹앱 (비개발자용)

- 브라우저에서 ① GNDVI tif(또는 zip) ② 바운더리 zip(선택) ③ vra.csv 를 업로드하면
  operation_main 파이프라인을 그대로 실행해 결과 zip 을 내려준다.
- Cloudflare 무료 터널(요청당 100MB 제한) 대응: 브라우저 JS가 파일을 64MB 조각으로
  잘라 /chunk(순차 append) → /finalize(작업 구성+실행) 로 전송한다.
- 동시 실행 방지 락, 1.5초 로그 폴링, VRA_NO_BROWSER=1 이면 브라우저 자동열림 억제.

실행: 처방맵_웹앱_실행.bat (로컬) / 처방맵_외부공개_실행.bat (Cloudflare Tunnel)
"""
import io
import os
import re
import sys
import glob
import json
import shutil
import zipfile
import datetime
import threading
import traceback
import webbrowser

import matplotlib
matplotlib.use("Agg")  # 백그라운드 스레드에서 그림 저장 (GUI 백엔드 금지)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../drone_vra_auto/webapp
ROOT_DIR = os.path.dirname(BASE_DIR)                           # .../drone_vra_auto
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
UPLOAD_DIR = os.path.join(JOBS_DIR, "_uploads")

sys.path.insert(0, ROOT_DIR)
import operation_main as om  # noqa: E402  (검증된 파이프라인 그대로 재사용)

from flask import Flask, request, jsonify, send_file, abort  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 조각(64MB)만 받으므로 여유값

# ------------------------------------------------------------------
# 작업 상태 관리
# ------------------------------------------------------------------
JOBS = {}                      # job_id -> {state, log, zip_path, images, output}
RUN_LOCK = threading.Lock()    # 동시 실행 방지 (파이프라인은 한 번에 하나만)
VALID_KINDS = ("data", "boundary", "vra")


def _safe_name(filename):
    """업로드 파일명 정리 — 경로 성분 제거, 위험 문자 치환 (한글 유지)."""
    name = os.path.basename(str(filename).replace("\\", "/"))
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()
    return name or "unnamed"


class _LogWriter(io.TextIOBase):
    """print() 출력을 작업 로그 리스트로 흘려보내는 writer."""

    def __init__(self, log_list):
        self.log = log_list
        self._buf = ""

    def write(self, s):
        self._buf += str(s)
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.log.append(line.rstrip())
        return len(s)

    def flush(self):
        pass


# ------------------------------------------------------------------
# 업로드 파일 배치 (finalize 단계)
# ------------------------------------------------------------------
def _place_data_uploads(src_folder, data_dir):
    """GNDVI 업로드 배치: tif는 그대로, zip이면 내부 tif를 풀어 넣는다."""
    for path in sorted(glob.glob(os.path.join(src_folder, "*"))):
        name = os.path.basename(path)
        low = name.lower()
        if low.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                for entry in zf.namelist():
                    if entry.endswith("/"):
                        continue
                    if entry.lower().endswith((".tif", ".tiff")):
                        out = os.path.join(data_dir, _safe_name(entry))
                        with zf.open(entry) as fsrc, open(out, "wb") as fdst:
                            shutil.copyfileobj(fsrc, fdst, 1024 * 1024)
        elif low.endswith((".tif", ".tiff")):
            os.replace(path, os.path.join(data_dir, name))


def _place_boundary_uploads(src_folder, boundary_dir):
    """바운더리 업로드 배치 — zip묶음 / 개별 zip / shp 세트 모두 허용.

    주의(실사용 버그 재발 방지): zip 핸들을 열어 둔 채 os.replace 하면
    Windows 에서 '파일 사용 중' 오류가 난다. 반드시 닫은 뒤 이동할 것.
    """
    for path in sorted(glob.glob(os.path.join(src_folder, "*"))):
        name = os.path.basename(path)
        low = name.lower()
        if low.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                entries = [e for e in zf.namelist() if not e.endswith("/")]
                inner_zips = [e for e in entries if e.lower().endswith(".zip")]
                has_shp = any(e.lower().endswith(".shp") for e in entries)
                if inner_zips:
                    # zip 묶음: 내부의 필지별 zip 들을 풀어 넣는다
                    for entry in inner_zips:
                        out = os.path.join(boundary_dir, _safe_name(entry))
                        with zf.open(entry) as fsrc, open(out, "wb") as fdst:
                            shutil.copyfileobj(fsrc, fdst, 1024 * 1024)
                    keep_as_zip = False
                elif has_shp:
                    # 개별 필지 zip (shp 세트 포함) → zip 그대로 사용
                    keep_as_zip = True
                else:
                    # 알 수 없는 구성 → 내용물을 평탄화해 풀어 넣는다
                    for entry in entries:
                        out = os.path.join(boundary_dir, _safe_name(entry))
                        with zf.open(entry) as fsrc, open(out, "wb") as fdst:
                            shutil.copyfileobj(fsrc, fdst, 1024 * 1024)
                    keep_as_zip = False
            # ↑ with 블록 종료 = zip 핸들 닫힘. 이제 이동해도 안전.
            if keep_as_zip:
                os.replace(path, os.path.join(boundary_dir, name))
        elif low.endswith((".shp", ".dbf", ".shx", ".prj", ".cpg", ".qpj")):
            os.replace(path, os.path.join(boundary_dir, name))


def _place_vra_upload(src_folder, job_dir):
    csvs = sorted(glob.glob(os.path.join(src_folder, "*.csv")))
    if not csvs:
        return None
    dst = os.path.join(job_dir, "vra.csv")
    os.replace(csvs[0], dst)
    return dst


# ------------------------------------------------------------------
# 파이프라인 실행 (백그라운드 스레드)
# ------------------------------------------------------------------
def _run_job(job_id, job_dir):
    job = JOBS[job_id]
    old_out, old_err = sys.stdout, sys.stderr
    writer = _LogWriter(job["log"])
    sys.stdout = writer
    sys.stderr = writer
    try:
        om.DATA_FOLDER = os.path.join(job_dir, "data")
        om.BOUNDARY_FOLDER = os.path.join(job_dir, "boundary")
        om.OUTPUT_FOLDER = os.path.join(job_dir, "output")
        om.VRA_CSV_PATH = os.path.join(job_dir, "vra.csv")

        print(f"[작업 시작] {job_id}")
        om.main()

        images = [os.path.basename(p)
                  for p in sorted(glob.glob(os.path.join(om.OUTPUT_FOLDER, "*_Result.png")))]
        produced = [p for p in glob.glob(os.path.join(om.OUTPUT_FOLDER, "**", "*"), recursive=True)
                    if os.path.isfile(p)]
        if not produced:
            raise RuntimeError("결과 파일이 생성되지 않았습니다. 로그의 경고/오류를 확인하세요 "
                               "(필지코드-CSV field 불일치가 가장 흔한 원인).")

        print("[압축] 결과 zip 생성 중...")
        zip_base = os.path.join(job_dir, f"VRA_result_{job_id}")
        zip_path = shutil.make_archive(zip_base, "zip", om.OUTPUT_FOLDER)

        job["images"] = images
        job["zip_path"] = zip_path
        job["state"] = "done"
        print(f"[완료] 결과 zip: {os.path.basename(zip_path)}")
    except Exception as e:
        job["state"] = "error"
        job["log"].append(f"[오류] {e}")
        for line in traceback.format_exc().splitlines():
            job["log"].append(line)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        RUN_LOCK.release()


# ------------------------------------------------------------------
# 라우트
# ------------------------------------------------------------------
@app.route("/chunk", methods=["POST"])
def chunk():
    """파일 조각 수신 — 같은 파일의 조각을 순서대로 append 한다."""
    try:
        upload_id = _safe_name(request.form.get("upload_id", ""))
        kind = request.form.get("kind", "")
        filename = _safe_name(request.form.get("filename", ""))
        index = int(request.form.get("index", -1))
        blob = request.files.get("chunk")

        if not upload_id or kind not in VALID_KINDS or not filename or index < 0 or blob is None:
            return jsonify(ok=False, error="잘못된 업로드 요청입니다."), 400

        folder = os.path.join(UPLOAD_DIR, upload_id, kind)
        os.makedirs(folder, exist_ok=True)
        target = os.path.join(folder, filename)

        mode = "wb" if index == 0 else "ab"
        with open(target, mode) as f:
            shutil.copyfileobj(blob.stream, f, 4 * 1024 * 1024)
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, error=f"조각 저장 실패: {e}"), 500


@app.route("/finalize", methods=["POST"])
def finalize():
    """업로드 완료 → 작업 폴더 구성 → 파이프라인 백그라운드 실행."""
    try:
        payload = request.get_json(silent=True) or {}
        upload_id = _safe_name(payload.get("upload_id", ""))
        upload_root = os.path.join(UPLOAD_DIR, upload_id)
        if not upload_id or not os.path.isdir(upload_root):
            return jsonify(ok=False, error="업로드된 파일을 찾을 수 없습니다. 다시 업로드하세요."), 400

        if not RUN_LOCK.acquire(blocking=False):
            return jsonify(ok=False, error="다른 작업이 실행 중입니다. 완료 후 다시 시도하세요."), 409

        try:
            job_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            job_dir = os.path.join(JOBS_DIR, job_id)
            data_dir = os.path.join(job_dir, "data")
            boundary_dir = os.path.join(job_dir, "boundary")
            for d in (data_dir, boundary_dir, os.path.join(job_dir, "output")):
                os.makedirs(d, exist_ok=True)

            _place_data_uploads(os.path.join(upload_root, "data"), data_dir) \
                if os.path.isdir(os.path.join(upload_root, "data")) else None
            if os.path.isdir(os.path.join(upload_root, "boundary")):
                _place_boundary_uploads(os.path.join(upload_root, "boundary"), boundary_dir)
            vra_path = _place_vra_upload(os.path.join(upload_root, "vra"), job_dir) \
                if os.path.isdir(os.path.join(upload_root, "vra")) else None

            shutil.rmtree(upload_root, ignore_errors=True)

            gndvi_files = glob.glob(os.path.join(data_dir, "*_GNDVI.tif"))
            if not gndvi_files:
                raise ValueError("'*_GNDVI.tif' 형식의 GNDVI 영상이 없습니다. "
                                 "파일명이 '필지코드_..._GNDVI.tif' 인지 확인하세요.")
            if vra_path is None:
                raise ValueError("VRA 설정 CSV가 업로드되지 않았습니다.")

            JOBS[job_id] = {"state": "running", "log": [], "zip_path": None, "images": []}
            t = threading.Thread(target=_run_job, args=(job_id, job_dir), daemon=True)
            t.start()
            return jsonify(ok=True, job_id=job_id,
                           n_gndvi=len(gndvi_files),
                           n_boundary=len(os.listdir(boundary_dir)))
        except Exception:
            RUN_LOCK.release()
            raise
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    except Exception as e:
        return jsonify(ok=False, error=f"작업 구성 실패: {e}"), 500


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify(ok=False, error="존재하지 않는 작업입니다."), 404
    return jsonify(ok=True, state=job["state"], log=job["log"],
                   images=job["images"], zip_ready=bool(job["zip_path"]))


@app.route("/download/<job_id>")
def download(job_id):
    job = JOBS.get(job_id)
    if job is None or not job.get("zip_path") or not os.path.exists(job["zip_path"]):
        abort(404)
    return send_file(job["zip_path"], as_attachment=True,
                     download_name=os.path.basename(job["zip_path"]))


@app.route("/preview/<job_id>/<path:filename>")
def preview(job_id, filename):
    if job_id not in JOBS or not filename.lower().endswith(".png"):
        abort(404)
    out_dir = os.path.join(JOBS_DIR, _safe_name(job_id), "output")
    target = os.path.join(out_dir, _safe_name(filename))
    if not os.path.exists(target):
        abort(404)
    return send_file(target, mimetype="image/png")


# ------------------------------------------------------------------
# 페이지 (단일 파일 유지를 위해 HTML 인라인)
# ------------------------------------------------------------------
PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>드론 VRA 처방맵 생성기</title>
<style>
  body { font-family: 'Malgun Gothic', sans-serif; background:#f4f6f8; margin:0; }
  .wrap { max-width: 860px; margin: 24px auto; padding: 0 16px; }
  h1 { font-size: 22px; color:#1a3c5e; }
  .card { background:#fff; border:1px solid #dde3ea; border-radius:10px; padding:18px 20px; margin-bottom:16px; }
  .card h2 { font-size:15px; margin:0 0 8px; color:#2c5170; }
  .hint { font-size:12.5px; color:#68788c; margin:4px 0 10px; }
  input[type=file] { font-size:13px; }
  button { background:#2c72b8; color:#fff; border:none; border-radius:8px;
           padding:12px 26px; font-size:15px; cursor:pointer; }
  button:disabled { background:#9db4c9; cursor:not-allowed; }
  #bar-outer { background:#e6ebf1; border-radius:6px; height:18px; overflow:hidden; display:none; margin-top:10px;}
  #bar { background:#39a35c; height:100%; width:0%; transition:width .2s; }
  #bar-label { font-size:12px; color:#4a5a6a; margin-top:4px; display:none; }
  #log { background:#101820; color:#c9e3c9; font-family:Consolas,monospace; font-size:12px;
         padding:12px; border-radius:8px; height:260px; overflow-y:auto; white-space:pre-wrap; display:none; }
  #msg { font-size:14px; margin-top:10px; }
  .err { color:#c0392b; font-weight:bold; }
  .ok { color:#1e8449; font-weight:bold; }
  #result { display:none; }
  #result img { max-width:100%; border:1px solid #ccd5de; border-radius:6px; margin-top:10px; }
  a.dl { display:inline-block; background:#1e8449; color:#fff; text-decoration:none;
         border-radius:8px; padding:11px 22px; font-size:15px; margin-top:6px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🚁 드론 VRA 처방맵 생성기</h1>

  <div class="card">
    <h2>① GNDVI 영상 (필수)</h2>
    <div class="hint">'필지코드_..._GNDVI.tif' 파일 여러 개 또는 이것들을 묶은 zip</div>
    <input type="file" id="f_data" multiple accept=".tif,.tiff,.zip">
  </div>

  <div class="card">
    <h2>② 필지 바운더리 (선택)</h2>
    <div class="hint">필지별 zip 여러 개 / zip 묶음 / shp 세트 — 없으면 영상에서 자동 감지(정확도 낮음)</div>
    <input type="file" id="f_boundary" multiple accept=".zip,.shp,.dbf,.shx,.prj,.cpg">
  </div>

  <div class="card">
    <h2>③ VRA 설정 CSV (필수)</h2>
    <div class="hint">컬럼: field,total,spread,crop,grid_size,sigma,masking,height,width,drone_type</div>
    <input type="file" id="f_vra" accept=".csv">
  </div>

  <div class="card">
    <button id="btn" onclick="start()">처방맵 생성 시작</button>
    <div id="bar-outer"><div id="bar"></div></div>
    <div id="bar-label"></div>
    <div id="msg"></div>
  </div>

  <div class="card"><pre id="log"></pre></div>

  <div class="card" id="result">
    <h2>결과</h2>
    <a class="dl" id="dl" href="#">📦 결과 zip 다운로드</a>
    <div id="previews"></div>
  </div>
</div>

<script>
const CHUNK = 64 * 1024 * 1024;   // 64MB — Cloudflare 100MB/요청 제한 대응
let polling = null, failCount = 0;

function setMsg(text, cls) {
  const m = document.getElementById('msg');
  m.textContent = text; m.className = cls || '';
}

function xhrSend(url, form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.upload.onprogress = e => { if (e.lengthComputable && onProgress) onProgress(e.loaded); };
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch (e) {}
      if (xhr.status >= 200 && xhr.status < 300 && data && data.ok) resolve(data);
      else reject(new Error((data && data.error) || ('서버 오류 (HTTP ' + xhr.status + ')')));
    };
    xhr.onerror = () => reject(new Error('서버에 연결할 수 없습니다 — 검은 창(서버)이 켜져 있는지 확인하세요.'));
    xhr.send(form);
  });
}

async function uploadFile(uploadId, kind, file, doneBytes, totalBytes) {
  const nChunks = Math.max(1, Math.ceil(file.size / CHUNK));
  for (let i = 0; i < nChunks; i++) {
    const blob = file.slice(i * CHUNK, Math.min(file.size, (i + 1) * CHUNK));
    const form = new FormData();
    form.append('upload_id', uploadId);
    form.append('kind', kind);
    form.append('filename', file.name);
    form.append('index', i);
    form.append('total', nChunks);
    form.append('chunk', blob, file.name + '.part');
    const base = doneBytes + i * CHUNK;
    await xhrSend('/chunk', form, loaded => updateBar(base + loaded, totalBytes));
  }
  return doneBytes + file.size;
}

function updateBar(done, total) {
  const pct = total > 0 ? Math.min(100, (done / total * 100)) : 0;
  document.getElementById('bar').style.width = pct.toFixed(1) + '%';
  document.getElementById('bar-label').textContent =
    '업로드 ' + (done / 1048576).toFixed(1) + ' / ' + (total / 1048576).toFixed(1) + ' MB (' + pct.toFixed(0) + '%)';
}

async function start() {
  const dataFiles = document.getElementById('f_data').files;
  const bndFiles = document.getElementById('f_boundary').files;
  const vraFiles = document.getElementById('f_vra').files;
  if (dataFiles.length === 0) { setMsg('GNDVI 영상을 선택하세요.', 'err'); return; }
  if (vraFiles.length === 0) { setMsg('VRA 설정 CSV를 선택하세요.', 'err'); return; }

  const btn = document.getElementById('btn');
  btn.disabled = true;
  document.getElementById('result').style.display = 'none';
  document.getElementById('bar-outer').style.display = 'block';
  document.getElementById('bar-label').style.display = 'block';
  setMsg('파일 업로드 중...');

  const uploadId = Date.now() + '_' + Math.random().toString(36).slice(2, 8);
  let total = 0;
  for (const f of dataFiles) total += f.size;
  for (const f of bndFiles) total += f.size;
  for (const f of vraFiles) total += f.size;

  try {
    let done = 0;
    for (const f of dataFiles) done = await uploadFile(uploadId, 'data', f, done, total);
    for (const f of bndFiles) done = await uploadFile(uploadId, 'boundary', f, done, total);
    for (const f of vraFiles) done = await uploadFile(uploadId, 'vra', f, done, total);

    setMsg('작업 구성 중...');
    const res = await fetch('/finalize', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ upload_id: uploadId })
    });
    let data = null;
    try { data = await res.json(); } catch (e) {}
    if (!res.ok || !data || !data.ok) {
      throw new Error((data && data.error) || ('서버 오류 (HTTP ' + res.status + ')'));
    }
    setMsg('처방맵 생성 중... (GNDVI ' + data.n_gndvi + '개)', 'ok');
    document.getElementById('log').style.display = 'block';
    failCount = 0;
    polling = setInterval(() => poll(data.job_id), 1500);
  } catch (e) {
    setMsg(e.message, 'err');
    btn.disabled = false;
  }
}

async function poll(jobId) {
  try {
    const res = await fetch('/status/' + jobId);
    const data = await res.json();
    failCount = 0;
    const logEl = document.getElementById('log');
    logEl.textContent = data.log.join('\\n');
    logEl.scrollTop = logEl.scrollHeight;

    if (data.state === 'done') {
      clearInterval(polling);
      setMsg('✅ 처방맵 생성 완료!', 'ok');
      document.getElementById('btn').disabled = false;
      document.getElementById('dl').href = '/download/' + jobId;
      const pv = document.getElementById('previews');
      pv.innerHTML = '';
      for (const img of data.images) {
        const el = document.createElement('img');
        el.src = '/preview/' + jobId + '/' + encodeURIComponent(img);
        pv.appendChild(el);
      }
      document.getElementById('result').style.display = 'block';
    } else if (data.state === 'error') {
      clearInterval(polling);
      setMsg('처리 중 오류가 발생했습니다. 아래 로그를 확인하세요.', 'err');
      document.getElementById('btn').disabled = false;
    }
  } catch (e) {
    failCount += 1;
    if (failCount >= 4) {
      clearInterval(polling);
      setMsg('서버와의 연결이 끊어졌습니다. 서버(검은 창)를 확인한 뒤 새로고침하세요.', 'err');
      document.getElementById('btn').disabled = false;
    }
  }
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return PAGE


# ------------------------------------------------------------------
# 실행
# ------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    url = "http://127.0.0.1:8000"
    if not os.environ.get("VRA_NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print("=" * 52)
    print("  드론 VRA 처방맵 웹앱")
    print(f"  브라우저 접속: {url}")
    print("  이 창을 닫으면 웹앱이 종료됩니다.")
    print("=" * 52)
    # threaded=True: 처방 생성 중에도 상태 폴링에 응답
    app.run(host="127.0.0.1", port=8000, threaded=True, debug=False)
