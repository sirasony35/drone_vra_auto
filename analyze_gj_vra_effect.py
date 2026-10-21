# -*- coding: utf-8 -*-
"""김제(GJ) 벼 변량시비 효과 분석 + 살포패턴(선무늬) 분석.

화성 analyze_vra_effect.py 와 동일한 균일도(CV) 방법론을 김제 벼 실증필지에 적용한다.
입력 데이터는 이동식 SSD `total_data` 안에 있으므로(PC마다 드라이브 문자 상이) 자동 탐색한다.

실증 설계(2026드론 균일_변량 내역.xlsx):
 - 변량(대동 드론 변량처방): GJ-R1~R6
 - 균일(대동 균등살포):     GJ-R11
 - 농진청 살포:              GJ-R14
 - 무처리(웃거름 살포안함):  GJ-R15, R16
 - 시비 前 = 03회차(0724/0729), 시비 後 = 04회차(0811)

산출물(SSD `.../00_데이터 분석/변량시비 리포트/gj_vra_effect_report/`):
 1) 김제_변량시비_효과분석_리포트.pdf  — 커버 + 요약(CV/P10) + 필지별 상세(前後 GNDVI·RGB·히스토그램·지표표)
 2) 김제_살포패턴(선무늬)_분석.pdf     — 대동 변량 필지의 미살포 줄무늬(stripe) vs 농진청/균일 비교

사용: python analyze_gj_vra_effect.py
의존성: rasterio, numpy, scipy, matplotlib, openpyxl, (선택)contextily, PIL
"""
import os, glob, math, sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 콘솔 한글/기호 출력 안전
except Exception:
    pass
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import openpyxl
from scipy.ndimage import gaussian_filter
try:
    import contextily as ctx
    HAS_CTX = True
except Exception:
    HAS_CTX = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from collections import OrderedDict
from PIL import Image

# ---------------- total_data 자동 탐색 (이동식 SSD, 드라이브 문자 가변) ----------------
def find_total():
    env = os.environ.get("TOTAL_DATA")
    if env and (Path(env) / "drone_26_data").exists():
        return Path(env)
    for d in "EFGDHIJKLMNOPQRSTUVWXYZC":
        p = Path(f"{d}:/total_data")
        if (p / "drone_26_data").exists():
            return p
    home = Path.home() / "mnt" / "total_data"
    if (home / "drone_26_data").exists():
        return home
    raise FileNotFoundError("total_data 볼륨을 찾지 못했습니다 (TOTAL_DATA 환경변수로 지정 가능).")


TOTAL = find_total()
VEA = TOTAL / "drone_26_data" / "1.김제" / "2.벼"
XLSX = TOTAL / "drone_26_data" / "00_데이터 분석" / "2026드론 균일_변량 내역.xlsx"
OUT = TOTAL / "drone_26_data" / "00_데이터 분석" / "변량시비 리포트" / "gj_vra_effect_report"
DET = OUT / "필지별"
STR = OUT / "선무늬"
for d in (OUT, DET, STR):
    d.mkdir(parents=True, exist_ok=True)

for n in ["Malgun Gothic", "NanumGothic", "Gulim", "Batang"]:
    if n in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = n
        break
plt.rcParams["axes.unicode_minus"] = False
GREEN = LinearSegmentedColormap.from_list("g", ["#8B4513", "#DEB887", "#FFFF99", "#66BB66", "#006400"])
DIVCM = LinearSegmentedColormap.from_list("d", ["#8e44ad", "#c39bd3", "#f2f2f2", "#f5b041", "#ca6f1e"])
RXCM = LinearSegmentedColormap.from_list("rx", ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"])
# 처방맵(Rx, kg/ha) 폴더 — drone_vra_auto 프로젝트 내(result/ 는 git 미추적이라 PC별로 있을 수 있음)
RX_DIR = Path(__file__).resolve().parent / "result" / "gj_1021" / "DJI" / "Rx"
VMIN, VMAX = 0.2, 1.0
BEFORE_RD, AFTER_RD = "03", "04"   # 시비 前/後 회차

# 보고용 그룹 라벨
VARIABLE = "변량"      # 대동 드론 변량
UNIFORM = "균일"       # 대동 균등
RDA = "농진청"         # 농진청 살포
NONE_ = "무처리"       # 웃거름 미살포


# ---------------- 래스터 유틸 ----------------
def load1(path, target=1600):
    with rasterio.open(path) as s:
        sc = max(1, int(max(s.width, s.height) / target))
        return s.read(1, out_shape=(s.height // sc, s.width // sc)).astype("float32")


def masked(a):
    return np.where((a > VMIN) & (a < VMAX) & (~np.isnan(a)), a, np.nan)


def gstat(a):
    v = a[(a > VMIN) & (a < VMAX) & (~np.isnan(a))]
    return dict(mean=float(v.mean()), cv=float(v.std() / v.mean() * 100),
                p10=float(np.percentile(v, 10)), low=float((v < 0.5).mean() * 100), vals=v)


def load_rgb(path, target=900):
    with rasterio.open(path) as s:
        sc = max(1, int(max(s.width, s.height) / target))
        h, w = s.height // sc, s.width // sc
        R = s.read(1, out_shape=(h, w)).astype("float32")
        G = s.read(2, out_shape=(h, w)).astype("float32")
        B = s.read(3, out_shape=(h, w)).astype("float32")
        al = s.read(4, out_shape=(h, w)).astype("float32") if s.count >= 4 else np.full((h, w), 255.0)
    return R, G, B, al


def rgb_img(path, gain=1.0):
    R, G, B, al = load_rgb(path)
    img = np.clip(np.dstack([R, G, B]) / 255.0 * gain, 0, 1)
    alpha = (al > 0).astype("float32")
    return np.dstack([img, alpha])


def exg_stat(path):
    R, G, B, al = load_rgb(path)
    br = R + G + B
    m = (al > 0) & (br > 30) & (br < 735)
    if m.sum() == 0:
        return dict(exg=0.0)
    exg = (2 * G - R - B)[m]
    return dict(exg=float(exg.mean()))


def bounds3857(path):
    with rasterio.open(path) as s:
        l, b, r, t = transform_bounds(s.crs, "EPSG:3857", *s.bounds)
    return (l, r, b, t)


def find_rx(code):
    """처방맵 Rx GeoTIFF. 프로젝트(result/gj_1021) → SSD 사본(_rx/) 순 탐색. 변량 필지만 존재."""
    for base in (RX_DIR, OUT / "_rx"):
        p = base / f"{code}.tif"
        if p.exists():
            return str(p)
        g = glob.glob(str(base / f"{code}_*.tif"))
        if g:
            return g[0]
    return None


def find_tif(code, rnd, kind):
    """{VEA}/GJ-Rn/{rnd}_*/1*원천데이터/{code}_{rnd}_*_{kind}.tif 탐색."""
    folder = f"GJ-R{code[3:]}"   # GJR1 -> GJ-R1 (파일코드는 하이픈 없음)
    pats = [str(VEA / folder / f"{rnd}_*" / "1*원천데이터" / f"{code}_{rnd}_*_{kind}.tif")]
    for p in pats:
        g = glob.glob(p)
        if g:
            return g[0]
    return None


# ---------------- 실증 메타(xlsx) ----------------
def load_meta():
    """김제 필지의 그룹/살포일/CASE 를 xlsx에서 읽는다."""
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb.active
    meta = OrderedDict()
    for row in ws.iter_rows(values_only=True):
        vals = list(row)
        if len(vals) < 20:
            continue
        region, code = vals[2], vals[3]
        if region != "김제" or not code or not str(code).startswith("GJ-R"):
            continue
        fcode = str(code).replace("-", "")   # GJ-R1 -> GJR1
        dronecol = (vals[15] or "").strip()
        if dronecol == "변량":
            grp = VARIABLE
        elif dronecol == "균일":
            grp = UNIFORM
        elif "농진청" in dronecol:
            grp = RDA
        else:
            grp = NONE_
        tv = vals[14]
        tdisp = tv.strftime("%m/%d") if hasattr(tv, "strftime") else str(tv or "").strip()
        early = tdisp.startswith(("05/", "5/", "5월"))   # 조기이앙(≈5월 말) = R1·R11·R14
        imo = (str(vals[12] or "").strip().upper().startswith("O"))  # 이모작 여부
        meta[fcode] = dict(
            code=fcode, name=str(code), grp=grp, dronecol=dronecol,
            case=vals[8], plan=(vals[9] or ""), farmer=(vals[5] or "").strip(),
            bags=vals[17], grid=vals[18], spray=(vals[19] or "").strip(),
            early=early, season=("조기이앙" if early else "만기이앙"), imo=imo, transplant=tdisp,
        )
    return meta


def verdict_of(grp, dcv, early):
    """이앙시기(03시점 생육단계)를 감안한 균일도 판정.
    조기이앙 필지는 시비 前 이미 성숙·고균일 → 작은 변화는 '악화'가 아니라 '개선 미비'."""
    if grp == NONE_:
        return "무처리(웃거름 없음)", "#7f8c8d"
    if dcv <= -1.5:
        return "균일도 개선", "#1e8449"
    if dcv <= 0.8:
        if early:
            return "개선 미비 (조기이앙·이미 성숙)", "#e67e22"
        return "균일도 유지 (개선 미비)", "#e67e22"
    return "균일도 소폭 악화", "#c0392b"


# ---------------- 선무늬(stripe) 지표 ----------------
def load_for_fft(path, px_m=0.05, maxdim=2200):
    """GNDVI를 약 px_m(m/px) 해상도로 읽어 마스크. 반환 (arr[nan], valid, res_m)."""
    with rasterio.open(path) as s:
        res0 = abs(s.res[0])
        sc = max(1, int(round(px_m / res0)))
        h, w = s.height // sc, s.width // sc
        if max(h, w) > maxdim:
            sc2 = math.ceil(max(h, w) / maxdim)
            sc *= sc2
            h, w = s.height // sc, s.width // sc
        a = s.read(1, out_shape=(h, w)).astype("float32")
        res_m = res0 * sc
    valid = (a > VMIN) & (a < VMAX) & np.isfinite(a)
    a = np.where(valid, a, np.nan)
    return a, valid, res_m


def crop_valid(a, valid, pad=4):
    ys, xs = np.where(valid)
    if len(ys) == 0:
        return a, valid
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, a.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, a.shape[1])
    return a[y0:y1, x0:x1], valid[y0:y1, x0:x1]


STRIPE_LAM_LO, STRIPE_LAM_HI = 2.5, 15.0   # 드론 살포 스와스 간격 대역(m)


def load_resid(path, px_m=0.06):
    """GNDVI를 읽어 대규모 추세를 제거한 잔차(resid)를 만든다.
    반환 dict(resid, valid, res_m, arr[실제GNDVI nan], mean)."""
    a, valid, res_m = load_for_fft(path, px_m=px_m)
    a, valid = crop_valid(a, valid)
    arr = np.where(valid, a, np.nan)
    mean_v = float(np.nanmean(arr))
    filled = np.where(valid, arr, mean_v)
    trend = gaussian_filter(filled, sigma=max(a.shape) / 50.0)
    resid = np.where(valid, filled - trend, 0.0)
    return dict(resid=resid, valid=valid, res_m=res_m, arr=arr, mean=mean_v)


def profile_at_angle(R, ang):
    """잔차를 ang도 회전시켜 '줄무늬에 수직' 방향의 1D 프로파일(평균 GNDVI 잔차)을 뽑는다."""
    from scipy.ndimage import rotate
    r = rotate(R["resid"], -ang, reshape=True, order=1, cval=0.0)
    v = rotate(R["valid"].astype("float32"), -ang, reshape=True, order=1, cval=0.0)
    m = v > 0.5
    rr = np.where(m, r, np.nan)
    cnt = m.sum(axis=1)
    with np.errstate(invalid="ignore"):
        prof = np.nansum(np.where(m, r, 0.0), axis=1) / np.maximum(cnt, 1)
    prof = prof[cnt > (0.3 * np.nanmax(cnt))]   # 유효 폭이 충분한 행만
    return prof


def profile_periodicity(prof, res_m):
    """1D 프로파일의 스와스 대역(2.5~15m) 주기성. 반환 (strength, wavelength_m, peak2trough).
    peak2trough = 지배 정현파의 진폭×2 ≈ '진한 띠 − 연한 띠' GNDVI 차."""
    p = prof[np.isfinite(prof)]
    if len(p) < 16:
        return 0.0, 0.0, 0.0
    p = p - p.mean()
    n = len(p)
    w = np.hanning(n)
    F = np.abs(np.fft.rfft(p * w))
    freqs = np.fft.rfftfreq(n, d=res_m)          # cycles/m
    lam = 1.0 / np.maximum(freqs, 1e-9)
    band = (lam >= STRIPE_LAM_LO) & (lam <= STRIPE_LAM_HI)
    if band.sum() < 2:
        return 0.0, 0.0, 0.0
    k = int(np.argmax(np.where(band, F, -1)))
    amp = 2.0 * F[k] / np.sum(w)                 # 정현파 진폭
    strength = float(F[k] / (np.median(F[band]) + 1e-9))
    return strength, float(lam[k]), float(2.0 * amp)


def scan_best_angle(R, step=5):
    """0~175도 스캔해 스와스 주기성이 가장 강한 방향을 찾는다."""
    best = (-1.0, 0.0, 0.0, 0.0, None)
    for ang in range(0, 180, step):
        prof = profile_at_angle(R, ang)
        s, lam, amp = profile_periodicity(prof, R["res_m"])
        if s > best[0]:
            best = (s, float(ang), lam, amp, prof)
    return dict(strength=best[0], angle=best[1], wavelength=best[2], amplitude=best[3], profile=best[4])


def eval_at_angle(R, ang):
    prof = profile_at_angle(R, ang)
    s, lam, amp = profile_periodicity(prof, R["res_m"])
    return dict(strength=s, angle=ang, wavelength=lam, amplitude=amp, profile=prof)


# ---------------- 필지 상세 페이지 ----------------
def field_page(m, rec):
    code, grp, farmer = m["code"], m["grp"], m["farmer"]
    bg = find_tif(code, BEFORE_RD, "GNDVI")
    ag = find_tif(code, AFTER_RD, "GNDVI")
    br = find_tif(code, BEFORE_RD, "RGB")
    ar = find_tif(code, AFTER_RD, "RGB")
    def f0(x):
        return f"{x:.0f}" if np.isfinite(x) else "-"
    dexg = (rec["exga"] - rec["exgb"]) if (np.isfinite(rec["exga"]) and np.isfinite(rec["exgb"])) else float("nan")
    rx = find_rx(code)
    b = masked(load1(bg))
    a = masked(load1(ag))
    allv = np.concatenate([b[~np.isnan(b)], a[~np.isnan(a)]])
    lo, hi = np.percentile(allv, 3), np.percentile(allv, 98)
    fig = plt.figure(figsize=(14, 10.6))
    gs = gridspec.GridSpec(3, 3, height_ratios=[2.0, 1.7, 1.5], hspace=0.34, wspace=0.16,
                           left=0.04, right=0.97, top=0.85, bottom=0.03)
    gcol = {VARIABLE: "#c0392b", UNIFORM: "#2874a6", RDA: "#1e8449", NONE_: "#7f8c8d"}[grp]
    dcv = rec["cva"] - rec["cvb"]
    verdict, vcol = verdict_of(grp, dcv, m["early"])
    grp_disp = {VARIABLE: "변량(대동)", UNIFORM: "균일(대동)", RDA: "농진청", NONE_: "무처리"}[grp]
    season = f"{m['season']}·이앙 {m['transplant']}" + ("·이모작" if m["imo"] else "·일모작")
    fig.suptitle(f"{code}  [{grp_disp}]   · 경작자 {farmer}  · {m['plan']}  · {season}", fontsize=17,
                 fontweight="bold", y=0.975, color=gcol)
    sp = f"살포 {m['spray']}" if m["spray"] and m["spray"] not in ("살포안함",) else m["dronecol"]
    fig.text(0.5, 0.945,
             f"CV {rec['cvb']:.1f}% → {rec['cva']:.1f}% ({dcv:+.1f}%p, {verdict})   |   "
             f"ExG(녹색도) {f0(rec['exgb'])}→{f0(rec['exga'])}   |   {sp}",
             ha="center", fontsize=12, color="#333")
    note2 = "※ CV(고르기 편차)는 낮을수록 균일 · ΔCV 음수(-)=개선  |  前=0724/0729, 後=0811  |  처방맵 진할수록(빨강) 비료 많이 처방"
    if m["early"] and grp != NONE_:
        note2 += "  |  조기이앙 성숙 → 변화 작게"
    fig.text(0.5, 0.915, note2, ha="center", fontsize=9.2, color="#888")
    # row1: 前 GNDVI / 처방맵(kg/ha) / 後 GNDVI  (화성 리포트와 동일)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(b, cmap=GREEN, vmin=lo, vmax=hi); ax.axis("off")
    ax.set_title("시비前 GNDVI (03회차)", fontsize=11)
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); ax.set_title("처방맵 (살포량 kg/ha)", fontsize=11)
    if rx:
        with rasterio.open(rx) as s:
            rr = s.read(1).astype("float32"); rr = np.where(rr > 0, rr, np.nan)
        im2 = ax.imshow(rr, cmap=RXCM); fig.colorbar(im2, ax=ax, fraction=0.045, pad=0.02)
    else:
        txt = "균등 살포\n(처방맵 없음)" if grp in (UNIFORM, RDA) else "처방맵 없음"
        ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=12, color="#999", transform=ax.transAxes)
    ax = fig.add_subplot(gs[0, 2]); im = ax.imshow(a, cmap=GREEN, vmin=lo, vmax=hi); ax.axis("off")
    ax.set_title("시비後 GNDVI (04회차)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    # row2: 前 RGB / 後 RGB / 히스토그램
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off"); ax.set_title("시비前 RGB", fontsize=11)
    if br: ax.imshow(rgb_img(br))
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); ax.set_title("시비後 RGB", fontsize=11)
    if ar: ax.imshow(rgb_img(ar))
    axh = fig.add_subplot(gs[1, 2]); bins = np.linspace(0.2, 0.95, 55)
    axh.hist(rec["vb"], bins=bins, alpha=0.55, color="#e67e22", density=True, label=f"前 {rec['cvb']:.1f}%")
    axh.hist(rec["va"], bins=bins, alpha=0.55, color="#27ae60", density=True, label=f"後 {rec['cva']:.1f}%")
    axh.set_title("GNDVI 분포 前/後", fontsize=10); axh.legend(fontsize=8); axh.set_yticks([])
    axh.set_xlabel("GNDVI", fontsize=9)
    # row3: 지표표 + 판정
    axt = fig.add_subplot(gs[2, :]); axt.axis("off")
    rows = [["지표", "시비前", "시비後", "Δ"],
            ["평균 GNDVI", f"{rec['mb']:.3f}", f"{rec['ma']:.3f}", f"{rec['ma']-rec['mb']:+.3f}"],
            ["CV (균일도)", f"{rec['cvb']:.1f}%", f"{rec['cva']:.1f}%", f"{dcv:+.1f}%p"],
            ["P10 (저위)", f"{rec['p10b']:.3f}", f"{rec['p10a']:.3f}", f"{rec['p10a']-rec['p10b']:+.3f}"],
            ["저활력<0.5", f"{rec['lowb']:.1f}%", f"{rec['lowa']:.1f}%", f"{rec['lowa']-rec['lowb']:+.1f}%p"],
            ["ExG 녹색도", f0(rec['exgb']), f0(rec['exga']), (f"{dexg:+.0f}" if np.isfinite(dexg) else "-")]]
    tb = axt.table(cellText=rows, cellLoc="center", loc="center", bbox=[0.0, 0.08, 0.56, 0.86])
    tb.auto_set_font_size(False); tb.set_fontsize(9.5)
    for (rr, cc), cell in tb.get_celld().items():
        if rr == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
    axt.text(0.62, 0.80, f"■ 판정: {verdict}", ha="left", fontsize=12, transform=axt.transAxes,
             color=vcol, fontweight="bold")
    axt.text(0.62, 0.58, f"P10 {rec['p10a']-rec['p10b']:+.3f} · 저활력 {rec['lowa']-rec['lowb']:+.1f}%p\n"
             f"평균GNDVI {rec['mb']:.2f}→{rec['ma']:.2f}",
             ha="left", fontsize=10, transform=axt.transAxes, color="#555", va="top")
    if rx and grp == VARIABLE:
        axt.text(0.62, 0.36, "※ 처방맵: 前 생육이 낮은(연한) 곳에 비료를\n   더 처방(빨강) → 均一化 의도",
                 ha="left", fontsize=9, transform=axt.transAxes, color="#8e44ad", va="top")
    if m["early"] and grp != NONE_:
        axt.text(0.62, 0.16, "※ 조기이앙으로 前 이미 성숙 → 변화 작음(악화 아님)",
                 ha="left", fontsize=9, transform=axt.transAxes, color="#e67e22", va="top")
    out = DET / f"{code}.png"
    fig.savefig(out, dpi=135, bbox_inches="tight"); plt.close(fig)
    return str(out)


# ---------------- 요약/커버 ----------------
def summary_page(recs, meta):
    fig = plt.figure(figsize=(14, 8.4)); fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1], hspace=0.55, wspace=0.22,
                           left=0.08, right=0.95, top=0.88, bottom=0.12)
    fig.suptitle("김제 벼 변량시비 효과 요약 — 작기(이앙시기) 감안 균일도(CV) 변화", fontsize=18, fontweight="bold")
    order = sorted(recs.items(), key=lambda kv: kv[1]["cva"] - kv[1]["cvb"])
    names = [k for k, _ in order]
    dcvs = [recs[k]["cva"] - recs[k]["cvb"] for k in names]
    gcolmap = {VARIABLE: "#c0392b", UNIFORM: "#2874a6", RDA: "#1e8449", NONE_: "#bdc3c7"}
    cols = [gcolmap[meta[k]["grp"]] for k in names]
    ax = fig.add_subplot(gs[0, :])
    bars = ax.bar(names, dcvs, color=cols)
    for k, bar in zip(names, bars):
        if meta[k]["early"]:                  # 조기이앙(5/27) 표시
            bar.set_hatch("///"); bar.set_edgecolor("white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ΔCV (%p) ↓개선/↑악화"); ax.set_title("필지별 균일도 변화 (음수=개선 ·빗금=조기이앙 5/27)", fontsize=12)
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(handles=[mpatches.Patch(color="#c0392b", label="변량(대동)"),
                       mpatches.Patch(color="#2874a6", label="균일(대동)"),
                       mpatches.Patch(color="#1e8449", label="농진청"),
                       mpatches.Patch(color="#bdc3c7", label="무처리"),
                       mpatches.Patch(facecolor="#999", hatch="///", label="조기이앙(5/27)")],
              fontsize=8.5, ncol=5)
    # 좌하: 작기별 변량 vs 대조 평균 ΔCV
    axg = fig.add_subplot(gs[1, 0])
    def avg(keys):
        v = [recs[k]["cva"] - recs[k]["cvb"] for k in keys if k in recs]
        return float(np.mean(v)) if v else float("nan")
    groups4 = [
        ("조기이앙\n변량(R1)", [k for k in recs if meta[k]["early"] and meta[k]["grp"] == VARIABLE], "#c0392b"),
        ("조기이앙\n균일·농진청", [k for k in recs if meta[k]["early"] and meta[k]["grp"] in (UNIFORM, RDA)], "#2874a6"),
        ("만기이앙\n변량(R2~6)", [k for k in recs if not meta[k]["early"] and meta[k]["grp"] == VARIABLE], "#c0392b"),
        ("만기이앙\n무처리(R15·16)", [k for k in recs if not meta[k]["early"] and meta[k]["grp"] == NONE_], "#bdc3c7"),
    ]
    xl, vv, cc = [], [], []
    for lab, keys, c in groups4:
        a = avg(keys)
        if np.isfinite(a):
            xl.append(lab); vv.append(a); cc.append(c)
    xpos = np.arange(len(xl))
    axg.bar(xpos, vv, color=cc, width=0.7); axg.axhline(0, color="black", lw=0.8)
    axg.set_xticks(xpos); axg.set_xticklabels(xl, fontsize=8)
    axg.set_title("작기별 평균 ΔCV (%p) — 같은 작기끼리 비교", fontsize=11)
    for i, v in enumerate(vv):
        axg.text(i, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontweight="bold", fontsize=9)
    # 우하: 해설
    axn = fig.add_subplot(gs[1, 1]); axn.axis("off")
    axn.text(0.0, 0.98, "■ 이앙시기(생육단계) 감안 해석", fontsize=12, fontweight="bold", transform=axn.transAxes, va="top")
    txt = ("• 조기이앙 R1·R11·R14(이앙 5/27, 일모작)는 시비 前(0724)\n"
           "  이미 생육 성숙·고균일(평균GNDVI 0.68~0.71) → 변화 작음.\n"
           "  이는 '악화'가 아니라 '개선 미비'(변량 R1 ΔCV +0.7 ≈\n"
           "  균일 R11 +0.4와 유사, 같은 생육단계).\n\n"
           "• 만기이앙 R2~R6(이앙 6/17)은 前 저생육(0.4~0.5) → 변화 큼.\n"
           "  단 저생육 필지는 성숙만으로도 CV가 내려가므로(무처리\n"
           "  R15·R16도 포함) 그룹 평균엔 성숙 효과가 섞임.\n\n"
           "→ 결론: 변량의 '균일도 악화'는 없음. 조기이앙은 개선 미비,\n"
           "   만기이앙은 개선되나 성숙 효과와 혼재.")
    axn.text(0.0, 0.86, txt, fontsize=9.3, transform=axn.transAxes, va="top")
    out = OUT / "_summary.png"; fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return str(out)


def cover_page(recs, meta):
    fig = plt.figure(figsize=(11.7, 8.3)); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.74, "김제 벼 변량시비 효과 분석 리포트", ha="center", fontsize=27, fontweight="bold")
    fig.text(0.5, 0.68, "시비 前(03회차) · 처방맵(kg/ha) · 시비 後(04회차, 0811) GNDVI+RGB 종합 · 균일도(CV) 검증",
             ha="center", fontsize=12.5, color="#333")
    fig.text(0.5, 0.605, "CV(변동계수)=논 안 생육 편차 · 낮을수록 균일 → ΔCV가 음수(-)일수록 '개선'",
             ha="center", fontsize=13, fontweight="bold", color="#1e5f2f",
             bbox=dict(boxstyle="round,pad=0.6", fc="#eafaf0", ec="#27ae60", lw=1.6))

    def gmean(glist, key):
        arr = [recs[k][key[1]] - recs[k][key[0]] for k in recs if meta[k]["grp"] in glist]
        return float(np.mean(arr)) if arr else float("nan")
    dv = gmean([VARIABLE], ("cvb", "cva")); du = gmean([UNIFORM, RDA], ("cvb", "cva"))
    pv = gmean([VARIABLE], ("p10b", "p10a")); pu = gmean([UNIFORM, RDA], ("p10b", "p10a"))
    nvar = sum(1 for k in recs if meta[k]["grp"] == VARIABLE)
    nuni = sum(1 for k in recs if meta[k]["grp"] in (UNIFORM, RDA))
    nnone = sum(1 for k in recs if meta[k]["grp"] == NONE_)
    lines = [
        f"■ 대상: 변량 {nvar}필지(대동) · 균일/농진청 {nuni}필지 · 무처리 {nnone}필지 (드론 GNDVI/RGB 2cm)",
        "■ 이앙시기: 조기이앙(5/27) R1·R11·R14  /  만기이앙(6/17) R2~R6·R15·R16",
        f"■ 균일도(CV) 변화:  변량 {dv:+.2f}%p  ·  균일/농진청 {du:+.2f}%p   (작기별 비교는 요약 페이지)",
        f"■ 저활력(P10) 회복:  변량 {pv:+.3f}   vs   균일/농진청 {pu:+.3f}",
        "■ 결론: 변량의 '균일도 악화'는 없음 — 조기이앙 필지는 '개선 미비', 만기이앙 필지는 개선(성숙 효과 혼재)",
    ]
    for i, ln in enumerate(lines):
        fig.text(0.1, 0.5 - i * 0.062, ln, fontsize=12.0, color="#c0392b" if i == 4 else "#222")
    fig.text(0.1, 0.135,
             "※ 해석 주의(교란): ① 조기이앙(R1·R11·R14, 5/27)은 시비 前(0724) 이미 성숙·고균일(평균GNDVI 0.68~0.71)\n"
             "   → 변화 작음 = '개선 미비'(악화 아님). ② 만기이앙(R2~6, 6/17)은 前 저생육(0.4~0.5)서 출발해 변화 큼.\n"
             "   ③ 다만 저생육 필지는 성숙만으로도 CV 감소(무처리 R15·R16) → 그룹평균엔 생육단계 효과가 섞임.\n"
             "   → 같은 이앙시기끼리 비교하고(요약), 살포 균등성은 별도 '선무늬(미살포 선)' 분석으로 교차확인.",
             fontsize=9.0, color="#666")
    out = OUT / "_cover.png"; fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)
    return str(out)


# ---------------- 선무늬(미살포 패턴) 분석 ----------------
def assess_stripe(code):
    """한 필지의 살포 줄무늬를 방향 스캔 + 횡단 프로파일로 정량 평가.
    - 後 영상에서 스와스 방향/간격/진폭(진한-연한 GNDVI 차)을 찾고,
    - 같은 방향으로 前 영상의 진폭을 재어 '시비 때 생겼는지'(emergence) 판정,
    - 前의 지배 방향(=모내기·기계자국)과 스와스 방향 차이를 잰다."""
    bg = find_tif(code, BEFORE_RD, "GNDVI"); ag = find_tif(code, AFTER_RD, "GNDVI")
    if not ag:
        return None
    Ra = load_resid(ag); Rb = load_resid(bg) if bg else None
    A = scan_best_angle(Ra)
    B = eval_at_angle(Rb, A["angle"]) if Rb else None
    plant = scan_best_angle(Rb) if Rb else None
    amp_a = A["amplitude"]; amp_b = B["amplitude"] if B else 0.0
    emerge = amp_a / (amp_b + 1e-6)
    dd = None
    if plant:
        d = abs(A["angle"] - plant["angle"]); dd = min(d, 180 - d)
    return dict(code=code, Ra=Ra, Rb=Rb, A=A, B=B, plant=plant,
                amp_a=amp_a, amp_b=amp_b, emerge=emerge, dir_diff=dd,
                mean_a=Ra["mean"], hi=Ra["mean"] + amp_a / 2, lo=Ra["mean"] - amp_a / 2)


def stripe_verdict(grp, S):
    """필지별 '미살포 줄무늬' 판정. amp=진한−연한 GNDVI 차, emerged=시비때 생김."""
    amp = S["amp_a"]
    emerged = (S["amp_a"] > S["amp_b"] * 1.3) and (S["amp_a"] - S["amp_b"] > 0.008)
    if grp in (RDA, UNIFORM):
        if amp < 0.025:
            return "고르게 살포 — 미살포 띠 거의 없음 (대조군)", "#1e8449", emerged
        return "띠 일부 있으나 대조군(참고)", "#2874a6", emerged
    if amp >= 0.030 and emerged:
        return "미살포 줄무늬 가능성 높음 (시비 後 생김)", "#c0392b", emerged
    if amp >= 0.030 and not emerged:
        return "띠 있으나 시비 前부터 존재 → 미살포로 단정 어려움", "#e67e22", emerged
    if amp >= 0.018:
        return "약한 띠 — 판단 보류", "#e67e22", emerged
    return "띠 미약 — 고르게 살포로 판단", "#1e8449", emerged


def _prof_m(prof, res):
    p = prof[np.isfinite(prof)]
    return np.arange(len(p)) * res, p


def draw_stripe_guide(ax, shape, ang, lam_m, res_m, label=None):
    """감지된 평행 띠의 방향·간격을 이미지 위에 박스 + 점선 가이드로 표시."""
    import matplotlib.patches as mp
    H, W = shape
    cx, cy = W / 2.0, H / 2.0
    lam_px = max(lam_m / res_m, 4.0)
    th = math.radians(ang)
    dx, dy = math.cos(th), math.sin(th)          # 띠(선)가 뻗는 방향
    nx, ny = -math.sin(th), math.cos(th)         # 띠 간격 방향(띠에 직각)
    Lb = 0.40 * max(H, W)                        # 박스 반길이(띠 방향)
    halfw = 1.6 * lam_px                         # 박스 반폭(약 3개 띠 포함)
    corners = [(cx + dx * Lb + nx * halfw, cy + dy * Lb + ny * halfw),
               (cx - dx * Lb + nx * halfw, cy - dy * Lb + ny * halfw),
               (cx - dx * Lb - nx * halfw, cy - dy * Lb - ny * halfw),
               (cx + dx * Lb - nx * halfw, cy + dy * Lb - ny * halfw)]
    ax.add_patch(mp.Polygon(corners, closed=True, fill=False, ec="#00e5ff", lw=2.2))
    for k in (-1, 0, 1):                          # 박스 안 평행 띠 위치 점선
        ox, oy = cx + nx * lam_px * k, cy + ny * lam_px * k
        ax.plot([ox - dx * Lb, ox + dx * Lb], [oy - dy * Lb, oy + dy * Lb],
                color="#00e5ff", lw=1.0, ls="--", alpha=0.85)
    if label:
        ax.text(0.02, 0.02, label, transform=ax.transAxes, fontsize=8.5, color="#00343d",
                va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.25", fc="#b8f2ff", ec="#00bcd4", alpha=0.9))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)


def stripe_overview(meta, assess):
    """1페이지: 판단 방법·가능성·한계 + 필지별 판정 요약표."""
    fig = plt.figure(figsize=(14, 9)); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.965, "살포 줄무늬(미살포 선) 판단이 가능한가? — 방법·근거·한계", ha="center",
             fontsize=19, fontweight="bold")
    fig.text(0.5, 0.925, "농가 제보: '드론이 날며 안 뿌리는 곳이 있다. 뿌린 곳은 진하게 올라오고 안 뿌린 곳은 연하다'",
             ha="center", fontsize=11.5, color="#555")
    # 좌: 방법/가능성/한계 텍스트
    axL = fig.add_axes([0.04, 0.08, 0.46, 0.80]); axL.axis("off")
    blocks = [
        ("■ 원리 (왜 보이나)", "#1a5276",
         "요소(N) 비료 → 엽록소↑ → GNDVI↑(진함). 안 뿌린 곳은\nGNDVI가 낮게(연하게) 남음. 드론은 밭을 '한 줄씩' 평행하게\n날며 뿌리므로, 안 뿌린 줄은 '일정 간격 평행 띠'로 보임."),
        ("■ 무엇을 측정했나", "#1a5276",
         "① 띠 방향(°)·띠 사이 간격(m)  ② 진한 띠 vs 연한 띠 GNDVI 차\n③ 시비 前→後 그 띠가 '생겼는지'(前엔 약→後 강)\n④ 모내기/기계자국 방향과 다른지."),
        ("■ 판단 가능 조건 (모두 충족시 '가능성 높음')", "#1e8449",
         "ⓐ 평행·규칙 간격 띠가 보임\nⓑ 시비 後에 생김(前엔 약함) = 살포 유래\nⓒ 진한/연한 차가 뚜렷(약 0.03 GNDVI 이상)\nⓓ 처방 그리드(5m)와 간격이 비슷."),
        ("■ 한계 (왜 '가능성'인가)", "#c0392b",
         "모내기 줄·써레 자국·배수 길도 비슷한 띠를 만듦 →\n前→後·방향으로 최대한 분리하나, 영상만으론 확정 불가.\n확정하려면 드론 '살포 비행로그/처방맵'을 영상에 겹쳐야 함."),
    ]
    y = 0.98
    for title, tc, body in blocks:
        axL.text(0.0, y, title, fontsize=12, fontweight="bold", color=tc, transform=axL.transAxes, va="top")
        axL.text(0.02, y - 0.045, body, fontsize=10, transform=axL.transAxes, va="top")
        y -= 0.045 + 0.052 * (body.count("\n") + 1) + 0.025
    # 우: 필지별 판정 요약표
    axR = fig.add_axes([0.53, 0.10, 0.44, 0.74]); axR.axis("off")
    axR.text(0.5, 1.02, "필지별 판정 요약", ha="center", fontsize=13, fontweight="bold", transform=axR.transAxes)
    rows = [["필지", "구분", "띠 간격", "진한-연한\nGNDVI차", "前→後", "판정"]]
    fieldorder = ["GJR1", "GJR2", "GJR3", "GJR4", "GJR5", "GJR6", "GJR11", "GJR14", "GJR15", "GJR16"]
    order = [c for c in fieldorder if c in assess] + [c for c in assess if c not in fieldorder]
    disp = {VARIABLE: "변량", UNIFORM: "균일", RDA: "농진청", NONE_: "무처리"}
    short = {"미살포 줄무늬 가능성 높음 (시비 後 생김)": "가능성 높음",
             "띠 있으나 시비 前부터 존재 → 미살포로 단정 어려움": "前부터 존재",
             "약한 띠 — 판단 보류": "판단 보류",
             "띠 미약 — 고르게 살포로 판단": "고르게",
             "고르게 살포 — 미살포 띠 거의 없음 (대조군)": "고르게(대조)",
             "띠 일부 있으나 대조군(참고)": "일부(대조)"}
    cell_colors = [["white"] * 6]
    for c in order:
        S = assess[c]; grp = meta[c]["grp"]
        v, vc, emerged = stripe_verdict(grp, S)
        rows.append([c, disp[grp], f"{S['A']['wavelength']:.1f}m",
                     f"{S['amp_a']:.3f}", ("생김" if emerged else "비슷"),
                     short.get(v, v)])
        cell_colors.append(["white", "white", "white", "white", "white", vc])
    tb = axR.table(cellText=rows, cellLoc="center", loc="upper center", bbox=[0, 0, 1, 1.0])
    tb.auto_set_font_size(False); tb.set_fontsize(8.5)
    for (rr, cc), cell in tb.get_celld().items():
        if rr == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif cc == 5:
            cell.set_facecolor(cell_colors[rr][5]); cell.set_text_props(color="white", fontweight="bold", fontsize=8)
    out = STR / "_overview.png"; fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return str(out)


def stripe_page(m, S):
    """한 필지의 살포 줄무늬 상세 — 필지별 구체 수치·판정."""
    code, grp = m["code"], m["grp"]
    ar = find_tif(code, AFTER_RD, "RGB")
    Ra, Rb, A = S["Ra"], S["Rb"], S["A"]
    v, vc, emerged = stripe_verdict(grp, S)
    grp_disp = {VARIABLE: "변량(대동)", UNIFORM: "균일(대동)", RDA: "농진청", NONE_: "무처리"}[grp]
    gcol = {VARIABLE: "#c0392b", UNIFORM: "#2874a6", RDA: "#1e8449", NONE_: "#7f8c8d"}[grp]
    fig = plt.figure(figsize=(14, 8.9))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.05, 1.0], hspace=0.28, wspace=0.18,
                           left=0.04, right=0.97, top=0.82, bottom=0.05)
    fig.suptitle(f"{code}  [{grp_disp}]  살포 줄무늬(미살포 선) 분석  · 이앙 {m['transplant']}", fontsize=17,
                 fontweight="bold", y=0.975, color=gcol)
    grid5 = " (처방 그리드 5m와 비슷)" if 3.5 <= A["wavelength"] <= 7.5 else ""
    fig.text(0.5, 0.932,
             f"띠 방향 {A['angle']:.0f}° · 간격 {A['wavelength']:.1f} m{grid5}  |  "
             f"진한 띠 GNDVI {S['hi']:.2f} vs 연한 띠 {S['lo']:.2f}  (차 {S['amp_a']:.3f})",
             ha="center", fontsize=12, color="#333")
    fig.text(0.5, 0.895,
             f"시비 前 같은 방향 띠 차 {S['amp_b']:.3f} → 後 {S['amp_a']:.3f}"
             f"  ({'시비 때 생김' if emerged else '前과 비슷(前부터 존재)'})",
             ha="center", fontsize=10.5, color="#666")
    # 상단: 後 GNDVI 실제 / 後 잔차 / 前 잔차
    lbl = f"평행 띠 방향 {A['angle']:.0f}° · 간격 {A['wavelength']:.1f}m"
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title("시비後 GNDVI (실제·진함=고) + 띠 위치 박스", fontsize=11)
    arr_a = Ra["arr"]; lo, hi = np.nanpercentile(arr_a, 3), np.nanpercentile(arr_a, 98)
    ax.imshow(arr_a, cmap=GREEN, vmin=lo, vmax=hi)
    draw_stripe_guide(ax, arr_a.shape, A["angle"], A["wavelength"], Ra["res_m"], label=lbl)
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); ax.set_title("시비後 띠 강조(추세제거) + 띠 박스", fontsize=11)
    r = np.where(Ra["valid"], Ra["resid"], np.nan); vm = np.nanpercentile(np.abs(r), 97)
    ax.imshow(r, cmap=DIVCM, vmin=-vm, vmax=vm)
    draw_stripe_guide(ax, r.shape, A["angle"], A["wavelength"], Ra["res_m"])
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off"); ax.set_title("시비前 띠 강조 (비교)", fontsize=11)
    if Rb is not None:
        rb = np.where(Rb["valid"], Rb["resid"], np.nan)
        ax.imshow(rb, cmap=DIVCM, vmin=-vm, vmax=vm)
    # 하단: 프로파일 前後 / RGB / 해설
    axp = fig.add_subplot(gs[1, 0]); axp.set_title("띠를 가로지른 단면 (평행 띠에 직각으로 자름)", fontsize=10)
    if S["B"] is not None and S["B"]["profile"] is not None:
        xb, pb = _prof_m(S["B"]["profile"], Rb["res_m"]); axp.plot(xb, pb, color="#9aa0a6", lw=0.9, label="시비前")
    if A["profile"] is not None:
        xa, pa = _prof_m(A["profile"], Ra["res_m"]); axp.plot(xa, pa, color="#c0392b", lw=1.1, label="시비後")
    axp.axhline(0, color="#888", lw=0.6); axp.set_xlabel("가로 위치 (m)", fontsize=8)
    axp.set_ylabel("GNDVI 잔차 (진한+/연한−)", fontsize=8); axp.tick_params(labelsize=7); axp.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); ax.set_title("시비後 RGB (대비강조)", fontsize=11)
    if ar: ax.imshow(rgb_img(ar, gain=1.25))
    axt = fig.add_subplot(gs[1, 2]); axt.axis("off")
    axt.text(0.5, 0.96, "■ 이 필지에서 보이는 것", ha="center", fontsize=11, fontweight="bold",
             transform=axt.transAxes, va="top")
    plant_txt = ""
    if S["plant"] is not None and S["dir_diff"] is not None:
        same = S["dir_diff"] <= 20
        plant_txt = (f"• 모내기/기계자국 방향 {S['plant']['angle']:.0f}°와 살포 띠 방향 "
                     f"{A['angle']:.0f}°는 {S['dir_diff']:.0f}° 차 → "
                     + ("같은 방향(구분 어려움)" if same else "다른 방향(살포 띠로 구분됨)"))
    lines = [
        f"• {A['angle']:.0f}° 방향으로 약 {A['wavelength']:.1f} m 간격 평행 띠.",
        f"• 진한 띠 GNDVI {S['hi']:.2f}, 연한 띠 {S['lo']:.2f} → 차 {S['amp_a']:.3f}.",
        f"• 시비 前 같은 방향 띠 차 {S['amp_b']:.3f} → 後 {S['amp_a']:.3f}"
        + (f" ({S['emerge']:.1f}배 ↑, 시비 때 생김)" if emerged else " (前과 비슷 = 前부터 존재)"),
    ]
    if plant_txt:
        lines.append(plant_txt)
    y = 0.86
    for ln in lines:
        axt.text(0.0, y, ln, fontsize=9.6, transform=axt.transAxes, va="top", wrap=True)
        y -= 0.052 * (ln.count("\n") + 1) + 0.045
    axt.text(0.0, y - 0.01, f"■ 판정: {v}", fontsize=11, fontweight="bold", color=vc,
             transform=axt.transAxes, va="top", wrap=True)
    out = STR / f"{code}.png"
    fig.savefig(out, dpi=135, bbox_inches="tight"); plt.close(fig)
    return str(out)


# ---------------- PDF 합치기 ----------------
def make_pdf(pages, out_path, a4=(1754, 1240), margin=36):
    def fit(p):
        cv = Image.new("RGB", a4, "white")
        im = Image.open(p).convert("RGB")
        iw, ih = im.size
        s = min((a4[0] - 2 * margin) / iw, (a4[1] - 2 * margin) / ih)
        nw, nh = int(iw * s), int(ih * s)
        cv.paste(im.resize((nw, nh), Image.LANCZOS), ((a4[0] - nw) // 2, (a4[1] - nh) // 2))
        return cv
    imgs = [fit(p) for p in pages]
    try:
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:])
        return out_path
    except PermissionError:
        alt = out_path.replace(".pdf", "_new.pdf")   # 뷰어에서 열려 잠긴 경우
        imgs[0].save(alt, save_all=True, append_images=imgs[1:])
        print(f"  [주의] {os.path.basename(out_path)} 잠김 → {os.path.basename(alt)}로 저장(뷰어 닫고 교체)")
        return alt


def main():
    meta = load_meta()
    print(f"[총 필지] {list(meta.keys())}")
    recs = {}
    for code, m in meta.items():
        bg = find_tif(code, BEFORE_RD, "GNDVI"); ag = find_tif(code, AFTER_RD, "GNDVI")
        if not bg or not ag:
            print(f"  !! {code} GNDVI 누락 (前={bool(bg)} 後={bool(ag)}) — 건너뜀"); continue
        sb = gstat(masked(load1(bg))); sa = gstat(masked(load1(ag)))
        br = find_tif(code, BEFORE_RD, "RGB"); arr = find_tif(code, AFTER_RD, "RGB")
        eb = exg_stat(br) if br else None; ea = exg_stat(arr) if arr else None
        recs[code] = dict(grp=m["grp"], mb=sb["mean"], ma=sa["mean"], cvb=sb["cv"], cva=sa["cv"],
                          p10b=sb["p10"], p10a=sa["p10"], lowb=sb["low"], lowa=sa["low"],
                          vb=sb["vals"], va=sa["vals"],
                          exgb=(eb["exg"] if eb else float("nan")),
                          exga=(ea["exg"] if ea else float("nan")))
        exgs = f"{eb['exg']:.0f}->{ea['exg']:.0f}" if (eb and ea) else "-"
        print(f"  {code} [{m['grp']}] CV {sb['cv']:.1f}->{sa['cv']:.1f} "
              f"mean {sb['mean']:.3f}->{sa['mean']:.3f} ExG {exgs}")

    order = [c for c in ["GJR1", "GJR2", "GJR3", "GJR4", "GJR5", "GJR6", "GJR11", "GJR14", "GJR15", "GJR16"]
             if c in recs]
    # ---- 효과분석 PDF ----
    cov = cover_page(recs, meta); summ = summary_page(recs, meta)
    dpages = {c: field_page(meta[c], recs[c]) for c in order}
    pages = [cov, summ] + [dpages[c] for c in order]
    pdf1 = make_pdf(pages, str(OUT / "김제_변량시비_효과분석_리포트.pdf"))
    print(f"[PDF1] {pdf1} ({len(pages)}p)")

    # ---- 선무늬(살포패턴) PDF ----
    # 변량 전필지 + 농진청/균일(대조) 분석
    targets = [c for c in order if meta[c]["grp"] in (VARIABLE, UNIFORM, RDA)]
    assess = {}
    for c in targets:
        S = assess_stripe(c)
        if S is None:
            continue
        assess[c] = S
        v, _, em = stripe_verdict(meta[c]["grp"], S)
        print(f"  [stripe] {c} [{meta[c]['grp']}] 방향 {S['A']['angle']:.0f}° 간격 {S['A']['wavelength']:.1f}m "
              f"진폭 前 {S['amp_b']:.3f}→後 {S['amp_a']:.3f} ({'생김' if em else '비슷'}) | {v}")
    ov = stripe_overview(meta, assess)
    spages = [stripe_page(meta[c], assess[c]) for c in order if c in assess]
    pdf2 = make_pdf([ov] + spages, str(OUT / "김제_살포패턴(선무늬)_분석.pdf"))
    print(f"[PDF2] {pdf2} ({1+len(spages)}p)")
    print("완료:", OUT)


if __name__ == "__main__":
    main()
