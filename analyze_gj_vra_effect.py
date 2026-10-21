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
import os, glob, math
from pathlib import Path
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


def stripe_metric(path):
    """살포 줄무늬 강도를 2D FFT 환형(annulus)에서 측정.
    반환: dict(strength, wavelength_m, angle_deg, residual, valid, res_m, fft_log, peak_rc, annulus)
    strength = 환형 내 최대 피크 / 환형 중앙값 (prominence). 높을수록 주기적 줄무늬 강함.
    파장대 = 2.5~15 m (드론 살포 스와스 간격 대역)."""
    a, valid, res_m = load_for_fft(path)
    a, valid = crop_valid(a, valid)
    # 대규모 추세 제거 → 잔차
    filled = np.where(valid, a, np.nan)
    mean_v = np.nanmean(filled)
    filled = np.where(valid, filled, mean_v)
    trend = gaussian_filter(filled, sigma=max(a.shape) / 60.0)
    resid = np.where(valid, filled - trend, 0.0)
    # 윈도우(해닝)로 경계 누설 억제
    hy = np.hanning(resid.shape[0])[:, None]
    hx = np.hanning(resid.shape[1])[None, :]
    win = resid * hy * hx
    F = np.fft.fftshift(np.abs(np.fft.fft2(win)))
    cy, cx = np.array(F.shape) // 2
    yy, xx = np.indices(F.shape)
    fy = (yy - cy) / F.shape[0]   # cycles/pixel
    fx = (xx - cx) / F.shape[1]
    freq = np.sqrt(fy**2 + fx**2)      # cycles/pixel
    freq_m = freq / res_m              # cycles/meter
    lam_m = np.where(freq_m > 0, 1.0 / np.maximum(freq_m, 1e-9), 1e9)  # wavelength m
    annulus = (lam_m >= 2.5) & (lam_m <= 15.0)
    if annulus.sum() < 10:
        return None
    vals = F[annulus]
    med = np.median(vals)
    peak = vals.max()
    strength = float(peak / (med + 1e-9))
    pr, pc = np.unravel_index(np.argmax(np.where(annulus, F, -1)), F.shape)
    wl = float(lam_m[pr, pc])
    ang = float((math.degrees(math.atan2(fy[pr, pc], fx[pr, pc])) + 90) % 180)  # 줄무늬 진행방향
    return dict(strength=strength, wavelength_m=wl, angle_deg=ang,
                residual=resid, valid=valid, res_m=res_m,
                fft_log=np.log1p(F), peak_rc=(pr, pc), annulus=annulus,
                cyx=(cy, cx))


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
    b = masked(load1(bg))
    a = masked(load1(ag))
    allv = np.concatenate([b[~np.isnan(b)], a[~np.isnan(a)]])
    lo, hi = np.percentile(allv, 3), np.percentile(allv, 98)
    fig = plt.figure(figsize=(14, 9.3))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.2, 1.9], hspace=0.30, wspace=0.16,
                           left=0.04, right=0.97, top=0.82, bottom=0.04)
    gcol = {VARIABLE: "#c0392b", UNIFORM: "#2874a6", RDA: "#1e8449", NONE_: "#7f8c8d"}[grp]
    dcv = rec["cva"] - rec["cvb"]
    verdict, vcol = verdict_of(grp, dcv, m["early"])
    grp_disp = {VARIABLE: "변량(대동)", UNIFORM: "균일(대동)", RDA: "농진청", NONE_: "무처리"}[grp]
    season = f"{m['season']}·이앙 {m['transplant']}" + ("·이모작" if m["imo"] else "·일모작")
    fig.suptitle(f"{code}  [{grp_disp}]   · 경작자 {farmer}  · {m['plan']}  · {season}", fontsize=17,
                 fontweight="bold", y=0.975, color=gcol)
    sp = f"살포 {m['spray']}" if m["spray"] and m["spray"] not in ("살포안함",) else m["dronecol"]
    fig.text(0.5, 0.935,
             f"CV {rec['cvb']:.1f}% → {rec['cva']:.1f}% ({dcv:+.1f}%p, {verdict})   |   "
             f"ExG(녹색도) {f0(rec['exgb'])}→{f0(rec['exga'])}   |   {sp}",
             ha="center", fontsize=12, color="#333")
    note2 = "※ CV(고르기 편차)는 낮을수록 균일 · ΔCV 음수(-)=개선  |  前=0724/0729, 後=0811"
    if m["early"] and grp != NONE_:
        note2 += "  |  조기이앙 성숙 → 03시점 이미 고생육, 변화 작게 나타남"
    fig.text(0.5, 0.90, note2, ha="center", fontsize=9.5, color="#888")
    # row1: 前 GNDVI / 後 GNDVI / 히스토그램
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(b, cmap=GREEN, vmin=lo, vmax=hi); ax.axis("off")
    ax.set_title("시비前 GNDVI (03회차)", fontsize=11)
    ax = fig.add_subplot(gs[0, 1]); im = ax.imshow(a, cmap=GREEN, vmin=lo, vmax=hi); ax.axis("off")
    ax.set_title("시비後 GNDVI (04회차)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    axh = fig.add_subplot(gs[0, 2]); bins = np.linspace(0.2, 0.95, 55)
    axh.hist(rec["vb"], bins=bins, alpha=0.55, color="#e67e22", density=True, label=f"前 {rec['cvb']:.1f}%")
    axh.hist(rec["va"], bins=bins, alpha=0.55, color="#27ae60", density=True, label=f"後 {rec['cva']:.1f}%")
    axh.set_title("GNDVI 분포 前/後", fontsize=10); axh.legend(fontsize=8); axh.set_yticks([])
    axh.set_xlabel("GNDVI", fontsize=9)
    # row2: 前 RGB / 後 RGB / 지표표
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off"); ax.set_title("시비前 RGB", fontsize=11)
    if br: ax.imshow(rgb_img(br))
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); ax.set_title("시비後 RGB", fontsize=11)
    if ar: ax.imshow(rgb_img(ar))
    axt = fig.add_subplot(gs[1, 2]); axt.axis("off")
    rows = [["지표", "시비前", "시비後", "Δ"],
            ["평균 GNDVI", f"{rec['mb']:.3f}", f"{rec['ma']:.3f}", f"{rec['ma']-rec['mb']:+.3f}"],
            ["CV (균일도)", f"{rec['cvb']:.1f}%", f"{rec['cva']:.1f}%", f"{dcv:+.1f}%p"],
            ["P10 (저위)", f"{rec['p10b']:.3f}", f"{rec['p10a']:.3f}", f"{rec['p10a']-rec['p10b']:+.3f}"],
            ["저활력<0.5", f"{rec['lowb']:.1f}%", f"{rec['lowa']:.1f}%", f"{rec['lowa']-rec['lowb']:+.1f}%p"],
            ["ExG 녹색도", f0(rec['exgb']), f0(rec['exga']), (f"{dexg:+.0f}" if np.isfinite(dexg) else "-")]]
    tb = axt.table(cellText=rows, cellLoc="center", loc="upper center", bbox=[0.0, 0.42, 1.0, 0.56])
    tb.auto_set_font_size(False); tb.set_fontsize(9)
    for (rr, cc), cell in tb.get_celld().items():
        if rr == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
    axt.text(0.5, 0.34, f"■ 판정: {verdict}", ha="center", fontsize=11, transform=axt.transAxes,
             color=vcol, fontweight="bold")
    axt.text(0.5, 0.24, f"P10 {rec['p10a']-rec['p10b']:+.3f} · 저활력 {rec['lowa']-rec['lowb']:+.1f}%p"
             f"  ·  평균GNDVI {rec['mb']:.2f}→{rec['ma']:.2f}",
             ha="center", fontsize=9.3, transform=axt.transAxes, color="#555")
    if m["early"] and grp != NONE_:
        axt.text(0.5, 0.14, "조기이앙으로 前 이미 성숙 → 변화 작음(악화 아님)",
                 ha="center", fontsize=8.8, transform=axt.transAxes, color="#e67e22")
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
    fig.text(0.5, 0.68, "시비 前(03회차) · 시비 後(04회차, 0811) GNDVI+RGB 종합 · 균일도(CV) 검증",
             ha="center", fontsize=13, color="#333")
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


# ---------------- 선무늬(미살포 패턴) 분석 페이지 ----------------
def stripe_page(m, sb, sa):
    """한 필지의 시비 前(sb)/後(sa) 줄무늬 비교 페이지."""
    code, grp = m["code"], m["grp"]
    ar = find_tif(code, AFTER_RD, "RGB")
    fig = plt.figure(figsize=(14, 8.8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.26, wspace=0.18,
                           left=0.04, right=0.97, top=0.84, bottom=0.05)
    grp_disp = {VARIABLE: "변량(대동)", UNIFORM: "균일(대동)", RDA: "농진청", NONE_: "무처리"}[grp]
    gcol = {VARIABLE: "#c0392b", UNIFORM: "#2874a6", RDA: "#1e8449", NONE_: "#7f8c8d"}[grp]
    ds = (sa["strength"] - sb["strength"]) if (sb and sa) else float("nan")
    fig.suptitle(f"{code}  [{grp_disp}]  살포 줄무늬(미살포 선) 분석", fontsize=18, fontweight="bold",
                 y=0.975, color=gcol)
    if sa:
        fig.text(0.5, 0.925,
                 f"줄무늬 강도  前 {sb['strength']:.1f} → 後 {sa['strength']:.1f}  (Δ {ds:+.1f})  |  "
                 f"띠 간격 ~{sa['wavelength_m']:.1f} m  ·  방향 {sa['angle_deg']:.0f}°",
                 ha="center", fontsize=12, color="#333")
    fig.text(0.5, 0.89,
             "※ 줄무늬 강도 = 시비 後 GNDVI의 주기적 띠(스와스) 세기(FFT). 높을수록 '선 긋듯' 미살포 띠가 뚜렷",
             ha="center", fontsize=9.3, color="#888")
    # 상단: 後 RGB / 後 GNDVI 잔차 / FFT
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.set_title("시비後 RGB (대비강조)", fontsize=11)
    if ar: ax.imshow(rgb_img(ar, gain=1.25))
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); ax.set_title("시비後 GNDVI 잔차(추세제거)", fontsize=11)
    if sa:
        r = np.where(sa["valid"], sa["residual"], np.nan)
        vmax = np.nanpercentile(np.abs(r), 97)
        ax.imshow(r, cmap=DIVCM, vmin=-vmax, vmax=vmax)
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off"); ax.set_title("FFT (주기성) · 환형=스와스 대역", fontsize=11)
    if sa:
        fl = sa["fft_log"]; cy, cx = sa["cyx"]
        ax.imshow(fl, cmap="magma")
        pr, pc = sa["peak_rc"]; ax.plot(pc, pr, "o", mfc="none", mec="#00e5ff", ms=14, mew=2)
        ax.plot(cx, cy, "+", color="w", ms=8)
    # 하단: 前 GNDVI 잔차 / 프로파일 / 설명
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off"); ax.set_title("시비前 GNDVI 잔차", fontsize=11)
    if sb:
        r = np.where(sb["valid"], sb["residual"], np.nan)
        vmax = np.nanpercentile(np.abs(r), 97)
        ax.imshow(r, cmap=DIVCM, vmin=-vmax, vmax=vmax)
    axp = fig.add_subplot(gs[1, 1]); axp.set_title("띠 횡단 프로파일(後, 평행띠에 수직)", fontsize=10)
    if sa:
        r = np.where(sa["valid"], sa["residual"], np.nan)
        # 줄무늬에 수직인 축(=평균 프로파일 변동이 큰 축)으로 투영
        with np.errstate(invalid="ignore"):
            px = np.nanmean(r, axis=0); py = np.nanmean(r, axis=1)
        prof = py if np.nanstd(py) >= np.nanstd(px) else px
        prof = prof[np.isfinite(prof)]
        axp.plot(np.arange(len(prof)) * sa["res_m"], prof, color="#c0392b", lw=0.9)
        axp.axhline(0, color="#888", lw=0.6); axp.set_xlabel("띠 횡단 위치 (m)", fontsize=8)
        axp.set_ylabel("GNDVI 잔차", fontsize=8); axp.tick_params(labelsize=7)
    axt = fig.add_subplot(gs[1, 2]); axt.axis("off")
    if sa:
        if grp == VARIABLE and ds >= 2:
            judge = "[검증됨] 변량 — 시비 후 평행 띠 증가"
            jc = "#c0392b"
        elif grp in (RDA, UNIFORM) and ds <= 0:
            judge = "[대조] 농진청/균일 — 띠 증가 없음(고르게 살포)"
            jc = "#1e8449"
        elif grp == VARIABLE and ds > 0:
            judge = "변량 — 띠 소폭 증가"
            jc = "#e67e22"
        else:
            judge = "띠 증가 없음(Δ≤0)"
            jc = "#555"
        axt.text(0.5, 0.88, judge, ha="center", fontsize=12.5, fontweight="bold", color=jc, transform=axt.transAxes)
        note = ("드론 변량살포는 처방 등급에 따라 살포량을 바꾸며 비행 띠(스와스)를 따라\n"
                "줄무늬가 생길 수 있음. 잔차맵·FFT의 일정 간격 평행 띠 = '선 긋듯' 미살포/과살포.\n\n"
                "※ 모내기 줄·기계 자국도 띠로 보이며 시비 前에도 존재 →\n"
                "   '살포 유래' 띠는 前→後 강도 증가(Δ>0)로 분리해 판단.\n"
                "   농진청/균일 살포는 전면 균등 → Δ≤0(대조군).")
        axt.text(0.03, 0.72, note, fontsize=9.5, transform=axt.transAxes, va="top", wrap=True)
    out = STR / f"{code}.png"
    fig.savefig(out, dpi=135, bbox_inches="tight"); plt.close(fig)
    return str(out)


def stripe_summary(meta, strengths):
    """필지별 살포유래 줄무늬 = Δ(後-前) 랭킹 요약.
    절대강도는 모내기 줄·기계자국을 포함하므로 Δ로 '살포 유래' 띠를 분리한다."""
    fig = plt.figure(figsize=(13.8, 8.2)); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.95, "살포 줄무늬(미살포 선) 검증 — 시비 前→後 변화(Δ)", ha="center",
             fontsize=20, fontweight="bold")
    fig.text(0.5, 0.905,
             "농가 제보: '농진청 살포는 고른데 대동 변량한 곳은 선 긋듯 안 뿌려진 띠가 있다'  →  드론영상 검증",
             ha="center", fontsize=11.5, color="#555")
    order = sorted(strengths.items(), key=lambda kv: -(kv[1]["after"] - kv[1]["before"]))
    names = [k for k, _ in order]
    before = [v["before"] for _, v in order]
    after = [v["after"] for _, v in order]
    delta = [a - b for a, b in zip(after, before)]
    gcolmap = {VARIABLE: "#c0392b", UNIFORM: "#2874a6", RDA: "#1e8449", NONE_: "#bdc3c7"}
    cols = [gcolmap[meta[k]["grp"]] for k in names]
    # 상단: Δ (살포유래 띠) — 핵심
    ax1 = fig.add_axes([0.08, 0.50, 0.88, 0.34])
    ax1.bar(names, delta, color=cols)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("Δ 줄무늬 (後−前)\n양수=시비때 띠 생김")
    ax1.set_title("① 살포 유래 줄무늬 (모내기 줄 제거한 순효과) — 양수일수록 '선 긋듯' 미살포", fontsize=12)
    for i, d in enumerate(delta):
        ax1.text(i, d, f"{d:+.1f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=9, fontweight="bold")
    ax1.set_xticklabels([])
    # 하단: 前/後 절대강도 (모내기 줄 포함, 참고)
    ax2 = fig.add_axes([0.08, 0.10, 0.88, 0.30])
    x = np.arange(len(names)); w = 0.38
    ax2.bar(x - w / 2, before, w, color="#bbbbbb", label="前(03)")
    ax2.bar(x + w / 2, after, w, color=cols, label="後(04)")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30)
    ax2.set_ylabel("절대 줄무늬 강도\n(모내기 줄 포함)")
    ax2.set_title("② 절대 강도 前/後 (참고 — 모내기 줄·기계자국 포함이라 기준선 높음)", fontsize=11)
    ax2.legend(handles=[mpatches.Patch(color="#c0392b", label="변량(대동)"),
                        mpatches.Patch(color="#2874a6", label="균일(대동)"),
                        mpatches.Patch(color="#1e8449", label="농진청"),
                        mpatches.Patch(color="#bbbbbb", label="前(03)")],
               fontsize=8.5, ncol=4, loc="upper right")
    out = STR / "_ranking.png"; fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)
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
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:])
    return out_path


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
    strengths = {}
    spages = []
    # 변량 전필지 + 농진청/균일(대조) 분석
    targets = [c for c in order if meta[c]["grp"] in (VARIABLE, UNIFORM, RDA)]
    for c in targets:
        bg = find_tif(c, BEFORE_RD, "GNDVI"); ag = find_tif(c, AFTER_RD, "GNDVI")
        sb = stripe_metric(bg) if bg else None
        sa = stripe_metric(ag) if ag else None
        if sa:
            strengths[c] = dict(before=sb["strength"] if sb else float("nan"), after=sa["strength"])
            print(f"  [stripe] {c} [{meta[c]['grp']}] 前 "
                  f"{sb['strength'] if sb else float('nan'):.1f} -> 後 {sa['strength']:.1f} "
                  f"(λ={sa['wavelength_m']:.1f}m)")
        spages.append(stripe_page(meta[c], sb, sa))
    rank = stripe_summary(meta, strengths)
    pdf2 = make_pdf([rank] + spages, str(OUT / "김제_살포패턴(선무늬)_분석.pdf"))
    print(f"[PDF2] {pdf2} ({1+len(spages)}p)")
    print("완료:", OUT)


if __name__ == "__main__":
    main()
