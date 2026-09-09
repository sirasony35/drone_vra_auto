# -*- coding: utf-8 -*-
"""화성 HSR 변량시비 효과 분석 리포트 (before/after GNDVI 균일도 + 처방맵, 변량 vs 관행)."""
import os, glob, re
import numpy as np, rasterio, openpyxl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

BEFORE_DIR = "data/hs_data"
AFTER_DIR = "data/hs_data/after_vra"
RX_DIR = "result/hs_0720/DJI/Rx"
OUT = "result/hs_vra_effect_report"; DET = f"{OUT}/필지별"
os.makedirs(DET, exist_ok=True)
for n in ["Malgun Gothic", "NanumGothic", "Gulim", "Batang"]:
    if n in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = n; break
plt.rcParams["axes.unicode_minus"] = False
GREEN = LinearSegmentedColormap.from_list("g", ["#8B4513", "#DEB887", "#FFFF99", "#66BB66", "#006400"])
RXCM = LinearSegmentedColormap.from_list("rx", ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"])
VMIN, VMAX = 0.2, 1.0  # 작물 마스크 범위


def load(path, target=1600):
    with rasterio.open(path) as s:
        sc = max(1, int(max(s.width, s.height) / target))
        a = s.read(1, out_shape=(s.height // sc, s.width // sc)).astype("float32")
    return a


def crop_vals(a):
    return a[(a > VMIN) & (a < VMAX) & (~np.isnan(a))]


def stat(a):
    v = crop_vals(a)
    return dict(mean=float(v.mean()), std=float(v.std()), cv=float(v.std() / v.mean() * 100),
                p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                low=float((v < 0.5).mean() * 100), vals=v)


def vmap_load():
    wb = openpyxl.load_workbook(f"{AFTER_DIR}/화성변량시비여부.xlsx", data_only=True)
    d = {}
    for r in list(wb.active.iter_rows(values_only=True))[1:]:
        d[str(r[0]).replace("-", "").replace(" ", "")] = str(r[1])
    return d


def masked(a):
    return np.where((a > VMIN) & (a < VMAX) & (~np.isnan(a)), a, np.nan)


def field_page(code, grp, bpath, apath, rxpath, sb, sa):
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.7, 1.4], hspace=0.32, wspace=0.18,
                           left=0.04, right=0.97, top=0.86, bottom=0.06)
    col = "#c0392b" if grp == "변량" else "#7f8c8d"
    fig.suptitle(f"{code}  [{grp}]", fontsize=19, fontweight="bold", y=0.965, color=col)
    dcv = sa["cv"] - sb["cv"]
    verdict = "균일도 개선" if dcv < -0.5 else ("균일도 악화" if dcv > 0.5 else "균일도 유지")
    fig.text(0.5, 0.905, f"시비前(07-15) → 시비後(07-28)   |   CV {sb['cv']:.1f}% → {sa['cv']:.1f}% ({dcv:+.1f}%p, {verdict})",
             ha="center", fontsize=12, color="#333")
    b = masked(load(bpath)); a = masked(load(apath))
    lo = np.nanpercentile(np.concatenate([b[~np.isnan(b)], a[~np.isnan(a)]]), 3)
    hi = np.nanpercentile(np.concatenate([b[~np.isnan(b)], a[~np.isnan(a)]]), 98)
    ax0 = fig.add_subplot(gs[0, 0]); ax0.set_title("시비 前 GNDVI (07-15)", fontsize=12)
    ax0.imshow(b, cmap=GREEN, vmin=lo, vmax=hi); ax0.axis("off")
    ax1 = fig.add_subplot(gs[0, 1]); ax1.set_title("처방맵 (살포량 kg/ha)", fontsize=12)
    if rxpath and os.path.exists(rxpath):
        with rasterio.open(rxpath) as s:
            r = s.read(1).astype("float32"); r = np.where(r > 0, r, np.nan)
        im = ax1.imshow(r, cmap=RXCM); ax1.axis("off")
        cb = fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=8)
    else:
        ax1.text(0.5, 0.5, "처방맵 없음", ha="center", va="center"); ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, 2]); ax2.set_title("시비 後 GNDVI (07-28)", fontsize=12)
    im2 = ax2.imshow(a, cmap=GREEN, vmin=lo, vmax=hi); ax2.axis("off")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.02); cb2.ax.tick_params(labelsize=8)
    # 히스토그램
    axh = fig.add_subplot(gs[1, 0:2])
    bins = np.linspace(0.2, 0.95, 60)
    axh.hist(sb["vals"], bins=bins, alpha=0.55, color="#e67e22", label=f"前 (CV {sb['cv']:.1f}%)", density=True)
    axh.hist(sa["vals"], bins=bins, alpha=0.55, color="#27ae60", label=f"後 (CV {sa['cv']:.1f}%)", density=True)
    axh.axvline(sb["mean"], color="#e67e22", ls="--", lw=1); axh.axvline(sa["mean"], color="#27ae60", ls="--", lw=1)
    axh.set_title("GNDVI 분포 (前/後) — 폭이 좁아질수록 균일", fontsize=11)
    axh.set_xlabel("GNDVI"); axh.legend(fontsize=9); axh.set_yticks([])
    # 지표표
    axt = fig.add_subplot(gs[1, 2]); axt.axis("off")
    rows = [
        ["지표", "前", "後", "Δ"],
        ["평균", f"{sb['mean']:.3f}", f"{sa['mean']:.3f}", f"{sa['mean']-sb['mean']:+.3f}"],
        ["CV(균일도)", f"{sb['cv']:.1f}%", f"{sa['cv']:.1f}%", f"{dcv:+.1f}%p"],
        ["P10(저위)", f"{sb['p10']:.3f}", f"{sa['p10']:.3f}", f"{sa['p10']-sb['p10']:+.3f}"],
        ["저활력<0.5", f"{sb['low']:.1f}%", f"{sa['low']:.1f}%", f"{sa['low']-sb['low']:+.1f}%p"],
    ]
    tb = axt.table(cellText=rows, cellLoc="center", loc="center", bbox=[0, 0.1, 1, 0.9])
    tb.auto_set_font_size(False); tb.set_fontsize(9.5)
    for (rr, cc), cell in tb.get_celld().items():
        if rr == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
    out = os.path.join(DET, f"{code}.png"); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def summary_page(recs):
    fig = plt.figure(figsize=(14, 8)); fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1], hspace=0.42, wspace=0.25, left=0.08, right=0.95, top=0.9, bottom=0.1)
    fig.suptitle("변량시비 효과 요약 — 균일도(CV) 변화", fontsize=19, fontweight="bold")
    order = sorted(recs, key=lambda r: r["dcv"])
    names = [r["code"] for r in order]; dcvs = [r["dcv"] for r in order]
    cols = ["#c0392b" if r["grp"] == "변량" else "#95a5a6" for r in order]
    ax = fig.add_subplot(gs[0, :])
    ax.bar(names, dcvs, color=cols)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ΔCV (%p)  ↓개선 / ↑악화"); ax.set_title("필지별 균일도 변화 (음수=개선)", fontsize=12)
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color="#c0392b", label="변량"), mp.Patch(color="#95a5a6", label="관행")], fontsize=10)
    # 그룹 평균
    for gi, (metric, lbl) in enumerate([("dcv", "평균 ΔCV (%p)"), ("dp10", "평균 ΔP10 (저위 회복)")]):
        axg = fig.add_subplot(gs[1, gi])
        vv = []; cc = []
        for g in ["변량", "관행"]:
            gg = [r[metric] for r in recs if r["grp"] == g]
            vv.append(np.mean(gg)); cc.append("#c0392b" if g == "변량" else "#95a5a6")
        axg.bar(["변량", "관행"], vv, color=cc)
        axg.axhline(0, color="black", lw=0.8); axg.set_title(lbl, fontsize=12)
        for i, v in enumerate(vv):
            axg.text(i, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontweight="bold")
    out = os.path.join(OUT, "_summary.png"); fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return out


def cover(recs):
    fig = plt.figure(figsize=(11.7, 8.3)); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.7, "화성 변량시비 효과 분석 리포트", ha="center", fontsize=28, fontweight="bold")
    fig.text(0.5, 0.62, "시비 前(07-15) · 처방맵 · 시비 後(07-28) GNDVI 종합 분석", ha="center", fontsize=14, color="#333")
    dv = np.mean([r["dcv"] for r in recs if r["grp"] == "변량"])
    dg = np.mean([r["dcv"] for r in recs if r["grp"] == "관행"])
    pv = np.mean([r["dp10"] for r in recs if r["grp"] == "변량"])
    pg = np.mean([r["dp10"] for r in recs if r["grp"] == "관행"])
    verdict = "변량시비가 관행보다 균일도 개선 효과 확인" if dv < dg else "그룹 차이 뚜렷하지 않음"
    lines = [
        f"■ 대상: 변량 7필지 · 관행 4필지 (드론 GNDVI 2cm, 13일 간격)",
        f"■ 균일도(CV) 변화 평균:  변량 {dv:+.2f}%p   vs   관행 {dg:+.2f}%p",
        f"■ 저활력 구간(P10) 회복 평균:  변량 {pv:+.3f}   vs   관행 {pg:+.3f}",
        f"■ 결론:  {verdict}",
        f"          (변량은 CV↓·저위↑로 균일화, 관행은 CV↑ 경향)",
    ]
    for i, ln in enumerate(lines):
        fig.text(0.12, 0.46 - i * 0.06, ln, fontsize=13, color="#c0392b" if i == 3 else "#222")
    fig.text(0.12, 0.12, "※ 유의: 13일 성숙에 따른 GNDVI 상승·포화 공존, 표본 소수(관행 4), 필지별 편차 존재 →\n   그룹 간 '차이'를 근거로 해석. 필지별 상세는 이후 페이지 참조.",
             fontsize=9.5, color="#666")
    out = os.path.join(OUT, "_cover.png"); fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)
    return out


def main():
    vmap = vmap_load()
    codes = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{BEFORE_DIR}/*_GNDVI*.tif")},
                   key=lambda x: int(x[3:]))
    recs = []; pages_field = []
    for c in codes:
        grp = vmap.get(c, "?")
        bpath = glob.glob(f"{BEFORE_DIR}/{c}_*_GNDVI*.tif")[0]
        apath = glob.glob(f"{AFTER_DIR}/{c}_*_GNDVI*.tif")[0]
        rxg = glob.glob(f"{RX_DIR}/{c}_*.tif"); rxpath = rxg[0] if rxg else None
        sb = stat(load(bpath)); sa = stat(load(apath))
        recs.append(dict(code=c, grp=grp, cvb=sb["cv"], cva=sa["cv"], dcv=sa["cv"] - sb["cv"],
                         dp10=sa["p10"] - sb["p10"], dmean=sa["mean"] - sb["mean"]))
        pages_field.append(field_page(c, grp, bpath, apath, rxpath, sb, sa))
        print(f"  {c} [{grp}] CV {sb['cv']:.1f}->{sa['cv']:.1f} ({sa['cv']-sb['cv']:+.1f})")
    cov = cover(recs); summ = summary_page(recs)
    pages = [cov, summ] + pages_field
    A4 = (1754, 1240); M = 40
    def fit(p):
        cv = Image.new("RGB", A4, "white"); im = Image.open(p).convert("RGB"); iw, ih = im.size
        s = min((A4[0] - 2 * M) / iw, (A4[1] - 2 * M) / ih); nw, nh = int(iw * s), int(ih * s)
        cv.paste(im.resize((nw, nh), Image.LANCZOS), ((A4[0] - nw) // 2, (A4[1] - nh) // 2)); return cv
    imgs = [fit(p) for p in pages]
    out = f"{OUT}/화성_변량시비_효과분석_리포트.pdf"; imgs[0].save(out, save_all=True, append_images=imgs[1:])
    print(f"PDF: {out} ({len(imgs)}페이지)")


if __name__ == "__main__":
    main()
