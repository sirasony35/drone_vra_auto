# -*- coding: utf-8 -*-
"""전체 필지 개요 지도 (overview_map.py)

결과 폴더의 DJI Rx GeoTIFF들을 위성 배경지도(Esri World Imagery) 위에 한 장으로 합쳐
개요 PNG를 생성한다. 필지 경계선·라벨(필지명 / 총량·포대수 / 평)·등급 범례·전체 합계 표기.

사용법:
    python overview_map.py [결과폴더] [출력파일명]
    - 결과폴더 기본값: result/ 하위 가장 최근의 *dji* 폴더
    - 출력 기본값: <결과폴더>/개요지도.png

구현 요점:
  - Rx GeoTIFF에는 등급이 아니라 kg/ha가 저장돼 있어, 같은 필지의 *_VRA.csv 등급별
    Rate(kg/ha)와 '최근접 매칭'으로 각 픽셀의 등급을 역산한다(필지마다 rate가 달라 필지별 매핑).
  - 배경 타일과 좌표계를 맞추기 위해 Rx bounds를 EPSG:3857로 변환해 imshow(extent=...).
  - contextily 실패(오프라인 등) 시 배경 없이 그리고 경고만 출력.
  - MAX_ZOOM=18: 소필지 1개만 그릴 때 자동 줌이 Esri 제공 한계를 넘어 회색 타일이 깔리는
    현상 방지(다필지 개요 화질에는 영향 없음).

※ 2026-08-25 집 PC 작업분이 커밋되지 않아 유실 → 2026-08-?? memory.md 스펙대로 복원.
"""
import os
import sys
import glob
import csv
import math
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import font_manager

try:
    import contextily as ctx
    HAS_CTX = True
except Exception:
    HAS_CTX = False

PYEONG_M2 = 3.3057851
BAG_KG = 20.0
MAX_ZOOM = 18
# DJI 5등급 색상(operation_main.save_map_image와 동일). 1=High(빨강)···5=Low(초록)
ZONE_COLORS = ['#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#008000']
ZONE_LABELS = ["1(High)", "2", "3", "4", "5(Low)"]


def _setup_korean_font():
    for name in ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "Gulim", "Batang", "Dotum"]:
        if name in {f.name for f in font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def _find_vra(res_folder, name):
    for pat in (f"{res_folder}/{name}_VRA.csv", f"{res_folder}/*{name}*_VRA.csv"):
        g = glob.glob(pat)
        if g:
            return g[0]
    return None


def _read_vra(path):
    """(등급,rate) 목록과 총량(kg) 반환. euc-kr 등 인코딩 자동 감지."""
    for enc in ("euc-kr", "utf-8-sig", "cp949", "utf-8"):
        try:
            rows = list(csv.DictReader(open(path, encoding=enc)))
            zone_rate = []
            total = 0.0
            for r in rows:
                try:
                    z = int(str(r["Zone"]).split("(")[0])
                    rate = float(r["Rate(kg/ha)"])
                    tk = float(r["Total(kg)"])
                except (ValueError, KeyError):
                    continue
                if 1 <= z <= 5 and rate > 0:
                    zone_rate.append((z, rate))
                total += tk
            return zone_rate, total
        except (UnicodeDecodeError, KeyError):
            continue
    return [], 0.0


def _rx_to_zone(a, zone_rate):
    """kg/ha 배열 → 등급 배열(최근접 rate). 0/nodata → NaN(skip)."""
    if not zone_rate:
        return a
    zones = np.array([z for z, _ in zone_rate])
    rates = np.array([r for _, r in zone_rate])
    out = np.full(a.shape, np.nan, dtype="float32")
    m = np.isfinite(a) & (a > 0)
    if m.any():
        idx = np.argmin(np.abs(a[m][:, None] - rates[None, :]), axis=1)
        out[m] = zones[idx]
    return out


def _calc_zoom(xmin, xmax, target_tiles=4):
    """3857 지도 폭 기준 적정 타일 줌 계산 후 MAX_ZOOM으로 상한."""
    world = 40075016.686
    mpt = max((xmax - xmin) / max(target_tiles, 1), 1.0)
    z = int(math.floor(math.log2(world / mpt)))
    return max(0, min(MAX_ZOOM, z))


def build_overview(res_folder, out_path):
    _setup_korean_font()
    rx_files = sorted(glob.glob(os.path.join(res_folder, "DJI", "Rx", "*.tif")))
    if not rx_files:
        print(f"[오류] {res_folder}/DJI/Rx 에 Rx GeoTIFF가 없습니다.")
        return False

    fig, ax = plt.subplots(figsize=(15, 11))
    xs, ys = [], []
    grand_total = 0.0
    n_field = 0
    for f in rx_files:
        name = os.path.splitext(os.path.basename(f))[0]
        vpath = _find_vra(res_folder, name)
        zone_rate, total = _read_vra(vpath) if vpath else ([], 0.0)
        with rasterio.open(f) as s:
            a = s.read(1).astype("float32")
            a = np.where(a > -1e4, a, np.nan)
            l, b, r, t = transform_bounds(s.crs, "EPSG:3857", *s.bounds)
        za = _rx_to_zone(a, zone_rate)
        ax.imshow(za, cmap=ListedColormap(ZONE_COLORS), vmin=1, vmax=5,
                  extent=[l, r, b, t], origin="upper", zorder=2, interpolation="nearest")
        # 경계선 + 면적(평)
        area_py = None
        shp = glob.glob(os.path.join(res_folder, "DJI", "ShapeFile", f"{name}.shp"))
        if shp:
            try:
                g3857 = gpd.read_file(shp[0]).to_crs("EPSG:3857")
                g3857.boundary.plot(ax=ax, color="cyan", linewidth=1.3, zorder=3)
                area_py = gpd.read_file(shp[0]).to_crs(5179).geometry.area.sum() / PYEONG_M2
            except Exception:
                pass
        cx, cy = (l + r) / 2, (b + t) / 2
        xs += [l, r]
        ys += [b, t]
        grand_total += total
        n_field += 1
        lab = f"{name}\n{total:,.0f}kg ({total / BAG_KG:.0f}포)"
        if area_py:
            lab += f" · {area_py:,.0f}평"
        ax.annotate(lab, (cx, cy), ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#000000aa", ec="none"))

    dx = (max(xs) - min(xs)) * 0.06 + 50
    dy = (max(ys) - min(ys)) * 0.06 + 50
    ax.set_xlim(min(xs) - dx, max(xs) + dx)
    ax.set_ylim(min(ys) - dy, max(ys) + dy)
    ax.set_axis_off()
    if HAS_CTX:
        try:
            ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Esri.WorldImagery,
                            zoom=_calc_zoom(min(xs) - dx, max(xs) + dx), zorder=1, attribution_size=5)
        except Exception as e:
            print(f"[경고] 배경지도 로드 실패 — 배경 없이 진행: {e}")
    else:
        print("[경고] contextily 미설치 — 배경 없이 진행 (pip install contextily)")

    ax.set_title(f"{os.path.basename(res_folder.rstrip('/'))} — 전체 {n_field}필지  |  "
                 f"총 {grand_total:,.0f}kg ({grand_total / BAG_KG:.0f}포)",
                 fontsize=15, fontweight="bold")
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(ZONE_COLORS, ZONE_LABELS)]
    ax.legend(handles=patches, loc="lower right", title="등급 (살포량 High→Low)", fontsize=9, framealpha=0.9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[완료] 개요지도 저장: {out_path}  ({n_field}필지, 총 {grand_total:,.0f}kg / {grand_total / BAG_KG:.0f}포)")
    return True


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else None
    if not res:
        cands = sorted(glob.glob("result/*dji*"), key=os.path.getmtime)
        res = cands[-1] if cands else None
    if not res or not os.path.isdir(res):
        print("결과 폴더를 찾을 수 없습니다.\n사용법: python overview_map.py [결과폴더] [출력파일명]")
        return
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(res, "개요지도.png")
    build_overview(res, out)


if __name__ == "__main__":
    main()
