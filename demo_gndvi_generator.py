"""시연(데모)용 합성 GNDVI 영상 생성기.

⚠️ 여기서 만드는 GeoTIFF는 **실제 촬영 데이터가 아니다**. 경계 shp만 있고 항공 영상이
없는 필지에서 '변량 시비 시연'을 하기 위해, 등급이 뚜렷하게 갈리도록 인위적으로 설계한
GNDVI 패턴이다. 생육 판단 근거로 쓰면 안 되고, 기체 동작·살포량 변화 시연 용도로만 쓴다.

동작:
  1. 경계(zip/shp)를 읽어 필지 주축 방향을 구한다.
  2. 주축을 따라 N개 띠(band)로 나누고 띠마다 다른 GNDVI 값을 부여한다.
     띠 순서를 섞어 배치해 기체가 비행하며 살포량이 여러 번 바뀌도록 한다.
  3. 경계 밖은 nodata(-10000)로 채워 footprint/클리핑이 정상 동작하게 한다.

사용:
    python demo_gndvi_generator.py <경계 zip|shp> <출력 tif> [GSD(m)]
"""
import os
import sys
import math

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import affinity

# 띠별 GNDVI 값 — 값이 낮을수록 생육 불량 = 살포량 많음(1등급).
# 5개 값이 분위수로 5등급에 1:1 대응되도록 충분히 벌려 둔다.
BAND_VALUES = [0.33, 0.47, 0.60, 0.68, 0.75]
# 배치 순서(주축을 따라) — 인접 띠의 살포량 차가 크도록 섞는다.
BAND_ORDER = [2, 0, 4, 1, 3]
NOISE_STD = 0.008          # 띠 안의 미세 변동(완전 균일하면 부자연스러움)
NODATA = -10000.0
TARGET_CRS = "EPSG:32652"  # 간척지 일대 UTM 52N (기존 GCJ 영상과 동일)


def load_boundary(path):
    gdf = gpd.read_file(("zip://" + path) if path.lower().endswith(".zip") else path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(TARGET_CRS)


def main_axis_angle(geom):
    """최소 회전 사각형의 긴 변 각도(도) — 띠를 필지 방향에 맞춰 자르기 위함."""
    coords = list(geom.minimum_rotated_rectangle.exterior.coords)
    best_len, angle = 0.0, 0.0
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len, angle = length, math.degrees(math.atan2(dy, dx))
    return angle


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    boundary_path, out_tif = sys.argv[1], sys.argv[2]
    gsd = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2

    gdf = load_boundary(boundary_path)
    geom = gdf.union_all()
    angle = main_axis_angle(geom)
    centroid = geom.centroid

    # 경계를 주축 기준으로 회전시켜 '띠 자르기'를 축 정렬 문제로 단순화
    rotated = affinity.rotate(geom, -angle, origin=centroid)
    rxmin, rymin, rxmax, rymax = rotated.bounds
    span = rxmax - rxmin
    n = len(BAND_VALUES)

    minx, miny, maxx, maxy = geom.bounds
    pad = gsd * 4
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
    width = int(math.ceil((maxx - minx) / gsd))
    height = int(math.ceil((maxy - miny) / gsd))
    transform = from_origin(minx, maxy, gsd, gsd)

    # 픽셀 중심 좌표 → 주축 기준 위치(0~1)로 환산해 띠 인덱스 결정
    cols = np.arange(width) + 0.5
    rows = np.arange(height) + 0.5
    xs = minx + cols * gsd
    ys = maxy - rows * gsd
    xx, yy = np.meshgrid(xs, ys)
    rad = math.radians(-angle)
    cx, cy = centroid.x, centroid.y
    rx = cx + (xx - cx) * math.cos(rad) - (yy - cy) * math.sin(rad)
    pos = np.clip((rx - rxmin) / span, 0, 0.999999)
    band_idx = (pos * n).astype(int)

    value_by_position = [BAND_VALUES[BAND_ORDER[i]] for i in range(n)]
    arr = np.take(np.array(value_by_position, dtype='float32'), band_idx)
    rng = np.random.default_rng(20260825)
    arr = arr + rng.normal(0.0, NOISE_STD, arr.shape).astype('float32')

    mask = rasterize([(geom, 1)], out_shape=(height, width), transform=transform,
                     fill=0, dtype='uint8')
    arr = np.where(mask == 1, arr, NODATA).astype('float32')

    os.makedirs(os.path.dirname(out_tif) or ".", exist_ok=True)
    profile = dict(driver="GTiff", height=height, width=width, count=1, dtype='float32',
                   crs=TARGET_CRS, transform=transform, nodata=NODATA, compress='lzw')
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr, 1)

    inside = arr[mask == 1]
    print(f"[데모 GNDVI] 저장: {out_tif}")
    print(f"  크기 {width}x{height} @ GSD {gsd}m | CRS {TARGET_CRS}")
    print(f"  필지 내 픽셀 {inside.size:,}개 | GNDVI {inside.min():.3f}~{inside.max():.3f}")
    print(f"  띠 배치(주축 방향): {[round(v, 2) for v in value_by_position]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
