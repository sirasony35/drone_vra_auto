import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely import affinity
import math
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter, generic_filter

# ======================================================
# 0. 설정 (Pix4D Zone Detail 완벽 재현)
# ======================================================
GNDVI_PATH = "test_data/GJR1_02_250724_GNDVI.tif"
BOUNDARY_ZIP_PATH = "test_data/GJR1_Boundary.zip"
OUTPUT_FOLDER = "result_pix4d_logic"

GRID_SIZE = 1.0
N_ZONES = 5

# Pix4D의 Zone Detail 슬라이더 역할 (3.0 ~ 5.0 추천)
# 값이 클수록 '평균화'가 심해져서 빨간색/파란색이 더 많이 사라지고 노란색이 됩니다.
SMOOTHING_SIGMA = 5.0

# 짜잘한 점 제거 (5 또는 7)
MAJORITY_FILTER_SIZE = 7


# ======================================================
# 1. 정규화 스무딩 (테두리 왜곡 방지 필수)
# ======================================================
def create_normalized_smoothed_raster(input_tif, output_tif, sigma=0):
    if sigma <= 0: return input_tif
    print(f"  [Processing] 정규화 가우시안 스무딩 (Sigma={sigma})...")

    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        data = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999
        valid_mask = (data != nodata) & (~np.isnan(data)) & (data != 0)

        U = data.copy()
        U[~valid_mask] = 0
        U = np.nan_to_num(U)
        V = valid_mask.astype(float)

        smooth_U = gaussian_filter(U, sigma=sigma, mode='constant', cval=0)
        smooth_V = gaussian_filter(V, sigma=sigma, mode='constant', cval=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            smoothed_data = smooth_U / smooth_V

        smoothed_data[smooth_V < 0.1] = nodata
        smoothed_data = np.nan_to_num(smoothed_data, nan=nodata)

        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(smoothed_data, 1)

    return output_tif


# ======================================================
# 2. 그리드 및 인덱싱
# ======================================================
def get_main_angle(geometry):
    rect = geometry.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    max_len = 0;
    main_angle = 0
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0];
        dy = coords[i + 1][1] - coords[i][1]
        length = math.sqrt(dx ** 2 + dy ** 2)
        if length > max_len: max_len = length; main_angle = math.degrees(math.atan2(dy, dx))
    return main_angle


def create_rotated_grid_with_indices(boundary_gdf, grid_size=1.0):
    boundary_geom = boundary_gdf.union_all()
    rotation_angle = get_main_angle(boundary_geom)
    centroid = boundary_geom.centroid
    rotated_boundary = affinity.rotate(boundary_geom, -rotation_angle, origin=centroid)
    xmin, ymin, xmax, ymax = rotated_boundary.bounds
    cols = np.arange(xmin, xmax, grid_size)
    rows = np.arange(ymin, ymax, grid_size)

    polygons = []
    indices = []
    for c_idx, x in enumerate(cols):
        for r_idx, y in enumerate(rows):
            polygons.append(Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]))
            indices.append((c_idx, r_idx))

    grid_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=boundary_gdf.crs)
    idx_df = pd.DataFrame(indices, columns=['mat_col', 'mat_row'])
    grid_gdf = pd.concat([grid_gdf, idx_df], axis=1)
    grid_gdf['geometry'] = grid_gdf['geometry'].apply(lambda g: affinity.rotate(g, rotation_angle, origin=centroid))
    clipped_grid = gpd.clip(grid_gdf, boundary_gdf).reset_index(drop=True)
    return clipped_grid


# ======================================================
# 3. 다수결 필터 (Logical Indexing)
# ======================================================
def apply_majority_filter_logical(grid_gdf, zone_col='Zone', size=3):
    print(f"  [Post-Processing] 다수결 필터(Despeckle) 적용 (Size={size}x{size})...")
    max_col = grid_gdf['mat_col'].max()
    max_row = grid_gdf['mat_row'].max()
    matrix = np.zeros((max_row + 1, max_col + 1), dtype=int)

    for _, row in grid_gdf.iterrows():
        matrix[int(row['mat_row']), int(row['mat_col'])] = row[zone_col]

    def mode_func(values):
        valid_vals = values[values != 0]
        if len(valid_vals) == 0: return 0
        vals, counts = np.unique(valid_vals, return_counts=True)
        return vals[np.argmax(counts)]

    cleaned_matrix = generic_filter(matrix, mode_func, size=size, mode='constant', cval=0)

    new_zones = []
    for _, row in grid_gdf.iterrows():
        new_val = cleaned_matrix[int(row['mat_row']), int(row['mat_col'])]
        new_zones.append(row[zone_col] if new_val == 0 else new_val)

    grid_gdf[zone_col] = new_zones
    return grid_gdf


# ======================================================
# 4. 유틸리티
# ======================================================
def load_boundary_from_zip(zip_path):
    safe_path = zip_path.replace("\\", "/")
    if not safe_path.startswith("zip://"): safe_path = f"zip://{safe_path}"
    try:
        gdf = gpd.read_file(safe_path)
        if gdf.crs is None: gdf.crs = "EPSG:4326"
        return gdf
    except Exception as e:
        print(f"Zip 로드 실패: {e}"); return None


def calculate_zonal_stats(grid_gdf, raster_path, col_name='GNDVI_Mean'):
    stats = []
    with rasterio.open(raster_path) as src:
        if grid_gdf.crs != src.crs: grid_gdf = grid_gdf.to_crs(src.crs)
        nodata = src.nodata if src.nodata is not None else -9999
        for _, row in grid_gdf.iterrows():
            try:
                out_image, _ = mask(src, [row['geometry']], crop=True)
                data = out_image[0]
                valid_data = data[(data != nodata) & (~np.isnan(data))]
                if valid_data.size > 0:
                    stats.append(np.mean(valid_data))
                else:
                    stats.append(np.nan)
            except:
                stats.append(np.nan)
    grid_gdf[col_name] = stats
    return grid_gdf


def save_map_image(gdf, output_path, title_suffix=""):
    colors = ['#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#008000']
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(column='Zone', cmap=cmap, linewidth=0.1, edgecolor='none', ax=ax)
    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, ["1(Low)", "2", "3", "4", "5(High)"])]
    ax.legend(handles=legend_patches, loc='lower right', title="Levels")
    ax.set_title(f"Zonation Map {title_suffix}", fontsize=15)
    ax.set_axis_off()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ======================================================
# 5. 메인 실행
# ======================================================
def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    boundary = load_boundary_from_zip(BOUNDARY_ZIP_PATH)
    if boundary is None: return
    if boundary.crs.is_geographic: boundary = boundary.to_crs(epsg=5179)

    print(">>> 1. 그리드 생성")
    grid = create_rotated_grid_with_indices(boundary, grid_size=GRID_SIZE)

    # -----------------------------------------------------------
    # [1단계] 원본(Raw) 데이터에서 'Quantile' 기준을 잡습니다.
    # -----------------------------------------------------------
    print(">>> 2. 원본 데이터 분석 (Quantile 기준점 확보)")
    grid = calculate_zonal_stats(grid, GNDVI_PATH, col_name='Raw_GNDVI')
    # 결측치 제거 후 기준 산정
    raw_valid = grid.dropna(subset=['Raw_GNDVI'])

    # Quantile Cutoff 계산 (이 기준값은 고정!)
    _, bins = pd.qcut(raw_valid['Raw_GNDVI'], q=N_ZONES, retbins=True, duplicates='drop')

    # 중복값 이슈 대비 (데이터가 너무 균일할 경우)
    if len(bins) < N_ZONES + 1:
        print("  [알림] Rank 기반 Quantile 적용")
        _, bins = pd.qcut(raw_valid['Raw_GNDVI'].rank(method='first'), q=N_ZONES, retbins=True)

    print(f"  - 원본 기준 Quantile Cutoff: {np.round(bins, 4)}")

    # -----------------------------------------------------------
    # [2단계] 스무딩된 데이터(Smoothed)를 만듭니다.
    # -----------------------------------------------------------
    print(">>> 3. 데이터 스무딩 (평균화)")
    temp_tif_path = os.path.join(OUTPUT_FOLDER, "temp_smooth.tif")
    processed_tif = create_normalized_smoothed_raster(GNDVI_PATH, temp_tif_path, sigma=SMOOTHING_SIGMA)

    grid = calculate_zonal_stats(grid, processed_tif, col_name='Smooth_GNDVI')

    # -----------------------------------------------------------
    # [3단계] '원본 기준'을 '스무딩 값'에 적용합니다. (이게 핵심!)
    # 스무딩으로 인해 극단값이 사라져서, 중간 등급으로 몰리게 됩니다.
    # -----------------------------------------------------------
    print(">>> 4. 등급 부여 (Raw Threshold -> Smooth Data)")

    # pd.cut 사용 (qcut 아님!) -> 우리가 정한 bins를 강제 적용
    grid['Zone'] = pd.cut(grid['Smooth_GNDVI'], bins=bins, labels=[1, 2, 3, 4, 5], include_lowest=True)

    # 스무딩 후 범위 벗어난 값 처리 (Min보다 작아지거나 Max보다 커진 경우)
    grid['Zone'] = grid['Zone'].cat.add_categories([0, 6])
    grid.loc[grid['Smooth_GNDVI'] < bins[0], 'Zone'] = 1
    grid.loc[grid['Smooth_GNDVI'] > bins[-1], 'Zone'] = 5
    grid['Zone'] = grid['Zone'].fillna(3).astype(int)  # 혹시 남은 NaN은 중간값(3) 처리

    # 1차 저장
    save_map_image(grid, os.path.join(OUTPUT_FOLDER, "Result_Step1_LogicCheck.png"),
                   "(Step 1: Raw Cutoff on Smooth Data)")

    # -----------------------------------------------------------
    # [4단계] 다수결 필터 (Despeckle)
    # -----------------------------------------------------------
    grid = apply_majority_filter_logical(grid, zone_col='Zone', size=MAJORITY_FILTER_SIZE)

    # 최종 결과 출력
    print("\n[최종 통계 - Pix4D Logic]")
    colors = ["Red", "Orange", "Yellow", "Lt.Green", "Green"]
    stats_df = grid.groupby('Zone')
    for z in range(1, N_ZONES + 1):
        if z in stats_df.groups:
            g = stats_df.get_group(z)
            pct = (len(g) / len(grid)) * 100
            print(f" Zone {z} ({colors[z - 1]}): {pct:.1f}% ({g.geometry.area.sum():.0f} m2)")

    save_map_image(grid, os.path.join(OUTPUT_FOLDER, "Result_Step2_Final.png"), "(Step 2: Final)")

    grid_save = grid.rename(columns={'Smooth_GNDVI': 'GNDVI_Sm', 'Raw_GNDVI': 'GNDVI_Rw'})
    save_cols = [c for c in grid_save.columns if c not in ['mat_col', 'mat_row']]
    grid_save[save_cols].to_file(os.path.join(OUTPUT_FOLDER, "Result_Final.shp"), encoding='euc-kr')

    if os.path.exists(temp_tif_path): os.remove(temp_tif_path)
    print("완료되었습니다.")


if __name__ == "__main__":
    main()