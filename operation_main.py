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
# 0. 설정 (Pix4D "Fine" Default 완벽 재현)
# ======================================================
GNDVI_PATH = "test_data/GJR1_02_250724_GNDVI.tif"
BOUNDARY_ZIP_PATH = "test_data/GJR1_02_250724_Boundary.zip"
OUTPUT_FOLDER = "result_pix4d_final_v7"

GRID_SIZE = 1.0
N_ZONES = 5

# [최종 튜닝 값] 스무딩 강도 (Sigma)
# Uniform 3.0은 너무 강했고(Coarse), 0은 너무 거칠었습니다(Raw).
# 0.8은 주변 값을 살짝만 참고하여, 빨간색 영역을 적당히 남기면서 노이즈만 잡습니다.
SMOOTHING_SIGMA = 0.8

# [청소] 다수결 필터 (필수)
# Pix4D 이미지는 픽셀 단위지만 '외톨이 점'이 없습니다. Size 3으로 정리합니다.
MAJORITY_FILTER_SIZE = 3


# ======================================================
# 1. 그리드 생성 및 인덱싱
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
# 2. 유틸리티 (로드, 통계)
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


def calculate_zonal_stats(grid_gdf, raster_path, col_name='Raw_Value'):
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


# ======================================================
# 3. [복귀] Gaussian 스무딩 (Sigma 0.8)
# ======================================================
def apply_grid_smoothing(grid_gdf, value_col, sigma=0.8):
    """
    Gaussian Filter with Sigma=0.8
    - Uniform(Size=3)보다 약하고, Raw(Size=0)보다는 부드러운 'Fine' 상태를 구현합니다.
    """
    print(f"  [Algorithm] Grid Matrix Gaussian Smoothing (Sigma={sigma})...")
    max_col = grid_gdf['mat_col'].max()
    max_row = grid_gdf['mat_row'].max()

    matrix = np.full((max_row + 1, max_col + 1), np.nan)
    for _, row in grid_gdf.iterrows():
        r, c = int(row['mat_row']), int(row['mat_col'])
        matrix[r, c] = row[value_col]

    valid_mask = ~np.isnan(matrix)
    if np.sum(valid_mask) == 0: return grid_gdf

    # 결측치 채우기
    mean_val = np.nanmean(matrix)
    filled_matrix = matrix.copy()
    filled_matrix[~valid_mask] = mean_val

    # [핵심] Gaussian Filter (mode='nearest')
    smoothed_matrix = gaussian_filter(filled_matrix, sigma=sigma, mode='nearest')

    smoothed_values = []
    for _, row in grid_gdf.iterrows():
        r, c = int(row['mat_row']), int(row['mat_col'])
        val = smoothed_matrix[r, c]
        smoothed_values.append(val)

    grid_gdf['Smooth_Value'] = smoothed_values
    return grid_gdf


# ======================================================
# 4. 다수결 필터 (Size 3 필수)
# ======================================================
def apply_majority_filter_logical(grid_gdf, zone_col='Zone', size=3):
    if size <= 1: return grid_gdf
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
# 5. 시각화 및 메인
# ======================================================
def save_map_image(gdf, output_path, title_suffix="", zone_col='Zone'):
    colors = ['#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#008000']
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(column=zone_col, cmap=cmap, linewidth=0.1, edgecolor='none', ax=ax)
    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, ["1(Low)", "2", "3", "4", "5(High)"])]
    ax.legend(handles=legend_patches, loc='lower right', title="Levels")
    ax.set_title(f"Zonation Map {title_suffix}", fontsize=15)
    ax.set_axis_off()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    boundary = load_boundary_from_zip(BOUNDARY_ZIP_PATH)
    if boundary is None: return
    if boundary.crs.is_geographic: boundary = boundary.to_crs(epsg=5179)

    print(">>> 1. 그리드 생성")
    grid = create_rotated_grid_with_indices(boundary, grid_size=GRID_SIZE)

    print(">>> 2. 원본 데이터 추출")
    grid = calculate_zonal_stats(grid, GNDVI_PATH, col_name='Raw_GNDVI')
    # 결측치 제거
    raw_valid = grid.dropna(subset=['Raw_GNDVI'])

    # [기준] Raw Data 기준
    print(">>> 3. 기준점(Bins) 계산 (Raw Quantile)")
    _, bins = pd.qcut(raw_valid['Raw_GNDVI'], q=N_ZONES, retbins=True, duplicates='drop')
    if len(bins) < N_ZONES + 1:
        _, bins = pd.qcut(raw_valid['Raw_GNDVI'].rank(method='first'), q=N_ZONES, retbins=True)
    print(f"  - Cutoffs: {np.round(bins, 4)}")

    # -----------------------------------------------------------
    # [핵심] Gaussian 0.8 (적당한 뭉개짐)
    # -----------------------------------------------------------
    print(f">>> 4. 스무딩 적용 (Sigma={SMOOTHING_SIGMA})")
    grid = apply_grid_smoothing(grid, value_col='Raw_GNDVI', sigma=SMOOTHING_SIGMA)

    print(">>> 5. 등급 부여 (Raw Cutoff -> Smooth Data)")
    grid['Zone'] = pd.cut(grid['Smooth_Value'], bins=bins, labels=[1, 2, 3, 4, 5], include_lowest=True)

    grid['Zone'] = grid['Zone'].cat.add_categories([0, 6])
    grid.loc[grid['Smooth_Value'] < bins[0], 'Zone'] = 1
    grid.loc[grid['Smooth_Value'] > bins[-1], 'Zone'] = 5
    grid['Zone'] = grid['Zone'].fillna(3).astype(int)

    # -----------------------------------------------------------
    # [핵심] 다수결 3x3 (깔끔함)
    # -----------------------------------------------------------
    grid = apply_majority_filter_logical(grid, zone_col='Zone', size=MAJORITY_FILTER_SIZE)

    # 저장
    print("\n[최종 통계 - Pix4D True Fine Mode]")
    colors = ["Red", "Orange", "Yellow", "Lt.Green", "Green"]
    stats_df = grid.groupby('Zone')
    for z in range(1, N_ZONES + 1):
        if z in stats_df.groups:
            g = stats_df.get_group(z)
            pct = (len(g) / len(grid)) * 100
            print(f" Zone {z} ({colors[z - 1]}): {pct:.1f}% ({g.geometry.area.sum():.0f} m2)")

    save_map_image(grid, os.path.join(OUTPUT_FOLDER, "Result_Pix4D_True_Fine.png"), "(Pix4D True Fine)",
                   zone_col='Zone')

    grid_save = grid.rename(columns={'Smooth_Value': 'GNDVI_Sm', 'Raw_GNDVI': 'GNDVI_Rw'})
    save_cols = [c for c in grid_save.columns if c not in ['mat_col', 'mat_row']]
    grid_save[save_cols].to_file(os.path.join(OUTPUT_FOLDER, "Result_Final.shp"), encoding='euc-kr')

    print("완료되었습니다.")


if __name__ == "__main__":
    main()