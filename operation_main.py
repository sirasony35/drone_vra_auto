import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from shapely.geometry import Polygon
from shapely import affinity
import math
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter, generic_filter
from scipy.interpolate import NearestNDInterpolator

# [경고 제거]
pd.set_option('future.no_silent_downcasting', True)

# ======================================================
# 0. 설정 (Pix4D Logic V30 - Clip First + Target Dist)
# ======================================================
GNDVI_PATH = "test_data/GJR2_02_250724_GNDVI.tif"
BOUNDARY_ZIP_PATH = "test_data/GJR2_02_250724_Boundary.zip"
OUTPUT_FOLDER = "result_pix4d_v30_final"

GRID_SIZE = 1.0
N_ZONES = 5

# [데이터 범위] -999 이상 (전체 범위 사용)
VALID_THRESHOLD = -999.0

# [스무딩 강도] Sigma 0.6 (디테일 유지)
ZONE_DETAIL_SIGMA = 1.0

# [다수결 필터]
MAJORITY_FILTER_SIZE = 5

# [최종 목표 분포] (Zone 1~5 비율 강제 적용)
# Zone 1(8.5%), Zone 2(25%), Zone 3(33%), Zone 4(25%), Zone 5(8.5%)
# 누적 비율 계산:
# Z1 End: 0.085
# Z2 End: 0.085 + 0.25 = 0.335
# Z3 End: 0.335 + 0.33 = 0.665
# Z4 End: 0.665 + 0.25 = 0.915
# Z5 End: 1.0
TARGET_QUANTILES = [0.0, 0.085, 0.335, 0.665, 0.915, 1.0]


# ======================================================
# 1. 유틸리티 함수
# ======================================================
def load_boundary_from_zip(zip_path):
    safe_path = zip_path.replace("\\", "/")
    if not safe_path.startswith("zip://"): safe_path = f"zip://{safe_path}"
    try:
        gdf = gpd.read_file(safe_path)
        if gdf.crs is None: gdf.crs = "EPSG:4326"
        return gdf
    except Exception as e:
        print(f"Zip 로드 실패: {e}")
        return None


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
            poly = Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)])
            polygons.append(poly)
            indices.append((c_idx, r_idx))

    grid_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=boundary_gdf.crs)
    idx_df = pd.DataFrame(indices, columns=['mat_col', 'mat_row'])
    grid_gdf = pd.concat([grid_gdf, idx_df], axis=1)

    grid_gdf['geometry'] = grid_gdf['geometry'].apply(lambda g: affinity.rotate(g, rotation_angle, origin=centroid))

    intersects_mask = grid_gdf.intersects(boundary_geom)
    filtered_grid = grid_gdf[intersects_mask].copy().reset_index(drop=True)

    return filtered_grid


# ======================================================
# 2. 래스터 클리핑 (메모리 저장)
# ======================================================
def clip_raster_to_boundary(raster_path, boundary_gdf):
    print(">>> [Pre-Processing] Clipping Raster by Boundary...")
    with rasterio.open(raster_path) as src:
        if boundary_gdf.crs != src.crs:
            boundary_gdf = boundary_gdf.to_crs(src.crs)

        out_image, out_transform = mask(src, boundary_gdf.geometry, crop=True, nodata=np.nan)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": np.nan,
            "dtype": 'float32'
        })

        memfile = MemoryFile()
        with memfile.open(**out_meta) as dataset:
            dataset.write(out_image)

        return memfile


# ======================================================
# 3. [Step 1] 1m Grid Mean
# ======================================================
def calculate_grid_mean_stats(grid_gdf, mem_raster, col_name='Raw_Value'):
    stats = []
    with mem_raster.open() as src:
        for _, row in grid_gdf.iterrows():
            try:
                out_image, _ = mask(src, [row['geometry']], crop=True)
                data = out_image[0]

                # 0값 및 NaN 제외
                valid_data = data[(~np.isnan(data)) & (data > VALID_THRESHOLD) & (data != 0)]

                if valid_data.size > 0:
                    stats.append(np.mean(valid_data))
                else:
                    stats.append(np.nan)
            except:
                stats.append(np.nan)
    grid_gdf[col_name] = stats
    return grid_gdf


# ======================================================
# 4. [Step 2] Zone Detail (Smoothing)
# ======================================================
def apply_zone_detail_smoothing(grid_gdf, value_col, sigma=0.5):
    print(f"  [Algorithm] Applying Zone Detail (Sigma={sigma})...")

    max_col = grid_gdf['mat_col'].max()
    max_row = grid_gdf['mat_row'].max()

    matrix = np.full((max_row + 1, max_col + 1), np.nan)
    for _, row in grid_gdf.iterrows():
        r, c = int(row['mat_row']), int(row['mat_col'])
        matrix[r, c] = row[value_col]

    mask_valid = ~np.isnan(matrix)
    if not mask_valid.any():
        grid_gdf['Smooth_Value'] = grid_gdf[value_col]
        return grid_gdf

    coords = np.argwhere(mask_valid)
    values = matrix[mask_valid]

    interp = NearestNDInterpolator(coords, values)
    all_coords = np.indices(matrix.shape).transpose(1, 2, 0).reshape(-1, 2)
    filled_flat = interp(all_coords)
    filled_matrix = filled_flat.reshape(matrix.shape)

    smoothed_matrix = gaussian_filter(filled_matrix, sigma=sigma, mode='mirror')

    smoothed_values = []
    for _, row in grid_gdf.iterrows():
        r, c = int(row['mat_row']), int(row['mat_col'])
        val = smoothed_matrix[r, c]
        if np.isnan(row[value_col]):
            smoothed_values.append(np.nan)
        else:
            smoothed_values.append(val)

    grid_gdf['Smooth_Value'] = smoothed_values
    return grid_gdf


# ======================================================
# 5. 후처리 (다수결 필터)
# ======================================================
def apply_majority_filter(grid_gdf, zone_col='Zone', size=3):
    print(f"  [Post-Processing] Majority Filter (Size={size})...")
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
# 6. 시각화 및 통계
# ======================================================
def save_map_image(gdf, output_path, title_suffix="", zone_col='Zone', boundary_gdf=None):
    colors = ['#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#008000']
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    plot_data = gdf.copy()
    plot_data[zone_col] = pd.to_numeric(plot_data[zone_col], errors='coerce').fillna(0).astype(int)
    plot_data.plot(column=zone_col, cmap=cmap, linewidth=0, edgecolor='none', ax=ax, vmin=1, vmax=5)

    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color='cyan', linewidth=1, alpha=0.7)

    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, ["1(Low)", "2", "3", "4", "5(High)"])]
    ax.legend(handles=legend_patches, loc='lower right', title="Levels")
    ax.set_title(f"Zonation Map {title_suffix}", fontsize=15)
    ax.set_axis_off()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_stats(grid, zone_col, bins, title):
    colors = ["Red", "Orange", "Yellow", "Lt.Green", "Green"]
    valid_zones = grid[grid[zone_col] != 0]
    total_valid = len(valid_zones)

    print("\n" + "=" * 95)
    print(f" [{title}]")
    print(f" {'Zone':<12} | {'Range':<20} | {'Area(m2)':<10} | {'Area(%)':<8} | {'Mean':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 95)

    stats_df = valid_zones.groupby(zone_col)
    value_col = 'Raw_GNDVI' if 'Raw' in zone_col else 'Smooth_Value'

    for z in range(1, N_ZONES + 1):
        lower = bins[z - 1]
        upper = bins[z]

        if z in stats_df.groups:
            g = stats_df.get_group(z)
            count = len(g)
            area_m2 = g.geometry.area.sum()  # m2
            pct = (count / total_valid) * 100

            mean_val = g[value_col].mean()
            min_val = g[value_col].min()
            max_val = g[value_col].max()

            print(
                f" {z} ({colors[z - 1]}) | {lower:.3f} ~ {upper:.3f}    | {area_m2:>10.1f} | {pct:>6.2f}%  | {mean_val:.3f}    | {min_val:.3f}    | {max_val:.3f}")
        else:
            print(
                f" {z} ({colors[z - 1]}) | {lower:.3f} ~ {upper:.3f}    |       0.0  |   0.00%  |   -      |   -      |   -   ")
    print("=" * 95 + "\n")


def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    boundary = load_boundary_from_zip(BOUNDARY_ZIP_PATH)
    if boundary is None: return
    if boundary.crs.is_geographic: boundary = boundary.to_crs(epsg=5179)

    # 1. 래스터 클리핑 (메모리 로드)
    mem_raster = clip_raster_to_boundary(GNDVI_PATH, boundary)

    print(">>> 1. 그리드 생성")
    grid = create_rotated_grid_with_indices(boundary, grid_size=GRID_SIZE)

    print(">>> 2. 1m 그리드 값 추출 (Using Clipped Raster)")
    grid = calculate_grid_mean_stats(grid, mem_raster, col_name='Raw_GNDVI')

    # -----------------------------------------------------------
    # [Step 2] Raw Data Quantile (20% Equal) & Save
    # -----------------------------------------------------------
    raw_valid = grid.dropna(subset=['Raw_GNDVI'])

    # Raw는 항상 20% Equal Quantile
    _, raw_bins = pd.qcut(raw_valid['Raw_GNDVI'], q=N_ZONES, retbins=True, duplicates='drop')
    if len(raw_bins) < N_ZONES + 1:
        _, raw_bins = pd.qcut(raw_valid['Raw_GNDVI'].rank(method='first'), q=N_ZONES, retbins=True)

    grid['Zone_Raw'] = pd.cut(grid['Raw_GNDVI'], bins=raw_bins, labels=[1, 2, 3, 4, 5], include_lowest=True)
    grid['Zone_Raw'] = pd.to_numeric(grid['Zone_Raw'], errors='coerce').fillna(0).astype(int)

    print_stats(grid, 'Zone_Raw', raw_bins, "Raw Data Statistics (20% Equal)")
    save_map_image(grid, os.path.join(OUTPUT_FOLDER, "Result_1_Raw_Quantile.png"),
                   "(Raw Data - 20% Equal)", zone_col='Zone_Raw', boundary_gdf=boundary)

    # -----------------------------------------------------------
    # [Step 3] Zone Detail 적용 (Smoothing)
    # -----------------------------------------------------------
    print(f">>> 3. Zone Detail 적용 (Sigma={ZONE_DETAIL_SIGMA})")
    grid = apply_zone_detail_smoothing(grid, value_col='Raw_GNDVI', sigma=ZONE_DETAIL_SIGMA)

    # -----------------------------------------------------------
    # [Step 4] 최종 등급 부여 (Target Distribution Applied)
    # -----------------------------------------------------------
    smooth_valid = grid.dropna(subset=['Smooth_Value'])
    print(f">>> 4. 최종 등급 부여 (Target: {TARGET_QUANTILES})")

    # [핵심] 스무딩된 데이터에서 목표 비율(8.5, 25, 33...)에 해당하는 Cutoff 찾기
    final_bins = smooth_valid['Smooth_Value'].quantile(TARGET_QUANTILES).values

    # 중복 값 방지 (Rank 기반)
    if len(np.unique(final_bins)) < len(final_bins):
        final_bins = smooth_valid['Smooth_Value'].rank(method='first').quantile(TARGET_QUANTILES).values

    grid['Zone'] = pd.cut(grid['Smooth_Value'], bins=final_bins, labels=[1, 2, 3, 4, 5], include_lowest=True)
    grid['Zone'] = pd.to_numeric(grid['Zone'], errors='coerce').fillna(0).astype(int)

    # 범위 보정
    mask_valid = grid['Smooth_Value'].notna()
    grid.loc[mask_valid & (grid['Zone'] < 1), 'Zone'] = 1
    grid.loc[mask_valid & (grid['Zone'] > 5), 'Zone'] = 5

    # -----------------------------------------------------------
    # [Step 5] 다수결 필터 & 최종 통계
    # -----------------------------------------------------------
    grid = apply_majority_filter(grid, zone_col='Zone', size=MAJORITY_FILTER_SIZE)

    print_stats(grid, 'Zone', final_bins, "Final Map Statistics (Target Dist)")

    save_map_image(grid, os.path.join(OUTPUT_FOLDER, "Result_Pix4D_V30_TargetDist.png"),
                   f"(Pix4D Final - Custom Target)", zone_col='Zone', boundary_gdf=boundary)

    grid_save = grid.rename(columns={'Smooth_Value': 'GNDVI_Sm', 'Raw_GNDVI': 'GNDVI_Rw'})
    save_cols = [c for c in grid_save.columns if c not in ['mat_col', 'mat_row', 'Zone_Raw']]
    grid_save[save_cols].to_file(os.path.join(OUTPUT_FOLDER, "Result_Final.shp"), encoding='euc-kr')

    mem_raster.close()
    print("완료되었습니다.")


if __name__ == "__main__":
    main()