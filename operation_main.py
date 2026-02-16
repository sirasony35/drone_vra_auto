import os
import glob
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from shapely.geometry import Polygon
from shapely import affinity
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter, generic_filter, binary_erosion
from scipy.interpolate import NearestNDInterpolator
import warnings

# [모듈 Import] 바운더리 감지기 클래스 가져오기
from boundary_detector import BoundaryDetector

# 경고 및 옵션 설정
warnings.filterwarnings("ignore")
pd.set_option('future.no_silent_downcasting', True)

# ======================================================
# 0. 설정 (Pix4D Logic V38 - Edge Boosting + Auto Boundary)
# ======================================================
DATA_FOLDER = "test_data"
OUTPUT_FOLDER = "result_final_auto"

GRID_SIZE = 1.0
N_ZONES = 5
VALID_THRESHOLD = -999.0
ZONE_DETAIL_SIGMA = 1.4  # V38 기준
MAJORITY_FILTER_SIZE = 3


# ======================================================
# 1. 유틸리티 및 분석 함수
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


def clip_raster_to_boundary(raster_path, boundary_gdf):
    # print(">>> [Pre-Processing] Clipping Raster by Boundary...")
    with rasterio.open(raster_path) as src:
        if boundary_gdf.crs != src.crs:
            boundary_gdf = boundary_gdf.to_crs(src.crs)
        out_image, out_transform = mask(src, boundary_gdf.geometry, crop=True, nodata=np.nan)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
            "transform": out_transform, "nodata": np.nan, "dtype": 'float32'
        })
        memfile = MemoryFile()
        with memfile.open(**out_meta) as dataset:
            dataset.write(out_image)
        return memfile


def calculate_grid_mean_stats(grid_gdf, mem_raster, col_name='Raw_Value'):
    stats = []
    with mem_raster.open() as src:
        for _, row in grid_gdf.iterrows():
            try:
                out_image, _ = mask(src, [row['geometry']], crop=True)
                data = out_image[0]
                valid_data = data[(~np.isnan(data)) & (data > VALID_THRESHOLD) & (data != 0)]
                if valid_data.size > 0:
                    stats.append(np.mean(valid_data))
                else:
                    stats.append(np.nan)
            except:
                stats.append(np.nan)
    grid_gdf[col_name] = stats
    return grid_gdf


def apply_zone_detail_smoothing(grid_gdf, value_col, sigma=0.5):
    # print(f"  [Algorithm] Applying Zone Detail (Sigma={sigma}, Strategy=EdgeBoosting)...")
    max_col = grid_gdf['mat_col'].max()
    max_row = grid_gdf['mat_row'].max()

    matrix = np.full((max_row + 1, max_col + 1), np.nan)
    for _, row in grid_gdf.iterrows():
        r, c = int(row['mat_row']), int(row['mat_col'])
        matrix[r, c] = row[value_col]

    mask_valid = ~np.isnan(matrix)
    # [V38] Edge Boosting Logic
    eroded_mask = binary_erosion(mask_valid, iterations=1)
    edge_mask = mask_valid & (~eroded_mask)

    valid_vals = matrix[mask_valid]
    robust_vals = valid_vals[valid_vals > 0.1]
    field_mean = np.mean(robust_vals) if len(robust_vals) > 0 else np.nanmean(matrix)

    boosted_matrix = matrix.copy()
    # 가장자리를 평균값과 섞어 중화시킴 (Red Edge 방지)
    boosted_matrix[edge_mask] = (boosted_matrix[edge_mask] * 0.4) + (field_mean * 0.6)

    filled_matrix = boosted_matrix.copy()
    filled_matrix[np.isnan(filled_matrix)] = field_mean

    smoothed_matrix = gaussian_filter(filled_matrix, sigma=sigma, mode='constant', cval=field_mean)

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


def apply_majority_filter(grid_gdf, zone_col='Zone', size=3):
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
        if len(bins) >= N_ZONES + 1:
            lower = bins[z - 1];
            upper = bins[z]
        else:
            lower = np.nan;
            upper = np.nan

        if z in stats_df.groups:
            g = stats_df.get_group(z)
            count = len(g)
            area_m2 = g.geometry.area.sum()
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


# ======================================================
# 3. 메인 프로세스
# ======================================================
def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    # 1. 파일 검색 및 바운더리 감지기 초기화
    tif_files = glob.glob(os.path.join(DATA_FOLDER, "*_GNDVI.tif"))
    detector = BoundaryDetector()  # 모듈 클래스 사용

    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        # 파일명 파싱 예시: "GJR1_02_250724_GNDVI.tif" -> "GJR1"
        try:
            field_code = filename.split("_")[0]
        except:
            field_code = "Unknown"

        print(f"\n>>> Processing: {filename} (Field Code: {field_code})")

        # 2. 바운더리 로드 또는 자동 생성
        boundary_path = os.path.join(DATA_FOLDER, f"{field_code}_Boundary.zip")
        boundary = None

        if os.path.exists(boundary_path):
            print(f"  - Found boundary file: {boundary_path}")
            boundary = detector.load_boundary_from_zip(boundary_path)
        else:
            print(f"  - Boundary file NOT found. Attempting Auto-Detection...")
            boundary = detector.detect_boundary_otsu(tif_path)

        if boundary is None:
            print("  - [FAIL] Boundary could not be established. Skipping file.")
            continue

        # 좌표계 통일 (권장: EPSG 5179, 필요시 소스 TIF 좌표계로 변환)
        # 여기서는 TIF가 대부분 5179/32652 등 미터 좌표계라고 가정
        if boundary.crs and boundary.crs.is_geographic:
            boundary = boundary.to_crs(epsg=5179)

        # -----------------------------------------------------------
        # [Zonation Logic Start] - V38 Edge Boosting
        # -----------------------------------------------------------
        try:
            # 1. 래스터 클리핑
            mem_raster = clip_raster_to_boundary(tif_path, boundary)

            # 2. 그리드 생성
            grid = create_rotated_grid_with_indices(boundary, grid_size=GRID_SIZE)

            # 3. Raw Data 추출
            grid = calculate_grid_mean_stats(grid, mem_raster, col_name='Raw_GNDVI')

            # 4. 기준점 계산 (Raw 20% Equal)
            raw_valid = grid.dropna(subset=['Raw_GNDVI'])
            if len(raw_valid) == 0:
                print("  - No valid data in grid. Skipping.")
                continue

            _, raw_bins = pd.qcut(raw_valid['Raw_GNDVI'], q=N_ZONES, retbins=True, duplicates='drop')
            if len(raw_bins) < N_ZONES + 1:
                _, raw_bins = pd.qcut(raw_valid['Raw_GNDVI'].rank(method='first'), q=N_ZONES, retbins=True)

            # Raw 통계 출력 (확인용)
            # grid['Zone_Raw'] = pd.cut(grid['Raw_GNDVI'], bins=raw_bins, labels=[1, 2, 3, 4, 5], include_lowest=True)
            # print_stats(grid, 'Zone_Raw', raw_bins, f"Raw Data Stats - {filename}")

            # 5. 스무딩 적용 (V38: Edge Boosting)
            grid = apply_zone_detail_smoothing(grid, value_col='Raw_GNDVI', sigma=ZONE_DETAIL_SIGMA)

            # 6. 최종 등급 부여 (Apply Raw Bins to Smooth Data)
            grid['Zone'] = pd.cut(grid['Smooth_Value'], bins=raw_bins, labels=[1, 2, 3, 4, 5], include_lowest=True)
            grid['Zone'] = pd.to_numeric(grid['Zone'], errors='coerce').fillna(0).astype(int)

            # 범위 보정
            mask_valid = grid['Smooth_Value'].notna()
            grid.loc[mask_valid & (grid['Zone'] < 1), 'Zone'] = 1
            grid.loc[mask_valid & (grid['Zone'] > 5), 'Zone'] = 5

            # 7. 필터링
            grid = apply_majority_filter(grid, zone_col='Zone', size=MAJORITY_FILTER_SIZE)

            # 8. 최종 결과 출력 및 저장
            print_stats(grid, 'Zone', raw_bins, f"Final Stats - {filename}")

            # 이미지 저장
            out_img_name = filename.replace(".tif", "_Result.png")
            save_map_image(grid, os.path.join(OUTPUT_FOLDER, out_img_name),
                           f"Result: {filename}", zone_col='Zone', boundary_gdf=boundary)

            # SHP 저장
            out_shp_name = filename.replace(".tif", "_Result.shp")
            grid_save = grid.rename(columns={'Smooth_Value': 'GNDVI_Sm', 'Raw_GNDVI': 'GNDVI_Rw'})
            save_cols = [c for c in grid_save.columns if c not in ['mat_col', 'mat_row', 'Zone_Raw']]
            grid_save[save_cols].to_file(os.path.join(OUTPUT_FOLDER, out_shp_name), encoding='euc-kr')

            mem_raster.close()
            print("  - Processing Complete.")

        except Exception as e:
            print(f"  - Error processing {filename}: {e}")
            # 디버깅을 위해 상세 에러 출력
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()