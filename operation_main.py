import os
import glob
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Polygon
from shapely import affinity
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter, generic_filter, binary_erosion
from skimage.filters import threshold_otsu  # [중요] Otsu 알고리즘 사용
import warnings

# [모듈 Import]
from boundary_detector import BoundaryDetector
from vra_calculator import VRACalculator

# 설정 및 경고 무시
warnings.filterwarnings("ignore")
pd.set_option('future.no_silent_downcasting', True)

# ======================================================
# 0. 설정
# ======================================================
DATA_FOLDER = "test_data"
OUTPUT_FOLDER = "result_final_dji_wgs84"
VRA_CSV_PATH = "vra.csv"

GRID_SIZE = 1.0
N_ZONES = 5
VALID_THRESHOLD = -999.0
ZONE_DETAIL_SIGMA = 1.35
MAJORITY_FILTER_SIZE = 3

# [NEW] 마스킹 최적화 파라미터
# 1. 완화 계수 (0.85 의미: Otsu가 찾은 값의 85%만 적용)
#    - 값을 낮출수록(0.7 ~ 0.8) 마스킹이 줄어들고 비료를 더 뿌립니다.
#    - 값을 높일수록(0.9 ~ 1.0) 마스킹이 엄격해집니다.
OTSU_RELAX_FACTOR = 0.3

# 2. 마스킹 절대 상한선 (Hard Limit)
#    - Otsu 결과가 아무리 높아도 이 값(GNDVI) 이상은 절대 마스킹하지 않음
#    - 작물이 심겨진 논이라면 0.40 이상은 무조건 작물로 보는 안전장치
MAX_MASK_THRESHOLD = 0.40


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
    return grid_gdf[intersects_mask].copy().reset_index(drop=True)


def clip_raster_to_boundary(raster_path, boundary_gdf):
    with rasterio.open(raster_path) as src:
        if boundary_gdf.crs != src.crs: boundary_gdf = boundary_gdf.to_crs(src.crs)
        out_image, out_transform = mask(src, boundary_gdf.geometry, crop=True, nodata=np.nan)
        out_meta = src.meta.copy()
        out_meta.update(
            {"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform,
             "nodata": np.nan, "dtype": 'float32'})
        memfile = MemoryFile()
        with memfile.open(**out_meta) as dataset: dataset.write(out_image)
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
    max_col = grid_gdf['mat_col'].max();
    max_row = grid_gdf['mat_row'].max()
    matrix = np.full((max_row + 1, max_col + 1), np.nan)
    for _, row in grid_gdf.iterrows():
        r, c = int(row['mat_row']), int(row['mat_col'])
        matrix[r, c] = row[value_col]
    mask_valid = ~np.isnan(matrix)
    eroded_mask = binary_erosion(mask_valid, iterations=1)
    edge_mask = mask_valid & (~eroded_mask)
    valid_vals = matrix[mask_valid]
    robust_vals = valid_vals[valid_vals > 0.1]
    field_mean = np.mean(robust_vals) if len(robust_vals) > 0 else np.nanmean(matrix)
    boosted_matrix = matrix.copy()
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
    max_col = grid_gdf['mat_col'].max();
    max_row = grid_gdf['mat_row'].max()
    matrix = np.zeros((max_row + 1, max_col + 1), dtype=int)
    for _, row in grid_gdf.iterrows(): matrix[int(row['mat_row']), int(row['mat_col'])] = row[zone_col]

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


def save_map_image(gdf, output_path, title_suffix="", zone_col='Zone', boundary_gdf=None):
    # [수정] Zone 6 (회색) 추가
    colors = ['#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#008000', '#808080']
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plot_data = gdf.copy()
    plot_data[zone_col] = pd.to_numeric(plot_data[zone_col], errors='coerce').fillna(0).astype(int)

    # vmin=1, vmax=6
    plot_data.plot(column=zone_col, cmap=cmap, linewidth=0, edgecolor='none', ax=ax, vmin=1, vmax=6)

    if boundary_gdf is not None: boundary_gdf.boundary.plot(ax=ax, color='cyan', linewidth=1, alpha=0.7)

    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in
                      zip(colors, ["1(High)", "2", "3", "4", "5(Low)", "6(Skip)"])]
    ax.legend(handles=legend_patches, loc='lower right', title="Levels")
    ax.set_title(f"Zonation Map {title_suffix}", fontsize=15)
    ax.set_axis_off()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_dji_files_wgs84(grid_gdf, vra_df, boundary_gdf, field_code):
    print(f"  [Output] Generating DJI Compatible Files (WGS84)...")
    rx_folder = os.path.join(OUTPUT_FOLDER, "DJI", "Rx")
    shp_folder = os.path.join(OUTPUT_FOLDER, "DJI", "ShapeFile")
    os.makedirs(rx_folder, exist_ok=True)
    os.makedirs(shp_folder, exist_ok=True)

    # 1. Boundary SHP
    boundary_4326 = boundary_gdf.to_crs(epsg=4326)
    boundary_out = os.path.join(shp_folder, f"{field_code}.shp")
    boundary_4326.to_file(boundary_out, encoding='euc-kr')

    # 2. Rx TIF
    rate_map = {}
    for _, row in vra_df.iterrows():
        try:
            zone_idx = int(str(row['Zone']).split('(')[0])
            rate_val = float(row['Rate(kg/ha)'])
            rate_map[zone_idx] = rate_val
        except:
            continue

    grid_gdf['Rx_Rate'] = grid_gdf['Zone'].map(rate_map).fillna(0)
    grid_4326 = grid_gdf.to_crs(epsg=4326)

    minx, miny, maxx, maxy = grid_4326.total_bounds
    pixel_size = 0.000005  # Approx 0.5m
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    transform = from_origin(minx, maxy, pixel_size, pixel_size)

    shapes = ((geom, value) for geom, value in zip(grid_4326.geometry, grid_4326['Rx_Rate']))
    out_image = rasterize(shapes=shapes, out_shape=(height, width), transform=transform, fill=0, dtype='float32')

    tif_out = os.path.join(rx_folder, f"{field_code}.tif")
    out_meta = {"driver": "GTiff", "height": height, "width": width, "count": 1, "dtype": 'float32', "crs": "EPSG:4326",
                "transform": transform, "nodata": 0}
    with rasterio.open(tif_out, "w", **out_meta) as dest:
        dest.write(out_image, 1)

    tfw_out = os.path.join(rx_folder, f"{field_code}.tfw")
    with open(tfw_out, "w") as f:
        f.write(f"{transform.a}\n");
        f.write(f"{transform.b}\n");
        f.write(f"{transform.d}\n");
        f.write(f"{transform.e}\n");
        f.write(f"{transform.c}\n");
        f.write(f"{transform.f}\n")
    print(f"    - Rx Map saved: {tif_out}")


# ======================================================
# 함수 수정: calculate_dynamic_threshold
# ======================================================
def calculate_dynamic_threshold(grid_gdf):
    """
    Otsu 알고리즘 값에 완화 계수(Relaxation)를 적용하여
    '약한 작물'은 살리고 '진짜 맨땅'만 타겟팅합니다.
    """
    valid_values = grid_gdf['Smooth_Value'].dropna()
    valid_values = valid_values[valid_values > 0]  # 0은 제외

    if len(valid_values) < 10:
        return -999

    try:
        # 1. Otsu로 수학적 경계값 찾기 (예: 0.45가 나왔다고 가정)
        raw_otsu = threshold_otsu(valid_values.values)

        # 2. 완화 계수 적용 (0.45 * 0.85 = 0.382) -> 기준을 낮춤
        relaxed_thresh = raw_otsu * OTSU_RELAX_FACTOR

        print(f"    [Threshold Info] Raw Otsu: {raw_otsu:.4f} -> Relaxed(x{OTSU_RELAX_FACTOR}): {relaxed_thresh:.4f}")

        # 3. 절대 상한선(Safety Clamp) 적용
        # 계산된 값이 0.40을 넘어가면 강제로 0.40으로 낮춤 (과도한 마스킹 방지)
        if relaxed_thresh > MAX_MASK_THRESHOLD:
            print(f"    [Adjustment] Threshold {relaxed_thresh:.4f} exceeds limit. Clamping to {MAX_MASK_THRESHOLD}.")
            final_thresh = MAX_MASK_THRESHOLD
        else:
            final_thresh = relaxed_thresh

        # 4. 너무 낮은 값(0.1 미만)은 사실상 마스킹 의미가 없으므로 무시할 수도 있음 (선택사항)
        if final_thresh < 0.1:
            return -999

        return final_thresh

    except Exception as e:
        print(f"    [Warning] Otsu calculation failed: {e}")
        return -999


# ======================================================
# 2. 메인 프로세스
# ======================================================
def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    detector = BoundaryDetector()
    vra_calc = VRACalculator(VRA_CSV_PATH)
    tif_files = glob.glob(os.path.join(DATA_FOLDER, "*_GNDVI.tif"))

    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        field_code = filename.split("_")[0] if "_" in filename else "Unknown"
        print(f"\n>>> Processing: {filename} (Field Code: {field_code})")

        boundary_path = os.path.join(DATA_FOLDER, f"{field_code}_Boundary.zip")
        if os.path.exists(boundary_path):
            boundary = detector.load_boundary_from_zip(boundary_path)
        else:
            boundary = detector.detect_boundary_otsu(tif_path)
        if boundary is None: continue
        if boundary.crs.is_geographic: boundary = boundary.to_crs(epsg=5179)

        try:
            mem_raster = clip_raster_to_boundary(tif_path, boundary)
            grid = create_rotated_grid_with_indices(boundary, grid_size=GRID_SIZE)
            grid = calculate_grid_mean_stats(grid, mem_raster, col_name='Raw_GNDVI')

            raw_valid = grid.dropna(subset=['Raw_GNDVI'])
            if len(raw_valid) == 0: continue

            _, raw_bins = pd.qcut(raw_valid['Raw_GNDVI'], q=N_ZONES, retbins=True, duplicates='drop')
            if len(raw_bins) < N_ZONES + 1:
                _, raw_bins = pd.qcut(raw_valid['Raw_GNDVI'].rank(method='first'), q=N_ZONES, retbins=True)

            grid = apply_zone_detail_smoothing(grid, value_col='Raw_GNDVI', sigma=ZONE_DETAIL_SIGMA)

            # 1. 기본 Zoning (1~5)
            grid['Zone'] = pd.cut(grid['Smooth_Value'], bins=raw_bins, labels=[1, 2, 3, 4, 5], include_lowest=True)
            grid['Zone'] = pd.to_numeric(grid['Zone'], errors='coerce').fillna(0).astype(int)

            # -----------------------------------------------------------
            # [NEW] 동적 마스킹 (Dynamic Masking)
            # -----------------------------------------------------------
            print("  - Calculating dynamic soil threshold...")
            soil_threshold = calculate_dynamic_threshold(grid)

            if soil_threshold > -900:
                print(f"    -> Detected Soil Threshold: {soil_threshold:.4f}")
                mask_bare = grid['Smooth_Value'] < soil_threshold
                count_bare = mask_bare.sum()
                if count_bare > 0:
                    print(f"    -> Masking {count_bare} grids as Zone 6 (Skip).")
                    grid.loc[mask_bare, 'Zone'] = 6
            else:
                print("    -> No masking applied (Threshold too high or invalid).")
            # -----------------------------------------------------------

            mask_valid = grid['Smooth_Value'].notna()
            # Zone 6가 아닌 것들만 범위 보정
            grid.loc[mask_valid & (grid['Zone'] < 1), 'Zone'] = 1
            grid.loc[mask_valid & (grid['Zone'] > 5) & (grid['Zone'] != 6), 'Zone'] = 5

            grid = apply_majority_filter(grid, zone_col='Zone', size=MAJORITY_FILTER_SIZE)

            # Zone Stats
            valid_zones = grid[grid['Zone'] != 0]
            stats_df = valid_zones.groupby('Zone')
            zone_stats = []

            # 1~6까지 루프
            for z in range(1, 7):
                if z in stats_df.groups:
                    g = stats_df.get_group(z)
                    zone_stats.append(
                        {'Zone': z, 'Area_m2': g.geometry.area.sum(), 'Mean_GNDVI': g['Smooth_Value'].mean()})
                else:
                    pass  # Empty zone

            print("  - Calculating VRA Prescription...")
            vra_df = vra_calc.calculate_prescription(field_code, zone_stats)

            if vra_df is not None:
                save_dji_files_wgs84(grid, vra_df, boundary, field_code)
                vra_out_name = filename.replace(".tif", "_VRA.csv")
                vra_df.to_csv(os.path.join(OUTPUT_FOLDER, vra_out_name), index=False, encoding='euc-kr')

            out_img_name = filename.replace(".tif", "_Result.png")
            save_map_image(grid, os.path.join(OUTPUT_FOLDER, out_img_name), f"Result: {filename}", zone_col='Zone',
                           boundary_gdf=boundary)
            mem_raster.close()
            print("  - Processing Complete.")

        except Exception as e:
            print(f"  - Error processing {filename}: {e}")
            import traceback;
            traceback.print_exc()


if __name__ == "__main__":
    main()