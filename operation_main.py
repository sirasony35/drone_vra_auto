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
OUTPUT_FOLDER = "result_final_dji"  # 폴더명 변경
VRA_CSV_PATH = "vra.csv"

GRID_SIZE = 1.0
N_ZONES = 5
VALID_THRESHOLD = -999.0
ZONE_DETAIL_SIGMA = 0.8
MAJORITY_FILTER_SIZE = 3


# ======================================================
# 1. 유틸리티 및 분석 함수 (기존 유지)
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
    max_col = grid_gdf['mat_col'].max()
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
    max_col = grid_gdf['mat_col'].max()
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
    colors = ['#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#008000']
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plot_data = gdf.copy()
    plot_data[zone_col] = pd.to_numeric(plot_data[zone_col], errors='coerce').fillna(0).astype(int)
    plot_data.plot(column=zone_col, cmap=cmap, linewidth=0, edgecolor='none', ax=ax, vmin=1, vmax=5)
    if boundary_gdf is not None: boundary_gdf.boundary.plot(ax=ax, color='cyan', linewidth=1, alpha=0.7)
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
    print(f" {'Zone':<12} | {'Range':<20} | {'Area(m2)':<10} | {'Area(%)':<8} | {'Mean':<8}")
    print("-" * 95)
    stats_df = valid_zones.groupby(zone_col)
    for z in range(1, N_ZONES + 1):
        if len(bins) >= N_ZONES + 1:
            lower = bins[z - 1]; upper = bins[z]
        else:
            lower = np.nan; upper = np.nan
        if z in stats_df.groups:
            g = stats_df.get_group(z)
            count = len(g)
            area_m2 = g.geometry.area.sum()
            pct = (count / total_valid) * 100
            mean_val = g['Smooth_Value'].mean()
            print(
                f" {z} ({colors[z - 1]}) | {lower:.3f} ~ {upper:.3f}    | {area_m2:>10.1f} | {pct:>6.2f}%  | {mean_val:.3f}")
        else:
            print(f" {z} ({colors[z - 1]}) | {lower:.3f} ~ {upper:.3f}    |       0.0  |   0.00%  |   -")
    print("=" * 95 + "\n")


# ======================================================
# [NEW] DJI 포맷 저장 함수 (Rx TIF/TFW & Boundary SHP)
# ======================================================
def save_dji_files(grid_gdf, vra_df, boundary_gdf, field_code, src_meta):
    """
    DJI 기체 호환용 파일 생성:
    1. DJI/Rx/{field_code}.tif (값=Rate)
    2. DJI/Rx/{field_code}.tfw (좌표 정보)
    3. DJI/ShapeFile/{field_code}.shp (바운더리)
    """
    print(f"  [Output] Generating DJI Compatible Files for {field_code}...")

    # 1. 폴더 생성
    rx_folder = os.path.join(OUTPUT_FOLDER, "DJI", "Rx")
    shp_folder = os.path.join(OUTPUT_FOLDER, "DJI", "ShapeFile")
    os.makedirs(rx_folder, exist_ok=True)
    os.makedirs(shp_folder, exist_ok=True)

    # 2. 바운더리 SHP 저장
    boundary_out = os.path.join(shp_folder, f"{field_code}.shp")
    boundary_gdf.to_file(boundary_out, encoding='euc-kr')
    print(f"    - Boundary saved: {boundary_out}")

    # 3. Rx TIF 생성 (Rasterize Rates)
    # Zone별 Rate 매핑 사전 생성
    rate_map = {}  # {Zone_Index: Rate_Value}
    for _, row in vra_df.iterrows():
        # Zone 문자열 "1(빨강)"에서 숫자 "1" 추출
        try:
            zone_idx = int(str(row['Zone']).split('(')[0])
            rate_val = float(row['Rate(kg/ha)'])
            rate_map[zone_idx] = rate_val
        except:
            continue

    # 그리드에 Rate 값 매핑
    # 매핑되지 않은(0등급 등) 곳은 0으로 처리
    grid_gdf['Rx_Rate'] = grid_gdf['Zone'].map(rate_map).fillna(0)

    # DataFrame -> Numpy Array 변환 (Rasterize보다 빠름)
    max_col = grid_gdf['mat_col'].max()
    max_row = grid_gdf['mat_row'].max()

    # 빈 래스터 생성 (0으로 초기화)
    rx_raster = np.zeros((max_row + 1, max_col + 1), dtype='float32')

    # 값 채우기
    # grid_gdf에는 유효한 그리드만 있으므로, 여기 없는 곳은 0(배경)이 됨
    for _, row in grid_gdf.iterrows():
        r, c = int(row['mat_row']), int(row['mat_col'])
        rx_raster[r, c] = row['Rx_Rate']

    # 원본 래스터 정보 복사 및 수정
    out_meta = src_meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": rx_raster.shape[0],
        "width": rx_raster.shape[1],
        "count": 1,
        "dtype": 'float32',
        "nodata": 0
    })

    # 4. TIF 저장
    tif_out = os.path.join(rx_folder, f"{field_code}.tif")
    tfw_out = os.path.join(rx_folder, f"{field_code}.tfw")

    with rasterio.open(tif_out, "w", **out_meta) as dest:
        dest.write(rx_raster, 1)
        # Transform 정보 가져오기
        transform = dest.transform

    # 5. TFW 파일 생성 (World File)
    # Line 1: Pixel X size
    # Line 2: Rotation (0)
    # Line 3: Rotation (0)
    # Line 4: Pixel Y size (Negative)
    # Line 5: Top Left X
    # Line 6: Top Left Y
    with open(tfw_out, "w") as f:
        f.write(f"{transform.a}\n")  # Pixel width
        f.write(f"{transform.b}\n")  # Rotation
        f.write(f"{transform.d}\n")  # Rotation
        f.write(f"{transform.e}\n")  # Pixel height
        f.write(f"{transform.c}\n")  # X origin
        f.write(f"{transform.f}\n")  # Y origin

    print(f"    - Rx Map saved: {tif_out}")
    print(f"    - World File saved: {tfw_out}")


# ======================================================
# 3. 메인 프로세스
# ======================================================
def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    detector = BoundaryDetector()
    vra_calc = VRACalculator(VRA_CSV_PATH)

    tif_files = glob.glob(os.path.join(DATA_FOLDER, "*_GNDVI.tif"))

    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        try:
            field_code = filename.split("_")[0]
        except:
            field_code = "Unknown"

        print(f"\n>>> Processing: {filename} (Field Code: {field_code})")

        # 1. Boundary Load/Detect
        boundary_path = os.path.join(DATA_FOLDER, f"{field_code}_Boundary.zip")
        if os.path.exists(boundary_path):
            boundary = detector.load_boundary_from_zip(boundary_path)
        else:
            boundary = detector.detect_boundary_otsu(tif_path)

        if boundary is None: continue
        if boundary.crs.is_geographic: boundary = boundary.to_crs(epsg=5179)

        try:
            # 원본 메타데이터 확보 (나중에 Rx 저장 시 사용)
            # Clip된 메모리 파일 대신 원본 파일에서 transform 정보 등을 참조해야 정확함
            # 단, Grid 생성 시 boundary 기준으로 잘랐으므로, Grid 생성 시 사용된 transform을 추적해야 함.
            # 여기서는 clip_raster_to_boundary에서 생성된 mem_raster의 메타데이터를 사용하겠습니다.

            # 2. Zonation Process
            mem_raster = clip_raster_to_boundary(tif_path, boundary)
            # 메타데이터 복사 (나중에 DJI 저장용)
            with mem_raster.open() as src:
                src_meta = src.meta.copy()

            grid = create_rotated_grid_with_indices(boundary, grid_size=GRID_SIZE)
            grid = calculate_grid_mean_stats(grid, mem_raster, col_name='Raw_GNDVI')

            raw_valid = grid.dropna(subset=['Raw_GNDVI'])
            if len(raw_valid) == 0: continue

            _, raw_bins = pd.qcut(raw_valid['Raw_GNDVI'], q=N_ZONES, retbins=True, duplicates='drop')
            if len(raw_bins) < N_ZONES + 1:
                _, raw_bins = pd.qcut(raw_valid['Raw_GNDVI'].rank(method='first'), q=N_ZONES, retbins=True)

            grid = apply_zone_detail_smoothing(grid, value_col='Raw_GNDVI', sigma=ZONE_DETAIL_SIGMA)
            grid['Zone'] = pd.cut(grid['Smooth_Value'], bins=raw_bins, labels=[1, 2, 3, 4, 5], include_lowest=True)
            grid['Zone'] = pd.to_numeric(grid['Zone'], errors='coerce').fillna(0).astype(int)

            mask_valid = grid['Smooth_Value'].notna()
            grid.loc[mask_valid & (grid['Zone'] < 1), 'Zone'] = 1
            grid.loc[mask_valid & (grid['Zone'] > 5), 'Zone'] = 5
            grid = apply_majority_filter(grid, zone_col='Zone', size=MAJORITY_FILTER_SIZE)

            # 3. Zone Stats for VRA
            valid_zones = grid[grid['Zone'] != 0]
            stats_df = valid_zones.groupby('Zone')

            zone_stats = []
            for z in range(1, 6):
                if z in stats_df.groups:
                    g = stats_df.get_group(z)
                    area_m2 = g.geometry.area.sum()
                    mean_gndvi = g['Smooth_Value'].mean()
                    zone_stats.append({'Zone': z, 'Area_m2': area_m2, 'Mean_GNDVI': mean_gndvi})
                else:
                    zone_stats.append({'Zone': z, 'Area_m2': 0, 'Mean_GNDVI': 0})

            # 4. VRA Calculation
            print("  - Calculating VRA Prescription...")
            vra_df = vra_calc.calculate_prescription(field_code, zone_stats)

            if vra_df is not None:
                # 5. [NEW] Save DJI Files (Rx TIF/TFW & Boundary SHP)
                save_dji_files(grid, vra_df, boundary, field_code, src_meta)

                # 통계 CSV 저장 (참고용)
                vra_out_name = filename.replace(".tif", "_VRA.csv")
                vra_df.to_csv(os.path.join(OUTPUT_FOLDER, vra_out_name), index=False, encoding='euc-kr')
            else:
                print("  - [SKIP] VRA Calculation failed (Missing config or data).")

            # 6. Visualization Maps (Optional)
            out_img_name = filename.replace(".tif", "_Result.png")
            save_map_image(grid, os.path.join(OUTPUT_FOLDER, out_img_name),
                           f"Result: {filename}", zone_col='Zone', boundary_gdf=boundary)

            mem_raster.close()
            print("  - Processing Complete.")

        except Exception as e:
            print(f"  - Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()