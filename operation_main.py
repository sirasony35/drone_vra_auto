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
from scipy.ndimage import gaussian_filter, generic_filter
from skimage.filters import threshold_otsu
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
DATA_FOLDER = "data/vra_data"
BOUNDARY_FOLDER = "data/ShapeFile"
OUTPUT_FOLDER = "result/result_final_dji_sm_rx_5m"
VRA_CSV_PATH = "vra_setting/sm_vra.csv"

DEFAULT_GRID_SIZE = 1.0
DEFAULT_CROP = 'rice'

N_ZONES = 5
VALID_THRESHOLD = -999.0
MAJORITY_FILTER_SIZE = 5  # Pix4D처럼 구역을 크게 모으기 위해 커널 크기 확장

MAX_MASK_THRESHOLD = 0.40


# ======================================================
# 1. 유틸리티 및 분석 함수
# ======================================================
def get_main_angle(geometry):
    rect = geometry.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    max_len = 0
    main_angle = 0
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        length = math.sqrt(dx ** 2 + dy ** 2)
        if length > max_len:
            max_len = length
            main_angle = math.degrees(math.atan2(dy, dx))
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


def calculate_optimal_sigma(grid_gdf, grid_size, value_col='Raw_GNDVI'):
    base_sigma = 1.35

    # 1. 그리드 크기 보정
    adjusted_sigma = base_sigma / (grid_size if grid_size > 0 else 1.0)

    # 2. 생육 편차 보정
    valid_vals = grid_gdf[value_col].dropna()
    if len(valid_vals) > 10:
        std_val = np.std(valid_vals)
        if std_val < 0.02:
            adjusted_sigma *= 0.5
        elif std_val > 0.08:
            adjusted_sigma *= 1.2

    return round(max(0.1, min(adjusted_sigma, 2.0)), 2)


def calculate_dynamic_threshold(grid_gdf, relax_factor=0.3):
    valid_values = grid_gdf['Raw_GNDVI'].dropna()
    valid_values = valid_values[valid_values > 0]
    if len(valid_values) < 10:
        return -999
    try:
        raw_otsu = threshold_otsu(valid_values.values)
        relaxed_thresh = raw_otsu * relax_factor
        print(f"    [Threshold Info] Raw Otsu: {raw_otsu:.4f} * Factor {relax_factor} = {relaxed_thresh:.4f}")

        if relaxed_thresh > MAX_MASK_THRESHOLD:
            final_thresh = MAX_MASK_THRESHOLD
        else:
            final_thresh = relaxed_thresh

        if final_thresh < 0.1:
            return -999
        return final_thresh
    except Exception as e:
        print(f"    [Warning] Otsu calculation failed: {e}")
        return -999


def apply_categorical_zone_smoothing(grid_gdf, zone_col='Raw_Zone', sigma=1.0, filter_size=5):
    max_col = grid_gdf['mat_col'].max()
    max_row = grid_gdf['mat_row'].max()
    matrix = np.full((max_row + 1, max_col + 1), np.nan)

    for _, row in grid_gdf.iterrows():
        r, c = int(row['mat_row']), int(row['mat_col'])
        val = row[zone_col]
        if pd.isna(val) or val == 6:
            matrix[r, c] = np.nan
        else:
            matrix[r, c] = val

    mask_valid = ~np.isnan(matrix)

    filled_matrix = matrix.copy()
    filled_matrix[np.isnan(filled_matrix)] = 3.0

    if sigma >= 0.2:
        smoothed_matrix = gaussian_filter(filled_matrix, sigma=sigma, mode='nearest')
    else:
        smoothed_matrix = filled_matrix

    rounded_matrix = np.round(smoothed_matrix).astype(int)
    rounded_matrix = np.clip(rounded_matrix, 1, 5)
    rounded_matrix[~mask_valid] = 0

    def mode_func(values):
        valid_vals = values[values > 0]
        if len(valid_vals) == 0:
            return 0
        vals, counts = np.unique(valid_vals, return_counts=True)
        return vals[np.argmax(counts)]

    cleaned_matrix = generic_filter(rounded_matrix, mode_func, size=filter_size, mode='constant', cval=0)

    final_zones = []
    for _, row in grid_gdf.iterrows():
        r, c = int(row['mat_row']), int(row['mat_col'])
        orig_val = row[zone_col]

        if orig_val == 6:
            final_zones.append(6)
        elif pd.isna(orig_val):
            final_zones.append(0)
        else:
            val = cleaned_matrix[r, c]
            final_zones.append(val if val > 0 else int(orig_val))

    grid_gdf['Zone'] = final_zones
    return grid_gdf


def save_map_image(gdf, output_path, title_suffix="", zone_col='Zone', boundary_gdf=None):
    colors = ['#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#008000', '#808080']
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plot_data = gdf.copy()
    plot_data[zone_col] = pd.to_numeric(plot_data[zone_col], errors='coerce').fillna(0).astype(int)
    plot_data.plot(column=zone_col, cmap=cmap, linewidth=0, edgecolor='none', ax=ax, vmin=1, vmax=6)
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color='cyan', linewidth=1, alpha=0.7)

    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in
                      zip(colors, ["1(High)", "2", "3", "4", "5(Low)", "6(Skip)"])]
    ax.legend(handles=legend_patches, loc='lower right', title="Levels")
    ax.set_title(f"Zonation Map {title_suffix}", fontsize=15)
    ax.set_axis_off()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# [업데이트] grid_size 파라미터 추가 및 위도 기반 실제 거리(Cos 보정) 정밀 계산 로직 적용
def save_dji_files_wgs84(grid_gdf, vra_df, boundary_gdf, field_code, flight_height=0, swath_width=0, grid_size=1.0):
    print(f"  [Output] Generating DJI Compatible Files (WGS84) with {grid_size}m grid resolution...")
    rx_folder = os.path.join(OUTPUT_FOLDER, "DJI", "Rx")
    shp_folder = os.path.join(OUTPUT_FOLDER, "DJI", "ShapeFile")
    os.makedirs(rx_folder, exist_ok=True)
    os.makedirs(shp_folder, exist_ok=True)

    if flight_height > 0 and swath_width > 0:
        filename_base = f"{field_code}_H{flight_height}m_W{swath_width}m"
    else:
        filename_base = f"{field_code}"

    boundary_4326 = boundary_gdf.to_crs(epsg=4326)
    boundary_out = os.path.join(shp_folder, f"{field_code}.shp")
    boundary_4326.to_file(boundary_out, encoding='euc-kr')

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

    # [핵심 수정] 위도에 따른 경도 거리 왜곡(찌그러짐) 현상 보정 (Cos 함수 적용)
    # 1. 필지의 중심 위도 계산
    center_y = (miny + maxy) / 2.0

    # 2. 위도(Y) 1도는 약 111,320m로 일정
    pixel_size_y = grid_size / 111320.0

    # 3. 경도(X) 1도는 위도에 따라 좁아지므로 math.cos(위도)로 보정하여 실제 길이를 맞춤
    pixel_size_x = grid_size / (111320.0 * math.cos(math.radians(center_y)))

    # 너비와 높이 계산 (각각 보정된 픽셀 사이즈 사용)
    width = int((maxx - minx) / pixel_size_x)
    height = int((maxy - miny) / pixel_size_y)

    # Transform 적용 시 X축과 Y축의 해상도를 다르게 입력
    transform = from_origin(minx, maxy, pixel_size_x, pixel_size_y)

    shapes = ((geom, value) for geom, value in zip(grid_4326.geometry, grid_4326['Rx_Rate']))
    out_image = rasterize(shapes=shapes, out_shape=(height, width), transform=transform, fill=0, dtype='float32')

    tif_out = os.path.join(rx_folder, f"{filename_base}.tif")
    out_meta = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": 'float32', "crs": "EPSG:4326", "transform": transform, "nodata": 0
    }
    with rasterio.open(tif_out, "w", **out_meta) as dest:
        dest.write(out_image, 1)

    # TFW 파일 작성 (TIF 파일의 지리적 위치/크기 정보)
    tfw_out = os.path.join(rx_folder, f"{filename_base}.tfw")
    with open(tfw_out, "w") as f:
        f.write(f"{transform.a}\n")  # X축 픽셀 크기 (보정됨)
        f.write(f"{transform.b}\n")  # 회전
        f.write(f"{transform.d}\n")  # 회전
        f.write(f"{transform.e}\n")  # Y축 픽셀 크기 (보정됨, 음수)
        f.write(f"{transform.c}\n")  # 좌상단 X 좌표
        f.write(f"{transform.f}\n")  # 좌상단 Y 좌표

    print(f"    - Rx Map saved: {tif_out}")


# ======================================================
# 2. 메인 프로세스
# ======================================================
def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    if not os.path.exists(BOUNDARY_FOLDER):
        print(f"[Info] '{BOUNDARY_FOLDER}' 폴더가 없습니다. 생성 후 .shp 또는 .zip 파일을 넣어주세요.")
        os.makedirs(BOUNDARY_FOLDER)

    detector = BoundaryDetector()
    vra_calc = VRACalculator(VRA_CSV_PATH)
    tif_files = glob.glob(os.path.join(DATA_FOLDER, "*_GNDVI.tif"))

    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        field_code = filename.split("_")[0] if "_" in filename else "Unknown"
        print(f"\n>>> Processing: {filename} (Field Code: {field_code})")

        field_info = vra_calc.get_field_info(field_code)

        current_crop = field_info['crop'] if field_info is not None and 'crop' in field_info else DEFAULT_CROP
        try:
            current_grid_size = float(
                field_info['grid_size']) if field_info is not None and 'grid_size' in field_info else DEFAULT_GRID_SIZE
        except:
            current_grid_size = DEFAULT_GRID_SIZE

        if field_info is not None and 'masking' in field_info and not pd.isna(field_info['masking']):
            try:
                current_relax_factor = float(field_info['masking'])
            except:
                current_relax_factor = 0.7 if current_crop in ['soybean', 'wheat'] else 0.3
        else:
            current_relax_factor = 0.7 if current_crop in ['soybean', 'wheat'] else 0.3

        # 바운더리 로드 우선순위
        zip_boundary_path_1 = os.path.join(BOUNDARY_FOLDER, f"{field_code}_Boundary.zip")
        zip_boundary_path_2 = os.path.join(BOUNDARY_FOLDER, f"{field_code}.zip")
        input_shp_path = os.path.join(BOUNDARY_FOLDER, f"{field_code}.shp")
        output_shp_path = os.path.join(OUTPUT_FOLDER, "DJI", "ShapeFile", f"{field_code}.shp")

        if os.path.exists(zip_boundary_path_1):
            boundary = detector.load_boundary_from_zip(zip_boundary_path_1)
        elif os.path.exists(zip_boundary_path_2):
            boundary = detector.load_boundary_from_zip(zip_boundary_path_2)
        elif os.path.exists(input_shp_path):
            boundary = detector.load_boundary_from_shp(input_shp_path)
        elif os.path.exists(output_shp_path):
            boundary = detector.load_boundary_from_shp(output_shp_path)
        else:
            boundary = detector.detect_boundary_otsu(tif_path, crop_type=current_crop)

        if boundary is None: continue
        if boundary.crs.is_geographic:
            boundary = boundary.to_crs(epsg=5179)

        try:
            mem_raster = clip_raster_to_boundary(tif_path, boundary)

            grid = create_rotated_grid_with_indices(boundary, grid_size=current_grid_size)
            grid = calculate_grid_mean_stats(grid, mem_raster, col_name='Raw_GNDVI')

            raw_valid = grid.dropna(subset=['Raw_GNDVI'])
            if len(raw_valid) == 0: continue

            # 동적 스무딩(Sigma) 계산
            if field_info is not None and 'sigma' in field_info and not pd.isna(field_info['sigma']):
                try:
                    current_sigma = float(field_info['sigma'])
                    print(f"    [Settings] Sigma loaded from CSV: {current_sigma}")
                except:
                    current_sigma = calculate_optimal_sigma(grid, current_grid_size)
                    print(f"    [Settings] Auto Dynamic Sigma: {current_sigma}")
            else:
                current_sigma = calculate_optimal_sigma(grid, current_grid_size)
                print(f"    [Settings] Auto Dynamic Sigma: {current_sigma}")

            print(
                f"    [Summary] Crop: {current_crop} | Grid: {current_grid_size}m | Mask Factor: {current_relax_factor}")

            # Step 1. 맨땅(흙) 6등급 사전 분리
            print("  - Calculating dynamic soil threshold...")
            grid['Raw_Zone'] = np.nan
            soil_threshold = calculate_dynamic_threshold(grid, relax_factor=current_relax_factor)
            if soil_threshold > -900:
                print(f"    -> Detected Soil Threshold: {soil_threshold:.4f}")
                mask_bare = grid['Raw_GNDVI'] < soil_threshold
                grid.loc[mask_bare, 'Raw_Zone'] = 6
            else:
                mask_bare = pd.Series(False, index=grid.index)

            # Step 2. 사전에 분위별로 구간화 (흙을 제외한 작물 영역만 20% 단위로 5등분)
            valid_crop_mask = grid['Raw_GNDVI'].notna() & (~mask_bare)
            crop_valid_data = grid.loc[valid_crop_mask, 'Raw_GNDVI']

            if len(crop_valid_data) > 0:
                _, raw_bins = pd.qcut(crop_valid_data, q=N_ZONES, retbins=True, duplicates='drop')
                if len(raw_bins) < N_ZONES + 1:
                    _, raw_bins = pd.qcut(crop_valid_data.rank(method='first'), q=N_ZONES, retbins=True)

                # [업데이트] Pandas 데이터 타입 충돌 방지를 위한 .astype(float) 적용
                grid.loc[valid_crop_mask, 'Raw_Zone'] = pd.cut(crop_valid_data, bins=raw_bins, labels=[1, 2, 3, 4, 5],
                                                               include_lowest=True).astype(float)

            # Step 3. 스무딩을 통해 정규분포화 하고, 구간별로 모으는 작업
            grid = apply_categorical_zone_smoothing(grid, zone_col='Raw_Zone', sigma=current_sigma,
                                                    filter_size=MAJORITY_FILTER_SIZE)

            valid_zones = grid[grid['Zone'] != 0]
            stats_df = valid_zones.groupby('Zone')
            zone_stats = []
            for z in range(1, 7):
                if z in stats_df.groups:
                    g = stats_df.get_group(z)
                    zone_stats.append({
                        'Zone': z,
                        'Area_m2': g.geometry.area.sum(),
                        'Mean_GNDVI': g['Raw_GNDVI'].mean()
                    })

            print("  - Calculating VRA Prescription...")
            vra_df = vra_calc.calculate_prescription(field_code, zone_stats)

            f_height = float(field_info.get('height', 0)) if field_info is not None else 0
            f_width = float(field_info.get('width', 0)) if field_info is not None else 0

            if vra_df is not None:
                # [업데이트] DJI 저장 시 current_grid_size 파라미터 전달
                save_dji_files_wgs84(grid, vra_df, boundary, field_code, flight_height=f_height, swath_width=f_width,
                                     grid_size=current_grid_size)
                vra_out_name = filename.replace(".tif", "_VRA.csv")
                vra_df.to_csv(os.path.join(OUTPUT_FOLDER, vra_out_name), index=False, encoding='euc-kr')

            out_img_name = filename.replace(".tif", "_Result.png")
            save_map_image(grid, os.path.join(OUTPUT_FOLDER, out_img_name), f"Result: {filename}", zone_col='Zone',
                           boundary_gdf=boundary)

            mem_raster.close()
            print("  - Processing Complete.")

        except Exception as e:
            print(f"  - Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()