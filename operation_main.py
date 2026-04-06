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
import datetime
import json
import uuid
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
DATA_FOLDER = "data/xag_test_data"
BOUNDARY_FOLDER = "data/ShapeFile"
OUTPUT_FOLDER = "result/xag_result_0406"
VRA_CSV_PATH = "vra_setting/gj_wol_vra.csv"

DEFAULT_GRID_SIZE = 1.0
DEFAULT_CROP = 'rice'
VALID_THRESHOLD = -999.0
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


# [NEW] 가장자리 낭비 방지 로직이 적용된 그리드 생성 함수
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

    # [핵심] 오버랩 비율 계산: 그리드 면적의 40% 이상이 바운더리 내부에 있을 때만 살포 구역으로 포함
    intersection_areas = grid_gdf.intersection(boundary_geom).area
    grid_areas = grid_gdf.area
    overlap_ratio = intersection_areas / grid_areas

    valid_mask = overlap_ratio >= 0.4

    return grid_gdf[valid_mask].copy().reset_index(drop=True)


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


def calculate_optimal_sigma(grid_gdf, grid_size, drone_type='DJI', value_col='Raw_GNDVI'):
    # XAG는 디테일(Pix4D 유사성)을 위해 0.7, DJI는 비행 안정을 위해 1.35
    base_sigma = 0.7 if drone_type == 'XAG' else 1.35

    adjusted_sigma = base_sigma / (grid_size if grid_size > 0 else 1.0)
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
        if relaxed_thresh > MAX_MASK_THRESHOLD:
            final_thresh = MAX_MASK_THRESHOLD
        else:
            final_thresh = relaxed_thresh
        if final_thresh < 0.1:
            return -999
        return final_thresh
    except Exception:
        return -999


def apply_categorical_zone_smoothing(grid_gdf, zone_col='Raw_Zone', sigma=1.0, filter_size=5, max_zone=5):
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

    middle_zone = 2 if max_zone == 3 else 3
    filled_matrix[np.isnan(filled_matrix)] = float(middle_zone)

    if sigma >= 0.2:
        smoothed_matrix = gaussian_filter(filled_matrix, sigma=sigma, mode='nearest')
    else:
        smoothed_matrix = filled_matrix

    rounded_matrix = np.round(smoothed_matrix).astype(int)
    rounded_matrix = np.clip(rounded_matrix, 1, max_zone)
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


def save_map_image(gdf, output_path, title_suffix="", zone_col='Zone', boundary_gdf=None, max_zone=5):
    if max_zone == 3:
        colors = ['#FF0000', '#FFFF00', '#008000', '#808080']
        labels = ["1(High)", "2(Medium)", "3(Low)", "6(Skip)"]
        vmin, vmax = 1, 6
    else:
        colors = ['#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#008000', '#808080']
        labels = ["1(High)", "2", "3", "4", "5(Low)", "6(Skip)"]
        vmin, vmax = 1, 6

    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plot_data = gdf.copy()

    plot_data['plot_zone'] = pd.to_numeric(plot_data[zone_col], errors='coerce').fillna(0).astype(int)
    if max_zone == 3:
        plot_data.loc[plot_data['plot_zone'] == 6, 'plot_zone'] = 4

    plot_data.plot(column='plot_zone', cmap=cmap, linewidth=0, edgecolor='none', ax=ax, vmin=1, vmax=len(colors))
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color='cyan', linewidth=1, alpha=0.7)

    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend_patches, loc='lower right', title="Levels")
    ax.set_title(f"Zonation Map {title_suffix}", fontsize=15)
    ax.set_axis_off()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ======================================================
# 2. DJI & XAG 내보내기 함수
# ======================================================
def save_dji_files_wgs84(grid_gdf, vra_df, boundary_gdf, field_code, flight_height=0, swath_width=0, grid_size=1.0):
    print(f"  [Output] Generating DJI Compatible Files (WGS84) with {grid_size}m grid resolution...")
    rx_folder = os.path.join(OUTPUT_FOLDER, "DJI", "Rx")
    shp_folder = os.path.join(OUTPUT_FOLDER, "DJI", "ShapeFile")
    os.makedirs(rx_folder, exist_ok=True)
    os.makedirs(shp_folder, exist_ok=True)

    current_date = datetime.datetime.now().strftime("%m%d")
    grid_str = f"{grid_size:g}"
    if flight_height > 0 and swath_width > 0:
        filename_base = f"{field_code}_DJI_{grid_str}m_H{flight_height}m_W{swath_width}m_{current_date}"
    else:
        filename_base = f"{field_code}_DJI_{grid_str}m_{current_date}"

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
    center_y = (miny + maxy) / 2.0
    pixel_size_y = grid_size / 111320.0
    pixel_size_x = grid_size / (111320.0 * math.cos(math.radians(center_y)))

    width = int((maxx - minx) / pixel_size_x)
    height = int((maxy - miny) / pixel_size_y)
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

    tfw_out = os.path.join(rx_folder, f"{filename_base}.tfw")
    with open(tfw_out, "w") as f:
        for val in [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]:
            f.write(f"{val}\n")
    print(f"    - DJI Rx Map saved: {tif_out}")


def save_xag_files_wgs84(grid_gdf, vra_df, boundary_gdf, field_code, grid_size=1.0):
    print(f"  [Output] Generating XAG Compatible Files (JSON & KML) with {grid_size}m grid resolution...")
    xag_folder = os.path.join(OUTPUT_FOLDER, "XAG")
    os.makedirs(xag_folder, exist_ok=True)

    current_date = datetime.datetime.now().strftime("%m%d")
    grid_str = f"{grid_size:g}"
    filename_base = f"{field_code}_XAG_{grid_str}m_{current_date}"

    # 1. 바운더리를 WGS84로 변환 및 멀티폴리곤 강제 병합
    boundary_4326 = boundary_gdf.to_crs(epsg=4326)
    geom = boundary_4326.union_all()
    if geom.geom_type == 'MultiPolygon':
        geom = max(geom.geoms, key=lambda a: a.area)

    # KML 및 WKT 좌표 포맷팅 (소수점 8자리, 띄어쓰기 엄격 통제)
    def format_coords_wkt(coords):
        return ",".join([f"{lon:.8f} {lat:.8f}" for lon, lat in coords])

    kml_coords = " ".join([f"{lon:.8f},{lat:.8f}" for lon, lat in geom.exterior.coords])

    wkt_str = f"POLYGON(({format_coords_wkt(geom.exterior.coords)})"
    for interior in geom.interiors:
        wkt_str += f",({format_coords_wkt(interior.coords)})"
    wkt_str += ")"

    # 2. XAG KML 생성
    kml_content = f"""<?xml version='1.0' encoding='utf-8'?>
<kml xmlns="http://www.opengis.net/kml/2.2">
 <Document id="root_doc">
  <Schema id="layer" name="layer">
   <SimpleField name="type" type="string"/>
   <SimpleField name="visualType" type="string"/>
  </Schema>
  <Folder>
   <name>Field_Boundary</name>
   <Placemark id="layer.1">
    <name>{filename_base}</name>
    <description>Boundaries</description>
    <Style>
     <LineStyle><color>ff0000ff</color></LineStyle>
     <PolyStyle><fill>0</fill></PolyStyle>
    </Style>
    <ExtendedData>
     <SchemaData schemaUrl="#layer">
      <SimpleData name="type">boundary</SimpleData>
      <SimpleData name="visualType">BOUNDARY</SimpleData>
     </SchemaData>
    </ExtendedData>
    <Polygon>
     <outerBoundaryIs>
      <LinearRing>
       <coordinates>{kml_coords}</coordinates>
      </LinearRing>
     </outerBoundaryIs>
    </Polygon>
   </Placemark>
  </Folder>
 </Document>
</kml>"""
    kml_out = os.path.join(xag_folder, f"{filename_base}_Boundary.kml")
    with open(kml_out, "w", encoding='utf-8') as f:
        f.write(kml_content)

    # 3. JSON 데이터 생성 및 수학적 격자 오차 완벽 보정
    grid_4326 = grid_gdf.to_crs(epsg=4326)
    grid_4326['XAG_Zone'] = grid_4326['Zone'].apply(lambda z: z if z in [1, 2, 3] else 0)

    minx, miny, maxx, maxy = grid_4326.total_bounds
    center_y = (miny + maxy) / 2.0
    pixel_size_y = grid_size / 111320.0
    pixel_size_x = grid_size / (111320.0 * math.cos(math.radians(center_y)))

    width = math.ceil((maxx - minx) / pixel_size_x)
    height = math.ceil((maxy - miny) / pixel_size_y)

    # 계산된 가로/세로 칸 수에 맞게 전체 Bounding Box 좌표를 역산 (오차 0%)
    exact_maxx = minx + (width * pixel_size_x)
    exact_miny = maxy - (height * pixel_size_y)

    transform = from_origin(minx, maxy, pixel_size_x, pixel_size_y)

    shapes = ((g, value) for g, value in zip(grid_4326.geometry, grid_4326['XAG_Zone']))
    out_image = rasterize(shapes=shapes, out_shape=(height, width), transform=transform, fill=0, dtype='int32')
    weight_data = out_image.flatten().tolist()

    data_type_level = []
    for _, row in vra_df.iterrows():
        try:
            zone_idx = int(str(row['Zone']).split('(')[0])
            rate_val = float(row['Rate(kg/ha)'])
            if zone_idx in [1, 2, 3]:
                # XAG의 dosage 단위(g/m²)에 맞게 kg/ha 값을 10으로 나눔
                dosage_g_m2 = rate_val / 10.0
                data_type_level.append({"dosage": round(dosage_g_m2, 2), "level": zone_idx})
        except:
            continue

    cell_size_val = int(grid_size) if grid_size == int(grid_size) else float(grid_size)

    # JSON 딕셔너리 조립 (Poly 오류 일으키던 중복 블록 삭제 완료)
    xag_json = {
        "borderWKT": wkt_str,
        "cellSize": cell_size_val,
        "columns": width,
        "dataType": 3,
        "dataTypeLevel": data_type_level,
        "guid": str(uuid.uuid4()),
        "name": filename_base,
        "originEndLat": float(f"{maxy:.14f}"),
        "originEndLng": float(f"{exact_maxx:.14f}"),
        "originLat": float(f"{exact_miny:.14f}"),
        "originLng": float(f"{minx:.14f}"),
        "rotation": 0,
        "rows": height,
        "source": "Pix4D",
        "version": 1,
        "weightData": weight_data,
        "workType": 2
    }

    json_out = os.path.join(xag_folder, f"{filename_base}_Prescription.json")
    with open(json_out, "w", encoding='utf-8') as f:
        json.dump(xag_json, f, indent=4)

    print(f"    - XAG KML saved: {kml_out}")
    print(f"    - XAG JSON saved: {json_out}")


# ======================================================
# 3. 메인 프로세스
# ======================================================
def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(BOUNDARY_FOLDER):
        os.makedirs(BOUNDARY_FOLDER)

    detector = BoundaryDetector()
    vra_calc = VRACalculator(VRA_CSV_PATH)
    tif_files = glob.glob(os.path.join(DATA_FOLDER, "*_GNDVI.tif"))

    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        field_code = filename.split("_")[0] if "_" in filename else "Unknown"
        print(f"\n>>> Processing: {filename} (Field Code: {field_code})")

        field_info = vra_calc.get_field_info(field_code)

        # [핵심] 기체 타입 인식
        drone_type = str(field_info.get('drone_type', 'DJI')).strip().upper() if field_info is not None else 'DJI'

        # 기체별 등급 및 필터 사이즈 동적 할당
        current_n_zones = 3 if drone_type == 'XAG' else 5
        current_filter_size = 3 if drone_type == 'XAG' else 5

        current_crop = field_info['crop'] if field_info is not None and 'crop' in field_info else DEFAULT_CROP

        try:
            current_grid_size = float(
                field_info['grid_size']) if field_info is not None and 'grid_size' in field_info else DEFAULT_GRID_SIZE
        except:
            current_grid_size = DEFAULT_GRID_SIZE

        # [NEW] XAG 기체 감지 시 CSV 설정값 무시하고 무조건 5m로 강제 고정
        if drone_type == 'XAG':
            current_grid_size = 5.0
            print(f"    [Info] XAG 기체 감지: 그리드 크기를 5.0m로 강제 고정합니다.")

        if field_info is not None and 'masking' in field_info and not pd.isna(field_info['masking']):
            try:
                current_relax_factor = float(field_info['masking'])
            except:
                current_relax_factor = 0.7 if current_crop in ['soybean', 'wheat'] else 0.3
        else:
            current_relax_factor = 0.7 if current_crop in ['soybean', 'wheat'] else 0.3

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

            if field_info is not None and 'sigma' in field_info and not pd.isna(field_info['sigma']):
                try:
                    current_sigma = float(field_info['sigma'])
                    print(f"    [Settings] Sigma loaded from CSV: {current_sigma}")
                except:
                    current_sigma = calculate_optimal_sigma(grid, current_grid_size, drone_type=drone_type)
                    print(f"    [Settings] Auto Dynamic Sigma: {current_sigma}")
            else:
                current_sigma = calculate_optimal_sigma(grid, current_grid_size, drone_type=drone_type)
                print(f"    [Settings] Auto Dynamic Sigma: {current_sigma}")

            print(
                f"    [Summary] Drone: {drone_type} | Zones: {current_n_zones} | Filter: {current_filter_size} | Grid: {current_grid_size}m")

            grid['Raw_Zone'] = np.nan
            soil_threshold = calculate_dynamic_threshold(grid, relax_factor=current_relax_factor)
            if soil_threshold > -900:
                mask_bare = grid['Raw_GNDVI'] < soil_threshold
                grid.loc[mask_bare, 'Raw_Zone'] = 6
            else:
                mask_bare = pd.Series(False, index=grid.index)

            valid_crop_mask = grid['Raw_GNDVI'].notna() & (~mask_bare)
            crop_valid_data = grid.loc[valid_crop_mask, 'Raw_GNDVI']

            if len(crop_valid_data) > 0:
                _, raw_bins = pd.qcut(crop_valid_data, q=current_n_zones, retbins=True, duplicates='drop')
                if len(raw_bins) < current_n_zones + 1:
                    _, raw_bins = pd.qcut(crop_valid_data.rank(method='first'), q=current_n_zones, retbins=True)

                labels_list = list(range(1, current_n_zones + 1))
                grid.loc[valid_crop_mask, 'Raw_Zone'] = pd.cut(crop_valid_data, bins=raw_bins, labels=labels_list,
                                                               include_lowest=True).astype(float)

            grid = apply_categorical_zone_smoothing(
                grid,
                zone_col='Raw_Zone',
                sigma=current_sigma,
                filter_size=current_filter_size,
                max_zone=current_n_zones
            )

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
                if drone_type == 'XAG':
                    save_xag_files_wgs84(grid, vra_df, boundary, field_code, grid_size=current_grid_size)
                else:
                    save_dji_files_wgs84(grid, vra_df, boundary, field_code, flight_height=f_height,
                                         swath_width=f_width, grid_size=current_grid_size)

                vra_out_name = f"{field_code}_{drone_type}_VRA.csv"
                vra_df.to_csv(os.path.join(OUTPUT_FOLDER, vra_out_name), index=False, encoding='euc-kr')

            out_img_name = f"{field_code}_{drone_type}_Result.png"
            save_map_image(grid, os.path.join(OUTPUT_FOLDER, out_img_name), f"Result: {field_code} ({drone_type})",
                           zone_col='Zone', boundary_gdf=boundary, max_zone=current_n_zones)

            mem_raster.close()
            print("  - Processing Complete.")

        except Exception as e:
            print(f"  - Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()