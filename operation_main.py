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
from matplotlib import font_manager
from scipy.ndimage import gaussian_filter, generic_filter
from skimage.filters import threshold_otsu
import warnings


def _setup_korean_font():
    """결과 지도(PNG) 제목·라벨의 한글 깨짐(□□□) 방지 — 한글 지원 폰트 지정.
    Windows 기본 '맑은 고딕' 우선, 없으면 나눔/기타 순으로 탐색."""
    candidates = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic",
                  "AppleGothic", "Gulim", "Batang", "Dotum"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    # 한글 폰트 사용 시 음수 부호(−)가 깨지지 않도록
    plt.rcParams["axes.unicode_minus"] = False


_setup_korean_font()

# [모듈 Import]
from boundary_detector import BoundaryDetector
from vra_calculator import VRACalculator

# 설정 및 경고 무시
warnings.filterwarnings("ignore")
pd.set_option('future.no_silent_downcasting', True)

# ======================================================
# 0. 설정
# ======================================================
DATA_FOLDER = "data/gj_data"
BOUNDARY_FOLDER = "data/gj_boundary"
OUTPUT_FOLDER = "result/gj_0731"
VRA_CSV_PATH = "vra_setting/gj_vra.csv"

DEFAULT_GRID_SIZE = 1.0
DEFAULT_CROP = 'rice'
VALID_THRESHOLD = -999.0
MAX_MASK_THRESHOLD = 0.40

# 경계 파일이 없을 때 자동 감지 방식:
#   'footprint' - 유효 데이터(non-nodata) 외곽선 (드론 정사영상이 필지대로 잘린 경우 권장)
#   'otsu'      - GNDVI 식생 임계값 기반 (사각형 영상에 주변 논/도로가 포함된 경우)
BOUNDARY_METHOD = 'footprint'


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


def find_boundary_zip(boundary_folder, field_code):
    """경계 폴더에서 필지코드에 해당하는 zip 파일 탐색.
    'GJR1.zip', 'GJR1_Boundary.zip', 'GJR1_boundary.zip' 등 네이밍 변형과
    대소문자 차이를 모두 허용한다 (파일명 첫 토큰 == 필지코드 기준)."""
    fc = str(field_code).lower()
    for zip_path in sorted(glob.glob(os.path.join(boundary_folder, "*.zip"))):
        base = os.path.splitext(os.path.basename(zip_path))[0].lower()
        if base == fc or base.split("_")[0] == fc:
            return zip_path
    return None


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

    intersection_areas = grid_gdf.intersection(boundary_geom).area
    grid_areas = grid_gdf.area
    overlap_ratio = intersection_areas / grid_areas

    # [UPDATE] 바운더리 밖으로 삐져나가는 픽셀을 완전히 제거하여 비료 모자람 해결 (95% 이상 겹칠 때만 포함)
    valid_mask = overlap_ratio >= 0.95

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
        # 격자는 분석 CRS(5179)로 생성되지만 클립 래스터는 원본 CRS를 유지할 수 있다
        # (예: EPSG:4326 드론 영상). 샘플링 시 격자를 래스터 CRS로 맞춰야 값이 잡힌다.
        if grid_gdf.crs is not None and src.crs is not None and grid_gdf.crs != src.crs:
            sample_geoms = grid_gdf.to_crs(src.crs).geometry.tolist()
        else:
            sample_geoms = grid_gdf.geometry.tolist()
        for geom in sample_geoms:
            try:
                out_image, _ = mask(src, [geom], crop=True)
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


def save_map_image(gdf, output_path, title_suffix="", zone_col='Zone', boundary_gdf=None, max_zone=5,
                   info_text=""):
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
    # 제목: 폰트 축소 + 상단 여백 확보(정보 줄과 겹치지 않도록)
    ax.set_title(f"Zonation Map {title_suffix}", fontsize=11, pad=22)
    # 확인용 정보(총 면적/총 비료량)를 제목 바로 아래 별도 줄로 배치
    if info_text:
        ax.text(0.5, 1.0, info_text, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=10, color='#1a3c5e', fontweight='bold')
    ax.set_axis_off()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ======================================================
# 2. DJI & XAG 내보내기 함수
# ======================================================
def short_field_name(field_code):
    """드론 화면 표시용 짧은 필지명 — 앞쪽 행정구역(도/시/군/구)을 제거하고
    읍/면/동/리부터 남긴다. 예: '경기도 화성시 만세구 우정읍 이화리 1409' → '우정읍 이화리 1409'.
    주소가 아닌 코드(예: 'HSR1', 'GJR5')는 공백/행정접미사가 없어 그대로 반환된다."""
    tokens = str(field_code).split()
    if not tokens:
        return str(field_code)
    drop = ('도', '시', '군', '구')
    i = 0
    while i < len(tokens) and tokens[i].endswith(drop):
        i += 1
    if i >= len(tokens):     # 전부 잘리면(비정상) 원본 유지
        return str(field_code)
    return " ".join(tokens[i:]).strip() or str(field_code)


def save_dji_files_wgs84(grid_gdf, vra_df, boundary_gdf, field_code, flight_height=0, swath_width=0, grid_size=1.0):
    print(f"  [Output] Generating DJI Compatible Files (WGS84) with {grid_size}m grid resolution...")
    rx_folder = os.path.join(OUTPUT_FOLDER, "DJI", "Rx")
    shp_folder = os.path.join(OUTPUT_FOLDER, "DJI", "ShapeFile")
    os.makedirs(rx_folder, exist_ok=True)
    os.makedirs(shp_folder, exist_ok=True)

    # 드론 표시용 짧은 이름 사용 (도/시/군/구 제거, 뒤 기체·격자·날짜 정보 생략)
    short = short_field_name(field_code)
    filename_base = short

    # Pix4D 호환: 경계 ShapeFile에 비행고도(height)/살포폭(line_space)/이름(name) 속성 포함 (UTF-8)
    boundary_4326 = boundary_gdf.to_crs(epsg=4326)
    boundary_geom_4326 = boundary_4326.union_all()
    boundary_attr = gpd.GeoDataFrame(
        {
            'height': [float(flight_height) if flight_height > 0 else 3.0],
            'line_space': [float(swath_width) if swath_width > 0 else 5.0],
            'name': [short],
        },
        geometry=[boundary_geom_4326], crs="EPSG:4326"
    )
    boundary_out = os.path.join(shp_folder, f"{short}.shp")
    boundary_attr.to_file(boundary_out, encoding='utf-8')

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
    # Pix4D 호환: nodata 태그 없이 저장 (0 = 살포 제외 구역의 실제 값)
    out_meta = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": 'float32', "crs": "EPSG:4326", "transform": transform, "nodata": None
    }
    with rasterio.open(tif_out, "w", **out_meta) as dest:
        dest.write(out_image, 1)

    # World File 표준: 5,6번째 줄은 좌상단 픽셀의 '중심' 좌표 (모서리 + 반 픽셀)
    tfw_out = os.path.join(rx_folder, f"{filename_base}.tfw")
    center_x = transform.c + transform.a / 2.0
    center_y = transform.f + transform.e / 2.0
    with open(tfw_out, "w") as f:
        for val in [transform.a, transform.d, transform.b, transform.e, center_x, center_y]:
            f.write(f"{val:.10f}\n")
    print(f"    - DJI Rx Map saved: {tif_out}")


def save_xag_files_wgs84(grid_gdf, vra_df, boundary_gdf, field_code, grid_size=1.0):
    print(f"  [Output] Generating XAG Compatible Files (JSON & KML) with {grid_size}m grid resolution...")
    xag_folder = os.path.join(OUTPUT_FOLDER, "XAG")
    os.makedirs(xag_folder, exist_ok=True)

    # 드론 표시용 짧은 이름 (도/시/군/구 제거, 뒤 정보 생략)
    filename_base = short_field_name(field_code)

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

    # 내부 링(구멍)도 KML에 반영 — WKT(borderWKT)와 경계 일치 보장
    inner_rings = ""
    for interior in geom.interiors:
        inner_coords = " ".join([f"{lon:.8f},{lat:.8f}" for lon, lat in interior.coords])
        inner_rings += f"""
    <innerBoundaryIs>
     <LinearRing>
      <coordinates>{inner_coords}</coordinates>
     </LinearRing>
    </innerBoundaryIs>"""

    # 2. XAG KML 생성 (Pix4D 구조 일치: Folder 래퍼 없이 Document 바로 아래 Placemark)
    kml_content = f"""<?xml version='1.0' encoding='utf-8'?>
<kml xmlns="http://www.opengis.net/kml/2.2">
 <Document id="root_doc">
  <Schema id="layer" name="layer">
   <SimpleField name="type" type="string"/>
   <SimpleField name="visualType" type="string"/>
  </Schema>
  <Placemark id="layer.1">
   <name>{filename_base}</name>
   <description>Boundaries</description>
   <Style>
    <LineStyle>
     <color>ff0000ff</color>
    </LineStyle>
    <PolyStyle>
     <fill>0</fill>
    </PolyStyle>
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
    </outerBoundaryIs>{inner_rings}
   </Polygon>
  </Placemark>
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
                # Pix4D 호환: dosage는 정수로 저장 (Pix4D 산출물이 정수 체계)
                dosage_g_m2 = int(round(rate_val / 10.0))
                data_type_level.append({"dosage": dosage_g_m2, "level": zone_idx})
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


PYEONG_M2 = 3.3057851   # 1평 = 400/121 m²
BAG_KG = 20.0           # 비료 1포대 = 20kg


def _calc_mode_label(field_info):
    """이 필지가 면적비율 모드인지 절대량 모드인지 라벨 반환."""
    if field_info is None:
        return "절대량"
    try:
        rk = field_info['rate_kg'] if 'rate_kg' in field_info.index else None
        ra = field_info['rate_area'] if 'rate_area' in field_info.index else None
        if rk is not None and ra is not None and not pd.isna(rk) and not pd.isna(ra):
            return "면적비율"
    except Exception:
        pass
    return "절대량"


def _field_str(field_info, key):
    """설정 행에서 문자열 값 안전 추출 (컬럼 없음/빈칸/NaN → '')."""
    if field_info is None:
        return ""
    try:
        if key not in field_info.index:
            return ""
        v = field_info[key]
        if pd.isna(v):
            return ""
        return str(v).strip()
    except Exception:
        return ""


def _fert_setting_label(field_info):
    """처음 세팅한 비료 기준을 표기용 문자열로. 예: 면적비율 '400평에 20kg', 절대 '60kg (절대)'."""
    if field_info is None:
        return ""
    rk = VRACalculator._num(field_info, 'rate_kg')
    ra = VRACalculator._num(field_info, 'rate_area')
    if rk is not None and ra is not None:
        unit = _field_str(field_info, 'area_unit') or '평'
        return f"{ra:g}{unit}에 {rk:g}kg"
    total = VRACalculator._num(field_info, 'total')
    if total is not None:
        return f"{total:g}kg (절대)"
    return ""


def consolidate_uniform_vra(vra_df):
    """균등 처방(살포 구역들의 kg/ha가 모두 동일)일 때 VRA.csv 표시를 정리.
    5개 등급이 같은 값으로 나열돼 변량처럼 보이는 혼동을 없애기 위해,
    살포 구역을 단일 '균등(Uniform)' 행으로 합치고 나지(Zone 6)는 0으로 유지한다.
    (실제 Rx 처방맵 생성은 원본 vra_df를 쓰므로 영향 없음 — 표시 전용)"""
    def zidx(z):
        try:
            return int(str(z).split('(')[0])
        except Exception:
            return -1
    spray = vra_df[vra_df['Zone'].apply(lambda z: zidx(z) != 6)]
    skip = vra_df[vra_df['Zone'].apply(lambda z: zidx(z) == 6)]
    if len(spray) <= 1:
        return vra_df
    rates = spray['Rate(kg/ha)'].round(2).unique()
    if len(rates) != 1:
        return vra_df   # 변량 → 그대로
    area = float(spray['Area(ha)'].sum())
    total = float(spray['Total(kg)'].sum())
    gndvi = float((spray['GNDVI'] * spray['Area(ha)']).sum() / area) if area > 0 else float(spray['GNDVI'].mean())
    row = {
        'Field': spray['Field'].iloc[0],
        'Zone': '균등(Uniform)',
        'GNDVI': round(gndvi, 4),
        'Area(ha)': round(area, 4),
        'Rate(kg/ha)': float(rates[0]),
        'Total(kg)': round(total, 2),
    }
    out = pd.DataFrame([row], columns=vra_df.columns)
    if len(skip) > 0:
        out = pd.concat([out, skip[vra_df.columns]], ignore_index=True)
    return out


def save_run_summary(rows, output_folder):
    """실행한 전체 필지의 처방 결과를 한 파일(엑셀)로 저장.
    xlsx 우선, 실패 시 CSV(utf-8-sig, 엑셀에서 한글 정상)로 대체."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    xlsx_path = os.path.join(output_folder, "처방요약.xlsx")
    try:
        df.to_excel(xlsx_path, index=False, sheet_name="처방요약")
        print(f"\n[요약] 전체 처방 요약 저장: {xlsx_path}")
    except Exception as e:
        csv_path = os.path.join(output_folder, "처방요약.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n[요약] xlsx 저장 실패({e}) → CSV로 저장: {csv_path}")


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
    # '*_GNDVI.tif' 및 '*_GNDVI.data.tif' 등 접미사 변형을 모두 허용 (필지코드는 첫 토큰 유지)
    tif_files = sorted(set(glob.glob(os.path.join(DATA_FOLDER, "*_GNDVI*.tif"))))

    summary_rows = []   # 전체 필지 처방 요약(엑셀 출력용)

    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        field_code = filename.split("_")[0].strip() if "_" in filename else "Unknown"
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

        zip_boundary_path = find_boundary_zip(BOUNDARY_FOLDER, field_code)
        input_shp_path = os.path.join(BOUNDARY_FOLDER, f"{field_code}.shp")
        output_shp_path = os.path.join(OUTPUT_FOLDER, "DJI", "ShapeFile", f"{field_code}.shp")

        if zip_boundary_path is not None:
            boundary = detector.load_boundary_from_zip(zip_boundary_path)
        elif os.path.exists(input_shp_path):
            boundary = detector.load_boundary_from_shp(input_shp_path)
        elif os.path.exists(output_shp_path):
            boundary = detector.load_boundary_from_shp(output_shp_path)
        elif BOUNDARY_METHOD == 'footprint':
            boundary = detector.detect_boundary_footprint(tif_path)
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
            # 실측 필지 전체 면적(m², 경계 기준) — 비율 모드(rate_kg/rate_area)의 자동 면적에 사용.
            # boundary는 이미 EPSG:5179(미터)로 변환된 상태.
            try:
                field_area_m2 = float(boundary.geometry.area.sum())
            except Exception:
                field_area_m2 = None
            vra_df = vra_calc.calculate_prescription(field_code, zone_stats, field_area_m2=field_area_m2)

            f_height = float(field_info.get('height', 0)) if field_info is not None else 0
            f_width = float(field_info.get('width', 0)) if field_info is not None else 0

            img_info = ""   # 확인용 이미지에 표시할 '총 O평 / 비료 O kg' 텍스트
            if vra_df is not None:
                if drone_type == 'XAG':
                    save_xag_files_wgs84(grid, vra_df, boundary, field_code, grid_size=current_grid_size)
                else:
                    save_dji_files_wgs84(grid, vra_df, boundary, field_code, flight_height=f_height,
                                         swath_width=f_width, grid_size=current_grid_size)

                vra_out_name = f"{short_field_name(field_code)}_VRA.csv"
                # CSV 표시용: 균등 처방이면 단일 '균등' 행으로 합쳐 저장 (Rx 생성은 위에서 원본 vra_df 사용 완료)
                consolidate_uniform_vra(vra_df).to_csv(
                    os.path.join(OUTPUT_FOLDER, vra_out_name), index=False, encoding='euc-kr')

                # 전체 요약(엑셀)용 1행 집계
                total_kg = float(vra_df['Total(kg)'].sum())
                spray_ha = float(vra_df['Area(ha)'].sum())
                spray_py = spray_ha * 10000.0 / PYEONG_M2
                field_py = (field_area_m2 / PYEONG_M2) if field_area_m2 else None
                avg_rate = (total_kg / spray_ha) if spray_ha > 0 else 0.0
                # 확인용 이미지 정보: 총 필지면적(평) / 총 비료량(kg, 포) / 그리드 크기
                area_disp = field_py if field_py else spray_py
                img_info = (f"총 {area_disp:,.0f}평  |  비료 {total_kg:,.1f}kg ({total_kg / BAG_KG:.1f}포)"
                            f"  |  그리드 {current_grid_size:g}m")
                summary_rows.append({
                    "필지": field_code,
                    "경작자": _field_str(field_info, '경작자'),
                    "기체": drone_type,
                    "계산방식": _calc_mode_label(field_info),
                    "비료기준": _fert_setting_label(field_info),
                    "총비료량(kg)": round(total_kg, 2),
                    "포대수(20kg)": round(total_kg / BAG_KG, 2),
                    "필요포대수(올림)": int(math.ceil(total_kg / BAG_KG)),
                    "필지면적(평)": round(field_py, 1) if field_py else "",
                    "살포면적(평)": round(spray_py, 1),
                    "살포면적(ha)": round(spray_ha, 4),
                    "평균살포량(kg/ha)": round(avg_rate, 1),
                })

            out_img_name = f"{short_field_name(field_code)}_Result.png"
            save_map_image(grid, os.path.join(OUTPUT_FOLDER, out_img_name),
                           f"Result: {short_field_name(field_code)} ({drone_type})",
                           zone_col='Zone', boundary_gdf=boundary, max_zone=current_n_zones,
                           info_text=img_info)

            mem_raster.close()
            print("  - Processing Complete.")

        except Exception as e:
            print(f"  - Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    # 전체 실행 결과 요약(엑셀) 저장
    save_run_summary(summary_rows, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()