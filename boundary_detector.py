import os
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from skimage.filters import threshold_otsu
from shapely.geometry import shape
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, binary_dilation, binary_erosion
from skimage.measure import label
import warnings


class BoundaryDetector:
    def __init__(self):
        pass

    def load_boundary_from_zip(self, zip_path):
        safe_path = zip_path.replace("\\", "/")
        if not safe_path.startswith("zip://"):
            safe_path = f"zip://{safe_path}"
        try:
            gdf = gpd.read_file(safe_path)
            if gdf.crs is None: gdf.crs = "EPSG:4326"
            print('    [Info] Loading boundary from zip file.')
            return gdf
        except Exception as e:
            print(f"    [Error] 바운더리 로드 실패: {e}")
            return None

    # [NEW] SHP 파일 로드 함수 추가
    def load_boundary_from_shp(self, shp_path):
        try:
            gdf = gpd.read_file(shp_path)
            # DJI ShapeFile은 보통 좌표계 정보(.prj)가 포함되어 있지만, 없을 경우 WGS84로 가정
            if gdf.crs is None:
                gdf.crs = "EPSG:4326"
            print(f"    [Info] 기존 바운더리 재사용: {os.path.basename(shp_path)}")
            return gdf
        except Exception as e:
            print(f"    [Error] SHP 파일 로드 실패: {e}")
            return None

    def detect_boundary_otsu(self, tif_path, crop_type='rice'):
        """
        crop_type: 'rice', 'soybean', 'wheat'
        """
        crop_type = str(crop_type).lower().strip()
        print(f"    [Info] 바운더리 감지 시작 (Type: {crop_type})...")

        try:
            with rasterio.open(tif_path) as src:
                # 1. 다운샘플링
                decimation = 4 if (src.width > 2000 or src.height > 2000) else 1
                out_shape = (src.count, int(src.height / decimation), int(src.width / decimation))
                data = src.read(1, out_shape=out_shape)

                transform = src.transform * src.transform.scale(
                    (src.width / out_shape[-1]), (src.height / out_shape[-2])
                )

                # 2. 유효 데이터 마스킹
                SAFE_FLOOR = 0.05
                valid_mask = ~np.isnan(data)
                if src.nodata is not None:
                    valid_mask &= (data != src.nodata)
                candidate_mask = valid_mask & (data > SAFE_FLOOR)

                if np.sum(candidate_mask) == 0:
                    print("    [Warning] 유효한 식생 데이터가 거의 없습니다.")
                    return None

                # 3. Otsu 임계값 계산
                valid_pixels = data[candidate_mask]
                try:
                    otsu_thresh = threshold_otsu(valid_pixels)
                except:
                    otsu_thresh = np.mean(valid_pixels)

                # 작물별 임계값 보정
                if crop_type == 'soybean':
                    thresh_factor = 0.85
                elif crop_type == 'wheat':
                    thresh_factor = 0.95
                else:
                    thresh_factor = 0.90

                final_thresh = otsu_thresh * thresh_factor
                if final_thresh < SAFE_FLOOR: final_thresh = SAFE_FLOOR
                print(f"    - Threshold (Lower): {final_thresh:.4f}")

                # 4. 이진화
                binary_img = (data > final_thresh)

                # 5. 형태학적 처리 (Morphology)
                if crop_type == 'soybean':
                    binary_img = binary_closing(binary_img, structure=np.ones((7, 7)))
                    binary_img = binary_dilation(binary_img, structure=np.ones((3, 3)), iterations=2)
                    binary_img = binary_fill_holes(binary_img)

                    labeled_img, num_features = label(binary_img, return_num=True, connectivity=2)
                    if num_features > 0:
                        sizes = np.bincount(labeled_img.ravel())
                        sizes[0] = 0
                        max_label = sizes.argmax()
                        binary_img = (labeled_img == max_label)

                    binary_img = binary_erosion(binary_img, structure=np.ones((3, 3)), iterations=2)
                    binary_img = binary_opening(binary_img, structure=np.ones((3, 3)))

                elif crop_type == 'wheat':
                    binary_img = binary_opening(binary_img, structure=np.ones((3, 3)))
                    binary_img = binary_closing(binary_img, structure=np.ones((5, 5)))
                    binary_img = binary_fill_holes(binary_img)

                else:  # Rice
                    open_structure = np.ones((3, 3))
                    binary_img = binary_opening(binary_img, structure=open_structure)
                    close_structure = np.ones((5, 5))
                    binary_img = binary_closing(binary_img, structure=close_structure)
                    binary_img = binary_fill_holes(binary_img)

                binary_img = binary_img.astype('uint8')

                # 벡터화
                shapes_gen = shapes(binary_img, mask=(binary_img == 1), transform=transform)
                polygons = []
                for geom, val in shapes_gen:
                    if val == 1:
                        polygons.append(shape(geom))

                if not polygons: return None

                gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=src.crs)

                if len(gdf) > 1:
                    gdf['area'] = gdf.geometry.area
                    gdf = gdf.sort_values('area', ascending=False).iloc[0:1]

                gdf['geometry'] = gdf.geometry.simplify(0.3)

                return gdf

        except Exception as e:
            print(f"    [Error] 바운더리 생성 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None