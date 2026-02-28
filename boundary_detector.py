import os
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from skimage.filters import threshold_otsu
from shapely.geometry import shape
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, binary_erosion
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

    def detect_boundary_otsu(self, tif_path, crop_type='rice'):
        """
        crop_type: 'rice', 'soybean', 'wheat'
        """
        crop_type = str(crop_type).lower().strip()
        print(f"    [Info] 바운더리 감지 시작 (Type: {crop_type})...")

        try:
            with rasterio.open(tif_path) as src:
                # 1. 다운샘플링 (대면적 필지 속도 최적화)
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

                # ---------------------------------------------------------
                # [분기 1] 작물별 임계값 설정 (Lower Cutoff)
                # ---------------------------------------------------------
                if crop_type == 'soybean':
                    # 콩: 배경(흙)과 분리. 너무 높게 잡으면 작물 손실되므로 적당히 완화
                    thresh_factor = 0.90
                elif crop_type == 'wheat':
                    thresh_factor = 0.95
                else:
                    # 벼: 배경(물)과 분리
                    thresh_factor = 0.90

                final_thresh = otsu_thresh * thresh_factor
                if final_thresh < SAFE_FLOOR:
                    final_thresh = SAFE_FLOOR

                print(f"    - Threshold: {final_thresh:.4f}")

                # 이진화 (Upper Limit 로직 삭제 -> 작물 보호 우선)
                binary_img = (data > final_thresh)

                # ---------------------------------------------------------
                # [분기 2] 형태학적 처리 (Morphology)
                # ---------------------------------------------------------
                if crop_type == 'soybean':
                    # [콩 전용 개선 로직]
                    # 1. Fill Holes (선 채움): 내부의 빈 공간을 먼저 메워서 덩어리를 단단하게 만듦
                    #    이걸 먼저 해야 나중에 깎을 때 안쪽이 무너지지 않음
                    binary_img = binary_fill_holes(binary_img)

                    # 2. Opening: 자잘한 잡초 점 제거 및 연결 고리 끊기
                    binary_img = binary_opening(binary_img, structure=np.ones((3, 3)))

                    # 3. Largest Component: 가장 큰 덩어리(본밭)만 남기기
                    labeled_img, num_features = label(binary_img, return_num=True, connectivity=2)
                    if num_features > 0:
                        sizes = np.bincount(labeled_img.ravel())
                        sizes[0] = 0
                        max_label = sizes.argmax()
                        binary_img = (labeled_img == max_label)

                    # 4. Gentle Erosion (살짝 깎기): 가장자리의 잡초층만 벗겨냄
                    # 다운샘플링 상태이므로 iteration을 1회만 줘도 1~2m 효과가 있음
                    binary_img = binary_erosion(binary_img, structure=np.ones((3, 3)), iterations=1)

                elif crop_type == 'wheat':
                    binary_img = binary_opening(binary_img, structure=np.ones((3, 3)))
                    binary_img = binary_closing(binary_img, structure=np.ones((5, 5)))
                    binary_img = binary_fill_holes(binary_img)

                else:
                    # Rice (기본)
                    open_structure = np.ones((3, 3))
                    binary_img = binary_opening(binary_img, structure=open_structure)
                    close_structure = np.ones((5, 5))
                    binary_img = binary_closing(binary_img, structure=close_structure)
                    binary_img = binary_fill_holes(binary_img)

                binary_img = binary_img.astype('uint8')

                # 벡터화 및 저장
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

                # 단순화
                gdf['geometry'] = gdf.geometry.simplify(0.3)

                return gdf

        except Exception as e:
            print(f"    [Error] 바운더리 생성 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None