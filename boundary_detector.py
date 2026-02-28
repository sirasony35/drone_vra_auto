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
                # 1. 다운샘플링 (동일)
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

                # 3. Otsu 임계값 (하한선)
                valid_pixels = data[candidate_mask]
                try:
                    otsu_thresh = threshold_otsu(valid_pixels)
                except:
                    otsu_thresh = np.mean(valid_pixels)

                if crop_type == 'soybean':
                    thresh_factor = 0.90  # 콩은 기준을 낮춰서 일단 넓게 잡음 (나중에 잡초 자름)
                elif crop_type == 'wheat':
                    thresh_factor = 0.95
                else:
                    thresh_factor = 0.90

                final_thresh = otsu_thresh * thresh_factor
                if final_thresh < SAFE_FLOOR: final_thresh = SAFE_FLOOR
                print(f"    - Threshold (Lower): {final_thresh:.4f}")

                # 4. 초기 이진화 (전체 식생 영역)
                binary_img = (data > final_thresh)

                # 내부 구멍 메우기 (콩을 위해 미리 채움)
                if crop_type == 'soybean':
                    binary_img = binary_fill_holes(binary_img)

                # ---------------------------------------------------------
                # [NEW] 콩 전용: 가장자리 잡초 제거 (Core vs Edge 통계 비교)
                # ---------------------------------------------------------
                if crop_type == 'soybean':
                    print("    [Soybean Logic] Analyzing Core vs Edge statistics to remove weeds...")

                    # (1) Core 영역 추출 (깊게 침식)
                    # iter=8 (다운샘플링 고려하면 실제 3~4m 안쪽)
                    core_mask = binary_erosion(binary_img, structure=np.ones((3, 3)), iterations=8)

                    if np.sum(core_mask) > 0:
                        # (2) Core 통계 계산 (순수 작물 영역)
                        core_pixels = data[core_mask]
                        core_mean = np.mean(core_pixels)
                        core_std = np.std(core_pixels)

                        # (3) 잡초 판별 기준: Core 평균보다 (표준편차 x 1.5) 이상 높으면 잡초
                        weed_threshold = core_mean + (core_std * 1.5)
                        print(f"      - Core Mean: {core_mean:.4f}, Weed Threshold: {weed_threshold:.4f}")

                        # (4) 가장자리 영역 (Edge)만 타겟팅
                        # Edge = 전체 - Core
                        edge_mask = binary_img & (~core_mask)

                        # (5) Edge 중에서 Weed Threshold보다 높은 픽셀 제거
                        # 잡초 조건: (Edge 영역임) AND (값이 Weed Threshold 보다 큼)
                        is_weed = edge_mask & (data > weed_threshold)
                        weed_count = np.sum(is_weed)

                        if weed_count > 0:
                            print(f"      - Detected {weed_count} weed pixels at edge. Removing...")
                            # 잡초 픽셀을 False(0)로 변경
                            binary_img[is_weed] = 0
                    else:
                        print("      - Field too small for core analysis. Skipping weed removal.")

                # ---------------------------------------------------------
                # 형태학적 후처리 (Morphology)
                # ---------------------------------------------------------
                if crop_type == 'soybean':
                    # 잡초가 빠지면서 생긴 자잘한 노이즈 정리
                    binary_img = binary_opening(binary_img, structure=np.ones((2, 2)))

                    # 가장 큰 덩어리만 유지
                    labeled_img, num_features = label(binary_img, return_num=True, connectivity=2)
                    if num_features > 0:
                        sizes = np.bincount(labeled_img.ravel())
                        sizes[0] = 0
                        max_label = sizes.argmax()
                        binary_img = (labeled_img == max_label)

                    # 다시 채우기 (내부 콩 보호)
                    binary_img = binary_fill_holes(binary_img)

                    # 살짝 다듬기 (침식 1회)
                    binary_img = binary_erosion(binary_img, structure=np.ones((3, 3)), iterations=1)

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

                gdf['geometry'] = gdf.geometry.simplify(0.3)

                return gdf

        except Exception as e:
            print(f"    [Error] 바운더리 생성 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None