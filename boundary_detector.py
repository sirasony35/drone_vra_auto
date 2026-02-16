import os
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from skimage.filters import threshold_otsu
from shapely.geometry import shape
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
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
            return gdf
        except Exception as e:
            print(f"    [Error] 바운더리 로드 실패: {e}")
            return None

    def detect_boundary_otsu(self, tif_path):
        """
        [V42 개선] 상대적 분포 기반 바운더리 감지 (Dynamic Threshold)
        - 고정된 0.25 기준 삭제 -> 필지별 데이터 분포에 따라 자동 결정
        - Safe Floor (0.05): 최소한의 노이즈(물/진흙)만 제거
        - Otsu Relaxation: 계산된 임계값을 살짝 낮춰서 약한 작물도 포함
        """
        print(f"    [Info] 필지 맞춤형 바운더리 감지 시작 (Dynamic Otsu)...")

        try:
            with rasterio.open(tif_path) as src:
                # 1. 다운샘플링 (속도 최적화)
                decimation = 4 if (src.width > 2000 or src.height > 2000) else 1
                out_shape = (src.count, int(src.height / decimation), int(src.width / decimation))
                data = src.read(1, out_shape=out_shape)

                # 변환 행렬 조정
                transform = src.transform * src.transform.scale(
                    (src.width / out_shape[-1]), (src.height / out_shape[-2])
                )

                # 2. 유효 데이터 및 최소 안전장치 (Safe Floor)
                # GNDVI -1 ~ 1 범위. 0.05 이하는 확실한 비식생(물, 진흙, 도로)으로 간주
                # 하지만 생육이 아주 안좋은 필지를 대비해 아주 낮게 설정
                SAFE_FLOOR = 0.05

                valid_mask = ~np.isnan(data)
                if src.nodata is not None:
                    valid_mask &= (data != src.nodata)

                # 최소한의 노이즈만 제거한 데이터 준비
                candidate_mask = valid_mask & (data > SAFE_FLOOR)

                if np.sum(candidate_mask) == 0:
                    print("    [Warning] 유효한 식생 데이터가 거의 없습니다.")
                    return None

                # 3. Otsu 알고리즘 (데이터 분포에 따른 동적 임계값)
                valid_pixels = data[candidate_mask]
                try:
                    otsu_thresh = threshold_otsu(valid_pixels)
                except:
                    otsu_thresh = np.mean(valid_pixels)  # 실패시 평균 사용

                # [핵심] 임계값 완화 (Relaxation)
                # Otsu가 칼같이 자르면 경계면의 약한 작물이 잘릴 수 있음.
                # 계산된 값의 90% 수준(혹은 -0.05)으로 낮춰서 안전하게 포함시킴
                final_thresh = otsu_thresh * 0.9

                # 너무 낮아지는 것 방지 (Safe Floor 보다는 커야 함)
                if final_thresh < SAFE_FLOOR:
                    final_thresh = SAFE_FLOOR

                print(f"    - 감지된 GNDVI 임계값: {final_thresh:.4f} (Otsu: {otsu_thresh:.4f})")

                # 이진화
                binary_img = (data > final_thresh)

                # 4. 모폴로지 연산 (다듬기)
                # Opening: 아주 작은 점(노이즈) 제거
                open_structure = np.ones((3, 3))
                binary_img = binary_opening(binary_img, structure=open_structure)

                # Closing: 작물 사이 틈새 메우기 (연결성 확보)
                # 작물이 듬성듬성 심겨진 경우를 대비해 커널을 적절히 유지
                close_structure = np.ones((5, 5))
                binary_img = binary_closing(binary_img, structure=close_structure)

                # Fill Holes: 필지 내부의 구멍 메우기
                binary_img = binary_fill_holes(binary_img)
                binary_img = binary_img.astype('uint8')

                # 5. 벡터화 (Polygon 변환)
                shapes_gen = shapes(binary_img, mask=(binary_img == 1), transform=transform)
                polygons = []
                for geom, val in shapes_gen:
                    if val == 1:
                        polygons.append(shape(geom))

                if not polygons:
                    return None

                # 6. 메인 필지 추출
                gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=src.crs)

                if len(gdf) > 1:
                    gdf['area'] = gdf.geometry.area
                    # 가장 큰 덩어리 1개만 선택 (메인 밭)
                    gdf = gdf.sort_values('area', ascending=False).iloc[0:1]

                # 7. 단순화 (Simplify)
                # 외곽선을 부드럽게 정리
                gdf['geometry'] = gdf.geometry.simplify(0.3)

                return gdf

        except Exception as e:
            print(f"    [Error] 바운더리 생성 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None