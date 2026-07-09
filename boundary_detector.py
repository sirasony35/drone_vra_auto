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
            print(f"    [Info] 바운더리 로드 성공: {os.path.basename(shp_path)}")
            return gdf
        except Exception as e:
            print(f"    [Error] SHP 파일 로드 실패: {e}")
            return None

    def detect_boundary_footprint(self, tif_path, simplify_tol=None, min_area_ratio=0.02):
        """유효 데이터(non-nodata) 풋프린트로 필지 경계를 추출.

        드론 정사영상이 이미 필지 모양대로 잘려(바깥이 NaN/nodata) 나온 경우,
        '값이 있는 영역의 외곽선 = 필지 경계'이므로 식생 임계값(Otsu) 없이
        데이터 존재 영역을 그대로 벡터화한다. 작물/생육시기와 무관하게 안정적.

        simplify_tol: 경계 단순화 허용오차(좌표계 단위). None이면 픽셀크기 기반 자동.
        min_area_ratio: 전체 유효면적 대비 이 비율 미만의 조각 폴리곤은 버림.
        """
        print("    [Info] 바운더리 감지 시작 (Type: footprint / 유효 데이터 외곽)...")
        try:
            with rasterio.open(tif_path) as src:
                decimation = 4 if (src.width > 2000 or src.height > 2000) else 1
                out_shape = (int(src.height / decimation), int(src.width / decimation))
                data = src.read(1, out_shape=out_shape).astype('float32')
                transform = src.transform * src.transform.scale(
                    (src.width / out_shape[-1]), (src.height / out_shape[-2])
                )

                # 유효 마스크: NaN 아님 + nodata 아님
                valid = ~np.isnan(data)
                if src.nodata is not None and not np.isnan(src.nodata):
                    valid &= (data != src.nodata)

                if np.sum(valid) == 0:
                    print("    [Warning] 유효한 데이터 픽셀이 없습니다.")
                    return None

                # 내부 구멍(물/그림자 등) 메우고 테두리 정리 후 최대 연결영역 선택
                valid = binary_fill_holes(valid)
                valid = binary_closing(valid, structure=np.ones((7, 7)))
                valid = binary_opening(valid, structure=np.ones((5, 5)))

                labeled, num = label(valid, return_num=True, connectivity=2)
                if num == 0:
                    return None
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0
                valid = (labeled == sizes.argmax())

                binary_img = valid.astype('uint8')
                shapes_gen = shapes(binary_img, mask=(binary_img == 1), transform=transform)
                polygons = [shape(geom) for geom, val in shapes_gen if val == 1]
                if not polygons:
                    return None

                gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=src.crs)
                gdf['area'] = gdf.geometry.area
                total_area = gdf['area'].sum()
                gdf = gdf[gdf['area'] >= total_area * min_area_ratio]
                gdf = gdf.sort_values('area', ascending=False).iloc[0:1]

                if simplify_tol is None:
                    px = abs(transform.a)
                    simplify_tol = px * 1.5  # 픽셀 스케일의 완만한 단순화
                gdf['geometry'] = gdf.geometry.simplify(simplify_tol)

                print(f"    - Footprint 경계 추출 완료 (단순화 tol={simplify_tol:.3g})")
                return gdf[['geometry']].reset_index(drop=True)

        except Exception as e:
            print(f"    [Error] Footprint 바운더리 생성 중 오류: {e}")
            import traceback
            traceback.print_exc()
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