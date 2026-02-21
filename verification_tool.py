import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings

# 경고 무시
warnings.filterwarnings("ignore")

# ======================================================
# 1. 설정 (폴더 경로)
# ======================================================
# Python 코드로 생성된 Rx 폴더
GEN_FOLDER = "result_final_dji_wgs84/DJI/Rx"

# Pix4D로 생성된 Rx 폴더 (사용자 환경에 맞게 수정하세요)
PIX_FOLDER = "pix4d_data/DJI/Rx"

# 결과 리포트 이미지가 저장될 폴더
REPORT_FOLDER = "verification_reports"


# ======================================================
# 2. 비교 분석 클래스 (Core Logic)
# ======================================================
class MapComparator:
    def __init__(self, generated_path, pix4d_path, field_code):
        self.gen_path = generated_path
        self.pix_path = pix4d_path
        self.field_code = field_code

        self.gen_data_aligned = None
        self.pix_data_aligned = None
        self.common_mask = None

        self.mean_gen = 0
        self.mean_pix = 0
        self.mae = 0
        self.corr = 0
        self.acc = 0
        self.kappa = 0

    def read_raw_data(self, path):
        """시각화용 원본 데이터 읽기"""
        with rasterio.open(path) as src:
            data = src.read(1)
            if src.nodata is not None:
                mask = (data != src.nodata) & (data > 0)
            else:
                mask = (data > 0)
            return data, mask

    def align_rasters_for_calc(self):
        """통계용 데이터 정렬 및 공통 마스크 생성"""
        try:
            # 1. Target (Pix4D) 읽기
            with rasterio.open(self.pix_path) as dst:
                self.pix_data_aligned = dst.read(1)
                dst_transform = dst.transform
                dst_crs = dst.crs
                dst_height = dst.height
                dst_width = dst.width

                mask_pix = (self.pix_data_aligned > 0)
                if dst.nodata is not None:
                    mask_pix &= (self.pix_data_aligned != dst.nodata)

            # 2. Source (Python) 읽기 및 재투영
            with rasterio.open(self.gen_path) as src:
                self.gen_data_aligned = np.zeros((dst_height, dst_width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=self.gen_data_aligned,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
                mask_gen = (self.gen_data_aligned > 0)

            # 3. 공통 마스크 (교집합)
            self.common_mask = mask_pix & mask_gen

            # 유효 데이터 확인
            if np.sum(self.common_mask) == 0:
                print(f"    [Skip] {self.field_code}: 겹치는 유효 영역이 없습니다.")
                return False

            return True
        except Exception as e:
            print(f"    [Error] 정렬 중 오류 발생: {e}")
            return False

    def compare_rates(self):
        val_gen = self.gen_data_aligned[self.common_mask]
        val_pix = self.pix_data_aligned[self.common_mask]

        self.mean_gen = np.mean(val_gen)
        self.mean_pix = np.mean(val_pix)
        diff = val_gen - val_pix
        self.mae = np.mean(np.abs(diff))

        if len(val_gen) > 1:
            self.corr = np.corrcoef(val_gen, val_pix)[0, 1]

    def compare_zones_by_rank(self):
        val_gen = self.gen_data_aligned[self.common_mask]
        val_pix = self.pix_data_aligned[self.common_mask]

        def rate_to_zone_rank(data_array):
            if len(data_array) == 0: return np.array([])
            s = pd.Series(data_array)
            ranks = s.rank(method='dense', ascending=False).astype(int)
            ranks[ranks > 5] = 5
            return ranks

        zone_gen = rate_to_zone_rank(val_gen)
        zone_pix = rate_to_zone_rank(val_pix)

        if len(zone_gen) > 0:
            self.acc = accuracy_score(zone_pix, zone_gen) * 100
            self.kappa = cohen_kappa_score(zone_pix, zone_gen)

    def visualize_comparison(self, output_folder):
        # 이미지 저장 경로
        save_path = os.path.join(output_folder, f"Report_{self.field_code}.png")

        fig = plt.figure(figsize=(18, 9))
        gs = fig.add_gridspec(2, 3, height_ratios=[5, 1.2])

        # 커스텀 컬러맵
        colors = ['red', 'orange', 'yellow', '#90EE90', 'green']
        cmap_zone = ListedColormap(colors)
        cmap_zone.set_under('white')

        def rate_to_zone_2d(data_2d, mask):
            valid_data = data_2d[mask]
            if len(valid_data) == 0: return np.zeros_like(data_2d, dtype=int)
            s = pd.Series(valid_data)
            ranks = s.rank(method='dense', ascending=False).astype(int)
            ranks[ranks > 5] = 5
            final_zones = np.zeros_like(data_2d, dtype=int)
            final_zones[mask] = ranks.values
            return final_zones

        # 1. Python Map (Raw)
        ax1 = fig.add_subplot(gs[0, 0])
        raw_gen, raw_gen_mask = self.read_raw_data(self.gen_path)
        zone_gen_2d = rate_to_zone_2d(raw_gen, raw_gen_mask)
        im1 = ax1.imshow(zone_gen_2d, cmap=cmap_zone, vmin=0.5, vmax=5.5, interpolation='none')
        ax1.set_title(f"Python Result ({self.field_code})")
        ax1.axis('off')

        # 2. Pix4D Map (Raw)
        ax2 = fig.add_subplot(gs[0, 1])
        raw_pix, raw_pix_mask = self.read_raw_data(self.pix_path)
        zone_pix_2d = rate_to_zone_2d(raw_pix, raw_pix_mask)
        im2 = ax2.imshow(zone_pix_2d, cmap=cmap_zone, vmin=0.5, vmax=5.5, interpolation='none')
        ax2.set_title(f"Pix4D Result ({self.field_code})")
        ax2.axis('off')

        # 3. Diff Map (Aligned)
        ax3 = fig.add_subplot(gs[0, 2])
        diff_map = self.gen_data_aligned - self.pix_data_aligned
        diff_map[~self.common_mask] = np.nan
        limit = np.nanmax(np.abs(diff_map)) if np.any(~np.isnan(diff_map)) else 1
        im3 = ax3.imshow(diff_map, cmap='bwr', vmin=-limit, vmax=limit, interpolation='none')
        ax3.set_title("Difference Map (Intersection)")
        ax3.axis('off')

        # 컬러바
        cbar_ticks = [1, 2, 3, 4, 5]
        cbar_labels = ['1(High)', '2', '3', '4', '5(Low)']
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, ticks=cbar_ticks).set_ticklabels(cbar_labels)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, ticks=cbar_ticks).set_ticklabels(cbar_labels)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Diff (kg/ha)")

        # 통계 텍스트
        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis('off')
        stats_text = (
            f"Verification Report: {self.field_code}\n"
            f"----------------------------------------------------------------------------------\n"
            f"1. Rate Comparison (kg/ha)\n"
            f"   - Mean Rate: Python {self.mean_gen:.2f} vs Pix4D {self.mean_pix:.2f}  |  Diff: {self.mean_gen - self.mean_pix:.2f}\n"
            f"   - MAE (Avg Error): {self.mae:.2f} kg/ha  |  Correlation: {self.corr:.4f}\n\n"
            f"2. Zone Classification Similarity\n"
            f"   - Zone Match Accuracy: {self.acc:.2f}%  |  Kappa Score: {self.kappa:.4f}"
        )
        ax_text.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
                     family='monospace', bbox=dict(boxstyle="round,pad=1", fc="#f0f0f0", ec="gray"))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"    -> Saved Report: {save_path}")


# ======================================================
# 3. 메인 실행 (Batch Process)
# ======================================================
def find_and_pair_files(gen_folder, pix_folder):
    """
    두 폴더를 스캔하여 필지코드(파일명 맨 앞)가 같은 파일끼리 짝을 지어줍니다.
    """
    # 1. Python 결과 파일 스캔 (예: GJR1.tif)
    gen_files = {}
    for f in glob.glob(os.path.join(gen_folder, "*.tif")):
        filename = os.path.basename(f)
        # GJR1.tif -> GJR1
        field_code = filename.split('.')[0].split('_')[0]
        gen_files[field_code] = f

    # 2. Pix4D 결과 파일 스캔 (예: GJR1_Pix4D.tif)
    pix_files = {}
    for f in glob.glob(os.path.join(pix_folder, "*.tif")):
        filename = os.path.basename(f)
        # GJR1_Pix4D.tif -> GJR1
        field_code = filename.split('.')[0].split('_')[0]
        pix_files[field_code] = f

    # 3. 교집합 찾기
    common_fields = set(gen_files.keys()) & set(pix_files.keys())

    pairs = []
    for field in sorted(common_fields):
        pairs.append((field, gen_files[field], pix_files[field]))

    return pairs


def main():
    # 저장 폴더 생성
    if not os.path.exists(REPORT_FOLDER):
        os.makedirs(REPORT_FOLDER)

    print(">>> 검증 도구 (일괄 처리 모드) 시작")
    print(f" - Python 폴더: {GEN_FOLDER}")
    print(f" - Pix4D 폴더:  {PIX_FOLDER}")

    # 파일 짝짓기
    pairs = find_and_pair_files(GEN_FOLDER, PIX_FOLDER)

    if not pairs:
        print("\n[Warning] 짝이 맞는 파일(필지코드 동일)을 찾을 수 없습니다.")
        return

    print(f"\n>>> 총 {len(pairs)}개의 필지 쌍을 발견했습니다.")

    # 순차 처리
    for i, (field, gen_path, pix_path) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Processing Field: {field} ...")

        comp = MapComparator(gen_path, pix_path, field)

        # 1. 정렬 및 공통 마스크 생성
        if comp.align_rasters_for_calc():
            # 2. 통계 계산
            comp.compare_rates()
            comp.compare_zones_by_rank()

            # 3. 리포트 이미지 저장
            comp.visualize_comparison(REPORT_FOLDER)
        else:
            print(f"    -> Skipped (정렬 실패)")

    print(f"\n>>> 모든 작업 완료. '{REPORT_FOLDER}' 폴더를 확인하세요.")


if __name__ == "__main__":
    main()