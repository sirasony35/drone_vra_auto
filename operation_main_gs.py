# find_best_sigma.py (기존 operation_main.py와 같은 폴더에 두고 실행)
import pandas as pd
import numpy as np
from operation_main import (
    load_boundary_from_zip, create_rotated_grid_with_indices,
    calculate_grid_mean_stats, apply_zone_detail_smoothing,
    BOUNDARY_ZIP_PATH, GNDVI_PATH, GRID_SIZE, N_ZONES
)


def find_best_sigma():
    print(">>> 최적의 Sigma 값 탐색 시작...")

    # 1. 데이터 로드 및 전처리
    boundary = load_boundary_from_zip(BOUNDARY_ZIP_PATH)
    if boundary.crs.is_geographic: boundary = boundary.to_crs(epsg=5179)
    grid = create_rotated_grid_with_indices(boundary, grid_size=GRID_SIZE)
    grid = calculate_grid_mean_stats(grid, GNDVI_PATH, col_name='Raw_GNDVI')
    raw_valid = grid.dropna(subset=['Raw_GNDVI'])

    # 2. Raw Data 기준 Quantile 계산
    _, bins = pd.qcut(raw_valid['Raw_GNDVI'], q=N_ZONES, retbins=True, duplicates='drop')
    if len(bins) < N_ZONES + 1:
        _, bins = pd.qcut(raw_valid['Raw_GNDVI'].rank(method='first'), q=N_ZONES, retbins=True)

    # 3. Sigma 테스트 (0.5 ~ 1.5 범위)
    test_sigmas = [0,0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0,6.0,7.0, 7.5, 9.0]

    print(f"\n{'Sigma':<6} | {'Zone 1 (%)':<10} | {'Target (8.58%)':<15} | {'Diff':<10}")
    print("-" * 50)

    for sigma in test_sigmas:
        # 스무딩 적용
        temp_grid = grid.copy()
        temp_grid = apply_zone_detail_smoothing(temp_grid, 'Raw_GNDVI', sigma=sigma)

        # 등급 부여
        temp_grid['Zone'] = pd.cut(temp_grid['Smooth_Value'], bins=bins, labels=[1, 2, 3, 4, 5], include_lowest=True)

        # 1등급 비율 계산
        total = len(temp_grid)
        z1_pct = (len(temp_grid[temp_grid['Zone'] == 1]) / total) * 100

        diff = abs(z1_pct - 8.58)
        print(f"{sigma:<6.2f} | {z1_pct:<10.2f} | 8.58%           | {diff:<10.2f}")


if __name__ == "__main__":
    find_best_sigma()