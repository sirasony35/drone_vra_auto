import pandas as pd
import os


class VRACalculator:
    def __init__(self, vra_csv_path):
        self.vra_data = self._load_vra_data(vra_csv_path)

    def _load_vra_data(self, path):
        try:
            df = pd.read_csv(path)
            # field 컬럼을 인덱스로 하여 검색 속도 향상
            df['field'] = df['field'].astype(str)
            return df.set_index('field')
        except Exception as e:
            print(f"    [Error] VRA 데이터 로드 실패: {e}")
            return None

    def calculate_prescription(self, field_code, zone_stats):
        """
        zone_stats: List of dicts [{'Zone': 1, 'Area_m2': ..., 'Mean_GNDVI': ...}, ...]
        Returns: DataFrame with prescription
        """
        if self.vra_data is None or field_code not in self.vra_data.index:
            print(f"    [Warning] '{field_code}'에 대한 VRA 설정값(vra.csv)을 찾을 수 없습니다.")
            return None

        # 1. 설정값 가져오기
        field_info = self.vra_data.loc[field_code]
        total_amount_kg = float(field_info['total'])
        spread = float(field_info['spread'])

        # 2. 전체 통계 계산
        total_area_m2 = sum(z['Area_m2'] for z in zone_stats)
        total_area_ha = total_area_m2 / 10000.0

        if total_area_ha == 0:
            return None

        # 가중 평균 GNDVI 계산
        weighted_sum_gndvi = sum(z['Mean_GNDVI'] * z['Area_m2'] for z in zone_stats)
        field_avg_gndvi = weighted_sum_gndvi / total_area_m2

        # 평균 시비량 (Flat Rate)
        flat_rate = total_amount_kg / total_area_ha

        # 3. Zone별 시비량 계산
        results = []
        zone_labels = {1: "빨강", 2: "주황", 3: "노랑", 4: "연두", 5: "초록"}

        for z in zone_stats:
            zone_idx = z['Zone']
            gndvi = z['Mean_GNDVI']
            area_ha = z['Area_m2'] / 10000.0

            # VRA 공식 적용
            if field_avg_gndvi > 0:
                # (평균 - 개별) / 평균 * Spread -> 낮을수록 양수(더 주기)
                # 공식: Rate = Flat * (1 + (Avg - Val)/Avg * Spread)
                # 또는 HTML 공식: Rate = Flat - (Flat * (Val - Avg)/Avg * Spread) -> 동일함
                rate_kg_ha = flat_rate * (1 - ((gndvi - field_avg_gndvi) / field_avg_gndvi) * spread)
            else:
                rate_kg_ha = 0

            # 음수 방지 (최소 0)
            rate_kg_ha = max(rate_kg_ha, 0)

            zone_total_kg = rate_kg_ha * area_ha

            results.append({
                'Field': field_code,
                'Zone': f"{zone_idx}({zone_labels.get(zone_idx, '')})",
                'GNDVI': round(gndvi, 4),
                'Area(ha)': round(area_ha, 4),
                'Rate(kg/ha)': round(rate_kg_ha, 2),
                'Total(kg)': round(zone_total_kg, 2)
            })

        return pd.DataFrame(results)