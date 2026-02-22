import pandas as pd
import os


class VRACalculator:
    def __init__(self, vra_csv_path):
        self.vra_data = self._load_vra_data(vra_csv_path)

    def _load_vra_data(self, path):
        try:
            df = pd.read_csv(path)
            df['field'] = df['field'].astype(str)
            return df.set_index('field')
        except Exception as e:
            print(f"    [Error] VRA 데이터 로드 실패: {e}")
            return None

    def calculate_prescription(self, field_code, zone_stats):
        """
        zone_stats: List of dicts
        """
        if self.vra_data is None or field_code not in self.vra_data.index:
            print(f"    [Warning] '{field_code}'에 대한 VRA 설정값을 찾을 수 없습니다.")
            return None

        field_info = self.vra_data.loc[field_code]
        total_amount_kg = float(field_info['total'])
        spread = float(field_info['spread'])

        # 1. Zone 6(Skip)를 제외한 '실제 살포 면적' 계산
        sprayable_zones = [z for z in zone_stats if z['Zone'] != 6]

        total_area_m2 = sum(z['Area_m2'] for z in sprayable_zones)
        total_area_ha = total_area_m2 / 10000.0

        if total_area_ha == 0:
            print("    [Warning] 살포 가능한 면적이 0입니다.")
            return None

        # 가중 평균 (Zone 6 제외)
        weighted_sum_gndvi = sum(z['Mean_GNDVI'] * z['Area_m2'] for z in sprayable_zones)
        field_avg_gndvi = weighted_sum_gndvi / total_area_m2

        # 평균 시비량 (Flat Rate) -> 살포 가능 면적 기준
        flat_rate = total_amount_kg / total_area_ha

        results = []
        zone_labels = {
            1: "빨강(High)", 2: "주황", 3: "노랑", 4: "연두", 5: "초록(Low)",
            6: "회색(Skip)"
        }

        for z in zone_stats:
            zone_idx = z['Zone']
            gndvi = z['Mean_GNDVI']
            area_ha = z['Area_m2'] / 10000.0

            # [핵심] Zone 6는 무조건 0kg
            if zone_idx == 6:
                rate_kg_ha = 0
                zone_total_kg = 0
            else:
                # Zone 1~5는 정상 계산
                if field_avg_gndvi > 0:
                    rate_kg_ha = flat_rate * (1 - ((gndvi - field_avg_gndvi) / field_avg_gndvi) * spread)
                else:
                    rate_kg_ha = 0

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