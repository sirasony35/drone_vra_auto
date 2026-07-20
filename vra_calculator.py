import pandas as pd
import os


class VRACalculator:
    def __init__(self, vra_csv_path):
        self.vra_data = self._load_vra_data(vra_csv_path)

    def _load_vra_data(self, path):
        # 한국 엑셀 'CSV로 저장'은 기본이 cp949(euc-kr)이고, 웹/UTF-8 저장은 utf-8/utf-8-BOM.
        # 어느 쪽이든 읽히도록 인코딩을 순서대로 시도한다. (utf-8류를 먼저 → 잘못된 바이트면
        # 예외 발생해 다음으로 넘어가고, cp949는 대부분 바이트를 무오류로 디코드하므로 마지막)
        encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
        df = None
        last_err = None
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc)
                if enc not in ("utf-8-sig", "utf-8"):
                    print(f"    [Info] VRA CSV 인코딩 자동 감지: {enc}")
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                last_err = e
                continue
            except Exception as e:
                # 인코딩 외 오류(형식 등)는 즉시 보고
                print(f"    [Error] VRA 데이터 로드 실패: {e}")
                return None
        if df is None:
            print(f"    [Error] VRA 데이터 로드 실패(인코딩 인식 불가): {last_err}")
            return None

        try:
            # 컬럼명·필지코드 앞뒤 공백 제거 (엑셀/수기 편집 편차 방어)
            df.columns = df.columns.str.strip()
            if 'field' not in df.columns:
                print(f"    [Error] VRA CSV에 'field' 컬럼이 없습니다. 컬럼: {list(df.columns)}")
                return None
            df['field'] = df['field'].astype(str).str.strip()
            return df.set_index('field')
        except Exception as e:
            print(f"    [Error] VRA 데이터 전처리 실패: {e}")
            return None

    def get_field_info(self, field_code):
        """필지의 모든 설정 정보(행)를 반환"""
        if self.vra_data is None or field_code not in self.vra_data.index:
            return None
        return self.vra_data.loc[field_code]

    # 면적 단위 → m² 환산 계수 (1평 = 400/121 m²)
    _UNIT_TO_M2 = {
        '평': 3.3057851, 'pyeong': 3.3057851, 'py': 3.3057851, 'p': 3.3057851,
        'ha': 10000.0, '헥타르': 10000.0, 'hectare': 10000.0,
        'm2': 1.0, 'm²': 1.0, 'sqm': 1.0, '㎡': 1.0,
    }

    @staticmethod
    def _num(field_info, key):
        """설정 행에서 숫자값을 안전하게 추출 (컬럼 없음/빈칸/NaN → None)."""
        try:
            if key not in field_info.index:
                return None
            v = field_info[key]
            if pd.isna(v):
                return None
            return float(v)
        except (KeyError, TypeError, ValueError):
            return None

    def _resolve_total_kg(self, field_code, field_info, zone_stats, field_area_m2=None):
        """농가별 비료 기준에 따라 목표 총량(kg)을 산출.

        - 면적 비율 모드: `rate_kg`, `rate_area`가 있으면
            total = rate_kg × (필지면적 ÷ rate_area)  (같은 area_unit 기준)
            필지면적 = `field_area`(직접 입력) 우선, 없으면 실측 면적 자동.
        - 절대량 모드: 위 컬럼이 없으면 기존 `total`(kg) 그대로.
        """
        rate_kg = self._num(field_info, 'rate_kg')
        rate_area = self._num(field_info, 'rate_area')

        if rate_kg is not None and rate_area is not None and rate_area > 0:
            # 단위 파악 (기본 평)
            unit_raw = '평'
            if 'area_unit' in field_info.index and not pd.isna(field_info.get('area_unit')):
                unit_raw = str(field_info['area_unit']).strip()
            m2_per_unit = self._UNIT_TO_M2.get(unit_raw, self._UNIT_TO_M2.get(unit_raw.lower(), 3.3057851))

            # 필지면적: 직접 입력(field_area) 우선, 없으면 실측(경계면적→격자합 순)
            entered = self._num(field_info, 'field_area')
            if entered is not None and entered > 0:
                area_in_unit = entered
                src = "입력"
            else:
                if field_area_m2 is None or field_area_m2 <= 0:
                    field_area_m2 = sum(z['Area_m2'] for z in zone_stats)
                area_in_unit = field_area_m2 / m2_per_unit
                src = "실측"

            total = rate_kg * (area_in_unit / rate_area)
            print(f"    [비율계산] {rate_kg:g}kg / {rate_area:g}{unit_raw} × 필지 {area_in_unit:.1f}{unit_raw}({src}) "
                  f"= 목표 총량 {total:.2f}kg")
            return total

        # 절대량 모드
        total = self._num(field_info, 'total')
        if total is None:
            print(f"    [Warning] '{field_code}': total(절대량)도 rate_kg/rate_area(비율)도 없어 총량을 정할 수 없습니다.")
            return None
        return total

    def calculate_prescription(self, field_code, zone_stats, field_area_m2=None):
        """
        zone_stats: List of dicts
        field_area_m2: 실측 필지 전체 면적(m², 경계 기준). 비율 모드 자동 면적에 사용.
        """
        if self.vra_data is None or field_code not in self.vra_data.index:
            print(f"    [Warning] '{field_code}'에 대한 VRA 설정값을 찾을 수 없습니다.")
            return None

        field_info = self.vra_data.loc[field_code]
        total_amount_kg = self._resolve_total_kg(field_code, field_info, zone_stats, field_area_m2)
        if total_amount_kg is None:
            return None
        spread_val = self._num(field_info, 'spread')
        spread = spread_val if spread_val is not None else 1.0

        # 기체 타입 인식 (csv에 drone_type 열이 없다면 DJI로 기본 인식)
        drone_type = str(field_info.get('drone_type', 'DJI')).strip().upper() if 'drone_type' in field_info else 'DJI'

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

        # 기체 타입별 CSV 라벨 동적 할당
        if drone_type == 'XAG':
            zone_labels = {
                1: "빨강(High)", 2: "노랑(Medium)", 3: "초록(Low)",
                6: "회색(Skip)"
            }
        else:
            zone_labels = {
                1: "빨강(High)", 2: "주황", 3: "노랑", 4: "연두", 5: "초록(Low)",
                6: "회색(Skip)"
            }

        # ---------------------------------------------------------
        # [STEP 1] 1차 계산: 최소 살포량(Base Rate) 보장 적용된 1차 시비량 계산
        # ---------------------------------------------------------
        # 생육이 아무리 좋아도 최소 50kg/ha 이상은 주도록 하한선 설정
        MIN_RATE_KG_HA = 50.0

        temp_zones = []
        preliminary_total_kg = 0.0  # <--- 아까 에러가 났던, 꼭 필요한 변수 초기화 부분입니다!

        for z in zone_stats:
            zone_idx = z['Zone']
            gndvi = z['Mean_GNDVI']
            area_ha = z['Area_m2'] / 10000.0

            if zone_idx == 6:
                rate_kg_ha = 0
                zone_total_kg = 0
            else:
                safe_denominator = max(abs(field_avg_gndvi), 0.1)
                rate_kg_ha = flat_rate * (1 - ((gndvi - field_avg_gndvi) / safe_denominator) * spread)

                # 0 대신 설정한 하한선(50kg/ha)으로 강제 끌어올림!
                rate_kg_ha = max(rate_kg_ha, MIN_RATE_KG_HA)
                zone_total_kg = rate_kg_ha * area_ha

            temp_zones.append({
                'zone_idx': zone_idx,
                'gndvi': gndvi,
                'area_ha': area_ha,
                'rate_kg_ha': rate_kg_ha,
                'zone_total_kg': zone_total_kg
            })
            preliminary_total_kg += zone_total_kg

        # ---------------------------------------------------------
        # [STEP 2] 2차 계산: 초과/미달분을 원래 목표 총량(Target)에 맞게 완벽 비율 보정
        # ---------------------------------------------------------
        correction_factor = 1.0
        if preliminary_total_kg > 0:
            correction_factor = total_amount_kg / preliminary_total_kg

        results = []
        for tz in temp_zones:
            zone_idx = tz['zone_idx']

            # 보정 계수를 곱하여 최종값 도출 (0인 곳은 곱해도 그대로 0 유지)
            final_rate_kg_ha = tz['rate_kg_ha'] * correction_factor
            final_total_kg = tz['zone_total_kg'] * correction_factor

            results.append({
                'Field': field_code,
                'Zone': f"{zone_idx}({zone_labels.get(zone_idx, '')})",
                'GNDVI': round(tz['gndvi'], 4),
                'Area(ha)': round(tz['area_ha'], 4),
                'Rate(kg/ha)': round(final_rate_kg_ha, 2),
                'Total(kg)': round(final_total_kg, 2)
            })

        return pd.DataFrame(results)