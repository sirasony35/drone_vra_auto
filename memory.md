# 프로젝트 개요

`drone_vra_auto`는 드론으로 촬영한 농경지의 식생지수(GNDVI) 정사영상(GeoTIFF)을 입력받아, 작물 생육 상태에 따라 농경지를 등급(Zone)으로 구분하고 **가변 시비/살포(VRA, Variable Rate Application) 처방도**를 자동으로 생성하는 파이썬 자동화 프로그램이다. 기존에 Pix4D 같은 상용 소프트웨어로 수작업하던 작업을 코드로 자동화하는 것이 목적이며, DJI 농업용 드론과 XAG(엑스에이지) 드론 두 기종에 맞는 처방 파일을 각각 출력한다.

주요 처리 단계는 다음과 같다.
1. GNDVI 영상에서 필지(농경지) 경계(Boundary) 추출
2. 경계 내부를 회전 격자(grid)로 분할하고 각 격자의 평균 GNDVI 계산
3. GNDVI 값을 분위수(quantile) 기준으로 등급화(Zone) 및 스무딩 처리
4. 등급별 면적·평균 GNDVI를 바탕으로 VRA 살포량(kg/ha) 계산
5. DJI용(GeoTIFF Rx + ShapeFile) 또는 XAG용(JSON + KML) 처방 파일 및 결과 지도 이미지 출력
6. (검증) 파이썬 결과물과 Pix4D 결과물을 비교하는 별도 도구 제공

본 폴더에는 동작 비교/검토용 발표자료인 `pix4d_자동화 비교.pptx` 파일도 존재한다(코드와 무관한 참고 자료).

---

# 폴더/파일 구조

```
drone_vra_auto/
├── operation_main.py          # 메인 실행 스크립트 (전체 VRA 파이프라인)
├── boundary_detector.py       # 필지 경계 추출 모듈 (BoundaryDetector 클래스)
├── vra_calculator.py          # VRA 처방량 계산 모듈 (VRACalculator 클래스)
├── verification_tool.py       # 파이썬 결과 vs Pix4D 결과 비교/검증 도구
│
├── vra_setting/               # 필지별 VRA 설정 CSV (코드 없음, 데이터만 존재)
│   ├── vra.csv                # 광주/월(GJR…) 벼 필지 설정
│   ├── sm_vra.csv             # SM… 밀 필지 설정
│   └── bd_vra.csv             # BD… 밀 필지 설정
│
├── data/                      # 입력 데이터
│   ├── sm_data/               # SM 필지 GNDVI / RGB GeoTIFF 입력 영상
│   └── ShapeFile/             # SM01~SM24 필지 경계 ShapeFile(.shp/.dbf/.shx/.prj …)
│
├── result/                    # 출력 결과 (실행 산출물)
│   ├── sm_result_dji_0330/    # SM 필지 DJI 결과 (DJI/Rx, DJI/ShapeFile, *_VRA.csv, *_Result.png)
│   ├── sm_fields/             # SM 필지 결과 (DJI Rx/ShapeFile, DJI.zip)
│   └── pix4d_data/            # Pix4D로 생성한 비교용 결과 (검증 도구 입력)
│       ├── bd/ , gj/          # 필지군별 Pix4D Rx/ShapeFile 및 비교 이미지
│
├── verification_reports/      # verification_tool.py 출력 비교 리포트 (Report_*.png)
│
├── pix4d_자동화 비교.pptx      # 자동화 vs Pix4D 비교 발표자료 (참고용, 코드 무관)
│
├── __pycache__/               # 파이썬 캐시 (.pyc) — git 무시 권장
└── .idea/                     # PyCharm 프로젝트 설정 — git 무시 권장
```

> 참고: `operation_main.py`의 경로 상수(`DATA_FOLDER`, `OUTPUT_FOLDER`, `VRA_CSV_PATH` 등)는 실행하는 작업마다 코드 상단에서 직접 수정하여 사용한다. 현재 커밋된 값은 `data/sm_hm_test_data`, `result/result_0604`, `vra_setting/sm_hm_vra.csv`이며(2026-06-30 기준), 위 폴더 구조의 실제 데이터 폴더명과 다를 수 있으니 실행 전 경로를 맞춰야 한다.

---

# 주요 스크립트 상세

## operation_main.py
전체 VRA 처방 파이프라인을 실행하는 메인 스크립트.

- **목적**: 입력 폴더의 `*_GNDVI.tif` 영상들을 순회하며 필지별로 경계 추출 → 격자화 → 등급화 → VRA 계산 → 기체별 처방 파일 출력까지 일괄 처리.
- **상단 설정 상수**
  - `DATA_FOLDER`: GNDVI 입력 영상 폴더
  - `BOUNDARY_FOLDER`: 필지 경계(zip/shp) 폴더 (`data/ShapeFile`)
  - `OUTPUT_FOLDER`: 결과 출력 폴더
  - `VRA_CSV_PATH`: 필지별 VRA 설정 CSV 경로
  - `DEFAULT_GRID_SIZE=1.0`, `DEFAULT_CROP='rice'`, `VALID_THRESHOLD=-999.0`, `MAX_MASK_THRESHOLD=0.40`
- **주요 함수**
  - `get_main_angle(geometry)`: 필지의 최소 회전 사각형으로 주축 각도 계산 (격자 회전용).
  - `create_rotated_grid_with_indices(boundary_gdf, grid_size)`: 필지 주축에 맞춰 회전된 정사각형 격자 생성. 각 격자에 행/열 인덱스(`mat_col`, `mat_row`) 부여. **격자 면적의 95% 이상이 경계 내부에 들어와야** 살포 구역으로 포함한다(`overlap_ratio >= 0.95`). 과거에는 40% 기준이었으나, 경계 밖으로 삐져나가는 가장자리 격자를 완전히 제거해 비료 모자람 문제를 해결하기 위해 2026-06-30 pull에서 95%로 상향됨.
  - `clip_raster_to_boundary(raster_path, boundary_gdf)`: 래스터를 경계로 잘라 메모리 파일(`MemoryFile`)로 반환.
  - `calculate_grid_mean_stats(grid_gdf, mem_raster, col_name)`: 각 격자별 유효 픽셀 평균값 계산(예: `Raw_GNDVI`).
  - `calculate_optimal_sigma(...)`: 기체 타입(XAG=0.7, DJI=1.35 기준)과 데이터 표준편차로 스무딩 sigma 자동 산출.
  - `calculate_dynamic_threshold(grid_gdf, relax_factor)`: Otsu 임계값 기반으로 나지(맨땅, Zone 6) 마스킹 임계값 동적 계산.
  - `apply_categorical_zone_smoothing(...)`: 등급 행렬에 가우시안 + 최빈값(mode) 필터를 적용해 등급 경계를 매끄럽게 정리. 최종 `Zone` 컬럼 생성.
  - `save_map_image(...)`: 등급 지도(3등급 또는 5등급 + Skip)를 PNG로 저장.
  - `save_dji_files_wgs84(...)`: **DJI용** 출력 — 격자 등급에 살포량을 매핑해 WGS84 GeoTIFF(Rx) + `.tfw` + 경계 ShapeFile 생성.
  - `save_xag_files_wgs84(...)`: **XAG용** 출력 — 경계 KML + 처방 JSON(borderWKT, weightData, dataTypeLevel 등) 생성. kg/ha를 g/m²(÷10)로 변환, 격자 오차를 역산 보정.
  - `main()`: 전체 루프. 필지코드는 파일명 맨 앞 토큰(`SM01_…` → `SM01`)으로 추출.
- **입력**: GNDVI GeoTIFF (`*_GNDVI.tif`), 필지 경계(zip 또는 shp, 없으면 Otsu로 자동 감지), VRA 설정 CSV.
- **출력**: 기체별 처방 파일(DJI: `.tif`/`.tfw`/`.shp`, XAG: `.kml`/`.json`), 필지별 `*_VRA.csv`, 결과 지도 `*_Result.png`.
- **기체별 분기**: CSV의 `drone_type`(없으면 DJI)으로 판별. DJI는 5등급/필터3→5·격자 CSV값, XAG는 3등급/필터3·격자 5m 강제 고정.

## boundary_detector.py
필지(농경지) 경계를 추출하는 `BoundaryDetector` 클래스.

- **목적**: 사전 제작된 경계 파일을 로드하거나, 없을 경우 GNDVI 영상에서 자동으로 경계를 생성.
- **주요 메서드**
  - `load_boundary_from_zip(zip_path)`: `zip://` 경로로 압축된 ShapeFile 로드 (좌표계 없으면 EPSG:4326 가정).
  - `load_boundary_from_shp(shp_path)`: 일반 `.shp` 파일 로드.
  - `detect_boundary_otsu(tif_path, crop_type)`: GNDVI 영상에 다운샘플링 → 유효 픽셀 마스킹 → Otsu 임계값(작물별 보정 계수: rice 0.90 / soybean 0.85 / wheat 0.95) → 이진화 → 형태학 처리(opening/closing/fill holes 등, 작물별 상이) → 최대 폴리곤 선택 → 단순화(simplify) 후 경계 폴리곤 반환.
- **입력**: 경계 zip/shp 또는 GNDVI GeoTIFF + 작물 종류.
- **출력**: GeoPandas `GeoDataFrame`(경계 폴리곤) 또는 실패 시 `None`.

## vra_calculator.py
VRA 살포량을 계산하는 `VRACalculator` 클래스.

- **목적**: 필지별 설정 CSV를 읽어, 등급별 면적·평균 GNDVI를 기반으로 살포 처방량을 계산.
- **주요 메서드**
  - `__init__(vra_csv_path)` / `_load_vra_data(path)`: 설정 CSV를 읽어 `field`를 인덱스로 한 DataFrame 보관.
  - `get_field_info(field_code)`: 해당 필지의 설정 행 전체 반환(없으면 `None`).
  - `calculate_prescription(field_code, zone_stats)`: 핵심 계산.
    - 목표 총량(`total`)과 변동 강도(`spread`)를 CSV에서 읽음.
    - Zone 6(Skip) 제외한 실제 살포 면적 산출, 면적 가중 평균 GNDVI 계산, 평균 시비량(flat rate) 도출.
    - **STEP 1**: 등급별로 GNDVI가 평균보다 낮으면 더 주고 높으면 덜 주는 방식으로 1차 살포량 계산(최소 50kg/ha 하한선 `MIN_RATE_KG_HA` 보장).
    - **STEP 2**: 1차 합계가 목표 총량과 맞도록 보정 계수(correction_factor)를 곱해 최종량 산출.
    - DJI는 5등급 라벨, XAG는 3등급 라벨로 동적 부여.
- **입력**: 필지 설정 CSV(`field,total,spread,crop,grid_size,sigma,masking` + 선택적 `drone_type,height,width`), 등급별 통계 리스트(`zone_stats`).
- **출력**: `Field, Zone, GNDVI, Area(ha), Rate(kg/ha), Total(kg)` 컬럼을 가진 처방 `DataFrame`.

## verification_tool.py
파이썬 산출물과 Pix4D 산출물을 비교/검증하는 독립 실행 도구.

- **목적**: 본 프로그램이 만든 Rx GeoTIFF와 Pix4D가 만든 Rx GeoTIFF를 같은 필지끼리 비교하여 정확도/유사도를 정량 평가하고 리포트 이미지를 생성.
- **상단 설정 상수**: `GEN_FOLDER`(파이썬 결과 Rx 폴더), `PIX_FOLDER`(Pix4D 결과 Rx 폴더), `REPORT_FOLDER`(리포트 출력 폴더, 기본 `verification_reports`). 실행 환경에 맞게 수정 필요.
- **주요 구성**
  - `MapComparator` 클래스: 두 래스터 정렬(`align_rasters_for_calc`, Pix4D 격자 기준으로 reproject), 살포량 비교(`compare_rates` → 평균/ MAE / 상관계수), 등급 순위 비교(`compare_zones_by_rank` → 정확도 accuracy / Cohen's Kappa), 시각화(`visualize_comparison` → Python/Pix4D/Diff 3분할 지도 + 통계 텍스트 PNG).
  - `find_and_pair_files(gen_folder, pix_folder)`: 파일명 맨 앞 필지코드가 같은 파일끼리 짝지음.
  - `main()`: 짝지은 필지를 순차 처리하여 리포트 생성.
- **입력**: 파이썬 Rx GeoTIFF, Pix4D Rx GeoTIFF (필지코드로 매칭).
- **출력**: 필지별 비교 리포트 이미지 `Report_<필지코드>.png` (정확도·MAE·상관계수·Kappa 포함).

---

# 데이터 흐름 / 실행 순서

기본 처방 생성 파이프라인 (`operation_main.py`):

1. `operation_main.py` 상단 설정 상수(`DATA_FOLDER`, `BOUNDARY_FOLDER`, `OUTPUT_FOLDER`, `VRA_CSV_PATH`)를 대상 작업에 맞게 수정.
2. `python operation_main.py` 실행.
3. `VRACalculator`가 설정 CSV 로드, `BoundaryDetector` 준비.
4. `DATA_FOLDER`의 `*_GNDVI.tif`를 하나씩 처리:
   - 파일명 맨 앞 토큰으로 **필지코드** 추출 → CSV에서 필지 설정(작물, 격자 크기, sigma, 기체 타입 등) 조회.
   - 경계 확보: `*_Boundary.zip` → `*.zip` → `*.shp`(입력) → `*.shp`(출력) 순으로 탐색, 없으면 GNDVI에서 Otsu 자동 감지. 지리좌표면 EPSG:5179로 변환.
   - 래스터 경계 클리핑 → 회전 격자 생성 → 격자별 평균 GNDVI 계산.
   - 나지(Zone 6) 마스킹 임계값 계산 → 나머지를 분위수로 N등급 분류 → 스무딩으로 최종 `Zone` 확정.
   - 등급별 면적·평균 GNDVI 집계 → `VRACalculator.calculate_prescription`로 살포량 계산.
   - 기체 타입에 따라 DJI(GeoTIFF+ShapeFile) 또는 XAG(JSON+KML) 처방 파일 + `*_VRA.csv` + `*_Result.png` 저장.
5. 모든 필지 처리 완료.

검증 파이프라인 (`verification_tool.py`, 선택):

1. 파이썬 결과 Rx 폴더와 Pix4D 결과 Rx 폴더 경로를 코드 상단에 설정.
2. `python verification_tool.py` 실행 → 필지코드로 파일 매칭 → 정렬·통계·시각화 → `verification_reports/Report_*.png` 생성.

> `boundary_detector.py`와 `vra_calculator.py`는 `operation_main.py`가 import하여 사용하는 모듈로, 단독 실행 진입점이 없다.

---

# 의존성

Python 3.12 기준 (`.pyc` 캐시가 cpython-312). 주요 외부 라이브러리:

- **geopandas** — 벡터(경계/격자) GeoDataFrame 처리
- **rasterio** — 래스터(GeoTIFF) 입출력, 클리핑(mask), 래스터화(rasterize), 재투영(reproject)
- **shapely** — 기하 연산(Polygon, affinity 회전 등)
- **numpy**, **pandas** — 수치/표 데이터 처리
- **scipy** (`scipy.ndimage`) — 가우시안/형태학/일반 필터링
- **scikit-image** (`skimage`) — Otsu 임계값, 라벨링, 형태 측정
- **scikit-learn** (`sklearn.metrics`) — 검증 도구의 accuracy, Cohen's Kappa (verification_tool.py 전용)
- **matplotlib** — 결과 지도 및 리포트 이미지 시각화

> `requirements.txt`는 현재 없음. git 공유 시 위 패키지 목록으로 작성해두면 좋다. 좌표계는 한국 측지계 EPSG:5179(분석)와 WGS84 EPSG:4326(출력)을 사용하므로 GDAL/PROJ가 포함된 환경(예: conda) 권장.

---

# 비고

- **경로 하드코딩**: 핵심 경로가 `operation_main.py`, `verification_tool.py` 상단 상수에 하드코딩되어 있다. 실행 전 반드시 자신의 데이터 폴더에 맞게 수정해야 하며, 현재 커밋된 값(`sm_hm_test_data`, `result_0604`, `sm_hm_vra.csv`)은 마지막 실험 흔적으로 실제 폴더명과 다를 수 있다.
- **필지코드 규칙**: 모든 매칭이 파일명 맨 앞 토큰(`_` 또는 `.` 앞)을 필지코드로 사용한다. 입력 영상·경계 파일·CSV의 `field` 값·출력 파일명이 같은 코드로 일치해야 한다 (예: `SM01`, `GJR1`, `BD01`).
- **기체별 동작 차이**: CSV의 `drone_type` 열로 DJI/XAG를 구분한다. 열이 없으면 DJI(기본). XAG는 3등급·격자 5m 강제 고정·JSON/KML 출력, DJI는 5등급·CSV 격자값·GeoTIFF/ShapeFile 출력.
- **VRA 설정 CSV 컬럼**: `field, total, spread, crop, grid_size, sigma, masking`이 기본이며, 코드는 추가로 `drone_type, height, width`도 참조한다(`vra_calculator.py`, `operation_main.py`). 2026-07-05부터 `vra.csv`/`gj_vra.csv`는 10컬럼(`height,width,drone_type` 포함) 체계 — `drone_type`은 DJI/XAG 분기(미기재 시 DJI), `height`/`width`는 비행고도/살포폭으로 경계 shp DBF와 파일명에 반영. `sm_vra.csv`/`bd_vra.csv`는 아직 기본 7컬럼. `sigma`가 비어 있으면 자동 계산, `masking`은 나지 마스킹 강도(relax_factor)로 쓰인다.
- **전체 처방 요약 엑셀 출력 (2026-07-20 추가, `operation_main.save_run_summary`)**: 한 번 실행한 전체 필지를 한 파일로. OUTPUT_FOLDER 최상위에 `처방요약.xlsx`(openpyxl, 실패 시 `처방요약.csv` utf-8-sig 폴백). 컬럼: 필지, 기체, 계산방식(면적비율/절대량), 총비료량(kg), 포대수(20kg), 필요포대수(올림), 필지면적(평), 살포면적(평), 살포면적(ha), 평균살포량(kg/ha). 1포대=20kg 상수(`BAG_KG`), 1평=3.3057851㎡(`PYEONG_M2`). main() 루프에서 필지별 1행 집계 후 루프 종료 시 저장. 웹앱 zip에 자동 포함. **exe 빌드에 `--collect-submodules openpyxl` 추가 필수**(없으면 xlsx 실패→csv 폴백). exe E2E 검증 통과.
- **비료량 입력 방식 2종 (2026-07-20 추가, `vra_calculator._resolve_total_kg`)**: 농가별 비료 기준 대응.
  - **절대량 모드(기존)**: `total`(kg) 그대로. 예: 2포대=40kg → `total=40`.
  - **면적 비율 모드(신규)**: `rate_kg` + `rate_area` 있으면 `total = rate_kg × (필지면적 ÷ rate_area)`. 예: 1500평당 60kg → `rate_kg=60, rate_area=1500`. 필지면적 = `field_area`(직접 입력) 우선, 없으면 **실측 자동**(경계 면적 m², `operation_main`이 `field_area_m2`로 전달). `area_unit`=평(기본)/ha/m². 1평=3.3057851㎡. `rate_kg`&`rate_area` 있으면 `total` 무시.
  - 하위호환: 비율 컬럼 없으면 기존 total 동작 그대로. total·비율 둘 다 없으면 경고 후 처방 스킵.
  - 검증(CLI+exe): 절대40→40kg, 비율자동실측, 등기2500평×60/1500=100kg, 0.5ha×100/ha=50kg 전부 정확. exe 재빌드 `dist\VRA_Webapp_20260720.zip`.
- **균등(비변량) 살포 처방맵**: 별도 모드 없이 **`spread=0`으로 설정**하면 됨 (2026-07-13 실증 — 전 등급 rate 동일 = total÷살포면적, 총량 정확 일치, Rx GeoTIFF 픽셀값 균일). `rate = flat_rate × (1 − 편차 × spread)` 공식에서 spread=0이면 편차항 소거. 나지(Zone 6) 자동 제외는 균등에서도 유지됨. 결과 PNG는 여전히 5색 등급 지도로 그려짐(GNDVI 분포 참고용) — 실살포량은 VRA.csv Rate 컬럼·Rx 픽셀값 기준. 중간값(0.5 등)으로 변량 강도 조절도 가능.
- **CSV 인코딩**: 출력 ShapeFile/VRA CSV는 `euc-kr`로 저장된다(한글 파일명/필드명 호환).
- **git 공유 시 권장 `.gitignore`**: `__pycache__/`, `.idea/`, 대용량 산출물(`result/`, `verification_reports/`, `.tif` 등). 코드와 `vra_setting/` 설정 CSV 위주로 공유 권장.
- **vra_setting 폴더에는 .py 파일이 없다** — 설정용 CSV 3개(`vra.csv`, `sm_vra.csv`, `bd_vra.csv`)만 존재한다.
- **`pix4d_자동화 비교.pptx`**: 본 자동화 결과와 Pix4D 결과를 비교한 발표/검토 자료. 코드 동작과는 직접 관련 없는 참고 문서.

---

# 변경 이력

- **2026-07-12 vra.csv 한글 인코딩 자동 감지 + yc(연천) 데이터 대응**:
  - **증상**: yc_data 신규 데이터(`경기도 연천군 전곡읍 은대리 1196(2)_260711_GNDVI.tif` = `지번주소_촬영일_GNDVI.tif`)에서 웹앱이 vra.csv 필지코드를 인식 못함. "한글 읽기 오류".
  - **원인**: `VRACalculator._load_vra_data`가 `pd.read_csv(path)`(기본 utf-8)로만 읽음. 한국 엑셀 'CSV로 저장'은 기본이 **cp949(euc-kr)** → `'utf-8' codec can't decode byte 0xb0` 예외 → vra_data=None → 전 필지 인식 실패. (필지코드 추출 `split("_")[0]`은 정상 — 주소가 공백 구분이라 전체 주소가 필지코드. 쉼표·괄호도 utf-8 CSV에선 자동 따옴표로 무해)
  - **수정**(`vra_calculator.py`): 인코딩 `[utf-8-sig, utf-8, cp949, euc-kr]` 순차 시도(utf-8류 먼저 → 잘못된 바이트면 예외로 넘어가고 cp949는 마지막). `field` 컬럼 없으면 명확 오류. 필지코드·컬럼명 `.str.strip()`. `operation_main.py` 필지코드 추출에도 `.strip()`.
  - **필지코드 = 전체 지번주소**(예: `경기도 연천군 전곡읍 은대리 1196(2)`). vra.csv의 field에 이 주소 문자열 그대로 넣어야 매칭. 출력 파일명도 이 주소(쉼표·괄호·공백 포함) — Windows 허용. **주의: 긴 한글주소 + 깊은 jobs 경로 → MAX_PATH(260) 초과 가능성**(exe를 얕은 경로에 설치 권장).
  - **`vra_setting/yc_vra.csv` 신규**: yc_data 68개 고유 필지코드 자동 추출 템플릿(utf-8-sig=엑셀 한글 정상, total 60/DJI 기본). 사용자가 필지별 total 조정 후 사용.
  - 검증: CLI + exe 양쪽에서 cp949 vra.csv + 한글주소 GNDVI(1196(2)) E2E 통과 — 인코딩 자동감지 로그, DJI Rx/tfw/ShapeFile/VRA.csv/PNG 전량 생성(한글 파일명 유지). exe 재빌드 → `dist\VRA_Webapp_20260712.zip`(170MB).
  - **결과 PNG 제목 한글 깨짐(□□□) 수정**: matplotlib 기본 폰트(DejaVu Sans)에 한글 글리프 없음. `operation_main.py`에 `_setup_korean_font()` 추가 — `font_manager.ttflist`에서 [Malgun Gothic, NanumGothic, Gulim, Batang, Dotum] 순 탐색해 `plt.rcParams["font.family"]` 지정 + `axes.unicode_minus=False`. Windows엔 맑은 고딕 기본 존재. **PyInstaller frozen에서도 시스템 폰트(C:\Windows\Fonts) 스캔 정상** — exe 산출 PNG로 한글 제목 렌더 확인. 이 수정도 0712 exe에 포함.

- **2026-07-10 웹앱 'PNG만 생성' UX 결함 수정 + exe 재빌드**:
  - **증상**: 사용자가 처방맵 zip을 받았는데 Result.png만 있고 DJI Rx/ShapeFile·XAG JSON(기체 로드용 데이터)이 없음.
  - **원인**: main()에서 `save_map_image`(PNG)는 `if vra_df is not None` 바깥이라 **항상** 저장되지만, DJI/XAG 파일과 VRA.csv는 처방 계산 성공(`vra_df is not None`) 시에만 저장. **vra.csv의 field 코드가 영상 파일명 첫 토큰(필지코드)과 불일치**하면 `calculate_prescription`이 None 반환 → PNG만 남음. 웹앱은 이걸 '성공'으로 처리해 png뿐인 zip을 내려줌(로그의 `[Warning] 'XXX'에 대한 VRA 설정값을 찾을 수 없습니다`를 사용자가 안 봄).
  - **수정**(`webapp/app.py` `_run_job`): om.main() 후 DJI Rx `*.tif` + XAG `*_Prescription.json` 개수 확인. **0개면 state=error로 실패 처리** + "vra.csv field가 영상 필지코드와 다릅니다, 영상 필지코드는 [...]" 안내. 일부 필지만 실패하면 로그에 실패 필지코드 나열 + "처방 N필지/실패 M필지" 요약. → png-only zip이 '성공'으로 나가지 않음.
  - **vra.csv 없을 때(Q)**: 웹앱은 finalize에서 `VRA 설정 CSV가 업로드되지 않았습니다` 오류(HTTP 400)로 이미 차단(정상). 단 CLI(operation_main 직접)는 csv 없으면 전 필지 png-only 됨 — 위 수정은 웹앱 한정.
  - **참고(정상 동작)**: DJI 산출물은 zip 안 `DJI/Rx/`(tif,tfw) + `DJI/ShapeFile/`(shp 세트) **하위 폴더**에 위치(Pix4D 호환 구조). 최상위엔 Result.png·VRA.csv만 보임 → 사용자가 하위 폴더 못 보고 'png만'으로 오해 가능. XAG는 `XAG/`(json,kml).
  - exe 재빌드 → `dist\VRA_Webapp_20260710.zip`(170MB). 불일치→오류 / 일치→DJI 전체 생성 두 시나리오 exe에서 검증 통과.
  - **프론트(JS) 오류 표시 강화**: 기존엔 error 시 "처리 중 오류가 발생했습니다. 로그를 확인하세요"만 떠서 사용자가 원인을 못 봄. 이제 로그에서 `[오류]` 줄을 추출해 화면 메시지에 직접 표시(❌ + 실제 원인), 일부 필지만 실패 시 done이어도 `[주의]` 줄을 노란 경고로 표시. Python PAGE 문자열 내 정규식은 `\\s`/`\\[`로 써야 JS에서 `\s`/`\[`로 렌더됨(확인). **사용자가 '불일치인데 오류 안 뜬다'고 한 건 수정 이전 exe 사용이 원인 — 0710 exe 배포 필요.**

- **2026-07-09 Footprint 경계 방식 추가 + CRS/glob 버그 2건 수정** (SC 김제 필지, 경계 파일 없음):
  - **glob 패턴 확장**: `main()`의 `*_GNDVI.tif` → `*_GNDVI*.tif`(sorted+set). SC 파일명이 `SC03_GNDVI.data.tif`(중간 `.data`)라 기존 패턴에 하나도 안 잡혀 루프가 빈 채로 조용히 종료되던 문제. 필지코드는 여전히 첫 토큰.
  - **CRS 불일치 버그 수정**(`calculate_grid_mean_stats`): SC GNDVI가 **EPSG:4326**인데 `clip_raster_to_boundary`는 래스터를 원본 CRS 유지, 격자는 5179 생성 → 4326 래스터를 5179 격자로 샘플링해 **전 셀 NaN → 조용히 continue(산출물 0개)**. 샘플링 시 `grid_gdf.to_crs(src.crs)`로 해결. 평균값은 CRS 무관, 면적은 5179로 정확. GJR(5179 영상)은 우연히 CRS가 맞아 동작했던 것 — **4326 드론 영상은 이 수정 없이는 처방 안 나옴**.
  - **`detect_boundary_footprint()` 신규**(`boundary_detector.py`): 유효 데이터(non-NaN/nodata) 외곽선을 경계로. 드론 정사영상이 필지대로 잘려(바깥 NaN) 나온 경우 Otsu보다 확실. fill_holes+closing+opening+최대연결영역+simplify. `operation_main.py`에 `BOUNDARY_METHOD='footprint'|'otsu'` 상수(기본 footprint), 경계 파일 없을 때 fallback. Otsu는 사각형 영상에 주변 논/도로 포함 시만.
  - **footprint > Otsu 근거**: Otsu는 GNDVI 임계 식생판정 → 가장자리 벼줄·낮은GNDVI 모서리 잘림(SC 실측 필지면적 91~96%로 축소, 톱니 경계). footprint는 외곽 100% 추적, 임계 불필요, 작물/시기 무관. RGB 불필요(같은 비행 RGB는 footprint 동일).
  - 검증: SC03~SC14 10필지 전량 성공. 총량 60.00±0.01kg, 면적 0.236~0.533ha, Rx(EPSG:4326,nodata=None)·경계 SHP 정상. 몽타주로 10필지 경계가 실제 외곽 정확 일치(불규칙 SC14·곡선 SC11 포함). 경로: DATA=data/sc_data/25_0721, OUTPUT=result/sc_result_0709, VRA=vra_setting/sc_vra.csv.

- **2026-07-08 웹앱 .exe 패키징 (PyInstaller)** — 로컬 서버 불안정 대응, PC별 독립 설치형:
  - `빌드_exe.bat` 신규: PyInstaller onedir 빌드 → `dist\VRA_Webapp\VRA_Webapp.exe` (426MB, zip 170MB). conda/파이썬 설치 없이 폴더 복사+더블클릭으로 각 PC에서 localhost 서버 독립 실행.
  - `webapp/app.py` frozen 지원 추가: `sys.frozen`이면 BASE_DIR=exe 폴더(쓰기 불가 시 `%LOCALAPPDATA%\VRA_Webapp` 폴백), jobs 폴더 exe 옆 생성. cp949 콘솔 크래시 방지 `reconfigure(errors="replace")`.
  - **빌드 함정 2개 (재빌드 시 필수)**: ① `rasterio.sample` 등 동적 모듈 → `--collect-submodules rasterio --collect-data rasterio --collect-submodules pyogrio --collect-data pyogrio --collect-data pyproj` 필요 (없으면 exe 시작 크래시). ② python312 env의 torch+CUDA(SAM용)가 통째로 끌려와 4GB → `--exclude-module torch/torchvision/cv2` 등으로 426MB. 플래그는 빌드_exe.bat에 고정됨.
  - **코드 수정 시 exe 재빌드 필수** (py가 exe에 번들). 검증: exe 기동→리스너 확인→DJI/XAG E2E 통과. `.gitignore`에 dist/·build/·*.spec 추가. 배포안내.md에 '방법 A. exe 배포' 절 추가.
  - 테스트 중 발견: 포트 8000에 이전 dev 서버가 살아있으면 exe가 안 떠도 HTTP 200이 나옴 — exe 검증 시 반드시 리스너 프로세스 확인(`Get-NetTCPConnection`).
  - vra_setting/gj_webapp_vra.csv 신규 (GJR1~8, DJI, 웹앱 업로드용). GJR5 실데이터(160MB) 웹앱 E2E 통과: 총량 60.00kg 정확, 처리 6초.

- **2026-07-07 XAG Pix4D 호환성 수정** (`operation_main.py` > `save_xag_files_wgs84`, 기준: `pix4d_data/GJR5_XAG` 실물 산출물 대조):
  - **KML `<Folder>` 래퍼 제거**: Pix4D는 Placemark가 `<Document>` 바로 아래에 위치 (기존 코드는 `<Folder><name>Field_Boundary</name>`로 감쌌음). XAG 앱 파서가 고정 경로로 읽을 가능성 대비 구조 일치화. Style 블록도 Pix4D와 동일한 멀티라인 들여쓰기로 변경.
  - **KML 내부 링(구멍) 반영**: 기존엔 WKT(borderWKT)에만 interior 포함하고 KML은 exterior만 기록 → 구멍 있는 필지에서 KML/JSON 경계 불일치 가능성. `<innerBoundaryIs>` 추가로 일치화 (GJ 필지는 단일 폴리곤이라 기존 산출물엔 무영향).
  - **dosage 정수화**: Pix4D 산출물이 정수 체계(19/18/17) → `round(x,2)` float에서 `int(round(x))`로 변경. 총량 정밀도 약간 희생(±수 kg), 실기체 float 허용 여부 미확인이라 안전측 선택.
  - **name 통일**: KML `<name>`/JSON `name`을 `{필지코드}_XAG`로 (기존 filename_base엔 grid·날짜 붙어 있었음, Pix4D는 'GJR5_XAG' 패턴). 출력 파일명은 기존 `{필지}_XAG_{grid}m_{날짜}` 유지.
  - 검증: 합성 데이터로 생성 후 Pix4D 원본과 프로그램 대조 — KML 요소 트리(경로 순서 포함)·태그 속성 완전 일치, JSON 17개 키 순서·값 타입 완전 일치, weightData 길이=rows×columns, dosage 정수 확인. DJI 호환성 수정(7/4)과 함께 양 기체 모두 Pix4D 구조 일치 완료.

- **2026-07-07 webapp/app.py 재작성 (회사 PC)**:
  - 7/5 웹앱은 집 PC에서 작성 후 **git에 커밋되지 않아** 회사 PC에 `webapp/` 폴더가 없었음 (bat·배포안내.md·memory.md만 커밋됨). memory.md 기록 스펙 그대로 회사 PC에서 재구현.
  - 구현 내용: 분할 업로드(`/chunk` 64MB 순차 append → `/finalize` 작업 구성+백그라운드 실행), 1.5초 `/status` 폴링(로그 실시간), `/download`(결과 zip)·`/preview`(Result.png), 동시 실행 방지 락, `MAX_CONTENT_LENGTH=256MB`, `threaded=True`, `VRA_NO_BROWSER=1` 지원, 포트 8000. 바운더리 업로드는 zip묶음/개별 zip/shp 세트 3경로 모두 지원하며 **zip 핸들을 닫은 뒤 os.replace** (7/5 발견 버그 재발 방지). 업로드 처리 전체 try/except → JSON 오류 응답, JS는 비JSON 응답 시 HTTP 상태 안내 + 폴링 4회 연속 실패 시 연결 끊김 안내.
  - matplotlib `Agg` 백엔드 강제(백그라운드 스레드에서 PNG 저장), `operation_main`을 import해 경로 상수만 작업 폴더(`webapp/jobs/<타임스탬프>/`)로 패치 후 `om.main()` 실행 — 기존 파이프라인 그대로 재사용.
  - E2E 검증 통과 (합성 GNDVI + 경계 zip + vra.csv): ① DJI 경로 — 다중 조각(6조각) 업로드→Rx tif/tfw+ShapeFile+VRA csv+Result png 생성→zip 다운로드→미리보기, ② XAG 경로 — zip묶음 바운더리→5m 격자 강제→KML+Prescription JSON 생성. 동시 실행 락 동작 확인.
  - **`.gitignore` 신규 추가** (webapp/jobs/, cloudflared.exe, data/, result/ 등) — webapp 미커밋 사고 재발 방지 목적. **webapp/app.py를 git에 커밋해 양쪽 PC 동기화 필요.**
  - 회사 PC python312 env에 flask 3.1.3 pip 설치함 (GIS 스택은 기존재).

- **2026-07-05 웹서비스 배포 방안 검토 (진행 대기)**:
  - 무료 호스팅(Render/PythonAnywhere 등)은 부적합 판정 — 업로드 용량(GNDVI zip 0.8~2.5GB), 메모리(GDAL 처리 시 수 GB, 무료 티어는 ~512MB), 처리 시간/슬립(cold start) 한계. 농지 좌표·작황 데이터의 외부 서버 업로드는 보안 검토도 필요.
  - 검토 결과 권장안: **사내 PC 상시 실행 + Cloudflare Tunnel(무료, 외부 어디서나 HTTPS 접속)** 또는 **Tailscale(무료, 지정 팀원만 접속)**. 차선책: Oracle Cloud Always Free VM(4코어/24GB, 리눅스 관리 필요), Hugging Face Spaces(16GB RAM이나 업로드 제약·슬립). 불특정 다수 대상 서비스로 확장 시에는 유료 VPS 권장.
  - **결정(2026-07-05): Cloudflare Tunnel 채택** — 사무실 PC 상시 실행 + 외부 접속. 구현 완료:
    - **분할 업로드 도입**: Cloudflare 무료 플랜의 요청당 100MB 제한 대응. 브라우저 JS가 파일을 64MB 조각으로 잘라 `/chunk`(순차 append) → `/finalize`(작업 구성+실행)로 전송. 진행률 바 표시. 기존 `/upload` 일괄 방식은 제거. `MAX_CONTENT_LENGTH`는 256MB로 축소, `threaded=True`(처리 중에도 폴링 응답), `VRA_NO_BROWSER=1` env로 서버 PC에서 브라우저 자동열림 억제.
    - **`처방맵_외부공개_실행.bat` 신규**: 웹앱(최소화 창) + cloudflared 빠른 터널 실행, trycloudflare.com 주소 발급. cloudflared.exe 없으면 자동 다운로드. `webapp/cloudflared.exe`(52MB) 다운로드해 둠.
    - 검증: 로컬 분할 업로드 E2E(831MB/13조각) 통과, **실제 터널 경유 E2E 통과**(공개 URL로 171MB tif 업로드→처방 생성→결과 다운로드).
    - 주의: 빠른 터널은 재시작마다 주소 변경. 주소 고정/접근 제한이 필요해지면 Cloudflare 계정+도메인으로 이름 있는 터널 + Cloudflare Access(무료 이메일 인증) 전환. 서버 host는 127.0.0.1 유지(터널이 로컬 프록시하므로 LAN 직노출 없음). 부팅 자동시작은 shell:startup에 bat 바로가기(배포안내.md 참조).

- **2026-07-05 비개발자용 웹앱 추가** (`webapp/app.py`, `처방맵_웹앱_실행.bat` 신규):
  - Flask 기반 로컬 웹앱. `처방맵_웹앱_실행.bat` 더블클릭 → 브라우저 자동 오픈(127.0.0.1:8000) → ① GNDVI tif(zip) ② 바운더리 zip(선택, zip묶음/개별zip/shp 모두 허용) ③ vra.csv 업로드 → 처방맵 생성 → 결과 zip 다운로드 + Result.png 미리보기.
  - 내부적으로 `operation_main`을 import해 경로 상수만 작업 폴더로 바꿔 `om.main()` 실행 — 검증된 파이프라인 그대로 재사용. 작업별 폴더는 `webapp/jobs/<타임스탬프>/`(data/boundary/output/vra.csv/결과zip). 진행 로그 실시간 표시(1.5초 폴링), 동시 실행 방지 락.
  - 의존성: python312 conda env에 Flask 3.1.3 설치함.
  - **배포 패키지 (2026-07-05 추가)**: `requirements.txt`(검증 버전 고정), `배포안내.md`(새 PC 설치 절차: Miniconda → `conda create -n vra -c conda-forge python=3.12 geopandas rasterio scipy scikit-image scikit-learn matplotlib flask`), bat는 anaconda3/miniconda3의 python312·vra env를 자동 탐색(ASCII 메시지, 못 찾으면 안내 후 종료). 전달 대상 = 코드 5개(.py) + bat + requirements + 배포안내 (data/result/jobs 등 대용량 불필요). 웹앱 오류 메시지 개선: 서버 미실행 시 "서버에 연결할 수 없습니다" 한글 안내, 폴링 중 연결 끊김 감지(4회 연속 실패 시).
  - E2E 테스트 통과: GJR1·2 zip 업로드 → 18개 파일 생성 → zip 다운로드(Rx tif/tfw, ShapeFile, VRA csv, Result png 포함) → 미리보기 정상.
  - **버그 수정 (실사용 중 발견)**: 바운더리 zip을 '개별 파일'로 업로드하면 `_save_boundary_uploads`가 zip 핸들을 연 채 `os.replace`를 호출 → Windows에서 "파일 사용 중" 오류 → 500 HTML 응답 → 화면에 "SyntaxError: Unexpected token '<'" 표시되던 문제. zip을 닫은 뒤 rename하도록 수정. 업로드 처리 전체를 try/except로 감싸 JSON 오류 응답 반환, JS는 비JSON 응답 시 HTTP 상태 안내. 사용자 시나리오(개별 zip 5개) 재현 테스트 통과. 교훈: zip묶음 경로만 테스트하고 개별 zip 경로를 미검증했었음 — 분기 경로 전부 테스트할 것.

- **2026-07-05 광주(GJ) 필지 자동화 테스트** (`operation_main.py`, `vra_setting/gj_vra.csv` 신규):
  - 경로 변경: `DATA_FOLDER=data/gj_data`, `BOUNDARY_FOLDER=data/gj_boundary`, `OUTPUT_FOLDER=result/gj_result_0705`, `VRA_CSV_PATH=vra_setting/gj_vra.csv`.
  - `gj_vra.csv` 신규 작성: GJR1~10, 벼, total 60kg, grid 1m, masking 0.5, height 3, width 5 (기존 vra.csv GJR 세팅 준용).
  - **`find_boundary_zip()` 함수 추가**: 경계 zip 탐색을 유연화 — `{필지}.zip`, `{필지}_Boundary.zip`, `{필지}_boundary.zip` 등 네이밍 변형·대소문자 차이를 모두 허용(파일명 첫 토큰==필지코드). 기존의 고정 파일명 2종 검사 대체. GJR1↔GJR10 같은 접두어 오매칭은 토큰 완전일치로 방지.
  - 실행 결과: 10필지 전부 성공. GJR1~5는 zip 경계, GJR6~10은 Otsu 자동 감지. 총량 60kg 정확히 일치, tfw 중심좌표·DBF height/line_space·nodata=None 모두 검증 통과.
  - **GJR9 주의**: 원본 GNDVI에 큰 나지 블록(좌측)과 하단 띠가 포함 → Otsu 경계가 식생 영역만 잡아 면적 0.233ha(타 필지 ~0.39ha)로 작음 → 동일 총량 60kg이 좁은 면적에 배분되어 살포량 264~288kg/ha로 높음. 나지가 필지 일부라면 경계 zip 제공 또는 total 조정 필요.
  - **대규모(새만금 SM) 적용 분석**: 95% 규칙 하에서 총량 소진 오류는 구조적으로 불가(STEP 2 보정으로 처방 적분=total, 전 격자 경계 내). 단 SM 세팅(5m 격자)은 가장자리 셀 제외 폭이 넓어 커버리지 92.5~95.4%(SM01~12 실측, 제외 4.24ha/65.4ha=6.5%), 내부 살포량 +6.9% 농축. 1m 격자면 손실 1.2%. DJI 살포폭(5m) 흩뿌림이 제외 밴드를 사실상 커버하므로 현행 유지 결정. ha당 기준 엄수 필요 시 total×0.935 보정 가능. 대규모 작업 시 경계 shp 필수(Otsu 의존 금지) + 실행 후 필지별 Area(ha) 대장 대조 권장.
  - **Pix4D 검증 (GJR1~5, verification_tool)**: 리포트 `verification_reports/gj_0705/`. 상관계수 0.87~0.91, Zone 일치도 69~77%, Kappa 0.60~0.70. 파이썬 평균 살포량이 전 필지 +6~12 kg/ha 높은 체계적 편차 존재 — 원인은 총량 차이가 아니라(양쪽 모두 총 투입량 ~60kg 일치) **살포 footprint 차이**: 파이썬은 95% overlap 규칙으로 가장자리 격자를 제거해 면적이 3~6% 작음(0.37ha vs Pix4D 0.39~0.40ha) → 같은 총량이 좁은 면적에 배분되어 단위면적당 살포량 상승. 2026-06-30 의도된 설계(경계 밖 낭비 방지)의 결과이며 오류 아님. `verification_tool.py` 경로 상수는 gj_0705 기준으로 변경됨.

- **2026-07-04 Pix4D 호환성 수정** (`operation_main.py` > `save_dji_files_wgs84`):
  - **경계 ShapeFile에 속성 추가**: Pix4D 산출물 분석 결과, DJI용 경계 shp의 DBF에 `height`(비행고도 m), `line_space`(살포폭 m), `name`(처방맵 이름) 3개 필드가 들어있음을 확인. 파이썬 출력도 동일 스키마로 저장하도록 수정 (CSV의 `height`/`width` 값 사용, 미정의 시 기본값 3.0/5.0 — Pix4D 세팅과 동일). 인코딩도 euc-kr → **UTF-8**(.cpg=UTF-8, Pix4D와 동일)로 변경. 경계는 union 후 1개 레코드로 저장.
  - **.tfw 반 픽셀 보정**: World File 표준은 5·6번째 줄이 좌상단 픽셀의 '중심' 좌표인데 기존 코드는 '모서리' 좌표를 기록 → 약 0.5m 북서쪽 오프셋 발생. `corner + pixel/2`로 수정하고 회전값 기록 순서도 표준(A,D,B,E,C,F)으로 정정. Pix4D .tfw와 동일한 소수점 10자리 포맷.
  - 참고: 살포량(kg/ha)은 Rx GeoTIFF 픽셀 값 자체에 저장되며(메타데이터 태그 없음), 비행고도·살포폭은 경계 shp DBF에만 저장됨.
  - **(추가) Rx GeoTIFF nodata 제거**: Pix4D는 nodata 태그 없이 0을 실제 픽셀 값(살포 제외)으로 저장 → 파이썬도 `nodata=None`으로 변경 (기존 `nodata=0`은 일부 SW에서 0 셀을 '데이터 없음'으로 무시할 수 있음).
  - **(추가) `vra_setting/vra.csv`에 `height,width` 컬럼 추가**: Pix4D 실측값 기준 GJR1~6·GR08 = 3m/5m, HSR1·HSR6 = 2m/5m (화성 Pix4D DBF에서 height=2 확인). GR08은 실측 미확인으로 GJR과 동일하게 3/5 가정. `sm_vra.csv`/`bd_vra.csv`는 실제 세팅 미확인이라 미수정(코드 기본값 3/5 적용됨).
  - 검증: Pix4D vs 파이썬 산출물 전체 메타데이터 비교 완료 — DBF 스키마(필드 타입/길이/소수자릿수), .prj, shp 지오메트리 타입(Polygon=5), GeoTIFF 프로파일(dtype/compress/interleave/colorinterp/태그) 모두 일치 확인. 파일명 H/W 표기는 `:g` 포맷으로 정수 표시(H3m).

- **2026-06-30 pull 반영** (`operation_main.py`, 커밋 "변수 수정"):
  - 작업 대상 설정 경로 변경: `DATA_FOLDER` → `data/sm_hm_test_data`, `OUTPUT_FOLDER` → `result/result_0604`, `VRA_CSV_PATH` → `vra_setting/sm_hm_vra.csv` (XAG 테스트 → SM/HM 데이터로 전환).
  - 격자 살포 구역 판정 임계값 상향: `create_rotated_grid_with_indices`의 `overlap_ratio` 기준이 **0.4 → 0.95**. 경계 밖으로 삐져나가는 가장자리 격자를 제거해 비료가 모자라던 문제를 해결하기 위한 로직 수정.
