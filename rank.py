#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일설명: Cross-Sectional Rank Feature Generator (rank.py)
         - 전처리된 데이터를 로드하여 일별(Daily) 횡단면 순위를 계산합니다.
         - Percentile Rank(0~1) 및 Quintile(1~5) 피처를 생성합니다.
         - 원본 데이터와 분리된 별도의 Parquet 파일로 저장하여 모듈성을 확보합니다.
         - 메모리 최적화를 위해 연도별로 스트리밍 처리합니다.
"""

import os
import glob
import shutil
import time
import json
import gc
import traceback
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# ==============================================================================
# 1. Configuration (설정)
# ==============================================================================
class Config:
    # --- 경로 설정 ---
    # GDrive 경로 (입/출력 최종 목적지)
    GDRIVE_ROOT = "/content/drive/MyDrive/rre/data"
    
    # 로컬 작업 경로 (고속 처리를 위한 임시 공간)
    LOCAL_ROOT  = "/content/rre_local_rank"
    
    # 입력 파일 (전처리된 원본 데이터)
    INPUT_ZIP_NAME = "processed_data_parquet.zip"
    INPUT_DIR_NAME = "processed_data_parquet"
    
    # 출력 파일 (생성될 랭크 데이터)
    OUTPUT_DIR_NAME = "rank_features_parquet"
    OUTPUT_ZIP_NAME = "rank_features_parquet.zip"

    # --- 랭크 변환 대상 피처 (Feature Registry) ---
    # 실제 데이터에 존재하는 컬럼만 동적으로 필터링하여 사용합니다.
    # 횡단면 비교가 의미 있는 팩터들 위주로 선정
    TARGET_FEATURES = [
        # 1. Momentum / Trend (추세 강도 비교)
        "RET_5", "RET_20", "RSI_14", "PPO_HIST",
        "DONCHIAN_POS_20", "CLOSE_Z_252", 
        
        # 2. Volatility / Risk (변동성 수준 비교)
        "RV_5", "RV_20", "ATR_14_REL", "BB_WIDTH_Z_60",
        "DRAWDOWN_60", "BETA_60", "IDIOSYNCRATIC_VOL_20",

        # 3. Liquidity / Volume (수급/거래 강도 비교)
        "VOLUME", "AMIHUD_REL", "VWAP_DIST_Z_60", "KER_20",

        # 4. Supply / Demand (Smart Money Flow)
        "외국인_SUM20", "개인_SUM20", "외국인_Z_60", "개인_Z_60",
        
        # 5. Relative Strength / Alpha
        "REL_RET_20", "ALPHA_20"
    ]

# ==============================================================================
# 2. Utility Functions (유틸리티)
# ==============================================================================
def setup_environment():
    """작업 디렉토리 초기화 및 정리"""
    print(f"[Init] 로컬 작업 경로 초기화: {Config.LOCAL_ROOT}")
    if os.path.exists(Config.LOCAL_ROOT):
        try:
            shutil.rmtree(Config.LOCAL_ROOT)
        except Exception as e:
            print(f"[Warn] 기존 폴더 삭제 실패 (무시됨): {e}")
            
    os.makedirs(Config.LOCAL_ROOT, exist_ok=True)

def copy_and_extract_data():
    """GDrive에서 원본 데이터를 가져와 로컬에 압축 해제"""
    src_path = os.path.join(Config.GDRIVE_ROOT, Config.INPUT_ZIP_NAME)
    dst_path = os.path.join(Config.LOCAL_ROOT, Config.INPUT_ZIP_NAME)
    extract_path = os.path.join(Config.LOCAL_ROOT, Config.INPUT_DIR_NAME)

    print(f"[IO] 데이터 복사 중... ({src_path} -> {dst_path})")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"원본 데이터가 없습니다: {src_path}\n run_preprocess.py가 성공적으로 완료되었는지 확인하세요.")

    shutil.copy2(src_path, dst_path)
    
    print(f"[IO] 압축 해제 중...")
    shutil.unpack_archive(dst_path, Config.LOCAL_ROOT)
    
    # 공간 확보를 위해 압축파일 즉시 삭제
    os.remove(dst_path) 
    
    # 압축 해제된 폴더 구조 확인 및 보정
    if not os.path.exists(extract_path):
        pass 
        
    return extract_path

def save_and_upload_results():
    """빈 폴더라도 강제로 생성하여 에러를 막고 업로드"""
    source_dir = os.path.join(Config.LOCAL_ROOT, Config.OUTPUT_DIR_NAME)
    output_zip_base = os.path.join(Config.LOCAL_ROOT, "rank_features_parquet") 
    final_local_zip = output_zip_base + ".zip"
    gdrive_dest = os.path.join(Config.GDRIVE_ROOT, Config.OUTPUT_ZIP_NAME)

    # 폴더가 없으면 강제로 생성
    if not os.path.exists(source_dir):
        print(f"\n[System] 랭크 결과 폴더가 없어 빈 폴더를 생성합니다: {source_dir}")
        os.makedirs(source_dir, exist_ok=True)
        # 빈 파일 하나라도 넣어둠
        with open(os.path.join(source_dir, "empty_result.txt"), "w") as f:
            f.write("No rank features generated.")

    print(f"\n[IO] 결과 압축 중... ({source_dir})")
    try:
        shutil.make_archive(
            base_name=output_zip_base, 
            format='zip', 
            root_dir=Config.LOCAL_ROOT, 
            base_dir=Config.OUTPUT_DIR_NAME
        )
        print(f"[IO] Google Drive 업로드 중... -> {gdrive_dest}")
        shutil.copy2(final_local_zip, gdrive_dest)
        print(f"✓ 업로드 완료.")
    except Exception as e:
        print(f"[ERROR] 압축 또는 업로드 실패: {e}")

# ==============================================================================
# 3. Core Logic (랭크 계산)
# ==============================================================================
def compute_rank_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    컬럼 매칭 상태를 정밀 진단합니다.
    """
    # 요청한 컬럼 중 실제로 존재하는 컬럼 확인
    valid_cols = [c for c in feature_cols if c in df.columns]
    
    # 디버깅: 왜 valid_cols가 비었는지 확인 (첫 호출 때만 상세 출력)
    if not hasattr(compute_rank_features, "_debug_printed"):
        print("\n" + "="*60)
        print("🔍 [DEBUG] Feature Column Inspection (First Batch)")
        print("="*60)
        
        # 1. Config에 있지만 데이터에 없는 것 (Missing)
        missing = set(feature_cols) - set(df.columns)
        # 2. 데이터에는 있지만 Config에 없는 것 (Extra - 중요 기술적 지표만)
        current = set(df.columns)
        
        print(f"1. Target Features (Config): {len(feature_cols)}개")
        print(f"2. Actual Data Columns: {len(df.columns)}개")
        print(f"3. Matched Columns (Valid): {len(valid_cols)}개")
        
        if missing:
            print(f"\n❌ [MISSING] 데이터에 없어서 계산 못하는 컬럼 ({len(missing)}개):")
            print(f"   {list(missing)[:10]} ...") 
            
            # 대소문자 문제인지 확인
            lower_current = {c.lower() for c in current}
            case_issues = [m for m in missing if m.lower() in lower_current]
            if case_issues:
                print(f"   💡 [HINT] 대소문자 불일치 의심 항목: {case_issues}")
                
        if valid_cols:
            print(f"\n✅ [OK] 정상 계산될 컬럼 예시: {valid_cols[:5]}")
        else:
            print("\n🚨 [CRITICAL] 계산 가능한 컬럼이 0개입니다! 랭크 데이터가 생성되지 않습니다.")
            # 데이터에 있는 컬럼 중 비슷한거라도 있는지 샘플 출력
            print(f"   (참고) 실제 데이터 컬럼 샘플: {list(df.columns)[:20]}")

        print("="*60 + "\n")
        compute_rank_features._debug_printed = True

    if not valid_cols:
        return pd.DataFrame()
        
    # 입력 데이터 확인
    if df.empty:
        print("    🚨 [CRITICAL] compute_rank_features에 빈 데이터프레임이 전달됨.")
        return pd.DataFrame()
        
    work_df = df.copy()
    
    if '날짜' not in work_df.columns or 'ticker' not in work_df.columns:
        work_df = work_df.reset_index()
        
    if '날짜' not in work_df.columns or 'ticker' not in work_df.columns:
        return pd.DataFrame()

    work_df['날짜'] = pd.to_datetime(work_df['날짜'])
    work_df = work_df.set_index(['날짜', 'ticker']).sort_index()

    target_df = work_df[valid_cols].replace([np.inf, -np.inf], np.nan)
    grp = target_df.groupby(level=0)

    rank_results = []
    
    for col in valid_cols:
        pct_rank = grp[col].rank(pct=True).astype('float32')
        pct_rank.name = f"R_{col}_PCT"
        
        quintile = np.ceil(pct_rank * 5).astype('float32')
        quintile = quintile.clip(lower=1.0, upper=5.0)
        quintile.name = f"R_{col}_Q5"
        
        rank_results.extend([pct_rank, quintile])

    if not rank_results:
        return pd.DataFrame()

    rank_df = pd.concat(rank_results, axis=1)
    rank_df = rank_df.reset_index()
    rank_df = rank_df.dropna(subset=['날짜'])
    rank_df = rank_df.set_index('날짜') 
    
    return rank_df

def process_by_year(input_dir: str, output_base_dir: str):
    """연도별 폴더를 순회하며 랭크 생성 및 저장"""
    
    # 연도 폴더 탐색 (숫자로 된 폴더만)
    year_dirs = sorted([
        d for d in os.listdir(input_dir) 
        if d.isdigit() and os.path.isdir(os.path.join(input_dir, d))
    ])
    
    print(f"[Process] 총 {len(year_dirs)}개 연도 데이터 처리 시작.")
    
    # 출력 폴더 미리 생성
    os.makedirs(output_base_dir, exist_ok=True)

    for year in tqdm(year_dirs, desc="Yearly Loop"):
        year_path = os.path.join(input_dir, year)
        parquet_files = glob.glob(os.path.join(year_path, "*.parquet"))
        
        if not parquet_files:
            continue
            
        # 1. 연도별 데이터 로드 (병합)
        try:
            # 필수 메타 컬럼
            meta_cols = ['날짜', 'ticker', 'Is_Tradable'] 
            # 요청할 전체 컬럼 (타겟 피처 + 메타)
            requested_cols = list(set(Config.TARGET_FEATURES + meta_cols))

            dfs = []
            for f in parquet_files:
                try:
                    # 1. 일단 전체 로드 (인덱스 포함)
                    df_part = pd.read_parquet(f)
                    
                    # 2. 인덱스(날짜)를 컬럼으로 끄집어내기
                    df_part = df_part.reset_index()

                    # 3. 컬럼명 표준화 (Date/Time/Index -> '날짜')
                    rename_map = {}
                    for c in df_part.columns:
                        if c.lower() in ['date', 'time', 'index']: 
                            rename_map[c] = '날짜'
                    df_part.rename(columns=rename_map, inplace=True)

                    # 4. 필수 컬럼 확인 (Ticker, 날짜)
                    if 'ticker' not in df_part.columns:
                        continue
                    
                    if '날짜' not in df_part.columns:
                        continue

                    # 5. 이제 필요한 컬럼만 남기기 (메모리 최적화)
                    existing_cols = set(df_part.columns)
                    
                    # keep_cols: (요청한 것 교집합 실제 있는 것) + (혹시 모를 ticker, 날짜 보장)
                    cols_to_keep = list(existing_cols.intersection(requested_cols))
                    
                    # 날짜, ticker는 필수니 한번 더 확인해서 리스트에 없으면 추가
                    for req in ['날짜', 'ticker', 'Is_Tradable']:
                        if req in df_part.columns and req not in cols_to_keep:
                            cols_to_keep.append(req)
                            
                    df_part = df_part[cols_to_keep].copy()

                    # 6. 요청했지만 파일에 없는 컬럼은 NaN 처리 (Schema Align)
                    for missing in set(requested_cols) - set(df_part.columns):
                        df_part[missing] = np.nan
                        
                    dfs.append(df_part)

                except Exception as e:
                    print(f"[Warn] 파일 읽기 실패: {f} - {e}")
                    continue

            if not dfs: continue
            
            df_year = pd.concat(dfs, ignore_index=True)

            print(f"\n🔍 [X-RAY DIAGNOSIS] {year}년도 데이터 정밀 진단 시작")
            
            if '날짜' in df_year.columns:
                # 1. 첫 번째 행의 실제 값 가져오기
                sample_val = df_year['날짜'].iloc[0]
                
                # 2. 타입과 원본 값 출력
                print(f"   1. [Raw Sample] 값: {sample_val!r}")
                print(f"   2. [Python Type] 타입: {type(sample_val)}")
                print(f"   3. [Pandas Dtype] 컬럼 타입: {df_year['날짜'].dtype}")
                
                # 3. 상위 5개 값 찍어보기
                print(f"   4. [Head 5 Values]:\n{df_year['날짜'].head(5).values}")

                # 4. 강제 변환 테스트
                print("   5. [Conversion Test] 단일 샘플 변환 시도...")
                try:
                    # 테스트 A: 기본 변환
                    res = pd.to_datetime(sample_val)
                    print(f"      -> (시도 A: 기본) 결과: {res} (Type: {type(res)})")
                except Exception as e:
                    print(f"      -> (시도 A: 실패) 에러: {e}")
                
                try:
                    # 테스트 B: 문자열 강제 변환 후 시도
                    res_str = pd.to_datetime(str(sample_val))
                    print(f"      -> (시도 B: str변환) 결과: {res_str}")
                except Exception as e:
                    print(f"      -> (시도 B: 실패) 에러: {e}")

            else:
                print("   🚨 [CRITICAL] '날짜' 컬럼이 데이터프레임에 없습니다!")
                print(f"   - 현재 컬럼 목록: {list(df_year.columns)}")
                print(f"   - 인덱스 이름: {df_year.index.name}")

            print("-" * 60)
            
            # 데이터 소멸 추적 로그
            cnt_raw = len(df_year)
            print(f"\n>>> [DEBUG] {year}년도 로드 직후 행 개수: {cnt_raw} rows")

            # 2. 날짜 변환 체크
            if '날짜' in df_year.columns:
                # (A) 이미 datetime 형식이면 건너뜀
                if not pd.api.types.is_datetime64_any_dtype(df_year['날짜']):
                    try:
                        # (B) YYYYMMDD 숫자/문자열 형식 우선 시도
                        df_year['날짜'] = pd.to_datetime(df_year['날짜'].astype(str), format='%Y%m%d', errors='raise')
                    except:
                        # (C) 실패 시 표준 파싱 시도 (YYYY-MM-DD 등)
                        df_year['날짜'] = pd.to_datetime(df_year['날짜'], errors='coerce')
                
                # NaT 체크
                nat_count = df_year['날짜'].isna().sum()
                if nat_count > 0:
                    print(f"    - [Warn] 날짜 변환 실패(NaT) 행: {nat_count}개 제거 (전체 {len(df_year)} 중)")
                    df_year = df_year.dropna(subset=['날짜'])
                
                if len(df_year) == 0:
                    print(f"    🚨 [CRITICAL] 날짜 변환 후 남은 데이터가 0개입니다! (날짜 포맷을 확인하세요)")
                    continue
                    
                df_year.set_index('날짜', inplace=True)
            
            df_year.sort_index(inplace=True) 
            
        except Exception as e:
            print(f"[Error] {year}년도 로드 중 오류: {e}")
            traceback.print_exc()
            continue
            
        # 2. 랭크 피처 계산
        rank_df = compute_rank_features(df_year, Config.TARGET_FEATURES)
        
        if rank_df.empty:
            continue

        # 3. 저장 (연도별 폴더 구조 유지)
        out_year_dir = os.path.join(output_base_dir, year)
        os.makedirs(out_year_dir, exist_ok=True)
        
        out_file = os.path.join(out_year_dir, f"rank_features_{year}.parquet")
        
        # 저장 직전 데이터 상태 확인
        if year == year_dirs[0]: 
            print(f"\n[DEBUG] {year}년도 저장 데이터 검증:")
            print(f"   - Reset 전 Index Name: {rank_df.index.name}")
            print(f"   - Reset 전 Columns: {list(rank_df.columns[:3])} ...")
            
            # Reset Index 시뮬레이션
            temp_df = rank_df.reset_index()
            print(f"   - Reset 후 Columns: {list(temp_df.columns)}")
            
            # '날짜' 컬럼 존재 여부 확실하게 체크
            if '날짜' in temp_df.columns:
                print("   ✅ 검증 성공: '날짜'가 컬럼으로 존재함.")
                print(f"   - 날짜 샘플: {temp_df['날짜'].iloc[0]}")
            else:
                print("   🚨 검증 실패: '날짜' 컬럼이 보이지 않음!")
                
            del temp_df

        # 압축 저장 (snappy 권장)
        # 인덱스를 리셋하여 '날짜'를 명시적인 컬럼으로 저장 (index=False)
        rank_df.reset_index().to_parquet(out_file, compression='snappy', index=False)
        
        # 메모리 정리
        del df_year, rank_df, dfs
        gc.collect()

# ==============================================================================
# 4. Main Execution
# ==============================================================================
def main():
    start_time = time.time()
    print("="*60)
    print(">>> [Rank Engine] Cross-Sectional Rank Feature Generator Start")
    print("="*60)

    try:
        # 1. 환경 초기화
        setup_environment()

        # 2. 데이터 가져오기 (GDrive -> Local)
        extract_path = copy_and_extract_data()
        
        # 명확한 타겟 디렉토리 우선 탐색 및 Depth 제한
        input_root = extract_path
        found = False
        
        # 1. 최상위 체크
        if any(d.isdigit() and len(d)==4 and os.path.isdir(os.path.join(input_root, d)) for d in os.listdir(input_root)):
            pass 
            found = True
        else:
            # 2. INPUT_DIR_NAME 바로 아래 체크
            candidate = os.path.join(input_root, Config.INPUT_DIR_NAME)
            if os.path.exists(candidate) and any(d.isdigit() and len(d)==4 for d in os.listdir(candidate)):
                input_root = candidate
                found = True
            else:
                # 3. 그래도 없으면 1-depth만 walk (너무 깊게 들어가지 않도록)
                for root, dirs, files in os.walk(input_root):
                    # 숨김 폴더(.으로 시작) 건너뛰기
                    dirs[:] = [d for d in dirs if not d.startswith('.')] 
                    
                    valid_years = [d for d in dirs if d.isdigit() and len(d)==4]
                    if valid_years:
                        # 연도 폴더 안에 실제 parquet 파일이 있는지 더블 체크
                        sample_year = os.path.join(root, valid_years[0])
                        if glob.glob(os.path.join(sample_year, "*.parquet")):
                            input_root = root
                            found = True
                            break
                
                if not found:
                    raise FileNotFoundError(f"연도별 데이터 폴더를 {extract_path} 하위에서 찾을 수 없습니다.")
        
        print(f"[Path] Resolved Input Data Root: {input_root}")

        # 3. 랭크 피처 생성 및 로컬 저장
        output_base_dir = os.path.join(Config.LOCAL_ROOT, Config.OUTPUT_DIR_NAME)
        process_by_year(input_root, output_base_dir)
        
        # 4. 결과 업로드 (Local -> GDrive)
        save_and_upload_results()
        
        elapsed = time.time() - start_time
        print("="*60)
        print(f">>> [Success] 모든 작업 완료. 소요시간: {elapsed:.1f}초")
        print(f">>> 출력 파일: {os.path.join(Config.GDRIVE_ROOT, Config.OUTPUT_ZIP_NAME)}")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f">>> [Critical Error] 작업 중단: {e}")
        traceback.print_exc()
        print("="*60)

if __name__ == "__main__":
    main()