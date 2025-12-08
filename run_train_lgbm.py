#!/usr/bin/env python3

import os
import glob
import time
import json
import pickle
import gc
import shutil
import traceback
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, log_loss, precision_score, recall_score, f1_score
import joblib
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuration & Hyperparameters

# 경로 설정
SOURCE_DATA_PATH = "/content/drive/MyDrive/rre/data"
LOCAL_DATA_PATH = "/content/rre_local_data_lgbm"
MODEL_OUTPUT_PATH = "/content/drive/MyDrive/rre/model_lgbm" # LightGBM 전용 폴더
RANK_FEATURES_DIR_NAME = "rank_features_parquet"

# 랭크 피처 파일명 정의
RANK_FILENAME = "rank_features_parquet.zip"

DATA_FILENAME = "processed_data.pkl"
MODEL_FILENAME = "best_lgbm.txt"
SCALER_FILENAME = "scaler_lgbm.pkl"

# 기간 설정
TRAIN_END_DATE = '2023-12-30'
VAL_END_DATE = '2025-01-01'

# Liquid Sharpe 설정 (학습 데이터 노이즈 필터링용)
MAX_LABELS_PER_DAY = int(os.getenv("MAX_LABELS_PER_DAY", "200"))
MIN_LIQUIDITY_THRESHOLD = float(os.getenv("MIN_LIQUIDITY_THRESHOLD", str(3_000_000_000)))
RAW_PRICE_COLS = {"open", "high", "low", "close", "volume"}

# LightGBM Hyperparameters
# [수정] 표준 파라미터로 리셋
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,       # [Reset] 표준 속도
    'num_leaves': 31,            # [Reset] 표준 복잡도
    'max_depth': -1,             # [Reset] 깊이 제한 해제
    'min_child_samples': 20,     # [Reset] 표준 민감도
    'subsample': 0.8,
    'colsample_bytree': 0.8,     # [Reset] 정보량 증가
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
    'verbose': -1,
    'seed': 42,
    # scale_pos_weight는 사용하지 않음 (Threshold Tuning으로 대체)
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100

# 2. Utility Functions

def copy_data_to_local(source_path, local_path, data_filename):
    """GDrive의 ZIP 파일을 로컬로 가져와 고속 해제"""
    print(f"[Phase 1] 📦 데이터(Zip) 로컬 복사 및 해제 프로세스 시작...")
    
    name, _ = os.path.splitext(data_filename)
    dir_name = f"{name}_parquet"
    archive_name = f"{dir_name}.zip"
    
    source_archive = os.path.join(source_path, archive_name)
    local_archive = os.path.join(local_path, archive_name)
    final_local_dir = os.path.join(local_path, dir_name)

    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path, exist_ok=True)

    if not os.path.exists(source_archive):
        print(f"[Error] 원본 데이터가 없습니다: {source_archive}")
        return None

    try:
        shutil.copy2(source_archive, local_archive)
        shutil.unpack_archive(local_archive, local_path)
        os.remove(local_archive)
        print(f"✓ 데이터 준비 완료: {final_local_dir}")
        return final_local_dir
    except Exception as e:
        print(f"[Fatal] 데이터 복사/해제 실패: {e}")
        return None

def load_rank_features(rank_root_dir, date_min=None, date_max=None):
    """랭크 피처 로드 및 날짜 컬럼 표준화"""
    if rank_root_dir is None or not os.path.isdir(rank_root_dir):
        return None

    file_list = sorted(glob.glob(os.path.join(rank_root_dir, "**", "*.parquet"), recursive=True))
    if not file_list:
        return None

    print(f"[Rank] Parquet 로딩 중... ({len(file_list)} files)")
    parts = []

    for fp in file_list:
        try:
            part = pd.read_parquet(fp)
        except Exception:
            continue

        if part.empty: continue

        # 날짜 컬럼 표준화
        if isinstance(part.index, pd.DatetimeIndex):
            part.index.name = '날짜'
            part = part.reset_index()
        
        rename_map = {}
        for col in part.columns:
            if col.lower() in ['date', 'time', 'index']:
                rename_map[col] = '날짜'
        if rename_map:
            part.rename(columns=rename_map, inplace=True)

        # [DEBUG START 1] 컬럼 확인
        if '날짜' not in part.columns:
            print(f"   [SKIP] {os.path.basename(fp)} -> '날짜' 컬럼 없음. (보유 컬럼: {list(part.columns)})")
            continue

        part['날짜'] = pd.to_datetime(part['날짜'], errors='coerce')
        
        # [DEBUG START 2] 날짜 변환 직후 확인
        if part['날짜'].isna().all():
            print(f"   [SKIP] {os.path.basename(fp)} -> 날짜 변환 실패 (All NaT). 원본 샘플: {part.index[:3] if not part.index.empty else 'Empty'}")
            continue

        if pd.api.types.is_datetime64_any_dtype(part['날짜']):
            # Timezone 제거 (매우 중요)
            part['날짜'] = part['날짜'].dt.tz_localize(None)
            
        part = part.dropna(subset=['날짜'])

        # [DEBUG START 3] 필터링 전후 비교
        rows_before = len(part)
        
        # 안전한 비교를 위해 date_min/max도 timezone 제거
        if date_min and date_min.tzinfo is not None: date_min = date_min.tz_localize(None)
        if date_max and date_max.tzinfo is not None: date_max = date_max.tz_localize(None)

        if date_min: 
            part = part[part['날짜'] >= date_min]
        if date_max: 
            part = part[part['날짜'] <= date_max]

        if part.empty:
            print(f"   [DROP] {os.path.basename(fp)} -> 날짜 필터링 후 0건 됨.")
            print(f"          (File Range: {part['날짜'].min()} ~ {part['날짜'].max()} / Target: {date_min} ~ {date_max})")
            # [CRITICAL DEBUG] 왜 안 맞는지 샘플 출력 (필터링 전 데이터가 있다면)
            if rows_before > 0:
                 print(f"          (Type Check - File: {part['날짜'].dtype}, Target: {type(date_min)})")
        else:
            print(f"   [KEEP] {os.path.basename(fp)} -> {len(part)} 행 유지됨.")
            parts.append(part)

    if not parts: return None

    rank_df = pd.concat(parts, ignore_index=True)
    rank_df.drop_duplicates(subset=['날짜', 'ticker'], keep='last', inplace=True)
    return rank_df

def merge_rank_features_with_main(df_main, rank_root_dir):
    """메인 df에 랭크 피처 병합 (MultiIndex 활용 최적화)"""
    if rank_root_dir is None:
        return df_main
        
    if df_main.empty or not os.path.isdir(rank_root_dir):
        return df_main

    date_min = df_main.index.min()
    date_max = df_main.index.max()

    print(f"[Rank] 메인 df와 랭크 피처 병합 시작 ({date_min.date()} ~ {date_max.date()})")

    rank_df = load_rank_features(rank_root_dir, date_min, date_max)
    if rank_df is None or rank_df.empty:
        return df_main

    # 효율적인 병합을 위한 Index 기반 Join
    
    # 1. Main DF 준비 (Index: 날짜, ticker)
    df_main_mi = df_main.reset_index().set_index(['날짜', 'ticker']).sort_index()
    
    # 2. Rank DF 준비
    rank_df['ticker'] = rank_df['ticker'].astype(str)
    rank_df = rank_df.set_index(['날짜', 'ticker']).sort_index()
    
    # 3. Join (Index 기반)
    new_rank_cols = list(set(rank_df.columns) - set(df_main_mi.columns))
    
    # [DEBUG] 병합 전 키 정합성 체크
    print(f"\n[DEBUG] Rank Merge 진단 시작")
    print(f"   - Main Index Sample: {df_main_mi.index[:3]}")
    print(f"   - Rank Index Sample: {rank_df.index[:3]}")
    
    # 교집합 개수 확인 (매우 중요)
    common_idx = df_main_mi.index.intersection(rank_df.index)
    print(f"   - Main Rows: {len(df_main_mi)}, Rank Rows: {len(rank_df)}")
    print(f"   - ⚡ 교차되는(매칭 성공) 행 개수: {len(common_idx)} ({(len(common_idx)/len(df_main_mi)*100):.1f}%)")
    
    if len(common_idx) == 0:
        print("   🚨 [CRITICAL WARNING] 병합되는 행이 0개입니다! 날짜 포맷이나 Ticker 타입을 확인하세요.")
        # 타입 강제 일치 시도 로그
        print(f"   - Main Index Types: {df_main_mi.index.get_level_values(0).dtype}, {df_main_mi.index.get_level_values(1).dtype}")
        print(f"   - Rank Index Types: {rank_df.index.get_level_values(0).dtype}, {rank_df.index.get_level_values(1).dtype}")

    if new_rank_cols:
        df_merged = df_main_mi.join(rank_df[new_rank_cols], how='left')
        
        # [DEBUG] 병합 후 데이터 확인
        nan_ratio = df_merged[new_rank_cols[0]].isna().mean()
        print(f"[Rank] 랭크 병합 완료. 추가된 컬럼 수: {len(new_rank_cols)}")
        print(f"   - 대표 랭크 컬럼({new_rank_cols[0]}) NaN 비율: {nan_ratio*100:.2f}%")
        if nan_ratio > 0.9:
             print("   🚨 [WARNING] 랭크 피처의 90% 이상이 NaN입니다. 병합이 잘못되었을 가능성이 큽니다.")
    else:
        print("[Rank] 추가할 랭크 컬럼이 없습니다.")
        df_merged = df_main_mi

    # 4. 최종 형태 복원 ('날짜'만 인덱스로)
    df_merged = df_merged.reset_index().set_index('날짜').sort_index()
    
    df_merged['ticker'] = df_merged['ticker'].astype('category')

    return df_merged

def load_data_from_parquet(local_base_path, rank_root=None):
    """Parquet 파일 일괄 로드 및 Timezone 처리"""
    print(f"[Phase 2] 로컬 Parquet 파일 로딩...")
    directory = os.path.dirname(local_base_path)
    name, _ = os.path.splitext(os.path.basename(local_base_path))
    parquet_root_dir = os.path.join(directory, f"{name}_parquet")

    file_list = sorted(glob.glob(os.path.join(parquet_root_dir, "**", "*.parquet"), recursive=True))
    if not file_list:
        return None, None

    try:
        # Fast Path: Arrow
        table = pq.read_table(file_list, use_threads=True, use_pandas_metadata=True)
        df = table.to_pandas()
    except Exception as e:
        # Fallback 전 원인 분석 출력
        print("\n[DEBUG] Arrow load failed! 상세 에러 로그:")
        print(traceback.format_exc())

        # 스키마 불일치 확인을 위한 긴급 점검
        print("[DEBUG] 파일별 스키마 불일치 점검 시작...")
        try:
            base_schema = pq.read_schema(file_list[0])
            base_cols = set(base_schema.names)
            print(f"   - 기준 파일: {os.path.basename(file_list[0])} (컬럼 {len(base_cols)}개)")
            
            for f in file_list[1:]:
                curr_schema = pq.read_schema(f)
                curr_cols = set(curr_schema.names)
                if curr_cols != base_cols:
                    print(f"   🚨 [MISMATCH] {os.path.basename(f)}")
                    print(f"      - 누락: {base_cols - curr_cols}")
                    print(f"      - 추가: {curr_cols - base_cols}")
                    break
        except Exception as inspect_err:
            print(f"   - 점검 중 에러 발생: {inspect_err}")

        print("Falling back to pandas concat... (속도가 느릴 수 있습니다)")
        dfs = [pd.read_parquet(f) for f in file_list]
        df = pd.concat(dfs, ignore_index=False)

    # 1. 인덱스가 DatetimeIndex라면 컬럼으로 끄집어냄
    if isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "날짜" # 이름 강제 지정
        df = df.reset_index()
    
    # [수정] 2. 모든 컬럼명을 소문자로 변환하여 표준화 (Raw Price 누수 방지)
    df.columns = [c.lower() for c in df.columns]

    # [수정] 3. 'index', 'date', 'time' 등을 '날짜'로 표준화
    rename_map = {}
    for col in df.columns:
        if col in ['index', 'date', 'time']:
            rename_map[col] = '날짜'
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # 4. 최종 확인
    if "날짜" not in df.columns:
        raise ValueError("Critical: '날짜' 컬럼을 찾을 수 없습니다. 전처리 데이터를 확인하세요.")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    
    # Timezone 정보가 있다면 제거하여 Rank 데이터와 병합 호환성 보장
    if pd.api.types.is_datetime64_any_dtype(df["날짜"]):
        if getattr(df["날짜"].dt, "tz", None) is not None:
             df["날짜"] = df["날짜"].dt.tz_localize(None)

    df = df.dropna(subset=["날짜"])
    df.set_index("날짜", inplace=True)
    df.sort_index(inplace=True)

    # [Optimization] 1. 먼저 거래 가능 종목만 남김 (메모리 확보 및 불필요한 병합 방지)
    if "is_tradable" in df.columns:
        # Is_Tradable은 이제 역할이 끝났으므로 drop
        df.drop(columns=['is_tradable'], inplace=True)

    # [Optimization] 2. 필터링된 데이터프레임에 대해서만 랭크 피처 병합
    df = merge_rank_features_with_main(df, rank_root)

    # [Fix] Feature Columns 정의 (메타데이터 및 Raw Data 제외)
    # 이미 df.columns를 소문자로 바꿨다고 가정.
    # 혹시 모를 대소문자 이슈를 방지하기 위해 set도 소문자로 구성
    exclude_cols = {"날짜", "ticker", "label", "is_tradable", "trading_missing_flag"} 
    raw_prices = {c.lower() for c in RAW_PRICE_COLS}
    base_cols_set = exclude_cols | raw_prices
    
    feature_cols = sorted([c for c in df.columns if c.lower() not in base_cols_set])
    
    return df, feature_cols

def apply_liquid_sharpe_label(df, max_pos_per_day, min_amt, label_col="label"):
    """
    [Liquid Sharpe] Train Set 노이즈 필터링
    - 미래 5일 수익/변동성 비율(Sharpe) 상위 K개만 label=1 유지
    - 나머지는 0으로 강제
    """
    print(f"[LiquidSharpe] 필터링 시작 (Top {max_pos_per_day}, MinAmt {min_amt/1e8:.1f}억)")
    
    # 필수 컬럼 체크
    req = {"close", "open", "volume", label_col, "ticker"}
    if not req.issubset(df.columns):
        return df

    df = df.sort_index().copy()
    df["_row_id"] = np.arange(len(df)) # 고유 ID

    # 거래대금 계산
    if "거래대금" in df.columns:
        amt_col = "거래대금"
    else:
        amt_col = "_amt_proxy"
        df[amt_col] = df["close"] * df["volume"]

    grp = df.groupby("ticker", sort=False)
    
    # 1. 유동성 (5일 평균 거래대금)
    df["_amt_ma5"] = grp[amt_col].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # 2. 미래 5일 수익률
    f_open_t1 = grp["open"].shift(-1)
    f_close_t5 = grp["close"].shift(-20) # holding days(20) 반영
    df["_fut_ret"] = (f_close_t5 - f_open_t1) / (f_open_t1 + 1e-9)

    # 3. 미래 5일 변동성 (일간 수익률 std)
    df["_daily_ret"] = grp["close"].pct_change()
    df["_fut_vol"] = grp["_daily_ret"].transform(
        lambda x: x.rolling(20, min_periods=10).std().shift(-20)
    )

    # 4. Score
    df["_score"] = df["_fut_ret"] / (df["_fut_vol"].abs() + 1e-4)

    # 5. Filter
    fail_liq = df["_amt_ma5"] < min_amt
    fail_fut = df["_fut_ret"].isna()
    
    # 기존 Label=1 중 살릴 것 선별
    pos_mask = df[label_col] == 1.0
    candidate_mask = pos_mask & (~fail_liq) & (~fail_fut)
    
    # 전체 0 초기화
    df[label_col] = 0.0
    
    if candidate_mask.sum() > 0:
        # .copy() 추가하여 SettingWithCopyWarning 방지
        cand_df = df.loc[candidate_mask, ["_score", "_row_id"]].copy()
        
        cand_df["_day_rank"] = cand_df.groupby(cand_df.index)["_score"].rank(ascending=False, method="first")
        
        keep_ids = cand_df.loc[cand_df["_day_rank"] <= max_pos_per_day, "_row_id"]
        df.loc[df["_row_id"].isin(keep_ids), label_col] = 1.0

    # 임시 컬럼 제거
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=drop_cols, inplace=True)
    
    return df

def run_causal_impute(df, feature_cols):
    """Iterative Causal Imputation (FFill + Expanding Median)"""
    print(f"   - [Info] 결측치 보정 (Causal Imputation) 수행 중... (전체 데이터 대상)")
    
    # [Optimization] 1. 속도 최적화를 위해 Ticker, 날짜 순으로 미리 정렬 (View 활용)
    # 날짜가 인덱스라고 가정
    df = df.sort_values(by=['ticker', '날짜']).copy()

    # Ticker 별로 그룹화
    # group_keys=False로 설정하여 불필요한 인덱스 레벨 생성을 막음
    try:
        grouped = df.groupby('ticker', observed=True, group_keys=False)
    except TypeError:
        grouped = df.groupby('ticker', group_keys=False)

    chunks = []
    for _, group in tqdm(grouped, desc="Imputing", leave=False):
        # 이미 정렬되었으므로 sort_index() 불필요
        
        # 1. Forward Fill
        # (SettingWithCopyWarning 방지를 위해 할당 방식 주의, 하지만 여기선 chunks로 모으므로 안전)
        g_filled = group[feature_cols].ffill()
        
        # 2. Expanding Median (Shift 1)
        if g_filled.isna().any().any():
            medians = g_filled.expanding(min_periods=1).median().shift(1)
            g_filled = g_filled.fillna(medians)
            
        # 원본 그룹의 피처 컬럼 교체
        group[feature_cols] = g_filled
        chunks.append(group)
    
    if not chunks: return df
    
    df_imputed = pd.concat(chunks)
    
    # [DEBUG] Imputation 효과 측정
    nan_before_zero = df_imputed[feature_cols].isna().sum().sum()
    total_cells = df_imputed[feature_cols].size
    
    # 3. 남은 NaN은 0으로 채움
    df_imputed[feature_cols] = df_imputed[feature_cols].fillna(0.0)
    
    # [DEBUG] 결과 리포트
    print(f"\n[DEBUG] Causal Imputation 리포트")
    print(f"   - FFill/Median 이후 남은 NaN: {nan_before_zero}개 ({(nan_before_zero/total_cells*100):.2f}%) -> 0.0으로 대체됨")
    
    # 특정 주요 피처의 0.0 비율 확인 (너무 높으면 데이터 품질 의심)
    check_col = feature_cols[0] if feature_cols else None
    if check_col:
        zero_cnt = (df_imputed[check_col] == 0).sum()
        print(f"   - '{check_col}' 컬럼의 0.0 값 비율: {(zero_cnt/len(df_imputed)*100):.2f}%")

    # [Critical] 시계열 모델링을 위해 다시 '날짜' 순으로 정렬 보장
    df_imputed.sort_index(inplace=True)
    
    return df_imputed

def preprocess_and_split(df, feature_cols):
    print("[Phase 3] 데이터 분할 및 스케일링...")

    # 1. Split (Train/Val/Test)
    df_train = df.loc[df.index <= TRAIN_END_DATE].copy()
    df_val = df.loc[(df.index > TRAIN_END_DATE) & (df.index <= VAL_END_DATE)].copy()
    df_test = df.loc[df.index > VAL_END_DATE].copy()

    # 2. Liquid Sharpe (Train Only)
    print("   - [Train Set] Liquid Sharpe 라벨 필터링 적용")
    # df_train = apply_liquid_sharpe_label(
    #     df_train, MAX_LABELS_PER_DAY, MIN_LIQUIDITY_THRESHOLD
    # )

    # 3. Label NaN 제거
    for name, d in zip(['Train', 'Val', 'Test'], [df_train, df_val, df_test]):
        before_len = len(d)
        d.dropna(subset=['label'], inplace=True)
        
        # [DEBUG] 세트별 상세 통계
        pos_cnt = (d['label'] == 1).sum()
        print(f"\n[DEBUG] {name} Set Status:")
        print(f"   - 기간: {d.index.min()} ~ {d.index.max()}")
        print(f"   - 데이터 수: {len(d)} (Label NaN 제거: {before_len - len(d)})")
        print(f"   - Positive Label(1): {pos_cnt} ({(pos_cnt/len(d)*100):.2f}%)")
        
        if pos_cnt < 10:
            print(f"   🚨 [CRITICAL] {name} 세트에 Positive Label이 너무 적습니다! 모델이 0만 예측할 것입니다.")

    # 4. Scaling 적용 (RobustScaler)
    print("\n   - Scaling 적용 (RobustScaler) 진행 중...")
    
    scaler = RobustScaler()
    
    # 결과값을 명시적으로 float32로 캐스팅하여 메모리 절약 (float64 -> float32)
    df_train[feature_cols] = scaler.fit_transform(
        df_train[feature_cols].astype(np.float32)
    ).astype(np.float32)
    
    df_val[feature_cols] = scaler.transform(
        df_val[feature_cols].astype(np.float32)
    ).astype(np.float32)
    
    df_test[feature_cols] = scaler.transform(
        df_test[feature_cols].astype(np.float32)
    ).astype(np.float32)

    return df_train, df_val, df_test, scaler

# 3. Main Execution

def main():
    # 전역 변수 의존성 제거

    start_time = time.time()
    
    # 1. 데이터 준비 (메인 데이터 복사 및 초기화)
    local_data_path = copy_data_to_local(SOURCE_DATA_PATH, LOCAL_DATA_PATH, DATA_FILENAME)
    if not local_data_path: return

    # --- 랭크 데이터 경로 확정 (Local Variable 사용) ---
    rank_features_path = None  # 기본값
    rank_zip_src = os.path.join(SOURCE_DATA_PATH, RANK_FILENAME)
    rank_zip_dst = os.path.join(LOCAL_DATA_PATH, RANK_FILENAME)
    
    if os.path.exists(rank_zip_src):
        print(f"[Phase 1-2] 랭크 피처 데이터 복사 및 해제... ({RANK_FILENAME})")
        try:
            shutil.copy2(rank_zip_src, rank_zip_dst)
            shutil.unpack_archive(rank_zip_dst, LOCAL_DATA_PATH) 
            os.remove(rank_zip_dst) 
            
            # [Fix] 압축 해제 후 실제 parquet 파일이 있는 폴더 찾기 (Robust Path Finding)
            potential_path = os.path.join(LOCAL_DATA_PATH, RANK_FEATURES_DIR_NAME)
            if os.path.exists(potential_path) and glob.glob(os.path.join(potential_path, "**", "*.parquet"), recursive=True):
                rank_features_path = potential_path
            else:
                # 폴더명이 다를 경우를 대비해 parquet이 있는 최상위 폴더 검색
                found_parquet = glob.glob(os.path.join(LOCAL_DATA_PATH, "**", "*.parquet"), recursive=True)
                if found_parquet:
                    rank_features_path = os.path.dirname(found_parquet[0])
                    # 연도별 폴더 구조일 경우 상위 폴더 지정
                    if os.path.basename(rank_features_path).isdigit():
                        rank_features_path = os.path.dirname(rank_features_path)
                else:
                    rank_features_path = None
            
            if rank_features_path:
                print(f"✓ 랭크 피처 준비 완료: {rank_features_path}")
            else:
                print(f"[Warn] 랭크 압축 해제 성공했으나 Parquet 파일을 찾을 수 없음.")

        except Exception as e:
            print(f"[Warn] 랭크 데이터 처리 중 오류 발생: {e}")
            rank_features_path = None
    else:
        print(f"[Info] 랭크 피처 파일({RANK_FILENAME})이 없습니다. 랭크 병합을 건너뜁니다.")
        # rank_features_path는 이미 None
    # ------------------------------------------

    local_data_file = os.path.join(LOCAL_DATA_PATH, DATA_FILENAME)
    
    # 명시적 인자 전달
    df, feature_cols = load_data_from_parquet(local_data_file, rank_root=rank_features_path)
    
    if df is None: return

    # 전체 데이터에 대해 먼저 결측치 보정 수행 (연속성 보장)
    print("[Info] 전체 데이터 Causal Imputation 수행 중...")
    df = run_causal_impute(df, feature_cols)

    # Ticker 범주형 변환도 전체 데이터에서 수행 (매핑 일치 보장)
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype('category')

    # 2. 분할 및 스케일링
    df_train, df_val, df_test, scaler = preprocess_and_split(df, feature_cols)
    
    # 메모리 정리
    del df
    gc.collect()

    print(f"[Info] 데이터 준비 완료:")
    print(f"   - Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
    print(f"   - Features: {len(feature_cols)}")

    # [DEBUG] 학습 직전 데이터 건전성 체크 (Health Check)
    print("\n[DEBUG] Final Data Health Check (Train Set)")
    
    # 무한대값 체크
    X_sample = df_train[feature_cols]
    inf_count = np.isinf(X_sample).sum().sum()
    if inf_count > 0:
        print(f"   🚨 [CRITICAL] Train Set에 무한대(Inf) 값이 {inf_count}개 있습니다! 학습이 터질 수 있습니다.")
        # 임시 조치: Clip
        df_train[feature_cols] = df_train[feature_cols].clip(-1e9, 1e9)
        df_val[feature_cols] = df_val[feature_cols].clip(-1e9, 1e9)
        print("   -> +/- 1e9 로 Clipping 적용함.")

    # 피처 스케일 확인 (RobustScaler가 정상 작동했는지)
    max_val = X_sample.max().max()
    min_val = X_sample.min().min()
    print(f"   - Feature Max Value: {max_val:.2f}")
    print(f"   - Feature Min Value: {min_val:.2f}")
    if abs(max_val) > 1000 or abs(min_val) > 1000:
        print("   ⚠️ [WARNING] 스케일링 후에도 값이 매우 큽니다. Outlier가 잡히지 않았을 수 있습니다.")

    # 3. LightGBM 데이터셋 생성

    # [수정] ticker 피처 제거
    # 이유 1: ticker가 피처에 포함되면 모델이 종목의 고유한 패턴(기술적 지표)이 아닌,
    #        '종목 코드 자체(이름)'를 외워서 점수를 매기는 과적합(Overfitting)이 발생함.
    # 이유 2: 과거에 잘 올랐던 종목이 미래에도 오른다는 보장이 없으므로,
    #        종목 식별자는 학습에서 배제하고 오직 팩터(Features)로만 승부해야 함.
    
    cat_feats = [] # 범주형 변수 리스트를 비움
    
    print(f"   - [Info] 학습 피처에서 'ticker' 제외 완료. 순수 팩터 개수: {len(feature_cols)}")

    # [수정] 데이터셋 생성 시 'feature_cols'만 사용 (cat_feats 더하기 제거)
    train_ds = lgb.Dataset(
        df_train[feature_cols], # <-- 여기에 + cat_feats를 제거하여 ticker가 들어가는 것을 차단
        label=df_train['label']
        # categorical_feature=cat_feats # <-- ticker가 없으므로 이 옵션도 제거하거나 빈 리스트 전달
    )
    val_ds = lgb.Dataset(
        df_val[feature_cols],   # <-- 여기도 동일하게 제거
        label=df_val['label'], 
        reference=train_ds
        # categorical_feature=cat_feats # <-- 제거
    )

    # 4. 학습
    print("\n[Phase 4] LightGBM 모델 학습 시작...")
    model = lgb.train(
        LGBM_PARAMS,
        train_ds,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_ds, val_ds],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=VERBOSE_EVAL)
        ]
    )

    # 5. 저장
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    model_save_path = os.path.join(MODEL_OUTPUT_PATH, MODEL_FILENAME)
    model.save_model(model_save_path)
    print(f"✓ 모델 저장 완료: {model_save_path}")

    if scaler is not None:
        scaler_save_path = os.path.join(MODEL_OUTPUT_PATH, SCALER_FILENAME)
        with open(scaler_save_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"✓ 스케일러 저장 완료: {scaler_save_path}")
    else:
        print("✓ 스케일러 저장 생략 (사용하지 않음)")

    manifest = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "model_type": "lightgbm",
        "created_at": pd.Timestamp.utcnow().isoformat()
    }
    with open(os.path.join(MODEL_OUTPUT_PATH, "features.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 6. 평가 (Test Set)
    print("\n[Phase 5] Test Set 최종 평가 (Threshold Tuning)...")
    
    # [Step A] Validation Set으로 최적의 임계값 찾기
    print("   - Finding best threshold using Validation Set...")
    # [수정] 평가 시에도 ticker 제외
    X_val = df_val[feature_cols]
    y_val = df_val['label']
    y_val_prob = model.predict(X_val)
    
    best_thr = 0.5
    best_score = 0.0
    
    # 0.1 ~ 0.6 까지 0.01 단위로 스캔
    for thr in np.arange(0.1, 0.61, 0.01):
        pred_lbl = (y_val_prob >= thr).astype(int)
        # F1 Score가 가장 높은 지점을 선택 (상황에 따라 Precision을 기준으로 할 수도 있음)
        score = f1_score(y_val, pred_lbl, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_thr = thr
            
    print(f"   -> Found Best Threshold: {best_thr:.2f} (Val F1: {best_score:.4f})")
    print(f"      (Score Stats - Min: {y_val_prob.min():.4f}, Mean: {y_val_prob.mean():.4f}, Max: {y_val_prob.max():.4f})")

    # [Step B] 찾은 임계값으로 Test Set 평가
    # [수정] 평가 시에도 ticker 제외
    X_test = df_test[feature_cols]
    y_test = df_test['label']
    
    y_pred_prob = model.predict(X_test)
    
    # [수정] 0.5가 아니라 best_thr를 기준으로 자름
    y_pred_label = (y_pred_prob >= best_thr).astype(int)

    # 각종 지표 계산
    auc_score = roc_auc_score(y_test, y_pred_prob)
    loss_val = log_loss(y_test, y_pred_prob)          # Log Loss (손실값)
    precision = precision_score(y_test, y_pred_label) # 정밀도
    recall = recall_score(y_test, y_pred_label)       # 재현율
    f1 = f1_score(y_test, y_pred_label)               # F1 Score

    print(f"=== Test Results ===")
    print(f"ROC AUC   : {auc_score:.4f}")
    print(f"Log Loss  : {loss_val:.4f}")   # Loss 출력
    print(f"Precision : {precision:.4f}")  # 정밀도 출력
    print(f"Recall    : {recall:.4f}")     # 재현율 출력
    print(f"F1 Score  : {f1:.4f}")

    # 상세 분류 리포트 출력
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred_label))
    
    print("\n[Info] Feature Importance 저장 중...")
    ax = lgb.plot_importance(model, max_num_features=20, importance_type='gain', figsize=(10, 8), title='LightGBM Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_PATH, "feature_importance.png"))
    plt.close()

    if os.path.exists(LOCAL_DATA_PATH):
        shutil.rmtree(LOCAL_DATA_PATH)
        print(f"\n[Clean Up] 로컬 임시 데이터 삭제 완료.")

    print(f"총 소요 시간: {time.time() - start_time:.2f}초")

if __name__ == "__main__":
    main()