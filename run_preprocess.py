#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import pandas_ta as ta
from tqdm import tqdm
import time
import json, hashlib
import re
from contextlib import suppress
import datetime
from datetime import datetime as _dt, UTC, timezone
import sys  # 미래데이터 누수 감지 시 강제 종료용
import traceback

import numba
import multiprocessing
import shutil
from functools import partial

# Pandas 향후 동작(다운캐스트) 옵션: 경고 억제/호환성
try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    pass

import gc

try:
    from debug_utils import (setup_logger, log_env, ensure_dir, dassert, df_quick_report,
                             nan_inf_audit, write_json, write_csv, log_mem, feature_signature, save_hist, check_time_splits)
except Exception:
    # Minimal safe fallbacks (필요 함수만 간단히 대체)
    class _Log:
        def info(self, m): print("[Info]", m)
        def warning(self, m): print("[Warn]", m)
        def error(self, m): print("[Error]", m)
    def setup_logger(*a, **k): return _Log()
    def log_env(*a, **k): pass
    def ensure_dir(p): os.makedirs(p, exist_ok=True)
    def dassert(c, msg, log=None):
        if not c:
            if log: log.error(msg)
            raise AssertionError(msg)
    def df_quick_report(*a, **k): pass
    def nan_inf_audit(*a, **k): return {}
    def write_json(path, obj):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import json; json.dump(obj, open(path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    def write_csv(*a, **k): pass
    def log_mem(*a, **k): pass
    def feature_signature(cols):
        import hashlib; return hashlib.md5(",".join(sorted(map(str, cols))).encode('utf-8')).hexdigest()
    def save_hist(*a, **k): pass
    def check_time_splits(*a, **k): pass


DEBUG = os.getenv("RRE_DEBUG","0") == "1"
LOG_JSON = os.getenv("RRE_LOG_JSON","0") == "1"
DBG_DIR = "/content/drive/MyDrive/rre/_debug/preprocess"
log = setup_logger(DEBUG, LOG_JSON, "rre.preprocess"); log_env(log)

# 미래데이터 누수 감지 플래그 (기본 OFF: 필요할 때만 켜서 사용)
LEAK_CHECK_ENABLED = os.getenv("RRE_LEAKCHECK", "0") == "1"

DIAG_ENABLED = os.getenv("RRE_DIAG", "1") == "1"   # 환경변수로 ON/OFF
DIAG_DIR = "/content/drive/MyDrive/rre/_diag/preprocess"
os.makedirs(DIAG_DIR, exist_ok=True)

# 피보나치 윈도우 (환경변수로 조절 가능, 기본 100)
FIB_WINDOW = int(os.getenv("FIB_WINDOW", "100"))
FIB_NAME_POS   = f"FIB_POS_{FIB_WINDOW}"
FIB_NAME_D382  = f"FIB_DIST_0382_{FIB_WINDOW}"
FIB_NAME_D618  = f"FIB_DIST_0618_{FIB_WINDOW}"

@numba.jit(nopython=True)
def _calc_rolling_beta_numba(stock_ret, market_ret, window):
    """
    CAPM Beta = Cov(Stock, Market) / Var(Market)
    이걸 Rolling Window로 고속 연산합니다.
    """
    n = len(stock_ret)
    betas = np.full(n, np.nan)
    
    # 윈도우 크기만큼 확보된 지점부터 루프
    for i in range(window, n):
        # 1. 윈도우 슬라이싱
        s_win = stock_ret[i-window : i]
        m_win = market_ret[i-window : i]
        
        # 2. 유효성 검사 (NaN이 너무 많으면 패스)
        # Numba에서는 np.isnan check 필요
        valid_mask = ~np.isnan(s_win) & ~np.isnan(m_win)
        if np.sum(valid_mask) < (window * 0.7): # 70% 이상 데이터 있어야 함
            continue
            
        s_clean = s_win[valid_mask]
        m_clean = m_win[valid_mask]

        # 3. 분산/공분산 계산
        var_m = np.var(m_clean)
        if var_m < 1e-9: # 분모 0 방지
            continue
            
        cov_sm = np.cov(s_clean, m_clean)[0, 1]
        
        # 4. 베타 저장
        betas[i] = cov_sm / var_m

    return betas

@numba.jit(nopython=True)
def _calc_robust_zscore_numba(arr, window):
    """
    Computes Rolling Robust Z-Score using Median and MAD (Median Absolute Deviation).
    Z = (X - Median) / (MAD * 1.4826)
    Specific for outlier resistance.
    """
    n = len(arr)
    out = np.full(n, np.nan)
    scale_factor = 1.4826 # Consistent estimator for Normal distribution sigma
    
    for i in range(window, n):
        # Slice window
        win_data = arr[i-window+1 : i+1] # Includes current day i
        
        # Remove NaNs
        valid_mask = ~np.isnan(win_data)
        if np.sum(valid_mask) < (window * 0.5): # Require 50% data
            continue
            
        clean_data = win_data[valid_mask]
        
        # Median
        med = np.median(clean_data)
        
        # MAD
        abs_diff = np.abs(clean_data - med)
        mad = np.median(abs_diff)
        
        if mad < 1e-9:
            # If MAD is 0 (e.g., flat line), return 0 or NaN. 
            # Returning 0 implies "at median".
            out[i] = 0.0
        else:
            out[i] = (arr[i] - med) / (mad * scale_factor)
            
    return out

def _robust_rolling_zscore(series, window):
    """Pandas wrapper for Numba robust z-score"""
    vals = series.values.astype(np.float64)
    return pd.Series(_calc_robust_zscore_numba(vals, window), index=series.index)


# 피처 스키마 레지스트리 정립
# PFI(Feature Importance) 분석 결과에 따른 정예 피처 27개 선정
# - 제거됨: Rank 관련 피처 전량, 중요도 0.001 미만 피처들, 중복성 지표
# - Is_Tradable: 피처에서 제외하고 메타 데이터로만 사용 (Leakage 방지)
OFFICIAL_FEATURE_SCHEMA = sorted([
    # 1. 구조 및 패턴 (Structure & Pattern) - 중요도 최상위
    'DONCHIAN_POS_20',      # (Rank 1) 현재가가 최근 고저폭의 어디쯤인가?
    'HIGH_REL',             # (Rank 2) 고가 상대 위치
    'LOW_REL',              # (Rank 4) 저가 상대 위치
    'CLOSE_Z_252',          # (Rank 5) 1년치 흐름 대비 현재 위치 (Mean Reversion)
    'FIB_POS_100',          # (Rank 16) 피보나치 위치
    'OPEN_REL',             # (Rank 18) 시가 상대 위치

    # 2. 추세 및 모멘텀 (Trend & Momentum)
    'ADX_14',               # (Rank 3) 추세의 강도 (방향 아님)
    'RET_5',                # (Rank 15) 1주일 수익률
    'RSI_14',               # (Rank 19) 과매수/과매도
    'SMA_60',               # (Rank 21) 이격도 관련
    'PPO_HIST',             # (Rank 23) MACD 히스토그램 (단기 모멘텀 변화)

    # 3. 수급 (Supply & Demand) - 개인/외국인 중요
    '개인',                 # (Rank 6) 개인 수급 강도
    '외국인_SUM5',          # (Rank 11) 외국인 1주일 누적
    '개인_SUM20',           # (Rank 12) 개인 1개월 누적
    '외국인_SUM20',         # (Rank 14) 외국인 1개월 누적
    '개인_Z_60',            # (Rank 17) 개인 수급 이상 징후 (Z-score)
    '외국인',               # (Rank 22) 외국인 수급 강도
    '외국인_Z_60',          # (Rank 27) 외국인 수급 이상 징후

    # 4. 변동성 및 리스크 (Volatility & Risk)
    'BETA_60',              # (Rank 7)  시장 민감도
    'ATR_14_REL',           # (Rank 8)  상대적 변동성 크기
    'BB_WIDTH_Z_60',        # (Rank 9)  볼린저 밴드 폭 (스퀴즈 감지)
    'RV_5',                 # (Rank 33) 단기 실현 변동성 (5일)
    'RV_20',                # (Rank 34) 중기 실현 변동성 (20일)
    'RV_RATIO_5_20',        # (Rank 35) 단기/중기 변동성 레짐 비율
    'NEG_VOL_20',           # (Rank 36) 하방 실현 변동성 (20일)
    'DRAWDOWN_60',          # (Rank 37) 60일 롤링 고점 대비 낙폭

    # 5. 기타 보조 지표 (Others)
    'VWAP_DIST_Z_60',       # (Rank 10) 거래량 가중 평균가 대비 이격
    'ALPHA_20',             # (Rank 20) 시장 대비 초과 수익률
    'AMIHUD_REL',           # (Rank 24) 비유동성 충격 (급락/급등 시 호가 공백)
    'VOLUME',               # (Rank 25) 거래량 변화율
    'BBP_20_2.0_CENTERED',  # (Rank 26) 볼린저 밴드 내 위치
    'KER_20',               # (Rank 28) 가격 효율성 (추세의 깨끗함 정도)

    # 6. 매크로 & 상대강도
    'US_MKT_RET_1',
    'FX_RET_1',
    'YIELD_SPREAD',
    'REL_RET_1',
    'REL_RET_20',

    # 7. VSA 기반 수급 효율성
    'VSA_REL',
])
# 추후 검증을 위해 Set으로 변환하여 사용
INTENT_FEATURES = set(OFFICIAL_FEATURE_SCHEMA)


# 피처 하드 클리핑(Guardrail) 설정
#  - 전부 "고정 상수"로만 경계를 잡습니다.
#  - 데이터 분포 기반 quantile, rolling 통계는 전혀 쓰지 않아서
#    미래 데이터 참조(누수) 위험이 0 입니다.
#  - 값이 비정상적으로 폭주하는 경우만 잘라내는 '안전 장치' 역할입니다.
FEATURE_CLIP_BOUNDS = {
    # 1. Supply / Demand (수급) - 이미 turnover 대비 비율이라 -10~10이면 충분히 넉넉
    '외국인': (-10.0, 10.0),
    '기관': (-10.0, 10.0),
    '개인': (-10.0, 10.0),
    '외국인_SUM5': (-25.0, 25.0),
    '외국인_SUM20': (-50.0, 50.0),
    '기관_SUM5': (-25.0, 25.0),
    '기관_SUM20': (-50.0, 50.0),
    '개인_SUM5': (-25.0, 25.0),
    '개인_SUM20': (-50.0, 50.0),
    '외국인_Z_60': (-8.0, 8.0),
    '기관_Z_60': (-8.0, 8.0),
    '개인_Z_60': (-8.0, 8.0),

    # 2. Smart Money & Structure (구조)
    'RVOL_5_LOG': (-6.0, 6.0),          # log 상대 거래량
    'BB_WIDTH_Z_60': (-8.0, 8.0),
    'DONCHIAN_POS_20': (0.0, 1.0),      # 이론상 [0, 1]
    'KER_20': (0.0, 1.0),               # 효율성 비율 [0, 1]
    'VWAP_DIST_Z_60': (-8.0, 8.0),
    'AMIHUD': (0.0, 1e3),               # 비정상치(폭주)만 잘라내는 넉넉한 상한
    'HL_SPREAD': (0.0, 1.0),            # 하루 고저폭이 100% 이상이면 거의 비정상
    'AMIHUD_REL': (-8.0, 8.0),
    'HL_SPREAD_REL': (0.0, 10.0),

    # 3. Price & Momentum (가격/모멘텀)
    'RSI_14': (-1.5, 1.5),              # -1~1 스케일 근처
    'SMA_60': (-8.0, 8.0),
    'FIB_POS_100': (0.0, 1.0),
    'RET_1': (-10.0, 10.0),
    'RET_5': (-10.0, 10.0),
    'RET_20': (-10.0, 10.0),
    'OPEN_REL': (-1.0, 1.0),            # -100% ~ +100% 갭이면 충분히 넉넉
    'HIGH_REL': (-1.0, 1.0),
    'LOW_REL': (-1.0, 1.0),
    'CLOSE_Z_252': (-10.0, 10.0),
    'BBP_20_2.0_CENTERED': (-1.0, 1.0),
    'BBWIDTH_PCT_20_2.0': (0.0, 5.0),
    'PPO_12_26_9': (-100.0, 100.0),
    'PPO_HIST': (-100.0, 100.0),
    'VOLUME': (-10.0, 10.0),
    'ADX_14': (0.0, 100.0),             # ADX 이론상 0~100
    'ATR_14': (0.0, 100.0),
    'ATR_14_REL': (0.0, 10.0),

    # 4. Alpha Factors (시장 중립화)
    'BETA_60': (-5.0, 5.0),
    'ALPHA_20': (-5.0, 5.0),
    'IDIOSYNCRATIC_VOL_20': (0.0, 5.0),

    # World-Class Volatility Features
    'RV_5': (0.0, 0.5),                 # 일간 log 수익률 ~±20% 가정 시 충분히 넉넉
    'RV_20': (0.0, 0.5),
    'RV_RATIO_5_20': (0.0, 5.0),        # 단기/중기 변동성 비율, 극단값 방지용 상한
    'NEG_VOL_20': (0.0, 0.5),           # 하방 변동성도 RV와 비슷한 스케일
    'DRAWDOWN_60': (-1.2, 0.2),         # 0%~ -100% 이상 낙폭 + 약간의 버퍼

    # 5. Macro & Relative Features
    'US_MKT_RET_1': (-0.10, 0.10),   # ±10% 초과는 이상치로 클리핑
    'FX_RET_1':     (-0.10, 0.10),   # 환율도 비슷한 스케일
    'YIELD_SPREAD': (-5.0,  5.0),    # 금리차, 수치 기준 넉넉하게

    'REL_RET_1':   (-0.30, 0.30),    # 일간 초과수익
    'REL_RET_20':  (-3.0,  3.0),     # 20일 누적 초과수익

    'VSA_REL':     (0.0,  10.0),     # 상대 효율성, 과도한 값 컷
}


def _apply_feature_clipping_inplace(df: pd.DataFrame) -> None:
    """
    OFFICIAL_FEATURE_SCHEMA 에 속한 피처들에 대해
    '상수 기반 하드 클리핑'을 적용합니다.

    - 데이터 분포를 보지 않고, 미리 정한 경계만 사용하므로
      미래 데이터 누수(leak)와는 무관합니다.
    - 극단값(폭주 값)만 잘라내고, 정상 구간은 그대로 유지해서
      모델이 이상치에 흔들리지 않도록 합니다.
    """
    for col, bounds in FEATURE_CLIP_BOUNDS.items():
        if col not in df.columns:
            continue
        s = df[col]
        # 숫자형이 아닌 경우 방어적으로 패스
        if not np.issubdtype(s.dtype, np.number):
            continue
        lo, hi = bounds
        df[col] = s.clip(lower=lo, upper=hi).astype('float32')


BASE_COLS = {'open','high','low','close','volume','ticker','label'}

def _sha1(lst):
    m = hashlib.sha1()
    for x in lst: m.update((x+"|").encode())
    return m.hexdigest()[:12]

def diag_write(event: str, payload: dict, ticker: str|None=None):
    """구조화 JSONL 한 줄로 던지기"""
    if not DIAG_ENABLED: return
    ts = _dt.now(timezone.utc).isoformat(timespec="milliseconds")
    ts = ts.replace("+00:00", "Z")
    row = {"ts": ts, "event": event, "ticker": ticker, **payload}
    try:
        with open(os.path.join(DIAG_DIR, "preprocess_diag.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    except Exception as e:
        print(f"[DiagErr] Failed to write diag log: {e}")

def feature_snapshot(df: pd.DataFrame, stage: str, ticker: str):
    """현재 컬럼 상태 스냅샷 + 의도 대비 차이 기록"""
    cols = [c for c in df.columns if c not in BASE_COLS]
    present = set(cols)
    missing = sorted(list(INTENT_FEATURES - present))
    extra   = sorted(list(present - INTENT_FEATURES))
    diag_write("snapshot", {
        "stage": stage,
        "n_features": len(cols),
        "missing": missing,
        "extra": extra,
        "hash": _sha1(sorted(cols)),
    }, ticker)

def bollinger_probe(df: pd.DataFrame, stage: str, ticker: str):
    """BB 컬럼 유무를 별도로 기록 (접미사 이슈 진단용)"""
    def has(*names): return [n for n in names if n in df.columns]
    diag_write("bollinger", {
        "stage": stage,
        "b_cols_2.0": has('BBL_20_2.0','BBM_20_2.0','BBU_20_2.0','BBB_20_2.0','BBP_20_2.0'),
        "b_cols_2":   has('BBL_20_2','BBM_20_2','BBU_20_2','BBB_20_2','BBP_20_2'),
    }, ticker)

def stage_timer(name):
    """with stage_timer('X'):  ...  형태로 구간 시간 로깅"""
    class _T:
        def __enter__(self_s): self_s.t0 = time.time(); return self_s
        def __exit__(self_s, *exc):
            diag_write("timing", {"stage": name, "sec": round(time.time()-self_s.t0,3)}, None)
    return _T()


# ==================================================
#           환경 설정
# ==================================================
# datacl.py에서 설정한 원본 데이터 기본 경로
BASE_DATA_PATH = "/content/drive/MyDrive/rre/data"

# 4가지 데이터 소스 경로
PATH_OHLCV = os.path.join(BASE_DATA_PATH, "all_stocks_ohlcv")
PATH_FUNDAMENTAL = os.path.join(BASE_DATA_PATH, "all_stocks_fundamental")
PATH_TRADING = os.path.join(BASE_DATA_PATH, "all_stocks_trading")
PATH_MACRO = os.path.join(BASE_DATA_PATH, "market_data")

# 경로 설정: 로컬 우선, GDrive는 최종 목적지
# 가공된 피처/레이블 데이터를 저장할 경로
# 1) 1차 저장: Colab 로컬 디스크
LOCAL_OUTPUT_PATH = "/content/rre_local/data"

# 2) 최종 목적지: Google Drive
FINAL_OUTPUT_PATH = "/content/drive/MyDrive/rre/data"

# Parquet 저장 시 기준이 될 파일명 (e.g., "processed_data_parquet" 폴더 생성)
OUTPUT_BASE_NAME = "processed_data"

# 아래는 기존 코드와 호환을 위해 OUTPUT_PATH는 로컬로 맞춰줌
OUTPUT_PATH = LOCAL_OUTPUT_PATH

# 단일 캔들 목표 수익률 설정
# 다음날 시가 대비 종가가 2% 이상 오르면 1, 아니면 0
LABEL_PROFIT_THRESHOLD = 0.02  # 2%

# ==================================================
# 동적(Dynamic) 익절/손절 설정 (ATR 기반)
# ==================================================
# - USE_DYNAMIC_TP_SL = True  이면 ATR_14 기반으로 TP/SL을 샘플별로 조정
# - False 로 두면 기존 정적 배수 (TAKE_PROFIT_PCT, STOP_LOSS_PCT) 그대로 사용
USE_DYNAMIC_TP_SL: bool = False

# 어떤 ATR 컬럼을 쓸지 (run_preprocess에서 이미 만드는 피처 중 하나)
ATR_COL_FOR_DYNAMIC_TP_SL: str = "ATR_14"

# ATR 기준 상대 변동성 = ATR / 가격 의 "기준값"
#   - 예) ATR_14 / 가격 ≈ 0.02 (2%) 를 "평균적인 일간 변동성"으로 가정
DYNAMIC_TP_SL_ATR_REF_PCT: float = 0.02  # 2% 기준

# 변동성에 따라 TP/SL을 몇 배까지 늘리고 줄일지 범위 (클램프)
#   - 예) 0.5 ~ 2.0 이면
#       * 저변동(ATR 낮음) 종목: TP/SL 기준폭을 0.5배로 축소
#       * 고변동(ATR 높음) 종목: TP/SL 기준폭을 최대 2배까지 확대
DYNAMIC_TP_SL_VOL_SCALE_MIN: float = 0.5
DYNAMIC_TP_SL_VOL_SCALE_MAX: float = 2.0

# [선택] 벤치마크 지수 CSV (예: KOSPI/KOSDAQ) - '날짜' 인덱스, '종가' 컬럼 필요
# 없으면 None 그대로 두세요.
BENCHMARK_CSV_PATH = None
# 예시: "/content/drive/MyDrive/rre/data/KOSPI_index.csv"

# [선택] 최적화 토글
USE_PRUNE = True    # True: 메모리 절약을 위해 최종 컬럼 가지치기
USE_DOWNCAST = True # True: 메모리 절약을 위해 숫자 타입 다운캐스트

# 스키마 보장 컬럼: 존재하지 않으면 생성할 컬럼 세트
SCHEMA_ENSURE_NAN = [
    # 펀더멘털 등 원본 데이터에 간헐적으로 누락될 수 있는 컬럼만 남김
    # (매크로 파생 변수 4종 제거 완료)
]
SCHEMA_ENSURE_ZERO = [
    # 수급(없으면 0으로 의미 안전)
    '외국인','기관','개인',
]


def downcast_numeric_inplace(df: pd.DataFrame):
    """
    모든 수치형 데이터를 float32로 통일하여 Parquet 스키마 불일치(ArrowInvalid) 방지.
    조건부 int32 변환은 파일마다 컬럼 타입이 달라지게 만들어 일괄 로딩 시 에러를 유발함.
    """
    # 1. 대상: 숫자형(float, int) 전체 식별
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # 2. 일괄 float32 변환 (데이터 일관성 및 메모리 최적화)
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].astype('float32')

    # 3. Ticker는 category로 최적화
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype('category')

def prune_columns_inplace(df: pd.DataFrame):
    """
    [Final Quality Gate]
    데이터프레임을 저장하기 전, 공식 스키마에 정의된 컬럼만 남기고,
    필수 컬럼이 누락되었는지 검사합니다.

    여기서 OHLCV(raw price/volume)는 라벨 후처리(LiquidSharpe 등)에만
    사용하고, 모델 피처에서는 제외할 것이므로
    parquet에는 계속 남겨둡니다.
    """
    # 1. 필수 보존 컬럼
    #    - BASE_COLS: {'open','high','low','close','volume','ticker','label'}
    #    - 여기에 거래 가능 플래그 컬럼까지 합집합으로 보존
    base_cols = set(BASE_COLS) | {'Is_Tradable', 'Trading_Missing_Flag'}
    
    # 2. 공식 피처 스키마 (위에서 정의한 상수)
    keep_features = set(OFFICIAL_FEATURE_SCHEMA)
    
    # 3. 최종 유지할 컬럼 집합
    final_cols = base_cols.union(keep_features)
    
    # 4. 실제 존재하는 컬럼 중 유지할 것만 선택 (Drop Unused)
    cols_to_drop = [c for c in df.columns if c not in final_cols]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # 5. [Validation] 필수 피처 누락 검사
    # (모든 피처가 모든 종목에 다 있을 순 없으므로 Warning 수준으로 처리하되,
    #  너무 많이(50% 이상) 없으면 에러로 간주)
    current_features = set(df.columns) - base_cols
    missing_features = set(OFFICIAL_FEATURE_SCHEMA) - current_features
    
    if len(missing_features) > len(keep_features) * 0.5:
        log.error(f"[Quality Fail] Major features missing: {len(missing_features)} items")
        # 심각한 경우 여기서 raise 하여 해당 종목을 저장하지 않게 할 수 있음
        # raise ValueError("Insufficient features")
    elif missing_features:
        # 누락된 피처는 NaN으로 생성하여 Parquet 스키마를 통일시킴 (매우 중요)
        for col in missing_features:
            df[col] = np.nan
            
    # 최종적으로 컬럼 순서 정렬 (Parquet 효율성 증대)
    # Ticker, Date 등은 인덱스 혹은 맨 앞, 나머지는 이름순
    # (단, inplace 연산이므로 reindex 사용 시 주의)


def _coerce_date_index(idx_like):
    s = pd.Index(idx_like)
    try:
        # 숫자형 'YYYYMMDD' (가장 흔한 케이스)
        if s.dtype == "object" and s.str.len().min() >= 8 and s.str.isnumeric().all():
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        # 'YYYY-MM-DD' (두번째 흔한 케이스)
        if s.dtype == "object" and s.str.contains("-").any():
            return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    except Exception:
        pass # 예외 발생 시 아래의 느린 경로로
    # 그 외는 마지막 수단 (느린 경로)
    return pd.to_datetime(s, errors="coerce")

def load_benchmark_series(csv_path: str | None):
    """
    '날짜'를 index(datetime), '종가' 컬럼을 갖는 벤치마크 CSV를 읽어 Series 반환.
    """
    if not csv_path:
        return None
    try:
        df = pd.read_csv(csv_path)
        if '날짜' not in df.columns or '종가' not in df.columns:
            raise ValueError(f"벤치마K리그 CSV 포맷 이상: {df.columns.tolist()}")
        
        df['날짜'] = _coerce_date_index(df['날짜'])
        df = df.loc[~df['날짜'].isna()].copy() # NaT 제거
        
        df.sort_values('날짜', inplace=True)
        s = pd.to_numeric(df['종가'], errors='coerce')
        s.index = df['날짜']
        s = s.dropna()
        s.sort_index(inplace=True) # 원본 코드의 sort_index 유지
        return s
    except Exception as e:
        print(f"[Warning] 벤치마크 로딩 실패: {e}")
        return None

# 1-1) 표준(정식) 컬럼명으로 강제 통일 함수 추가
CANONICAL_RENAME_MAP = {
    # Donchian: 환경에 따라 DCL_20 으로 나오기도 함 → 모두 _20_20 으로 통일
    'DCL_20': 'DCL_20_20',
    'DCM_20': 'DCM_20_20',
    'DCU_20': 'DCU_20_20',
    # ATR: 퍼센트 ATR(ATRr_14)이 생기는 환경이 있음 → 표준은 ATR_14 로 통일
    'ATRr_14': 'ATR_14',
}

def _fix_bb_cols(cols):
    fixed = []
    for c in cols:
        if c.startswith(("BBL_", "BBM_", "BBU_", "BBB_", "BBP_")):
            # 뒤에 같은 숫자 토큰이 2번 연속이면 하나만 남김
            parts = c.split("_")
            if len(parts) >= 4 and parts[-1] == parts[-2]:
                parts = parts[:-1]
            c = "_".join(parts)
        fixed.append(c)
    return fixed


def enforce_canonical_names_inplace(df: pd.DataFrame):
    try:
        df.columns = _fix_bb_cols(df.columns)
    except Exception as e:
        log.warning(f"[_fix_bb_cols] 볼린저 밴드 이름 교정 실패: {e}")

    rename_map = {}
    for src, dst in CANONICAL_RENAME_MAP.items():
        if src in df.columns:
            if dst in df.columns:
                # dst가 이미 있으면 src만 버림
                df.drop(columns=[src], inplace=True, errors="ignore")
            else:
                # dst가 없으면 rename
                rename_map[src] = dst

    if rename_map:
        df.rename(columns=rename_map, inplace=True)


def _compute_features_core(group: pd.DataFrame,
                           ticker: str | None = None):
    """
    [World-Class Feature Engineering]
    1. Log Returns for all momentum calculations.
    2. Risk-Adjusted Momentum (Return / Volatility).
    3. Robust Scaling (MAD-based) instead of standard Z-score where applicable.
    4. Microstructure-aware Liquidity features.
    """
    group.sort_index(kind='mergesort', inplace=True)
    EPSILON = 1e-9

    # --- 1. Base Transformations (Log Returns) ---
    # Log Return is additive and symmetric, better for ML
    group['log_close'] = np.log(group['close'] + EPSILON)
    group['log_ret_1'] = group['log_close'].diff()
    
    # Robust Volatility (60-day Median Absolute Deviation proxy or standard std)
    # Using standard std for volatility normalization is acceptable, 
    # but we limit extreme values later.
    vol_60d = group['log_ret_1'].rolling(window=60, min_periods=20).std() + EPSILON

    # --- 2. Price & Momentum (Risk-Adjusted & Market-Relative) ---
    # RET_X: Risk-Adjusted Momentum (Sharpe-like ratio over the window)
    # Logic: Sum(LogRet) / (Std(LogRet) * Sqrt(N))
    
    for w in [5, 20]:
        # Cumulative Log Return
        ret_cum = group['log_ret_1'].rolling(window=w, min_periods=w).sum()
        # Window Volatility
        vol_win = group['log_ret_1'].rolling(window=w, min_periods=w).std() * np.sqrt(w)
        # Risk Adjusted Momentum
        group[f'RET_{w}'] = (ret_cum / (vol_win + EPSILON)).astype('float32')

    # RET_1: Normalized by long-term volatility
    group['RET_1'] = (group['log_ret_1'] / vol_60d).astype('float32')

    # SMA: Distance from SMA, normalized by Volatility
    # (Close - SMA) / (Price * Vol) ~= Z-score
    for w in [5, 20, 60]:
        sma = group['close'].rolling(window=w, min_periods=int(w/2)).mean()
        # Log distance is better: ln(Close / SMA)
        # Normalized by 60d Volatility to make it comparable across stocks
        dist = np.log(group['close'] / (sma + EPSILON))
        group[f'SMA_{w}'] = (dist / (vol_60d * np.sqrt(w/20) + EPSILON)).astype('float32')

    # RSI: Keep standard definition, but scale to 0-1 and center
    # RSI is already bounded 0-100. Let's map 50 -> 0, 0/100 -> +/- 1 approx
    try:
        rsi = group.ta.rsi(close='close', length=14)
        if rsi is not None:
             group['RSI_14'] = ((rsi - 50.0) / 50.0).astype('float32')
        else:
             group['RSI_14'] = np.nan
    except: group['RSI_14'] = np.nan

    # --- 3. Price Structure & Reversion ---
    # CLOSE_Z_252: Rank within last 1 year (Robust Position)
    # Using (LogPrice - RollingMedian) / RollingMAD
    try:
        group['CLOSE_Z_252'] = _robust_rolling_zscore(group['log_close'], 252).astype('float32')
    except Exception:
        group['CLOSE_Z_252'] = np.nan

    # GAP / INTRA / RANGE: Volatility Normalized
    prev_close = group['close'].shift(1)
    
    # Gap: ln(Open / PrevClose) / Vol
    gap_val = np.log(group['open'] / (prev_close + EPSILON))
    group['GAP_PCT'] = (gap_val / vol_60d).astype('float32')

    # Intra: ln(Close / Open) / Vol
    intra_val = np.log(group['close'] / (group['open'] + EPSILON))
    group['INTRA_RET'] = (intra_val / vol_60d).astype('float32')
    
    # Range: (High - Low) / Close / Vol (Intraday Volatility vs Daily Volatility)
    # Using Log High - Log Low approximation
    hl_range = np.log(group['high'] / (group['low'] + EPSILON))
    group['RANGE_PCT'] = (hl_range / vol_60d).astype('float32')
    
    # OHLC Relative
    group['OPEN_REL'] = (group['open'] / (group['close'] + EPSILON) - 1.0).astype('float32')
    group['HIGH_REL'] = (group['high'] / (group['close'] + EPSILON) - 1.0).astype('float32')
    group['LOW_REL']  = (group['low']  / (group['close'] + EPSILON) - 1.0).astype('float32')

    # --- 3-A. Volatility Regime & Tail Risk ---
    #   - RV_5, RV_20: Realized Volatility (short / mid)
    #   - RV_RATIO_5_20: Vol regime (short vs mid)
    #   - NEG_VOL_20: Downside Volatility (Sortino 스타일)
    #   - DRAWDOWN_60: 최근 60일 롤링 최대 낙폭
    try:
        # 1) 로그 일간 수익률 (이미 상단에서 계산됨)
        log_ret = pd.to_numeric(group.get("log_ret_1"), errors="coerce")

        if log_ret is None:
            # log_ret_1 이 생성되지 못했으면 전부 NaN으로 유지
            raise KeyError("log_ret_1 missing for volatility features")

        # (a) 실현 변동성: 5일 / 20일
        ret_sq = log_ret.pow(2)

        rv_5_mean = ret_sq.rolling(window=5, min_periods=3).mean()
        rv_20_mean = ret_sq.rolling(window=20, min_periods=10).mean()

        rv_5 = np.sqrt(rv_5_mean.clip(lower=0.0))
        rv_20 = np.sqrt(rv_20_mean.clip(lower=0.0))

        group["RV_5"] = rv_5.astype("float32")
        group["RV_20"] = rv_20.astype("float32")

        # (b) 단기 vs 중기 변동성 레짐
        rv_ratio = rv_5 / (rv_20 + EPSILON)
        group["RV_RATIO_5_20"] = rv_ratio.astype("float32")

        # (c) 하방 변동성 (Downside Volatility, Semi-deviation)
        neg_ret = log_ret.where(log_ret < 0.0)
        neg_ret_sq = neg_ret.pow(2)

        neg_vol_20_mean = neg_ret_sq.rolling(window=20, min_periods=10).mean()
        neg_vol_20 = np.sqrt(neg_vol_20_mean.clip(lower=0.0))

        group["NEG_VOL_20"] = neg_vol_20.astype("float32")

        # (d) 60일 롤링 최대 낙폭 (Drawdown)
        rolling_max_60 = group["close"].rolling(window=60, min_periods=20).max()
        dd_60 = group["close"] / (rolling_max_60 + EPSILON) - 1.0
        group["DRAWDOWN_60"] = dd_60.astype("float32")

    except Exception:
        # 예외가 나더라도 스키마는 항상 유지
        group["RV_5"] = group.get("RV_5", np.nan).astype("float32")
        group["RV_20"] = group.get("RV_20", np.nan).astype("float32")
        group["RV_RATIO_5_20"] = group.get("RV_RATIO_5_20", np.nan).astype("float32")
        group["NEG_VOL_20"] = group.get("NEG_VOL_20", np.nan).astype("float32")
        group["DRAWDOWN_60"] = group.get("DRAWDOWN_60", np.nan).astype("float32")

    # ==================================================================
    # 3-B. Market-Relative & Macro Context
    #   (미래 데이터 누수 없이, "그 시점까지 알려진 정보"만 사용)
    # ==================================================================

    # 1) 미국 S&P500 전일 수익률
    if "SP500_Close" in group.columns:
        sp500 = pd.to_numeric(group["SP500_Close"], errors="coerce")
        sp500_ret = np.log(sp500 / (sp500.shift(1) + EPSILON))

        # 한국 시간 T일 기준으로, 실제로 사용할 수 있는 건 "T-1 미국 장 수익률"
        group["US_MKT_RET_1"] = sp500_ret.shift(1).astype("float32")

    # 2) 원/달러 환율 변동률 (당일 종가 vs 전일 종가)
    if "USD_KRW_Close" in group.columns:
        fx = pd.to_numeric(group["USD_KRW_Close"], errors="coerce")
        fx_ret = np.log(fx / (fx.shift(1) + EPSILON))
        group["FX_RET_1"] = fx_ret.astype("float32")

    # 3) 장단기 금리차 (경기 사이클)
    spread = None
    if {"UST10Y", "UST2Y"}.issubset(group.columns):
        y10 = pd.to_numeric(group["UST10Y"], errors="coerce")
        y2  = pd.to_numeric(group["UST2Y"], errors="coerce")
        spread = y10 - y2
    elif "UST_SLOPE_10Y2Y" in group.columns:
        spread = pd.to_numeric(group["UST_SLOPE_10Y2Y"], errors="coerce")

    if spread is not None:
        group["YIELD_SPREAD"] = spread.astype("float32")

    # 4) KOSPI 대비 상대 수익률 (Alpha Proxy)
    if "KOSPI_Close" in group.columns:
        mkt = pd.to_numeric(group["KOSPI_Close"], errors="coerce")
        mkt_log = np.log(mkt + EPSILON)
        mkt_log_ret = mkt_log.diff()

        # log_ret_1 은 이미 상단에서 log_close.diff() 로 계산됨
        stock_log_ret = group["log_ret_1"]

        rel_ret_1 = stock_log_ret - mkt_log_ret
        group["REL_RET_1"] = rel_ret_1.astype("float32")

        group["REL_RET_20"] = (
            rel_ret_1.rolling(window=20, min_periods=5).sum()
        ).astype("float32")

    # --- 4. Volume & Liquidity (Microstructure) ---
    vol_sma_20 = group['volume'].rolling(20, min_periods=5).mean()
    group['VOLUME'] = np.log1p(group['volume'] / (vol_sma_20 + EPSILON)).astype('float32')
    
    # RVOL_5_LOG: Short-term volume anomaly
    # Log(Vol_5_Avg / Vol_60_Median) -> Robust Relative Volume
    vol_sma_5 = group['volume'].rolling(5, min_periods=1).mean()
    vol_med_60 = group['volume'].rolling(60, min_periods=20).median()
    group['RVOL_5_LOG'] = np.log1p(vol_sma_5 / (vol_med_60 + EPSILON)).astype('float32')

    # Amihud Illiquidity: Average(|Ret| / DollarVolume)
    # Dollar Volume = Close * Volume
    # We use a robust transformation: Log(Amihud) -> Z-Score
    try:
        dollar_vol = group['close'] * group['volume'] + EPSILON
        illiquidity = group['log_ret_1'].abs() / dollar_vol
        
        # Raw Amihud (20 days)
        amihud_20 = illiquidity.rolling(20, min_periods=10).mean()
        
        # Log-Transform (to fix skewness)
        log_amihud = np.log(amihud_20 + EPSILON)
        
        # Self-Relative (Z-score against own history of liquidity)
        # "Is it harder to trade today than usual?"
        group['AMIHUD_REL'] = _robust_rolling_zscore(log_amihud, 252).astype('float32')
        group['AMIHUD'] = log_amihud.astype('float32') # Keep absolute level (log scale)
        
    except Exception:
        group['AMIHUD_REL'] = np.nan
        group['AMIHUD'] = np.nan

    # HL Spread (Effective Spread proxy)
    try:
        # High-Low spread as % of price
        hl_spread = (group['high'] - group['low']) / (group['close'] + EPSILON)
        # Relative to recent history
        group['HL_SPREAD'] = hl_spread.astype('float32')
        group['HL_SPREAD_REL'] = (
            hl_spread / (hl_spread.rolling(60).median() + EPSILON)
        ).astype('float32')
    except Exception:
        group['HL_SPREAD'] = np.nan
        group['HL_SPREAD_REL'] = np.nan

    # ==================================================================
    # VSA Interaction: Effort vs Result
    #   - 거래량 1단위당 몸통 길이 → "거래 효율성"
    #   - 과거 20일 평균 대비 상대 비율
    # ==================================================================
    try:
        spread_body = (group["close"] - group["open"]).abs()
        vsa_raw = spread_body / (group["volume"] + EPSILON)

        vsa_ma_20 = vsa_raw.rolling(window=20, min_periods=5).mean()
        group["VSA_REL"] = (vsa_raw / (vsa_ma_20 + EPSILON)).astype("float32")
    except Exception:
        group["VSA_REL"] = np.nan

    # --- 5. Structure / Oscillators ---
    # Bollinger Bands: Width Z-Score
    try:
        # Use simple pandas rolling for speed and control
        bb_sma = group['close'].rolling(20).mean()
        bb_std = group['close'].rolling(20).std()
        
        # %B Centered: (Price - SMA) / (2 * Std) -> Z-score of price relative to bands
        # Range -1 to +1 implies inside bands (approx)
        group['BBP_20_2.0_CENTERED'] = ((group['close'] - bb_sma) / (2 * bb_std + EPSILON)).astype('float32')
        
        # Bandwidth: (Upper - Lower) / SMA = (4 * Std) / SMA
        bb_width = (4 * bb_std) / (bb_sma + EPSILON)
        group['BBWIDTH_PCT_20_2.0'] = bb_width.astype('float32')
        
        # Robust Z-score of Bandwidth (Volatility of Volatility)
        group['BB_WIDTH_Z_60'] = _robust_rolling_zscore(bb_width, 60).astype('float32')

    except Exception:
        group['BBP_20_2.0_CENTERED'] = np.nan
        group['BBWIDTH_PCT_20_2.0'] = np.nan
        group['BB_WIDTH_Z_60'] = np.nan

    # PPO (MACD Percentage): Inherently robust to price levels
    try:
        ema12 = group['close'].ewm(span=12, adjust=False).mean()
        ema26 = group['close'].ewm(span=26, adjust=False).mean()
        ppo = (ema12 - ema26) / (ema26 + EPSILON) * 100
        ppo_sig = ppo.ewm(span=9, adjust=False).mean()
        
        group['PPO_12_26_9'] = ppo.astype('float32')
        group['PPO_HIST'] = (ppo - ppo_sig).astype('float32')
    except:
        group['PPO_12_26_9'] = np.nan
        group['PPO_HIST'] = np.nan
        
    # ADX / ATR
    try:
        # Use pandas_ta for complex logic
        group.ta.adx(length=14, append=True)
        # Normalize ADX: 0~100 -> 0~1
        if 'ADX_14' in group.columns:
            group['ADX_14'] = (group['ADX_14'] / 100.0).astype('float32')
            
        group.ta.atr(length=14, append=True)
        # Relative ATR: ATR / Close (Percentage Volatility)
        # Then standardized against history
        if 'ATRr_14' in group.columns: # Percent ATR
            group['ATR_14_REL'] = _robust_rolling_zscore(group['ATRr_14'], 60).astype('float32')
            group['ATR_14'] = group['ATRr_14'].astype('float32')
        elif 'ATR_14' in group.columns:
            atr_pct = group['ATR_14'] / (group['close'] + EPSILON)
            group['ATR_14_REL'] = _robust_rolling_zscore(atr_pct, 60).astype('float32')
    except: pass

    # Donchian: Position in range (0~1)
    try:
        d_high = group['high'].rolling(20).max()
        d_low  = group['low'].rolling(20).min()
        group['DONCHIAN_POS_20'] = ((group['close'] - d_low) / (d_high - d_low + EPSILON)).astype('float32')
    except: pass
    
    # VWAP Distance: Standardized
    try:
        # VWAP approx using Typical Price
        tp = (group['high'] + group['low'] + group['close']) / 3
        vp = tp * group['volume']
        vwap = vp.rolling(20).sum() / (group['volume'].rolling(20).sum() + EPSILON)
        
        dist_vwap = np.log(group['close'] / (vwap + EPSILON))
        group['VWAP_DIST_Z_60'] = _robust_rolling_zscore(dist_vwap, 60).astype('float32')
    except: pass

    # KER (Efficiency Ratio): Abs(NetChange) / Sum(AbsChange)
    try:
        change_abs = (group['close'] - group['close'].shift(20)).abs()
        path_len = group['close'].diff().abs().rolling(20).sum()
        group['KER_20'] = (change_abs / (path_len + EPSILON)).astype('float32')
    except: pass

    # Fibonacci (Position relative to window High/Low)
    try:
        win = FIB_WINDOW
        fh = group['high'].rolling(win).max()
        fl = group['low'].rolling(win).min()
        fpos = (group['close'] - fl) / (fh - fl + EPSILON)
        
        group[FIB_NAME_POS] = fpos.astype('float32')
        group[FIB_NAME_D382] = (fpos - 0.382).astype('float32') # Signed distance
        group[FIB_NAME_D618] = (fpos - 0.618).astype('float32')
    except: pass

    # Time Encodings (Cyclical)
    try:
        if hasattr(group.index, 'dayofyear'):
            doy = group.index.dayofyear
        else:
            doy = pd.to_datetime(group.index).dayofyear
        # Normalize to 0~1 range for sin/cos
        rad = 2 * np.pi * doy / 365.0
        group['DAY_SIN'] = np.sin(rad).astype('float32')
        group['DAY_COS'] = np.cos(rad).astype('float32')
    except:
        pass

    # 피처별 하드 클리핑 적용
    # OFFICIAL_FEATURE_SCHEMA 에 속한 모든 피처에 대해
    # 미리 정의한 안전 범위로 값 폭주를 막아줍니다.
    _apply_feature_clipping_inplace(group)

    # Clean up temp cols (안전하게, 존재 여부 확인 + errors="ignore")
    temp_cols = [c for c in ("log_close", "log_ret_1") if c in group.columns]
    if temp_cols:
        group.drop(columns=temp_cols, inplace=True, errors="ignore")


def compute_features_inplace(group: pd.DataFrame,
                             ticker: str | None = None):
    """
    [Leak Hunter]
    시계열 절단 검증(Time-Truncation Test)을 통해 미래 데이터 누수를 
    비트(bit) 단위로 감지하고, 적발 시 즉시 프로그램을 폭파시킵니다.
    """
    # 함수 진입 시점의 컬럼 목록 기억 (Leak Hunter에서 사용)
    original_cols = set(group.columns)

    # 1. 실제 피처 계산 (Main Run)
    _compute_features_core(group, ticker=ticker)
    
    # 누수 체크가 꺼져있거나, 데이터가 너무 적으면 패스
    if (not LEAK_CHECK_ENABLED) or (len(group) < 65):
        return

    # ==========================================================
    # 🕵️‍♂️ LEAK HUNTER: Counterfactual Validation
    # "미래(마지막 날)가 사라졌을 때, 과거(어제)의 값이 변한다면 그것은 누수다."
    # ==========================================================
    try:
        # 검증 대상: 마지막에서 두 번째 행 (T_end-1)
        target_idx = group.index[-2] 
        
        # A. 원본(Full) 상태에서의 값 추출
        check_cols = group.select_dtypes(include=[np.number]).columns
        vec_full = group.loc[target_idx, check_cols].copy()

        # B. 시계열 절단 시뮬레이션 (Truncated Simulation)
        sim_group = group.iloc[:-1].copy(deep=True)
        
        # [Hunter Fix] 검증 전, 계산된 피처들을 모두 날리고(Clean Slate) 재계산
        newly_created = set(group.columns) - original_cols
        sim_group.drop(columns=list(newly_created), inplace=True, errors='ignore')

        # Raw 상태에서 재계산 수행
        _compute_features_core(sim_group, ticker=ticker)
        
        # 원본에는 있었지만('등락률' 등), 재계산 후에는 없는(로직상 불필요한) 컬럼은 비교에서 제외
        valid_cols = [c for c in check_cols if c in sim_group.columns]
        
        vec_trunc = sim_group.loc[target_idx, valid_cols]
        vec_full_subset = vec_full[valid_cols] # 원본에서도 동일한 부분집합만 추출

        # C. 정밀 비교 (Tolerance: 1e-6)
        diff = (vec_full_subset.fillna(0) - vec_trunc.fillna(0)).abs()
        
        if diff.max() > 1e-6: # 허용 오차 초과 시
            leaking_cols = diff[diff > 1e-6].index.tolist()
            
            error_msg = (
                f"\n[🚨 CRITICAL LEAK DETECTED] 미래 데이터 누수 감지!\n"
                f"종목: {ticker}\n"
                f"원인: 마지막 날짜 데이터를 지웠더니 과거({target_idx})의 값이 변했습니다.\n"
                f"누수 의심 컬럼({len(leaking_cols)}개): {leaking_cols[:5]}...\n"
                f"최대 차이: {diff.max():.8f}\n"
                f"→ 프로세스를 강제 종료합니다."
            )
            print(error_msg)
            if 'log' in globals(): log.error(error_msg)
            sys.exit(1)

    except Exception as e:
        # 검증 로직 자체 에러는 경고만 하고 넘어감
        print(f"[LeakHunter] 검증 중 예외 발생(Pass): {e}")


def apply_labeling_simple(group, threshold=0.02):
    """
    [단순화된 레이블링]
    조건: (다음날 종가 / 다음날 시가 - 1) >= threshold
    """
    # 1. 다음날 시가와 종가 가져오기 (Shift -1)
    # 같은 종목 내에서 시간순 정렬되어 있다고 가정
    next_open = group['open'].shift(-1)
    next_close = group['close'].shift(-1)
    
    # 2. 수익률 계산 (Intraday Return)
    # 0으로 나누기 방지 (EPSILON)
    ret = (next_close - next_open) / (next_open + 1e-9)
    
    # 3. 레이블 생성 (True/False -> 1.0/0.0)
    # threshold(0.02) 이상이면 1.0, 아니면 0.0
    group['label'] = (ret >= threshold).astype('float32')
    
    # 4. 마지막 행 처리
    # shift(-1)을 하면 마지막 날짜는 NaN이 되므로 레이블도 NaN이어야 함 (삭제 대상)
    group.loc[next_open.isna(), 'label'] = np.nan
    
    return group


def save_by_year_parquet(df: pd.DataFrame, out_root: str, base_name: str, ticker: str):
    # 인덱스가 날짜여야 함
    years = df.index.year.to_numpy()
    df = df.copy()
    # 'ticker' 컬럼은 stream_pipeline에서 이미 추가되고 downcast_numeric_inplace에서 'category'로 변환됨.
    
    for y in np.unique(years):
        sub = df[years == y].copy() # .copy() 추가 (SettingWithCopyWarning 방지)
        if sub.empty:
            continue
            
        if DEBUG:
            print(f"[{ticker}] (Save) 연도: {y}, Shape: {sub.shape} (저장 전 스키마 보장 작업 시작)")

        # 스키마 보장: 누락 컬럼 생성
        for c in SCHEMA_ENSURE_NAN:
            if c not in sub.columns:
                sub[c] = np.nan
        for c in SCHEMA_ENSURE_ZERO:
            if c not in sub.columns:
                sub[c] = 0.0

        # 컬럼 순서 안정화(가독/일관성)
        # OHLCV는 prune에서 버림 → 여기선 정렬 기준에서 제외
        base_first = ['ticker','label']
        others = [c for c in sub.columns if c not in base_first]
        sub = sub[[c for c in base_first if c in sub.columns] + sorted(others)]

        year_dir = os.path.join(out_root, f"{base_name}_parquet", str(y))
        os.makedirs(year_dir, exist_ok=True)
        # 파일명: processed_data_{ticker}_{year}.parquet
        out_path = os.path.join(year_dir, f"{base_name}_{ticker}_{y}.parquet")
        
        sub.to_parquet(out_path, engine='pyarrow', compression='snappy', index=True)

        if DEBUG:
            if not os.path.exists(out_path):
                 print(f"[WARN] [{ticker}] (Save) {out_path} 파일이 저장되지 않았습니다!")
            else:
                 print(f"[{ticker}] (Save) {out_path} 저장 완료.")


def sync_local_to_gdrive(local_root: str, gdrive_root: str, base_name: str):
    """
    로컬 폴더를 ZIP으로 고속 압축하여 GDrive로 전송 (I/O 병목 해결)
    """
    dir_name = f"{base_name}_parquet"
    src_dir = os.path.join(local_root, dir_name)
    
    # 로컬에서 생성될 압축 파일 경로 (확장자 제외하고 입력)
    local_archive_base = os.path.join(local_root, dir_name)
    local_archive_file = local_archive_base + ".zip"
    
    # GDrive 목적지 파일 경로
    gdrive_dest_file = os.path.join(gdrive_root, f"{dir_name}.zip")

    if not os.path.exists(src_dir):
        print(f"[SYNC] ⚠️ 소스 디렉토리가 없습니다: {src_dir}")
        return

    print(f"[SYNC] 📦 데이터 압축 시작 (Target: {local_archive_file})...")
    t0 = time.time()
    
    try:
        # 1. 압축 (ZIP) - root_dir와 base_dir을 분리하여 폴더 구조 깔끔하게 유지
        shutil.make_archive(
            base_name=local_archive_base, 
            format='zip', 
            root_dir=local_root, 
            base_dir=dir_name
        )
        compress_time = time.time() - t0
        file_size_mb = os.path.getsize(local_archive_file) / (1024 * 1024)
        print(f"[SYNC] ✓ 압축 완료 ({compress_time:.1f}초, {file_size_mb:.1f} MB)")

        # 2. GDrive 전송
        print(f"[SYNC] 🚀 Google Drive 업로드 시작: {gdrive_dest_file} ...")
        os.makedirs(gdrive_root, exist_ok=True)
        
        # 기존 파일이 있으면 덮어쓰기 위해 copy2 사용 (메타데이터 보존)
        shutil.copy2(local_archive_file, gdrive_dest_file)
        
        upload_time = time.time() - t0 - compress_time
        print(f"[SYNC] ✓ 전송 완료 (업로드: {upload_time:.1f}초). 총 소요시간: {time.time() - t0:.1f}초")
        
    except Exception as e:
        print(f"[SYNC] ❌ 동기화 실패: {e}")
        # 실패 시 불완전한 압축 파일 제거 시도
        if os.path.exists(local_archive_file):
            os.remove(local_archive_file)


def _read_macro_value_csv(path, fname):
    f = os.path.join(path, fname)
    if not os.path.exists(f): 
        return None
    df = pd.read_csv(f, parse_dates=True, index_col=0)
    df = df.rename_axis("날짜")
    return df

def _read_fred_vintages_as_pit(market_path, nice_name, start, end):
    """
    FRED *_vintages.csv (date, vintage_date, value) → 
    'vintage_date ≤ t' 조건을 만족하는 완전 PIT 일별 시계열로 변환.
    - 각 관측일(date)에 대해 최초 공시(첫 vintage)만 사용
    - first_vintage_date 이후로만 값이 보이고 그 전에는 NaN
    """
    vint_path = os.path.join(market_path, f"{nice_name}_vintages.csv")
    if not os.path.exists(vint_path):
        print(f"[WARN] FRED vintages 파일 없음: {vint_path}")
        return None

    df_long = pd.read_csv(vint_path)
    if df_long.empty:
        print(f"[WARN] {nice_name}_vintages.csv 가 비어있습니다.")
        return None

    # 날짜 파싱
    df_long["date"] = pd.to_datetime(df_long["date"], errors="coerce")
    df_long["vintage_date"] = pd.to_datetime(df_long["vintage_date"], errors="coerce")
    mask = df_long["date"].notna() & df_long["vintage_date"].notna()
    df_long = df_long.loc[mask].copy()
    if df_long.empty:
        print(f"[WARN] {nice_name}: 유효한 date/vintage_date 가 없습니다.")
        return None

    # (1) 관측일(date)별 "최초 공시"만 사용 → 리비전 제거
    df_first = (
        df_long.sort_values(["date", "vintage_date"])
               .groupby("date", as_index=False)
               .head(1)
               .rename(columns={"vintage_date": "first_vintage_date"})
    )

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    # (2) first_vintage_date 기준으로, end 이전까지만 사용
    df_first = df_first.loc[df_first["first_vintage_date"] <= end_ts].copy()
    if df_first.empty:
        print(f"[WARN] {nice_name}: {end} 이전에 발표된 데이터가 없습니다.")
        return None

    # (3) 발표일 기준 정렬
    df_first = df_first.sort_values("first_vintage_date")

    # 중복된 발표일(first_vintage_date) 처리
    # 만약 서로 다른 관측일(date)의 데이터가 같은 날 최초 공시된 경우,
    # 가장 최신 관측일(date)의 데이터만 남깁니다.
    # 1. (first_vintage_date, date) 순서로 정렬
    df_first = df_first.sort_values(["first_vintage_date", "date"])
    # 2. first_vintage_date 기준 중복 제거 (keep='last'로 최신 date 행 보존)
    df_first = df_first.drop_duplicates(subset=["first_vintage_date"], keep="last")

    # (4) 발표일(first_vintage_date) 기준으로 계단함수 생성 후,
    #     영업일(business day) 인덱스에 맞춰 ffill
    first_release = df_first["first_vintage_date"].min()
    # 시작일은 'start'와 '첫 발표일' 중 더 이른 날
    start_all = min(start_ts, first_release)

    bidx_full = pd.bdate_range(start_all, end_ts, freq="B")
    s_release = (
        df_first.set_index("first_vintage_date")["value"]
                .sort_index()
                .reindex(bidx_full)
                .ffill()
    )

    # 우리가 실제로 쓰는 건 [start, end] 구간
    s_pit = s_release.loc[start_ts:end_ts]

    out = pd.DataFrame(s_pit)
    out.columns = [nice_name]
    out.index.name = "날짜"
    return out


def load_macro_plus_once(market_path, start, end):
    """
    datacl.py가 만든 FRED *_vintages.csv를 사용해
    - (1) 각 관측일(date)에 대해 '최초 공시(first vintage)'만 선택하고
    - (2) 발표일(first_vintage_date) 기준으로 주식 영업일에 맞춰 ffill 하여
      완전 Point-in-Time(PIT) 매크로 시계열을 생성한다.
    """
    # flat CSV가 아니라, long vintage 파일 기반으로 읽어올 시리즈 목록
    fred_series = [
        "UST10Y","UST2Y","FEDFUNDS","EFFR",
        "HY_OAS","IG_OAS","CPI","PPI","BREAKEVEN10Y",
        "EPU_US","UNRATE","GDP_REAL_QOQ"
    ]

    parts = []
    for name in fred_series:
        df = _read_fred_vintages_as_pit(market_path, name, start, end)
        if df is None:
            continue
        parts.append(df)

    if not parts:
        print("[WARN] load_macro_plus_once: 사용할 FRED 시리즈가 없습니다.")
        return None

    # 모든 시리즈를 '날짜'(영업일) 기준으로 outer-join
    out = pd.concat(parts, axis=1).sort_index()

    # === 파생 피처 생성 ===
    # (1) 금리/수익률 곡선
    if {"UST10Y","UST2Y"}.issubset(out.columns):
        out["UST_SLOPE_10Y2Y"] = out["UST10Y"] - out["UST2Y"]

    # (2) 신용스프레드 & 차이
    if {"HY_OAS","IG_OAS"}.issubset(out.columns):
        out["OAS_DIFF_HY_IG"] = out["HY_OAS"] - out["IG_OAS"]

    # (3) 인플레 YoY (%): CPI/PPI는 index값 → 12개월(≈252영업일) 전 대비 변화율
    if "CPI" in out.columns:
        out["CPI_YOY"] = out["CPI"].pct_change(252) * 100.0
    if "PPI" in out.columns:
        out["PPI_YOY"] = out["PPI"].pct_change(252) * 100.0

    # (4) 정책금리 변화(모멘텀): 일간(EFFR) 21영업일 변화
    if "EFFR" in out.columns:
        out["EFFR_d21"] = out["EFFR"].diff(21)

    # (5) 기대 인플레(브레이크이븐) 단기 변화
    if "BREAKEVEN10Y" in out.columns:
        out["BREAKEVEN10Y_d21"] = out["BREAKEVEN10Y"].diff(21)

    # (6) EPU/실업률/GDP 3개월(≈63영업일) 변화
    for col in ["EPU_US","UNRATE","GDP_REAL_QOQ"]:
        if col in out.columns:
            out[f"{col}_d63"] = out[col].diff(63)

    # (7) 금리/스프레드 단기변화(21영업일)
    for col in ["UST10Y","UST2Y","UST_SLOPE_10Y2Y","HY_OAS","IG_OAS","OAS_DIFF_HY_IG"]:
        if col in out.columns:
            out[f"{col}_d21"] = out[col].diff(21)

    # ---- 결측 마무리: "과거 → 미래" 방향으로만 ffill (bfill 제거) ----
    #   → 발표 이전 구간은 계속 NaN으로 유지되어, "발표 전 정보"를 보지 않게 됨.
    out = out.ffill()

    # '신호'(YOY, d63 등)는 남기고 '잡음'(원본 Level)은 제거합니다.
    cols_to_drop_noise = [
        'CPI', 'PPI',              # (신호: CPI_YOY, PPI_YOY 사용)
        'UNRATE', 'GDP_REAL_QOQ',  # (신호: UNRATE_d63, GDP_REAL_QOQ_d63 사용)
        'FEDFUNDS',                # (신호: 일간 데이터인 EFFR, EFFR_d21 사용)
        'HY_OAS_bps', 'IG_OAS_bps' # (중복 별칭 제거용, 실제 컬럼명과 다르면 무시됨)
    ]
    cols_to_drop_exist = [c for c in cols_to_drop_noise if c in out.columns]
    if cols_to_drop_exist:
        out.drop(columns=cols_to_drop_exist, inplace=True, errors='ignore')
        print(f"[Signal Clean] 'load_macro_plus_once' 원본(Level) 피처 {len(cols_to_drop_exist)}개 제거 완료.")

    return out


def load_macro_data_once(macro_path, start_date, end_date):
    """
    스트리밍 시작 전, 매크로 데이터를 한 번만 로드하여 병합용 DataFrame 생성
    """
    print(f"[INFO] 매크로 데이터(KOSPI, KOSDAQ, S&P500, 환율, VIX, KOSPI200 선물, KODEX200) 선-로드 중... ({start_date} ~ {end_date})")
    macro_df = pd.DataFrame(index=pd.date_range(start_date, end_date, freq='D'))
    
    # datacl.py에서 정의한 매크로 에셋
    macro_assets = {
        "KOSPI": "KOSPI.csv",
        "KOSDAQ": "KOSDAQ.csv",
        "SP500": "SP500.csv",
        "USD_KRW": "USD_KRW.csv",
        "VIX": "VIX.csv",
        "KOSPI200_FUT": "KOSPI200_FUT.csv",
        "KODEX200": "KODEX200.csv",
    }
    
    for name, filename in macro_assets.items():
        filepath = os.path.join(macro_path, filename)
        if not os.path.exists(filepath):
            print(f"[Warning] 매크로 파일 없음: {filepath}")
            continue
        
        df = pd.read_csv(filepath, index_col=0)

        # 1) 날짜 파싱
        idx = pd.to_datetime(df.index, errors="coerce", utc=False)
        mask = ~idx.isna()
        if not mask.all():
            print(f"[Fix] {name}: 파싱 실패 {(~mask).sum()}건 → 제거")
        df = df.loc[mask].copy()
        df.index = idx[mask]

        # 2) 인덱스 중복 제거 + 정렬
        dup_cnt = int(df.index.duplicated(keep=False).sum())
        if dup_cnt:
            print(f"[Fix] {name}: 중복 날짜 {dup_cnt}건 → keep='last'")
        df = df[~df.index.duplicated(keep='last')].sort_index()

        # 3) 모든 컬럼 수치형으로 캐스팅
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # 4) 자산별로 사용할 컬럼 선택 & macro_df에 매핑
        if name in ("KOSPI", "KOSDAQ", "SP500", "USD_KRW", "VIX"):
            col_to_use = '종가' if name in ('KOSPI', 'KOSDAQ') else 'Close'
            if col_to_use not in df.columns:
                print(f"[Warning] {name}에 '{col_to_use}' 컬럼이 없습니다. (컬럼: {df.columns})")
                continue
            macro_df[f'{name}_Close'] = df[col_to_use].reindex(macro_df.index)

        elif name == "KOSPI200_FUT":
            # 선물 미결제약정(OI)
            if '미결제약정' not in df.columns:
                print(f"[Warning] {name}에 '미결제약정' 컬럼이 없습니다. (컬럼: {df.columns})")
                continue
            macro_df['KOSPI200_FUT_OI'] = df['미결제약정'].reindex(macro_df.index)

        elif name == "KODEX200":
            # ETF 거래량
            if '거래량' not in df.columns:
                print(f"[Warning] {name}에 '거래량' 컬럼이 없습니다. (컬럼: {df.columns})")
                continue
            macro_df['KODEX200_VOL'] = df['거래량'].reindex(macro_df.index)

    # 매크로 데이터 휴일 처리: ffill
    macro_df = macro_df.infer_objects(copy=False)
    macro_df.ffill(inplace=True)
    return macro_df


def process_ticker_file(file, config, macro_all, benchmark_close):
    # EPSILON 정의를 최상단으로 확실하게 위치시킴 (Scope Error 방지)
    EPSILON = 1e-9  
    
    """
    병렬 처리를 위한 Worker 함수
    단일 'file'에 대해 기존 stream_pipeline의 for 루프 내부 로직을 수행합니다.
    """
    
    # config 딕셔너리에서 필요한 설정값들을 가져옵니다.
    ticker = os.path.basename(file).split('.')[0]
    
    # 설정값 로드
    PATH_FUNDAMENTAL = config["PATH_FUNDAMENTAL"]
    PATH_TRADING = config["PATH_TRADING"]
    # 단일 캔들 목표 수익률 설정
    LABEL_THRESHOLD = config.get("LABEL_PROFIT_THRESHOLD", 0.02)
    USE_PRUNE = config["USE_PRUNE"]
    USE_DOWNCAST = config["USE_DOWNCAST"]
    DEBUG = config["DEBUG"]
    DBG_DIR = config["DBG_DIR"]
    DIAG_ENABLED = config["DIAG_ENABLED"]
    OUTPUT_PATH = config["OUTPUT_PATH"]
    OUTPUT_BASE_NAME = config["OUTPUT_BASE_NAME"]
    SCHEMA_ENSURE_NAN = config["SCHEMA_ENSURE_NAN"]
    SCHEMA_ENSURE_ZERO = config["SCHEMA_ENSURE_ZERO"]
    INTENT_FEATURES = config["INTENT_FEATURES"] # (스키마 검증용)
    
    g = None # finally 블록을 위한 초기화
    
    try:
        if DEBUG:
            # (로그 객체 대신 print 사용 - 병렬처리 시 로깅은 복잡함)
            print(f"[{ticker}] --- 처리 시작 ---")
        
        # ---------------------------------------------------------
        # 1. [기본] OHLCV 로드 및 전처리
        # ---------------------------------------------------------
        g = pd.read_csv(file)
        
        g['날짜'] = _coerce_date_index(g['날짜'])
        g = g.loc[~g['날짜'].isna()].copy() # NaT 제거
        
        g.set_index('날짜', inplace=True)
        for c in g.columns:
            if c != 'ticker': g[c] = pd.to_numeric(g[c], errors='coerce')
        g.sort_index(kind='mergesort', inplace=True)
        
        # 이름 변경을 먼저 해야 'open', 'close' 등을 사용할 수 있음
        OHLCV_RENAME_MAP = {"시가":"open","고가":"high","저가":"low","종가":"close","거래량":"volume"}
        g.rename(columns=OHLCV_RENAME_MAP, inplace=True)

        # 불필요한 Raw 데이터 컬럼 제거
        drop_candidates = ['등락률', '대비', '거래대금'] 
        g.drop(columns=[c for c in drop_candidates if c in g.columns], inplace=True, errors='ignore')

        # Is_Tradable 계산 위치 이동 - 데이터 삭제 전 원본 사용
        if '거래대금' in g.columns:
            daily_amt = g['거래대금']
        else:
            daily_amt = g['close'] * g['volume']
        # 120일 평균 (데이터가 짧아도 있는 만큼만 계산하도록 min_periods=1 유지)
        ma120_amt = daily_amt.rolling(window=120, min_periods=1).mean().fillna(0)
        MIN_AMT = 3_000_000_000
        g['Is_Tradable'] = (ma120_amt >= MIN_AMT).astype('int8')

        # [데이터 클리닝] 가격이 0 이하인 행 제거 (로그 경고 원천 차단 + 데이터 무결성)
        # 이제 컬럼명이 영문(open, close...)으로 바뀌었으므로 안전하게 체크 가능
        if {'open','high','low','close'}.issubset(g.columns):
            mask_valid = (g['open'] > 0) & (g['high'] > 0) & (g['low'] > 0) & (g['close'] > 0)
            if (~mask_valid).any():
                if DEBUG:
                    # 제거되는 개수 확인용 (필요시 주석 해제)
                    # print(f"[{ticker}] 0원 데이터 {(~mask_valid).sum()}개 제거")
                    pass
                g = g.loc[mask_valid].copy()
        
        if g.empty:
            return (ticker, "SKIP: All rows had 0 price or empty")
        
        # 날짜 갭 확인 로직 추가
        # 인덱스가 날짜라고 가정
        day_diff = g.index.to_series().diff().dt.days
        gap_mask = day_diff > 5 # 5일 이상 데이터가 비어있으면 갭으로 간주

        if DEBUG:
            print(f"[{ticker}] (1. OHLCV 로드) Shape: {g.shape}, "
                  f"기간: {g.index.min().date()} ~ {g.index.max().date()}")
        
        oh = {"open","high","low","close","volume"}
        if not oh.issubset(g.columns):
            print(f"[ERROR] [OHLCV MISSING] {ticker} need={oh} have={set(g.columns)}")
            if DIAG_ENABLED: diag_write("skip_no_ohlcv", {"have": list(g.columns)}, ticker)
            return (ticker, "SKIP: No OHLCV") # continue 대신 return
        
        # (dassert, df_quick_report 등은 log 객체 대신 print를 쓰도록 수정하거나 제거 필요)
        
        if DIAG_ENABLED:
            diag_write("ticker_begin", {"rows": int(len(g))}, ticker)
            feature_snapshot(g, "A_after_ohlcv", ticker)

        
        
        # ==================================================================
        # 3. 수급 로드 & Intensity Normalization
        # ==================================================================
        
        # 파일 확장자 대소문자 호환성 처리 (.csv, .CSV 확인)
        trading_file = os.path.join(PATH_TRADING, f"{ticker}.csv")
        if not os.path.exists(trading_file):
            trading_file_upper = os.path.join(PATH_TRADING, f"{ticker}.CSV")
            if os.path.exists(trading_file_upper):
                trading_file = trading_file_upper
            else:
                trading_file = None # 파일 없음

        # 표준 수급 컬럼
        standard_trading_cols = ["외국인", "기관", "개인"]
        g['Trading_Missing_Flag'] = 0.0

        if trading_file is not None:
            try:
                trading_df = pd.read_csv(trading_file)
                
                # 날짜 파싱 및 정리
                trading_df["날짜"] = _coerce_date_index(trading_df["날짜"])
                trading_df = trading_df.loc[~trading_df["날짜"].isna()].copy()
                trading_df.set_index("날짜", inplace=True)

                # 중복 제거 (Keep Last)
                if trading_df.index.duplicated().any():
                    trading_df = trading_df[~trading_df.index.duplicated(keep='last')]
                
                RENAME_MAP = {
                    "기관합계": "기관", "기관계": "기관", "외국인합계": "외국인",
                    "개인": "개인", "개인투자자": "개인", "연기금등": "기관",
                    "순매수거래대금_외국인": "외국인", "순매수거래대금_기관": "기관", "순매수거래대금_개인": "개인",
                }
                exist_map = {src: dst for src, dst in RENAME_MAP.items() if src in trading_df.columns}
                if exist_map: trading_df.rename(columns=exist_map, inplace=True)
                
                # 없는 컬럼 채우기
                for c in standard_trading_cols:
                    if c not in trading_df.columns: trading_df[c] = 0.0
                
                trading_cols_exist = [c for c in standard_trading_cols if c in trading_df.columns]
                trading_df = trading_df[trading_cols_exist]

                if trading_cols_exist:
                    # Left Join
                    g = pd.merge(g, trading_df, left_index=True, right_index=True, how="left")
                    g[trading_cols_exist] = g[trading_cols_exist].fillna(0)

                    # Intensity Normalization
                    # 수급 금액(Net Buying Amount)을 그대로 쓰면 시총 큰 종목만 값이 커짐.
                    # 거래대금(Turnover) 대비 순매수 비중으로 변환하여 "매수 강도"를 측정.
                    # Intensity = NetBuyAmount / (DailyTurnover + Epsilon)
                    
                    daily_turnover = (g['close'] * g['volume']).replace(0, np.nan)
                    # 거래대금 이동평균(5일)을 사용하여 분모 안정화
                    turnover_sma = daily_turnover.rolling(5, min_periods=1).mean().fillna(daily_turnover)

                    for col in trading_cols_exist:
                        # 1. Intensity Calculation (Ratio)
                        # -1.0 ~ 1.0 (이론상)
                        raw_flow = g[col]
                        intensity = raw_flow / (turnover_sma + EPSILON)
                        
                        # 2. Robust Z-Score Normalization (Self-Relative)
                        # 과거 60일 대비 현재 매수 강도가 얼마나 이례적인가?
                        # Using the new Numba helper directly here for speed
                        z_name = f"{col}_Z_60"
                        intensity_vals = intensity.values.astype(np.float64)
                        g[z_name] = _calc_robust_zscore_numba(intensity_vals, window=60).astype('float32')
                        
                        # Replace original column with the Intensity Ratio (Slightly smoothed)
                        # We clip extreme ratios to +/- 0.5 (meaning net buy was 50% of total turnover)
                        g[col] = intensity.clip(-0.5, 0.5).astype('float32')

                    # 누적 피처 (Intensity Sum)
                    # "최근 5일간 거래대금 대비 순매수 누적 비율"
                    for col in ["외국인", "기관", "개인"]:
                        if col in g.columns:
                            g[f"{col}_SUM5"] = g[col].rolling(5).sum().astype("float32")
                            g[f"{col}_SUM20"] = g[col].rolling(20).sum().astype("float32")

                    # NaN Handling
                    cols_to_clean = trading_cols_exist + [f"{c}_Z_60" for c in trading_cols_exist] + \
                                    [f"{c}_SUM5" for c in trading_cols_exist] + \
                                    [f"{c}_SUM20" for c in trading_cols_exist]
                    
                    g[cols_to_clean] = g[cols_to_clean].fillna(0).replace([np.inf, -np.inf], 0)

            except Exception as e:
                print(f"[{ticker}] [TRD-ERR] 수급 처리 중 오류: {e}")
                trading_file = None
        
        if trading_file is None:
            g['Trading_Missing_Flag'] = 1.0
            for c in standard_trading_cols:
                g[c] = 0.0
                g[f"{c}_Z_60"] = 0.0
                g[f"{c}_SUM5"] = 0.0
                g[f"{c}_SUM20"] = 0.0

        # ==================================================================
        # 4. 매크로 병합 및 시점 정렬 (Alignment)
        # ==================================================================
        if macro_all is not None:
            # 우리가 실제로 피처에서 쓸 매크로 컬럼만 선별
            macro_cols = [
                "KOSPI_Close",      # 베타/REL_RET 기준
                "SP500_Close",      # US_MKT_RET_1
                "USD_KRW_Close",    # FX_RET_1
                "UST10Y", "UST2Y",  # YIELD_SPREAD
                "UST_SLOPE_10Y2Y",  # 없으면 위 둘로 직접 계산
            ]
            macro_cols = [c for c in macro_cols if c in macro_all.columns]

            if macro_cols:
                # 1. Merge (Left Join)
                g = pd.merge(
                    g,
                    macro_all[macro_cols],
                    left_index=True,
                    right_index=True,
                    how="left",
                )

                # 2. Sort Index (시계열 순서 보장)
                g.sort_index(inplace=True)

                # 3. 매크로 결측은 "과거 → 미래" 방향으로만 ffill
                for c in macro_cols:
                    g[c] = g[c].ffill()

            # 4. Return Calculation (로그 수익률)
            # 오늘 종가(t) / 어제 종가(t-1) -> 오늘의 수익률(t)
            # 이 값은 장 마감 후에 확정되므로, 다음날 시가 진입 전략에 사용 가능합니다.
            g['r_stock'] = np.log(g['close'] / g['close'].shift(1) + 1e-9)
            # 갭이 있는 행은 NaN 처리
            g.loc[gap_mask, 'r_stock'] = np.nan
            
            # KOSPI_Close가 없으면 r_market 계산 불가 -> NaN 처리
            if 'KOSPI_Close' in g.columns:
                g['r_market'] = np.log(g['KOSPI_Close'] / g['KOSPI_Close'].shift(1) + 1e-9)
                
                # 5. Rolling Beta Calculation (Numba Optimized)
                stock_ret_vals = g['r_stock'].values
                market_ret_vals = g['r_market'].values
                
                # _calc_rolling_beta_numba는 (i-window ~ i)를 슬라이싱하므로
                # 현재 행(i)의 수익률을 포함합니다. 이는 '오늘 장 마감 기준 베타'이므로 적절합니다.
                beta_vals = _calc_rolling_beta_numba(stock_ret_vals, market_ret_vals, window=60)
                g['BETA_60'] = beta_vals.astype('float32')
                
                # 6. Alpha Calculation
                # 20일 누적 수익률 계산
                g['log_ret_20_stock'] = g['r_stock'].rolling(20).sum()
                g['log_ret_20_market'] = g['r_market'].rolling(20).sum()
                
                beta_filled = g['BETA_60'].fillna(1.0)
                g['ALPHA_20'] = (g['log_ret_20_stock'] - (beta_filled * g['log_ret_20_market'])).astype('float32')
                
                # 잔차 변동성 (Idiosyncratic Volatility)
                daily_resid = g['r_stock'] - (beta_filled * g['r_market'])
                g['IDIOSYNCRATIC_VOL_20'] = daily_resid.rolling(20, min_periods=10).std().astype('float32')

                # REL_RET_* 및 Leak Hunter 입력으로 사용할 KOSPI_Close 는 남겨둔다.
                drop_targets = ['r_stock', 'r_market',
                                'log_ret_20_stock', 'log_ret_20_market']
                g.drop(columns=[c for c in drop_targets if c in g.columns],
                       inplace=True, errors='ignore')
            else:
                 # 매크로(KOSPI)가 아예 없으면 관련 피처 NaN
                 g['BETA_60'] = np.nan
                 g['ALPHA_20'] = np.nan
                 g['IDIOSYNCRATIC_VOL_20'] = np.nan
        else:
            # 매크로 데이터가 없을 경우 NaN 처리 (스키마 유지)
            g['BETA_60'] = np.nan
            g['ALPHA_20'] = np.nan
            g['IDIOSYNCRATIC_VOL_20'] = np.nan
        
        # (기존의 단순 KOSPI_RET20, DAY_SIN 등은 아예 계산하지 않음으로써 제거)
        # ==================================================================

        if DIAG_ENABLED: feature_snapshot(g, "B_after_merge", ticker)

        g['ticker'] = ticker  # 티커 추가

        # 5. [기존] 피처 계산
        if DIAG_ENABLED: before_cols = set(g.columns)
        
        # (log 객체 대신 print를 사용하도록 compute_features_inplace 내부 수정 필요하나,
        #  일단은 log 객체 없이 실행되도록 함)
        # 누수 감지 기능이 포함된 래퍼 함수 호출
        # 5. 피처 계산 (누수 탐지 기능이 내장된 래퍼 호출)
        compute_features_inplace(g, ticker=ticker)
        enforce_canonical_names_inplace(g)
        
        # Cold Start 구간 삭제
        # 피처 생성 후, 충분한 데이터(예: 60일)가 쌓이지 않아 NaN인 초기 행들을 과감히 삭제
        # 레이블링(미래 참조) 전에 수행해야 데이터 정합성이 맞음
        g.dropna(inplace=True)
        
        if g.empty:
            return (ticker, "SKIP: Empty after feature dropna")
        
        if DEBUG:
            print(f"[{ticker}] (5. 피처 계산 완료) Shape: {g.shape}, "
                  f"신규 피처 샘플(RSI_14 mean): {g['RSI_14'].mean(skipna=True):.2f}")
        
        if DIAG_ENABLED:
            after_cols = set(g.columns)
            new_cols = sorted(list(after_cols - before_cols))
            diag_write("features_added", {"stage":"C_compute_features", "added":new_cols, "n_added":len(new_cols)}, ticker)
            bollinger_probe(g, "C_after_compute", ticker)
            feature_snapshot(g, "C_after_compute", ticker)
        
        # 6. 레이블링 (단순화된 로직 적용)
        # "다음날 시가 대비 종가가 2% 이상인가?"
        g = apply_labeling_simple(g, threshold=LABEL_THRESHOLD)
        
        # Safety Fix: 무한대(inf) 값 제거 (학습 폭발 방지)
        g.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # ==================================================================
        # 상장폐지(Delisting) 방어 로직 (생존 편향 해결)
        # ==================================================================
        # 전체 수집 종료일(오늘)과 해당 종목의 마지막 거래일 비교
        collection_end_date = config.get("COLLECTION_END_DATE")
        
        if collection_end_date is not None and not g.empty:
            last_date_stock = g.index[-1]
            # 안전 마진 30일 (휴장일 고려)
            # 30일 이상 데이터가 끊겨 있다면 상장폐지로 간주
            if (collection_end_date - last_date_stock).days > 30:
                # 마지막 행의 레이블이 NaN일 텐데(미래 데이터가 없어서),
                # 이를 0.0 (실패)으로 강제 주입하여 dropna에서 살아남게 함
                # (상장폐지는 주주에게 '실패' 경험이므로 0.0 처리)
                if pd.isna(g['label'].iloc[-1]):
                    # iloc을 사용하여 마지막 행의 'label' 컬럼 위치에 값 할당
                    lbl_idx = g.columns.get_loc('label')
                    g.iloc[-1, lbl_idx] = 0.0
                    
                    if DEBUG:
                        print(f"[{ticker}] 💀 상장폐지 감지 (Last: {last_date_stock.date()}) -> 마지막 레이블 0.0 강제 할당")
        # ==================================================================

        # 7. NaN 레이블 제거
        if DEBUG: shape_before_drop = g.shape
        g.dropna(subset=['label'], inplace=True) 
        if DEBUG:
            shape_after_drop = g.shape
            dropped_rows = shape_before_drop[0] - shape_after_drop[0]
            print(f"[{ticker}] (7. NaN 레이블 제거) Shape: {shape_after_drop}, 제거된 행: {dropped_rows}개")
        
        if g.empty:
            if DIAG_ENABLED: diag_write("ticker_empty_after_label", {"reason":"all_label_nan"}, ticker)
            return (ticker, "SKIP: Empty after labeling") # continue 대신 return
            
        g['label'] = g['label'].astype('float32') 

        # [수정 2: 기존 Is_Tradable 계산 로직 제거됨 - 상단으로 이동]
        
        # [Optional] 데이터가 너무 적은 경우(상장 초기 등) 아예 분석 불가한 경우만 예외적으로 Skip
        # 단, 여기서는 '데이터 갯수' 자체를 체크하는 것이지, '조건 미달'을 체크하는 게 아님.
        if len(g) < 60: 
            return (ticker, "SKIP: Insufficient History (<60 rows)")
        
        # (삭제 로직 제거됨)
        # g.drop(columns=['Amount_MA120'], ...) -> 불필요

        if DIAG_ENABLED: feature_snapshot(g, "D_before_prune", ticker)

        # 8. Prune
        if USE_PRUNE:
            # (prune_columns_inplace 내부의 log.info/warn/error는 print로 변경해야 함)
            prune_columns_inplace(g) # (일단 log 객체 없이 실행)
            if DEBUG: print(f"[{ticker}] (8. Prune 완료) Shape: {g.shape}")
        
        # 9. Downcast
        if USE_DOWNCAST:
            downcast_numeric_inplace(g)
            if DEBUG: print(f"[{ticker}] (9. Downcast 완료) Shape: {g.shape}")

        if DIAG_ENABLED:
            bollinger_probe(g, "E_after_prune", ticker)
            feature_snapshot(g, "E_after_prune", ticker)
            feature_snapshot(g, "F_before_assert", ticker)

        # ... (기존 스키마 보장 및 Strict 모드 로직 동일) ...
        base_cols = {'open','high','low','close','volume','ticker','label','Is_Tradable','Trading_Missing_Flag'}
        feature_cols_now = [c for c in g.columns if c not in base_cols]
        EXPECTED_FEATURES = list(INTENT_FEATURES)
        nontech_keep = [c for c in g.columns if c not in base_cols and c not in INTENT_FEATURES]
        missing_tech = set(EXPECTED_FEATURES).difference(g.columns)
        extra_items  = set(feature_cols_now).difference(INTENT_FEATURES | set(nontech_keep))
        if missing_tech or extra_items:
            print(f"[WARN] [SCHEMA-FIX] {ticker} schema-fix with nontech preservation | "
                  f"miss_tech={sorted(list(missing_tech))[:5]}... "
                  f"extra_items={sorted(list(extra_items))[:5]}...")
            all_keep_cols = [c for c in base_cols if c in g.columns] + EXPECTED_FEATURES + nontech_keep
            g = g.reindex(columns=all_keep_cols, fill_value=np.nan)
        bad_cols = [c for c in EXPECTED_FEATURES if c in g and g[c].isna().all()]
        if len(bad_cols) >= 5:
            print(f"[WARN] [QUALITY] {ticker} has {len(bad_cols)} empty features: {bad_cols[:10]}...")
        strict = os.getenv("STRICT_FEATURE_SCHEMA", "0") == "1"
        if strict:
            feature_cols_final = [c for c in g.columns if c not in base_cols]
            expected_final_set = set(EXPECTED_FEATURES) | set(nontech_keep)
            missing_final = sorted(list(expected_final_set.difference(g.columns)))
            extra_final   = sorted(list(set(feature_cols_final).difference(expected_final_set)))
            expected_count = len(expected_final_set)
            if missing_final or extra_final:
                print(f"[ERROR] [ASSERT] {ticker} feature schema mismatch | "
                      f"missing={missing_final} extra={extra_final} "
                      f"expected_count={expected_count} now={len(feature_cols_final)}")
                raise AssertionError(f"Strict schema mismatch: missing/extra items found.")

        # 10. 저장
        save_by_year_parquet(g, OUTPUT_PATH, OUTPUT_BASE_NAME, ticker)
        
        if DEBUG:
            print(f"[{ticker}] (10. Parquet 저장 완료) --- 처리 종료 ---")
        
        if DIAG_ENABLED:
            diag_write("ticker_saved", {"n_features": len(feature_cols_now)}, ticker)

        # 성공 리턴
        return (ticker, "OK")
        
    except Exception as e:
        import traceback
        # 오류 상세 내용 캡처
        tb_str = traceback.format_exc()
        
        # 로그에는 간략히 출력하되, traceback 일부를 포함
        print(f"[ERROR] [{ticker}] 처리 중 심각한 오류 발생: {e}\n{tb_str}")
        
        if DIAG_ENABLED:
            diag_write("ticker_error", {"err": str(e), "trace": tb_str}, ticker)
        
        # 리턴 값에도 에러 타입을 명시하여 집계 시 확인 가능하게 함
        return (ticker, f"Error: {str(e)}")
    finally:
        # g가 DataFrame일 경우 메모리 해제
        if isinstance(g, pd.DataFrame):
            del g


def stream_pipeline():
    start_time = time.time()
    pd.set_option("mode.copy_on_write", True)

    ensure_dir(DBG_DIR)
    # (주의: log 객체는 병렬 처리 시 안전하게 전달/사용하기 어려움)
    print(f"[CONFIG] OUTPUT_BASE_NAME={OUTPUT_BASE_NAME}, USE_PRUNE={USE_PRUNE}, USE_DOWNCAST={USE_DOWNCAST}")

    ohlcv_files = glob.glob(os.path.join(PATH_OHLCV, "*.csv"))
    if not ohlcv_files:
        print(f"[Error] '{PATH_OHLCV}'에 CSV가 없습니다.")
        return
        
    TICKER_FILTER = os.getenv("RRE_TICKER_FILTER")
    MAX_TICKERS = int(os.getenv("RRE_MAX_TICKERS", "0")) or None
    
    if TICKER_FILTER:
        allow = set([t.strip() for t in TICKER_FILTER.split(",") if t.strip()])
        ohlcv_files = [p for p in ohlcv_files if os.path.splitext(os.path.basename(p))[0] in allow]
    if MAX_TICKERS:
        ohlcv_files = ohlcv_files[:MAX_TICKERS]

    # ===== 매크로 데이터 선-로드 (기존과 동일) =====
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    start_date_str = "2010-01-01"
    base_macro = load_macro_data_once(PATH_MACRO, start_date_str, end_date)
    macro_plus = load_macro_plus_once(PATH_MACRO, start_date_str, end_date)
    
    if macro_plus is not None and base_macro is not None:
        print(f"[INFO] 신규 매크로 피처 {macro_plus.shape[1]}개 + 기본 매크로 {base_macro.shape[1]}개 병합 중...")
        macro_all = pd.concat([base_macro, macro_plus], axis=1)
    elif base_macro is not None:
        macro_all = base_macro
    elif macro_plus is not None:
        macro_all = macro_plus
    else:
        macro_all = None
    
    benchmark_close = load_benchmark_series(BENCHMARK_CSV_PATH)
    
    # out_root는 OUTPUT_PATH (즉, LOCAL_OUTPUT_PATH)를 사용
    out_root = OUTPUT_PATH
    base_name = OUTPUT_BASE_NAME

    # 병렬 처리를 위한 설정값 패키징
    # DEBUG, log 등은 worker 함수에서 직접 사용하기 어려우므로, 
    # DEBUG 플래그만 전달하여 worker 내부에서 print를 사용하도록 합니다.
    
    # 오늘 날짜(또는 수집 종료 기준일)를 config에 추가
    collection_end_dt = pd.to_datetime(end_date)
    
    static_config = {
        "PATH_FUNDAMENTAL": PATH_FUNDAMENTAL,
        "PATH_TRADING": PATH_TRADING,
        "LABEL_PROFIT_THRESHOLD": LABEL_PROFIT_THRESHOLD, 
        "USE_PRUNE": USE_PRUNE,
        "USE_DOWNCAST": USE_DOWNCAST,
        "DEBUG": DEBUG, # DEBUG 플래그 전달
        "DBG_DIR": DBG_DIR,
        "DIAG_ENABLED": DIAG_ENABLED,
        "DIAG_DIR": DIAG_DIR,
        "OUTPUT_PATH": OUTPUT_PATH, # LOCAL_OUTPUT_PATH가 전달됨
        "OUTPUT_BASE_NAME": OUTPUT_BASE_NAME,
        "SCHEMA_ENSURE_NAN": SCHEMA_ENSURE_NAN,
        "SCHEMA_ENSURE_ZERO": SCHEMA_ENSURE_ZERO,
        "INTENT_FEATURES": INTENT_FEATURES, # (Worker의 스키마 검증용)
        "COLLECTION_END_DATE": collection_end_dt, 
    }
    
    # functools.partial을 사용해 worker 함수에 고정 인자(설정, 매크로 데이터)를 미리 바인딩
    worker_func = partial(process_ticker_file, 
                          config=static_config, 
                          macro_all=macro_all, 
                          benchmark_close=benchmark_close)

    # 병렬 처리 루프
    print(f"[Streaming] 총 {len(ohlcv_files)}개 종목 병렬 처리 시작 (CPU: {os.cpu_count()}개 사용)...")
    ok_cnt = 0
    err_cnt = 0
    skip_cnt = 0
    
    with stage_timer("stream_total"):
        try:
            # CPU 코어 수만큼 풀 생성 (os.cpu_count()는 가상 코어 포함)
            # Colab 환경 등에서 코어 수를 제한해야 할 수 있습니다. (예: max(1, os.cpu_count() // 2))
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                
                # pool.imap_unordered: 작업이 완료되는 순서대로 결과를 반환 (더 효율적)
                # tqdm으로 감싸서 진행 상황 표시
                results = list(tqdm(pool.imap_unordered(worker_func, ohlcv_files), 
                                    total=len(ohlcv_files), 
                                    desc="병렬 스트리밍 처리"))
            
            # 병렬 처리 완료 후 결과 집계
            for ticker, status in results:
                if status == "OK":
                    ok_cnt += 1
                elif status.startswith("SKIP"):
                    skip_cnt += 1
                    if DEBUG: print(f"[{ticker}] 건너뜀: {status}")
                else:
                    err_cnt += 1
                    print(f"[ERROR] [{ticker}] 처리 실패: {status}") # 실패한 작업만 에러 로그 출력

        except KeyboardInterrupt:
            tqdm.write("[INFO] 사용자가 중단했습니다.")
            if 'pool' in locals():
                pool.terminate() # 풀 강제 종료
        finally:
            with suppress(Exception):
                tqdm._instances.clear()

    # 로컬 저장 완료 후 GDrive로 일괄 복사
    elapsed = time.time() - start_time
    print(f"✓ 스트리밍 완료: {ok_cnt}개 저장, {skip_cnt}개 건너뜀, {err_cnt}개 오류. "
          f"(총 {len(ohlcv_files)}개 시도), 소요 {elapsed:.1f}초")
    print(f"[Output-LOCAL] {os.path.join(out_root, base_name + '_parquet')}/*/<여러 parquet 파일>")

    # 신규: 로컬 -> GDrive 일괄 복사
    try:
        sync_local_to_gdrive(LOCAL_OUTPUT_PATH, FINAL_OUTPUT_PATH, OUTPUT_BASE_NAME)
        print(f"[Output-GDRIVE] {os.path.join(FINAL_OUTPUT_PATH, OUTPUT_BASE_NAME + '_parquet')}/*/<여러 parquet 파일>")
    except Exception as e:
        print(f"[SYNC ERROR] GDrive로 복사 중 오류 발생: {e}")


def main():
    # 스트리밍 파이프라인 (권장)
    stream_pipeline()

if __name__ == "__main__":
    main()