#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프로젝트: RRE (Robust Rolling Engine)
파일설명: LightGBM 모델 기반 통합 백테스팅 엔진 (run_backtest.py)
         - 데이터 로드 -> 전처리/추론 -> 시뮬레이션 -> 성과분석을 단일 파일로 수행
         - Look-ahead Bias(미래 참조) 원천 차단 설계
         - Vectorized Inference + Event-Driven Simulation
         - 로깅 및 데이터 정합성 검사 강화 버전
"""

import os
import sys
import glob
import time
import json
import gc
import shutil
import pickle
import traceback
import logging
import hashlib
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import pyarrow.parquet as pq

# 시각화 설정
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.unicode_minus'] = False

# 전역 로거 (main에서 초기화)
log = logging.getLogger("rre.backtest")

# 로거(Logger) 강화 (파일 저장 & 포맷 고도화)
def setup_logger(level: str = "DEBUG") -> logging.Logger:
    """콘솔 + 파일 양방향 로깅, 라인 넘버 포함"""
    logger = logging.getLogger("rre.backtest")

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    
    # 포맷 강화: 파일명과 라인번호 추가
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d ➤ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. 콘솔 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 2. 파일 핸들러 (backtest_debug.log 에 저장)
    file_handler = logging.FileHandler("backtest_debug.log", mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger

def _sha1_text(text: str) -> str:
    """간단한 SHA1 해시 (features 시그니처 확인용)."""
    m = hashlib.sha1()
    m.update(text.encode("utf-8"))
    return m.hexdigest()[:12]

# ==============================================================================
# 1. Config: 백테스트 환경 설정
# ==============================================================================
class Config:
    # --- 경로 설정 ---
    GDRIVE_MOUNT_PATH = "/content/drive/MyDrive/rre"
    SOURCE_DATA_PATH  = os.path.join(GDRIVE_MOUNT_PATH, "data")
    MODEL_ROOT_PATH   = os.path.join(GDRIVE_MOUNT_PATH, "model_lgbm")
    
    # 로컬 고속 처리를 위한 임시 경로
    LOCAL_DATA_PATH   = "/content/rre_backtest_data"
    OUTPUT_RESULT_DIR = "/content/backtest_results"

    # 파일명
    DATA_FILENAME     = "processed_data.pkl"
    RANK_FILENAME     = "rank_features_parquet.zip"
    MODEL_FILENAME    = "best_lgbm.txt"
    SCALER_FILENAME   = "scaler_lgbm.pkl"
    FEATURE_JSON      = "features.json"

    # --- 로깅 설정 ---
    LOG_LEVEL         = os.getenv("RRE_LOG_LEVEL", "DEBUG")

    # --- 백테스트 기간 설정 ---
    TEST_START_DATE   = "2024-01-01"
    TEST_END_DATE     = datetime.today().strftime("%Y-%m-%d")

    # --- 자금 및 포트폴리오 설정 ---
    INITIAL_CASH      = 100_000_000   # 초기 자본금 1억 원
    MAX_POSITIONS     = 5             # 최대 보유 종목 수 (분산 투자)
    
    # --- 비용 설정 (보수적 접근) ---
    FEE_RATE          = 0.000       # 수수료 (0.015%)
    TAX_RATE          = 0.0000        # 거래세 (0.20%, 매도 시)
    SLIPPAGE_RATE     = 0.0000        # 슬리피지 (0.50% 가정 - 호가 공백 고려)

    # --- 매매 전략 설정 ---
    BUY_THRESHOLD     = 0.2          # 매수 진입 확률 임계값 (이 점수 이상만 매수)
    
    # TP/SL 설정 (기본값)
    HOLDING_DAYS      = 20            # 최대 보유 기간
    TARGET_PROFIT_PCT = 1.07          # 익절: +15%
    STOP_LOSS_PCT     = 0.97          # 손절: -10%

    # [고급] ATR 기반 동적 TP/SL 사용 여부
    USE_DYNAMIC_EXIT  = False
    ATR_MULTIPLIER_TP = 3.0           # 익절 = ATR * 3
    ATR_MULTIPLIER_SL = 2.0           # 손절 = ATR * 2

# ==============================================================================
# 2. Utils: 헬퍼 함수
# ==============================================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_data_to_local(source_path, local_path, base_name="processed_data"):
    # 1. 기본 데이터 복사
    print(f"[System] 데이터 로컬 복사 중... ({source_path} -> {local_path})")
    zip_name = f"{base_name}_parquet.zip"
    source_zip = os.path.join(source_path, zip_name)
    local_zip = os.path.join(local_path, zip_name)
    extract_path = os.path.join(local_path, f"{base_name}_parquet")
    
    ensure_dir(local_path)
    
    if not os.path.exists(extract_path):
        if os.path.exists(source_zip):
            shutil.copy2(source_zip, local_zip)
            shutil.unpack_archive(local_zip, local_path)
            os.remove(local_zip)
        else:
             # 폴더 복사 시도 로직
             source_dir = os.path.join(source_path, f"{base_name}_parquet")
             if os.path.exists(source_dir):
                 shutil.copytree(source_dir, extract_path)

    # 2. 랭크 데이터 복사 로직
    rank_zip_src = os.path.join(source_path, Config.RANK_FILENAME)
    rank_zip_dst = os.path.join(local_path, Config.RANK_FILENAME)
    rank_extract_dir = os.path.join(local_path, "rank_features_parquet")

    if os.path.exists(rank_zip_src) and not os.path.exists(rank_extract_dir):
        print(f"[System] 랭크 피처 복사 중... ({Config.RANK_FILENAME})")
        shutil.copy2(rank_zip_src, rank_zip_dst)
        shutil.unpack_archive(rank_zip_dst, local_path)
        os.remove(rank_zip_dst)
    
    return extract_path

# ==============================================================================
# Utils: 결측치 처리 함수
# ==============================================================================
def run_causal_impute(df, feature_cols):
    """
    학습 코드와 동일한 Causal Imputation 로직.
    1. FFill (직전 값)
    2. Expanding Median Shift(1) (과거 누적 중앙값, 미래 참조 방지)
    3. 남은 NaN은 0으로 채움
    """
    chunks = []
    if 'ticker' in df.columns:
        grouper = df.groupby('ticker', group_keys=False)
    else:
        return df.fillna(0)

    print("[Brain] Applying Causal Imputation (FFill + Expanding Median)...")
    for _, group in tqdm(grouper, desc="Imputing", leave=False):
        group = group.sort_index()
        
        # 1. Forward Fill
        group[feature_cols] = group[feature_cols].ffill()
        
        # 2. Expanding Median (Shift 1 to avoid leakage)
        if group[feature_cols].isna().any().any():
            medians = group[feature_cols].expanding(min_periods=1).median().shift(1)
            group[feature_cols] = group[feature_cols].fillna(medians)
        
        chunks.append(group)
    
    if not chunks: 
        return df
    
    df_imputed = pd.concat(chunks)
    # 3. 남은 NaN은 0으로 채움
    df_imputed[feature_cols] = df_imputed[feature_cols].fillna(0.0)
    return df_imputed

# --- 랭크 데이터 로딩 함수 (컬럼 소문자화 및 타입 매칭) ---
def load_rank_features(rank_root_dir, date_min=None, date_max=None):
    if rank_root_dir is None or not os.path.isdir(rank_root_dir):
        return None
    file_list = sorted(glob.glob(os.path.join(rank_root_dir, "**", "*.parquet"), recursive=True))
    if not file_list: return None

    print(f"[Rank] Parquet 로딩 중... ({len(file_list)} files)")
    parts = []
    for fp in file_list:
        try:
            part = pd.read_parquet(fp)
        except: continue
        if part.empty: continue

        # 로드 즉시 컬럼 소문자화
        part.columns = [c.lower() for c in part.columns]

        if isinstance(part.index, pd.DatetimeIndex):
            part.index.name = '날짜'
            part = part.reset_index()
        
        rename_map = {}
        for col in part.columns:
            if col in ['date', 'time', 'index']: rename_map[col] = '날짜'
        if rename_map: part.rename(columns=rename_map, inplace=True)
        
        # Ticker 타입 강제 (병합 실패 방지)
        if 'ticker' in part.columns:
            part['ticker'] = part['ticker'].astype(str)
        
        if '날짜' not in part.columns: continue
        part['날짜'] = pd.to_datetime(part['날짜'], errors='coerce')
        part = part.dropna(subset=['날짜'])

        if date_min: part = part[part['날짜'] >= date_min]
        if date_max: part = part[part['날짜'] <= date_max]
        if not part.empty: parts.append(part)

    if not parts: return None
    rank_df = pd.concat(parts, ignore_index=True)
    rank_df.drop_duplicates(subset=['날짜', 'ticker'], keep='last', inplace=True)
    return rank_df

def merge_rank_features_with_main(df_main, rank_root_dir):
    if rank_root_dir is None or not os.path.exists(rank_root_dir):
        return df_main
    if df_main.empty: return df_main

    date_min = df_main.index.min()
    date_max = df_main.index.max()
    print(f"[Rank] 메인 df와 랭크 피처 병합 시작 ({date_min.date()} ~ {date_max.date()})")

    rank_df = load_rank_features(rank_root_dir, date_min, date_max)
    if rank_df is None or rank_df.empty: return df_main

    df_reset = df_main.reset_index()
    df_reset['ticker'] = df_reset['ticker'].astype(str)
    rank_df['ticker'] = rank_df['ticker'].astype(str)

    df_reset.set_index(['날짜', 'ticker'], inplace=True)
    rank_df.set_index(['날짜', 'ticker'], inplace=True)

    new_rank_cols = [c for c in rank_df.columns if c not in df_reset.columns]
    
    if new_rank_cols:
        df_merged = df_reset.join(rank_df[new_rank_cols], how='left')
        df_merged = df_merged.reset_index()
        print(f"[Rank] 랭크 병합 완료: 추가된 컬럼 {len(new_rank_cols)}개")
    else:
        df_merged = df_reset.reset_index()

    df_merged['날짜'] = pd.to_datetime(df_merged['날짜'])
    df_merged.set_index('날짜', inplace=True)
    df_merged.sort_index(inplace=True)
    return df_merged

# ==============================================================================
# 3. ModelPredictor: 데이터 로드, 전처리, AI 추론 (Brain)
# ==============================================================================
class ModelPredictor:
    def __init__(self, config):
        self.cfg = config
        self.model: Optional[lgb.Booster] = None
        self.scaler = None
        self.features: list[str] = []
        
        # 메타데이터 로드
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """모델 및 피처 메타데이터 로드"""
        print("[Brain] 모델 및 아티팩트 로드 중...")

        feat_path = os.path.join(self.cfg.MODEL_ROOT_PATH, self.cfg.FEATURE_JSON)
        model_path = os.path.join(self.cfg.MODEL_ROOT_PATH, self.cfg.MODEL_FILENAME)

        # 파일 존재 여부 동시 체크
        if not os.path.exists(feat_path) or not os.path.exists(model_path):
            msg = (
                f"[Brain] Critical Error: Artifacts missing.\n"
                f"  - Features: {feat_path} ({'Found' if os.path.exists(feat_path) else 'MISSING'})\n"
                f"  - Model:    {model_path} ({'Found' if os.path.exists(model_path) else 'MISSING'})\n"
                "  => Please check if 'run_train_lgbm.py' completed successfully."
            )
            print(msg)
            log.error(msg)
            raise FileNotFoundError(msg)

        # 1) feature manifest 로드
        with open(feat_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            # 데이터가 이미 소문자화되어 있으므로, 피처 목록도 소문자로 통일하여 매칭
            self.features = [str(c).lower() for c in meta.get("feature_cols", [])]

        if not self.features:
            msg = "[Brain] feature_cols가 비어 있습니다. 학습 시 저장된 features.json을 확인하세요."
            print(msg)
            log.error(msg)
            raise RuntimeError(msg)

        feat_hash = _sha1_text(",".join(self.features))
        log.info(
            "[Brain] Feature manifest loaded. n_features=%d hash=%s path=%s",
            len(self.features),
            feat_hash,
            feat_path,
        )

        # 2) scaler (선택 사항)
        scaler_path = os.path.join(self.cfg.MODEL_ROOT_PATH, self.cfg.SCALER_FILENAME)
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                log.info("[Brain] Scaler loaded from %s", scaler_path)
            except Exception:
                log.warning(
                    "[Brain] Failed to load scaler from %s. Proceeding without scaler.",
                    scaler_path,
                )
                self.scaler = None
        else:
            log.info(
                "[Brain] Scaler file not found (%s). Using raw (unscaled) features.",
                scaler_path,
            )
            self.scaler = None

        # 3) 모델 파일
        self.model = lgb.Booster(model_file=model_path)
        log.info("[Brain] Model loaded from %s", model_path)
        print(f"   -> 모델 로드 완료. Feature 수: {len(self.features)} (hash={feat_hash})")

    def load_and_predict(self) -> pd.DataFrame:
        """
        테스트 기간의 데이터를 로드하고, 전처리 후 예측 점수(Score)를 생성.
        """
        # 1. 데이터 준비
        local_parquet_dir = copy_data_to_local(
            self.cfg.SOURCE_DATA_PATH,
            self.cfg.LOCAL_DATA_PATH,
        )

        # Cold Start 방지를 위해 1년 전 데이터부터 로드 (Buffer)
        test_start_dt = pd.to_datetime(self.cfg.TEST_START_DATE)
        start_year = test_start_dt.year - 1  # 1년 전부터 로드하여 Expanding 통계 안정화
        end_year = int(self.cfg.TEST_END_DATE[:4])
        years = range(start_year, end_year + 1)

        files = []
        for y in years:
            yr_path = os.path.join(local_parquet_dir, str(y))
            files.extend(glob.glob(os.path.join(yr_path, "*.parquet")))

        if not files:
            raise RuntimeError("테스트 기간에 해당하는 데이터 파일이 없습니다.")

        print(f"[Brain] Parquet 파일 로드 중... (Buffering from {start_year}) files={len(files)}")
        log.info(
            "[Brain] Loading parquet files for backtest. years=%s files=%d base=%s",
            list(years),
            len(files),
            local_parquet_dir,
        )

        # 3. PyArrow 로딩
        try:
            table = pq.read_table(files)
            df = table.to_pandas()
        except Exception:
            dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)

        # 컬럼명 소문자화
        df.columns = [c.lower() for c in df.columns]
        
        # 날짜/티커 표준화
        rename_map = {}
        for c in df.columns:
            if c in ['date', 'time', 'index']: rename_map[c] = '날짜'
        df.rename(columns=rename_map, inplace=True)
        
        if "날짜" in df.columns:
            df["날짜"] = pd.to_datetime(df["날짜"])
            df.set_index("날짜", inplace=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            df.index.name = '날짜'
        else:
            df.index = pd.to_datetime(df.index)
            df.index.name = '날짜'
        
        df.sort_index(inplace=True)

        # === 랭크 피처 병합 로직 ===
        rank_dir = os.path.join(self.cfg.LOCAL_DATA_PATH, "rank_features_parquet")
        df = merge_rank_features_with_main(df, rank_dir) 
        
        # 병합된 랭크 컬럼도 확실하게 소문자화
        df.columns = [c.lower() for c in df.columns]

        # 전처리(Imputation) 수행
        cols = set(df.columns)
        feature_set = set(self.features)
        missing_feats = [c for c in self.features if c not in cols]
        
        if missing_feats:
            log.error("[Brain] Feature mismatch BEFORE Imputation. missing=%d sample=%s", len(missing_feats), missing_feats[:10])
            raise RuntimeError("Backtest <-> model feature mismatch. (Check lowercase issues)")

        # 데이터 무결성 체크 (Imputation 전)
        log.debug(f"[Data Check] Before Imputation: Shape={df.shape}, NaN Count={df[self.features].isna().sum().sum()}")
        
        df = run_causal_impute(df, self.features) 
        
        # 데이터 무결성 체크 (Imputation 후)
        nan_remain = df[self.features].isna().sum().sum()
        if nan_remain > 0:
            log.critical(f"[CRITICAL] NaN remains after imputation! Count: {nan_remain}")
            nan_cols = df[self.features].columns[df[self.features].isna().any()].tolist()
            log.critical(f"NaN Columns: {nan_cols}")
        else:
            log.debug("[Data Check] Imputation Clean. No NaNs found.")

        # 전처리 후 실제 테스트 기간만 Slicing
        mask = (df.index >= self.cfg.TEST_START_DATE) & (
            df.index <= self.cfg.TEST_END_DATE
        )
        df = df.loc[mask].copy()
        
        # 6. 거래 가능 종목 필터링
        if "is_tradable" in df.columns:
            df = df[df["is_tradable"] == 1].copy()
        elif "Is_Tradable" in df.columns:
            df = df[df["Is_Tradable"] == 1].copy()

        n_rows, n_cols = df.shape
        n_tickers = df["ticker"].nunique() if "ticker" in df.columns else -1
        date_min = df.index.min()
        date_max = df.index.max()
        log.info(
            "[Brain] Test frame ready. rows=%d cols=%d tickers=%d date_range=[%s .. %s]",
            n_rows,
            n_cols,
            n_tickers,
            getattr(date_min, "date", lambda: date_min)(),
            getattr(date_max, "date", lambda: date_max)(),
        )

        print(f"[Brain] 전처리 및 추론 시작 (Data Shape: {df.shape})...")

        # 9. Scaling & Clipping
        X = df[self.features].astype("float32").values

        if self.scaler is not None:
            log.info(
                "[Brain] Applying scaler '%s' to features.",
                self.cfg.SCALER_FILENAME,
            )
            X = self.scaler.transform(X)
        else:
            log.info("[Brain] No scaler in use. Using raw features (same as training).")

        X = np.clip(X, -10.0, 10.0).astype(np.float32)

        if self.model is None:
            raise RuntimeError("[Brain] Model is not loaded. _load_artifacts() check.")

        scores = self.model.predict(X)
        df["score"] = scores

        # 데이터 무결성 체크 - Score 분포 분석
        s_series = pd.Series(scores)
        log.info(f"\n=== [Model Score Stats] ===")
        log.info(f"Min: {s_series.min():.4f} | Max: {s_series.max():.4f} | Mean: {s_series.mean():.4f}")
        log.info(f"Quantiles: 25%={s_series.quantile(0.25):.4f}, 50%={s_series.median():.4f}, 75%={s_series.quantile(0.75):.4f}, 99%={s_series.quantile(0.99):.4f}")
        log.info(f"Over Threshold({self.cfg.BUY_THRESHOLD}): {(s_series >= self.cfg.BUY_THRESHOLD).sum()} rows")
        log.info("===========================\n")

        # 10. next_* 컬럼 구성 (ticker 별 순서 보장)
        df.reset_index(inplace=True)  # '날짜'를 컬럼으로
        df.sort_values(["ticker", "날짜"], inplace=True)

        grp = df.groupby("ticker", group_keys=False)
        df["next_open"] = grp["open"].shift(-1)
        df["next_high"] = grp["high"].shift(-1)
        df["next_low"] = grp["low"].shift(-1)
        df["next_close"] = grp["close"].shift(-1)
        df["next_date"] = grp["날짜"].shift(-1)

        # 마지막 캔들 제거
        df.dropna(subset=["next_open"], inplace=True)

        # 거래일 공백 방지용 date_diff
        df["date_diff"] = (df["next_date"] - df["날짜"]).dt.days
        df.loc[df["date_diff"] > 10, "score"] = -999.0

        if "atr_14" not in df.columns:
            if "ATR_14" in df.columns:
                df["atr_14"] = df["ATR_14"]
            else:
                df["atr_14"] = 0.0
        
        # open, high, low 필수 추가
        sim_cols = [
            "날짜", "ticker", "score", "open", "high", "low", "close", "atr_14",
            "next_open", "next_high", "next_low", "next_close",
        ]

        log.info(
            "[Brain] Simulation frame ready. rows=%d tickers=%d",
            len(df),
            df["ticker"].nunique(),
        )
        return df[sim_cols]

# ==============================================================================
# 4. BacktestEngine: 매매 시뮬레이터 (Heart)
# ==============================================================================
class BacktestEngine:
    def __init__(self, config, data):
        self.cfg = config
        self.data = data
        
        # 상태 변수
        self.cash = self.cfg.INITIAL_CASH
        self.portfolio: dict[str, dict] = {}
        self.history: list[dict] = []
        self.equity_curve: list[dict] = []

    def run(self):
        print("[Engine] 매매 시뮬레이션 시작...")
        n_rows = len(self.data)
        n_tickers = self.data["ticker"].nunique() if "ticker" in self.data.columns else -1
        n_days = self.data["날짜"].nunique() if "날짜" in self.data.columns else -1

        log.info(
            "[Engine] Simulation start. days=%d rows=%d tickers=%d initial_cash=%.0f",
            n_days,
            n_rows,
            n_tickers,
            self.cfg.INITIAL_CASH,
        )
        
        # 날짜별 그룹화 (속도 최적화)
        grouped_data = self.data.groupby('날짜')
        sorted_dates = sorted(grouped_data.groups.keys())

        for idx, today in enumerate(tqdm(sorted_dates, desc="Daily Loop"), start=1):
            daily_df = grouped_data.get_group(today)
            
            # 1. 보유 종목 관리 (청산 판단) -> 오늘(T)의 OHLC로 TP/SL/만기 판단
            self._manage_positions(today, daily_df)
            
            # 2. 신규 진입 판단 (매수) -> T+1 시가 진입
            self._check_entries(today, daily_df)
            
            # 3. 자산 평가
            self._update_equity(today, daily_df)

            # 주기적인 스냅샷 로그 (1일차 / 20일 단위 / 마지막 날)
            if idx == 1 or idx % 20 == 0 or idx == len(sorted_dates):
                self._log_daily_snapshot(idx, today)

        log.info(
            "[Engine] Simulation finished. days=%d trades=%d final_cash=%.0f open_pos=%d",
            len(sorted_dates),
            len(self.history),
            self.cash,
            len(self.portfolio),
        )
        return pd.DataFrame(self.equity_curve), pd.DataFrame(self.history)

    # 자산 평가 디버깅 (일별 스냅샷)
    def _log_daily_snapshot(self, step: int, today):
        """엔진 상태 스냅샷 + 보유 종목 요약"""
        if self.equity_curve:
            last = self.equity_curve[-1]
            total_value = last["total_value"]
            cash = last["cash"]
            stock_value = last["stock_value"]
            n_pos = last["n_pos"]
        else:
            total_value = self.cash
            cash = self.cash
            stock_value = 0.0
            n_pos = len(self.portfolio)

        dstr = today.strftime("%Y-%m-%d") if hasattr(today, "strftime") else str(today)
        
        # 보유 종목 리스트 간단 출력
        holding_tickers = list(self.portfolio.keys())
        
        log.info(
            f"[Day End] {dstr} (Step {step}) | "
            f"Total: {total_value:,.0f} | Cash: {cash:,.0f} | Stock: {stock_value:,.0f} | "
            f"Pos({n_pos}): {holding_tickers}"
        )

    # 청산 로직 (TP/SL) 상세화
    def _manage_positions(self, today, daily_df):
        """보유 종목의 TP/SL/만기 청산 로직 수행"""
        current_prices = daily_df.set_index('ticker')[['open', 'high', 'low', 'close']].to_dict('index')
        
        sell_list = []

        for ticker, pos in self.portfolio.items():
            if ticker not in current_prices:
                continue # 데이터 없음(거래정지 등)

            # 현재 루프 날짜(T)의 가격
            p_data = current_prices[ticker]
            cur_open = p_data['open']
            cur_high = p_data['high']
            cur_low  = p_data['low']
            cur_close = p_data['close']
            
            # 보유 기간 계산
            holding_days = (today - pos['entry_date']).days
            
            exit_price = None
            exit_reason = None

            # 포지션 상태 상세 로깅
            log.debug(
                f"[Pos Check] {ticker} | Held: {holding_days}d | "
                f"Cur: O={cur_open}, H={cur_high}, L={cur_low}, C={cur_close} | "
                f"Target: TP={pos['tp']:.2f}, SL={pos['sl']:.2f}"
            )

            # 1. 손절 (SL)
            if cur_low <= pos['sl']:
                exit_price = min(cur_open, pos['sl']) 
                exit_reason = 'SL'
                log.info(f"  -> [Hit SL] {ticker}: Low({cur_low}) <= SL({pos['sl']:.2f})")
            
            # 2. 익절 (TP)
            elif cur_high >= pos['tp']:
                exit_price = max(cur_open, pos['tp'])
                exit_reason = 'TP'
                log.info(f"  -> [Hit TP] {ticker}: High({cur_high}) >= TP({pos['tp']:.2f})")
            
            # 3. 만기 청산 (Time Exit)
            elif holding_days >= self.cfg.HOLDING_DAYS:
                exit_price = cur_close
                exit_reason = 'Time'
                log.info(f"  -> [Hit Time] {ticker}: Held {holding_days} days >= Limit {self.cfg.HOLDING_DAYS}")

            # 청산 실행
            if exit_price:
                revenue = exit_price * pos['qty']
                fee = revenue * (self.cfg.FEE_RATE + self.cfg.TAX_RATE)
                slippage = revenue * self.cfg.SLIPPAGE_RATE
                
                net_revenue = revenue - fee - slippage
                self.cash += net_revenue
                
                # 수익률 기록
                pnl = (net_revenue / (pos['buy_price'] * pos['qty'])) - 1
                
                self.history.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': today, # 실제로는 T+1일이지만 데이터 매핑상 today로 기록
                    'ticker': ticker,
                    'buy_price': pos['buy_price'],
                    'sell_price': exit_price,
                    'qty': pos['qty'],
                    'reason': exit_reason,
                    'pnl_pct': pnl * 100
                })
                log.info(
                    "[Engine][EXIT] %s %s qty=%d reason=%s pnl=%.2f%%",
                    today,
                    ticker,
                    pos["qty"],
                    exit_reason,
                    pnl * 100,
                )
                sell_list.append(ticker)

        # 포트폴리오에서 제거
        for t in sell_list:
            del self.portfolio[t]

    # 진입 로직 (Entry) 상세화
    def _check_entries(self, today, daily_df):
        """신규 매수 진입 로직"""
        # 현재 보유 종목 수가 꽉 찼으면 패스
        if len(self.portfolio) >= self.cfg.MAX_POSITIONS:
            return

        # 점수 높은 순으로 정렬
        candidates = daily_df.sort_values('score', ascending=False)
        
        # 금일 Top 3 후보 로깅
        top_3 = candidates.head(3)[['ticker', 'score', 'close']].to_dict('records')
        log.debug(f"[Candidates] Top 3: {top_3}")
        
        for _, row in candidates.iterrows():
            if len(self.portfolio) >= self.cfg.MAX_POSITIONS:
                log.debug(f"[Entry Skip] Portfolio Full ({len(self.portfolio)}/{self.cfg.MAX_POSITIONS}). Stop scanning.")
                break
                
            ticker = row['ticker']
            score = row['score']
            
            # 이미 보유 중이면 패스
            if ticker in self.portfolio:
                continue
                
            # 매수 임계값 미만
            if score < self.cfg.BUY_THRESHOLD:
                log.debug(f"[Entry Stop] {ticker} Score({score:.4f}) < Threshold({self.cfg.BUY_THRESHOLD}). No better candidates.")
                break
            
            # --- 진입 실행 ---
            # 예산 계산 디버깅
            target_alloc = self.equity_curve[-1]['total_value'] if self.equity_curve else self.cfg.INITIAL_CASH
            per_stock_budget = target_alloc / self.cfg.MAX_POSITIONS
            actual_budget = min(self.cash, per_stock_budget)
            
            # 예상 가격 및 수량
            # 갭상승(점상) 대비 버퍼를 15%로 설정 (보수적 자금 운용)
            est_entry_price = row['close'] * 1.15
            est_qty = int(actual_budget // est_entry_price)

            if est_qty <= 0:
                log.warning(f"[Entry Fail] {ticker}: Budget({actual_budget:.0f}) insufficient for Price({est_entry_price:.0f}). Qty=0")
                continue

            # 2. 실제 진입가 (내일 시가)
            real_entry_price = row['next_open'] * (1 + self.cfg.SLIPPAGE_RATE)
            real_cost = est_qty * real_entry_price
            real_fee = real_cost * self.cfg.FEE_RATE

            # 3. 현금 확인
            if self.cash < (real_cost + real_fee):
                # 현금 부족 시 수량 재조정 (혹은 스킵)
                est_qty = int(self.cash // (real_entry_price * (1 + self.cfg.FEE_RATE)))
                if est_qty <= 0: continue
                # 재계산
                real_cost = est_qty * real_entry_price
                real_fee = real_cost * self.cfg.FEE_RATE

            # 4. 현금 차감 및 포트폴리오 추가
            self.cash -= (real_cost + real_fee)
            
            # 다음 TP/SL 로직 호환을 위해 변수 매핑
            entry_price = real_entry_price
            qty = est_qty
            
            # 4. TP/SL 설정
            atr = row['atr_14']
            if self.cfg.USE_DYNAMIC_EXIT and atr > 0:
                tp_price = entry_price + (atr * self.cfg.ATR_MULTIPLIER_TP)
                sl_price = entry_price - (atr * self.cfg.ATR_MULTIPLIER_SL)
            else:
                tp_price = entry_price * self.cfg.TARGET_PROFIT_PCT
                sl_price = entry_price * self.cfg.STOP_LOSS_PCT
            
            # 5. 포트폴리오 등록
            self.portfolio[ticker] = {
                'entry_date': today,
                'buy_price': entry_price,
                'qty': qty,
                'tp': tp_price,
                'sl': sl_price
            }
            
            # 진입 성공 로그 강화
            log.info(
                f"[ORDER EXEC] BUY {ticker} | Score: {score:.4f} | "
                f"Cash Used: {actual_budget:.0f} | Est.Qty: {est_qty}"
            )
            
            # 기존 상세 로그 (호환성을 위해 유지)
            log.info(
                "[Engine][ENTRY] %s %s qty=%d entry=%.2f tp=%.2f sl=%.2f score=%.4f",
                today,
                ticker,
                qty,
                entry_price,
                tp_price,
                sl_price,
                score,
            )

    def _update_equity(self, today, daily_df):
        """일별 자산 평가 업데이트"""
        stock_val = 0
        # 미래(next_close)가 아니라 오늘 종가 기준으로 평가
        price_map = daily_df.set_index('ticker')['close'].to_dict()
        
        for ticker, pos in self.portfolio.items():
            curr_price = price_map.get(ticker, pos['buy_price']) # 데이터 없으면 매수가로 평가
            stock_val += curr_price * pos['qty']
            
        total_val = self.cash + stock_val
        self.equity_curve.append({
            'date': today,
            'total_value': total_val,
            'cash': self.cash,
            'stock_value': stock_val,
            'n_pos': len(self.portfolio)
        })

# ==============================================================================
# 5. Reporter: 결과 분석 및 리포팅
# ==============================================================================
class Reporter:
    def __init__(self, equity_df, trade_df, config):
        self.equity_df = equity_df
        self.trade_df = trade_df
        self.cfg = config
        
        ensure_dir(self.cfg.OUTPUT_RESULT_DIR)

    def report(self):
        if self.equity_df.empty:
            print("[Report] 거래 내역이 없어 리포트를 생성할 수 없습니다.")
            return

        self.equity_df['date'] = pd.to_datetime(self.equity_df['date'])
        self.equity_df.set_index('date', inplace=True)
        
        # 1. 주요 지표 계산
        initial = self.cfg.INITIAL_CASH
        final = self.equity_df['total_value'].iloc[-1]
        
        total_ret = (final / initial) - 1
        days = (self.equity_df.index[-1] - self.equity_df.index[0]).days
        cagr = (final / initial) ** (365 / days) - 1 if days > 0 else 0
        
        # MDD
        self.equity_df['peak'] = self.equity_df['total_value'].cummax()
        self.equity_df['dd'] = (self.equity_df['total_value'] / self.equity_df['peak']) - 1
        mdd = self.equity_df['dd'].min()
        
        # 승률 (run_preprocess.py의 LABEL_PROFIT_THRESHOLD 2% 기준 적용)
        if not self.trade_df.empty:
            # pnl_pct는 퍼센트 단위이므로 0.02가 아닌 2.0으로 비교
            win_rate = (self.trade_df['pnl_pct'] >= 2.0).mean() * 100
            avg_pnl = self.trade_df['pnl_pct'].mean()
            n_trades = len(self.trade_df)
        else:
            win_rate = 0.0
            avg_pnl = 0.0
            n_trades = 0

        # 로그 요약 (모니터링용)
        log.info(
            "[Report] Summary period=%s~%s days=%d final=%.0f total_ret=%.2f%% "
            "CAGR=%.2f%% MDD=%.2f%% trades=%d win_rate=%.2f%% avg_pnl=%.2f%%",
            self.equity_df.index[0].date(),
            self.equity_df.index[-1].date(),
            days,
            final,
            total_ret * 100,
            cagr * 100,
            mdd * 100,
            n_trades,
            win_rate,
            avg_pnl,
        )

        # 2. 텍스트 출력
        print("\n" + "="*50)
        print(f" >>> BACKTEST RESULT REPORT <<<")
        print("="*50)
        print(f" 기간: {self.equity_df.index[0].date()} ~ {self.equity_df.index[-1].date()} ({days}일)")
        print(f" 초기 자본: {initial:,.0f} 원")
        print(f" 최종 자본: {final:,.0f} 원")
        print(f" --------------------------------")
        print(f" 누적 수익률 : {total_ret*100:6.2f} %")
        print(f" 연환산 수익 (CAGR): {cagr*100:6.2f} %")
        print(f" 최대 낙폭 (MDD)  : {mdd*100:6.2f} %")
        print(f" --------------------------------")
        print(f" 총 매매 횟수: {n_trades} 회")
        print(f" 승률 (Win Rate): {win_rate:6.2f} %")
        print(f" 평균 손익률: {avg_pnl:6.2f} %")
        print("="*50)

        # 상세 매매 기록 출력
        if not self.trade_df.empty:
            print("\n[상세 매매 기록]")
            print(f"{'진입일':<12} {'종목코드':<10} {'결과':<6} {'수익률':<10} {'청산사유'}")
            print("-" * 55)
            
            # 진입일 기준 정렬
            sorted_trades = self.trade_df.sort_values('entry_date')
            
            for _, row in sorted_trades.iterrows():
                # 날짜 포맷팅
                d_str = row['entry_date'].strftime("%Y-%m-%d") if hasattr(row['entry_date'], "strftime") else str(row['entry_date'])[:10]
                
                # 승/패 판별 (수익률이 0보다 크면 WIN)
                outcome = "WIN" if row['pnl_pct'] > 0 else "LOSE"
                
                # 수익률 포맷팅 (이미 % 단위로 저장되어 있음)
                pnl_str = f"{row['pnl_pct']:+.2f}%"
                
                # 출력
                print(f"{d_str:<12} {row['ticker']:<10} {outcome:<6} {pnl_str:<10} {row['reason']}")
            print("-" * 55 + "\n")

        # 3. 파일 저장
        self.equity_df.to_csv(os.path.join(self.cfg.OUTPUT_RESULT_DIR, 'daily_equity.csv'))
        self.trade_df.to_csv(os.path.join(self.cfg.OUTPUT_RESULT_DIR, 'trade_log.csv'))
        
        # 4. 차트 그리기
        self._plot_result()

    def _plot_result(self):
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity Curve
        axes[0].plot(self.equity_df.index, self.equity_df['total_value'], label='Portfolio Value', color='blue')
        axes[0].set_title('Backtest Equity Curve')
        axes[0].set_ylabel('Value (KRW)')
        axes[0].legend()
        
        # Drawdown
        axes[1].fill_between(self.equity_df.index, self.equity_df['dd'], 0, color='red', alpha=0.3)
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown (%)')
        
        plt.tight_layout()
        img_path = os.path.join(self.cfg.OUTPUT_RESULT_DIR, 'backtest_chart.png')
        plt.savefig(img_path)
        print(f"[Report] 차트 저장 완료: {img_path}")
        plt.show()

# ==============================================================================
# 6. Main: 실행 진입점
# ==============================================================================
def main() -> None:
    # 1. 인자 파싱
    parser = argparse.ArgumentParser(description="RRE Backtest Engine")
    parser.add_argument("--mode", type=str, default="normal", help="Execution mode: 'normal' or 'uui'")
    args = parser.parse_args()

    try:
        config = Config()

        # 전역 로거 초기화
        global log
        log = setup_logger(config.LOG_LEVEL)
        
        if args.mode == 'uui':
            print("\n" + "="*60)
            print(" >>> MODE: UUI (Threshold Sensitivity Analysis) <<<")
            print("="*60)
            log.info("==== Backtest Start (Mode: UUI) ====")
        else:
            log.info("==== Backtest Start (Mode: Normal) ====")

        # ---------------------------------------------------------
        # [Common] 데이터 로드 및 추론 (한 번만 수행하여 재사용)
        # ---------------------------------------------------------
        predictor = ModelPredictor(config)
        simulation_data = predictor.load_and_predict()
        
        if simulation_data.empty:
            print("[Main] 테스트 기간에 데이터가 없습니다. 종료합니다.")
            log.warning("[Main] No data in test period. Backtest aborted.")
            return

        # ---------------------------------------------------------
        # [Mode 1] UUI 모드: 임계값별 반복 시뮬레이션
        # ---------------------------------------------------------
        if args.mode == 'uui':
            # 테스트할 임계값 범위 설정 (0.1 ~ 0.9, 0.1 단위)
            # preprocess의 라벨링 기준과 동일한 승률 산출을 위해 Step 변경
            thresholds = np.arange(0.10, 1.0, 0.01)
            
            results = []
            print(f"\n[UUI] 총 {len(thresholds)}개의 임계값에 대해 시뮬레이션을 시작합니다...\n")
            print(f"{'Threshold':<10} {'Trades':<10} {'Win Rate(%)':<15} {'Return(%)':<12} {'MDD(%)':<10}")
            print("-" * 65)

            # 로깅 레벨 일시 상향 (반복 실행 시 로그 폭탄 방지)
            logging.getLogger("rre.backtest").setLevel(logging.WARNING)

            for th in thresholds:
                # Config의 임계값 동적 변경
                config.BUY_THRESHOLD = float(th)
                
                # 엔진 초기화 및 실행 (데이터는 재사용)
                engine = BacktestEngine(config, simulation_data)
                equity, trades = engine.run()
                
                # 결과 집계
                n_trades = len(trades)
                if n_trades > 0:
                    # 라벨링 기준(2%)과 동일하게 >= 2.0% 승률 계산
                    win_rate = (trades['pnl_pct'] >= 2.0).mean() * 100
                    avg_pnl = trades['pnl_pct'].mean()
                else:
                    win_rate = 0.0
                    avg_pnl = 0.0

                if not equity.empty:
                    final_val = equity['total_value'].iloc[-1]
                    total_ret = (final_val / config.INITIAL_CASH - 1) * 100
                    
                    # MDD 계산
                    equity['peak'] = equity['total_value'].cummax()
                    dd = (equity['total_value'] / equity['peak']) - 1
                    mdd = dd.min() * 100
                else:
                    total_ret = 0.0
                    mdd = 0.0
                
                # 결과 리스트 저장 및 한 줄 출력
                results.append({
                    'threshold': th,
                    'trades': n_trades,
                    'win_rate': win_rate,
                    'return': total_ret,
                    'mdd': mdd
                })
                print(f"{th:<10.2f} {n_trades:<10d} {win_rate:<15.2f} {total_ret:<12.2f} {mdd:<10.2f}")

            print("-" * 65)
            
            # 최적 결과 추천 (수익률 기준)
            if results:
                best_res = sorted(results, key=lambda x: x['return'], reverse=True)[0]
                print(f"\n[Recommendation] Best Return at Threshold: {best_res['threshold']:.2f}")
                print(f" -> Return: {best_res['return']:.2f}%, Trades: {best_res['trades']}, Win Rate: {best_res['win_rate']:.2f}%")

        # ---------------------------------------------------------
        # [Mode 2] Normal 모드: 단일 설정 실행
        # ---------------------------------------------------------
        else:
            log.info(
                "[Config] INITIAL_CASH=%.0f BUY_TH=%.3f HOLD_DAYS=%d",
                config.INITIAL_CASH, config.BUY_THRESHOLD, config.HOLDING_DAYS
            )
            
            engine = BacktestEngine(config, simulation_data)
            equity, trades = engine.run()
            
            reporter = Reporter(equity, trades, config)
            reporter.report()

        log.info("==== Backtest Finished ====")
        
    except Exception:
        msg = "\n[Critical Error] 백테스트 실행 중 오류 발생:"
        print(msg)
        print(traceback.format_exc())
        logging.getLogger("rre.backtest").exception("[Critical] Backtest crashed.")

if __name__ == "__main__":
    main()