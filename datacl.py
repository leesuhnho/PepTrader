#!/usr/bin/env python3
# ==============================================================================
# 프로젝트: rre (딥러닝 주식 모델)
# 파일설명: KOSPI + KOSDAQ 종목의 OHLCV, 펀더멘털, 수급 데이터 및
#           주요 매크로 지표(시장 지수, 환율 등)를 수집하는 스크립트
# ==============================================================================

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import random 
from pykrx import stock
import yfinance as yf 
import requests_cache 

import gc 
from pathlib import Path 
import glob 

from datetime import timedelta 
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import BoundedSemaphore
import threading

# ==================================================
#            환경 설정
# ==================================================
# 데이터 수집 기간
START_DATE = "20190101"
END_DATE = "20251117" 
# KRX 서버 부하를 줄이기 위한 기본 대기 시간 (초)
TIME_SLEEP = 0.5
# API 요청 실패 시 최대 재시도 횟수
MAX_RETRIES = 3
# ==================================================

def setup_environment():
    """
    구글 드라이브 마운트 확인 및 프로젝트 경로 설정
    """
    print("="*50)
    print("[INFO] Google Drive 마운트 및 경로 설정을 시작합니다.")
    print("="*50)

    google_drive_path = "/content/drive/MyDrive"
    if os.path.exists(google_drive_path):
        print("✓ Google Drive가 이미 마운트되어 있습니다.")
        google_drive_active = True
    elif os.path.exists("/content/drive"):
        print("⚠ /content/drive 폴더는 존재하지만 MyDrive가 없습니다.")
        print("  Colab 노트북에서 다음 명령을 먼저 실행해주세요:")
        print("  from google.colab import drive; drive.mount('/content/drive')")
        google_drive_active = False
    else:
        print("✗ Google Drive가 마운트되지 않았습니다. 로컬 경로를 사용합니다.")
        google_drive_active = False

    project_name = "rre"
    if google_drive_active:
        base_path = f"/content/drive/MyDrive/{project_name}"
    else:
        base_path = f"./{project_name}"

    # 4가지 데이터 유형별 저장 경로 설정
    paths = {
        "ohlcv": os.path.join(base_path, "data", "all_stocks_ohlcv"),
        "fundamental": os.path.join(base_path, "data", "all_stocks_fundamental"),
        "trading": os.path.join(base_path, "data", "all_stocks_trading"),
        "macro": os.path.join(base_path, "data", "market_data")
    }

    print(f"프로젝트명: {project_name}")
    print(f"기본 경로: {base_path}")
    print("-" * 50)
    
    # 모든 경로 생성
    for name, path in paths.items():
        os.makedirs(path, exist_ok=True)
        print(f"✓ {name} 데이터 저장 경로 생성 완료: {path}")
        
    print("="*50)
    return paths

def get_all_tickers_in_range(start_date, end_date):
    """
    생존 편향 방지를 위해 기간 내 모든 티커(상장폐지 포함) 수집
    """
    print(f"\n[INFO] {start_date} ~ {end_date} 기간의 모든 티커(상장폐지 포함) 수집을 시작합니다.")
    
    all_tickers = set()
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    # 매년 1월 1일 (또는 해당 연도 첫 거래일) 기준으로 티커 수집
    dates_to_check = [f"{year}0101" for year in range(start_year, end_year + 1)]
    # 마지막 날짜도 포함하여 최신성 보장
    if end_date not in dates_to_check:
         dates_to_check.append(end_date)

    for date in tqdm(dates_to_check, desc="연도별 티커 목록 수집"):
        try:
            kospi_tickers = stock.get_market_ticker_list(date, market="KOSPI")
            all_tickers.update(kospi_tickers)
            time.sleep(TIME_SLEEP)
        except (IndexError, ValueError):
            # 해당 날짜가 휴일이거나 데이터가 없는 경우 (정상)
            pass 
        except Exception as e:
            # 네트워크 오류 등 예상치 못한 오류
            tqdm.write(f"  - [WARN] {date} KOSPI 티커 수집 중 오류: {e}")

        try:
            kosdaq_tickers = stock.get_market_ticker_list(date, market="KOSDAQ")
            all_tickers.update(kosdaq_tickers)
            time.sleep(TIME_SLEEP)
        except (IndexError, ValueError):
            # 해당 날짜가 휴일이거나 데이터가 없는 경우 (정상)
            pass
        except Exception as e:
            # 네트워크 오류 등 예상치 못한 오류
            tqdm.write(f"  - [WARN] {date} KOSDAQ 티커 수집 중 오류: {e}")

    print(f"[SUCCESS] {start_year}~{end_year} 기간 동안 총 {len(all_tickers)}개의 고유 티커 발견.")
    
    if all_tickers:
        print("... (샘플 종목)")
        sample_tickers = list(all_tickers)[:2]
        for ticker in sample_tickers:
            try:
                print(f"  - {ticker}: {stock.get_market_ticker_name(ticker)}")
            except (KeyError, IndexError, ValueError, RuntimeError) as e:
                # 상장폐지되었거나 이름 조회가 불가능한 경우
                tqdm.write(f"  - [INFO] 샘플 티커 {ticker} 이름 조회 실패 (상장폐지/오류): {e}")
    
    return list(all_tickers)


def fetch_and_save_macro(filepath, fetch_function, index_label=None, **kwargs):
    """
    매크로 데이터 수집을 위한 재시도 및 원자적 쓰기 래퍼 함수
    """
    if os.path.exists(filepath):
        print(f"  - [SKIP] {kwargs.get('name', filepath)} 데이터가 이미 존재합니다.")
        return

    temp_filepath = filepath + ".tmp"
    retries = 0
    
    while retries < MAX_RETRIES:
        try:
            # 1. API 호출
            df = fetch_function()
            
            if df.empty:
                print(f"  - [WARN] {kwargs.get('name', 'Data')} 데이터가 비어있습니다.")
                return # 빈 데이터도 성공으로 간주하고 종료

            # 2. Atomic Write (원자적 쓰기)
            df.to_csv(temp_filepath, index=True, index_label=index_label)
            os.rename(temp_filepath, filepath)
            
            print(f"  - [OK] {kwargs.get('name', filepath)} 데이터 저장 완료.")
            time.sleep(TIME_SLEEP) # 성공 시 휴식
            return # 성공 시 함수 종료

        except (ValueError, KeyError, IndexError) as e:
            # 데이터 없음 오류. 재시도 불필요.
            print(f"  - [INFO] {kwargs.get('name', 'Data')} 데이터 없음: {e}")
            return # 다음 작업으로 넘어감

        except Exception as e:
            # 네트워크 오류 또는 기타 예외
            retries += 1
            sleep_time = TIME_SLEEP * (2 ** retries) + random.uniform(0, 1) # Exponential Backoff
            print(f"  - [RETRY {retries}/{MAX_RETRIES}] {kwargs.get('name', 'Data')} 오류: {e}. {sleep_time:.1f}초 대기...")
            time.sleep(sleep_time)

    print(f"  - [FAIL] {kwargs.get('name', 'Data')} 최종 수집 실패.")
    # 실패 시 .tmp 파일이 남아있다면 삭제
    if os.path.exists(temp_filepath):
        os.remove(temp_filepath)


def collect_macro_data(path, start_date, end_date):
    """
    매크로/시장 데이터 수집 (pykrx + yfinance)
    """
    print(f"\n[INFO] {start_date} ~ {end_date} 매크로/시장 데이터 수집을 시작합니다...")
    
    # 1. 국내 시장 지수 (pykrx)
    indices = {
        "KOSPI": "1001",
        "KOSDAQ": "2001",
    }
    for name, code in indices.items():
        filepath = os.path.join(path, f"{name}.csv")
        # 람다 함수를 사용하여 fetch_function 인자 전달
        fetch_function = lambda code=code: stock.get_index_ohlcv_by_date(start_date, end_date, code)
        fetch_and_save_macro(filepath, fetch_function, name=name, index_label='날짜')

    # 1-2. KODEX 200 ETF (pykrx)
    etfs = {
        "KODEX200": "069500",
    }
    for name, code in etfs.items():
        filepath = os.path.join(path, f"{name}.csv")

        def fetch_function(code=code):
            return stock.get_etf_ohlcv_by_date(start_date, end_date, code)

        fetch_and_save_macro(
            filepath,
            fetch_function,
            name=name,
            index_label="날짜",
        )

    # 2. 글로벌 매크로 지표 (yfinance)
    # yfinance는 end 날짜를 포함하지 않으므로 +1일
    yf_start = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    yf_end = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    macro_assets = {
        "SP500": "^GSPC",      # S&P 500
        "USD_KRW": "KRW=X",    # USD/KRW 환율
        "VIX": "^VIX",         # 변동성 지수
    }

    tickers_list = list(macro_assets.values())
    print(f"  - [INFO] yfinance 티커 일괄 다운로드: {tickers_list}")
    
    try:
        # (1) 일괄 다운로드 + (2) auto_adjust=False 명시
        all_macro_data = yf.download(
            tickers_list, 
            start=yf_start, 
            end=yf_end, 
            auto_adjust=False # 경고 제거 및 과거 데이터 일관성 유지
        )
    except Exception as e:
        print(f"  - [FAIL] yfinance 일괄 다운로드 실패: {e}")
        all_macro_data = pd.DataFrame() # 빈 DF로 설정하여 루프 스킵

    for name, ticker in macro_assets.items():
        filepath = os.path.join(path, f"{name}.csv")
        full_name = f"{name} ({ticker})"
        
        # 일괄 다운로드된 데이터에서 이 티커의 DF만 재구성하는 함수
        def create_fetch_function(t):
            def fetch_from_bulk():
                if all_macro_data.empty:
                    return pd.DataFrame() # 빈 DF 반환
                
                # 다운로드된 데이터가 1개 티커뿐이면 MultiIndex가 아님
                if len(tickers_list) == 1:
                    return all_macro_data # 그냥 반환
                    
                # MultiIndex에서 단일 티커의 DF를 재구성
                try:
                    # 동적으로 이 티커에 해당하는 컬럼들만 수집
                    ticker_cols = [col for col in all_macro_data.columns if col[1] == t]
                    if not ticker_cols:
                        return pd.DataFrame()
                        
                    df_one = all_macro_data[ticker_cols]
                    # MultiIndex ('Open', '^VIX') -> SingleIndex ('Open')
                    df_one.columns = df_one.columns.droplevel(1)
                    # yfinance 원본 순서와 유사하게 정렬
                    cols_original_order = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                    cols_to_use = [c for c in cols_original_order if c in df_one.columns]
                    return df_one[cols_to_use]
                except Exception as e:
                    print(f"  - [WARN] {t} 데이터 슬라이싱 실패: {e}")
                    return pd.DataFrame()
            return fetch_from_bulk

        fetch_function = create_fetch_function(ticker) 
        
        fetch_and_save_macro(filepath, fetch_function, name=full_name, index_label='Date')
            
    print("[COMPLETE] 매크로/시장 데이터 수집 완료.")

def get_all_vintage_dates(series_id, session, fred_vd_url, api_key):
    """
    fred/series/vintages로 대상 빈티지 날짜 전체를 수집
    """
    params = {
        "api_key": api_key, 
        "file_type": "json", 
        "series_id": series_id,
        "realtime_start": "1776-07-04",
        "realtime_end": "9999-12-31",
    }
    
    # 재시도 로직
    retries = 0
    while retries < MAX_RETRIES:
        try:
            r = session.get(fred_vd_url, params=params, timeout=60)
            r.raise_for_status()
            vintage_dates = r.json()["vintage_dates"]
            return pd.to_datetime(vintage_dates)
        except Exception as e:
            retries += 1
            sleep_time = TIME_SLEEP * (2 ** retries) + random.uniform(0, 1)
            print(f"  - [RETRY {retries}/{MAX_RETRIES}] FRED {series_id} vintagedates: {e}. {sleep_time:.1f}s 대기...")
            time.sleep(sleep_time)
            
    print(f"  - [FAIL] FRED {series_id} vintagedates 최종 실패.")
    return pd.DatetimeIndex([]) # 실패 시 빈 인덱스 반환

def fetch_fred_vintages_bulk(series_id, obs_start, obs_end, vintage_dates, 
                             session, fred_obs_url, api_key):
    """
    fred/series/observations에 vintage_dates 일괄 호출
    """
    frames, CHUNK = [], 100  # FRED JSON 권장 2000개 이하로 청크
    
    if vintage_dates.empty:
        return pd.DataFrame(columns=["date", "vintage_date", "value"])

    for i in tqdm(range(0, len(vintage_dates), CHUNK), desc=f"  - {series_id} 빈티지 수집", leave=False):
        vd_chunk = vintage_dates[i:i+CHUNK]
        vstr = ",".join(d.strftime("%Y-%m-%d") for d in vd_chunk)
        params = {
            "api_key": api_key, "file_type": "json", "series_id": series_id,
            "output_type": 2,  # 빈티지별 전체 관측치 컬럼으로 반환
            "vintage_dates": vstr,
            "observation_start": obs_start,
            "observation_end": obs_end,
        }
        
        # 재시도 로직
        retries = 0
        while retries < MAX_RETRIES:
            try:
                r = session.get(fred_obs_url, params=params, timeout=90)
                r.raise_for_status()
                frames.append(pd.json_normalize(r.json()["observations"]))
                break # (while True) 성공
            except Exception as e:
                retries += 1
                if retries >= MAX_RETRIES:
                    print(f"  - [FAIL] FRED {series_id} observations 청크 {i//CHUNK} 실패: {e}")
                    break # (while True) 최종 실패
                sleep_time = TIME_SLEEP * (2 ** retries) + random.uniform(0, 1)
                print(f"  - [RETRY {retries}/{MAX_RETRIES}] FRED {series_id} obs 청크: {e}. {sleep_time:.1f}s 대기...")
                time.sleep(sleep_time)
    
    if not frames:
        return pd.DataFrame(columns=["date", "vintage_date", "value"])

    raw = pd.concat(frames, ignore_index=True)

    # 'date', f'{series_id}_YYYY-MM-DD' ... 형태 → long 포맷
    value_cols = [c for c in raw.columns if c != "date"]
    if not value_cols:
        return pd.DataFrame(columns=["date", "vintage_date", "value"])
        
    long = raw.melt(id_vars="date", value_vars=value_cols,
                    var_name="vintage_col", value_name="value")
    
    # .이 누락값임
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    
    long["vintage_date"] = pd.to_datetime(long["vintage_col"].str.rsplit("_", n=1).str[-1])
    long["date"] = pd.to_datetime(long["date"])
    
    long = long.drop(columns=["vintage_col"])\
               .sort_values(["vintage_date","date"])\
               .reset_index(drop=True)
    return long

def collect_fred_macro(path, start_date, end_date, session):
    """
    FRED 빈티지(Real-Time) 값으로 일괄 수집
    - 'series/vintages' 1회 + 'series/observations' 1~2회 (일괄)
    - 'long' 포맷 (date, vintage_date, value)으로 저장
    - 캐시된 세션 사용
    """
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    if not FRED_API_KEY:
        raise RuntimeError("환경변수 FRED_API_KEY가 필요합니다. (FRED API 키)")

    FRED_SERIES = {
        "UST10Y": "DGS10",
        "UST2Y": "DGS2",
        "FEDFUNDS": "FEDFUNDS",
        "EFFR": "EFFR",
        "HY_OAS": "BAMLH0A0HYM2",
        "IG_OAS": "BAMLC0A0CM",
        "CPI": "CPIAUCSL",
        "PPI": "PPIACO",
        "BREAKEVEN10Y": "T10YIE",
        "EPU_US": "USEPUINDXD",
        "UNRATE": "UNRATE",
        "GDP_REAL_QOQ": "A191RL1Q225SBEA",
    }
    
    # 수집할 관측치(observation)의 기간
    obs_start = "2019-01-01" 
    obs_end = "2025-11-17"   
    
    # API 엔드포인트
    url_obs = "https://api.stlouisfed.org/fred/series/observations"
    url_vint = "https://api.stlouisfed.org/fred/series/vintagedates"

    # 수집할 빈티지(vintage)의 기간
    vintage_start = pd.to_datetime(start_date)
    vintage_end = pd.to_datetime(end_date)

    print(f"\n[INFO] {start_date} ~ {end_date} FRED 빈티지(일괄) 수집을 시작합니다...")

    for nice_name, series_id in FRED_SERIES.items():
        filepath = os.path.join(path, f"{nice_name}.csv")
        
        # --- 람다 함수 정의 ---
        def create_fetch_function(sid):
            def fetcher():
                # 1. 모든 빈티지 날짜 수집
                all_vd = get_all_vintage_dates(sid, session, url_vint, FRED_API_KEY)
                
                # 2. 수집할 기간(vintage_start ~ vintage_end) 내의 날짜만 필터링
                vd_in_range = all_vd[(all_vd >= vintage_start) & (all_vd <= vintage_end)]
                
                if vd_in_range.empty:
                    print(f"  - [INFO] FRED {sid}: 수집 기간 내 빈티지 날짜 없음.")
                    return pd.DataFrame(columns=["date", "vintage_date", "value"])

                # 3. 필터링된 빈티지 날짜에 대해서만 일괄 호출
                long_df = fetch_fred_vintages_bulk(
                    sid, obs_start, obs_end, vd_in_range, 
                    session, url_obs, FRED_API_KEY
                )
                return long_df
            return fetcher
        # --- 람다 함수 종료 ---

        fetch_function = create_fetch_function(series_id)
        
        if os.path.exists(filepath):
            print(f"  - [SKIP] {nice_name} (ALFRED:{series_id}) 데이터가 이미 존재합니다.")
            continue
        
        # 파일 경로 분리: flat(표준 단일 값) + vintages(long)
        flat_path = filepath 
        vint_path = os.path.join(path, f"{nice_name}_vintages.csv")
        
        # 임시 파일 경로
        tmp_f = flat_path + ".tmp"
        tmp_v = vint_path + ".tmp"

        try:
            # 1) Long 포맷 저장 (현재 형식 유지)
            df_long = fetch_function()  # [date, vintage_date, value]
            
            if df_long.empty:
                print(f"  - [WARN] {nice_name} 데이터가 비어있습니다.")
                continue
            
            df_long.to_csv(tmp_v, index=False)
            os.rename(tmp_v, vint_path)

            # 2) 표준 단일 값 포맷 생성 → run_preprocess.py 호환
            # 관측일(date)별 "첫 번째" 빈티지(최초 공시 값)만 남김 → 리비전 기반 누수 차단
            df_flat = (
                df_long.sort_values(["date", "vintage_date"])
                       .groupby("date", as_index=False)
                       .head(1)[["date", "value"]]      # head(1)
                       .set_index("date")
                       .rename(columns={"value": nice_name})
            )

            df_flat.to_csv(tmp_f, index=True)
            os.rename(tmp_f, flat_path)
            
            print(f"  - [OK] {nice_name} 저장 완료 (flat + vintages)")

        except Exception as e:
            print(f"  - [FAIL] {nice_name} (ALFRED:{series_id}) 최종 수집 실패: {e}")
            # 실패 시 생성된 .tmp 파일 모두 삭제
            if os.path.exists(tmp_f):
                os.remove(tmp_f)
            if os.path.exists(tmp_v):
                os.remove(tmp_v)
                
    print("[COMPLETE] FRED(빈티지) 매크로 수집 완료.")

def collect_data_template(tickers, path, start_date, end_date, 
                          data_type, fetch_function, save_with_index=True):
    """
    개별 종목 데이터 수집을 위한 공통 템플릿 함수
    """
    print(f"\n[INFO] {start_date} ~ {end_date} {data_type} 데이터 수집을 시작합니다...")
    
    desc = f"{data_type} 데이터 수집 ({len(tickers)}개 종목)"
    for ticker in tqdm(tickers, desc=desc):
        filepath = os.path.join(path, f"{ticker}.csv")
        
        if os.path.exists(filepath):
            continue
            
        temp_filepath = filepath + ".tmp"
        retries = 0
        
        while retries < MAX_RETRIES:
            try:
                # 1. API 호출
                df = fetch_function(start_date, end_date, ticker)
                
                if df.empty:
                    # 데이터가 없는 경우 (정상 응답) 루프 탈출
                    break 

                # 2. Atomic Write (원자적 쓰기)
                if save_with_index:
                    df.to_csv(temp_filepath, index=True)
                else:
                    # Trading(수급) 데이터는 index가 날짜가 아니므로 index=False로 저장
                    df.reset_index(inplace=True)
                    df.to_csv(temp_filepath, index=False)
                    
                os.rename(temp_filepath, filepath)
                
                time.sleep(TIME_SLEEP) # 성공 시 휴식
                break # 성공, while 루프 탈출

            except (ValueError, KeyError, IndexError):
                # 데이터가 존재하지 않는 경우 (오류 응답)
                break 

            except Exception as e:
                # 네트워크 오류, 서버 오류 등
                retries += 1
                sleep_time = TIME_SLEEP * (2 ** retries) + random.uniform(0, 1) # Exponential Backoff
                tqdm.write(f"  - [RETRY {retries}/{MAX_RETRIES}] {ticker} ({data_type}) 오류: {e}. {sleep_time:.1f}초 대기...")
                time.sleep(sleep_time)

        if retries == MAX_RETRIES:
            tqdm.write(f"  - [FAIL] {ticker} ({data_type}) 최종 수집 실패.")
            # 실패 시 .tmp 파일이 남아있다면 삭제
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

    print(f"[COMPLETE] {data_type} 데이터 수집 완료.")


def collect_ohlcv_data(tickers, path, start_date, end_date):
    """
    개별 종목 OHLCV 데이터 수집
    """
    collect_data_template(tickers, path, start_date, end_date, 
                          "OHLCV", stock.get_market_ohlcv_by_date)

def collect_ohlcv_data_fast(tickers, path, start_date, end_date):
    """
    OHLCV 수집을 소규모 병렬화(기본 6 workers) + 레이트리미트(초당 6회)로 가속.
    파일 스키마/경로는 기존과 동일하게 저장하여 run_preprocess.py와 100% 호환.
    """
    print(f"\n[INFO] {start_date} ~ {end_date} OHLCV 병렬 수집 시작...")
    os.makedirs(path, exist_ok=True)

    # 환경변수로 조절 가능 (없으면 보수적 기본값)
    max_workers = int(os.getenv("RRE_OHLCV_WORKERS", "6"))
    rate_per_sec = int(os.getenv("RRE_OHLCV_RPS", "6"))  # 전체 초당 호출 한도
    token = BoundedSemaphore(rate_per_sec)

    # 초마다 토큰을 재충전 (private 속성 없이 안전하게)
    def refill_tokens():
        while True:
            for _ in range(rate_per_sec):
                try:
                    token.release()
                except ValueError:
                    break
            time.sleep(1.0)

    threading.Thread(target=refill_tokens, daemon=True).start()

    def _one(tk):
        filepath = os.path.join(path, f"{tk}.csv")
        if os.path.exists(filepath):
            return tk, "SKIP"

        # 초당 호출 제한 준수
        token.acquire()
        try:
            df = stock.get_market_ohlcv_by_date(start_date, end_date, tk)
        except Exception as e:
            return tk, f"ERR:{e}"

        # 비어있으면 파일 생성하지 않음(다음 실행에서 다시 시도 가능)
        if df is None or df.empty:
            return tk, "EMPTY"

        # Atomic write
        tmp = filepath + ".tmp"
        df.to_csv(tmp, index=True)  # (기존과 동일: 날짜가 index로 저장)
        os.replace(tmp, filepath)
        # 살짝 지터를 줘서 호출이 한 번에 몰리지 않게 함
        time.sleep(0.02 + random.random()*0.05)
        return tk, "OK"

    done = skip = empty = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, t) for t in tickers]
        desc = f"OHLCV 병렬 수집 ({len(tickers)}개)"
        pbar = tqdm(as_completed(futs), total=len(tickers), desc=desc)
        for f in pbar:
            st = f.result()
            if isinstance(st, tuple):
                _, tag = st
            else:
                tag = st # 오류 메시지 등이 직접 올 수 있음 (ERR:...)

            if tag == "OK":
                done += 1
            elif tag == "SKIP":
                skip += 1
            elif tag == "EMPTY":
                empty += 1
            else:
                fail += 1

    print(f"[COMPLETE] OHLCV: OK={done}, SKIP={skip}, EMPTY={empty}, FAIL={fail}")

def fetch_trading_value_for_ticker(start_date: str, end_date: str, ticker: str) -> pd.DataFrame:
    """
    개별 종목의 투자자별 순매수 거래대금 시계열을 가져온다.
    - index=날짜, columns=기관합계/외국인합계/개인/... 인 DataFrame
    """
    # pykrx: 투자자별 (순매수) 거래대금 일별 추이
    df = stock.get_market_trading_value_by_date(start_date, end_date, ticker)

    if df is None or df.empty:
        return pd.DataFrame()

    # 나중에 CSV로 저장할 때 '날짜' 컬럼이 생기도록 index 이름 명시
    df.index = pd.to_datetime(df.index)
    df.index.name = "날짜"

    # run_preprocess에서 RENAME_MAP_TRADING으로 표준화하므로 여기서는 원본 유지
    return df

def collect_trading_data(tickers, path, gdrive_trading_path, start_date, end_date):
    """
    [FAST] 종목별 Trading(수급) 데이터 수집 (병렬 + 레이트리밋 버전)
    - pykrx stock.get_market_trading_value_by_date(start, end, ticker) 사용
    - 결과는 /{path}/{티커}.csv 로 저장 (CSV 컬럼: '날짜' + 투자자별 컬럼들)
    - 완료 후 Google Drive(gdrive_trading_path)로 일괄 복사
    """
    print(f"\n[INFO] {start_date} ~ {end_date} Trading(수급) 데이터 수집을 시작합니다... (종목별 병렬 수집)")

    # 1) 로컬 수급 경로 보장
    os.makedirs(path, exist_ok=True)

    # 2) 병렬 설정 (필요시 환경변수로 튜닝)
    max_workers = int(os.getenv("RRE_TRADING_WORKERS", "4"))
    rate_per_sec = int(os.getenv("RRE_TRADING_RPS", "4"))  # 초당 호출 한도
    token = BoundedSemaphore(rate_per_sec)

    # 초마다 토큰 재충전
    def refill_tokens():
        while True:
            for _ in range(rate_per_sec):
                try:
                    token.release()
                except ValueError:
                    # 이미 토큰이 가득 찬 경우
                    break
            time.sleep(1.0)

    threading.Thread(target=refill_tokens, daemon=True).start()

    # 3) 단일 종목 수급 수집 함수 (ThreadPool에서 실행)
    def _one(ticker: str):
        filepath = os.path.join(path, f"{ticker}.csv")
        if os.path.exists(filepath):
            # 이미 수집된 종목은 스킵
            return ticker, "SKIP"

        temp_filepath = filepath + ".tmp"
        retries = 0

        while retries < MAX_RETRIES:
            # 레이트리밋: 초당 rate_per_sec 회 이하로 pykrx 호출
            token.acquire()
            try:
                # pykrx 호출 (index=날짜, name="날짜" 인 DataFrame 반환)
                df = fetch_trading_value_for_ticker(start_date, end_date, ticker)

                # 데이터 없음: 파일 생성하지 않고 EMPTY 처리
                if df is None or df.empty:
                    # 혹시 남아있을지 모를 임시파일 정리
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
                    return ticker, "EMPTY"

                # CSV 스키마: '날짜'를 컬럼으로 빼서 저장 (run_preprocess 호환)
                df_reset = df.reset_index()
                df_reset.to_csv(temp_filepath, index=False)
                os.replace(temp_filepath, filepath)

                # 호출이 한 번에 몰리지 않도록 소량 지터 추가
                time.sleep(0.02 + random.random() * 0.05)
                return ticker, "OK"

            except (ValueError, KeyError, IndexError):
                # pykrx가 "데이터 없음" 계열 예외를 던지는 경우 → EMPTY로 취급
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                return ticker, "EMPTY"

            except Exception as e:
                # 네트워크/서버 오류 등 → 지수 백오프 후 재시도
                retries += 1
                if retries >= MAX_RETRIES:
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
                    return ticker, f"ERR:{e}"

                sleep_time = TIME_SLEEP * (2 ** retries) + random.uniform(0, 1)
                tqdm.write(
                    f"  - [RETRY {retries}/{MAX_RETRIES}] {ticker} (Trading) 오류: {e}. "
                    f"{sleep_time:.1f}초 대기..."
                )
                time.sleep(sleep_time)

        # 이론상 도달하지 않지만, 방어적 코드
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return ticker, "FAIL"

    # 4) ThreadPoolExecutor로 병렬 실행
    done = skip = empty = fail = 0
    desc = f"Trading(수급) 병렬 수집 ({len(tickers)}개 종목)"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_one, tk) for tk in tickers]
        for future in tqdm(as_completed(futures), total=len(tickers), desc=desc):
            result = future.result()

            # 방어적으로 tuple / 문자열 모두 처리
            if isinstance(result, tuple):
                ticker, tag = result
            else:
                ticker, tag = None, result

            if tag == "OK":
                done += 1
            elif tag == "SKIP":
                skip += 1
            elif tag == "EMPTY":
                empty += 1
            else:
                fail += 1

    print(f"[COMPLETE] Trading(수급): OK={done}, SKIP={skip}, EMPTY={empty}, FAIL={fail}")

    # 5) 로컬 -> GDrive 일괄 복사 (기존 로직 유지)
    if gdrive_trading_path is not None:
        os.makedirs(gdrive_trading_path, exist_ok=True)

        print("\n" + "=" * 50)
        print("[INFO] 로컬에 저장된 Trading(수급) 데이터를 Google Drive로 복사 시작.")
        print(f"  - 원본 (Local): {path}")
        print(f"  - 대상 (GDrive): {gdrive_trading_path}")

        # -a : 권한/타임스탬프 보존, -n : 이미 존재하는 파일은 덮어쓰지 않음
        exit_code = os.system(f'cp -a -n "{path}/." "{gdrive_trading_path}"')

        if exit_code == 0:
            print("[SUCCESS] Trading(수급) 복사 완료.")
        else:
            print(f"[WARN] Trading(수급) 복사 중 오류 발생 (exit code={exit_code}).")

    print(f"[COMPLETE] Trading(수급) 데이터 수집 완료.")


def run_leakage_check(paths_to_check, end_date_str, sample_size=10):
    """
    데이터 수집 완료 후, END_DATE 이후의 데이터가
    저장되었는지(미래 데이터 누수) 확인하는 디버깅 함수
    
    Args:
        paths_to_check (dict): 검사할 GDrive 경로 딕셔너리
        end_date_str (str): 설정된 수집 종료일
        sample_size (int): 각 폴더당 검사할 파일 샘플 개수
    """
    print("\n" + "="*55)
    print(f"[DEBUG] 🔍 미래 데이터 누수 점검 시작 (END_DATE: {end_date_str})...")
    print(f"        (각 폴더당 최대 {sample_size}개 CSV 파일을 샘플링하여 날짜 인덱스 확인)")
    
    try:
        max_allowed_date = pd.to_datetime(end_date_str)
    except Exception as e:
        print(f"  - [ERROR] END_DATE '{end_date_str}'를 날짜로 변환 실패: {e}")
        return

    found_leakage = False

    # paths_to_check는 GDrive 경로가 포함된 딕셔너리
    for data_type, directory in paths_to_check.items():
        print(f"\n  --- {data_type} ({directory}) 점검 중 ---")
        
        # .complete 플래그 파일 등은 제외
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        
        if not csv_files:
            print("  - [INFO] 점검할 CSV 파일이 없습니다.")
            continue
            
        # 샘플링
        sample_files = random.sample(csv_files, min(len(csv_files), sample_size))
        
        for filepath in sample_files:
            filename = os.path.basename(filepath)
            try:
                # FRED 데이터는 인덱스가 없으므로 'date' 또는 'vintage_date' 컬럼을 기준으로 확인
                if data_type == "macro" and "_vintages.csv" in filename: 
                    df = pd.read_csv(filepath)
                    if df.empty: continue
                    
                    df["date"] = pd.to_datetime(df["date"], errors='coerce')
                    df["vintage_date"] = pd.to_datetime(df["vintage_date"], errors='coerce')
                    
                    max_date_in_file = df["date"].max()
                    max_vintage_in_file = df["vintage_date"].max()
                    
                    if max_date_in_file > max_allowed_date or max_vintage_in_file > max_allowed_date:
                        print(f"    - [!! LEAKAGE DETECTED !!] {filename}")
                        if max_date_in_file > max_allowed_date:
                            print(f"      > 🔴 파일 최대 날짜(date): {max_date_in_file.strftime('%Y-%m-%d')}")
                        if max_vintage_in_file > max_allowed_date:
                            print(f"      > 🔴 파일 최대 날짜(vintage): {max_vintage_in_file.strftime('%Y-%m-%d')}")
                        print(f"      > 🟢 허용 최대 날짜: {max_allowed_date.strftime('%Y-%m-%d')}")
                        found_leakage = True
                    
                else: 
                    # 수급(trading) 데이터는 날짜 컬럼이 '날짜'임
                    is_trading_or_ohlcv = (data_type == "ohlcv" or (data_type == "macro" and "_vintages" not in filename))

                    if is_trading_or_ohlcv:
                        # OHLCV, Macro(Flat) 등: index_col=0 (날짜가 인덱스)
                        df = pd.read_csv(filepath, index_col=0)
                        if df.empty: continue
                        df.index = pd.to_datetime(df.index, errors='coerce')
                        df = df[~df.index.isna()] # NaT 제거
                        if df.empty or not isinstance(df.index, pd.DatetimeIndex): continue
                        max_date_in_file = df.index.max()
                    else:
                        # Trading(수급) 등: '날짜' 컬럼 사용 (index_col=False)
                        df = pd.read_csv(filepath) 
                        if df.empty or "날짜" not in df.columns: continue
                        df["날짜"] = pd.to_datetime(df["날짜"], errors='coerce')
                        df = df.dropna(subset=["날짜"])
                        if df.empty: continue
                        max_date_in_file = df["날짜"].max()
                    
                    if max_date_in_file > max_allowed_date:
                        print(f"    - [!! LEAKAGE DETECTED !!] {filename}")
                        print(f"      > 🔴 파일 최대 날짜: {max_date_in_file.strftime('%Y-%m-%d')}")
                        print(f"      > 🟢 허용 최대 날짜: {max_allowed_date.strftime('%Y-%m-%d')}")
                        found_leakage = True
                    
            except Exception as e:
                print(f"    - [WARN] {filename} 파일 점검 중 오류: {e}")
    
    print("="*55)
    if found_leakage:
        print(f"[DEBUG] ❌ 점검 완료. 미래 데이터 누수가 의심되는 파일이 발견되었습니다!")
    else:
        print(f"[DEBUG] ✅ 점검 완료. {end_date_str} 기준 데이터 누수가 발견되지 않았습니다.")
    print("="*55)

def main():
    """
    메인 실행 함수
    """
    # 1. 환경 설정 및 경로 생성을 먼저 호출
    paths = setup_environment() # GDrive 경로가 설정됨

    # 2. 명시적 CachedSession 생성
    #    (pykrx는 POST를 사용하므로 allowable_methods 추가)
    cache_path = "/content/pykrx_cache"
    
    print(f"[INFO] requests-cache 세션 활성화 (유효기간: 1일).")
    print(f"     캐시 저장 위치: {cache_path}.sqlite") 
    
    cached_session = requests_cache.CachedSession(
        cache_name=cache_path,
        backend="sqlite",
        expire_after=timedelta(days=1), 
        allowable_methods=('GET', 'POST') 
    )
    
    # 3. 티커 목록 가져오기
    tickers = get_all_tickers_in_range(START_DATE, END_DATE)

    # OHLCV/수급 데이터 로컬 저장 경로 설정
    # OHLCV도 파일이 많아 Drive 직접쓰기보다 로컬→일괄복사가 빠름
    gdrive_ohlcv_path   = paths["ohlcv"]
    gdrive_trading_path = paths["trading"]

    local_ohlcv_path   = "/content/rre_local_ohlcv_data"
    local_trading_path = "/content/rre_local_trading_data"
    os.makedirs(local_ohlcv_path, exist_ok=True)
    os.makedirs(local_trading_path, exist_ok=True)

    print("-" * 50)
    print(f"[INFO] ⚠ Google Drive 할당량 방지를 위해 'OHLCV'와 '수급' 데이터는")
    print(f"       로컬({local_ohlcv_path}, {local_trading_path})에 임시 저장 후,")
    print(f"       마지막에 각 GDrive 경로로 일괄 복사합니다.")
    print(f"       (OHLCV 대상: {gdrive_ohlcv_path})")
    print(f"       (수급 대상:  {gdrive_trading_path})")
    print("-" * 50)

    # 실제 수집 경로를 로컬로 바꿔치기
    paths["ohlcv"]   = local_ohlcv_path
    paths["trading"] = local_trading_path

    if tickers:
        # 3. 매크로 데이터 수집 (GDrive에 저장)
        collect_macro_data(paths["macro"], START_DATE, END_DATE)
        
        # 3-1. FRED 지표 추가 수집 (캐시 세션 전달)
        collect_fred_macro(paths["macro"], START_DATE, END_DATE, cached_session)
        
        # 4. OHLCV 데이터 수집 (로컬 임시 경로에 저장)
        #의도적으로 끈거임 버그 아님
        try:
            collect_ohlcv_data_fast(tickers, paths["ohlcv"], START_DATE, END_DATE)
        finally:
            # 다른 단계에 영향 없도록 반드시 해제
            requests_cache.uninstall_cache()
        
        # OHLCV 수집이 끝나자마자 바로 Google Drive로 저장
        print("\n" + "="*50)
        print(f"[INFO] 로컬에 저장된 OHLCV 데이터를 Google Drive로 복사 시작...")
        print(f"  - 원본 (Local): {local_ohlcv_path}")
        print(f"  - 대상 (GDrive): {gdrive_ohlcv_path}")
        os.makedirs(gdrive_ohlcv_path, exist_ok=True)

        # 존재하는 파일만 덮어쓰지 않고 복사(-n). 실패 시 로컬 유지
        ret_ohlcv = os.system(f"cp -a -n -v {local_ohlcv_path}/. {gdrive_ohlcv_path}/")
        if ret_ohlcv == 0:
            print("[SUCCESS] OHLCV 복사 완료.")
            # 공간 확보를 위해 삭제 (선택 사항)
            os.system(f"rm -r {local_ohlcv_path}") 
        else:
            print("[ERROR] OHLCV 복사 실패! 로컬 폴더 유지됨:", local_ohlcv_path)

        # 6. 수급 데이터 수집 (local_trading_path에 저장됨)
        # collect_trading_data 함수는 내부에서 GDrive 복사까지 수행함
        collect_trading_data(tickers, paths["trading"], gdrive_trading_path, START_DATE, END_DATE)
        
        print(f"[INFO] 모든 수급 데이터 처리가 완료되어 로컬 임시 데이터를 삭제합니다...")
        print(f"  - 삭제 대상 (Local): {local_trading_path}")
        
        try:
            if os.path.exists(local_trading_path):
                os.system(f"rm -r {local_trading_path}") # 로컬 런타임 저장 공간 확보
                print("[INFO] 로컬 임시 데이터 삭제 완료.")
            else:
                print("[INFO] 삭제할 로컬 임시 데이터 폴더가 없습니다.")
        except Exception as e:
            print(f"[ERROR] 로컬 임시 데이터 삭제 실패: {e}")
            print(f"       [!!!] 로컬 데이터가 {local_trading_path} 에 남아있을 수 있습니다.")
        
        # ================== [디버깅 코드] ==================
        # 8. 최종 데이터 누수 점검 (GDrive 경로 대상)
        # GDrive 경로로 다시 복원
        paths["ohlcv"] = gdrive_ohlcv_path
        paths["trading"] = gdrive_trading_path 
        
        run_leakage_check(paths, END_DATE, sample_size=10)
        # =========================================================

        print("="*50)
        print("[ALL COMPLETE] 모든 데이터 수집 작업이 완료되었습니다.")
        print("="*50)

    else:
        print("\n[STOP] 수집할 Ticker 목록이 비어있어 데이터 수집을 진행하지 않았습니다.")
        if os.path.exists(local_trading_path):
             os.system(f"rm -r {local_trading_path}")
        if os.path.exists(local_ohlcv_path):
             os.system(f"rm -r {local_ohlcv_path}")


if __name__ == "__main__":
    # Colab 등에서 !pip install pykrx tqdm pandas yfinance requests-cache httpx aiolimiter nest_asyncio 먼저 실행 필요
    main()