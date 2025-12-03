# # tick, 뉴스 데이터를 timespan 단위로 변환하여 저장
# # 작업 성격상, 한 달 데이터를 사용해야 함
# import os
# import pandas as pd
# import gzip
# import shutil
# import meta
# import shutil
# import config
# import numpy as np

# from multiprocessing import Pool, cpu_count
# import warnings
# warnings.filterwarnings("ignore")

# minutes = config.minutes
# chunk_size = config.chunk_size

# use_stock = pd.read_excel("use_stock.xlsx", dtype = str)
# new_index = {row[1] : row[0] for _, row in use_stock.iterrows()}
# MAX_FIRM_NUM = len(use_stock)

# TICK_ZIPFILE_PATH = "D:\\data\\tick_kosdaq"
# NEWS_PATH = "D:\\data\\news"

# def build_timespan(date, minutes, chunk_size) :
    
#     build_timespan_tick(date, minutes, chunk_size)
#     build_timespan_news(date, minutes)


# def build_timespan_tick(date, minutes, chunk_size) :
#     print("[ Building timespan tick data ]")

#     # 압축해제
#     print(" - Decompressing tick data...")
    
#     zip_nm = f"SQSNXTRDIJH_{date.year}_{date.month:02}.dat.gz"

#     file_nm = zip_nm[:-3]
#     zipfile = os.path.join(TICK_ZIPFILE_PATH, zip_nm)
#     os.makedirs('tick', exist_ok = True)
#     output = os.path.join('tick', file_nm)

#     with gzip.open(zipfile, 'rb') as f_in :
#         with open(output, "wb") as f_out:
#             shutil.copyfileobj(f_in, f_out)

#     # timespan으로 데이터 쪼개기
#     print(f" - Spliting tick data...")
#     needed_cols = [v['col_nm_eng'] for v in meta.tick_kosdaq]

#     use_cols = [
#         'JONG_INDEX'       ,   # 종목 인덱스
#         'TIME_SIN'         ,   # 시간 벡터화
#         'TIME_COS'         ,   # 시간 벡터화
#         'TRD_PRC'          ,   # 체결가격
#         'TRDVOL'           ,   # 체결수량
#         'BID_MBR_NO'       ,   # 매수회원번호
#         'BIDORD_TP_CD'     ,   # 매수호가유형코드
#         'BID_INVST_TP_CD'  ,   # 매수투자자구분코드
#         'ASK_MBR_NO'       ,   # 매도회원번호
#         'ASKORD_TP_CD'     ,   # 매도호가유형코드
#         'ASK_INVST_TP_CD'  ,   # 매도투자자구분코드
#         'LST_ASKBID_TP_CD',    # 최종매도매수구분코드
#         ]
    
#     col_name = needed_cols[:len(pd.read_csv(output, sep='|', nrows=0).columns)]
    
#     chunks = pd.read_csv(output, 
#                          sep="|", header=None, chunksize=chunk_size,
#                          dtype={v['no']: v['datatype'] for v in meta.tick_kosdaq})

#     os.makedirs('D:\\data\\timespan_tick', exist_ok = True)
    
#     with Pool(cpu_count()) as pool:
#         for _ in pool.starmap(process_chunk_save, [(chunk, col_name, minutes, use_cols) for chunk in chunks]):
#             pass
        
#     # 원본 파일 삭제
#     shutil.rmtree("tick")
#     print()

# def build_timespan_news(date, minutes=10) :
#     print("[ Building timespan news data ]")
    
#     news_path = os.listdir(NEWS_PATH)
#     news_list = [f for f in news_path if date.strftime('%Y%m') in f]
    
#     for file in news_list : 
        
#         print(f" - Processing news date: {date.strftime('%Y-%m')} | file : {file}", end = '\r')
        
#         file_path = os.path.join(NEWS_PATH, file)
#         news = pd.read_json(file_path)
#         news = news.dropna()
#         news['posttime'] = pd.to_datetime(news['posttime'])
#         news['period_start'] = news['posttime'].dt.floor(f"{minutes}min")
#         news['article'] = news['article'].str.replace('\n', ' ')
#         news['news'] = "[title]" + news['title'] + "\t[article]" + news['article']
#         news['post_date'] = news['posttime'].dt.strftime("%Y%m%d")
#         news['post_time'] = news['posttime'].dt.strftime("%H%M%S")
        
#         use_cols = [
#             'post_date',
#             'post_time',
#             'news',
#         ]

#         os.makedirs('D:\\data\\timespan_news', exist_ok = True)
#         for period, group in news.groupby('period_start'): 
#             out_file = os.path.join('D://data//timespan_news', f'[{minutes}min]{period.strftime("%Y-%m-%d %H-%M")}.csv')
#             group[use_cols].to_csv(out_file, sep = "\t", index = False)
#     print()
#     print()
        
# def process_chunk_save(chunk, col_name, minutes, use_cols):

#     chunk.columns = col_name
#     chunk = chunk[chunk['ISIN_CODE'].isin(use_stock['isin_code'])]
#     chunk['JONG_INDEX'] = chunk['ISIN_CODE'].map(new_index)
#     tm = chunk['TRD_TM'].astype(str).str.zfill(9)
#     sec = (tm.str[:2].astype(int)*3600 +
#            tm.str[2:4].astype(int)*60 +
#            tm.str[4:6].astype(int) +
#            tm.str[6:9].astype(int)/1000)
#     sec_norm = sec / 86400
#     chunk['TIME_SIN'] = np.round(np.sin(2*np.pi*sec_norm), 10)
#     chunk['TIME_COS'] = np.round(np.cos(2*np.pi*sec_norm), 10)
    
#     # 날짜 변환
#     chunk['TRADE_DATE'] = pd.to_datetime(chunk['TRADE_DATE'])
#     chunk['TRADE_TIME'] = chunk['TRADE_DATE'] + pd.to_timedelta(sec, unit='s')
#     chunk['PERIOD_START'] = chunk['TRADE_TIME'].dt.floor(f'{minutes}min')
    
#     out_dir = 'D:\\data\\timespan_tick'
#     os.makedirs(out_dir, exist_ok=True)

#     # 기간별로 나누어 저장
#     for period, group in chunk.groupby('PERIOD_START'): 
#         out_file = os.path.join(out_dir, f"[DATE_{period.strftime('%Y-%m-%d')}][TIME_{period.strftime('%H-%M')}].csv")
#         group[use_cols].to_csv(out_file, sep='\t', index=False, mode='a', header=not os.path.exists(out_file))
        
        
# if __name__ == "__main__" :

#     date_range = pd.date_range(start='2018-02-01', end = '2018-09-30', freq = "MS")
#     # (25.12.01.) 2018-01-01~2018-09-30 누락

#     for date in date_range :
#         print(f'[[ DATE : {date.strftime("%Y-%m")} ]]')

#         build_timespan(date, minutes, chunk_size)
#         build_timespan_news(date, minutes)


# tick, 뉴스 데이터를 timespan 단위로 변환하여 저장
# 작업 성격상, 한 달 데이터를 사용해야 함

import os
import pandas as pd
import gzip
import meta
import config
import numpy as np
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG / PATH
# ==========================================================
minutes = config.minutes
chunk_size = int(config.chunk_size)

use_stock = pd.read_excel("use_stock.xlsx", dtype=str)
use_set = set(use_stock["isin_code"])
new_index = {row[1]: row[0] for _, row in use_stock.iterrows()}
MAX_FIRM_NUM = len(use_stock)

TICK_ZIPFILE_PATH = "D:\\data\\tick_kosdaq"
NEWS_PATH = "D:\\data\\news"
TIMESPAN_TICK_PATH = "D:\\data\\timespan_tick"
TIMESPAN_NEWS_PATH = "D:\\data\\timespan_news"

# tick에서 실제로 사용할 컬럼
TICK_USE_COLS = [
    "JONG_INDEX",        # 종목 인덱스
    "TIME_SIN",          # 시간 벡터화
    "TIME_COS",          # 시간 벡터화
    "TRD_PRC",           # 체결가격
    "TRDVOL",            # 체결수량
    "BID_MBR_NO",        # 매수회원번호
    "BIDORD_TP_CD",      # 매수호가유형코드
    "BID_INVST_TP_CD",   # 매수투자자구분코드
    "ASK_MBR_NO",        # 매도회원번호
    "ASKORD_TP_CD",      # 매도호가유형코드
    "ASK_INVST_TP_CD",   # 매도투자자구분코드
    "LST_ASKBID_TP_CD",  # 최종매도매수구분코드
]

# ==========================================================
# MAIN DISPATCH
# ==========================================================
def build_timespan(date, minutes, chunk_size):
    """한 달 단위로 tick + news timespan 데이터 생성"""
    build_timespan_tick(date, minutes, chunk_size)
    build_timespan_news(date, minutes)


# ==========================================================
# (1) TICK: gzip 스트리밍 + chunk 처리
# ==========================================================
def build_timespan_tick(date, minutes, chunk_size):
    print("[ Building timespan tick data ]")

    # ------------------------------
    # 1) gzip 파일 경로
    # ------------------------------
    zip_nm = f"SQSNXTRDIJH_{date.year}_{date.month:02}.dat.gz"
    zipfile = os.path.join(TICK_ZIPFILE_PATH, zip_nm)

    if not os.path.exists(zipfile):
        print(f"  ! Tick zip not found: {zipfile}")
        return

    print(f" - Reading gzip directly: {zipfile}")

    # meta에서 전체 컬럼 이름 가져오기
    needed_cols = [v["col_nm_eng"] for v in meta.tick_kosdaq]

    # timespan tick 저장 폴더
    os.makedirs(TIMESPAN_TICK_PATH, exist_ok=True)

    # ------------------------------
    # 2) gzip 파일을 바로 스트리밍으로 읽어서 chunk 단위 처리
    # ------------------------------
    # 주의: Windows 멀티프로세싱 때문에, 이 함수는 반드시
    #       if __name__ == "__main__" 아래에서만 호출되어야 함.
    with gzip.open(zipfile, "rt", encoding="utf-8", newline="") as f:
        reader = pd.read_csv(
            f,
            sep="|",
            header=None,
            chunksize=chunk_size,
            names=needed_cols,
            dtype={v["no"]: v["datatype"] for v in meta.tick_kosdaq},
        )

        # generator로 바로 전달해서 메모리 폭발 방지
        with Pool(cpu_count()) as pool:
            pool.starmap(
                process_chunk_save,
                ((chunk, minutes) for chunk in reader),
            )

    print(" - Tick timespan build complete.\n")


def process_chunk_save(chunk: pd.DataFrame, minutes: int):
    """
    개별 chunk를 받아서:
    1) use_stock 필터링
    2) JONG_INDEX 매핑
    3) 시간 인코딩 (TIME_SIN, TIME_COS)
    4) PERIOD_START 계산
    5) PERIOD_START별로 파일 append 저장
    """

    # ---------------------------
    # 1) 필터링: use_stock에 포함된 종목만
    # ---------------------------
    # use_set은 전역 set (isin보다 빠름)
    chunk = chunk[chunk["ISIN_CODE"].isin(use_set)]
    if chunk.empty:
        return

    chunk["JONG_INDEX"] = chunk["ISIN_CODE"].map(new_index)

    # ---------------------------
    # 2) 시간 변환 (TRD_TM → 초)
    # ---------------------------
    tm = chunk["TRD_TM"].astype(str).str.zfill(9)
    sec = (
        tm.str[:2].astype(int) * 3600
        + tm.str[2:4].astype(int) * 60
        + tm.str[4:6].astype(int)
        + tm.str[6:9].astype(int) / 1000
    )

    sec_norm = sec / 86400
    chunk["TIME_SIN"] = np.round(np.sin(2 * np.pi * sec_norm), 10)
    chunk["TIME_COS"] = np.round(np.cos(2 * np.pi * sec_norm), 10)

    # ---------------------------
    # 3) 날짜 + 시각 → PERIOD_START
    # ---------------------------
    # 기존 로직 유지 (TRADE_DATE를 datetime으로 변환)
    chunk["TRADE_DATE"] = pd.to_datetime(chunk["TRADE_DATE"])
    chunk["TRADE_TIME"] = chunk["TRADE_DATE"] + pd.to_timedelta(sec, unit="s")
    chunk["PERIOD_START"] = chunk["TRADE_TIME"].dt.floor(f"{minutes}min")

    # ---------------------------
    # 4) PERIOD_START별로 파일 append
    # ---------------------------
    out_dir = TIMESPAN_TICK_PATH
    os.makedirs(out_dir, exist_ok=True)

    for period, group in chunk.groupby("PERIOD_START"):
        out_file = os.path.join(
            out_dir,
            f"[DATE_{period.strftime('%Y-%m-%d')}][TIME_{period.strftime('%H-%M')}].csv",
        )
        group[TICK_USE_COLS].to_csv(
            out_file,
            sep="\t",
            index=False,
            mode="a",
            header=not os.path.exists(out_file),
        )


# ==========================================================
# (2) NEWS PROCESSING
# ==========================================================
def build_timespan_news(date, minutes=10):
    print("[ Building timespan news data ]")

    if not os.path.exists(NEWS_PATH):
        print(f"  ! NEWS_PATH not found: {NEWS_PATH}")
        return

    news_path = os.listdir(NEWS_PATH)
    # 해당 월(YYYYMM)에 해당하는 파일만 선택
    news_list = [f for f in news_path if date.strftime("%Y%m") in f]

    if not news_list:
        print(f"  - No news files for {date.strftime('%Y-%m')}\n")
        return

    os.makedirs(TIMESPAN_NEWS_PATH, exist_ok=True)

    for file in news_list:
        print(
            f" - Processing news date: {date.strftime('%Y-%m')} | file : {file}",
            end="\r",
        )

        file_path = os.path.join(NEWS_PATH, file)
        news = pd.read_json(file_path)
        news = news.dropna()

        news["posttime"] = pd.to_datetime(news["posttime"])
        news["period_start"] = news["posttime"].dt.floor(f"{minutes}min")

        news["article"] = news["article"].str.replace("\n", " ")
        news["news"] = "[title]" + news["title"] + "\t[article]" + news["article"]
        news["post_date"] = news["posttime"].dt.strftime("%Y%m%d")
        news["post_time"] = news["posttime"].dt.strftime("%H%M%S")

        use_cols = ["post_date", "post_time", "news"]

        for period, group in news.groupby("period_start"):
            out_file = os.path.join(
                TIMESPAN_NEWS_PATH,
                f"[{minutes}min]{period.strftime('%Y-%m-%d %H-%M')}.csv",
            )
            group[use_cols].to_csv(out_file, sep="\t", index=False)

    print("\n - News timespan build complete.\n")


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":

    # (예시) 2018-01 ~ 2018-09
    date_range = pd.date_range(start="2018-03-01", end="2018-09-30", freq="MS")    
    date_range = date_range.append(pd.date_range(start = '2020-01-01', end = '2023-12-31', freq = 'MS'))
    

    for date in date_range:
        try : 
            print(f'[[ DATE : {date.strftime("%Y-%m")} ]]')
            build_timespan(date, minutes, chunk_size)

        except Exception as e :
            err = f"[Error] {e} | [Date] {date.strftime('%Y-%m')}"
            with open("error_log.txt", "a", encoding="utf-8") as f:
                f.write(err)