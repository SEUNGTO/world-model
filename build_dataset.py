# tick, 뉴스 데이터를 timespan 단위로 변환하여 저장
# 작업 성격상, 한 달 데이터를 사용해야 함
import os
import pandas as pd
import gzip
import shutil
import meta
import torch
import numpy as np
from multiprocessing import Pool, cpu_count

def build_tesnor_process(date, minutes, MAX_OBS_TICKS = 2**13, TICK_FEAT_DIM=13) :
    
    build_timespan_tick(date, minutes=minutes, chunk_size=1000000)
    build_timespan_news(date, minutes=minutes)
    build_tensor_data(MAX_OBS_TICKS = MAX_OBS_TICKS, TICK_FEAT_DIM = TICK_FEAT_DIM)


def build_timespan_tick(date, minutes=10, chunk_size = 100000) :
    print("[ Building timespan tick data ]")

    # 압축해제
    print(" - Decompressing tick data...")
    TICK_ZIPFILE_PATH = "D:\\data\\tick_kosdaq"
    zip_nm = f"SQSNXTRDIJH_{date.year}_{date.day:02}.dat.gz"
    file_nm = zip_nm[:-3]
    zipfile = os.path.join(TICK_ZIPFILE_PATH, zip_nm)
    os.makedirs('tick', exist_ok = True)
    output = os.path.join('tick', file_nm)

    with gzip.open(zipfile, 'rb') as f_in :
        with open(output, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # timespan으로 데이터 쪼개기
    print(f" - Spliting tick data...")
    needed_cols = [v['col_nm_eng'] for v in meta.tick_kosdaq]

    use_cols = [
        'TIME_SIN'         ,   # 시간 벡터화 'TIME_SIN',
        'TIME_COS'         ,   # 시간 벡터화 'TIME_COS',
        'TRD_PRC'          ,   # 체결가격
        'TRDVOL'           ,   # 체결수량
        'BID_MBR_NO'       ,   # 매수회원번호
        'BIDORD_TP_CD'     ,   # 매수호가유형코드
        'BID_INVST_TP_CD'  ,   # 매수투자자구분코드
        'ASK_MBR_NO'       ,   # 매도회원번호
        'ASKORD_TP_CD'     ,   # 매도호가유형코드
        'ASK_INVST_TP_CD'  ,   # 매도투자자구분코드
        'LST_ASKBID_TP_CD',    # 최종매도매수구분코드
        ]
    
    col_name = needed_cols[:len(pd.read_csv(output, sep='|', nrows=0).columns)]
    
    chunks = pd.read_csv(output, 
                         sep="|", header=None, chunksize=chunk_size,
                         dtype={v['no']: v['datatype'] for v in meta.tick_kosdaq})
    
    os.makedirs('timespan_tick', exist_ok = True)
    with Pool(cpu_count()) as pool:
        for _ in pool.starmap(process_chunk_save, [(chunk, col_name, minutes, use_cols) for chunk in chunks]):
            pass
    print()

def build_timespan_news(date, minutes=10) :
    print("[ Building timespan news data ]")
    
    news_path = os.listdir('news')
    news_list = [f for f in news_path if date.strftime('%Y%m') in f]
    
    for file in news_list : 
        
        print(f" - Processing news date: {date.strftime('%Y-%m')} | file : {file}", end = '\r')
        
        file_path = os.path.join("news", file)
        news = pd.read_json(file_path)
        news = news.dropna()
        news['posttime'] = pd.to_datetime(news['posttime'])
        news['period_start'] = news['posttime'].dt.floor(f"{minutes}min")
        news['article'] = news['article'].str.replace('\n', ' ')
        news['news'] = "[title]" + news['title'] + "\t[article]" + news['article']
        news['post_date'] = news['posttime'].dt.strftime("%Y%m%d")
        news['post_time'] = news['posttime'].dt.strftime("%H%M%S")
        
        use_cols = [
            'post_date',
            'post_time',
            'news',
        ]

        os.makedirs('timespan_news', exist_ok = True)
        for period, group in news.groupby('period_start'): 
            out_file = f'timespan_news/[{minutes}min]{period.strftime("%Y-%m-%d %H-%M")}.csv'
            group[use_cols].to_csv(out_file, sep = "\t", index = False)
    print()
    print()

def build_tensor_data(MAX_OBS_TICKS=2**13, TICK_FEAT_DIM=13):

    print("[ Building tensor ]")
    
    tick_dir = 'timespan_tick'
    news_dir = 'timespan_news'
    save_dir = 'processed_dataset'
    os.makedirs(save_dir, exist_ok=True)

    tick_data = sorted(os.listdir(tick_dir))   # 정렬 필수 (시간 순서 보장)
    idx = 0

    for t1, t2 in zip(tick_data, tick_data[1:]):

        # read tick
        t1_tick = pd.read_csv(os.path.join(tick_dir, t1), sep="\t", nrows = MAX_OBS_TICKS)
        t2_tick = pd.read_csv(os.path.join(tick_dir, t2), sep="\t", nrows = MAX_OBS_TICKS)
        
        t1_tick = t1_tick.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        t2_tick = t2_tick.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # read news
        news_path = os.path.join(news_dir, t1)
        if os.path.exists(news_path):
            t1_news = pd.read_csv(news_path, sep="\t")
            news = t1_news["news"].tolist()
        else:
            news = []

        # pad obs tick
        n_obs = len(t1_tick)
        obs_padded = torch.zeros((MAX_OBS_TICKS, TICK_FEAT_DIM), dtype=torch.float32)
        obs_padded[:n_obs] = torch.from_numpy(t1_tick.values).float()
        obs_mask = torch.zeros(MAX_OBS_TICKS, dtype=torch.bool)
        obs_mask[:n_obs] = 1
        
        # pad next tick
        n_next = len(t2_tick)
        nxt_padded = torch.zeros((MAX_OBS_TICKS, TICK_FEAT_DIM), dtype=torch.float32)
        nxt_padded[:n_next] = torch.from_numpy(t2_tick.values).float()
        nxt_mask = torch.zeros(MAX_OBS_TICKS, dtype=torch.bool)
        nxt_mask[:n_next] = 1

        # save as .pt
        sample = {
            "obs_tick": obs_padded,
            "obs_mask": obs_mask,
            "next_tick": nxt_padded,
            "next_mask": nxt_mask,
            "news": news,
        }
        torch.save(sample, os.path.join(save_dir, f"{idx:06d}.pt"))
        idx += 1

    print(f" - Saved {idx} data samples to {save_dir}")
    print()
        
def process_chunk_save(chunk, col_name, minutes, use_cols):
    chunk.columns = col_name
    tm = chunk['TRD_TM'].astype(str).str.zfill(9)
    sec = (tm.str[:2].astype(int)*3600 +
           tm.str[2:4].astype(int)*60 +
           tm.str[4:6].astype(int) +
           tm.str[6:9].astype(int)/1000)
    sec_norm = sec / 86400
    chunk['TIME_SIN'] = np.round(np.sin(2*np.pi*sec_norm), 10)
    chunk['TIME_COS'] = np.round(np.cos(2*np.pi*sec_norm), 10)
    
    # 날짜 변환
    chunk['TRADE_DATE'] = pd.to_datetime(chunk['TRADE_DATE'])
    chunk['TRADE_TIME'] = chunk['TRADE_DATE'] + pd.to_timedelta(sec, unit='s')
    chunk['PERIOD_START'] = chunk['TRADE_TIME'].dt.floor(f'{minutes}min')
    
    out_dir = 'timespan_tick'
    os.makedirs(out_dir, exist_ok=True)
    
    for period, group in chunk.groupby('PERIOD_START'):
        out_file = os.path.join(out_dir, f"[{minutes}min]{period.strftime('%Y-%m-%d %H-%M')}.csv")
        group[use_cols].to_csv(out_file, sep='\t', index=False, mode='a', header=not os.path.exists(out_file))