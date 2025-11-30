# tick, 뉴스 데이터를 timespan 단위로 변환하여 저장
# 작업 성격상, 한 달 데이터를 사용해야 함
import os
import pandas as pd
import torch
import pickle
import pdb


minutes = 10
chunk_size = 1e6
MAX_OBS_TICKS = 2**10
TICK_FEAT_DIM = 11

use_stock = pd.read_excel("use_stock.xlsx", dtype = str)
new_index = {row.iloc[1] : row.iloc[0] for _, row in use_stock.iterrows()}
MAX_FIRM_NUM = len(use_stock)

TICK_DIR = 'D:\\data\\timespan_tick'
NEWS_DIR = 'D:\\data\\timespan_news'
INVEST_DIR = 'D:\\data\\invest_info'
SAVE_DIR = 'processed_dataset'

use_cols = [
        'JONG_INDEX'       ,   # 종목 인덱스
        'TIME_SIN'         ,   # 시간 벡터화
        'TIME_COS'         ,   # 시간 벡터화
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


def build_tensor_data(MAX_OBS_TICKS, TICK_FEAT_DIM, MAX_FIRM_NUM):
    # (N, T, F) 형태의 텐서로 저장
    # N : 종목 수 (MAX_FIRM_NUM)
    # T : 최대 틱 수 (MAX_OBS_TICKS)
    # F : 특징 수 (TICK_FEAT_DIM)

    print("[ Building tensor ]")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    tick_data = sorted(os.listdir(TICK_DIR))

    idx = 0

    for t1, t2 in zip(tick_data, tick_data[1:]):
        
        print(f" - Processing sample {idx:010d} : {t1} & {t2}")
        
        # read tick
        t1_tick = pd.read_csv(os.path.join(TICK_DIR, t1), sep="\t")
        t2_tick = pd.read_csv(os.path.join(TICK_DIR, t2), sep="\t")
        
        t1_tick = t1_tick.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        t2_tick = t2_tick.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # read news
        news_path = os.path.join(NEWS_DIR, t1)
        if os.path.exists(news_path):
            t1_news = pd.read_csv(news_path, sep="\t")
            news = t1_news["news"].tolist()
        else:
            news = []
            
        # read investment info
        invest_path = os.listdir(INVEST_DIR)
        bs_dt = t1[6:16]
        f_name = [f for f in invest_path if bs_dt in f]
        if len(f_name) > 0 :
            with open(os.path.join(INVEST_DIR, f_name[0]), 'rb') as f :
                d = pickle.load(f)
                news += d['content'].str.replace("\n", "").tolist()
        
        obs_firm_mask = torch.zeros(MAX_FIRM_NUM, dtype=torch.bool)
        nxt_firm_mask = torch.zeros(MAX_FIRM_NUM, dtype=torch.bool)

        # 종목별로 차원 나누기
        for N in range(MAX_FIRM_NUM) :

            t1_tick_firm = t1_tick[t1_tick['JONG_INDEX'] == N].drop(columns=['JONG_INDEX']).copy()
            t2_tick_firm = t1_tick[t1_tick['JONG_INDEX'] == N].drop(columns=['JONG_INDEX']).copy()
            
            n_obs = len(t1_tick_firm)
            obs_padded = torch.zeros((MAX_FIRM_NUM, MAX_OBS_TICKS, TICK_FEAT_DIM), dtype=torch.float32)
            obs_mask = torch.zeros((MAX_FIRM_NUM, MAX_OBS_TICKS), dtype=torch.bool)
            if n_obs > 0 :
                obs_padded[N, :n_obs] = torch.from_numpy(t1_tick_firm.values[:MAX_OBS_TICKS]).float()
                obs_mask[N, :n_obs] = 1
                obs_firm_mask[N] = 1

            n_next = len(t2_tick_firm)
            nxt_padded = torch.zeros((MAX_FIRM_NUM, MAX_OBS_TICKS, TICK_FEAT_DIM), dtype=torch.float32)
            nxt_mask = torch.zeros((MAX_FIRM_NUM, MAX_OBS_TICKS), dtype=torch.bool)
            if n_next > 0 :
                nxt_padded[N, :n_next] = torch.from_numpy(t2_tick_firm.values[:MAX_OBS_TICKS]).float()
                nxt_mask[N, :n_next] = 1
                nxt_firm_mask[N] = 1

        # save as .pt
        sample = {
            "obs_tick": obs_padded,
            "obs_mask": obs_mask,
            "obs_firm_mask": obs_mask,

            "next_tick": nxt_padded,
            "next_mask": nxt_mask,
            "next_firm_mask": nxt_firm_mask,

            "news": news,
        }
        torch.save(sample, os.path.join(SAVE_DIR, f"{idx:010d}.pt"))
        idx += 1

    print(f" - Saved {idx} data samples to {SAVE_DIR}")
    print()

if __name__ == "__main__" :
    build_tensor_data(MAX_OBS_TICKS, TICK_FEAT_DIM, MAX_FIRM_NUM)