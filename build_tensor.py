# ============================================
# Tick & News → Tensor Dataset Builder (with subsampling + shuffle)
# ============================================

import os
import pandas as pd
import torch
import pickle
import numpy as np
import config

# ---------------------------
# Config
# ---------------------------
minutes        = config.minutes
chunk_size     = config.chunk_size
MAX_OBS_TICKS  = config.MAX_OBS_TICKS
TICK_FEAT_DIM  = config.TICK_FEAT_DIM

use_stock = pd.read_excel("use_stock.xlsx", dtype=str)
new_index = {row.iloc[1]: row.iloc[0] for _, row in use_stock.iterrows()}
MAX_FIRM_NUM = len(use_stock)

TICK_DIR  = 'D:\\data\\timespan_tick'
NEWS_DIR  = 'D:\\data\\timespan_news'
INVEST_DIR = 'D:\\data\\invest_info'
SAVE_DIR  = 'D:\\processed_dataset'

use_cols = [
    'JONG_INDEX',
    'TIME_SIN', 'TIME_COS',
    'TRD_PRC', 'TRDVOL',
    'BID_MBR_NO', 'BIDORD_TP_CD', 'BID_INVST_TP_CD',
    'ASK_MBR_NO', 'ASKORD_TP_CD', 'ASK_INVST_TP_CD',
    'LST_ASKBID_TP_CD',
]


# =====================================================
# (A) Subsampling + Shuffle 함수 (핵심)
# =====================================================
def subsample_and_shuffle(df, max_len, feat_dim, keep_ratio=0.6):
    """
    시계열 자기상관을 줄이기 위해
    - 일정 비율 random subsampling
    - shuffle
    - padding + mask 생성
    """

    padded = torch.zeros(max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(max_len, dtype=torch.bool)

    n = len(df)
    if n == 0:
        return padded, mask

    # (1) subsample
    keep_n = max(1, int(n * keep_ratio))
    idx = np.random.choice(n, keep_n, replace=False)

    # (2) shuffle
    np.random.shuffle(idx)
    df_sub = df.iloc[idx]

    # (3) pad
    use_len = min(keep_n, max_len)
    padded[:use_len] = torch.from_numpy(df_sub.values[:use_len]).float()
    mask[:use_len] = 1

    return padded, mask


# =====================================================
# (B) Main tensor builder
# =====================================================
def build_tensor_data(MAX_OBS_TICKS, TICK_FEAT_DIM, MAX_FIRM_NUM):

    print("[ Building tensor dataset ]")
    os.makedirs(SAVE_DIR, exist_ok=True)

    tick_files = sorted(os.listdir(TICK_DIR))[6006:]
    idx = 6006

    for t1, t2 in zip(tick_files, tick_files[1:]):
        try : 

            print(f" - Processing sample {idx:010d} : {t1} & {t2}")

            # ------------------------------------
            # Tick 데이터 로드
            # ------------------------------------
            t1_tick = pd.read_csv(os.path.join(TICK_DIR, t1), sep="\t")
            t2_tick = pd.read_csv(os.path.join(TICK_DIR, t2), sep="\t")

            t1_tick = t1_tick.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            t2_tick = t2_tick.apply(pd.to_numeric, errors='coerce').fillna(0.0)

            # ------------------------------------
            # 뉴스 데이터 로드
            # ------------------------------------
            news_path = os.path.join(NEWS_DIR, t1)
            if os.path.exists(news_path):
                _news = pd.read_csv(news_path, sep="\t")
                news_list = _news["news"].tolist()
            else:
                news_list = []

            # ------------------------------------
            # 투자정보 (invest_info) 로드
            # ------------------------------------
            bs_dt = t1[6:16]
            invest_files = os.listdir(INVEST_DIR)
            match = [f for f in invest_files if bs_dt in f]

            if len(match) > 0:
                with open(os.path.join(INVEST_DIR, match[0]), 'rb') as f:
                    d = pickle.load(f)
                    news_list += d['content'].str.replace("\n", "").tolist()

            # ------------------------------------
            # 텐서 초기화
            # ------------------------------------
            obs_padded = torch.zeros((MAX_FIRM_NUM, MAX_OBS_TICKS, TICK_FEAT_DIM), dtype=torch.float32)
            obs_mask   = torch.zeros((MAX_FIRM_NUM, MAX_OBS_TICKS), dtype=torch.bool)
            obs_firm_mask = torch.zeros(MAX_FIRM_NUM, dtype=torch.bool)

            nxt_padded = torch.zeros((MAX_FIRM_NUM, MAX_OBS_TICKS, TICK_FEAT_DIM), dtype=torch.float32)
            nxt_mask   = torch.zeros((MAX_FIRM_NUM, MAX_OBS_TICKS), dtype=torch.bool)
            nxt_firm_mask = torch.zeros(MAX_FIRM_NUM, dtype=torch.bool)

            # ------------------------------------
            # Firm 단위 데이터 처리 (subsample + shuffle)
            # ------------------------------------
            for N in range(MAX_FIRM_NUM):

                # (A) OBS
                t1_firm = t1_tick[t1_tick['JONG_INDEX'] == N].drop(columns=['JONG_INDEX']).copy()
                obs_vec, obs_m = subsample_and_shuffle(t1_firm, MAX_OBS_TICKS, TICK_FEAT_DIM, keep_ratio=0.6)

                obs_padded[N] = obs_vec
                obs_mask[N] = obs_m
                if obs_m.sum() > 0:
                    obs_firm_mask[N] = 1

                # (B) NEXT
                t2_firm = t2_tick[t2_tick['JONG_INDEX'] == N].drop(columns=['JONG_INDEX']).copy()
                nxt_vec, nxt_m = subsample_and_shuffle(t2_firm, MAX_OBS_TICKS, TICK_FEAT_DIM, keep_ratio=0.6)

                nxt_padded[N] = nxt_vec
                nxt_mask[N] = nxt_m
                if nxt_m.sum() > 0:
                    nxt_firm_mask[N] = 1

            # ------------------------------------
            # (최종 저장)
            # ------------------------------------
            sample = {
                "obs_tick":       obs_padded,
                "obs_mask":       obs_mask,
                "obs_firm_mask":  obs_firm_mask,

                "next_tick":      nxt_padded,
                "next_mask":      nxt_mask,
                "next_firm_mask": nxt_firm_mask,

                "news":           news_list,
            }

            torch.save(sample, os.path.join(SAVE_DIR, f"{idx:010d}.pt"))
            idx += 1
        except Exception as e:
            err = f"[Error] {e} | {t1}, {t2}"
            print(e, t1, t2)
           # with open("tensor_error_log.txt", "a", encoding="utf-8") as f:
           #     f.write(err)
            continue

    print(f" - Saved {idx} samples to: {SAVE_DIR}")
          
        


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    build_tensor_data(MAX_OBS_TICKS, TICK_FEAT_DIM, MAX_FIRM_NUM)
