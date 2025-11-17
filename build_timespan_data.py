import os
import pandas as pd
from tqdm import tqdm
import meta
from multiprocessing import Pool, cpu_count

# 저장 폴더 생성
os.makedirs('timespan_data', exist_ok=True)

# 필요한 컬럼
needed_cols = [v['col_nm_eng'] for v in meta.tick_kosdaq]

def split_tick_to_period(file_nm, minutes=5, chunk_size=100000):
    """
    chunk 단위로 읽고, time span별로 바로 CSV 저장
    """
    # 컬럼 수 확인
    _len = len(pd.read_csv(file_nm, sep="|", nrows=0).columns)
    col_name = needed_cols[:_len]

    # 청크 단위 읽기
    for chunk in pd.read_csv(file_nm, sep="|", header=None, chunksize=chunk_size,
                             dtype={v['no']: v['datatype'] for v in meta.tick_kosdaq}):
        chunk.columns = col_name

        # 날짜 컬럼 변환
        chunk['TRADE_DATE'] = pd.to_datetime(chunk['TRADE_DATE'])

        # TRD_TM 벡터화 → timedelta
        tm = chunk['TRD_TM'].astype(str).str.zfill(9)
        chunk['TRADE_TIME'] = (
            chunk['TRADE_DATE'] +
            pd.to_timedelta(tm.str.slice(0, 2).astype(int), unit='h') +
            pd.to_timedelta(tm.str.slice(2, 4).astype(int), unit='m') +
            pd.to_timedelta(tm.str.slice(4, 6).astype(int), unit='s') +
            pd.to_timedelta(tm.str.slice(6, 9).astype(int), unit='ms')
        )

        # period 시작 시간 계산 (floor)
        period_start = chunk['TRADE_TIME'].dt.floor(f'{minutes}min')
        chunk['PERIOD_START'] = period_start

        # period별 그룹화 후 CSV append
        for period, group in chunk.groupby('PERIOD_START'):
            out_file = f'timespan_data/[TimeSpan_{minutes}]{period.strftime("%Y-%m-%d %H-%M")}.csv'
            group.to_csv(out_file, sep="\t", index=False,
                         mode='a', header=not os.path.exists(out_file))



def process_file(file_nm, minutes):
    """
    멀티프로세싱에서 각 파일 처리
    """

    # 작업
    split_tick_to_period(file_nm, minutes=minutes)

    # 완료 시     
    try:
        os.remove(file_nm)
        print(f"삭제 완료: {file_nm}")
    except Exception as e:
        print(f"파일 삭제 실패: {file_nm}, {e}")


if __name__ == '__main__':
    # 예: 2014년 1월 1일~1월 31일
    date_range = pd.date_range('2010-01', '2014-12-31', freq="MS")

    # time span 단위
    # minutes = 5     # 5분 단위
    minutes = 10    # 10분 단위
    # minutes = 30    # 30분 단위
    # minutes = 60  # 60분 단위
    # minutes = 1440  # 하루 단위

    # 처리할 파일 리스트
    file_list = os.listdir('data')
    file_list = [os.path.join('data', f) for f in file_list]

    # 멀티프로세싱: CPU 코어 수 활용
    args_list = [(f, minutes) for f in file_list]
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.starmap(process_file, args_list), total=len(args_list)))