import os
import re
import json
import time
import asyncio
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse

NAVER_URL = 'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=101'

# 동시에 너무 많이 보내면 차단 위험 있으니 제한
SEM = asyncio.Semaphore(10)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36'}
# TIMEOUT = aiohttp.ClientTimeout(connect=5, total=15)

# async def fetch(session, url):
#     async with SEM:
#         async with session.get(url, headers=HEADERS) as resp:
#             return await resp.text()


async def fetch(session, url):
    async with SEM:
        async with session.get(url, headers=HEADERS) as resp:
            raw = await resp.read()

            # 인코딩 자동 판별
            for enc in ('utf-8', 'euc-kr', 'cp949'):
                try:
                    return raw.decode(enc)
                except:
                    pass

            # 그래도 안되면 손실 복구 방식
            return raw.decode('utf-8', errors='ignore')


async def fetch_last_page(session, datestr):
    url = f"{NAVER_URL}&date={datestr}&page=9999"
    html = await fetch(session, url)
    soup = BeautifulSoup(html, 'html.parser')
    strong = soup.find('div', class_='paging').find('strong')
    return int(strong.text)


async def fetch_article(session, url):
    html = await fetch(session, url)

    # 특수 토큰 정리 (Naver 이미지 placeholder)
    html = re.sub(r'<!\[\[.*?\]\]>', '', html)

    # 1차: lxml 파서
    try:
        soup = BeautifulSoup(html, 'lxml')
    except Exception:
        # 2차: html5lib fallback
        try:
            soup = BeautifulSoup(html, 'html5lib')
        except Exception:
            return None, None

    dic_area = soup.find('article', id='dic_area')
    article = dic_area.get_text(" ", strip=True) if dic_area else None

    datestamp = soup.find('span', {'class': re.compile('media_end_head_info_datestamp_time')})
    posttime = datestamp['data-date-time'] if datestamp else None

    return posttime, article


async def fetch_news_page(session, datestr, page):
    list_url = f"{NAVER_URL}&date={datestr}&page={page}"
    html = await fetch(session, list_url)
    soup = BeautifulSoup(html, 'html.parser')

    list_body = soup.find('div', class_='list_body')
    items = list_body.find_all('li')

    tasks = []
    for item in items:
        link = item.find_all('a')[-1]
        title = link.get_text(strip=True)
        url = link['href']

        tokens = urlparse(url).path.split('/')
        doc_id = 'nn-' + '-'.join(tokens[-2:])

        tasks.append((doc_id, title, url))

    results = []
    for doc_id, title, url in tasks:
        posttime, article = await fetch_article(session, url)
        results.append({
            'doc_id': doc_id,
            'title': title,
            'url': url,
            'posttime': posttime,
            'article': article,
        })
    return results


async def crawl_date(datestr):
    start = time.perf_counter()  # 시간 측정 시작

    async with aiohttp.ClientSession() as session:
        last_page = await fetch_last_page(session, datestr)

        tasks = [fetch_news_page(session, datestr, p) for p in range(1, last_page + 1)]
        pages = await asyncio.gather(*tasks)

        data = [item for sub in pages for item in sub]

        os.makedirs('data', exist_ok=True)
        with open(f'data/news_{datestr}.json', "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    end = time.perf_counter()  # 시간 측정 종료
    print(f">>> {datestr} 완료 | 기사 {len(data)}개 | 소요 시간: {end - start:.2f}초")


async def main():

    start_date = '20150306'
    end_date = '20171231'
    date_list = pd.date_range(start_date, end_date, freq='d')
    for date in date_list:
        datestr = date.strftime('%Y%m%d')
        print(f"\n===== {datestr} 크롤링 시작 =====")
        await crawl_date(datestr)


if __name__ == "__main__":
    asyncio.run(main())
