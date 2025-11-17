# 네이버파이낸스 투자정보 리포트 크롤링
# 기간 : 2014-01-01 ~ 2023-12-31

import os
import json
import requests
from bs4 import BeautifulSoup

def fetch_invest_content(nid) : 
    url = f'https://finance.naver.com/research/invest_read.naver?nid={nid}'
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    header = soup.find('th', {'class' : 'view_sbj'}).text
    title = header.strip().split("\t")[0].strip()

    others = header.strip().split("\t")[-1].strip()
    researcher, postdate, _ = others.split("|")

    researcher = researcher.strip()
    postdate = postdate.strip().replace(".", "-")

    content = soup.find("td", {'class' : 'view_cnt'}).text

    item = {
        'title' : title,
        'researcher' : researcher,
        'postdate' : postdate,
        'content' : content,
    }

    return item

if __name__ == '__main__' :

    start_nid = 7653
    last_nid = 29846
    

    data = []
    for nid in range(start_nid, last_nid + 1) :
        item = fetch_invest_content(nid)
        data.append(item)

    # 폴더 생성
    os.makedirs('data', exist_ok=True)
    with open(f'data/invest_info.json', "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)