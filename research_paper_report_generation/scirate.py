import requests
import bs4
import os


scirate_url = 'https://scirate.com'


def scirate_request(category, date, range_):
    r = requests.get(url=os.path.join(scirate_url, 'arxiv', category),
                     params={
                         'date': date,
                         'range': range_,
                     })
    return r.text


def extract_info(row):
    title_element = row.select_one('.title > a')
    paper_id = title_element['href'].split('/')[-1]
    return {
        'title': title_element.decode_contents(),
        'id': paper_id,
    }


def fetch_top_arxiv_paper_ids(category, date, range_=1, max_result=5):
    soup = bs4.BeautifulSoup(scirate_request(category, date, range_),
                             'html.parser')
    return [
        extract_info(x)['id']
        for x in soup.select_one('ul.papers').select('.row', limit=max_result)
    ]
