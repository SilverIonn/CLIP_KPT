from pprint import pprint
from urllib.request import urlopen, Request

import json

def get_related_words(word):
    if ' ' not in word:
        url = f"https://relatedwords.org/api/related?term={word}"
    else:
        # print(word)
        url = f"https://relatedwords.org/api/related?term={'%20'.join(word.split(' '))}"
        # print(url)
        
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}
    req = Request(url=url, headers=headers)
    html = urlopen(req).read()
    text = html.decode('utf-8')
    return json.loads(text)