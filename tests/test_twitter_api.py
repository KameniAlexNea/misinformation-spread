import missinformation
import json


def test_search_by_id():
    api = missinformation.api
    res = api.request('statuses/show/:%d' % 1124350496712556544)

    data = res.json()
    for key, value in data.items():
        print(key, value)
    
    data = json.loads(res.text)