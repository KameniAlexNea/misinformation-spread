import json
import os
import time
from ctypes import Union
from glob import glob

import numpy as np
import pandas as pd
import tweepy
from tqdm import tqdm


def read_conf():
    data = json.load(open("./tweet_keys_file.json", "r"))
    return data


def get_api(data, wait_on_rate_limit=True):
    api = tweepy.Client(
        data["oauth_token"],
        data["app_key"],
        data["app_secret"],
        wait_on_rate_limit=wait_on_rate_limit,
    )
    return api


def load_json(pth):
    return json.load(open(pth, "r"))


def look_up_tweets(client: tweepy.Client, ids: list, base_pth: str):
    for i in tqdm(range(len(ids) // 100 + 1)):
        tweets_ids = ids[100 * i: 100 * (i + 1)]
        if len(tweets_ids) < 1:
            return
        response = client.get_tweets(
            ids[100 * i: 100 * (i + 1)],
            expansions=[
                "author_id",
                "in_reply_to_user_id",
                "referenced_tweets.id",
                "referenced_tweets.id.author_id",
            ],
            **{
                "tweet.fields": ["created_at", "public_metrics", "lang"],
                "user.fields": ["created_at", "public_metrics", "id"],
            },
        )
        try:
            tweets = response.data
            for tweet in tweets:
                json.dump(tweet.data, open(base_pth + "/" + str(tweet.id) + ".json", "w"))
        except Exception as ex:
            print(ex)


def load_pandas(pth, columns=Union[str, list], join: bool = False):
    columns = list(columns)
    data = pd.read_csv(pth)
    result = [list(set(data[col].values)) for col in columns]
    if join:
        result = list(set([j for i in result for j in i]))
    return result
