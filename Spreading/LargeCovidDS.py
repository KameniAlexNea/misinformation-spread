import json
import os
import time
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
        time.sleep(1)


def load_fake_news(pth="./dataset/corona_tweets_732.csv", to_remove=[]):
    df = pd.read_csv(pth, header=None)
    df.columns = ["tweet_id", "sentiment"]
    to_fetch = df.tweet_id.unique()
    if to_remove:
        data = pd.DataFrame()
        for pth in to_remove:
            tmp = pd.read_csv(pth, header=None)
            tmp.columns = ["tweet_id", "sentiment"]
            data = pd.concat([data, tmp], ignore_index=True)
        unique_remove = data.tweet_id.unique()
        to_fetch = np.setdiff1d(to_fetch, unique_remove)
    client = get_api(read_conf())
    base_pth = "./tweets"
    look_up_tweets(client, list(to_fetch), base_pth)


load_fake_news(pth="./dataset/corona_tweets_731.csv", to_remove=["./dataset/corona_tweets_732.csv"])
