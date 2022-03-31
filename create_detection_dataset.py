from glob import glob

import pandas as pd

from load_detection_dataset import load_json


def list_detection_json():
    files = glob("./dataset/detection/*/data/*/*.json")
    return files


def get_class(tweet_pth: str):
    tweet_cls = "real" if ("Real" in tweet_pth) else "fake"
    return tweet_cls


def join_to_json(tweet_pths: list):
    tweets = [
        load_json(tweet_pth) for tweet_pth in tweet_pths
    ]
    tweet_classes = [
        get_class(tweet_pth) for tweet_pth in tweet_pths
    ]
    return tweets, tweet_classes


def create_pandas(tweets, tweet_classes):
    df = pd.DataFrame(tweets)
    df["label"] = tweet_classes
    return df


tweet_pths = list_detection_json()
tweets, tweet_classes = join_to_json(tweet_pths)
df = create_pandas(tweets, tweet_classes)

print(df.head(10))

df.to_json("./Detecting/dataset.json")
