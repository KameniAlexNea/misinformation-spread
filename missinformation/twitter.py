import json

class Twitter:
    def __init__(self, consumer_key, consumer_secret, access_token_key):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token_key = access_token_key
    
    @classmethod
    def from_json(cls, twitter_json):
        return cls(
            twitter_json["API Key"],
            twitter_json["API Key Secret"],
            twitter_json["Bearer Token"]
        )
    
    @staticmethod
    def get_twitter_credentials():
        with open("./twitter/api.json", "r") as conf:
            configuration = json.load(conf)
            twitter = Twitter.from_json(configuration)
        return twitter