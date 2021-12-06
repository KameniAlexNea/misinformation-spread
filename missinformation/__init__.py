__version__ = "1.1.0a1+SNAPSHOT"

from TwitterAPI import TwitterAPI

from missinformation.twitter import Twitter

twitter = Twitter.get_twitter_credentials()

api = TwitterAPI(
    consumer_key = twitter.consumer_key,
    consumer_secret = twitter.consumer_secret,
    access_token_key = twitter.access_token_key,
    auth_type="oAuth2"
)