import numpy as np
import pandas as pd


import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def train_test_split(people, frac=0.75):
    """
        params:
            people: pd.DataFrame
            frac: number
        return:
            train, test split using frac saving the macro_disparate_impact
    """
    np.random.seed(40)
    train = people[["gender", "job"]].reset_index(   # need to keep the index as a column
        ).groupby(["gender", "job"]                  # split by "group"
        ).apply(lambda x: x.sample(frac=frac) # in each group, do the random split
        ).reset_index(drop=True              # index now is group id - reset it
        ).set_index("Id")                 # reset the original index
    test = people.drop(train.index)
    return train, test

def macro_disparate_impact(people):
    counts = people[['job', 'gender']].groupby(['job', 'gender']).size().unstack('gender')
    counts['disparate_impact'] = counts[['M', 'F']].max(axis='columns') / counts[['M', 'F']].min(axis='columns')
    return counts['disparate_impact'].mean()

def shuffle_transform(data_x, names, data_y=None):
    """
        params:
            data_x: pd.DataFrame
            names: dict
    """
    np.random.seed(40)
    if data_y is None:
        x = data_x.sample(frac=1.)
        return x
    else:
        y = data_y.replace({j:i for i,j in names.items()})
        y = y.sample(frac=1.)
        x = data_x.loc[y.index]
        return x, y



def process_text(text):
    """Process text function.
    Input:
        text: a string containing a tweet
    Output:
        texts_clean: a list of words containing the processed text

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    # text = re.sub(r'\$\w*', '', text)
    # remove old style retweet text "RT"
    # text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'http?:\/\/.*[\r\n]*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    # text = re.sub(r'#', '', text)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    text_tokens = tokenizer.tokenize(text)

    texts_clean = []
    for word in text_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            texts_clean.append(stem_word)

    return texts_clean

def build_freqs(texts, ys):
    """Build frequencies.
    Input:
        texts: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, text in zip(yslist, texts):
        for word in process_text(text):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def test_lookup(func):
    freqs = {('sad', 0): 4,
             ('happy', 1): 12,
             ('oppressed', 0): 7}
    word = 'happy'
    label = 1
    if func(freqs, word, label) == 12:
        return 'SUCCESS!!'
    return 'Failed Sanity Check!'


def lookup(freqs, word, label):
    '''
    Input:
        freqs: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears.
    '''
    pair = (word, label)
    return freqs.get(pair, 0) # freqs.get((word, label), 0)

def add_feature(X, feature_to_add):
    """
        Returns sparse feature matrix with added feature.
        feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')