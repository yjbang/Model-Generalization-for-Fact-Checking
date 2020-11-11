import pandas as pd
import re
from collections import Counter
from utils.utils import preprocess_tweet

TRAIN_PATH = './data/train.tsv'
VAL_PATH =  './data/valid.tsv'

def count_urls(tweet_list):
    counts = [1 if re.search(r'(https://\S+)', tweet) else 0 for tweet in tweet_list ]
    return sum(counts)

def find_hashtages(tweet_list):
    hashtags_all = []
    for tweet in tweet_list:
        hashtags_per_tweet = []
        for t in tweet.split():
            if '#' in t:
                hashtags_per_tweet.append(t.replace(':',''))
        hashtags_all.append(hashtags_per_tweet)
    return hashtags_all

def find_mentions(tweet_list):
    mentions_all = []
    for tweet in tweet_list:
        mentions_per_tweet = []
        for t in tweet.split():
            if '@' in t and re.search(r'@\w+', t.replace(':','')):
                mentions_per_tweet.append(re.search(r'@\w+', t.replace(':',''))[0])
        mentions_all.append(mentions_per_tweet)

    return mentions_all

def find_most(list_all, k=-1):
    if k == -1:
        k = len(list_all)
    return Counter(sum(list_all, [])).most_common(k)

if __name__ == "__main__":
    df = pd.read_csv(TRAIN_PATH, sep='\t')
    real_data = df[df['label'] == 'real']
    fake_data = df[df['label'] == 'fake']

    real_tweets = [row['tweet'] for idx, row in real_data.iterrows()]
    fake_tweets = [row['tweet'] for idx, row in fake_data.iterrows()]
    real_tweets_preprocessed = [preprocess_tweet(row['tweet']) for idx, row in real_data.iterrows()]
    fake_tweets_preprocessed = [preprocess_tweet(row['tweet']) for idx, row in fake_data.iterrows()]

    real_tweets_len = len(real_tweets)
    fake_tweets_len = len(fake_tweets)

    print("Real: {}\tFake: {}".format(len(real_tweets), len(fake_tweets)))
    print('=' * 80)

    print("Mentions")
    print("Real:", find_most(find_mentions(real_tweets_preprocessed), 10))
    print('\n')
    print("Fake:", find_most(find_mentions(fake_tweets_preprocessed), 10))

    print('=' * 80)
    print("URL Number avg.")
    print("Real:", round(count_urls(real_tweets) / real_tweets_len, 3))
    print("Fake:", round(count_urls(fake_tweets) / fake_tweets_len, 3))

    print('=' * 80)
    print("Real:", find_most(find_hashtages(real_tweets_preprocessed), 10))
    print('\n')
    print("Fake:", find_most(find_hashtages(fake_tweets_preprocessed), 10))


