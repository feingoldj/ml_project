import pandas as p
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import copy

nltk.download('stopwords')
nltk.download('punkt')

# Convert date format
def convert_date(date):
  month_to_num = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
                  "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"}
  date_components = date.split(" ")
  year = date_components[5]
  month = date_components[1]
  day = date_components[2]
  month_num = month_to_num[month.lower()]
  return month_num + "/" + day + "/" + year

def load_data():

    convert_date_vec = np.vectorize(convert_date)

    # read in data from first dataset
    df = p.read_csv("tweets.csv")
    df = df[df['sentiment'] != 'not_relevant']

    y = df['sentiment']
    X = df['text']

    D = df['date']
    D = p.Series(convert_date_vec(D))

    # read in data from second dataset
    df = p.read_csv("dataset2Edited.csv")

    y_new = df['Sentiment']
    X_new = df['TweetText']
    y_new = y_new.astype(str)

    D_new = df['TweetDate']
    D_new = p.Series(convert_date_vec(D_new))


    # combine both datasets
    y = y.append(y_new)
    X = X.append(X_new)

    D = p.concat([D, D_new])
    D = D.reset_index(drop = True)

    # read in stock data
    sd = p.read_csv("stock_data.csv", sep=", ")
    sd_list = []
    for date in D:
        relevant_sd = sd.loc[sd.Date == date]

        if(relevant_sd.empty):
            sd_list.append(0)
        else:
            open = float(relevant_sd["Open"].item().replace("$", ""))
            close = float(relevant_sd["CloseLast"].item().replace("$", ""))
            stock_diff = (close - open)/open
            sd_list.append(stock_diff)

    sd = p.Series(sd_list)

    return X, y, sd

def process_tweets(list_of_tweets):
    processedTweets=[]
    for tweet in list_of_tweets:
        processedTweets.append(_processTweet(tweet))
    return processedTweets

def _processTweet(tweet):
    stopwords = set(list(punctuation))
    tweet = tweet.lower() # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
    return [word for word in tweet if word not in stopwords]

def add_stock_feature(features, stock_changes):
    new_features = []
    for f in range(len(features)):
        new_feat = copy.deepcopy(features[f])
        if(type(new_feat) == list):
            new_feat.append(stock_changes[f])
        else:
            new_feat = np.append(new_feat, stock_changes[f])
        new_features.append(new_feat)
    return new_features
