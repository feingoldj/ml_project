import pandas as p
import re
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

def load_data():
    # read in data from first dataset
    df = p.read_csv("tweets.csv")
    df = df[df['sentiment'] != 'not_relevant']

    y = df['sentiment']
    X = df['text']

    # D = df['date']
    # D = p.Series(convert_date_vec(D))

    # read in data from second dataset 
    df = p.read_csv("dataset2Edited.csv")

    y_new = df['Sentiment']
    X_new = df['TweetText']
    y_new = y_new.astype(str)

    # D_new = df['TweetDate']
    # D_new = p.Series(convert_date_vec(D_new))

    
    # combine both datasets
    y = y.append(y_new)
    X = X.append(X_new)

    # D = p.concat([D, D_new])
    # D = D.reset_index(drop = True)

    # read in stock data
    # sd = p.read_csv("Stock_data.csv", sep=", ")
    # sd_list = []
    # for date in D:
    #     relevant_sd = sd.loc[sd.Date == date]

    #     if(relevant_sd.empty):
    #         sd_list.append(0)
    #     else:
    #         open = float(relevant_sd["Open"].item().replace("$", ""))
    #         close = float(relevant_sd["CloseLast"].item().replace("$", ""))
    #         stock_diff = (close - open)/open
    #         sd_list.append(stock_diff)

    # sd = p.Series(sd_list)

    return X, y # , sd

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
