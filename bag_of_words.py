import nltk
from tqdm import tqdm
from nltk.classify import apply_features

def buildVocabulary(preprocessedTrainingData):
    all_words = []
    
    for words in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    
    return word_features

def extract_features(tweet, vocab):
    tweet_words = set(tweet)
    features = []
    for word in vocab:
        features.append(1 if word in tweet_words else 0)
    return features

def bow_build_X(X):
    vocab = buildVocabulary(X)
    features_array = []
    for row in X:
        features_array.append(extract_features(row, vocab))
    return features_array

