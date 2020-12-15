import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

def update_dict(d, w):
    if(w in d):
        d[w] += 1
    else:
        d[w] = 1
    return d

def sort_dict(d):
    sorted_tuples = sorted(d.items() ,  key=lambda x: x[1])
    new_dict = {}
    for elem in sorted_tuples:
        new_dict[elem[0]] = elem[1]
    return new_dict

def compute_positive_negative_neutral(tokenized_tweets, labels):
    positive_words = {}
    negative_words = {}
    neutral_words = {}
    for t in range(len(tokenized_tweets)):
        for w in tokenized_tweets[t]:
            if(labels[t] == "1"):
                negative_words = update_dict(negative_words, w)
            elif(labels[t] == "5"):
                positive_words = update_dict(positive_words, w)
            elif(labels[t] == "3"):
                neutral_words = update_dict(neutral_words, w)

    positive_words = sort_dict(positive_words)
    negative_words = sort_dict(negative_words)
    neutral_words = sort_dict(neutral_words)

    return positive_words, negative_words, neutral_words


def split_to_xy(d):
    labels = list(d.keys())
    values = []
    for key in d:
        values.append(d[key])
    return np.arange(0,len(labels)), values, labels

def plot_word_categories(positive_words, negative_words, neutral_words):
    p_x, p_y, p_l = split_to_xy(positive_words)
    plt.figure(figsize=(15,10))
    plt.barh(p_l[len(p_l)-50:], p_y[len(p_y)-50:])
    plt.show()

    n_x, n_y, n_l = split_to_xy(negative_words)
    plt.figure(figsize=(15,10))
    plt.barh(n_l[len(n_l)-50:], n_y[len(n_y)-50:])
    plt.show()

    neu_x, neu_y, neu_l = split_to_xy(neutral_words)
    plt.figure(figsize=(15,10))
    plt.barh(neu_l[len(neu_l)-50:], neu_y[len(neu_y)-50:])
    plt.show()

def plot_word_cloud(positive_words, negative_words, neutral_words):
    posWordCloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate_from_frequencies(positive_words)
    plt.figure(figsize=(5,5))
    plt.imshow(posWordCloud)

    negWordCloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate_from_frequencies(negative_words)

    plt.figure(figsize=(5,5))
    plt.imshow(negWordCloud)

    neutralWordCloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate_from_frequencies(neutral_words)

    plt.figure(figsize=(5,5))
    plt.imshow(neutralWordCloud)
