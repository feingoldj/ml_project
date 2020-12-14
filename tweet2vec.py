import spacy

def tweet2vec_build_X(X):
    nlp = spacy.load("en_core_web_lg")

    tweet2vec = []

    for tweet in X:
        model = nlp(tweet)
        tweet2vec.append(model.vector)

    return tweet2vec
