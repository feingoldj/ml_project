from preprocessing import load_data, process_tweets
from bag_of_words import bow_build_X
from models import train_models, train_and_test_NB
from tweet2vec import tweet2vec_build_X

def main():
    # X, y, stock_data = load_data()
    print("1. Loading data")
    X, y = load_data()
    print("Finished loading data\n")

    print("2. Preprocessing tweets")
    processed_X = process_tweets(X)
    print("Finished preprocessing tweets\n")

    print("3. Extracting bag-of-words features")
    X_bow_features = bow_build_X(processed_X)
    print("Finished extracting bag-of-word features\n")

    print("4. Train and test NB")
    train_and_test_NB(X_bow_features, y)
    print("Finished training and testing NB\n")

    print("5. Training models")
    train_models(X_bow_features, y, True)
    print("Finished training models\n")

    print("6. Extracting word2vec features")
    X_tweet2vec_features = tweet2vec_build_X(X)
    print("Finished extracting word2vec features\n")

    print("7. Train and test NB")
    train_and_test_NB(X_tweet2vec_features, y)
    print("Finished training and testing NB\n")

    print("8. Training models")
    train_models(X_tweet2vec_features, y)
    print("Finished training models\n")

if __name__ == "__main__":
    main()