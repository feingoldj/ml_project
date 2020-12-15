from preprocessing import load_data, process_tweets, add_stock_feature
from bag_of_words import bow_build_X
from models import train_models, train_and_test_NB
from tweet2vec import tweet2vec_build_X

def main():
    # X, y, stock_data = load_data()
    print("1. Loading data")
    X, y, sd = load_data()
    print("Finished loading data\n")

    print("2. Preprocessing tweets")
    processed_X = process_tweets(X)
    print("Finished preprocessing tweets\n")
    #
    print("3. Extracting bag-of-words features")
    X_bow_features = bow_build_X(processed_X)
    print("Finished extracting bag-of-word features\n")

    X_features_stocks = add_stock_feature(X_bow_features,sd)

    # #
    print("4. Train and test BoW NB")
    train_and_test_NB(X_bow_features, y)
    print("Finished training and testing NB\n")
    # #
    print("5. Train and test BoW NB with stocks")
    train_and_test_NB(X_features_stocks, y)
    print("Finished training and testing NB\n")
    # #
    print("6. Training BoW models")
    train_models(X_bow_features, y, True)
    print("Finished training models\n")

    print("7. Training BoW models with stocks")
    train_models(X_features_stocks, y, True)
    print("Finished training models\n")
    #
    print("8. Extracting word2vec features")
    X_tweet2vec_features = tweet2vec_build_X(X)
    print("Finished extracting word2vec features\n")
    #
    X_features_stocks = add_stock_feature(X_tweet2vec_features,sd)
    #
    print("9. Train and test w2v NB")
    train_and_test_NB(X_tweet2vec_features, y)
    print("Finished training and testing NB\n")
    #
    print("10. Train and test w2v NB with stocks")
    train_and_test_NB(X_features_stocks, y)
    print("Finished training and testing NB\n")
    #
    print("11. Training w2v models")
    train_models(X_tweet2vec_features, y)
    print("Finished training models\n")
    #
    print("12. Training w2v models with stocks")
    train_models(X_features_stocks, y)
    print("Finished training models\n")

if __name__ == "__main__":
    main()
