import nltk
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn import metrics
from nltk.classify import apply_features
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_validate
from sklearn import metrics
from nltk.classify import apply_features
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def train_and_test_NB(X_features, y):
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.4, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    y_train_pred = gnb.fit(X_train, y_train).predict(X_train)

    print(f"Accuracy: {(y_test == y_pred).sum() / len(y_test)}")
    print(f"Training Accuracy: {(y_train == y_train_pred).sum() / len(y_train)}")


def cross_validation(X_features, y, model):
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_validate(model, X_features, y, cv=cv, scoring = ['accuracy', 'recall_weighted', 'precision_weighted'], return_train_score = True)
    # recall_scores = cross_val_score(model, X_features, y, cv=cv, scoring = 'recall_weighted')
    # precision_scores = cross_val_score(model, X_features, y, cv=cv, scoring = 'precision_weighted')
    accuracy = scores["test_accuracy"]
    training_accuracy = scores["train_accuracy"]
    recall = scores["test_recall_weighted"]
    precision = scores["test_precision_weighted"]
    print(f"Average testing accuracy across splits: {accuracy.mean()}")
    print(f"Average training accuracy across splits: {training_accuracy.mean()}")
    print(f"Average recall across splits: {recall.mean()}")
    print(f"Average precision across splits: {precision.mean()}")

def train_models(X_features, y, bag_of_words=False):
    models = []

    models.append(LinearSVC())

    if bag_of_words:
        pipeline = Pipeline([('tfidf', TfidfTransformer()), ('mi', SelectKBest(mutual_info_classif, k=1000)), ('nb', MultinomialNB())])
        models.append(pipeline)

    models.append(SVC(kernel='poly', degree=2))

    models.append(SVC(kernel='rbf'))

    for model in tqdm(models):
        cross_validation(X_features, y, model)

def accuracy(classifier, test_set):
    correct = 0
    for (tweet, expected_label) in tqdm(test_set):
      predicted_label = classifier.classify(tweet)
      if predicted_label == expected_label:
        correct += 1

    print(f"Accuracy: {correct / len(test_set)}")
