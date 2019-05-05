# import libraries
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet'])
import re
import pickle
import warnings

warnings.filterwarnings("ignore")


def load_data(database_filepath):
    """Load data from database and split it into feature and target variables X and Y.

    Keyword arguments:
    database_name -- the name of the database the data is being extracted from
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessagesCategories', engine)
    X = df.iloc[:, 1]
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Tokenize sentence into words and lemmatize the tokens.

    Keyword arguments:
    text -- String/sentence to be tokenized into words
    """
    # splitting the sentence(s) into tokens/words
    tokens = word_tokenize(text)

    # initializing the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token.lower().strip())
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    """Build a pipeline with CountVectorizer, Tfidf-Transformer and RandomForestClassifier."""

    # set up the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])
    # set up a parameter grid for cross validation
    parameters = {
    'clf__estimator__n_estimators' : [10,100,200],
    'clf__estimator__min_samples_split' : [2,3,5],
    }
    # initialize cross validatiom
    model = GridSearchCV(pipeline, parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the fitted model on test data. Print classification report for each label.
    
    Keyword arguments:
    model -- the fitted model
    X_test -- the test features
    Y_test -- the test labels
    category_names -- the column names of the labels
    """
    # make predictions for test data
    preds = model.predict(X_test)
    preds = pd.DataFrame(preds, columns=Y_test.columns)

    # print classification report for each label
    for category in category_names:
        print(category.upper())
        print(classification_report(Y_test[category], preds[category]))
        print('-----------------------------------------------------')


def overall_performance(model, X_test, Y_test, category_names):
    """Evaluate the fitted model on test data. Returns macro-averaged accuracy, f1-score, precision and recall.
    The "macro" averaging is being used for recall, precision and f1-score to calculate the metrics for each label, and find their unweighted mean.
    This does not take label imbalance into account.

    Keyword arguments:
    model -- the fitted model
    X_test -- the test features
    Y_test -- the test labels
    category_names -- the column names of the labels
    """
    # make predictions for test data
    preds = model.predict(X_test)
    preds = pd.DataFrame(preds, columns=Y_test.columns)

    # instantiate empty lists for the metrics
    acc = []
    f1 = []
    recall = []
    prec = []
    # loop over all columns/labels in the dataset and calculate the 4 metrics for each of them
    for category in category_names:
        acc_ = accuracy_score(Y_test[category], preds[category])
        f1_ = f1_score(Y_test[category], preds[category], average='macro')
        recall_ = recall_score(Y_test[category], preds[category], average='macro')
        prec_ = precision_score(Y_test[category], preds[category], average='macro')
        acc.append(acc_)
        f1.append(f1_)
        recall.append(recall_)
        prec.append(prec_)
    print('Average accuracy: {}'.format(np.mean(acc)))
    print('Average f1-score: {}'.format(np.mean(f1)))
    print('Average recall: {}'.format(np.mean(recall)))
    print('Average precision: {}'.format(np.mean(prec)))


def save_model(model, model_filepath):
    """Saves trained model as pickle file.

    Keyword arguments:
    model -- the fitted model
    model_filepath -- the path to where the file is supposed to be stored
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Performance across all labels:')
        overall_performance(model, X_test, Y_test, category_names)
        print('--------------------------------------------------------------')

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()