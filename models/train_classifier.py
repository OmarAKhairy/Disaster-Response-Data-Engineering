import sys
import nltk
import pickle
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    '''
    :param database_filepath:
    :return: X, Y, category_names
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', con=engine)
    X = df["message"]
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    '''
    :param text:
    :return: clean_tokens
    '''
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    :param
    :return: grid_search
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)], 
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4]
    }
    grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=5)
    return grid_search

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    :param model, X_test, Y_test, category_names
    :return:
    '''
    y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        y_pred_column = y_pred[:, i]
        report = classification_report(Y_test.iloc[:, i], y_pred_column)
        print(f"Category: {column}")
        print(report)


def save_model(model, model_filepath):
    '''
    :param model, model_filepath
    :return:
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    print("Model exported successfully as:", model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        best_estimator = model.best_estimator_
        
        print('Evaluating model...')
        evaluate_model(best_estimator, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_estimator, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
