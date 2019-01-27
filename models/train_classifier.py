import sys
import re
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """Load the data from the path provided as input."""
    # Initialize the DB connection
    engine = create_engine('sqlite:///' + database_filepath)
    
    # Read the table from the DB
    df = pd.read_sql_table('DisasterCategories', engine)
    
    # Partition the dataframe on X and Y
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    return X, Y, list(Y.columns)


def tokenize(text):
    """Tokenize the text provided as input."""
    # Get rid of characters that aren't letters or numbers
    tokens = word_tokenize(re.sub(r'[^\w\s]','',text))
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Get the lemma of each word in a standard form
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """Build the Pipeline of the model."""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC()))
    ])
    
    # Define the parameters to use in the GridSearchCV
    parameters = { 
                'clf__estimator__C': [0.1, 1], # [0.001, 0.01, 0.1, 1, 10]
                'clf__estimator__max_iter':[100, 150] # [100, 200, 500]
    }

    # Use GridSearchCV to find the best parameters for the model
    model = GridSearchCV(pipeline, param_grid=parameters, cv=5, scoring='accuracy', n_jobs=-1)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """Get a report of the model showing precision, recall, f1-score, support and accuracy.

    Keyword arguments:
    model -- model to be evaluated
    X_test -- test partition of the dataset
    Y_test -- target of the test dataset
    category_names -- list of the category names
    """
    print(classification_report(Y_test, model.predict(X_test), target_names = category_names))
    print("\nAccuracy:", accuracy_score(model.predict(X_test), Y_test))

def save_model(model, model_filepath):
    """Save model into pickle file.

    Keyword arguments:
    model -- model to be saved
    model_filepath -- path to save the pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        # Save the arguments as variables
        database_filepath, model_filepath = sys.argv[1:]
        
        # Load the data
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # Build the model
        print('Building model...')
        model = build_model()
        
        # Train the model
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # Evaluate the model
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        # Save the model as a pickle file
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
