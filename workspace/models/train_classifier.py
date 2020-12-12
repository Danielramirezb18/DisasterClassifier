"""
This Python code builds a multiclass classification model
for disaster input messages.

Processed data is brought.
Then the pipeline model is defined using the best parameters
found in Jupyter Notebook estimation.
After that, the model is trained and evaluated to interpret
the result of the training process
Finally, the model is saved in pickle format for ease of use
in a web application
"""
# import libraries
import sys
#Data handling
import pandas as pd
from sqlalchemy import create_engine
import pickle

#NLP
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#Machine Learnig
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#Define variables
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def load_data(database_filepath):
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('dataset_clean', engine)
    #Split the train data from target varible
    X = df['message']
    columns = list(df.columns)
    columns = columns[4:]
    y = df[columns]
    #Replace a value tha was detected to be wrong
    y['related'].replace(2, 1, inplace=True)
    
    return X, y, columns


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    #Define the best ML pipeline with two estimators and one transformer
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5, max_features=5000, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    #Calculate predictions
    y_pred = model.predict(X_test)
    #Model evaluation
    for i, v in enumerate(category_names):
        print('Class:',v)
        print(classification_report(Y_test[v], y_pred[:,i]))
        print('\n')
        
    pass


def save_model(model, model_filepath):
    
    #Save the results of gridsearchcv on a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))
    
    pass


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
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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