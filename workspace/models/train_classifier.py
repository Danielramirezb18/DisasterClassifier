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
    
    '''
    This function load the processed dataframe from the
    created database on proces_data.py , then it divide
    the data into features and target values to be returned
    to the main function
    '''
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('dataset_clean', engine)
    #Split the train data from target varible
    X = df['message']
    columns = list(df.columns)
    columns = columns[4:]
    y = df[columns]
    
    return X, y, columns


def tokenize(text):
    
    '''
    The tokenize function performs NLP processing of the training 
    data, takes a block of text and filters the special catacters 
    it contains, then the function breaks the text into words. 
    Finally, all stopwords are removed and the tagline is extracted
    from each word.
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    
    '''
    This function performs and establishes the ML pipeline, uses two 
    estimators and a transformer. Then we specify the other hyperparameters 
    based on the GridSearchCV performed in the Jupyter Notebook analysis.
    '''
    
    #Define the best ML pipeline with two estimators and one transformer
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    #Define the parameters of param grid
    #Some of them are commented because the training time increases significantly
    parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__max_df': (0.5, 1.0),
            'vect__max_features': (None, 5000),
            'tfidf__use_idf': (True, False)
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    The model evaluation function, estimates the model predictions and 
    then we use some defined metrics such as F1 score, support and 
    precision to evaluate how the model is performing on the training data.
    '''
    
    #Calculate predictions
    y_pred = model.predict(X_test)
    #Model evaluation
    for i, v in enumerate(category_names):
        print('Class:',v)
        print(classification_report(Y_test[v], y_pred[:,i]))
        print('\n')
        
    pass


def save_model(model, model_filepath):
    
    '''
    This function takes de train model on pickle format this is usefull
    because after saving that it can be use on other programs
    '''
    
    #Save the results of gridsearchcv on a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))
    
    pass


def main():
    
    '''
    The main function process all the functions step by step to train de classifier
    '''
    
    
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