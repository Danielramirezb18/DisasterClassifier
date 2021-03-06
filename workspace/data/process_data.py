"""
This python code process all the data of messages and categories
It merge both sources and make some transformation to get 
the multiclass target variables for the model.
Finaly all the clean data is stored on SQLite database
"""

#Libraries import
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''
    The data load function takes the path of the source files and uses 
    them to load these in the dataframe format, then we combine both 
    dataframes to get all the complete information to generate the 
    training base.
    '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories,on='id',how='left')
    
    return df


def clean_data(df, categories_filepath):
    
    '''    
    This function takes the category basis and processes it, this file 
    contains all the multiclass classification. First we divide the 
    data by the delimiter and remove the category names implicit in 
    the data. Finally we filter out the non-binary values, merge the 
    two dataframes, and remove the duplicate records.
    '''
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # create a dataframe of the 36 individual category columns
    categories = categories.join(categories.categories.str.split(';',expand=True))
    del categories['categories']

    # select the first row of the categories dataframe
    row = list(categories.iloc[0])
    row[0] = 'id'
    row = [i[0:i.find('-')] for i in row]
    row[0] = 'id'
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:

        if column!= 'id':
            # set each value to be the last character of the string
            categories[column] = categories[column].apply(lambda x: x[-1])

            # convert column from string to numeric
            categories[column] = pd.to_numeric(categories[column])
            
    #Dekete values that are not binary
    categories = categories[categories.related != 2]

    # drop the original categories column from `df`
    del df['categories']

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, on='id', how='inner')

    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    
    '''
    This function takes the clean dataframe and save it on a SQLite database
    '''
    
    #the clean dataframe is saved on a SQLite database table
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('dataset_clean', engine, index=False, if_exists = 'replace')
    
    pass  


def main():
    
    '''
    The main function process all the functions step by step to process
    and clean the data.
    '''
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories_filepath)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()