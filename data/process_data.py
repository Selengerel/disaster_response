import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load Data From CSV Files
    
    Args:
        messages_filepath : Path to messages file
        categories_filepath : Path to categories file
    Returns:
        df : DataFrame after merging both messages ans categories file
    '''
    # load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge these datasets
    df = pd.merge(left=messages,right=categories, how='inner',on=['id'])
    return df

def clean_data(df):
    '''
    Split `categories` into separate category columns.
    Split the values in the `categories` column on the `;` character so that each value becomes a separate column.
    Use the first row of categories dataframe to create column names for the categories data.
    Rename columns of `categories` with new column names.
    
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0].str.split('-', expand = True)
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = list(row[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    #categories.head()
    #print(category_colnames)
    #convert first row value in categories columns to labels
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(-1)
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)
    categories.head()
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join="inner").drop_duplicates()
    return df

def save_data(df, database_filename):
    '''
    Save Cleaned Data to a SQLite database
    Args:
        df : DataFrame which is to be saved
        database_filename : Path to the database file
    
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('dis_resp_mes', engine, index=False, if_exists='replace')
    
def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n MESSAGES: {}\n CATEGORIES: {}'
        .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)
        print('Saving data...\n DATABASE: {}'.format(database_filepath))
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
