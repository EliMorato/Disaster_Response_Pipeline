import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load two dataframes from two provided paths amd join them into one dataframe.
    Returns the resultant dataframe."""
    # Load messages and categories from the specified filepaths
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge messages and categories by id
    df = messages.merge(categories, how='inner', on='id')

    return df


def clean_data(df):
    """Clean the dataframe provided as input and return the cleaned dataframe. """
    # Split the categories column to get each category into a column
    categories = df['categories'].str.split(pat=';', expand=True)

    # Assign category names to df columns
    row = categories.iloc[0]
    category_colnames = [i.split('-')[0] for i in row]
    categories.columns = category_colnames

    # Get only the numeric value from each column
    for column in categories:
        categories[column] = categories[column].str.extract('(?P<digit>\d)', expand=False).replace('2','1')
        categories[column] = categories[column].astype(int)

    # Drop categories column
    df.drop('categories', axis=1, inplace=True)

    # Concat df with all the categories columns
    df = pd.concat([df, categories], axis=1)

    # Remove duplicated rows
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """Save dataframe into database.

    Keyword arguments:
    df -- dataframe to be saved
    database_filename -- path to save the dataframe
    """
    # Create a database connection 
    engine = create_engine('sqlite:///' + database_filename)
    
    # Insert df into DisasterCategories table
    df.to_sql('DisasterCategories', engine, index=False)  


def main():
    if len(sys.argv) == 4:
        # Save the arguments as variables
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # Load the dataframe
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # Clean the dataframe
        print('Cleaning data...')
        df = clean_data(df)
        
        # Save the data in a database
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