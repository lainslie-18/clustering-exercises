import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import os
import env

########## Acquire ##########

def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''
    This function takes in user credentials from an env.py file and a database name and creates a connection to the Codeup database through a connection string 
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
mall_customers_sql_query = '''
                    SELECT * FROM customers;
                    '''

def query_mall_customers_data():
    '''
    This function uses the get_connection function to connect to the mall customers database and reads the sql query 
    into a pandas dataframe
    '''
    return pd.read_sql(mall_customers_sql_query,get_connection('mall_customers'))


def get_mall_customers_data():
    '''
    This function checks for a local mall_customers.csv file and reads it into a pandas dataframe, if it exists. If 
    not, it uses the get_connection & query_mall_customers_data functions to query the data and write it locally to a 
    csv file
    '''
    # If csv file exists locally, read in data from csv file.
    if os.path.isfile('mall_customers.csv'):
        df = pd.read_csv('mall_customers.csv', index_col=0)
        
    else:
        
        # Query and read data from mall_customers database
        df = query_mall_customers_data()
        
        # Cache data
        df.to_csv('mall_customers.csv')
        
    return df

########## Remove Outliers ##########

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

########## Split ##########

# Split the data into train, validate, and test
def split_data(df, random_state=369, stratify=None):
    '''
    This function takes in a dataframe and splits the data into train, validate and test samples. 
    Test, validate, and train are 20%, 24%, & 56% of the original dataset, respectively. 
    The function returns train, validate and test dataframes.
    '''
   
    if stratify == None:
        # split dataframe 80/20
        train_validate, test = train_test_split(df, test_size=.2, random_state=random_state)

        # split larger dataframe from previous split 70/30
        train, validate = train_test_split(train_validate, test_size=.3, random_state=random_state)
    else:

        # split dataframe 80/20
        train_validate, test = train_test_split(df, test_size=.2, random_state=random_state, stratify=df[stratify])

        # split larger dataframe from previous split 70/30
        train, validate = train_test_split(train_validate, test_size=.3, 
                            random_state=random_state,stratify=train_validate[stratify])

    # results in 3 dataframes
    return train, validate, test

########## Scale ##########

# create function that scales train, validate, and test datasets using min_maxscaler
def scale_data_min_maxscaler(train, validate, test):
    '''
    This function takes in train, validate, and test data sets, scales them using sklearn's Min_MaxScaler
    and returns three scaled data sets
    '''
    # Create the scaler
    scaler = sklearn.preprocessing.MinMaxScaler(copy=True, feature_range=(0,1))

    # Fit scaler on train dataset
    scaler.fit(train)

    # Transform and rename columns for all three datasets
    train_scaled = pd.DataFrame(scaler.transform(train), columns = train.columns.tolist())
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns = train.columns.tolist())
    test_scaled = pd.DataFrame(scaler.transform(test), columns = train.columns.tolist())

    return train_scaled, validate_scaled, test_scaled

def prep_mall_customers_data():
    '''
    This function pulls in the mall customers data, preps it, scales it, splits it
    and returns three dataframes.
    '''

    # use function to pull in mall customers dataframe
    df = get_mall_customers_data()
    # use function to remove outliers
    col_list = [col for col in df if col in ['age','annual_income']]
    df = remove_outliers(df,1.5,col_list)

    # encode the only categorical column, concatenate back onto original dataframe, and drop gender column
    dummy_df = pd.get_dummies(df.gender, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1).drop(columns = ['gender'])

    # use previously created function to split data into 3 dataframes
    train, validate, test = split_data(df)

    # use previously created function to scale data using min_max scaler
    train_sc, validate_sc, test_sc = scale_data_min_maxscaler(train, validate, test)

    return train_sc, validate_sc, test_sc