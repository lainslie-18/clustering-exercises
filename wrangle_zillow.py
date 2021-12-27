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


zillow_sql_query =  '''
                    select *
                    from properties_2017
                    join predictions_2017 using(parcelid)
                    join propertylandusetype using(propertylandusetypeid)
                    where propertylandusedesc = 'Single Family Residential'
                    and transactiondate like '2017%%';
                    '''

def query_zillow_data():
    '''
    This function uses the get_connection function to connect to the zillow database and returns the zillow_sql_query read into a pandas dataframe
    '''
    return pd.read_sql(zillow_sql_query,get_connection('zillow'))


def get_zillow_data():
    '''
    This function checks for a local zillow.csv file and reads it into a pandas dataframe, if it exists. If not, it uses the get_connection & query_zillow_data functions to query the data and write it locally to a csv file
    '''
    # If csv file exists locally, read in data from csv file.
    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Query and read data from zillow database
        df = query_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df


########## Clean ##########

def handle_missing_values(df, col_min_required_pct = .5, row_min_required_pct = .7):
    '''
    This function takes in a dataframe and percentage requirements for both columns and rows and returns a dataframe
    with columns and rows removed that are missing more than those percentages of their values.
    '''
    # specify threshhold for column values
    threshold = int(round(col_min_required_pct*len(df.index),0))
    # drop columns with less than the specified percentage of values
    df.dropna(axis=1, thresh=threshold, inplace=True)
    # specify threshhold for row values
    threshold = int(round(row_min_required_pct*len(df.columns),0))
    # drop rows with less than the specified percentage of values
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def remove_outliers(df, k, col_list):
    ''' 
    This function takes in a dataframe, value of k, and a list of columns, removes outliers greater than k times IQR above the 75th percentile and lower than k times IQR below the 25th percentile from the list of columns specified and returns a dataframe
    '''
    
    # loop through each column
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df



def clean_zillow_data(df):
    '''
    This function narrows down the zillow dataset to single unit properties, cleans it, and returns a dataframe.
    '''
   
    # Narrow down to single unit properties
    single_unit = [260,261,262,263,264,265,266,268,273,275,276,279]
    df = df[df.propertylandusetypeid.isin(single_unit)]
    
    # Remove properties with less than 1 bath and bed and <500 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) &
            ((df.unitcnt<=1)|df.unitcnt.isnull()) &
            (df.calculatedfinishedsquarefeet>500)]
    
    # Use function to handle missing values
    df = handle_missing_values(df)
    
    # drop columns that are unnecessary or still missing too many values
    cols_to_drop =['parcelid','id','propertylandusetypeid','finishedsquarefeet12','heatingorsystemtypeid',
               'buildingqualitytypeid','propertyzoningdesc','rawcensustractandblock','censustractandblock','id.1',
               'heatingorsystemdesc','unitcnt','regionidcity']
    df = df.drop(columns=cols_to_drop)
    
    # fill NaNs with mode or other values based on best judgment
    df.calculatedbathnbr.fillna(df.calculatedbathnbr.mode()[0], inplace=True)
    df.fullbathcnt.fillna(df.fullbathcnt.mode()[0], inplace=True)
    df.lotsizesquarefeet.fillna(df.lotsizesquarefeet.mode()[0], inplace=True)
    df.regionidzip.fillna(df.regionidzip.mode()[0], inplace=True)
    df.yearbuilt.fillna(df.yearbuilt.mode()[0], inplace=True)
    df.structuretaxvaluedollarcnt.fillna((df.structuretaxvaluedollarcnt.quantile(.25) + 
                                     df.structuretaxvaluedollarcnt.quantile(.75)) / 2, inplace=True)
    
    # drop rest of observations with null values
    df=df.dropna()

    # add an age column based on year built
    df['age'] = 2020 - df.yearbuilt.astype(int)

    # replace fips number with city they represent for readability
    df.fips = df.fips.replace({6037:'los_angeles',6059:'orange',6111:'ventura'})

    # create dummy variables for fips column and concatenate back onto original dataframe
    dummy_df = pd.get_dummies(df['fips'])
    df = pd.concat([df, dummy_df], axis=1)

    # drop columns that are unnecessary or contain duplicate information
    df = df.drop(columns=['calculatedbathnbr', 'fullbathcnt', 'roomcnt', 'yearbuilt'])

    # change dataypes where it makes sense
    int_col_list = ['bedroomcnt', 'calculatedfinishedsquarefeet', 'latitude', 'longitude', 'lotsizesquarefeet',  'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount']
    obj_col_list = ['regionidcounty', 'regionidzip','assessmentyear']

    for col in df:
        if col in int_col_list:
            df[col] = df[col].astype(int)
        if col in obj_col_list:
            df[col] = df[col].astype(int).astype(object)

    # rename columns for clarity
    df.rename(columns={'bathroomcnt':'bathrooms', 'bedroomcnt':'bedrooms', 'calculatedfinishedsquarefeet':'area',
                       'fips':'counties', 'lotsizesquarefeet':'lot_area','propertycountylandusecode':'landusecode',
                       'structuretaxvaluedollarcnt':'structuretaxvalue', 'taxvaluedollarcnt':'taxvalue',
                       'landtaxvaluedollarcnt':'landtaxvalue','propertylandusedesc':'landusedesc'}, inplace=True)

    df = remove_outliers(df, 1.5, ['bathrooms', 'bedrooms', 'area', 'lot_area', 'structuretaxvalue', 'taxvalue', 'landtaxvalue', 'taxamount', 'logerror','age'])

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

########## Prep ##########

def prep_zillow_data():
    '''
    This function takes in the zillow dataset, cleans it, splits it, scales it and returns train, validate, and test datasets
    ready for exploring and modeling
    '''
    # use a function to acquire and clean the data
    df = clean_zillow_data(get_zillow_data())

    # use a function to split the dataframe into train, validate, and test datasets
    train, validate, test = split_data(df)

     # drop object type columns to prepare for scaling
    train_model = train.drop(columns = ['counties', 'landusecode','regionidcounty','regionidzip',
                                    'assessmentyear','transactiondate','landusedesc'])
    validate_model = validate.drop(columns = ['counties', 'landusecode','regionidcounty','regionidzip',
                                    'assessmentyear','transactiondate','landusedesc'])
    test_model = test.drop(columns = ['counties', 'landusecode','regionidcounty','regionidzip',
                                    'assessmentyear','transactiondate','landusedesc'])
    
    # use a function to scale data for modeling
    train_scaled, validate_scaled, test_scaled = scale_data_min_maxscaler(train_model, validate_model, test_model)
    
    # split scaled data into X_train and y_train
    X_train = train_scaled.drop(columns='logerror')
    y_train = train_scaled.logerror
    X_validate = validate_scaled.drop(columns='logerror')
    y_validate = validate_scaled.logerror
    X_test = test_scaled.drop(columns='logerror')
    y_test = test_scaled.logerror

    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test