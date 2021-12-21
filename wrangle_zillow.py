import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import os
import env

def prep_zillow_data():
    '''
    This function pulls in the zillow data, narrows it down to single unit properties, drops or fills in null values,
    and returns a dataframe.
    '''
    # use function to pull in data
    df = get_zillow_data()
    
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