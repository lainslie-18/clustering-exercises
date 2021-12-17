import pandas as pd
import numpy as np

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