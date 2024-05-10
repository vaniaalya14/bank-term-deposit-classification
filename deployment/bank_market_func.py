# Import related library
import numpy as np

# Function for Binary encoding
def binary_mapping(df, var):
    df[var+'_enc'] = df[var].map({'yes': 1, 'no': 0})
    return df

# Function for null imputation
def null_imputer(df, variable, null_value):
    df[variable+'_NA'] = np.where(df[variable] == null_value, 1, 0)
    return df
