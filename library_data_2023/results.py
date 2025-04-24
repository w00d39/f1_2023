import pandas as pd
import os
from sklearn import preprocessing

def open_results_data():
    """
    Load the laps data from a CSV file, cleans it, and prepares it for analysis.
    This is accessed as a module to keep everything tidy.

    """
    # Load the results data
    file_path = 'library_data_2023/data_2023/Results.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    results_df = pd.read_csv(file_path)


    #Dropping unnecessary columns: Q1, Q2, Q3
    results_df.drop(columns=['Q1', 'Q2', 'Q3'], inplace=True)
    #Cleaning the dataframe
    
    #ints
    int_columns = ['DriverNumber', 'GridPosition', 'GridPosition', 'Points', 'RoundNumber']

    for i in int_columns: #loops thru the columns and converts them to int
        results_df[i] = results_df[i].astype(int)


    #str
    str_columns = ['BroadcastName', 'Abbreviation', 'TeamName', 'TeamColor', 'FirstName',
                   'LastName', 'FullName', 'Status', 'Race Type', 'EventName']
    
    for i in str_columns: # loops thru the columns and converts them to string
        results_df[i] = results_df[i].astype(str)
    #timedelta
    td_columns = ['Time']

    for i in td_columns: # loops thru the columns and converts them to timedelta
        results_df[i] = pd.to_timedelta(results_df[i], errors='coerce')

    #encoding
    encode_columns = ['Race Type', 'Status']

    label_encoder = preprocessing.LabelEncoder() #creating the label encoder object

    for i in encode_columns: #encodes the column but also formats it for us
        results_df[f'{i}_index'] = label_encoder.fit_transform(results_df[i])

    return results_df

