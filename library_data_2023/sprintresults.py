import pandas as pd
import os
from sklearn import preprocessing

def open_sprintresults_data():
    """
    Load the sprint results data from a CSV file, cleans it, and prepares it for analysis.
    This is accessed as a module to keep everything tidy.

    """
    # Load the sprint results data
    file_path = 'library_data_2023/data_2023/Sprint_Results.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    sprintresults_df = pd.read_csv(file_path)

    #dropping bc its all null
    sprintresults_df.drop(columns=['Q1', 'Q2', 'Q3'], inplace=True)

    int_columns = ['RoundNumber','Position', 'GridPosition']

    for i in int_columns:  # loops thru the columns and converts them to int
        sprintresults_df[i] = sprintresults_df[i].astype(int)


    float_columns = ['Points']

    for i in float_columns:  # loops thru the columns and converts them to float
        sprintresults_df[i] = sprintresults_df[i].astype(float)

    #str
    str_columns = ['DriverNumber', 'BroadcastName', 'Abbreviation', 'TeamName', 'TeamColor',
                   'FirstName', 'LastName', 'FullName', 'Status', 'Race Type', 'EventName']
    
    for i in str_columns:  # loops thru the columns and converts them to string
        sprintresults_df[i] = sprintresults_df[i].astype(str)
    #timedelta
    td_columns = ['Time']

    for i in td_columns:  # loops thru the columns and converts them to timedelta
        sprintresults_df[i] = pd.to_timedelta(sprintresults_df[i], errors='coerce')
    #encoding
    encode_columns = ['Race Type', 'Status']
    label_encoder = preprocessing.LabelEncoder() #creating the label encoder object

    for i in encode_columns:
        sprintresults_df[f'{i}_index'] = label_encoder.fit_transform(sprintresults_df[i])

    return sprintresults_df
