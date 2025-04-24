import pandas as pd
import os
from sklearn import preprocessing

def open_sprintlaps_data():
    """
    Load the sprint laps data from a CSV file, cleans it, and prepares it for analysis.
    This is accessed as a module to keep everything tidy.

    """
    # Load the sprint laps data
    file_path = 'library_data_2023/data_2023/Sprint_Laps.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    

    sprintlaps_df = pd.read_csv(file_path)

    # Cleaning the dataframe

    # ints
    int_columns = ['LapNumber', 'TyreLife', 'Stint', 'TrackStatus', 'RoundNumber']
    for i in int_columns:  # loops thru the columns and converts them to int
        sprintlaps_df[i] = sprintlaps_df[i].astype(int)

    #floats
    float_columns = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
    for i in float_columns:  # loops thru the columns and converts them to float
        sprintlaps_df[i] = sprintlaps_df[i].astype(float)

    #bool
    bool_columns = ['IsPersonalBest', 'IsAccurate', 'FreshTyre']
    for i in bool_columns:  # loops thru the columns and converts them to bool
        sprintlaps_df[i] = sprintlaps_df[i].astype(bool)

    #str
    str_colummns = ['DriverNumber', 'Compound', 'Team', 'Driver', 'Race Type', 'EventName']
    for i in str_colummns:  # loops thru the columns and converts them to string
        sprintlaps_df[i] = sprintlaps_df[i].astype(str)

    #timedelta

    dt_columns = ['Time', 'PitOutTime', 'PitInTime', 'Sector1Time', 
                  'Sector2Time', 'Sector3Time', 'LapStartTime', 'LapTime',
                  'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime']
    for i in dt_columns:  # loops thru the columns and converts them to timedelta
        sprintlaps_df[i] = pd.to_timedelta(sprintlaps_df[i], errors='coerce')
    
    #timedate
    timedate_columns = ['LapStartDate']
    for i in timedate_columns:  # loops thru the columns and converts them to datetime
        sprintlaps_df[i] = pd.to_datetime(sprintlaps_df[i], errors='coerce')
    
    #encoding
    encode_column = ['Compound', 'Race Type'] #only one to encode
    label_encoder = preprocessing.LabelEncoder() #creating the label encoder object

    # Loop through the columns and convert them to the appropriate type
    for i in encode_column:
        sprintlaps_df[f'{i}_index'] = label_encoder.fit_transform(sprintlaps_df[i])

    return sprintlaps_df
