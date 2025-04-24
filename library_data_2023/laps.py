import pandas as pd
import os
from sklearn import preprocessing


def open_laps_data(): 
    """
    Load the laps data from a CSV file, cleans it, and prepares it for analysis.
    This is accessed as a module to keep everything tidy.
    
    """
    # Load the laps data
    file_path = 'library_data_2023/data_2023/Laps.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    laps_df = pd.read_csv(file_path)

    #Cleaning the dataframe
    
    #ints
    int_columns = ['LapNumber', 'Stint', 'TrackStatus', 'RoundNumber']

    for i in int_columns: #loops thru the columns and converts them to int
        laps_df[i] = laps_df[i].astype(int)

    #floats
    float_columns = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'TyreLife']

    for i in float_columns:# loops thru the columns and converts them to float
        laps_df[i] = laps_df[i].astype(float)

    #bool
    bool_columns = ['IsPersonalBest', 'IsAccurate', 'FreshTyre']

    for i in bool_columns: # loops thru the columns and converts them to bool
        laps_df[i] = laps_df[i].astype(bool)

    #str
    str_columns = ['DriverNumber', 'Compound', 'Team', 'Driver', 'EventName']

    for i in str_columns: # loops thru the columns and converts them to string
        laps_df[i] = laps_df[i].astype(str)


    #timedelta
    dt_columns = ['Time', 'LapTime', 'PitOutTime', 'PitInTime', 'Sector1Time',
                   'Sector2Time', 'Sector3Time', 'Sector1SessionTime',
                   'Sector2SessionTime', 'Sector3SessionTime', 'LapStartTime']
    
    for i in dt_columns: # loops thru the columns and converts them to timedelta
        laps_df[i] = pd.to_timedelta(laps_df[i], errors='coerce')

    #encoding
    encode_column = ['Compound'] #only one to encode
    label_encoder = preprocessing.LabelEncoder() #creating the label encoder object
    for i in encode_column: #encodes the column but also formats it for us
        laps_df[f'{i}_index'] = label_encoder.fit_transform(laps_df[i])

    return laps_df

