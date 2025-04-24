import pandas as pd
import os
from sklearn import preprocessing

def open_fastestlaps_data():
    """
    Load the fastest laps data from a CSV file and cleans it for analysis.
    This function is accessed as a module to keep everything tidy.
    """
    # Load the fastest laps data
    file_path = 'library_data_2023/data_2023/Fastest_Laps.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    fastestlaps_df = pd.read_csv(file_path)

    # Clean the dataframe
   
    #drop the pitouttime, and pitintime because they have nothing in them
    fastestlaps_df.drop(columns=['PitOutTime', 'PitInTime'], inplace=True)
    #simple data type conversons

    #ints
    int_columns = ['LapNumber', 'Stint', 'TrackStatus', 'RoundNumber']

    for i in int_columns: #loops thru the columns and converts them to int
        fastestlaps_df[i] = fastestlaps_df[i].astype(int)
    #floats
    float_columns = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'TyreLife']

    for i in float_columns:# loops thru the columns and converts them to float
        fastestlaps_df[i] = fastestlaps_df[i].astype(float)

    #bool
    bool_columns = ['IsPersonalBest', 'IsAccurate', 'FreshTyre']

    for i in bool_columns: # loops thru the columns and converts them to bool
        fastestlaps_df[i] = fastestlaps_df[i].astype(bool)

    #string
    str_columns = ['DriverNumber', 'Compound', 'Team', 'Driver', 'EventName']

    for i in str_columns: # loops thru the columns and converts them to string
        fastestlaps_df[i] = fastestlaps_df[i].astype(str)

    #timedelta conversion
    dt_columns = ['Time', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime',
                   'Sector2SessionTime', 'Sector3SessionTime', 'LapStartTime']
    
    for i in dt_columns: # loops thru the columns and converts them to timedelta
        fastestlaps_df[i] = pd.to_timedelta(fastestlaps_df[i], errors='coerce')

    #encoding
    encode_column = ['Compound'] #only one to encode
    label_encoder = preprocessing.LabelEncoder() #creating the label encoder object
    for i in encode_column: #encodes the column but also formats it for us
        fastestlaps_df[f'{i}_index'] = label_encoder.fit_transform(fastestlaps_df[i])

    return fastestlaps_df

