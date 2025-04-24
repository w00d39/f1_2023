import pandas as pd 
import os
from sklearn import preprocessing

def open_telemetry_data():
    """
    Load the telemetry data from a CSV file, cleans it, and prepares it for analysis.
    This is accessed as a module to keep everything tidy.
    """

    file_path = 'library_data_2023/data_2023/Telemetry.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    telemetry_df = pd.read_csv(file_path)

    # Cleaning the dataframe

    #ints
    int_columns = ['nGear', 'Throttle', 'DRS', 'RoundNumber']
    for i in int_columns:  # loops thru the columns and converts them to int
        telemetry_df[i] = telemetry_df[i].astype(int)


    #floats
    float_columns = ['RPM', 'Speed']
    for i in float_columns:  # loops thru the columns and converts them to float
        telemetry_df[i] = telemetry_df[i].astype(float)

    #bools
    bool_columns = ['Brake']
    for i in bool_columns:  # loops thru the columns and converts them to bool
        telemetry_df[i] = telemetry_df[i].astype(bool)

    #str
    str_columns = ['DriverNumber', 'EventName', 'Source']
    for i in str_columns:  # loops thru the columns and converts them to string
        telemetry_df[i] = telemetry_df[i].astype(str)

    #timedelta
    td_columns = ['Time', 'SessionTime']
    for i in td_columns:  # loops thru the columns and converts them to timedelta
        telemetry_df[i] = pd.to_timedelta(telemetry_df[i], errors='coerce')

    #timedate
    timedate_columns = ['Date']
    for i in timedate_columns:  # loops thru the columns and converts them to datetime
        telemetry_df[i] = pd.to_datetime(telemetry_df[i], errors='coerce')


    #encoding
    encode_columns = ['Source']
    label_encoder = preprocessing.LabelEncoder()  # creating the label encoder object

    for i in encode_columns:
        telemetry_df[f'{i}_index'] = label_encoder.fit_transform(telemetry_df[i])

    return telemetry_df

