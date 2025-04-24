import pandas as pd
import os
from sklearn import preprocessing

def open_weather_data():
    """
    Load the weather data from a CSV file, cleans it, and prepares it for analysis.
    This is accessed as a module to keep everything tidy.
    

    """
    # Load the weather data
    file_path = 'library_data_2023/data_2023/Weather.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    weather_df = pd.read_csv(file_path)

    # Cleaning the dataframe

    #ints
    int_columns = ['WindDirection', 'RoundNumber']
    for i in int_columns:  # loops thru the columns and converts them to int
        weather_df[i] = weather_df[i].astype(int)

    #floats
    float_columns = ['AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindSpeed']
    for i in float_columns:  # loops thru the columns and converts them to float
        weather_df[i] = weather_df[i].astype(float)

    #bool
    bool_columns = ['Rainfall']
    for i in bool_columns:  # loops thru the columns and converts them to bool
        weather_df[i] = weather_df[i].astype(bool)

    #str
    str_columns = ['EventName']
    for i in str_columns:  # loops thru the columns and converts them to string
        weather_df[i] = weather_df[i].astype(str)

    #timedelta
    dt_columns = ['Time']
    for i in dt_columns:  # loops thru the columns and converts them to timedelta
        weather_df[i] = pd.to_timedelta(weather_df[i], errors='coerce')

    return weather_df

