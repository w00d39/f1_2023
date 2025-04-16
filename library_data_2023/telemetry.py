import pandas as pd 

def open_telemetry_data():
    """
    Load the telemetry data from a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the telemetry data.
    """
    # Load the telemetry data
    telemetry_df = pd.read_csv('data_2023/Telemetry.csv')
    
    return telemetry_df

def about_telemetry_data():
    about_telemetry = open_telemetry_data()

    return print(about_telemetry.columns)