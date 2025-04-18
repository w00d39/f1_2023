import pandas as pd

def open_weather_data():
    """
    Load the weather data from a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the weather data.
    """
    # Load the weather data
    weather_df = pd.read_csv('data_2023/Weather.csv')
    
    return weather_df

def about_weather_data():
    about_weather = open_weather_data()
    
    return print(about_weather.columns)