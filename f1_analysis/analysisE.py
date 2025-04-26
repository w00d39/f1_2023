import sys, os


# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 



from library_data_2023 import laps, events, weather

def var_lap_times():

    laps_df = laps.open_laps_data()
    events_df = events.open_events_data()
    weather_df = weather.open_weather_data()

    return laps_df, events_df, weather_df

if __name__ == "__main__":
    laps_df, events_df, weather_df = var_lap_times()

    # Display the first few rows of the DataFrames
    print("Laps DataFrame:")
    print(laps_df.head())
    print("\nEvents DataFrame:")
    print(events_df.head())
    print("\nWeather DataFrame:")
    print(weather_df.head())