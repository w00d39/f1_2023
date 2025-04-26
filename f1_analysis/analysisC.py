import sys, os


# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from library_data_2023 import weather, laps, sprintlaps

def weather_impact():

    weather_df = weather.open_weather_data
    laps_df = laps.open_laps_data()
    sprintlaps_df = sprintlaps.open_sprintlaps_data()

    return weather_df, laps_df, sprintlaps_df


if __name__ == "__main__":
    weather_df, laps_df, sprintlaps_df = weather_impact()

    # Display the first few rows of the DataFrames
    print("Weather DataFrame:")
    print(weather_df.head())
    print("\nLaps DataFrame:")
    print(laps_df.head())
    print("\nSprint Laps DataFrame:")
    print(sprintlaps_df.head())