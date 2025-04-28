import os
import laps
import sprintlaps
import fastestlaps
import events
import results
import telemetry
import weather
print(os.getcwd())






weather_df = weather.open_weather_data()
print(weather_df.head())
print(weather_df.columns)

