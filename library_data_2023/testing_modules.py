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
laps_df = laps.open_laps_data()
sprintlaps_df = sprintlaps.open_sprintlaps_data()

print(weather_df.head())
print(laps_df.head())
print(sprintlaps_df.head())

print(weather_df.columns)
print(laps_df.columns)
print(sprintlaps_df.columns)