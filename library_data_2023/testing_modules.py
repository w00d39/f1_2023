import os
import laps
import sprintlaps
import fastestlaps
import events
import results
import telemetry
import weather
print(os.getcwd())





laps_df = laps.open_laps_data()

print(laps_df['TrackStatus'].unique())


