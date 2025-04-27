import os
import laps
import sprintlaps
import fastestlaps
import events
import results
import telemetry
import weather
print(os.getcwd())

fastestlaps_df = fastestlaps.open_fastestlaps_data()
print(fastestlaps_df.head())
print(fastestlaps_df.columns)