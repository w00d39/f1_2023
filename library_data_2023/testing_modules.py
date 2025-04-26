import os
import laps
import sprintlaps
import telemetry
import weather
print(os.getcwd())

laps_df = laps.open_laps_data()
sprintlaps_df = sprintlaps.open_sprintlaps_data()

print(laps_df.columns)

