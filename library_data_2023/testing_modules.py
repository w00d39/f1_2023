import os
import events
import fastestlaps
print(os.getcwd())

events_df = events.open_events_data()
print(events_df.dtypes)
print(events_df['Session1Date'].head())