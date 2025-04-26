import sys, os


# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from library_data_2023 import laps, sprintlaps

def pit_stops():
    laps_df = laps.open_laps_data()
    sprintlaps_df = sprintlaps.open_sprintlaps_data()

    return laps_df, sprintlaps_df

if __name__ == "__main__":
    laps_df, sprintlaps_df = pit_stops()
    print("Laps DataFrame:")
    print(laps_df.head())
    print("\nSprint Laps DataFrame:")
    print(sprintlaps_df.head())