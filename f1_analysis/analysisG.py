import sys, os


# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from library_data_2023 import results, events, laps

def team_improvement():
    results_df = results.open_results_data()
    events_df = events.open_events_data()
    laps_df = laps.open_laps_data()

    return results_df, events_df, laps_df

if __name__ == "__main__":
    results_df, events_df, laps_df = team_improvement()
    print("Results DataFrame:")
    print(results_df.head())
    print("\nEvents DataFrame:")
    print(events_df.head())
    print("\nLaps DataFrame:")
    print(laps_df.head())