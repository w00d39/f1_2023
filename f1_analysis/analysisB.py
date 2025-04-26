import sys, os


# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from library_data_2023 import results, laps, telemetry, events

def mclaren_improvement():
    """
    This function analyzes the performance of the McLaren team in the 2023 Formula 1 season.
    It compares the results of Lando Norris and Oscar Piastri, focusing on their finishing positions
    and the number of laps completed in each
    """

    results_df = results.open_results_data()
    laps_df = laps.open_laps_data()
    telemetry_df = telemetry.open_telemetry_data()
    events_df = events.open_events_data()

    return results_df, laps_df, telemetry_df, events_df


if __name__ == "__main__":
    results_df, laps_df, telemetry_df, events_df = mclaren_improvement()
    print("Results DataFrame:")
    print(results_df.head())
    print("\nLaps DataFrame:")
    print(laps_df.head())
    print("\nTelemetry DataFrame:")
    print(telemetry_df.head())
    print("\nEvents DataFrame:")
    print(events_df.head())
