import sys, os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from library_data_2023 import results, laps, telemetry, events

def mclaren_linreg():
    """
    This function analyzes the performance of the McLaren team in the 2023 Formula 1 season.
   Focusing on linear regression analysis, it retrieves data from results, laps, telemetry, and events.
    """

    results_df = results.open_results_data()
    laps_df = laps.open_laps_data()
    telemetry_df = telemetry.open_telemetry_data()
    events_df = events.open_events_data()



    return results_df, laps_df, telemetry_df, events_df

def mclaren_forestreg():
    """
    This function analyzes the performance of the McLaren team in the 2023 Formula 1 season.
    Focusing on random forest regression analysis, it retrieves data from results, laps, telemetry, and events.
    """

    results_df = results.open_results_data()
    laps_df = laps.open_laps_data()
    telemetry_df = telemetry.open_telemetry_data()
    events_df = events.open_events_data()

    return results_df, laps_df, telemetry_df, events_df

def mclaren_TimeSeriesCv():
    """
    This function analyzes the performance of the McLaren team in the 2023 Formula 1 season.
    Focusing on time series cross-validation, it retrieves data from results, laps, telemetry, and events.
    """

    results_df = results.open_results_data()
    laps_df = laps.open_laps_data()
    telemetry_df = telemetry.open_telemetry_data()
    events_df = events.open_events_data()

    return results_df, laps_df, telemetry_df, events_df

def mclaren_ridge():
    """
    This function analyzes the performance of the McLaren team in the 2023 Formula 1 season.
    Focusing on ridge regression analysis, it retrieves data from results, laps, telemetry, and events.
    """

    results_df = results.open_results_data()
    laps_df = laps.open_laps_data()
    telemetry_df = telemetry.open_telemetry_data()
    events_df = events.open_events_data()

    return results_df, laps_df, telemetry_df, events_df


if __name__ == "__main__":
    print("McLaren Linear Regression Analysis:")
    results_df, laps_df, telemetry_df, events_df = mclaren_linreg()
 

