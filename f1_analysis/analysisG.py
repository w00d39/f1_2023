import sys, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from library_data_2023 import results, events, laps

def team_improvement_data():
    """
    Opens results, events, and laps data.
    Returns all three datasets ready for team improvement analysis.
    """
    results_df = results.open_results_data()
    events_df = events.open_events_data()
    laps_df = laps.open_laps_data()
    return results_df, events_df, laps_df

def analyze_team_improvement(results_df, events_df, laps_df):
    data = laps_df.copy()

    # Prepare lap_time_seconds (always positive)
    data['lap_time_seconds'] = pd.to_timedelta(data['LapTime']).dt.total_seconds().abs()

    # Group by Team and RoundNumber
    team_avg_lap = data.groupby(['Team', 'RoundNumber'])['lap_time_seconds'].mean().reset_index()

    # Quick check
    print("\n[DEBUG] Team Average Lap Times Sample:")
    print(team_avg_lap.head())

    # Unique teams
    teams = team_avg_lap['Team'].dropna().unique()

    for i, team in enumerate(teams, start=1):
        print(f"[G] ({i}/{len(teams)}) Analyzing improvement for team: {team}")

        team_data = team_avg_lap[team_avg_lap['Team'] == team]
        X = team_data[['RoundNumber']]
        y = team_data['lap_time_seconds']

        if len(X) < 2:
            print(f"[WARNING] Not enough data points for {team}. Skipping...")
            continue

        # Linear Regression model
        linreg = LinearRegression()
        linreg.fit(X, y)
        y_pred_lin = linreg.predict(X)

        # Random Forest Regressor model
        rfreg = RandomForestRegressor(random_state=42)
        rfreg.fit(X, y)
        y_pred_rf = rfreg.predict(X)

        # Calculate RMSE
        rmse_lin = np.sqrt(mean_squared_error(y, y_pred_lin))
        rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))

        print(f"[G] Linear Regression RMSE: {rmse_lin:.2f} seconds")
        print(f"[G] Random Forest Regressor RMSE: {rmse_rf:.2f} seconds")

        # Plot actual lap times vs predictions
        plt.figure(figsize=(10, 6))
        plt.plot(team_data['RoundNumber'], y, 'o-', label='Actual Lap Times')
        plt.plot(team_data['RoundNumber'], y_pred_lin, 's--', label='Linear Regression Prediction')
        plt.plot(team_data['RoundNumber'], y_pred_rf, 'd-.', label='Random Forest Prediction')
        plt.xlabel('Round Number', fontsize=14)
        plt.ylabel('Average Lap Time (seconds)', fontsize=14)
        plt.title(f'Average Lap Time Improvement for {team}', fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'team_improvement_{team.replace(' ', '_').replace('/', '_')}.png', dpi=300)
        plt.show()

if __name__ == "__main__":
    results_df, events_df, laps_df = team_improvement_data()
    analyze_team_improvement(results_df, events_df, laps_df)

    print("\nResults DataFrame:")
    print(results_df.head())

    print("\nEvents DataFrame:")
    print(events_df.head())

    print("\nLaps DataFrame:")
    print(laps_df.head())


