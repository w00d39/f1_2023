import sys, os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from library_data_2023 import results, events, laps

def team_improvement():
    """
    Opens results, events, and laps data.
    Returns merged datasets for team improvement analysis.
    """
    results_df = results.open_results_data()
    events_df = events.open_events_data()
    laps_df = laps.open_laps_data()

    return results_df, events_df, laps_df

def analyze_team_improvement(results_df, events_df, laps_df):
    # Merge dataframes smartly
    data = results_df.merge(events_df, on='event_id').merge(laps_df, on='event_id')

    # Calculate 'improvement' (negative change in race position)
    data['improvement'] = data.groupby('team_id')['position'].diff(periods=-1) * -1
    data['improvement'] = data['improvement'].fillna(0)

    #Feature engineering for modeling
    X = data[['event_round', 'lap_time']].fillna(0)
    y = data['improvement']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Support Vector Regression (SVR)
    svr = SVR()
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)
    print(f"\n[G] SVR RMSE on Improvement Prediction: {mean_squared_error(y_test, y_pred_svr, squared=False):.2f}")

    # ElasticNet Regression
    elastic = ElasticNetCV()
    elastic.fit(X_train, y_train)
    y_pred_elastic = elastic.predict(X_test)
    print(f"[G] ElasticNet RMSE: {mean_squared_error(y_test, y_pred_elastic, squared=False):.2f}")

    # Ridge Regression
    ridge = RidgeCV()
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    print(f"[G] RidgeCV RMSE: {mean_squared_error(y_test, y_pred_ridge, squared=False):.2f}")

    #Plot: True vs Predicted Improvement (Ridge)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='True Improvement', marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred_ridge, label='Predicted Improvement (RidgeCV)', marker='x', linestyle='--', alpha=0.7)
    plt.title('True vsPredicted Team Improvement Over Season', fontsize=16)
    plt.xlabel('Test Samples', fontsize=14)
    plt.ylabel('Improvement (Position Change)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Bonus: Top Improving Teams
    if 'team_name' in data.columns:
        team_improvement_avg = data.groupby('team_name')['improvement'].mean().sort_values(ascending=False)
        print("\nTop 5 Teams by Average Improvement:")
        print(team_improvement_avg.head(5))
    else:
        print("\nTeam names not available for ranking.")
if __name__ == "__main__":
    results_df, events_df, laps_df = team_improvement()
    analyze_team_improvement(results_df, events_df, laps_df)
    print("Results DataFrame:")
    print(results_df.head())
    print("\nEvents DataFrame:")
    print(events_df.head())
    print("\nLaps DataFrame:")
    print(laps_df.head())