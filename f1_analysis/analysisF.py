import sys, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from library_data_2023 import laps, sprintlaps

def pit_stops():
    """
    Opens laps and sprintlaps data.
    Returns both datasets ready for pit stop analysis.
    """
    laps_df = laps.open_laps_data()
    sprintlaps_df = sprintlaps.open_sprintlaps_data()
    return laps_df, sprintlaps_df

def analyze_pit_stops(laps_df, sprintlaps_df):
    # Step 1: Create pit_stop_time correctly (always positive)
    sprintlaps_df['PitInTime'] = pd.to_timedelta(sprintlaps_df['PitInTime'])
    sprintlaps_df['PitOutTime'] = pd.to_timedelta(sprintlaps_df['PitOutTime'])
    sprintlaps_df['pit_stop_time'] = (sprintlaps_df['PitOutTime'] - sprintlaps_df['PitInTime']).dt.total_seconds().abs()

    # Step 2: Create fast_stop label
    sprintlaps_df['fast_stop'] = (sprintlaps_df['pit_stop_time'] < sprintlaps_df['pit_stop_time'].median()).astype(int)

    # Step 3: Classification models
    X = sprintlaps_df[['pit_stop_time']].fillna(0)
    y = sprintlaps_df['fast_stop']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred_log = logreg.predict(X_test)

    rf_class = RandomForestClassifier()
    rf_class.fit(X_train, y_train)
    y_pred_rf = rf_class.predict(X_test)

    print(f"\n[F] Logistic Regression Accuracy on Fast Pit Stops: {accuracy_score(y_test, y_pred_log):.2f}")
    print(f"[F] Random Forest Classifier Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")

    # Step 4: Regression model (Random Forest Regressor)
    regression_data = sprintlaps_df[['pit_stop_time']].dropna()
    X_reg = regression_data[['pit_stop_time']]
    y_reg = regression_data['pit_stop_time']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = rf_reg.predict(X_test_reg)

    print(f"[F] Random Forest Regressor RMSE for Pit Stop Time: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.2f} seconds")

    # Step 5: Plot histogram of pit stop times
    plt.figure(figsize=(10, 6))
    plt.hist(sprintlaps_df['pit_stop_time'].dropna(), bins=25, color='steelblue', edgecolor='black')
    plt.title('Distribution of Pit Stop Times', fontsize=16)
    plt.xlabel('Pit Stop Time (seconds)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('histogram_pit_stop_times.png', dpi=300)
    plt.show()

    # Step 6: Rank Teams by Median Pit Stop Time
    print("\nAttempting to rank teams by fastest pit stops...")

    try:
        pit_stop_ranking = sprintlaps_df.groupby('Team')['pit_stop_time'].median().sort_values()

        print("\nTop 5 Teams by Fastest Median Pit Stop Time:")
        print(pit_stop_ranking.head(5))

        # Create the bar plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(pit_stop_ranking.index, pit_stop_ranking.values, color='mediumseagreen')
        plt.xlabel('Median Pit Stop Time (seconds)', fontsize=14)
        plt.title('Median Pit Stop Time by Team', fontsize=16)
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle='--', alpha=0.5)

        # Add text labels next to bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                     va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('team_median_pit_stop_times.png', dpi=300)
        plt.show()

        # EXTRA: Print the fastest team
        fastest_team = pit_stop_ranking.index[0]
        fastest_time = pit_stop_ranking.iloc[0]
        print(f"\n✅ The fastest team is **{fastest_team}** with a median pit stop time of {fastest_time:.2f} seconds.")

    except Exception as e:
        print(f"[ERROR] Could not rank teams: {e}")

if __name__ == "__main__":
    laps_df, sprintlaps_df = pit_stops()
    analyze_pit_stops(laps_df, sprintlaps_df)

    print("\nLaps DataFrame:")
    print(laps_df.head())

    print("\nSprint Laps DataFrame:")
    print(sprintlaps_df.head())