import sys, os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from library_data_2023 import laps, sprintlaps

def pit_stops():
    """
    Opens laps and sprinlaps data.
    Returns both datasets ready for pit stop analysis.
    """
    laps_df = laps.open_laps_data()
    sprintlaps_df = sprintlaps.open_sprintlaps_data()

    return laps_df, sprintlaps_df

def analyze_pit_stops(laps_df, sprintlaps_df):
    # Creat "fast_stop" label (1 = fast, 0 = not fast)
    sprintlaps_df['fast_stop'] = (sprintlaps_df['pit_stop_time'] < sprintlaps_df['pit_stop_time'].median()).astype(int)

    # Simple feature: pit_stop_time to classify fast stops
    X = sprintlaps_df[['pit_stop_time']].fillna(0)
    y = sprintlaps_df['fast_stop']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    # Logistic Regression Model (simple but mighty)
    logreg = LogisticRegression
    logreg.fit(X_train, y_train)
    y_pred_log = logreg.predict(X_test)

    # Random Forest Classifier Model
    rf_class = RandomForestClassifier()
    rf_class.fit(X_train, y_train)
    y_pred_rf = rf_class.predict(X_test)

    print(f"\n[F] Logistic Regression Accuracy on Fast Pit Stops: {accuracy_score(y_test, y_pred_log):.2f}")
    print(f"[F] Random Forest Classifier Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
    
    # Random Forest Regressor to predict pit stop time itself
    X_reg = sprintlaps_df[['pit_stop_time']].fillna(0)
    y_reg = sprintlaps_df['pit_stop_time']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = rf_reg.predict(X_test_reg)

    print(f"[F] Random Forest Regressor RMSE for Pit Stop Time: {mean_squared_error(y_test_reg, y_pred_reg, squared=False):.2f} seconds")

    # Histogram of pit stop times
    plt.figure(figsize=(10, 6))
    plt.hist(sprintlaps_df['pit_stop_time'].dropna(), bins=25, color='steelblue', edgecolor='black')
    plt.title('Distribution of Pit Stop Times', fontsize=16)
    plt.xlabel('Pit Stop Time (seconds)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Rank teams by median pit stop time
    print("\nTop 5 Teams by Fastest Median Pit Stop Time:")
    if 'team' in sprintlaps_df.columns:
        pit_stop_ranking = sprintlaps_df.groupby('team')['pit_stop_time'].median().sort_values()
        print(pit_stop_ranking.head(5))
    else:
        print("Team information not available in sprintlaps data.")

if __name__ == "__main__":
    laps_df, sprintlaps_df = pit_stops()
    analyze_pit_stops(laps_df, sprintlaps_df)
    print("Laps DataFrame:")
    print(laps_df.head())
    print("\nSprint Laps DataFrame:")
    print(sprintlaps_df.head())