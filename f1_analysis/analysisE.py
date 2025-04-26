import sys, os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 

from library_data_2023 import laps, events, weather

def var_lap_times():

    laps_df = laps.open_laps_data()
    events_df = events.open_events_data()
    weather_df = weather.open_weather_data()

    return laps_df, events_df, weather_df

def analyze_lap_times(laps_df, events_df, weather_df):
    #Merge dataframes on event_id
    data = laps_df.merge(events_df, on= 'event_id').merge(weather_df, on= 'event_id')
    
    #Feature selection for clustering
    features = data[['lap_time', 'track_temp', 'air_temp']].dropna()
    
    #Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    #Kmeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(scaled_features)

    #Plot clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(features['track_temp'], features['lap_time'], c=data['Cluster'], cmap='viridis', s=50, edgecolor='k')
    plt.title('Lap Times Clustered by Track Temperature and Air Temp', fontsize=16)
    plt.xlabel('Track Temperature', fontsize=14)
    plt.ylabel('Lap Time (s)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()

    #Gradient Boosting Regressor
    X = data[['track_temp', 'air_temp', 'humidity']].fillna(0)
    y = pd.to_timedelta(data['lap_time']).dt.total_seconds()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gbr = GradientBoostingRegressor
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)

    print(f"\n[E] Gradient Boosting RMSE on Lap Times: {mean_squared_error(y_test, y_pred, squared=False):.2f} seconds")

    #Print average lap time per cluster
    print("\nAverage Lap Time per Cluster:")
    avg_lap_by_cluster = data.groupby('cluster')['lap_time'].apply(lambda x: pd.to_timedelta(x).dt.total_seconds().mean())
    print(avg_lap_by_cluster)
  
if __name__ == "__main__":
    laps_df, events_df, weather_df = var_lap_times()

    # Display the first few rows of the DataFrames
    print("Laps DataFrame:")
    print(laps_df.head())
    print("\nEvents DataFrame:")
    print(events_df.head())
    print("\nWeather DataFrame:")
    print(weather_df.head())