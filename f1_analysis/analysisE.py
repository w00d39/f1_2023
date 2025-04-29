import sys, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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
    """
    Merge datasets, perform KMeans clustering on weather features, 
    and build a Gradient Boosting Regressor to predict lap times.
    """

    # Merge datasets
    data = laps_df.merge(events_df, on='RoundNumber').merge(weather_df, on='RoundNumber')
    data['lap_time_seconds'] = pd.to_timedelta(data['LapTime']).dt.total_seconds()

    features = data[['TrackTemp', 'AirTemp', 'Humidity', 'lap_time_seconds']].dropna()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[['TrackTemp', 'AirTemp', 'Humidity']])

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    features['cluster'] = kmeans.fit_predict(scaled_features)

    # Gradient Boosting Regressor
    X = features[['TrackTemp', 'AirTemp', 'Humidity']]
    y = features['lap_time_seconds']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)

    print(f"\n[E] Gradient Boosting RMSE on Lap Times: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} seconds")

    avg_lap_by_cluster = features.groupby('cluster')['lap_time_seconds'].mean()
    print("\nAverage Lap Time per Cluster:")
    print(avg_lap_by_cluster)

    # 1. Hexbin plot
    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(
        features['TrackTemp'],
        features['AirTemp'],
        C=features['cluster'],
        gridsize=40,
        cmap='viridis',
        reduce_C_function=np.mean
    )
    plt.colorbar(hb, label='Average Cluster')
    plt.xlabel('Track Temperature (°C)', fontsize=14)
    plt.ylabel('Air Temperature (°C)', fontsize=14)
    plt.title('Clusters Based on Track and Air Temperature', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('hexbin_clusters.png', dpi=300)
    plt.show()

    # 2. Boxplot by Cluster (with outliers removed)
    # Filter out extreme outliers (keep 1%–99% range) 
    q_low = features['lap_time_seconds'].quantile(0.01)
    q_high = features['lap_time_seconds'].quantile(0.99)
    features_filtered = features[(features['lap_time_seconds'] >= q_low) & (features['lap_time_seconds'] <= q_high)]

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x='cluster',
        y='lap_time_seconds',
        data=features_filtered,
        palette='Set2',
        width=0.6,
        showfliers=False  # Hides dots
    )
    plt.title('Lap Times Distribution by Cluster (Cleaned)', fontsize=18)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Lap Time (seconds)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('better_boxplot_lap_times_by_cluster.png', dpi=300)
    plt.show()

    # Barplot by Event
    event_lap_times = data.groupby('EventName')['lap_time_seconds'].mean().sort_values()

    plt.figure(figsize=(12, 10))
    event_lap_times.plot(kind='barh', color='steelblue')
    plt.xlabel('Average Lap Time (seconds)', fontsize=14)
    plt.ylabel('Event Name', fontsize=14)
    plt.title('Average Lap Time by Event', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('barplot_average_lap_time_by_event.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    laps_df, events_df, weather_df = var_lap_times()
    analyze_lap_times(laps_df, events_df, weather_df)

    # Display the first few rows of the DataFrames
    print("Laps DataFrame:")
    print(laps_df.head())
    print("\nEvents DataFrame:")
    print(events_df.head())
    print("\nWeather DataFrame:")
    print(weather_df.head())
