import sys, os
import pandas as pd
import numpy as np
from sklearn. preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from library_data_2023 import telemetry, laps, events, weather, results, fastestlaps

telemetry_df = telemetry.open_telemetry_data()
laps_df = laps.open_laps_data()
events_df = events.open_events_data()
weather_df = weather.open_weather_data()
results_df = results.open_results_data()
fastestlaps_df = fastestlaps.open_fastestlaps_data()

def straights_speed():
    #grabbing checo's telemetry data and laps data
    per_laps_df = laps_df[laps_df['DriverNumber'] == '11']
    per_telemetry = telemetry_df[telemetry_df['DriverNumber'] == '11']
    
    # Extract lap times for mapping telemetry to laps
    per_lap_times = per_laps_df[['RoundNumber', 'LapNumber', 'LapStartTime', 'LapTime']].copy()
    
    # Convert LapTime to seconds if needed for later calculations
    if per_lap_times['LapTime'].dtype == 'object':  
        per_lap_times['LapTime'] = per_lap_times['LapTime'].apply(
            lambda x: float(x.split(':')[0])*60 + float(x.split(':')[1]) if isinstance(x, str) and ':' in x else None)
    
    # Calcing lap end time
    per_lap_times['LapEndTime'] = per_lap_times['LapStartTime'] + per_lap_times['LapTime']
    
    # Function to find which lap a telemetry point belongs to
    def assign_lap(telemetry_df, lap_times_df):
        # Initialize empty lists to store lap numbers
        lap_numbers = []
        
        # For each telemetry data point it loops through the telemetry data and finds the lap number
        for _, row in telemetry_df.iterrows():
            session_time = row['SessionTime']
            round_num = row['RoundNumber']
            
            # Find the lap where this telemetry point belongs
            lap_info = lap_times_df[
                (lap_times_df['RoundNumber'] == round_num) & 
                (lap_times_df['LapStartTime'] <= session_time) & 
                (lap_times_df['LapEndTime'] >= session_time)
            ]
            
            if len(lap_info) > 0: #if there is a lap info to be had
                lap_numbers.append(lap_info.iloc[0]['LapNumber']) #iloc is used to get the lap number specificalyl bc it kept giving me a warning of depreication
            else:
                # If no matching lap is found, use nan
                lap_numbers.append(np.nan)
        
        return lap_numbers
    
    # Assign lap numbers to telemetry data
    print("Assigning lap numbers to telemetry data...") #sanity check
    sample_size = 50000  # in case it was too big and going to crash my mcbook
    
    if len(per_telemetry) > sample_size: #if the telemetry data is too big which it will be bc we are not dealing with logan sargeant here
        print(f"  Processing a sample of {sample_size} rows for Perez...")
        per_telemetry_sample = per_telemetry.sample(sample_size, random_state= 301) #sampling the telemetry data 
        per_telemetry_sample['LapNumber'] = assign_lap(per_telemetry_sample, per_lap_times) #assigning the lap numbers to the telemetry data
    else: #error handling copilot did this 
        print(f"  Processing all {len(per_telemetry)} rows for Perez...")
        per_telemetry['LapNumber'] = assign_lap(per_telemetry, per_lap_times)
        per_telemetry_sample = per_telemetry
        
    # Needing to make sense of the DRS numerics
    print("Creating lap-level DRS features...") #sanity check
    per_telemetry_sample = per_telemetry_sample.dropna(subset=['LapNumber']) #dropping the telemetry data that does not have lap numbers
    
    #grouping by round number and lap number to get the DRS data and do math on it
    per_drs_by_lap = per_telemetry_sample.groupby(['RoundNumber', 'LapNumber']).agg({
        'DRS': [
            ('DRS_used', lambda x: (x > 0).any()), #drs is used if it is greater than 0
            ('DRS_usage_pct', lambda x: (x > 0).mean()), #how much checo was using drs
            ('DRS_max', 'max') #max drs value
        ],
        'Speed': [('max_speed', 'max')] #max speed in telemetry data with drs
    })
    
    # Flattenig the MultiIndex columns so it can be used in the merge
    per_drs_by_lap.columns = [col[1] for col in per_drs_by_lap.columns]
    per_drs_by_lap = per_drs_by_lap.reset_index()
    
    # Merging DRS features with lap data
    print("Merging DRS data with lap data...")
    per_laps_df = pd.merge(
        per_laps_df,
        per_drs_by_lap,
        on=['RoundNumber', 'LapNumber'],
        how='left'
    )
    
    # Fill missing values with 0 or False
    per_laps_df['DRS_used'] = per_laps_df['DRS_used'].fillna(False).astype(int)
    per_laps_df['DRS_usage_pct'] = per_laps_df['DRS_usage_pct'].fillna(0)
    per_laps_df['DRS_max'] = per_laps_df['DRS_max'].fillna(0)
    per_laps_df['max_speed'] = per_laps_df['max_speed'].fillna(per_laps_df['SpeedST'])
    
    # Drop rows with missing SpeedST data bc they are not useful for analysis
    per_laps_df = per_laps_df.dropna(subset=['SpeedST'])
    
    # Get unique rounds and sort them in order
    rounds = sorted(per_laps_df['RoundNumber'].unique())
    
    # Add race names for better context for later
    race_names = {}
    for round_num in rounds:
        race_info = events_df[events_df['RoundNumber'] == round_num]
        if len(race_info) > 0:
            race_names[round_num] = race_info.iloc[0].get('EventName', f"Round {round_num}")
        else:
            race_names[round_num] = f"Round {round_num}"
    
    
    print("Creating straight speed trend visualization...") #sanity check
    
    # Calculate average straight speeds by round
    speed_trends = per_laps_df.groupby('RoundNumber').agg({
        'SpeedST': ['mean', 'max', 'std'],
        'max_speed': ['mean', 'max'],
        'DRS_usage_pct': 'mean'
    }).reset_index()

    # Flatten the multi-index columns
    speed_trends.columns = ['RoundNumber', 'SpeedST_mean', 'SpeedST_max', 'SpeedST_std', 
                            'max_speed_mean', 'max_speed_max', 'DRS_usage_pct_mean']

    # Merge with race names
    speed_trends['Race'] = speed_trends['RoundNumber'].map(race_names)

    # This is mostly copilot bc matplotlib is a pain to work with
    plt.figure(figsize=(16, 12))

    # Plot 1: Average straight line speed trends
    plt.subplot(2, 1, 1)
    plt.plot(speed_trends['RoundNumber'], speed_trends['SpeedST_mean'], 'bo-', 
             linewidth=2, markersize=8, label='Avg Straight Speed')
    plt.plot(speed_trends['RoundNumber'], speed_trends['SpeedST_max'], 'ro-', 
             linewidth=2, markersize=8, label='Max Straight Speed')
    plt.plot(speed_trends['RoundNumber'], speed_trends['max_speed_mean'], 'go--', 
             linewidth=1.5, alpha=0.7, label='Avg Top Speed')

    # Add error bands for straight speeds
    plt.fill_between(
        speed_trends['RoundNumber'], 
        speed_trends['SpeedST_mean'] - speed_trends['SpeedST_std'],
        speed_trends['SpeedST_mean'] + speed_trends['SpeedST_std'],
        color='blue', alpha=0.2, label='Standard Deviation'
    )

    # Add race names as x-tick labels
    plt.xticks(speed_trends['RoundNumber'], 
               [f"R{r}\n{n[:5]}" for r, n in zip(speed_trends['RoundNumber'], speed_trends['Race'])], 
               rotation=45, ha='right')
    plt.title("Perez: Straight Speed Evolution Across Races", fontsize=14)
    plt.ylabel("Speed (km/h)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    # Plot 2: Relationship between DRS usage and straight speed
    plt.subplot(2, 1, 2)
    plt.bar(speed_trends['RoundNumber'], speed_trends['DRS_usage_pct_mean']*100, 
            alpha=0.6, label='DRS Usage %')
    
    # Add second y-axis for speed
    ax2 = plt.gca().twinx()
    ax2.plot(speed_trends['RoundNumber'], speed_trends['SpeedST_mean'], 'bo-', 
             linewidth=2, label='Avg Straight Speed')
    ax2.set_ylabel('Speed (km/h)', color='b')

    # Add race names as x-tick labels
    plt.xticks(speed_trends['RoundNumber'], 
               [f"R{r}\n{n[:5]}" for r, n in zip(speed_trends['RoundNumber'], speed_trends['Race'])], 
               rotation=45, ha='right')
    plt.title("DRS Usage vs. Straight Speed by Race", fontsize=14)
    plt.ylabel("DRS Usage (%)")
    plt.grid(True, alpha=0.3)

    # Combine legends from both axes
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('perez_straight_speed_trends.png')
    plt.show()
    
    #back to me now 
    # Storage for round-by-round results
    results_by_round = {}
    per_feature_imp_by_round = {}
    round_metrics = []
    
    # Also track all predictions for overall evaluation
    all_per_actual = []
    all_per_pred_lin = []
    all_per_pred_ridge = []
    all_per_pred_rf = []
    
    # Processing each round separately
    for round_num in rounds:
        print(f"\nProcessing {race_names[round_num]} (Round {round_num})...") #sanity check
        
        # Filtering data for this round
        per_round = per_laps_df[per_laps_df['RoundNumber'] == round_num] 
        
        # Skip if not enough data #probably where it is dropping mexico???
        if len(per_round) < 10:
            print(f"  Skipping: insufficient data (only {len(per_round)} laps)")
            continue
            
        # featrs
        per_features = per_round[['Compound_index', 'TyreLife', 'LapNumber', 
                                 'Stint', 'FreshTyre', 'DRS_used',
                                 'DRS_usage_pct', 'DRS_max', 'max_speed']].copy()
        
        # Converting bools to integers
        if per_features['FreshTyre'].dtype == 'bool':
            per_features['FreshTyre'] = per_features['FreshTyre'].astype(int)
            
        # Target variable (straight line speed)
        per_target = per_round['SpeedST']
        
        # Train-test split :)
        x_train_per, x_test_per, y_train_per, y_test_per = train_test_split(
            per_features, per_target, test_size=0.2, random_state=301)
            
        # mini lin reg for later 
        per_lin_model = LinearRegression()
        per_lin_model.fit(x_train_per, y_train_per)
        per_lin_pred = per_lin_model.predict(x_test_per)
        
        # my boi ridge
        # alpja selection I did have copilot make up alphas bc i am tired
        alphas = [.001, .01, .1, 1, 10, 100, 1000]
        ridge_per = Ridge() #ridge!
        params_per = {'alpha': alphas}
        cv_per = GridSearchCV(ridge_per, params_per, cv=5, scoring='neg_mean_squared_error') #my beloved gridsearch
        cv_per.fit(x_train_per, y_train_per)
        best_alpha_per = cv_per.best_params_['alpha'] #best performing alpha
        
        # Training with best alpha
        per_ridge_model = Ridge(alpha=best_alpha_per)
        per_ridge_model.fit(x_train_per, y_train_per)
        per_ridge_pred = per_ridge_model.predict(x_test_per)
        
        # rf model 
        per_rf_model = RandomForestRegressor(n_estimators=100, random_state=301)
        per_rf_model.fit(x_train_per, y_train_per)
        per_rf_pred = per_rf_model.predict(x_test_per)
        
        # metrics
        per_lin_mse = mean_squared_error(y_test_per, per_lin_pred)
        per_lin_r2 = r2_score(y_test_per, per_lin_pred)
        
        per_ridge_mse = mean_squared_error(y_test_per, per_ridge_pred)
        per_ridge_r2 = r2_score(y_test_per, per_ridge_pred)
        
        per_rf_mse = mean_squared_error(y_test_per, per_rf_pred)
        per_rf_r2 = r2_score(y_test_per, per_rf_pred)
        
        # feature importance
        per_feature_importance = pd.DataFrame({
            'Feature': per_features.columns,
            'Importance': per_rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Store this round's feature importance
        per_feature_imp_by_round[round_num] = per_feature_importance
        
        # Add metrics to round-by-round tracking
        round_metrics.append({
            'Round': round_num,
            'Race': race_names[round_num],
            'Linear_R2': per_lin_r2,
            'Ridge_R2': per_ridge_r2,
            'RF_R2': per_rf_r2,
            'Linear_MSE': per_lin_mse,
            'Ridge_MSE': per_ridge_mse,
            'RF_MSE': per_rf_mse,
            'Ridge_Alpha': best_alpha_per,
            'Top_Feature': per_feature_importance.iloc[0]['Feature'],
            'Top_Feature_Importance': per_feature_importance.iloc[0]['Importance'],
            'Sample_Size': len(per_round)
        })
        
        # Store results
        results_by_round[round_num] = {
            'race_name': race_names[round_num],
            'linear': {'mse': per_lin_mse, 'r2': per_lin_r2},
            'ridge': {'mse': per_ridge_mse, 'r2': per_ridge_r2, 'alpha': best_alpha_per},
            'random_forest': {'mse': per_rf_mse, 'r2': per_rf_r2},
            'feature_importance': per_feature_importance.to_dict('records'),
            'sample_size': len(per_round)
        }
        
        # Aggregate predictions for overall evaluation
        all_per_actual.extend(y_test_per)
        all_per_pred_lin.extend(per_lin_pred)
        all_per_pred_ridge.extend(per_ridge_pred)
        all_per_pred_rf.extend(per_rf_pred)
        
        # Print round results
        print(f"  {race_names[round_num]} Results:")
        print(f"  Linear R²={per_lin_r2:.4f}, Ridge R²={per_ridge_r2:.4f}, RF R²={per_rf_r2:.4f}")
        print(f"  Top Feature: {per_feature_importance.iloc[0]['Feature']} (Importance: {per_feature_importance.iloc[0]['Importance']:.4f})")
        
        #back to copilot for the plotting
        # Create round-specific plots
        plt.figure(figsize=(15, 6))
        
        # Model comparison plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_test_per, per_lin_pred, alpha=0.5, label='Linear')
        plt.scatter(y_test_per, per_ridge_pred, alpha=0.5, label='Ridge')
        plt.scatter(y_test_per, per_rf_pred, alpha=0.5, label='Random Forest')
        plt.plot([y_test_per.min(), y_test_per.max()], 
                 [y_test_per.min(), y_test_per.max()], 'k--')
        plt.title(f"Perez - {race_names[round_num]} (R{round_num})")
        plt.xlabel("Actual Speed (km/h)")
        plt.ylabel("Predicted Speed (km/h)")
        plt.legend()
        
        # Feature importance plot
        plt.subplot(1, 2, 2)
        top_n = min(8, len(per_feature_importance))  # Show top 8 features
        top_features = per_feature_importance.head(top_n)
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.title(f"Feature Importance - {race_names[round_num]}")
        plt.xlabel("Importance")
        plt.gca().invert_yaxis()  # To show most important at the top
        
        plt.tight_layout()
        plt.savefig(f'perez_round_{round_num}_{race_names[round_num].replace(" ", "_")}.png')
        plt.close()
    
    # Convert metrics to DataFrame for easy analysis
    rounds_df = pd.DataFrame(round_metrics)
    
    # RACE-BY-RACE PROGRESSION PLOTS
    # Create trend plots showing how performance and features evolve over races
    plt.figure(figsize=(15, 12))
    
    # Plot performance metrics by round
    plt.subplot(2, 1, 1)
    plt.plot(rounds_df['Round'], rounds_df['Linear_R2'], marker='o', label='Linear R²')
    plt.plot(rounds_df['Round'], rounds_df['Ridge_R2'], marker='s', label='Ridge R²')
    plt.plot(rounds_df['Round'], rounds_df['RF_R2'], marker='^', label='RF R²')
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.3, label='R²=0.8 threshold')
    
    # Add race names as x-tick labels
    plt.xticks(rounds_df['Round'], [f"R{r}\n{n[:5]}" for r, n in zip(rounds_df['Round'], rounds_df['Race'])], 
              rotation=45, ha='right')
    
    plt.title("Perez: Model Performance by Race")
    plt.ylabel("R² Score")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot top feature changes across races
    plt.subplot(2, 1, 2)
    
    # Get all unique features that were important
    all_top_features = set()
    for _, importance_df in per_feature_imp_by_round.items():
        top_3 = importance_df.head(3)['Feature'].values
        all_top_features.update(top_3)
    
    # Create matrix to store importance values
    feature_trends = pd.DataFrame(0, 
                                 index=sorted(rounds_df['Round']), 
                                 columns=sorted(all_top_features))
    
    # Fill with actual importance values
    for round_num, importance_df in per_feature_imp_by_round.items():
        for _, row in importance_df.iterrows():
            if row['Feature'] in all_top_features:
                feature_trends.loc[round_num, row['Feature']] = row['Importance']
    
    # Plot each feature's importance trend
    for feature in feature_trends.columns:
        plt.plot(feature_trends.index, feature_trends[feature], 
                marker='o', label=feature)
    
    plt.xticks(rounds_df['Round'], [f"R{r}\n{n[:5]}" for r, n in zip(rounds_df['Round'], rounds_df['Race'])], 
              rotation=45, ha='right')
    plt.title("Feature Importance Evolution Across Races")
    plt.ylabel("Importance")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('perez_race_by_race_evolution.png')
    plt.show()  # Explicitly show the plot
    
    # OVERALL METRICS
    all_per_actual = np.array(all_per_actual)
    all_per_pred_lin = np.array(all_per_pred_lin)
    all_per_pred_ridge = np.array(all_per_pred_ridge)
    all_per_pred_rf = np.array(all_per_pred_rf)
    
    # Calculate overall metrics
    per_lin_mse_overall = mean_squared_error(all_per_actual, all_per_pred_lin)
    per_lin_r2_overall = r2_score(all_per_actual, all_per_pred_lin)
    per_ridge_mse_overall = mean_squared_error(all_per_actual, all_per_pred_ridge)
    per_ridge_r2_overall = r2_score(all_per_actual, all_per_pred_ridge)
    per_rf_mse_overall = mean_squared_error(all_per_actual, all_per_pred_rf)
    per_rf_r2_overall = r2_score(all_per_actual, all_per_pred_rf)
    
    # Print race-by-race comparison
    print("\n--- RACE-BY-RACE PERFORMANCE ---")
    print(rounds_df[['Round', 'Race', 'Linear_R2', 'Ridge_R2', 'RF_R2']].to_string(index=False))
    
    print("\n--- OVERALL PERFORMANCE ---")
    print(f"LINEAR: MSE={per_lin_mse_overall:.2f}, R²={per_lin_r2_overall:.4f}")
    print(f"RIDGE: MSE={per_ridge_mse_overall:.2f}, R²={per_ridge_r2_overall:.4f}")
    print(f"RF: MSE={per_rf_mse_overall:.2f}, R²={per_rf_r2_overall:.4f}")
    
    # Return results with race-by-race focus
    return {
        'by_race': results_by_round,
        'race_progression': rounds_df.to_dict('records'),
        'feature_evolution': {
            round_num: {
                'race': race_names[round_num],
                'features': importance.to_dict('records')
            }
            for round_num, importance in per_feature_imp_by_round.items()
        },
        'overall': {
            'linear': {'mse': per_lin_mse_overall, 'r2': per_lin_r2_overall},
            'ridge': {'mse': per_ridge_mse_overall, 'r2': per_ridge_r2_overall},
            'random_forest': {'mse': per_rf_mse_overall, 'r2': per_rf_r2_overall}
        },
        'speed_trends': speed_trends.to_dict('records')
    }
    
    
def corners():
    """
    looking at checo's cornering performance throughout the season. Also any print statements through out were added by copilot for sanity checkers bc this was a mess to debug
    """
#randomforest [telemetry, laps, weather]
    try:
        #randomforest [telemetry, laps, weather]
        print("Starting corners() function...")
        per_telemetry = telemetry_df[telemetry_df['DriverNumber'] == '11'] #grabbing telemetry data for checo
        per_laps = laps_df[laps_df['DriverNumber'] == '11'] #grabbing laps data for checo
        print(f"Found {len(per_telemetry)} telemetry rows and {len(per_laps)} laps") #sanity check
        
        # the sample is done before merging with weather data bc this was the only way to get it to work and it is very slow function
        sample_size = min(50000, len(per_telemetry))
        per_telemetry_sample = per_telemetry.sample(sample_size, random_state=301)
        print(f"Using sample of {len(per_telemetry_sample)} rows for merging with weather")
            
        weather_columns = ['RoundNumber'] #round number as always is our hailmary for merging
        for col in ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']: #weather columns we need for this
            if col in weather_df.columns:
                weather_columns.append(col)
                
        # debugging check just to make sure i didn;t change my mind on columndata types when setting up the modules
        if per_telemetry_sample['RoundNumber'].dtype != weather_df['RoundNumber'].dtype:
            print(f"Converting RoundNumber types to match: {per_telemetry_sample['RoundNumber'].dtype} -> {weather_df['RoundNumber'].dtype}")
            per_telemetry_sample['RoundNumber'] = per_telemetry_sample['RoundNumber'].astype(str)
            weather_df['RoundNumber'] = weather_df['RoundNumber'].astype(str)

        
        # Merge telemetry sample with weather data for checo error handling to nail down where it is losing it
        try:
            merged_weather = pd.merge(
                per_telemetry_sample,  # Use the sample instead of full dataset bc its slow enough as is
                weather_df[weather_columns],
                on=['RoundNumber'],
                how='left'
            )
            print(f"Merged weather data shape: {merged_weather.shape}")
        except Exception as e:
            print(f"Error in weather merge: {e}")
            return None

        per_lap_times = per_laps[['RoundNumber', 'LapNumber', 'LapStartTime']].copy() #so we are not directly modifying the original laps data with how bugged this is rn

        #converting to seconds for consistent comparison so it doesnt errror more
        per_lap_times['LapStartTime_seconds'] = per_lap_times['LapStartTime'].dt.total_seconds() 

        #lil function to assign lap numbers to times in the telemetry modded to closest lap time
        #cloud computing is coming in handy here with this function
        def assign_lap(telemetry_df, lap_times_df):
            print(f"Assigning laps to {len(telemetry_df)} telemetry rows...")
            
            # list to store lap results from this assignment function
            lap_numbers = []
            
            # Group lap times by round for faster lookup to speed this up
            lap_times_by_round = {}
            for round_num in lap_times_df['RoundNumber'].unique():
                round_laps = lap_times_df[lap_times_df['RoundNumber'] == round_num].copy()
                lap_times_by_round[round_num] = round_laps
            
            # Process in batches by round number 
            for round_num in sorted(telemetry_df['RoundNumber'].unique()):
                print(f"  Processing round {round_num}...")
                round_mask = telemetry_df['RoundNumber'] == round_num #this is a mask for the round number which will be used to filter the telemetry data
                round_count = round_mask.sum()
                # this is for if it cannot find a lap time for the telemetry data 
                if round_num not in lap_times_by_round:
                    print(f"  No lap data for round {round_num}, assigning NaN")
                    lap_numbers.extend([np.nan] * round_count)
                    continue
                #
                round_telemetry = telemetry_df[round_mask] # assigning that mask to the telemetry data after the if 
                round_laps = lap_times_by_round[round_num] #grouping the laps
                
                # Processing each session time in this round
                for session_time in round_telemetry['SessionTime']:
                    try:
                        session_time_seconds = session_time.total_seconds() #this converts the session time to seconds 
                        
                        # Find closest lap
                        round_laps['time_diff'] = abs(round_laps['LapStartTime_seconds'] - session_time_seconds) #grabbing the lap start time in seconds subtracting the session time in seconds to get the difference
                        closest_lap_idx = round_laps['time_diff'].idxmin() #finding the closest lap index
                        closest_lap = round_laps.loc[closest_lap_idx] #assigning the closest lap to the lap times
                        
                        if closest_lap['time_diff'] < 120:  # 2 minute threshold 
                            lap_numbers.append(closest_lap['LapNumber']) #appending the lap number to the lap numbers list
                        else:
                            lap_numbers.append(np.nan) #nan if the lap time is too far away
                    except:
                        lap_numbers.append(np.nan) #nan if error
                
            print(f"  Lap assignment complete, {sum(~np.isnan(lap_numbers))}/{len(lap_numbers)} assigned") #so ik how many laps were assigned and sanity check
            return lap_numbers
        
        #now i assign lap numbers to telemetry data via sampling
        #Use merged_weather instead of per_telemetry bc it has the weather data and telemetry data
        telemetry_sample = merged_weather
        try:
            print("Starting lap assignment...")
            telemetry_sample['LapNumber'] = assign_lap(telemetry_sample, per_lap_times) #assigning the lap numbers
            print("Lap assignment complete")
        except Exception as e:
            print(f"Error in lap assignment: {e}")
            return None

        # getting id of rows that could not be assigned laps
        telemetry_sample = telemetry_sample.dropna(subset=['LapNumber'])
        print(f"After dropping NAs: {len(telemetry_sample)} rows")

        #no labelled corners so I gotta do some math :p
        try:
            print("Creating speed percentiles...")
            speed_percentiles = telemetry_sample.groupby(['RoundNumber', 'LapNumber'])['Speed'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack() #the percentiles for the speed data in telemetry which allow us to try and get the corners later

            telemetry_sample = pd.merge(telemetry_sample, speed_percentiles.reset_index(), on=['RoundNumber', 'LapNumber'], how='left') #adding the percentiles to the telemetry sample
            print("Speed percentiles created and merged")
        except Exception as e:
            print(f"Error creating speed percentiles: {e}")
            return None

        #low speed segments are probably corners so we have to find them
        print("Identifying low-speed segments...") #sanity check
        telemetry_sample['IsLowSpeed'] = telemetry_sample['Speed'] <= telemetry_sample[0.25] #this is the low speed threshold for corners we cant exactly find high speed corners so we have focus on low speed ones

        telemetry_sample['SegmentChange'] = telemetry_sample['IsLowSpeed'].diff().fillna(0) != 0 #this is creating segment change for the telemetry data
        telemetry_sample['SegmentID'] = telemetry_sample['SegmentChange'].cumsum() #this is creating the segment id for the telemetry data the cumsum function is used to create a unique id for each segment

        low_speed_segments = telemetry_sample[telemetry_sample['IsLowSpeed']] #this is filtering the telemetry data for low speed segments
        print(f"Found {len(low_speed_segments)} low-speed segment points") #sanity check and also to see how many low speed segments we have for each track
        
        # Print first 10 columns to avoid overwhelming output
        print(f"First 10 columns: {low_speed_segments.columns.tolist()[:10]}")

        print("Calculating segment features...")
        try:

            # Convert boolean Brake column to integers (0=False, 1=True) so i can try to get some kind of brake data
            low_speed_segments['Brake_numeric'] = low_speed_segments['Brake'].astype(int)
            #this aggregating the telemetry data for low speed segments grouping by round number, lap number, and segment id 
            segment_features = low_speed_segments.groupby(['RoundNumber', 'LapNumber', 'SegmentID']).agg({ 
                'Speed': ['min', 'mean', 'std'],
                'Throttle': ['min', 'mean', 'max'],
                'Brake_numeric': ['mean', 'max'],
                'nGear': ['min', 'max', 'median'] if 'nGear' in low_speed_segments.columns else ['min'],
                'RPM': ['mean', 'max']
            })
        except Exception as e:
            print(f"Error calculating segment features: {e}")
            return None

        #flattening the multiindex columns so we can work with them w o errors
        segment_features.columns = ['_'.join(col).strip() for col in segment_features.columns.values]
        segment_features = segment_features.reset_index()
        print(f"Created {len(segment_features)} segment features") #sanity check

        # Using SessionTime for duration calculations to see how long the segments are
        if 'SessionTime' in low_speed_segments.columns: 
            print("Using SessionTime for segment duration calculations") #sanity check
            # grouping by round number, lap number, and segment id to get the max and min session time for each segment
            segment_duration = low_speed_segments.groupby(['RoundNumber', 'LapNumber', 'SegmentID']).agg({
                'SessionTime': ['max', 'min']
            })

            # Flattening the multiindex columns
            segment_duration.columns = ['_'.join(col).strip() for col in segment_duration.columns.values]
            segment_duration = segment_duration.reset_index()
            
            # Calculate duration only so if the session time is not in the data it will not error out
            segment_duration['Segment_Duration'] = segment_duration['SessionTime_max'] - segment_duration['SessionTime_min']
            
            # Merging duration with segment features
            segment_features = pd.merge(
                segment_features,
                segment_duration[['RoundNumber', 'LapNumber', 'SegmentID', 'Segment_Duration']],
                on=['RoundNumber', 'LapNumber', 'SegmentID'],
                how='left'
            )
            
            # (Speed in km/h, Duration in seconds)
            # Distance (m) = Speed (km/h) * time (s) / 3.6
            segment_features['Segment_Length'] = segment_features['Speed_mean'] * segment_features['Segment_Duration'].dt.total_seconds() / 3.6
            print("Estimated segment length using speed and duration") #sanity check
            
        else: #copilot did this in error handling but it works so i am not touching it
            print("WARNING: SessionTime column not found. Creating proxy metrics.") 
            # Create alternative segment metrics
            segment_count = low_speed_segments.groupby(['RoundNumber', 'LapNumber', 'SegmentID']).size().reset_index(name='Segment_Points')
            
            segment_features = pd.merge(
                segment_features,
                segment_count[['RoundNumber', 'LapNumber', 'SegmentID', 'Segment_Points']],
                on=['RoundNumber', 'LapNumber', 'SegmentID'],
                how='left'
            )
            
            # Add placeholder values
            segment_features['Segment_Duration'] = pd.to_timedelta(segment_features['Segment_Points'] * 0.1, unit='s')
            segment_features['Segment_Length'] = segment_features['Speed_mean'] * segment_features['Segment_Points'] * 0.1 / 3.6

        # Getting average weather conditions for each corner
        # FIXED: Now safely checks if columns are strings before applying 'in' operator <-copilot fixed this
        #copilot added alot of error handling to nail down where it was not working and silently failing
        print("Checking for weather columns...")
        string_columns = [col for col in telemetry_sample.columns if isinstance(col, str)] #this is checking for string columns in the telemetry data in case i oopsied bc i am a silly goosy
        weather_cols = [col for col in string_columns if 'Temp' in col or col in ['Humidity', 'WindSpeed']] 
        print(f"Found weather columns: {weather_cols}")
        
        if 'AirTemp' in string_columns:
            print("Weather data found in telemetry_sample")
            weather_agg = {}
            for col in ['TrackTemp', 'AirTemp', 'Humidity', 'WindSpeed']:
                if col in low_speed_segments.columns:
                    weather_agg[col] = 'mean'
                    
            if weather_agg:
                segment_weather = low_speed_segments.groupby(['RoundNumber', 'LapNumber', 'SegmentID']).agg(weather_agg).reset_index()
                
                # Mergeing with la segment features
                segment_features = pd.merge(
                    segment_features,
                    segment_weather,
                    on=['RoundNumber', 'LapNumber', 'SegmentID'],
                    how='left'
                )
            else:
                print("Warning: No weather columns found to aggregate")
        else:
            print("Warning: Weather data not available in telemetry_sample")

        #marrying the lap data and tyre info
        print("Adding tire information...")
        segment_features = pd.merge(
            segment_features,
            per_laps[['RoundNumber', 'LapNumber', 'TyreLife', 'Compound', 'Stint', 'FreshTyre']],
            on = ['RoundNumber', 'LapNumber'],
            how = 'left'
        )
        #the label encoding never worked properly so I just made a mapping cloud computing coming in handy again
        compound_mapping = {
            'SOFT': 1,
            'MEDIUM': 2,
            'HARD': 3,
            'INTERMEDIATE': 4,
            'WET': 5
        }
        #love map reduce, but this is only mappig at this point
        segment_features['Compound_index'] = segment_features['Compound'].map(compound_mapping)

        #in case there are still nans left in this house
        segment_features = segment_features.dropna()
        print(f"After removing NAs: {len(segment_features)} segment features")

        # Calculating additional metrics 
        segment_features['Speed_to_Distance_Ratio'] = segment_features['Speed_mean'] / segment_features['Segment_Length'] #the ratio here is important so we can get a better idea of how fast the car is going in relation to the distance dude is sending it
        segment_features['Braking_Intensity'] = segment_features['Brake_max'] * segment_features['Segment_Duration'].dt.total_seconds() #how hard we think he is braking in the segment based on this
        
        # Create performance category for classification (based on minimum speed)
        # FIXED: Added error handling for qcut operation
        try:
            print("Creating speed percentiles...")
            segment_features['Speed_Percentile'] = segment_features.groupby('RoundNumber')['Speed_min'].transform(
                lambda x: pd.qcut(x, 3, labels=False, duplicates='drop') 
                if len(x) > 3 and len(x.unique()) >= 3 
                else pd.Series([1] * len(x), index=x.index)
            )
        except Exception as e:
            print(f"Error in qcut: {e}, using alternate approach")
            # Fallback to a simple cut if qcut fails (equal-width bins)
            segment_features['Speed_Percentile'] = segment_features.groupby('RoundNumber')['Speed_min'].transform(
                lambda x: pd.cut(x, 3, labels=False) 
                if len(x) > 3 
                else pd.Series([1] * len(x), index=x.index)
            )
        
        # Fix any NaN values that might have been produced
        segment_features['Speed_Percentile'] = segment_features['Speed_Percentile'].fillna(1).astype(int)
        
        segment_features['Performance'] = segment_features['Speed_Percentile'].map(
            {0: 'Slow', 1: 'Medium', 2: 'Fast'}
        )

        #prepping the homemade features for model
        print("Preparing model features...")
        features = segment_features[[
            'Speed_min', 'Speed_mean', 'Speed_std',
            'Throttle_min', 'Throttle_mean', 'Throttle_max',
            'Brake_mean', 'Brake_max',
            'Segment_Length', 
            'Compound_index', 'TyreLife', 'Stint'
        ]].copy()

        # Add calculated feature for duration in seconds
        features['Segment_Duration'] = segment_features['Segment_Duration'].dt.total_seconds()

        #nGears finallly!!!
        if 'nGear_min' in segment_features.columns:
            features['nGear_min'] = segment_features['nGear_min']
            features['nGear_max'] = segment_features['nGear_max']

        # addign the weather features
        for col in ['TrackTemp', 'AirTemp', 'Humidity']:
            if col in segment_features.columns:
                features[col] = segment_features[col]

        # Making sure FreshTyre is numeric bc im cookied
        if 'FreshTyre' in segment_features.columns:
            if segment_features['FreshTyre'].dtype == 'bool':
                features['FreshTyre'] = segment_features['FreshTyre'].astype(int)
            else:
                features['FreshTyre'] = segment_features['FreshTyre']

        #finally the target variable 
        target = segment_features['Performance']
        print(f"Target variable distribution: {target.value_counts().to_dict()}")

        # Train-test split time
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=301, stratify=target)
        print(f"Training set: {x_train.shape}, Test set: {x_test.shape}")

        #finally rhe random forest model good lord and I have ran out of interesting variable names and comments
        print("Training Random Forest model...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=301)
        rf_model.fit(x_train, y_train)

        #evaluating the model
        print("Evaluating model...")
        rf_pred = rf_model.predict(x_test)
        
        # Use confusion matrix instead of accuracy_score
        cm = confusion_matrix(y_test, rf_pred)
        accuracy = cm.diagonal().sum() / cm.sum()
        print(f"Model accuracy: {accuracy:.4f}")

        feature_importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        #top 10 most important features
        top_features = feature_importance.head(10)
        print("Top 10 Features:", top_features) 

        #My one true op matplotlib copilot is doing the heavy lifting here with matplotlib but I did try
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Feature importance
        plt.subplot(2, 2, 1)
        top_features = feature_importance.head(10)
        plt.barh(
            np.arange(len(top_features)), 
            top_features['Importance'],
            color='skyblue'
        )
        plt.yticks(np.arange(len(top_features)), top_features['Feature'])
        plt.title('Top Features for Low-Speed Segment Performance', fontsize=14)
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()  # Most important at the top
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Plot 2: Low-speed segment performance by tire compound
        plt.subplot(2, 2, 2)
        compound_performance = pd.crosstab(
            segment_features['Compound'], 
            segment_features['Performance'],
            normalize='index'
        ) * 100
        
        compound_performance.plot(
            kind='bar', 
            stacked=False,
            ax=plt.gca()
        )
        
        plt.title('Low-Speed Segment Performance by Tire Compound', fontsize=14)
        plt.xlabel('Tire Compound')
        plt.ylabel('Percentage of Segments')
        plt.legend(title='Performance')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Plot 3: Speed vs. Throttle colored by performance
        plt.subplot(2, 2, 3)
        performance_colors = {'Slow': 'red', 'Medium': 'gold', 'Fast': 'green'}
        
        for perf, color in performance_colors.items():
            mask = segment_features['Performance'] == perf
            plt.scatter(
                segment_features.loc[mask, 'Speed_min'],
                segment_features.loc[mask, 'Throttle_mean'],
                c=color,
                alpha=0.6,
                label=perf
            )
        
        plt.title('Minimum Speed vs. Throttle Application', fontsize=14)
        plt.xlabel('Minimum Speed (km/h)')
        plt.ylabel('Mean Throttle (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Performance distribution by track
        plt.subplot(2, 2, 4)
        track_performance = pd.crosstab(
            segment_features['RoundNumber'], 
            segment_features['Performance'],
            normalize='index'
        ) * 100
        
        # Get race names for better labels
        race_names = {}
        for round_num in track_performance.index:
            race_info = events_df[events_df['RoundNumber'] == round_num]
            if len(race_info) > 0:
                race_names[round_num] = race_info.iloc[0].get('EventName', f"Round {round_num}")
            else:
                race_names[round_num] = f"Round {round_num}"
        
        # Update index with race names
        track_performance.index = [f"R{r}-{race_names[r][:5]}" for r in track_performance.index]
        
        track_performance.plot(
            kind='bar',
            stacked=True,
            colormap='RdYlGn',
            ax=plt.gca()
        )
        
        plt.title('Low-Speed Segment Performance by Track', fontsize=14)
        plt.xlabel('Race')
        plt.ylabel('Percentage of Segments')
        plt.legend(title='Performance')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('perez_low_speed_segment_analysis.png')
        plt.show()
        
        # 12. Additional analysis: Speed by segment length
        plt.figure(figsize=(12, 8))
        
        # Create bins for segment length
        segment_features['Length_Category'] = pd.cut(
            segment_features['Segment_Length'],
            bins=[0, 50, 100, 200, 500, np.inf],
            labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
        )
        
        # Plot speed distribution by segment length category
        sns.boxplot(
            x='Length_Category',
            y='Speed_min',
            data=segment_features,
            palette='viridis'
        )
        
        plt.title('Minimum Speed by Low-Speed Segment Length', fontsize=14)
        plt.xlabel('Segment Length Category (m)')
        plt.ylabel('Minimum Speed (km/h)')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('perez_segment_length_speed.png')
        plt.show()
        
        # Return analysis results
        return {
            'model_accuracy': accuracy,
            'feature_importance': feature_importance.to_dict('records'),
            'segments_analyzed': len(segment_features),
            'performance_by_track': track_performance.to_dict(),
            'performance_by_compound': compound_performance.to_dict()
        }

    except Exception as e:
        print(f"Error in corners() function: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

    


if __name__ == "__main__":
    #print(straights_speed())
    #print(tyre_deg())

    results = corners()
    if results:
        print("Function completed successfully")
    else:
        print("Function returned None (failed)")
