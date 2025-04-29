import sys, os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# What kind of weather impacted lap times?

from library_data_2023 import weather, laps, sprintlaps

def preprocess_weather():
    weather_df = weather.open_weather_data()
    laps_df = laps.open_laps_data()
    sprintlaps_df = sprintlaps.open_sprintlaps_data()

    #prepping and combining the laps and sprint laps data
    sprintlaps_df['IsSprintLap'] = True
    laps_df['IsSprintLap'] = False

    #coomon columns to concat
    common_columns = list(set(laps_df.columns).intersection(sprintlaps_df.columns))
    all_laps_df = pd.concat([laps_df[common_columns], sprintlaps_df[common_columns]])

    #mergin the  weather and laps data
    merged_df = pd.merge(all_laps_df, weather_df, on=['RoundNumber', 'EventName'], how='inner', suffixes=('_lap', '_weather'))

     #closest weather reading for each lap
    merged_df['TimeDiff'] = abs((merged_df['Time_lap'] - merged_df['Time_weather']).dt.total_seconds())
    merged_df = (merged_df.sort_values('TimeDiff').groupby(['RoundNumber', 'EventName', 'DriverNumber', 'LapNumber']).first().reset_index())

    #converting lap time to seconds to make this easier
    merged_df['LapTimeSeconds'] = merged_df['LapTime'].dt.total_seconds()

    #getting rid of any missing data
    merged_df = merged_df.dropna(subset=['LapTimeSeconds', 'AirTemp', 'TrackTemp', 'Humidity'])

    #filtering outlier lap times such as pit stops, yellows, safety cars
    track_medians = merged_df.groupby('EventName')['LapTimeSeconds'].median()
    track_sds = merged_df.groupby('EventName')['LapTimeSeconds'].std()

    valid_laps = []

    for _, i in merged_df.iterrows(): #for each lap in merged_df we are going tocheck if its a valid lap 
        track = i['EventName'] #track name
        time = i['LapTimeSeconds'] #lap time in seconds

        #laps need to be within 3 standard deviations of the median lap time
        if time <= track_medians[track] + 3 * track_sds[track]: 
            valid_laps.append(True)
        else:
            valid_laps.append(False)

    merged_df['ValidLap'] = valid_laps #adds the valid lap column to the merged_df
    merged_df = merged_df[merged_df['ValidLap']]

    return weather_df, laps_df, sprintlaps_df, merged_df


def feature_forging(df):
    #in case I make an oopsy being a silly goosy
    enhanced_df = df.copy()
    #weather interactions as features
    enhanced_df['TempHumidityInteraction'] = enhanced_df['AirTemp'] * enhanced_df['Humidity'] / 100 #humidity * airtemp / 100
    enhanced_df['TrackAirTempDelta'] = enhanced_df['TrackTemp'] - enhanced_df['AirTemp'] #tracktemp - air temp

    #wind direction being circluar if i can
    if 'WindDirection' in enhanced_df.columns:
        enhanced_df['WindDirSin'] = np.sin(enhanced_df['WindDirection'] * np.pi/180) #wind sin direction in radians
        enhanced_df['WindDirCos'] = np.cos(enhanced_df['WindDirection'] * np.pi/180) #wind cos direction in radians

    #one must simply account for rain impact as a feature
    if 'Rainfall' in enhanced_df.columns: # if rainfall exists
        enhanced_df['RainfallPresent'] = (enhanced_df['Rainfall'] > 0).astype(int) # #rainfall present as a binary feature

    # Create track-specific features based on general characteristics
    high_speed_tracks = ['Monza', 'Spa', 'Azerbaijan', 'Jeddah', 'Las Vegas']
    high_downforce_tracks = ['Monaco', 'Hungary', 'Singapore', 'Zandvoort']
    
    enhanced_df['IsHighSpeedTrack'] = enhanced_df['EventName'].apply( 
        lambda x: any(track in x for track in high_speed_tracks)).astype(int)
    enhanced_df['IsHighDownforceTrack'] = enhanced_df['EventName'].apply(
        lambda x: any(track in x for track in high_downforce_tracks)).astype(int)
    
     # tire compound feature
    if 'Compound' in enhanced_df.columns:
        # Assign numeric values to compounds based on hardness
        compound_hardness = {'Hard': 1, 'Medium': 2, 'Soft': 3, 'Intermediate': 4, 'Wet': 5}
        enhanced_df['CompoundHardness'] = enhanced_df['Compound'].map(
            lambda x: compound_hardness.get(x, 0))

    #  compound x temperature interaction
        enhanced_df['CompoundTempInteraction'] = enhanced_df['CompoundHardness'] * enhanced_df['TrackTemp']

    #tyre life effects 
    if 'TyreLife' in enhanced_df.columns:
        enhanced_df['TireAgeFactor'] = np.log1p(enhanced_df['TyreLife']) 

         #tire age x temperature interaction
        if 'Compound' in enhanced_df.columns:
            enhanced_df['TireAgeCompoundEffect'] = enhanced_df['TireAgeFactor'] * enhanced_df['CompoundHardness']
        
    #categorizing lap times by track
    print("Categorizing lap times by track...") #comment this out later just a check for sanity
    enhanced_df['LapTimeCategory'] = enhanced_df.groupby('EventName').apply(
        lambda x: pd.qcut(x['LapTimeSeconds'], q=3, labels=['Fast', 'Medium', 'Slow'])
    ).reset_index(level=0, drop=True)

    return enhanced_df


def weather_gb(df, tuning = False):
    print("Building weather impact model...") #comment later sanity checker

    #features weather data
    weather_features = ['AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']
    #forged features = homemade features
    forged_features = ['TempHumidityInteraction', 'TrackAirTempDelta', 'WindDirSin', 'WindDirCos', 'RainfallPresent', 'IsHighSpeedTrack', 'IsHighDownforceTrack']

    tyre_features = []
    tyre_features.append('Compound')
        
    if 'CompoundHardness' in df.columns:
        tyre_features.extend(['CompoundHardness', 'CompoundTempInteraction'])
        
    if 'TyreLife' in df.columns:
        tyre_features.extend(['TireAgeFactor', 'TireAgeCompoundEffect'])


    #Make one huge feature stew
    all_features = weather_features + forged_features + tyre_features

    # Filter to features actually available in the dataset
    features = [f for f in all_features if f in df.columns]
    print(f"Using {len(features)} features: {', '.join(features)}") #sanity checker

    #prep
    X = pd.get_dummies(df[features], drop_first=True)
    y = df['LapTimeCategory']


    #one can never have too many missing value checkers
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

     # Scale numerical features for better performance
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 301, stratify = y)


    if tuning: #rip computer if so
        print("Performing hyperparameter tuning (this may take a while)...")
        # Hyperparameter tuning with grid search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=301),
            param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring='accuracy',
            verbose=1,
            n_jobs=-1  # Use all CPUs
        )


        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        
        print(f"Best parameters: {best_params}")
        model = grid_search.best_estimator_
        
    else:
        # Using optimized parameters directly
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=301
        )
        model.fit(X_train, y_train)

      # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Test an alternative model - Gradient Boosting
    print("Testing Gradient Boosting model for comparison...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=301
    )
    
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)   

    print(f"Gradient Boosting accuracy: {gb_accuracy:.4f}")
    
    # Choose better model
    if gb_accuracy > accuracy:
        print("Using Gradient Boosting model (better performance)")
        final_model = gb_model
        final_accuracy = gb_accuracy
        y_pred = gb_pred
        
        # Get GB feature importance
        gb_feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': gb_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        feature_importance = gb_feature_importance
    else:
        print("Using Random Forest model (better performance)")
        final_model = model
        final_accuracy = accuracy
    
    results = {
        'model': final_model,
        'accuracy': final_accuracy,
        'feature_importance': feature_importance,
        'merged_data': df,
        'x_train': X_train,
        'x_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return results

def plot_weather_analysis(results):
    """Create comprehensive visualizations of weather impact on lap times"""
    print("Generating weather impact visualizations...")
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(14, 8))
    top_features = results['feature_importance'].head(15)
    sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
    plt.title('Weather Features Impact on Lap Time Classification', fontsize=16)
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig('weather_feature_importance.png', dpi=300)
    plt.show()
    
    # 2. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = results['confusion_matrix']
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Fast', 'Medium', 'Slow'],
        yticklabels=['Fast', 'Medium', 'Slow']
    )
    plt.title('Model Performance: Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Lap Category', fontsize=14)
    plt.ylabel('Actual Lap Category', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
    
    # 3. Temperature Effects
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Air Temperature
    sns.boxplot(
        x='LapTimeCategory', y='AirTemp', 
        data=results['merged_data'],
        palette='viridis', order=['Fast', 'Medium', 'Slow'],
        ax=axes[0]
    )
    axes[0].set_title('Air Temperature Effect on Lap Times', fontsize=14)
    axes[0].set_xlabel('Lap Time Category', fontsize=12)
    axes[0].set_ylabel('Air Temperature (°C)', fontsize=12)
    
    # Track Temperature
    sns.boxplot(
        x='LapTimeCategory', y='TrackTemp', 
        data=results['merged_data'],
        palette='viridis', order=['Fast', 'Medium', 'Slow'],
        ax=axes[1]
    )
    axes[1].set_title('Track Temperature Effect on Lap Times', fontsize=14)
    axes[1].set_xlabel('Lap Time Category', fontsize=12)
    axes[1].set_ylabel('Track Temperature (°C)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('temperature_effects.png', dpi=300)
    plt.show()
    
    # 4. Rainfall Effect if available
    if 'Rainfall' in results['merged_data'].columns:
        plt.figure(figsize=(10, 6))
        # Group by presence of rain (binary)
        results['merged_data']['RainPresent'] = (results['merged_data']['Rainfall'] > 0).astype(str)
        rain_counts = results['merged_data'].groupby(['RainPresent', 'LapTimeCategory']).size().reset_index(name='Count')
        
        # Create a percentage-based plot
        rain_pcts = rain_counts.groupby('RainPresent').apply(
            lambda x: 100 * x['Count'] / x['Count'].sum()
        ).reset_index(name='Percentage')
        
        sns.barplot(
            x='RainPresent', y='Percentage', hue='LapTimeCategory',
            data=rain_pcts, palette='viridis', order=['False', 'True']
        )
        plt.title('Effect of Rainfall on Lap Time Categories', fontsize=16)
        plt.xlabel('Rain Present', fontsize=14)
        plt.ylabel('Percentage of Laps', fontsize=14)
        plt.tight_layout()
        plt.savefig('rainfall_effect.png', dpi=300)
        plt.show()
    
    # 5. Tire Compound Effect if available
    if 'Compound' in results['merged_data'].columns:
        plt.figure(figsize=(12, 6))
        
        # Remove compounds with very few data points
        compound_counts = results['merged_data']['Compound'].value_counts()
        valid_compounds = compound_counts[compound_counts > 10].index
        compound_df = results['merged_data'][results['merged_data']['Compound'].isin(valid_compounds)]
        
        sns.boxplot(
            x='Compound', y='LapTimeSeconds', hue='LapTimeCategory',
            data=compound_df, palette='viridis'
        )
        plt.title('Lap Time by Tire Compound', fontsize=16)
        plt.xlabel('Tire Compound', fontsize=14)
        plt.ylabel('Lap Time (seconds)', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('compound_effect.png', dpi=300)
        plt.show()

def weather_rfc(tuning=False):
    """Main function to analyze weather impact on F1 lap times"""
    # Load and preprocess data
    weather_df, laps_df, sprintlaps_df, merged_df = preprocess_weather()
    
    # Engineer features
    enhanced_df = feature_forging(merged_df)
    
    # Build model
    results = weather_gb(enhanced_df, tuning)
    
    return weather_df, laps_df, sprintlaps_df, results

if __name__ == "__main__":
    # Set to True for hyperparameter tuning (slower but potentially better results)
    perform_tuning = False
    
    # Run the analysis
    weather_df, laps_df, sprintlaps_df, results = weather_rfc(tuning=perform_tuning)
    
    # Print model performance
    print("\n" + "="*50)
    print("WEATHER IMPACT ON LAP TIMES - MODEL RESULTS")
    print("="*50)
    print(f"Model accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Print top weather features
    print("\nTop Weather Features Impacting Lap Times:")
    for i, (feature, importance) in enumerate(zip(
        results['feature_importance']['Feature'].head(10),
        results['feature_importance']['Importance'].head(10))):
        print(f"{i+1}. {feature}: {importance:.4f}")
    plt.figure(figsize=(12, 8))

    #top 10 features its looking at
    top_features = results['feature_importance'].head(10)
    sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
    plt.title('Weather Feautres Impact on Lap Time Classification', fontsize = 16)
    plt.xlabel('Importance', fontsize = 14)
    plt.ylabel('Feature', fontsize = 14)
    plt.tight_layout()
    plt.savefig('weather_features_importance.png', dpi = 300)
    plt.show()

    #affects of temps on lap times
    plt.figure(figsize = (10, 6))
    sns.boxplot(
        x = 'LapTimeCategory',
        y ='TrackTemp',
        data = results['merged_data'],
        palette = 'viridis',
        order = ['Fast', 'Medium', 'Slow']
    )

    plt.title('Track Temp vs Lap Time Category', fontsize = 16)
    plt.xlabel('Lap Time Category', fontsize = 14)
    plt.ylabel('Track Temp (°C)', fontsize = 14)
    plt.tight_layout()
    plt.savefig('track_temp_effect.png', dpi = 300)
    plt.show()
