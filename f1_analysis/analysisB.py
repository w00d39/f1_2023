import sys, os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# How has McLaren improved throughout the year?

from library_data_2023 import fastestlaps, results, laps, telemetry, events

def mclaren_linreg():
    """
    This function analyzes the performance of the McLaren team in the 2023 Formula 1 season.
   Focusing on linear regression analysis, it retrieves data from results, laps, telemetry, and events.
    """
    #home made modules we are using

    fastestlaps_df = fastestlaps.open_fastestlaps_data()
    results_df = results.open_results_data()
    laps_df = laps.open_laps_data()
    telemetry_df = telemetry.open_telemetry_data()
    events_df = events.open_events_data()

   #Filter for Mclaren, papaya is easier to type than McLaren
    papaya_df = results_df[results_df['TeamName'] == 'McLaren']

    #rounds but i need them in order so we can go proper order for progression
    rounds = events_df.sort_values('EventDate')['RoundNumber'].unique()

    #df for storing agg race papaya performance
    papaya_performance = []

    for i in rounds:
        #event deets
        event_deets = events_df[events_df['RoundNumber'] == i].iloc[0]

        #papaya results for this round
        round_results = papaya_df[papaya_df['RoundNumber'] == i]

        if round_results.empty: #error handling :)
            continue

        #fastest lap for this round
        fastest_lap = fastestlaps_df[fastestlaps_df['RoundNumber'] == i]

        if len(fastest_lap) > 0: #fastest lap exists 
            fastest_lap_time = fastest_lap['LapTime'].min()

            papaya_fastest_lap = fastest_lap[fastest_lap['Team'] == 'McLaren'] #fastest lap is for the papaya team

            if len(papaya_fastest_lap) > 0: #if the papaya team has a fastest lap
                papaya_fastest_lap_time = papaya_fastest_lap['LapTime'].min() #get the lap time for the papaya team
                gap_to_fastest = (papaya_fastest_lap_time - fastest_lap_time).total_seconds() #convert to total seconds
            else:
                gap_to_fastest = None #if papaya team has no fastest lap
        else:
            gap_to_fastest = None #if no fastest lap exists

        #avg positions and points
        avg_positions = round_results['Position'].mean()
        avg_grid = round_results['GridPosition'].mean()
        total_points = round_results['Points'].sum()

    #storing our calculated metrics
        papaya_performance.append({
            'RoundNumber': i,
            'EventName': event_deets['EventName'],
            'EventDate': event_deets['EventDate'],
            'AvgPosition': avg_positions,
            'AvgGridPosition': avg_grid,
            'TotalPoints': total_points,
            'GapToFastest': gap_to_fastest
        })

    # convert to DataFrame
    mclaren_df = pd.DataFrame(papaya_performance)


    #this is is the x index for lin reg
    mclaren_df['RaceIndex'] = range(1, len(mclaren_df) + 1)

    #lin reg time >:D
    x = mclaren_df[['RaceIndex']]

    reg_results = {}

    for metric in ['AvgPosition', 'AvgGridPosition', 'TotalPoints', 'GapToFastest']:
        #mets w missing data 
        if mclaren_df[metric].isna().sum() > 0:
            y = mclaren_df.dropna(subset=[metric])[metric]
            x_clean = mclaren_df.dropna(subset = [metric])[['RaceIndex']]
        else:
            y = mclaren_df[metric]
            x_clean = x
        
        if len(y) >= 3: #if we have enough data points
            # Create and fit the model
            papaya_model = LinearRegression() # lin reg
            papaya_model.fit(x_clean, y)

            reg_results[metric] = { #metrics
                'coefficient': papaya_model.coef_[0],
                'intercept': papaya_model.intercept_,
                'r_squared': papaya_model.score(x_clean, y)
            }

    return {
        'mclaren_df': mclaren_df,
        'regression_results': reg_results
    }

def mclaren_linreg_plots(mclaren_data, regression_results):
    """
    This function generates plots to visualize the performance of the McLaren team in the 2023 Formula 1 season.
    """
    plt.figure(figsize=(14, 12))
    
    # Plot 1: Race Positions (lower is better)
    plt.subplot(2, 2, 1)
    plt.scatter(mclaren_data['RaceIndex'], mclaren_data['AvgPosition'], color='#FF8000', s=100, label='Race Results')

     #race names as annotations
    for i, row in mclaren_data.iterrows():
        plt.annotate(row['EventName'].split(' Grand')[0], 
                    (row['RaceIndex'], row['AvgPosition']),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=8,
                    rotation=45)
        
    #reg line
    if 'AvgPosition' in regression_results:
        coef = regression_results['AvgPosition']['coefficient']
        intercept = regression_results['AvgPosition']['intercept']
        x_range = range(1, len(mclaren_data) + 1)
        plt.plot(x_range, [intercept + coef * x for x in x_range], 'b--', 
                    label=f'Trend: {coef:.3f} pos/race (R²: {regression_results["AvgPosition"]["r_squared"]:.2f})')
        plt.legend()

    plt.title('McLaren Average Race Position')
    plt.xlabel('Race Number')
    plt.ylabel('Position (lower is better)')
    plt.gca().invert_yaxis()  # Invert y-axis so better positions are higher on the plot
    plt.grid(True, alpha=0.3)

    #plot 2: Grid Positions >:p

    plt.subplot(2, 2, 2)

    plt.scatter(mclaren_data['RaceIndex'], mclaren_data['AvgGridPosition'], color='#FF8000', s=100, label='Grid Positions')

    #reg line
    if 'AvgGridPosition' in regression_results:
        coef = regression_results['AvgGridPosition']['coefficient']
        intercept = regression_results['AvgGridPosition']['intercept']
        x_range = range(1, len(mclaren_data) + 1)
        plt.plot(x_range, [intercept + coef * x for x in x_range], 'b--', 
                 label=f'Trend: {coef:.3f} pos/race (R²: {regression_results["AvgGridPosition"]["r_squared"]:.2f})')
        plt.legend()

    plt.title('McLaren Average Grid Position')
    plt.xlabel('Race Number')
    plt.ylabel('Grid Position (lower is better)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)

    #plot 3: Total Points
    plt.subplot(2, 2, 3)
    plt.scatter(mclaren_data['RaceIndex'], mclaren_data['TotalPoints'], color='#FF8000', s=100, label='Total Points')

    #reg line
    if 'TotalPoints' in regression_results:
        coef = regression_results['TotalPoints']['coefficient']
        intercept = regression_results['TotalPoints']['intercept']
        x_range = range(1, len(mclaren_data) + 1)
        plt.plot(x_range, [intercept + coef * x for x in x_range], 'b--', 
                 label=f'Trend: {coef:.3f} pts/race (R²: {regression_results["TotalPoints"]["r_squared"]:.2f})')
        plt.legend()

    plt.title('McLaren Points per Race')
    plt.xlabel('Race Number')
    plt.ylabel('Points')
    plt.grid(True, alpha=0.3)  



    plt.tight_layout()
    plt.suptitle('McLaren 2023 Season Progression', fontsize=16, y=0.98)
    plt.savefig('mclaren_progression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nMcLaren 2023 Season Progression Summary:")
    for metric, results in regression_results.items():
        trend = "improving" if (metric in ['AvgPosition', 'AvgGridPosition', 'GapToFastest'] and results['coefficient'] < 0) or \
                              (metric == 'TotalPoints' and results['coefficient'] > 0) else "declining"
        print(f"- {metric}: {trend} at a rate of {abs(results['coefficient']):.3f} per race (R²: {results['r_squared']:.2f})")

    """
    This function analyzes the performance of the McLaren team in the 2023 Formula 1 season.
    Focusing on random forest regression analysis, it retrieves data from results, laps, telemetry, and events.
    """

    results_df = results.open_results_data()
    laps_df = laps.open_laps_data()
    telemetry_df = telemetry.open_telemetry_data()
    events_df = events.open_events_data()

    return results_df, laps_df, telemetry_df, events_df


if __name__ == "__main__":
    # Get analysis results
    analysis_results = mclaren_linreg()
    mclaren_df = analysis_results['mclaren_df']
    regression_results = analysis_results['regression_results']
    
    # Display basic information
    print("McLaren DataFrame:")
    print(mclaren_df.head())
    print("\nRegression Results:")
    for metric, results in regression_results.items():
        print(f"{metric}: {results}")
    
    # Generate plots
    mclaren_linreg_plots(mclaren_df, regression_results)
 
