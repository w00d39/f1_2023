import sys, os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from library_data_2023 import laps, sprintlaps, events

def highest_tyre_deg():
    """
    Utilizes laps and sprintlaps modules to analyze tracks with the highest tyre degradation. 
    The model will be K Means clustering.
    """

    laps_df = laps.open_laps_data() #opens the laps data
    sprintlaps_df = sprintlaps.open_sprintlaps_data() #opens the sprintlaps data
    events_df = events.open_events_data() #opens the events data
  
    track_names = events_df[['RoundNumber', 'OfficialEventName']].drop_duplicates() # Get unique track names w/o dupes

     #features i need for this beautiful and hopefully sensical analysis
    laps_df = laps_df[['RoundNumber', 'EventName', 'Compound', 'Compound_index', 'TyreLife', 'LapTime', 'Stint']]
    sprintlaps_df = sprintlaps_df[['RoundNumber', 'EventName', 'Compound', 'Compound_index', 'TyreLife', 'LapTime', 'Stint']]

    #Converting the laptimes to seconds so i can use them in the analysis
    laps_df['LapTime'] = pd.to_timedelta(laps_df['LapTime']).dt.total_seconds() #double check this later?
    sprintlaps_df['LapTime'] = pd.to_timedelta(sprintlaps_df['LapTime']).dt.total_seconds()
    # Combine laps and sprintlaps dataframes
    combined_df = pd.concat([laps_df, sprintlaps_df], ignore_index=True)

    deg_data = [] # initializing to store deg data later
    #makes groups where each one has laps from the same round and compound giving specific race and compound combo
    for (round_number, compound), group in combined_df.groupby(['RoundNumber', 'Compound']):
        if len(group) < 5:  # i need at least 5 laps to give this model a chance of working :p
            continue # i am lazy

        # x is tyre lifem, y is laptime as it was converted to just seconds
        x = group['TyreLife'].values
        y = group['LapTime'].values

        if len(x) > 1: #needs to be at least 2 points to make a line 
            # Removing nulls before fit checking
            mask = ~np.isnan(x) & ~np.isnan(y) # will return true if both values are not null
            x_clean = x[mask] #reseparates into x and y but cleared for nulls
            y_clean = y[mask]
            
            if len(x_clean) > 1: #triple checking for the sneaky nulls >:p
                try:
                    #lil lin reg to calc tyre deg
                    #slope is all we care about 
                    #positive slope = lap times increase w tyre age normal
                    #sharper positive slope = faster deg, more time lost per lap bc the tyres are toast
                    slope, intercept = np.polyfit(x_clean, y_clean, 1) # x = tire ages aka tyre life, y = lap times in sec, the 1 is the degree of the polyfit
                    deg_data.append({
                        'RoundNumber': round_number,  #race identifier
                        'Compound': compound, #tyre compound identifier
                        'DegradationRate': slope, #the holy slope
                        'MedianLapTime': np.median(y_clean), #baseline performance
                        'MedianTyreLife': np.median(x_clean), #baseline tyre life for the dataset
                        'LapNumber': len(x_clean) #laps analysed
                    })
                except:
                    # Skips if my red bull fuelled genius fails
                    continue

    deg_df = pd.DataFrame(deg_data) #converts the data into a dataframe

    deg_df = deg_df.merge(track_names, on='RoundNumber', how='left') #merge the track names with the data

    # Add a direct ranking of tracks by degradation rate
    track_deg_ranking = deg_df.groupby(['RoundNumber', 'OfficialEventName'])['DegradationRate'].mean().reset_index()
    track_deg_ranking = track_deg_ranking.sort_values('DegradationRate', ascending=False)
    
    # Feature engineering: Create a weighted degradation feature that emphasizes degradation more
    deg_df['WeightedDegradation'] = deg_df['DegradationRate'] * 3  # Triple the importance
    
    # Use the weighted feature for clustering
    features = ['WeightedDegradation', 'MedianLapTime']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(deg_df[features])
    
    # Rest of your K-means code remains the same
    wcss = []
    max_clusters = min(10, len(deg_df))
    
    
    n_clusters = 5  # Try 5 clusters instead of 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=301, n_init=10)
    
    deg_df['Cluster'] = kmeans.fit_predict(scaled_features)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    cluster_stats = deg_df.groupby('Cluster')['DegradationRate'].mean().sort_values(ascending=False)
    high_deg_cluster = cluster_stats.index[0]
    
    return deg_df, centroids, high_deg_cluster, features, track_deg_ranking


if __name__ == "__main__":


    deg_df, centroids, high_deg_cluster, features, track_deg_ranking = highest_tyre_deg()

    # BEEFY K-MEANS PLOT
    plt.figure(figsize=(20, 12))
    
    # Set up a custom color palette that matches F1 tire compounds
    colors = {
        0: '#0F2E68',  # Navy blue (highest degradation)
        1: '#2C75FF',  # Bright blue
        2: '#00B3A1',  # Teal
        3: '#7B4DFF',  # Purple
        4: '#0CD1E8',  # Electric blue (lowest degradation)
    }
    
    # Sort clusters by degradation rate (highest first)
    cluster_order = deg_df.groupby('Cluster')['DegradationRate'].mean().sort_values(ascending=False).index
    cluster_labels = {i: f"Cluster {cluster_order[i]} (Deg: {deg_df[deg_df['Cluster'] == cluster_order[i]]['DegradationRate'].mean():.4f}s/lap)" for i in range(len(cluster_order))}
    
    # Get list of top tracks to annotate (top 5 + Bahrain and Spain)
    top_track_names = track_deg_ranking.head(5)['OfficialEventName'].tolist()
    important_tracks = list(set(top_track_names + ['Bahrain Grand Prix', 'Spanish Grand Prix']))
    
    # Create reference dictionary for letter labels
    reference_dict = {}
    letter_idx = 0
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    # Plot each cluster with advanced styling
    for i, cluster in enumerate(cluster_order):
        cluster_data = deg_df[deg_df['Cluster'] == cluster]
        
        # Calculate compound distribution in this cluster
        compound_counts = cluster_data['Compound'].value_counts()
        most_common = compound_counts.idxmax() if len(compound_counts) > 0 else "N/A"
        
        # Main scatter plot for this cluster - now more visible without all annotations
        plt.scatter(
            cluster_data['DegradationRate'], 
            cluster_data['MedianLapTime'],
            label=f"{cluster_labels[i]} - Most common: {most_common}",
            color=colors[i],
            alpha=0.8,
            s=120,  # Bigger points
            edgecolors='black',
            linewidth=1
        )
        
        # Only annotate important tracks with letters
        for _, row in cluster_data.iterrows():
            if row['OfficialEventName'] in important_tracks and letter_idx < len(letters):
                # Use a letter instead of full track name
                letter = letters[letter_idx]
                letter_idx += 1
                
                # Store reference info
                reference_dict[letter] = {
                    'Track': row['OfficialEventName'].split(' Grand')[0],
                    'Compound': row['Compound'],
                    'DegradationRate': row['DegradationRate'],
                    'Color': colors[i]
                }
                
                # Add letter annotation
                plt.annotate(
                    letter,
                    (row['DegradationRate'], row['MedianLapTime']),
                    color='white',
                    weight='bold',
                    fontsize=12,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle="circle", fc=colors[i], ec="black", alpha=0.9)
                )
    
    # Plot centroids with beefier markers
    for i, centroid in enumerate(centroids):
        plt.scatter(
            centroid[0]/3,  # Divide by 3 since we multiplied by 3 in the weighting
            centroid[1],
            marker='X',
            s=300,  # Huge centroids
            color='black',
            edgecolors=colors[i],
            linewidth=3,
            alpha=0.9,
            label=f"Centroid {i}"
        )
    
    # Add grid lines for better orientation
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add special markers for Bahrain and Spain
    bahrain_spain_data = deg_df[deg_df['OfficialEventName'].isin(['Bahrain Grand Prix', 'Spanish Grand Prix'])]
    for _, row in bahrain_spain_data.iterrows():
        plt.scatter(
            row['DegradationRate'],
            row['MedianLapTime'],
            marker='*',
            s=350,  # Bigger stars
            color='red',
            edgecolors='black',
            linewidth=2,
            alpha=0.9,
            zorder=100  # Ensure it's on top
        )
    
    # Add legend with special styling
    legend = plt.legend(title="Tire Degradation Clusters", 
                       title_fontsize=14,
                       fontsize=12,
                       loc="upper left",
                       bbox_to_anchor=(1.01, 1),
                       fancybox=True, 
                       shadow=True)
    
    # Style the plot 
    plt.title('F1 Track Tire Degradation Analysis (2023 Season)', fontsize=20, pad=20)
    plt.xlabel('Degradation Rate (seconds/lap)', fontsize=14, labelpad=10)
    plt.ylabel('Median Lap Time (seconds)', fontsize=14, labelpad=10)
 
    # Add insights box with positioning on the right side
    insight_text = (
        "INSIGHTS:\n"
        f"• Highest degradation cluster: Cluster {cluster_order[0]}\n"
        f"• Avg degradation: {deg_df[deg_df['Cluster'] == cluster_order[0]]['DegradationRate'].mean():.4f} sec/lap\n"
        f"• Most common tire: {deg_df[deg_df['Cluster'] == cluster_order[0]]['Compound'].value_counts().idxmax()}\n"
        f"• Top degradation track: {track_deg_ranking.iloc[0]['OfficialEventName'].split(' Grand')[0]}\n"
        f"• Top track deg rate: {track_deg_ranking.iloc[0]['DegradationRate']:.4f} sec/lap"
    )
    
    # Position the insights box on the right side
    plt.figtext(0.82, 0.25, insight_text, fontsize=12,
              bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=1'))
    
    # Create and position reference box for letter labels
    reference_text = "TRACK REFERENCE:\n"
    for letter, info in reference_dict.items():
        reference_text += f"• {letter}: {info['Track']} - {info['Compound']} ({info['DegradationRate']:.3f}s/lap)\n"
    
    plt.figtext(0.82, 0.55, reference_text, fontsize=11,
              bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1', ec='black'))
    
    # Update layout to accommodate the right-side boxes
    plt.tight_layout(rect=[0, 0.07, 0.80, 1])
    
    plt.savefig('f1_tire_degradation_kmeans.png', dpi=300, bbox_inches='tight')
    
    # Print direct ranking alongside cluster analysis
    print("\nDirect ranking of tracks by degradation rate:")
    for _, row in track_deg_ranking.head(10).iterrows():
        print(f"{row['OfficialEventName']}: {row['DegradationRate']:.4f} sec/lap")
    
    plt.show()