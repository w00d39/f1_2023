import pandas as pd

def open_fastestlaps_data():
    """
    Load the fastest laps data from a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the fastest laps data.
    """
    # Load the fastest laps data
    fastestlaps_df = pd.read_csv('data_2023/Fastest_Laps.csv')
    
    
    return fastestlaps_df