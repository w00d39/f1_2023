import pandas as pd

def open_sprintlaps_data():
    """
    Load the sprint laps data from a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the sprint laps data.
    """
    # Load the sprint laps data
    sprintlaps_df = pd.read_csv('data_2023/Sprint_Laps.csv')
    
    return sprintlaps_df