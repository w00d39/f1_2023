import pandas as pd

def open_laps_data(): 
    """
    Load the laps data from a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the laps data.
    """
    # Load the laps data
    laps_df = pd.read_csv('data_2023/Laps.csv')
    
    
    return laps_df


def about_laps():
    about_laps = open_laps_data()
    
    return print(about_laps.columns)