import pandas as pd

def open_events_data():
    """
    Load the events data from a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the events data.
    """
    # Load the events data
    events_df = pd.read_csv('data_2023/Events.csv')
    

    
    return events_df