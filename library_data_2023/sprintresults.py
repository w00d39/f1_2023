import pandas as pd

def open_sprintresults_data():
    """
    Load the sprint results data from a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the sprint results data.
    """
    # Load the sprint results data
    sprintresults_df = pd.read_csv('data_2023/Sprint_Results.csv')
    
    
    return sprintresults_df

def about_sprintresults():
    about_sprintresults = open_sprintresults_data()

    return print(about_sprintresults.columns)