import pandas as pd

def open_results_data():
    """
    Load the results data from a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the results data.
    """
    # Load the results data
    results_df = pd.read_csv('data_2023/Results.csv')
    
    return results_df

def about_results():

    about_results = open_results_data()

    return print(about_results.columns)