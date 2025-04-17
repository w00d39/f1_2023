import pandas as pd
import os
from sklearn import preprocessing

def open_fastestlaps_data():
    """
    Load the fastest laps data from a CSV file
    """
    # Load the fastest laps data
    file_path = 'library_data_2023/data_2023/FastestLaps.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    fastestlaps_df = pd.read_csv(file_path)

    # Clean the dataframe
    print("hi")
    #drop the pitouttime, and pitintime because they have nothing in them

    #simple data type conversons


    #encoding

    #datatime conversion
    
    
    return fastestlaps_df

