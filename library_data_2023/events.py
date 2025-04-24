import pandas as pd 
from sklearn import preprocessing
import os

def open_events_data():
    """
    This function loads the events data from a CSV file, cleans it, and prepares it for analysis. 
    This is accessed as a module to keep everything tidy.
    """
  
    file_path = 'library_data_2023/data_2023/Events.csv' #the file path for the csv file
    if not os.path.exists(file_path): #check if the file exists 
        raise FileNotFoundError(f"The file {file_path} does not exist.") #raise an error if it does not
    events_df = pd.read_csv(file_path) #read the csv file into a dataframe
 
    #cleaning and converting the dataset so it can be manipulated as needed for various questions.

    #dropping column F1ApiSupport because it is not needed for analysis, inplace so we do not take up precious memory with multiple copies
    events_df.drop(columns=['F1ApiSupport'], inplace=True) 

    #converting utilizng label encoder so we can preserve the data itself to be readable
    label_encoder_column_list = ['Country', 'Location', 'EventFormat', 'Session1',
                                  'Session2', 'Session3', 'Session4', 'Session5']
    label_encoder = preprocessing.LabelEncoder() #creating the label encoder object

    for i in label_encoder_column_list:
        events_df[f'{i}_index'] = label_encoder.fit_transform(events_df[i])

    #converting the official event name column to string
    events_df['OfficialEventName'] = events_df['OfficialEventName'].astype(str) #converting the column to string    

   #converting the times to be usable for analysis
    time_columns = ['EventDate','Session1Date', 'Session2Date', 'Session3Date', 'Session4Date', 'Session5Date']
    for k in time_columns:
        events_df[k] = pd.to_datetime(events_df[k], errors = 'coerce') #converting the columns to datetime format
 
    return events_df

    


