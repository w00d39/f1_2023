import os
import laps
import results
print(os.getcwd())

results_df = results.open_results_data()
print(results_df.head())