import csv
import pandas as pd


csv_file_list = ["csv/down.csv", "csv/left.csv", "csv/lower_left.csv", "csv/lower_right.csv",
                 "csv/middle.csv", "csv/right.csv", "csv/up.csv", "csv/upper_left.csv", "csv/upper_right.csv"]

list_of_dataframes = []
for filename in csv_file_list:
    list_of_dataframes.append(pd.read_csv(filename))

merged_df = pd.concat(list_of_dataframes)

merged_df.to_csv('intial_dataset.csv')

print(merged_df.shape)
