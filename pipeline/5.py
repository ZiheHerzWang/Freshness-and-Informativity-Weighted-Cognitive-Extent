import pandas as pd
import glob

# Use glob to get all the CSV files that match the pattern
all_files = glob.glob(rf"word_counts_fits_process_*.csv")

list_data = []

# Iterate through the files and append them to the list
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    list_data.append(df)

# Concatenate all dataframes into one
merged_data = pd.concat(list_data, axis=0, ignore_index=True)

# Filter out rows where 'Area Ratio' is negative
merged_data = merged_data[merged_data['Area Ratio'] >= 0]

# Cap the 'Area Ratio' to 1 if it exceeds 1
merged_data['Area Ratio'] = merged_data['Area Ratio'].apply(lambda x: 1 if x > 1 else x)

# Save the filtered and merged data to a new CSV file
merged_data.to_csv(rf"newAreaCalculation_no_negative_filter_2.csv", index=False)

print("Merged, filtered, and capped 'Area Ratio' CSV saved to originMethod\newAreaCalculation.csv")
