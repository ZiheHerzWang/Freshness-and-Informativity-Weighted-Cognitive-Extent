import pandas as pd
import glob

# Use glob to get all the CSV files that match the pattern
#originMethod\word_counts_and_fits_no_negative_filter_word_counts_11111.csv_2_0.csv
all_files = glob.glob(rf"originMethod\word_counts_and_fits_no_negative_filter_word_counts_11111.csv_2_*.csv")

list_data = []

# Iterate through the files and append them to the list
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    list_data.append(df)

# Concatenate all dataframes into one
merged_data = pd.concat(list_data, axis=0, ignore_index=True)

# Filter out rows where 'Area Ratio' or 'Area' is negative
merged_data = merged_data[merged_data['Area Ratio'] >= 0] 
# merged_data = merged_data[merged_data['Area'] >= 0]

# Save the filtered and merged data to a new CSV file
merged_data.to_csv(rf"originMethod\newAreaCalculation_no_negative_filter.csv", index=False)

print("Merged and filtered CSV saved to originMethod\newAreaCalculation.csv")
#word_counts_and_fits_no_negative_filter_word_counts_11111.csv_2_19