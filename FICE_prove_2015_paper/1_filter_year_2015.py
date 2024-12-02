import pandas as pd

# File path
file_path = "FICE_Area_Preparation/references.csv"

# Load the CSV file
data = pd.read_csv(file_path)

# Filter for rows where the year is 2015
filtered_data = data[data['Year'] == 2015]

# Specify the output file path
output_file_path = "FICE_prove_2015_paper/filtered_2015.csv"

# Save the filtered data to a new CSV file
filtered_data.to_csv(output_file_path, index=False)

print(f"Filtered data saved to {output_file_path}")
