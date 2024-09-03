import pandas as pd

# Load the data from the CSV file
file_path = 'AllMergedRealSlope_by_Frequency_by_length.csv'
data = pd.read_csv(file_path)

# Remove duplicate entries based on 'Unique Word' and 'Year'
filtered_data = data.drop_duplicates(subset=['Unique Word', 'Year'], keep='first')

# Save the cleaned data to a new CSV file
filtered_data.to_csv('Cleaned_AllMergedRealSlope_by_Frequency_by_length.csv', index=False)

print("Duplicates removed and new file saved.")
