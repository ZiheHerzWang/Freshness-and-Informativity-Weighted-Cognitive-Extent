import pandas as pd

# Load the CSV file into a pandas DataFrame
file_path = rf'originMethod/papers_with_novelty_scores_filter_30.csv'
df = pd.read_csv(file_path)

# Filter rows where 'Novelty Score' is less than 0
filtered_df = df[df['Novelty Score'] < 0]

# Display the filtered dataframe
print(filtered_df)
