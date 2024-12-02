#%%
import pandas as pd

# Load the CSV file
file_path = r'FICE_Area_Preparation/csv file/word_similarity_scores_with_bins.csv'
df = pd.read_csv(file_path)
df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce')

# Iterate over each row, showing word1 and word2, and prompting for human_label input
for index, row in df.iterrows():
    if pd.isna(row['human_label']):  # Only ask for human label if it's missing
        print(f"word1: {row['word1']}, word2: {row['word2']}")
        while True:
            try:
                # Take human input for the label
                human_label = input("Enter human_label (0 or 1): ")
                if human_label in ["0", "1"]:
                    df.at[index, 'human_label'] = int(human_label)
                    break
                else:
                    print("Please enter 0 or 1.")
            except ValueError:
                print("Invalid input. Please enter 0 or 1.")
    else:
        print(f"word1: {row['word1']}, word2: {row['word2']} already labeled.")

# Save the updated CSV file with human labels
df.to_csv(file_path, index=False)

print(f"Human labels have been saved to {file_path}")
