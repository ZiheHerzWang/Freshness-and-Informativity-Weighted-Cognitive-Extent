import pandas as pd

# Load your data
data = pd.read_csv(f"FICE_Area_Preparation/csv file/all_gpt_extract.csv")

# Fill NA/NaN values
data['Scientific Entity'] = data['Scientific Entity'].fillna("")

# Extract unique words
unique_words = set()

# Iterate over each row in the data
for _, row in data.iterrows():
    important_words = row['Scientific Entity'].split(', ')
    for word in important_words:
        unique_words.add(word.lower())  # Convert word to lowercase for consistency

# Convert the unique words to a DataFrame
df = pd.DataFrame(unique_words, columns=['Word'])

# Write the DataFrame to a CSV file
df.to_csv(f'FICE_Area_Preparation/csv file/Unique_word.csv', index=False)
