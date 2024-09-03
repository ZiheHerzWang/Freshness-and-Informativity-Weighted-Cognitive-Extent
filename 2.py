import pandas as pd
from collections import Counter

# Load your data
data = pd.read_csv('FAISS_new_output_gpt4_all_new.csv')

# Fill NA/NaN values 
data['Important Words'] = data['Important Words'].fillna("")

# Prepare Counter to store word occurrences
word_counts = Counter()

# Iterate over each row in the data
for _, row in data.iterrows():
    important_words = row['Important Words'].split(', ')

    # Count each word
    for word in important_words:
        word_counts[word.lower()] += 1  # Convert word to lowercase to ensure case-insensitive matching

# Prepare data for writing to CSV
rows = []
for word, count in word_counts.items():
    rows.append([word, count])

# Convert the data to a DataFrame
df = pd.DataFrame(rows, columns=['Unique_Word', 'Count'])

# Sort the DataFrame by Count, in descending order
df = df.sort_values(by='Count', ascending=False)

# Write the DataFrame to a CSV file
df.to_csv('word_counts_11111.csv', index=False)
