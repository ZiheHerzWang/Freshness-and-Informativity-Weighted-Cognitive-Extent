
import pandas as pd
from collections import Counter

device_total_number = 20 # modify it to your device thread
# Load your data
data = pd.read_csv(r'FICE_Area_Preparation/temp_all_gpt_extract.csv')
# Fill NA/NaN values 
data['Scientific Entity Disambigious'] = data['Scientific Entity Disambigious'].fillna("")

# Prepare Counter to store word occurrences
word_counts = Counter()

# Iterate over each row in the data
for _, row in data.iterrows():
    important_words = row['Scientific Entity Disambigious'].split(', ')

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
df.to_csv(f'C:/Users/zihe0/Desktop/try/Novelty-Life-Time/FICE_Area_Preparation/Unique_word_with_count.csv', index=False)

df = pd.read_csv(rf'C:/Users/zihe0/Desktop/try/Novelty-Life-Time/FICE_Area_Preparation/Unique_word_with_count.csv')

chunk_size = -(-len(df) // device_total_number)  

for i in range(device_total_number):
    start = i * chunk_size
    end = start + chunk_size
    
    chunk_df = df.iloc[start:end]
    
    chunk_filename = rf'C:/Users/zihe0/Desktop/try/Novelty-Life-Time/FICE_Area_Preparation/parallel/Unique_word_with_count_{i}.csv'
    chunk_df.to_csv(chunk_filename, index=False)

