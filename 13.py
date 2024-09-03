import pandas as pd
import numpy as np
import math

def calculate_novelty(slope, k=1):
    return 1 / (1 + math.exp(-k * slope))

# Load the DataFrame containing papers and their important words
df = pd.read_csv('classified_by_quartile.csv')

# Load the merged file with slope data for each word
merged_file = pd.read_csv('AllMergedRealSlope.csv')

# Define a function to compute the weighted novelty score for a paper
def calculate_paper_novelty(important_words, year):
    if isinstance(important_words, str):  # Ensure important_words is a string
        words = important_words.split(', ')
        summary_data = []
        total_frequency = 0

        for word in words:
            word = word.title()
            print(word)  # Print each word
            word_data = merged_file[(merged_file['Unique Word'].str.lower() == word.lower()) & (merged_file['Year'] == year)]
            print(word_data)  # Print the data of the word
            if not word_data.empty:
                frequency = word_data['Count'].sum()  # Total occurrences of the word up to the publication year
                total_frequency += frequency
                slope = word_data.iloc[0]['Slope']
                novelty = calculate_novelty(slope)
                print(novelty)  # Print the novelty calculated
                summary_data.append((novelty, frequency))

        # Calculate weighted average novelty if data exists
        if summary_data:
            weighted_novelty = sum(nov * freq for nov, freq in summary_data) / total_frequency
            return weighted_novelty
    return 0  # Return 0 if important_words is not a string

def apply_novelty(row):
    try:
        year = int(row['Year'])  # Attempt to convert year to an integer
        print(year)  # Print the year
        return calculate_paper_novelty(row['Important Words'], year)
    except ValueError:
        print(f"Skipping row with invalid year: {row['Year']}")  # Print error message if conversion fails
        return np.nan  # Return NaN if the year is not valid

# Calculate novelty score for each paper and store in a new column
df['Novelty Score'] = df.apply(apply_novelty, axis=1)

# Save the updated DataFrame
df.to_csv('papers_with_novelty_scores.csv', index=False)
