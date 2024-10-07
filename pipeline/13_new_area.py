import pandas as pd
import numpy as np
import math

# Load the merged file with area ratios and counts
merged_file = pd.read_csv("newAreaCalculation_no_negative_filter_2.csv")
print(merged_file.columns)

def calculate_novelty(area_ratio):
    """Calculate the novelty score using the area ratio."""
    return 1-area_ratio

df = pd.read_csv('classified_by_quartile.csv')

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df.dropna(subset=['Year'], inplace=True)

def filter_important_words(important_words, year):
    """Filter words based on area ratio from the merged file."""
    if isinstance(important_words, str):
        words = important_words.split(', ')
        valid_words = []
        for word in words:
            word_data = merged_file[(merged_file['Unique Word'].str.lower() == word.lower()) &
                                    (merged_file['Year'] == year)]
            if not word_data.empty:
                total_area_ratio = word_data['Area Ratio'].sum()
                print(f"Word: {word}, Area Ratio in {year}: {total_area_ratio}")
                if total_area_ratio >= 0.1:  # Adjust threshold as needed
                    valid_words.append(word)
        return valid_words
    return []

def calculate_paper_novelty(important_words, year):
    """Calculate weighted novelty score for the paper based on area ratio and normalized counts up to the paper's publication year."""
    words = filter_important_words(important_words, year)
    if words:
        novelty_scores = []
        total_weights = []

        # Find the min and max count for normalization
        word_data_up_to_year = merged_file[(merged_file['Year'] <= year)]
        min_occurrences = word_data_up_to_year['Count'].min()
        max_occurrences = word_data_up_to_year['Count'].max()

        for word in words:
            # Get occurrences of the word up to the paper's publication year
            word_data_up_to_year = merged_file[(merged_file['Unique Word'].str.lower() == word.lower()) &
                                               (merged_file['Year'] <= year)]
            total_occurrences_up_to_year = word_data_up_to_year['Count'].sum()

            # Check if the total occurrences are negative, which shouldn't happen
            if total_occurrences_up_to_year < 0:
                raise ValueError(f"Error: Negative total occurrences found for word '{word}' in year {year}. Total occurrences: {total_occurrences_up_to_year}")

            # Calculate novelty using reverse occurrence weight
            word_data_in_year = word_data_up_to_year[word_data_up_to_year['Year'] == year]
            if not word_data_in_year.empty:
                area_ratio = word_data_in_year.iloc[0]['Area Ratio']
                novelty = calculate_novelty(area_ratio)

                # Normalize total_occurrences_up_to_year to get a weight in the range (0, 1]
                if max_occurrences > min_occurrences:  # To avoid division by zero
                    weight = 1 - ((total_occurrences_up_to_year - min_occurrences) / (max_occurrences - min_occurrences))
                else:
                    weight = 1  # If all counts are the same, treat as maximally novel

                novelty_scores.append(novelty)
                total_weights.append(weight)
        
        if novelty_scores and sum(total_weights) > 0:
            # Weighted average of the novelty scores using the normalized weights
            weighted_novelty = np.average(novelty_scores, weights=total_weights)
            print(f"Calculating weighted novelty for {important_words} in year {year}: {weighted_novelty}")
            return weighted_novelty
        else:
            # Fallback to simple average if no valid weights
            return np.mean(novelty_scores) if novelty_scores else 0
    return 0



def apply_novelty(row):
    year = int(row['Year'])
    return calculate_paper_novelty(row['Important Words'], year)

# Filter out papers where no important words meet the criteria
df['Filtered Words'] = df.apply(lambda row: filter_important_words(row['Important Words'], row['Year']), axis=1)
df = df[df['Filtered Words'].map(len) > 0]
print(f"Total papers after filtering: {len(df)}")

# Calculate novelty score for each paper
df['Novelty Score'] = df.apply(lambda row: apply_novelty(row), axis=1)

# Save the results
df.to_csv("newAreaCalculation_by_length.csv", index=False)

print("Novelty scores calculated and saved to 'newAreaCalculation_by_length.csv'")
