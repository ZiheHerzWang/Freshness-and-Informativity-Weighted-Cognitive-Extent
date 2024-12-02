#%%
import pandas as pd
from tqdm import tqdm
import numpy as np

tqdm.pandas()
# Load the merged file with area ratios and counts
merged_file = pd.read_csv(f"FICE_Area_Preparation/parallel/area_csv_file/merged_area_file.csv")

def calculate_FICE(area_ratio):
    return 1-area_ratio

df = pd.read_csv(f'FICE_prove_2015_paper/2015_scientific_entity.csv')

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
                if total_area_ratio > 1:
                    raise ValueError(f"The words are {words} and the weighted FICE score is {total_area_ratio}, which exceeds the maximum allowable value of 1.")
                if total_area_ratio >= 0.1:  
                    valid_words.append(word)
        return valid_words
    return []

def calculate_paper_FICE(important_words, year):
    """Calculate weighted FICE score for the paper based on area ratio and normalized counts up to the paper's publication year."""
    words = filter_important_words(important_words, year)
    if words:
        FICE_scores = []
        total_weights = []

        # Find the min and max count for normalization
        word_data_up_to_year = merged_file[(merged_file['Year'] <= year) & (merged_file['Unique Word'].str.lower().isin([w.lower() for w in words]))]
        word_counts = word_data_up_to_year.groupby('Unique Word')['Count'].sum()

        # Calculate the minimum and maximum of these summed counts
        min_occurrences = word_counts.min()
        max_occurrences = word_counts.max()
        # print(f"max: {max_occurrences} and min: {min_occurrences}")
        for word in words:
            # Get occurrences of the word up to the paper's publication year
            word_data_up_to_year = merged_file[(merged_file['Unique Word'].str.lower() == word.lower()) &
                                               (merged_file['Year'] <= year)]
            total_occurrences_up_to_year = word_data_up_to_year['Count'].sum()

            # Check if the total occurrences are negative, which shouldn't happen
            if total_occurrences_up_to_year < 0:
                raise ValueError(f"Error: Negative total occurrences found for word '{word}' in year {year}. Total occurrences: {total_occurrences_up_to_year}")

            # Calculate FICE using reverse occurrence weight
            word_data_in_year = word_data_up_to_year[word_data_up_to_year['Year'] == year]
            if not word_data_in_year.empty:
                area_ratio = word_data_in_year.iloc[0]['Area Ratio']
                FICE = calculate_FICE(area_ratio)
                if max_occurrences > min_occurrences:  # To avoid division by zero
                    weight = 1 - ((total_occurrences_up_to_year - min_occurrences) / (max_occurrences - min_occurrences))
                else:
                    weight = 1  # If all counts are the same, treat as maximally novel

                FICE_scores.append(FICE)
                total_weights.append(weight)
        
        if FICE_scores and sum(total_weights) > 0:
            # Weighted average of the FICE scores using the normalized weights
            normalized_weights = total_weights / np.sum(total_weights)
            # Calculate the weighted average using the normalized weights
            weighted_FICE = sum(FICE_scores * normalized_weights)
            return weighted_FICE
        else:
            # Fallback to simple average if no valid weights
            return np.mean(FICE_scores) if FICE_scores else 0
    return 0

def apply_FICE(row):
    year = int(row['Year'])
    return calculate_paper_FICE(row['Scientific Entity Disambigious'], year)


# Calculate FICE score for each paper
df['FICE Score'] = df.progress_apply(lambda row: apply_FICE(row), axis=1)

# Save the results
df.to_csv(f"FICE_prove_2015_paper/2015_paper_FICE_score.csv", index=False)

