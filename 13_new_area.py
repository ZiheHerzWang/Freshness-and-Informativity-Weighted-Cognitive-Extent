import pandas as pd
import numpy as np
import math

# Load the merged file with area ratios and counts
merged_file = pd.read_csv(rf"originMethod\newAreaCalculation_no_negative_filter_less10_delete_length_check.csv")

def calculate_novelty(area_ratio):
    """Calculate the novelty score using the area ratio."""
    return math.cos(area_ratio * (math.pi / 2))

df = pd.read_csv(rf'originMethod\classified_by_quartile.csv')

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
    """Calculate weighted novelty score for the paper based on area ratio."""
    words = filter_important_words(important_words, year)
    if words:
        novelty_scores = []
        total_counts = []
        for word in words:
            word_data = merged_file[(merged_file['Unique Word'].str.lower() == word.lower()) & (merged_file['Year'] == year)]
            if not word_data.empty:
                area_ratio = word_data.iloc[0]['Area Ratio']
                count = word_data.iloc[0]['Count']
                novelty = calculate_novelty(area_ratio)
                novelty_scores.append(novelty)
                total_counts.append(count)
        if novelty_scores and sum(total_counts) > 0:
            # Weighted average of the novelty scores using the Count column
            weighted_novelty = np.average(novelty_scores, weights=total_counts)
            print(f"Calculating weighted novelty for {important_words} in year {year}: {weighted_novelty}")
            return weighted_novelty
        else:
            # If total_counts sum to zero, fallback to a simple average
            if sum(total_counts) == 0:
                return np.mean(novelty_scores)
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
df.to_csv(rf"originMethod\newAreaCalculation_by_length.csv", index=False)

print("Novelty scores calculated and saved to 'papers_with_novelty_scores_filter_30.csv'")