import pandas as pd
import numpy as np
import math

def calculate_novelty(slope, k=1):
    if -0.01 <= slope <= 0.01:
        return -1  
    return 1 / (1 + math.exp(-k * slope))

df = pd.read_csv('classified_by_quartile.csv')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df.dropna(subset=['Year'], inplace=True)

merged_file = pd.read_csv('AllMergedRealSlope.csv')

def filter_important_words(important_words, year):
    if isinstance(important_words, str):
        return important_words.split(', ')
    return []

def calculate_paper_novelty(important_words, year, k=1):
    words = filter_important_words(important_words, year)
    if words:
        novelty_scores = []
        total_counts = []
        for word in words:
            word_data = merged_file[(merged_file['Unique Word'].str.lower() == word.lower()) & (merged_file['Year'] == year)]
            if not word_data.empty:
                slope = word_data.iloc[0]['Slope']
                count = word_data.iloc[0]['Count']
                novelty = calculate_novelty(slope, k)
                novelty_scores.append(novelty)
                total_counts.append(count)
        if novelty_scores:
            # Use a minimum value of 0.001 to avoid division by zero errors
            weights = [1 / (count if count > 0 else 0.001) for count in total_counts]
            weighted_novelty = np.average(novelty_scores, weights=weights)
            print(f"Calculating weighted novelty for {important_words} in year {year}: {weighted_novelty}")
            return weighted_novelty
    return 0

def apply_novelty(row, k=1):
    year = int(row['Year'])
    return calculate_paper_novelty(row['Important Words'], year, k)

df['Filtered Words'] = df.apply(lambda row: filter_important_words(row['Important Words'], row['Year']), axis=1)
df = df[df['Filtered Words'].map(len) > 0] 
print(f"Total papers after filtering: {len(df)}")

df['Novelty Score'] = df.apply(lambda row: apply_novelty(row, k=1), axis=1)
df.to_csv('papers_with_novelty_scores_non.csv', index=False)
