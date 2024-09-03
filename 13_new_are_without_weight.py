import pandas as pd
import numpy as np
import math

merged_file = pd.read_csv(rf"originMethod\newAreaCalculation_no_negative_filter_less10_delete.csv")

def calculate_novelty(area_ratio):
    return math.cos(area_ratio * (math.pi / 2))

df = pd.read_csv(rf'originMethod\classified_by_quartile.csv')

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df.dropna(subset=['Year'], inplace=True)

def filter_important_words(important_words, year):
    if isinstance(important_words, str):
        words = important_words.split(', ')
        valid_words = []
        for word in words:
            word_data = merged_file[(merged_file['Unique Word'].str.lower() == word.lower()) &
                                    (merged_file['Year'] == year)]
            if not word_data.empty:
                total_area_ratio = word_data['Area Ratio'].sum()
                print(f"Word: {word}, Area Ratio in {year}: {total_area_ratio}")
                if total_area_ratio >= 0.1:  
                    valid_words.append(word)
        return valid_words
    return []

def calculate_paper_novelty(important_words, year):
    words = filter_important_words(important_words, year)
    if words:
        novelty_scores = []
        for word in words:
            word_data = merged_file[(merged_file['Unique Word'].str.lower() == word.lower()) & (merged_file['Year'] == year)]
            if not word_data.empty:
                area_ratio = word_data.iloc[0]['Area Ratio']
                novelty = calculate_novelty(area_ratio)
                novelty_scores.append(novelty)
        if novelty_scores:
            simple_novelty = np.mean(novelty_scores)
            print(f"Calculating simple novelty for {important_words} in year {year}: {simple_novelty}")
            return simple_novelty
    return 0

def apply_novelty(row):
    year = int(row['Year'])
    return calculate_paper_novelty(row['Important Words'], year)

df['Filtered Words'] = df.apply(lambda row: filter_important_words(row['Important Words'], row['Year']), axis=1)
df = df[df['Filtered Words'].map(len) > 0] 
print(f"Total papers after filtering: {len(df)}")
df['Novelty Score'] = df.apply(lambda row: apply_novelty(row), axis=1)
df.to_csv(rf"originMethod\newAreaCalculation_by_length_no_weight.csv", index=False)
print("Novelty scores calculated and saved to 'newAreaCalculation_by_length_no_weight.csv'")
