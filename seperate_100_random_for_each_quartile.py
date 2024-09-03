import pandas as pd
import numpy as np
import os
import langid

# Load the dataset
file_path = 'papers_with_novelty_scores.csv'
df = pd.read_csv(file_path)

# Ensure 'Quartile' column is standardized to uppercase
df['Quartile'] = df['Quartile'].str.upper()

# Filter papers with English titles
def is_english(title):
    try:
        if pd.isna(title):
            return False
        lang, _ = langid.classify(title)
        return lang == 'en'
    except:
        return False

df['is_english'] = df['Title'].apply(is_english)
df = df[df['is_english']]

# Define the sample size
sample_size = 100

# Create directory structure for output files
output_dir = 'Quartile_Samples'
os.makedirs(output_dir, exist_ok=True)

# Sample and save the papers for each quartile
for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
    quartile_df = df[df['Quartile'] == quartile]
    if len(quartile_df) < sample_size:
        sampled_papers = quartile_df.sample(n=len(quartile_df), random_state=42)
    else:
        sampled_papers = quartile_df.sample(n=sample_size, random_state=42)
    output_file = os.path.join(output_dir, f'{quartile}_papers.csv')
    sampled_papers.to_csv(output_file, index=False)

# Display the directory structure to the user
output_files = os.listdir(output_dir)
output_files
