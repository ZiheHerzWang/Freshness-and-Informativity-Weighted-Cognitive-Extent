import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import UnivariateSpline
from numpy.fft import rfft, irfft, rfftfreq
from tqdm import tqdm
import os
import re

def calculate_adjusted_cutoff(counts):
    std_dev = np.std(counts)
    normalized_std_dev = (std_dev - np.min(counts)) / (np.max(counts) - np.min(counts))
    cutoff_frequency = 0.08 + normalized_std_dev * 0.13
    return cutoff_frequency

def calculate_novelty(slope, delta, a=1, b=1):  
    sigmoid_value = 1 / (1 + math.exp(-slope))
    if delta == 0:
        return 0
    return sigmoid_value

def smooth_signal_with_fft(y, cutoff_frequency):
    yf = rfft(y)
    xf = rfftfreq(len(y), 1)
    yf[xf > cutoff_frequency] = 0
    return irfft(yf, n=len(y))

output_results = pd.read_csv('processed_output_with_important_words_new.csv')
merged_file = pd.read_csv('AllMergedRealSlope.csv')

output_results['Citations'] = pd.to_numeric(output_results['Citations'], errors='coerce')

def plot_paper_trends(words, merged_file, title, publication_year, output_dir):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black']

    for i, word in enumerate(words):
        if len(word.strip()) <= 1:
            continue
        
        word_data = merged_file[merged_file['Unique Word'].str.lower() == word.lower().strip()]
        if word_data.empty:
            continue

        years = word_data['Year'].unique()
        years.sort()
        counts = word_data.groupby('Year')['Count'].sum()

        x = np.array(years)
        y = np.array([counts[year] for year in years])

        cutoff_frequency = 0.125
        y_smooth = smooth_signal_with_fft(y, cutoff_frequency)
        slopes = np.gradient(y_smooth)
        xs = np.linspace(years.min(), years.max(), len(y_smooth))
        ys = np.interp(xs, x, y_smooth)

        plt.bar(x, y, alpha=0.5, label=f'{word} (Real Frequency)', color=colors[i % len(colors)], width=0.4)
        plt.plot(xs, ys, label=f'{word} (Smoothed)', color=colors[i % len(colors)])
        plt.axvline(x=publication_year, color='r', linestyle='--', label=f'Publication Year {publication_year}')

        if publication_year in x:
            idx = np.where(x == publication_year)[0][0]
            slope_val = slopes[idx]
            plt.text(publication_year, ys[idx], f'{word} (Slope: {slope_val:.2f})', verticalalignment='bottom')
        else:
            plt.text(publication_year, plt.ylim()[1], f'{word} (Slope: N/A)', verticalalignment='top')

    plt.title(f'Trends for "{title}"')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()

    # Clean the title for a valid filename
    if pd.isna(title):
        title = "Untitled"
    valid_title = re.sub(r'[^\w\s-]', '', str(title)).replace(' ', '_')
    valid_title = valid_title[:255]  # Limit filename length to avoid OS limits
    output_path = os.path.join(output_dir, f"{valid_title}.png")

    try:
        plt.savefig(output_path)
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
    
    plt.close()

def process_quartile_files(input_dir):
    # Count total papers
    total_papers = 0
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            quartile_file_path = os.path.join(input_dir, file_name)
            quartile_papers = pd.read_csv(quartile_file_path)
            total_papers += len(quartile_papers)

    # Process files with progress bar
    with tqdm(total=total_papers, desc="Processing papers") as pbar:
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.csv'):
                quartile_file_path = os.path.join(input_dir, file_name)
                output_dir = os.path.join(input_dir, file_name.replace('.csv', ''))
                os.makedirs(output_dir, exist_ok=True)

                quartile_papers = pd.read_csv(quartile_file_path)

                for _, paper in quartile_papers.iterrows():
                    important_words = paper['Important Words'].split(', ')
                    publication_year = int(paper['Year'])
                    plot_paper_trends(important_words, merged_file, paper['Title'], publication_year, output_dir)
                    pbar.update(1)

process_quartile_files('Quartile_Samples')
