import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import re

output_dir = 'originMethod'
os.makedirs(output_dir, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")
def gaussian(x, *params):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.zeros_like(x, dtype=torch.float32, device=device)
    for i in range(0, len(params), 3):
        amp = torch.tensor(params[i], device=device)
        ctr = torch.tensor(params[i+1], device=device)
        wid = torch.tensor(params[i+2], device=device)
        y += amp * torch.exp(-((x - ctr) / wid)**2)
    return y.cpu().numpy()

def fit_gaussian_two_peaks(x, y, maxfev=10000):
    initial_guess = [
        max(y), x[np.argmax(y)], np.std(x),  
        max(y) / 2, x[np.argmax(y)] + np.std(x), np.std(x) / 2 
    ]

    try:
        params, _ = curve_fit(gaussian, x, y, p0=initial_guess, maxfev=maxfev)
        return params
    except (RuntimeError, ValueError, TypeError):
        return None

def calculate_novelty(area_ratio):
    return np.cos(area_ratio * (np.pi / 2))

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
        y = np.array([counts.get(year, 0) for year in years])

        best_params = fit_gaussian_two_peaks(x, y)
        if best_params is None:
            continue

        extended_years = np.linspace(min(x) - 20, max(x) + 20, 1000)
        full_fitted_curve = gaussian(extended_years, *best_params)
        
        threshold = 0.001  
        start_index = np.where(full_fitted_curve > threshold)[0][0]
        end_index = np.where(full_fitted_curve > threshold)[0][-1]
        valid_years = extended_years[start_index:end_index + 1]
        valid_fitted_curve = full_fitted_curve[start_index:end_index + 1]

        plt.bar(x, y, alpha=0.5, label=f'{word} (Real Frequency)', color=colors[i % len(colors)], width=0.4)
        plt.plot(valid_years, valid_fitted_curve, label=f'{word} (Fitted Gaussian)', color='red')
        plt.axvline(x=publication_year, color='r', linestyle='--')

        area_under_curve = simpson(valid_fitted_curve, valid_years)
        area_ratio = area_under_curve / simpson(y, x)
        novelty_score = calculate_novelty(area_ratio)

        plt.text(publication_year, plt.ylim()[1] * 0.9, f'Area Ratio: {area_ratio:.2f}\nNovelty: {novelty_score:.2f}', 
                 verticalalignment='top', color='black')

    plt.title(f'Trends for "{title}"')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()

    if pd.isna(title):
        title = "Untitled"
    valid_title = re.sub(r'[^\w\s-]', '', str(title)).replace(' ', '_')
    valid_title = valid_title[:255]
    output_path = os.path.join(output_dir, f"{valid_title}.png")

    try:
        plt.savefig(output_path)
    except Exception as e:
        print(f"Error saving {output_path}: {e}")

    plt.close()

def process_papers_and_save_plots(input_file, merged_file_path, output_dir):
    output_results = pd.read_csv(input_file)
    merged_file = pd.read_csv(merged_file_path)

    output_results = output_results[pd.to_numeric(output_results['Year'], errors='coerce').notna()]
    output_results['Year'] = pd.to_numeric(output_results['Year'], errors='coerce')

    for _, paper in output_results.iterrows():
        important_words = paper['Important Words'].split(', ')
        publication_year = int(paper['Year'])
        plot_paper_trends(important_words, merged_file, paper['Title'], publication_year, output_dir)

input_file = rf'originMethod\processed_output_with_important_words_new.csv'
merged_file_path = rf'originMethod\newAreaCalculation.csv'
output_dir = rf'originMethod\Paper_Trends_Plots'

os.makedirs(output_dir, exist_ok=True)

process_papers_and_save_plots(input_file, merged_file_path, output_dir)
