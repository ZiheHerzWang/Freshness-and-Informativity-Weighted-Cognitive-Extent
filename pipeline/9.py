import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from sklearn.model_selection import KFold

def gaussian(x, *params):
    y = np.zeros_like(x, dtype=np.float64)
    for i in range(0, len(params), 3):
        amp = params[i]
        ctr = params[i+1]
        wid = params[i+2]
        y += amp * np.exp(-((x - ctr) / wid)**2)
    return y

def fit_gaussian_dynamic(x, y, max_peaks=5, wid_range=(3, 7), n_splits=3, maxfev=10000):
    best_params = None
    best_residual = np.inf
    x = x.astype(np.float64)
    max_peaks = min(max_peaks, len(x) // 3)
    n_splits = min(n_splits, len(x))
    kf = KFold(n_splits=n_splits)
    
    for n_peaks in range(1, max_peaks + 1):
        if len(x) < n_peaks * 3:
            continue
        
        for initial_wid in np.linspace(wid_range[0], wid_range[1], num=3):
            residuals = []
            for train_index, test_index in kf.split(x):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                initial_guess = []
                for i in range(n_peaks):
                    initial_guess += [max(y_train), x_train[np.argmax(y_train)], initial_wid]

                try:
                    if len(x_train) < len(initial_guess) // 3:
                        continue

                    params, _ = curve_fit(gaussian, x_train, y_train, p0=initial_guess, maxfev=maxfev)
                    fitted_curve = gaussian(x_test, *params)
                    residual = np.sum((y_test - fitted_curve) ** 2)
                    residuals.append(residual)
                except (RuntimeError, ValueError, TypeError):
                    residuals.append(np.inf)

            avg_residual = np.mean(residuals)
            if avg_residual < best_residual:
                best_residual = avg_residual
                best_params = params

    return best_params

def calculate_novelty(slope, delta, a=1, b=1):
    sigmoid_value = 1 / (1 + math.exp(-slope))
    if delta == 0:
        return 0
    return sigmoid_value

def plot_paper_trends(words, merged_file, title, publication_year):
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
        best_params = fit_gaussian_dynamic(x, y)
        if best_params is None:
            continue

        fitted_curve = gaussian(x, *best_params)
        plt.bar(x, y, alpha=0.5, label=f'{word} (Real Frequency)', color=colors[i % len(colors)], width=0.4)
        plt.plot(x, fitted_curve, label=f'{word} (Fitted Gaussian)', color='red')
        plt.axvline(x=publication_year, color='r', linestyle='--')
        slopes = np.gradient(fitted_curve)
        if publication_year in x:
            idx = np.where(x == publication_year)[0][0]
            slope_val = slopes[idx]
            plt.text(publication_year, fitted_curve[idx], f'{word} (Slope: {slope_val:.2f})', verticalalignment='bottom')
        else:
            plt.text(publication_year, plt.ylim()[1], f'{word} (Slope: N/A)', verticalalignment='top')

    plt.title(f'Trends for "{title}"')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

def generate_paper_summary_table(paper, merged_file, publication_year):
    citations = paper['Citations']
    important_words = paper['Important Words'].split(', ')
    summary_data = []
    total_frequency = 0  
    
    for word in important_words:
        word_data = merged_file[(merged_file['Unique Word'].str.lower() == word.lower()) &
                                (merged_file['Year'] == publication_year)]
        if not word_data.empty:
            frequency = word_data['Count'].sum()
            total_frequency += frequency
            slope = word_data.iloc[0]['Slope']
            novelty = calculate_novelty(slope, frequency)
            summary_data.append({'Word': word, 'Slope': slope, 'Frequency': frequency, 'Novelty': novelty, 'Citations': citations})

    summary_table = pd.DataFrame(summary_data)

    if not summary_table.empty and total_frequency > 0:
        summary_table['Weight'] = summary_table['Frequency'] / total_frequency
        summary_table['Weighted Novelty'] = summary_table['Novelty'] * summary_table['Weight']
        paper_novelty_score = summary_table['Weighted Novelty'].sum()
        summary_table['Paper Novelty Score'] = paper_novelty_score
        return summary_table
    return pd.DataFrame() 

output_results = pd.read_csv('processed_output_with_important_words_new.csv')
merged_file = pd.read_csv('newAreaCalculation.csv')

output_results['Citations'] = pd.to_numeric(output_results['Citations'], errors='coerce')

titles_to_find = [
    'An Empirical Study of Diversity of Word Alignment and its Symmetrization Techniques for System Combination'
]

selected_papers = output_results[output_results['Title'].isin(titles_to_find)]

for _, paper in selected_papers.iterrows():
    important_words = paper['Important Words'].split(', ')
    print(important_words)
    publication_year = int(paper['Year']) 
    plot_paper_trends(important_words, merged_file, paper['Title'], publication_year)
    
for index, paper in selected_papers.iterrows():
    publication_year = int(paper['Year'])
    summary_table = generate_paper_summary_table(paper, merged_file, publication_year)
    if not summary_table.empty:
        print(f"Summary for paper: {paper['Title']} (Year: {paper['Year']}, Citations: {paper['Citations']})")
        print(summary_table[['Word', 'Slope', 'Frequency', 'Novelty', 'Weighted Novelty', 'Paper Novelty Score']])
        print("\n" + "-"*80 + "\n")
    else:
        print(f"No data found for paper: {paper['Title']}")
