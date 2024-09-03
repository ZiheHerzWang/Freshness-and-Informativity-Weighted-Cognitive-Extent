import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
import multiprocessing as mp  

output_dir = 'originMethod'
os.makedirs(output_dir, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Gaussian function using PyTorch (GPU)
def gaussian(x, *params):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.zeros_like(x, dtype=torch.float32, device=device)  # Ensure y is float32 to avoid casting issues
    for i in range(0, len(params), 3):
        amp = torch.tensor(params[i], device=device)
        ctr = torch.tensor(params[i+1], device=device)
        wid = torch.tensor(params[i+2], device=device)
        y += amp * torch.exp(-((x - ctr) / wid)**2)
    return y.cpu().numpy()  # Convert back to numpy array for compatibility with scipy

def fit_gaussian_dynamic(x, y, max_peaks=5, wid_range=(3, 7), n_splits=3, maxfev=10000):
    best_params = None
    best_residual = np.inf
    x = x.astype(np.float64)  # Ensure x is float64 to avoid casting issues
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

def count_papers(data, unique_word):
    counts = {}

    for row in data.itertuples(index=False):
        year = row.Year
        words = str(row[data.columns.get_loc('Important Words')]).lower()
        word_count = words.count(unique_word)

        if word_count > 0:
            counts[year] = counts.get(year, 0) + word_count  

    return counts

def analyze_data(args):
    filename, count_howmany = args
    unique_words_data = pd.read_csv(filename, encoding='utf-8')
    data = pd.read_csv(os.path.join(output_dir, 'new_output_gpt4_all.csv'))

    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    data['Important Words'] = data['Important Words'].fillna("")

    all_errors = []

    for unique_word in unique_words_data['Unique_Word'].str.lower().unique():
        if pd.isna(unique_word) or not isinstance(unique_word, str):
            continue  

        print(f"Processing '{unique_word}':")
        counts = count_papers(data, unique_word)

        years = np.array(sorted(counts.keys()), dtype=np.float64)
        counts_list = np.array([counts.get(year, 0) for year in years])

        if len(counts_list) < 3:  # Ensure at least 3 data points for fitting
            print(f" - Not enough data points for '{unique_word}'")
            continue

        # Fit Gaussian dynamically
        best_params = fit_gaussian_dynamic(years, counts_list)

        if best_params is not None:
            # Extend the years based on the fitted Gaussian curve
            extended_years = np.arange(min(years) - 20, max(years) + 100, 1)  # Extend both before and after the data range
            fitted_curve = gaussian(extended_years, *best_params)
            threshold = 1e-3  # Define a small threshold to determine when the curve reaches zero
            start_index = np.where(fitted_curve > threshold)[0]
            end_index = np.where(fitted_curve > threshold)[0]
            if len(start_index) > 0 and len(end_index) > 0:
                valid_years = extended_years[start_index[0]:end_index[-1] + 1]
                valid_fitted_curve = fitted_curve[start_index[0]:end_index[-1] + 1]
                for year, fitted_val in zip(valid_years, valid_fitted_curve):
                    if year in years:  
                        groundtruth_value = counts.get(year, 0)
                        error = np.abs(groundtruth_value - fitted_val)
                        all_errors.append(error)

        count_howmany.value += 1
        print(f"Total processed so far: {count_howmany.value}\n")

    return all_errors

def process_in_parallel(num_files):
    with mp.Manager() as manager:
        count_howmany = manager.Value('i', 0)
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(analyze_data, [(os.path.join(output_dir, f'word_counts_11111.csv_2_{i}.csv'), count_howmany) for i in range(num_files)])
        pool.close()
        pool.join()

        # Flatten the list of errors and calculate the overall average error
        all_errors = [error for sublist in results for error in sublist]
        overall_average_error = np.mean(all_errors)
        print('Overall Average Error:', overall_average_error)

if __name__ == '__main__':
    process_in_parallel(20)
