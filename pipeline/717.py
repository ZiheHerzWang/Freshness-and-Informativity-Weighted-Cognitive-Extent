import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.signal import find_peaks
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

# Standard Gaussian function
def gaussian(x, *params):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.zeros_like(x, dtype=torch.float32, device=device)
    for i in range(0, len(params), 3):
        amp = torch.tensor(params[i], device=device)
        ctr = torch.tensor(params[i+1], device=device)
        wid = torch.tensor(params[i+2], device=device)
        y += amp * torch.exp(-((x - ctr) / wid)**2)
    return y.cpu().numpy()

# Loss function (MSE + L2 regularization)
# Loss function (MSE + Dynamic Peak Number Regularization with Nonlinear Adjustment)
def loss_with_dynamic_and_nonlinear_peak_regularization(y_true, y_pred, n_peaks, peak_lambda=1):
    # 计算均方误差
    mse_loss = np.mean((y_true - y_pred) ** 2)
    
    # 动态峰数量惩罚，非线性调整，标准化通过 y_true 的方差
    y_std = np.std(y_true)
    peak_reg = peak_lambda * (np.exp(n_peaks)) * (mse_loss / (y_std + 1e-5)) 
    
    return mse_loss + peak_reg



def generate_initial_guess(x, y, n_peaks):
    distance = max(1, len(x)//n_peaks)
    peaks, _ = find_peaks(y, distance=distance)  

    if len(peaks) < n_peaks:
        peaks = np.append(peaks, np.linspace(0, len(x)-1, n_peaks-len(peaks), dtype=int))

    # 构建初始猜测
    initial_guess = []
    for peak in peaks[:n_peaks]:  # 仅取前 n_peaks 个峰
        initial_guess += [y[peak], x[peak], np.std(x)/n_peaks]  # 假设每个峰的宽度为标准差/n_peaks
    
    return initial_guess

# 动态拟合高斯峰，并防止过拟合
def fit_gaussian_dynamic_peaks(x, y, max_peaks=5, reg_lambda=0.1, maxfev=10000):
    best_params = None
    best_loss = np.inf
    best_peaks = 1  # 记录最佳峰值数量
    
    # 限制最大峰值数量，确保参数数量不超过数据点数量
    max_allowed_peaks = min(max_peaks, len(x) // 3)
    
    for n_peaks in range(1, max_allowed_peaks + 1):
        initial_guess = generate_initial_guess(x, y, n_peaks)  # 每次生成新的初始猜测
        
        try:
            # 使用 Levenberg-Marquardt 算法进行拟合
            params, _ = curve_fit(gaussian, x, y, p0=initial_guess, maxfev=maxfev, method='lm')
            fitted_curve = gaussian(x, *params)
            
            # 计算带正则化的损失
            loss = loss_with_dynamic_and_nonlinear_peak_regularization(y, fitted_curve, n_peaks)
            print(f"Number of peaks: {n_peaks}, Loss: {loss:.4f}")  # 输出每个峰值的损失
            if loss < best_loss:
                best_loss = loss
                best_params = params
                best_peaks = n_peaks
        except (RuntimeError, ValueError, TypeError) as e:
            print(f"Failed fitting with {n_peaks} peaks: {str(e)}")
            continue
    
    return best_params, best_peaks

# Novelty calculation
def calculate_novelty(area_ratio):
    return np.cos(area_ratio * (np.pi / 2))

# Plot trends and the best fitting Gaussian
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

        # 动态峰值数量高斯拟合，使用 Levenberg-Marquardt 算法
        best_params, best_peaks = fit_gaussian_dynamic_peaks(x, y)
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
        plt.plot(valid_years, valid_fitted_curve, label=f'{word} (Fitted Gaussian, {best_peaks} Peaks)', color='red')
        plt.axvline(x=publication_year, color='r', linestyle='--')

        # Calculate area under the curve
        area_under_curve = simpson(y=valid_fitted_curve, x=valid_years)
        area_ratio = area_under_curve / simpson(y=y, x=x)
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

input_file = 'processed_output_with_important_words_new.csv'
merged_file_path = 'newAreaCalculation.csv'
output_dir = 'Paper_Trends_Plots'

os.makedirs(output_dir, exist_ok=True)

process_papers_and_save_plots(input_file, merged_file_path, output_dir)
