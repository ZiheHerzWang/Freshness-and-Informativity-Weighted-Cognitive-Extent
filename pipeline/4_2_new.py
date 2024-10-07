import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import multiprocessing as mp  
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.signal import find_peaks
import re

output_dir = 'originMethod'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义高斯模型
class GaussianModel(torch.nn.Module):
    def __init__(self, n_peaks):
        super(GaussianModel, self).__init__()
        self.amplitude = torch.nn.Parameter(torch.randn(n_peaks, device=device))
        self.center = torch.nn.Parameter(torch.randn(n_peaks, device=device))
        self.width = torch.nn.Parameter(torch.abs(torch.randn(n_peaks, device=device)))  # 确保宽度为正值
        self.n_peaks_param = torch.nn.Parameter(torch.tensor(float(n_peaks), device=device))  # 使 n_peaks 可微

    def forward(self, x):
        y = torch.zeros_like(x, device=device)
        # 确保 n_peaks 不超过 5 且为正整数
        n_peaks_rounded = torch.round(torch.clamp(self.n_peaks_param, 1, 5)).int()
        for i in range(n_peaks_rounded):
            y += self.amplitude[i] * torch.exp(-((x - self.center[i]) ** 2) / (2 * self.width[i] ** 2))
        return y

# 自动检测 n_peaks 的数量
def detect_number_of_peaks(y_data):
    peaks, _ = find_peaks(y_data, distance=5)
    return max(1, len(peaks))  # 确保返回至少 1 个峰值

# 生成初始猜测
def generate_initial_guess(x, y, n_peaks):
    distance = max(1, len(x) // n_peaks)
    peaks, _ = find_peaks(y, distance=distance)

    if len(peaks) < n_peaks:
        peaks = np.append(peaks, np.linspace(0, len(x) - 1, n_peaks - len(peaks), dtype=int))

    initial_amplitude = np.random.uniform(5.0, 10.0, size=n_peaks)
    initial_center = x[peaks]
    initial_width = np.std(x) / n_peaks

    return initial_amplitude, initial_center, initial_width

# 训练过程
def train_gaussian(x_data, y_data, n_peaks, lr=1e-2, epochs=1, reg_lambda=0.001, total_count=1):
    x = torch.tensor(x_data, dtype=torch.float32, device=device)
    y_true = torch.tensor(y_data, dtype=torch.float32, device=device)

    # 初始化高斯模型
    amplitude, center, width = generate_initial_guess(x_data, y_data, n_peaks)
    model = GaussianModel(n_peaks).to(device)
    
    # 使用初始猜测值初始化参数
    with torch.no_grad():
        model.amplitude.copy_(torch.tensor(amplitude, device=device))
        model.center.copy_(torch.tensor(center, device=device))
        model.width.copy_(torch.tensor(width, device=device))

    # 损失函数
    criterion = torch.nn.MSELoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        mse_loss = criterion(y_pred, y_true)

        # 添加正则化项，限制高斯函数的宽度和振幅以防止过拟合
        reg_term = reg_lambda * (torch.sum(torch.exp(model.width)) + torch.sum(torch.abs(model.amplitude))) * int(n_peaks)
        loss = mse_loss + reg_term

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

    return model, loss.item()

# 保存拟合图像
def save_fitting_plot(years, counts_list, extended_years, fitted_curve, unique_word, process_id):
    plt.figure(figsize=(10, 6))
    plt.bar(years, counts_list, alpha=0.5, label='Ground Truth')
    plt.plot(extended_years, fitted_curve, 'r-', label='Fitted Gaussian')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title(f"Fitting for '{unique_word}' (Process {process_id})")
    plt.legend()
    
    # 生成文件名并尝试保存
    sanitized_word = re.sub(r'[<>:"/\\|?*]', '', unique_word)
    process_output_dir = os.path.join(output_dir, f'process_{process_id}')
    os.makedirs(process_output_dir, exist_ok=True)
    filename = os.path.join(process_output_dir, f"fitting_{sanitized_word}.png")
    
    try:
        plt.savefig(filename)
    except Exception as e:
        print(f"无法保存图片 '{unique_word}': {e}")
    plt.close()

# 计算面积比
def calculate_area_ratio(extended_years, fitted_curve, current_year):
    total_area = simpson(y=fitted_curve, x=extended_years)
    valid_years_idx = extended_years <= current_year
    valid_years = extended_years[valid_years_idx]
    valid_fitted_curve = fitted_curve[valid_years_idx]
    area_up_to_current = simpson(y=valid_fitted_curve, x=valid_years)
    return area_up_to_current / total_area if total_area > 0 else 0

# 保存结果到 CSV
def save_results_extended(filename, unique_word, years, counts_list, extended_years, fitted_curve):
    results = []
    ground_truth_dict = dict(zip(years, counts_list))
    
    for year, fitted_value in zip(extended_years, fitted_curve):
        area_ratio = calculate_area_ratio(extended_years, fitted_curve, year)
        ground_truth_value = ground_truth_dict.get(year, 0)
        results.append([unique_word, year, ground_truth_value, fitted_value, area_ratio])

    df = pd.DataFrame(results, columns=['Unique Word', 'Year', 'Count', 'Fitted Value', 'Area Ratio'])
    if not os.path.isfile(filename):
        df.to_csv(filename, mode='w', header=True, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

# 统计包含特定词汇的论文数
def count_papers(data, unique_word):
    counts = {}
    for row in data.itertuples(index=False):
        year = row.Year
        words = str(row[data.columns.get_loc('Important Words')]).lower()
        word_count = words.count(unique_word)
        if word_count > 0:
            counts[year] = counts.get(year, 0) + word_count
    return counts

# 分析数据并进行高斯拟合
def analyze_data(args):
    output_dir = 'originMethod'
    os.makedirs(output_dir, exist_ok=True)

    filename, count_howmany, process_id = args
    unique_words_data = pd.read_csv(filename, encoding='utf-8')
    data = pd.read_csv('new_output_gpt4_all.csv')

    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    data['Important Words'] = data['Important Words'].fillna("")

    result_filename = os.path.join(output_dir, f"word_counts_fits_process_{process_id}.csv")

    # 初始化累计的总 loss 和总词汇数
    total_loss = 0
    total_word_count = 0

    for unique_word in unique_words_data['Unique_Word'].str.lower().unique():
        if pd.isna(unique_word) or not isinstance(unique_word, str):
            continue

        counts = count_papers(data, unique_word)
        years = np.array(sorted(counts.keys()), dtype=np.float64)
        counts_list = np.array([counts.get(year, 0) for year in years])

        if len(counts_list) < 3:
            continue

        # 动态检测 n_peaks
        n_peaks = detect_number_of_peaks(counts_list)

        # 计算该词的总出现次数
        total_count = np.sum(counts_list)

        # 根据总出现次数动态确定 epochs
        epochs = max(total_count, 500)
        epochs = epochs * 10
        
        # 使用自定义的高斯模型进行拟合
        model, loss = train_gaussian(
            years,
            counts_list,
            n_peaks=n_peaks,
            lr=1e-2,
            epochs=epochs,
            reg_lambda=0.001,
            total_count=total_count
        )

        extended_years = np.arange(min(years) - 20, max(years) + 100, 1)
        extended_years_tensor = torch.tensor(extended_years, dtype=torch.float32, device=device)
        fitted_curve = model(extended_years_tensor).cpu().detach().numpy()

        # 保存拟合结果和图像
        save_fitting_plot(years, counts_list, extended_years, fitted_curve, unique_word, process_id)
        save_results_extended(result_filename, unique_word, years, counts_list, extended_years, fitted_curve)

        # 累计总 loss 和总词汇数
        total_loss += loss / total_count  # 按词汇出现次数加权 loss
        total_word_count += total_count

        count_howmany.value += 1
        print(f"Total processed so far: {count_howmany.value}")

    # 返回所有词的平均 loss
    average_loss = total_loss / total_word_count if total_word_count > 0 else 0
    print(f"Average Loss: {average_loss}")

# 并行处理多个文件
def process_in_parallel(num_files):
    with mp.Manager() as manager:
        count_howmany = manager.Value('i', 0)
        pool = mp.Pool(mp.cpu_count())
        pool.map(analyze_data, [
            (f'word_counts_11111.csv_2_{i}.csv', count_howmany, i) for i in range(num_files)
        ])
        pool.close()
        pool.join()

        print(f"Total processed: {count_howmany.value}")

if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    process_in_parallel(20)
