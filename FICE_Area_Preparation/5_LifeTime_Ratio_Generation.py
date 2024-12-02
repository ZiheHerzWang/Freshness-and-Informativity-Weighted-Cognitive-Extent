#%%
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


output_dir = f"C:/Users/zihe0/Desktop/try/Novelty-Life-Time/FICE_Area_Preparation/parallel/area_csv_file"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_total_number =  20 # Modify it to your cpu  thread
print(device)
#%%
# Define the Gaussian model
class GaussianModel(torch.nn.Module):
    def __init__(self, n_peaks):
        super(GaussianModel, self).__init__()
        self.amplitude = torch.nn.Parameter(torch.randn(n_peaks, device=device))
        self.center = torch.nn.Parameter(torch.randn(n_peaks, device=device))
        self.width = torch.nn.Parameter(torch.abs(torch.randn(n_peaks, device=device)))  # Ensure width is positive
        self.n_peaks_param = torch.nn.Parameter(torch.tensor(float(n_peaks), device=device))  # Make n_peaks differentiable

    def forward(self, x):
        y = torch.zeros_like(x, device=device)
        # Ensure n_peaks is between 1 and 5 and rounded to an integer
        n_peaks_rounded = torch.round(torch.clamp(self.n_peaks_param, 1, 5)).int()
        for i in range(n_peaks_rounded):
            y += self.amplitude[i] * torch.exp(-((x - self.center[i]) ** 2) / (2 * self.width[i] ** 2))
        return y

# Automatically detect the number of peaks
def detect_number_of_peaks(y_data):
    peaks, _ = find_peaks(y_data, distance=5)
    return max(1, len(peaks))  # Ensure at least one peak is returned

# Generate the initial guess for the Gaussian model
def generate_initial_guess(x, y, n_peaks):
    distance = max(1, len(x) // n_peaks)
    peaks, _ = find_peaks(y, distance=distance)

    if len(peaks) < n_peaks:
        peaks = np.append(peaks, np.linspace(0, len(x) - 1, n_peaks - len(peaks), dtype=int))

    initial_amplitude = np.random.uniform(5.0, 10.0, size=n_peaks)
    initial_center = x[peaks]
    initial_width = np.std(x) / n_peaks

    return initial_amplitude, initial_center, initial_width

# Train the Gaussian model
def train_gaussian(x_data, y_data, n_peaks, lr=1e-2, epochs=1, reg_lambda=0.001, total_count=1, early_stop_threshold=0.4):
    x = torch.tensor(x_data, dtype=torch.float32, device=device)
    y_true = torch.tensor(y_data, dtype=torch.float32, device=device)

    # Initialize the Gaussian model
    amplitude, center, width = generate_initial_guess(x_data, y_data, n_peaks)
    model = GaussianModel(n_peaks).to(device)
    
    # Use initial guesses to initialize parameters
    with torch.no_grad():
        model.amplitude.copy_(torch.tensor(amplitude, device=device))
        model.center.copy_(torch.tensor(center, device=device))
        model.width.copy_(torch.tensor(width, device=device))

    # Loss function
    criterion = torch.nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop with early stopping
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        mse_loss = criterion(y_pred, y_true)

        # Add regularization term to limit the width and amplitude to prevent overfitting
        reg_term = reg_lambda * (torch.sum(torch.exp(model.width)) + torch.sum(torch.abs(model.amplitude))) * int(n_peaks)
        loss = mse_loss + reg_term

        # Early stopping condition
        if loss.item() < early_stop_threshold:
            print(f'Early stopping at epoch {epoch} with loss: {loss.item()}')
            break

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

    # Retrieve the final values of the parameters
    final_amplitude = model.amplitude.detach().cpu().numpy()
    final_center = model.center.detach().cpu().numpy()
    final_width = model.width.detach().cpu().numpy()

    # Return model, final loss, and the parameters as a tuple
    return model, loss.item(), final_amplitude, final_center, final_width


# Save the fitting plot
def save_fitting_plot(years, counts_list, extended_years, fitted_curve, unique_word, process_id, final_amplitude, final_center, final_width, n_peak):
    # Calculate area ratios for filtering
    # print(fitted_curve)
    # print(f"fitting_{sanitized_word}_{final_amplitude}_{final_center}_{final_width}_{n_peak}")
    area_ratios = [calculate_area_ratio(extended_years, fitted_curve, year) for year in extended_years]

    # Filter data to include only area ratios between 0 and 1
    filtered_years = []
    filtered_fitted_curve = []

    for year, value, ratio in zip(extended_years, fitted_curve, area_ratios):
        if 0 <= ratio <= 1:
            filtered_years.append(year)
            filtered_fitted_curve.append(value)

    filtered_years = [year for year in filtered_years if year <= 2040]
    filtered_fitted_curve = [value for year, value in zip(filtered_years, filtered_fitted_curve) if year <= 2040]

    plt.figure(figsize=(10, 6))

    # Plot ground truth and fitted Gaussian within area ratio between 0 and 1
    plt.bar(years, counts_list, alpha=0.5, label='Ground Truth')
    plt.plot(filtered_years, filtered_fitted_curve, 'r-', label='Fitted Gaussian')

    # Set labels and legend
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()

    # Sanitize filename and save the plot
    sanitized_word = re.sub(r'[<>:"/\\|?*]', '', unique_word)
    process_output_dir = os.path.join(output_dir, f'process_{process_id}')
    os.makedirs(process_output_dir, exist_ok=True)
    filename = os.path.join(process_output_dir, f"fitting_{sanitized_word}_{final_amplitude}_{final_center}_{final_width}_{n_peak}.png")

    try:
        plt.savefig(filename)
    except Exception as e:
        print(f"Unable to save image '{unique_word}': {e}")

    # Close the figure to free memory
    plt.close()

# Calculate the area ratio
def calculate_area_ratio(extended_years, fitted_curve, current_year):
    total_area = simpson(y=fitted_curve, x=extended_years)
    valid_years_idx = extended_years <= current_year
    valid_years = extended_years[valid_years_idx]
    valid_fitted_curve = fitted_curve[valid_years_idx]
    
    # Calculate the area up to the current year
    area_up_to_current = simpson(y=valid_fitted_curve, x=valid_years)
    
    if total_area > 0:
        return area_up_to_current / total_area
    return 0

# Save the results to CSV and stop calculation after area ratio reaches 1
def save_results_extended(filename, unique_word, years, counts_list, extended_years, fitted_curve):
    results = []
    ground_truth_dict = dict(zip(years, counts_list))

    for year, fitted_value in zip(extended_years, fitted_curve):
        area_ratio = calculate_area_ratio(extended_years, fitted_curve, year)

        # Cap the area ratio at 1
        area_ratio = min(area_ratio, 1)

        ground_truth_value = ground_truth_dict.get(year, 0)
        results.append([unique_word, int(year), ground_truth_value, area_ratio])

    # Save results to CSV
    df = pd.DataFrame(results, columns=['Unique Word', 'Year', 'Count', 'Area Ratio'])
    if not os.path.isfile(filename):
        df.to_csv(filename, mode='w', header=True, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

# Count papers containing the specific word
def count_papers(data, unique_word):
    counts = {}
    unique_word = unique_word.lower()  # Ensure `unique_word` is lowercase for consistent matching
    
    for row in data.itertuples(index=False):
        year = row.Year
        # Get the `Scientific Entity Disambigious` column and split by commas, convert to lowercase
        words = str(row[data.columns.get_loc('Scientific Entity Disambigious')]).lower().split(',')
        
        # Strip whitespace and check for exact match with `unique_word`
        word_count = sum(1 for word in words if word.strip() == unique_word)
        
        if word_count > 0:
            counts[year] = counts.get(year, 0) + word_count
            
    return counts


# Analyze the data and perform Gaussian fitting
def analyze_data(args):
    os.makedirs(output_dir, exist_ok=True)

    filename, count_howmany, process_id = args
    unique_words_data = pd.read_csv(filename, encoding='utf-8')
    data = pd.read_csv(f'C:/Users/zihe0/Desktop/try/Novelty-Life-Time/FICE_Area_Preparation/all_gpt_extract.csv')

    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    data['Scientific Entity Disambigious'] = data['Scientific Entity Disambigious'].fillna("")

    result_filename = os.path.join(output_dir, f"word_counts_fits_process_new_most_new_{process_id}.csv")

    # Initialize cumulative loss and word count
    total_loss = 0
    total_word_count = 0

    for unique_word in unique_words_data['Unique_Word'].str.lower().unique():
        if pd.isna(unique_word) or not isinstance(unique_word, str):
            continue

        counts = count_papers(data, unique_word)
        years = np.array(sorted(counts.keys()), dtype=np.float64)
        counts_list = np.array([counts.get(year, 0) for year in years])

        # if len(counts_list) < 3:
        #     continue

        # Dynamically detect number of peaks
        n_peaks = detect_number_of_peaks(counts_list)

        # Compute total occurrences of the word
        total_count = np.sum(counts_list)

        # Dynamically determine epochs based on the total occurrences
        epochs = max(total_count, 500)
        epochs = epochs * 10
        
        # Perform fitting using the custom Gaussian model
        model, loss, final_amplitude, final_center, final_width = train_gaussian(
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
        n_peaks_int = int(torch.round(model.n_peaks_param).item())
        # print(f"n_peaks_int is {n_peaks_int}")
        # Save fitting results and images
        save_fitting_plot(years, counts_list, extended_years, fitted_curve, unique_word, process_id, final_amplitude, final_center, final_width, n_peaks_int)
        save_results_extended(result_filename, unique_word, years, counts_list, extended_years, fitted_curve)

        # Accumulate total loss and word count
        total_loss += loss / total_count  # Weight loss by occurrences
        total_word_count += total_count

        count_howmany.value += 1
        print(f"Total processed so far: {count_howmany.value}")

    # Return average loss across all words
    average_loss = total_loss / total_word_count if total_word_count > 0 else 0
    print(f"Average Loss: {average_loss}")

# Parallel processing of multiple files
def process_in_parallel(num_files):
    # Use 'spawn' as the start method for multiprocessing to handle CUDA
    mp.set_start_method('spawn', force=True)
    
    with mp.Manager() as manager:
        count_howmany = manager.Value('i', 0)
        pool = mp.Pool(mp.cpu_count())
        pool.map(analyze_data, [
            (f'C:/Users/zihe0/Desktop/try/Novelty-Life-Time/FICE_Area_Preparation/parallel/Unique_word_with_count_{i}.csv', count_howmany, i) for i in range(num_files)
        ])
        pool.close()
        pool.join()

        print(f"Total processed: {count_howmany.value}")

if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    process_in_parallel(device_total_number)