import pandas as pd
import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import multiprocessing as mp

def calculate_adjusted_cutoff(counts):
    # Calculate the standard deviation
    std_dev = np.std(counts)
    # Normalize the standard deviation to be in the range of 0.08 to 0.16
    normalized_std_dev = (std_dev - np.min(counts)) / (np.max(counts) - np.min(counts))
    # Scale the normalized std_dev to the desired range (0.08 to 0.16)
    cutoff_frequency = 0.08 + normalized_std_dev * 0.13
    return cutoff_frequency

def smooth_signal_with_fft(y, cutoff_frequency):
    yf = rfft(y)
    xf = rfftfreq(len(y), 1)  

    yf[xf > cutoff_frequency] = 0

    return irfft(yf, n=len(y))

def count_papers(data, unique_word):
    # Removed the category argument and filtering
    min_year, max_year = np.inf, -np.inf
    counts = {}

    for row in data.itertuples(index=False):
        year = row.Year  # Ensuring that 'Year' is used correctly after conversion
        words = str(row[data.columns.get_loc('Important Words')]).lower()  # Accessing by column index
        word_count = words.count(unique_word)

        if word_count > 0:
            min_year = min(min_year, year)
            max_year = max(max_year, year)
            counts[year] = counts.get(year, 0) + word_count  

    return min_year, max_year, counts


def analyze_data(args):
    filename, count_how_many = args
    unique_words_data = pd.read_csv(filename, encoding='utf-8')
    data = pd.read_csv('originMethod\new_output_gpt4_all.csv')

    data['Year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)  # Convert year to int
    data['Important Words'] = data['Important Words'].fillna("")

    result_list = []

    for unique_word in unique_words_data['Unique_Word'].str.lower().unique():
        if pd.isna(unique_word) or not isinstance(unique_word, str):
            continue  

        print(unique_word + ":")
        min_year, max_year, counts = count_papers(data, unique_word)
        print(min_year, max_year, counts)
        if min_year == np.inf or max_year == -np.inf:
            continue

        years = list(range(min_year, max_year + 1))
        counts_list = [counts.get(year, 0) for year in years]

        x = np.array(years)
        y = np.array(counts_list)

        if len(y) < 2:  # Ensure there are enough data points
            print(f"Not enough data points for {unique_word}")
            continue

        cutoff_frequency = 0.125
        y_smooth = smooth_signal_with_fft(y, cutoff_frequency)
        if len(y_smooth) >= 2:  # Check again after smoothing
            slope_smooth = np.gradient(y_smooth)

            for year, count, slope_val in zip(years, counts_list, slope_smooth):
                result_list.append({
                    'Unique Word': unique_word,
                    'Year': year,
                    'Count': count,
                    'Slope': slope_val
                })

        count_how_many.value += 1
        print(count_how_many.value)

    result = pd.DataFrame(result_list)
    result.to_csv(f'originMethod\word_counts_and_slopes_{filename}.csv', index=False, encoding='utf-8')

# The rest of your code remains unchanged

def process_in_parallel(num_files):
    with mp.Manager() as manager:
        count_how_many = manager.Value('i', 0)
        pool = mp.Pool(mp.cpu_count())
        pool.map(analyze_data, [(f'originMethod\word_counts_11111.csv_2_{i}.csv', count_how_many) for i in range(num_files)])
        pool.close()
        pool.join()
        print('Total number of unique words processed:', count_how_many.value)

if __name__ == '__main__':
    process_in_parallel(20)
