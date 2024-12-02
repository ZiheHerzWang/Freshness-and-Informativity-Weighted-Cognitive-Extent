#%%
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from numpy.polynomial.polynomial import Polynomial
# Load the dataset
df = pd.read_csv(f'FICE_prove_2015_paper/2015_paper_FICE_score.csv')

# First, let's sort the dataset by 'Citations' in ascending order
df_sorted = df.sort_values(by='citations', ascending=True)
M = 500

# Now, we need to group the sorted data into bins of size M, or as close as possible
bins = [df_sorted[i:i + int(M)] for i in range(0, len(df_sorted), int(M))]

# If the last bin has fewer than M/3 papers, we will re-bin it with the (K-1)th bin
if len(bins) > 1 and len(bins[-1]) < M / 3:
    bins[-2] = pd.concat([bins[-2], bins[-1]])  # Combine the last two bins
    bins = bins[:-1]  # Remove the last bin


# For each bin, calculate the average C_5 (Citations in 5-year period) and plot the novelty scores
avg_c5 = []
avg_novelty = []
std_novelty = []

for bin_data in bins:
    avg_c5.append(bin_data['citations'].mean())
    avg_novelty.append(bin_data['FICE Score'].mean())
    std_novelty.append(bin_data['FICE Score'].std())

# Filter out entries where avg_c5 is zero and align avg_novelty and std_novelty accordingly
filtered_avg_c5 = [x for x in avg_c5 if x > 0]
filtered_avg_novelty = [avg_novelty[i] for i, x in enumerate(avg_c5) if x > 0]
filtered_std_novelty = [std_novelty[i] for i, x in enumerate(avg_c5) if x > 0]

# Apply natural log transformation to filtered avg_c5
log_avg_c5 = np.log(filtered_avg_c5)

degree = 1  # You can adjust the degree as needed
p = Polynomial.fit(log_avg_c5, filtered_avg_novelty, degree)
x_fit = np.linspace(min(log_avg_c5), max(log_avg_c5), 100)
y_fit = p(x_fit)

plt.rcParams.update({'font.size': 20})
correlation, p_value = pearsonr(log_avg_c5, filtered_avg_novelty)
print(filtered_std_novelty)
# Plot novelty scores with error bars
plt.figure(figsize=(10, 7))
plt.errorbar(log_avg_c5, filtered_avg_novelty, yerr=filtered_std_novelty, fmt='o', color='b', ecolor='r', capsize=5, label='Novelty Scores')
# plt.plot(x_fit, y_fit, color='g', linestyle='--', linewidth=2, label=f'{degree}-degree Polynomial Fit')

# Labels, grid, and title
plt.xlabel(r'$C_5$ (Log of Average Citations in 5-Year Period)')
plt.ylabel('average FICE')
plt.grid(False)

bin_size = int(M)  # Set bin size to match your calculation for M
plt.text(0.05, 0.95, f'Bin size = {M}\nCorrelation coefficient = {correlation:.3f} (p<0.001)',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Show the plot and print correlation
plt.show()
print(f"correlation is: {correlation}")
print(f"p_value is: {p_value}")