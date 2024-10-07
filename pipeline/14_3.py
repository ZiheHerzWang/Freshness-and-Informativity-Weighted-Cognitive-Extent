import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# Load the dataset
df = pd.read_csv('different_try_of_area_calculation/newAreaCalculation_by_length_updated.csv')
# df = pd.read_csv('new_novelty_calculation_by_first_occurrence.csv')


# Filter data to include only the 1% to 99% percentiles for citations
lower_bound = df['Citations'].quantile(0.01)
upper_bound = df['Citations'].quantile(0.99)

df_filtered = df[(df['Citations'] >= lower_bound) & (df['Citations'] <= upper_bound)]

# Enlarge the font size globally to be 2x bigger
plt.rcParams.update({'font.size': 10})

# Plot citation vs novelty for the filtered data (no log transform)
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['Citations'], df_filtered['Novelty Score'], alpha=0.7, label='Paper')

# Add vertical lines for citation quantile boundaries
plt.axvline(x=7.0, color='red', linestyle='--', label=r'50th Percentile ($7.0$ Citations)')  
plt.axvline(x=26.0, color='blue', linestyle='--', label=r'80th Percentile ($26.0$ Citations)')
plt.axvline(x=95.8, color='green', linestyle='--', label=r'95th Percentile ($95.0$ Citations)')

# Linear fitting
coefficients = np.polyfit(df_filtered['Citations'], df_filtered['Novelty Score'], 1)
linear_fit = np.poly1d(coefficients)

# Generate x values for the fitted line
x_values = np.linspace(df_filtered['Citations'].min(), df_filtered['Citations'].max(), 100)
y_values = linear_fit(x_values)

# Plot the linear fit line
plt.plot(x_values, y_values, color='orange', linestyle='-', label='Linear Fit')

# Labels, and grid (no title as requested)
plt.xlabel(r'$C_5$ (Citations in 5-Year Period)')
plt.ylabel('Novelty Score')
plt.grid(True)

# Move the legend to the upper right inside the plot
plt.legend(loc='upper right', prop={'size': 8}, ncol=2)  # Adjust font size and use 2 columns


# Save the plot
plt.savefig('citations_vs_novelty_filtered_with_boundaries_and_linear_fit_updated.png', bbox_inches='tight')
plt.show()

# Calculate Pearson correlation coefficient and p-value for original citations
correlation_coefficient, p_value = pearsonr(df_filtered['Citations'], df_filtered['Novelty Score'])

# Output the results
print("Correlation coefficient (filtered): ", correlation_coefficient)
print("P-value (filtered): ", p_value)

# Find and print titles where the novelty score is below 0
titles_below_zero = df_filtered[df_filtered['Novelty Score'] < 0]['Title']

if titles_below_zero.empty:
    print("No titles found with a novelty score below 0.")
else:
    print("Titles with a novelty score below 0:")
    for title in titles_below_zero:
        print(title)
