#%%
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd

# Load the CSV file from the specified path
file_path = "FICE_Area_Preparation/all_gpt_extract.csv"
df = pd.read_csv(file_path)

# Function to count unique entities in a bin
def count_unique_entities(entities):
    return len(set(entities))

merged_file = pd.read_csv(f"FICE_Area_Preparation/parallel/area_csv_file/merged_area_file.csv")

# Prepare data for plotting
df_sorted = df.sort_values(by='Year')
years = df_sorted['Year'].unique()
binned_cognitive_extent_entities_125 = []
binned_cognitive_extent_entities_250 = []
binned_cognitive_extent_entities_500 = []
binned_cognitive_extent_words = []

# Loop through each year for bin sizes 250 and 500
for year in years:
    # Filter the data for the specific year
    year_data = df_sorted[df_sorted['Year'] == year]

    # Bin size of 125
    bin_size_125 = 125
    bin_size_125 = int(np.floor(len(year_data) / bin_size_125))
    bins_125 = [year_data.iloc[i*bin_size_125: (i+1)*bin_size_125] for i in range(bin_size_125)]
    
    # Bin size of 250
    bin_size_250 = 250
    num_bins_250 = int(np.floor(len(year_data) / bin_size_250))
    bins_250 = [year_data.iloc[i*bin_size_250: (i+1)*bin_size_250] for i in range(num_bins_250)]
    
    # Bin size of 500
    bin_size_500 = 500
    num_bins_500 = int(np.floor(len(year_data) / bin_size_500))
    bins_500 = [year_data.iloc[i*bin_size_500: (i+1)*bin_size_500] for i in range(num_bins_500)]

    # Calculate cognitive extent for each bin (Entities) for bin size 125
    for bin_data in bins_125:
        entities = ",".join(bin_data['Scientific Entity Disambigious']).split(',')
        # print(entities)
        unique_count_entities = count_unique_entities(entities)
        binned_cognitive_extent_entities_125.append((year, unique_count_entities))
    
    # Calculate cognitive extent for each bin (Entities) for bin size 250
    for bin_data in bins_250:
        entities = ",".join(bin_data['Scientific Entity Disambigious']).split(',')
        unique_count_entities = count_unique_entities(entities)
        binned_cognitive_extent_entities_250.append((year, unique_count_entities))
    
    # Calculate cognitive extent for each bin (Entities) for bin size 500
    for bin_data in bins_500:
        entities = ",".join(bin_data['Scientific Entity Disambigious']).split(',')
        unique_count_entities = count_unique_entities(entities)
        binned_cognitive_extent_entities_500.append((year, unique_count_entities))

    # Calculate cognitive extent for each bin (Words)
    for bin_data in bins_250:
        entities = ",".join(bin_data['Scientific Entity']).split(',')
        unique_count_words = count_unique_entities(entities)
        binned_cognitive_extent_words.append((year, unique_count_words))

# Convert results to DataFrames for plotting
binned_entities_df_125 = pd.DataFrame(binned_cognitive_extent_entities_125, columns=['Year', 'Cognitive Extent'])
binned_entities_df_250 = pd.DataFrame(binned_cognitive_extent_entities_250, columns=['Year', 'Cognitive Extent'])
binned_entities_df_500 = pd.DataFrame(binned_cognitive_extent_entities_500, columns=['Year', 'Cognitive Extent'])
binned_words_df = pd.DataFrame(binned_cognitive_extent_words, columns=['Year', 'Cognitive Extent'])

# Fit polynomial regression models (degree 3 for a smooth curve)
p_entities_125 = Polynomial.fit(binned_entities_df_125['Year'].values, binned_entities_df_125['Cognitive Extent'].values, deg=3)
p_entities_250 = Polynomial.fit(binned_entities_df_250['Year'].values, binned_entities_df_250['Cognitive Extent'].values, deg=3)
p_entities_500 = Polynomial.fit(binned_entities_df_500['Year'].values, binned_entities_df_500['Cognitive Extent'].values, deg=3)
p_words = Polynomial.fit(binned_words_df['Year'].values, binned_words_df['Cognitive Extent'].values, deg=3)

# Generate a range of years to plot the smooth curves
x_range = np.linspace(binned_entities_df_250['Year'].min(), binned_entities_df_250['Year'].max(), 300)
x_range_125 = np.linspace(binned_entities_df_125['Year'].min(), binned_entities_df_125['Year'].max(), 300)
x_range_500 = np.linspace(binned_entities_df_500['Year'].min(), binned_entities_df_500['Year'].max(), 300)
y_pred_entities_125 = p_entities_125(x_range_125)
y_pred_entities_250 = p_entities_250(x_range)
y_pred_entities_500 = p_entities_500(x_range_500)
y_pred_words = p_words(x_range)

# Plot the combined diagram
plt.rcParams.update({'font.size': 20})  
plt.figure(figsize=(10, 6))

# Disambiguated (dot plot + smooth line for bin size 125)
plt.plot(binned_entities_df_125['Year'], binned_entities_df_125['Cognitive Extent'], 
         marker='s', linestyle='None', markersize=4, color='black', zorder=1, label='Disambiguated (Dots, bin=125)')
plt.plot(x_range_125, y_pred_entities_125, color='orange', linestyle='-', linewidth=2, zorder=2, label='Disambiguated (Line, bin=125)')

# Disambiguated (dot plot + smooth line for bin size 250)
plt.plot(binned_entities_df_250['Year'], binned_entities_df_250['Cognitive Extent'], 
         marker='o', linestyle='None', markersize=4, color='black', zorder=1, label='Disambiguated (Dots, bin=500)')
plt.plot(x_range, y_pred_entities_250, color='blue', linestyle='-', linewidth=2, zorder=2, label='Disambiguated (Line, bin=500)')

# Disambiguated (triangle plot + smooth line for bin size 500)
plt.plot(binned_entities_df_500['Year'], binned_entities_df_500['Cognitive Extent'], 
         marker='^', linestyle='None', markersize=4, color='black', zorder=1, label='Disambiguated (Triangles, bin=250)')
plt.plot(x_range_500, y_pred_entities_500, color='green', linestyle='-', linewidth=2, zorder=2, label='Disambiguated (Line, bin=250)')

# Not Disambiguated (smooth line)
plt.plot(x_range, y_pred_words, color='red', linestyle='--', linewidth=2, zorder=1, label='Un-disambiguated')

plt.xlabel('Year')
plt.ylabel('Number of Unique Entities per Bin')
plt.grid(False)
plt.legend(fontsize=9)
plt.show()