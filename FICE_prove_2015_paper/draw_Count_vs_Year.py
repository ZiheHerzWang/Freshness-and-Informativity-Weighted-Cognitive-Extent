#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial

# Load the CSV file
file_path = rf"FICE_Area_Preparation/csv file/all_gpt_extract.csv"
df = pd.read_csv(file_path)

# Prepare data for unique entities count per year
df_sorted = df.sort_values(by='Year')
years = df_sorted['Year'].unique()

# Data structures for storing results
entity_counts_per_year_importants = []
entity_counts_per_year_cluster = []
first_time_entity_counts_per_year_importants = []
first_time_entity_counts_per_year_cluster = []

all_seen_entities_importants = set()
all_seen_entities_cluster = set()

# Count the number of papers per year
paper_counts_per_year = df_sorted['Year'].value_counts().sort_index()

# Loop through each year and count entities
for year in years:
    year_data = df_sorted[df_sorted['Year'] == year]

    # For "Scientific Entity"
    entities_importants = ",".join(year_data['Scientific Entity']).split(',')
    unique_entities_importants = set(entities_importants)
    entity_counts_per_year_importants.append((year, len(unique_entities_importants)))
    first_time_entities_importants = unique_entities_importants - all_seen_entities_importants
    first_time_entity_counts_per_year_importants.append((year, len(first_time_entities_importants)))
    all_seen_entities_importants.update(unique_entities_importants)

    # For "Scientific Entity Disambigious"
    entities_cluster = ",".join(year_data['Scientific Entity Disambigious']).split(',')
    unique_entities_cluster = set(entities_cluster)
    entity_counts_per_year_cluster.append((year, len(unique_entities_cluster)))
    first_time_entities_cluster = unique_entities_cluster - all_seen_entities_cluster
    first_time_entity_counts_per_year_cluster.append((year, len(first_time_entities_cluster)))
    all_seen_entities_cluster.update(unique_entities_cluster)

# Convert results to DataFrames for plotting
entity_counts_df_importants = pd.DataFrame(entity_counts_per_year_importants, columns=['Year', 'Unique Entities Count'])
entity_counts_df_cluster = pd.DataFrame(entity_counts_per_year_cluster, columns=['Year', 'Unique Entities Count'])
first_time_entity_counts_df_importants = pd.DataFrame(first_time_entity_counts_per_year_importants, columns=['Year', 'First Time Entities Count'])
first_time_entity_counts_df_cluster = pd.DataFrame(first_time_entity_counts_per_year_cluster, columns=['Year', 'First Time Entities Count'])

plt.rcParams.update({'font.size': 20})

# Plot the number of papers per year (bar plot) with the adjusted x-axis limits
plt.figure(figsize=(14, 7))
plt.bar(paper_counts_per_year.index[:-3], paper_counts_per_year.values[:-3], color='darkgray', label='Number of Papers')

# Fit a polynomial to the number of papers per year data
degree = 3  # Degree of the polynomial
x_values = paper_counts_per_year.index.values[:-3]  # Years
y_values = paper_counts_per_year.values[:-3]        # Number of Papers
p = Polynomial.fit(x_values, y_values, degree)

# Generate x values for the fitted line and evaluate the polynomial at these points
x_fit = np.linspace(x_values.min(), x_values.max(), 100)
y_fit = p(x_fit)

# Plot the polynomial fitting line
plt.plot(x_fit, y_fit, color='orange', linestyle='--', linewidth=2, label=f'{degree}-degree Polynomial Fit')

# Overlay the total unique entities count per year (line plots)
plt.plot(entity_counts_df_importants['Year'][:-3], entity_counts_df_importants['Unique Entities Count'][:-3], marker='o', linestyle='-', color='blue', label='Un-disambiguated (Scientific Entity)')
plt.plot(entity_counts_df_cluster['Year'][:-3], entity_counts_df_cluster['Unique Entities Count'][:-3], marker='o', linestyle='-', color='green', label='Disambiguated (Scientific Entity)')

# Plot the first-time unique entities count for disambiguated scientific entities
plt.plot(first_time_entity_counts_df_cluster['Year'][:-3], first_time_entity_counts_df_cluster['First Time Entities Count'][:-3], marker='o', linestyle='-', color='red', label='First Time Appearence Disambiguated (Scientific Entity)')

# Labels and Legend
plt.xlabel('Year')
plt.ylabel('Count')
plt.grid(False)
plt.legend()
plt.ylim(bottom=0)
# Adjust x-axis limits to remove space between start point and box line
plt.xlim(x_values.min(), x_values.max())
plt.show()