import pandas as pd

# Load your CSV file
file_path = 'newAreaCalculation_by_length.csv'
df = pd.read_csv(file_path)

sorted_citations = df['Citations'].sort_values().values

# Calculate the index for each threshold
index_50 = int(0.50 * (len(sorted_citations) - 1))  # 50th percentile index
index_80 = int(0.80 * (len(sorted_citations) - 1))  # 80th percentile index
index_95 = int(0.95 * (len(sorted_citations) - 1))  # 95th percentile index

# Get the citation counts at those indices
threshold_50 = sorted_citations[index_50]
threshold_80 = sorted_citations[index_80]
threshold_95 = sorted_citations[index_95]

# Print the thresholds for verification
print(f"50th Percentile (Q2 upper limit): {threshold_50}")
print(f"80th Percentile (Q3 upper limit): {threshold_80}")
print(f"95th Percentile (Q4 lower limit): {threshold_95}")

# Function to classify based on citation count with adjusted thresholds
def classify_by_citation_count(value):
    if value <= threshold_50:
        return "Q1"
    elif value <= threshold_80:
        return "Q2"
    elif value <= threshold_95:
        return "Q3"
    else:
        return "Q4"

# Apply classification based on adjusted thresholds
df['Quartile'] = df['Citations'].apply(classify_by_citation_count)

# Save the updated data with new Quartile information
output_file = 'different_try_of_area_calculation/newAreaCalculation_by_length_updated.csv'
df.to_csv(output_file, index=False)

print("Reclassification done with adjusted thresholds, and file saved.")
