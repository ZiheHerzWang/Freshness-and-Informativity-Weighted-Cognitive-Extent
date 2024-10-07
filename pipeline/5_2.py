import pandas as pd

# Sample data loading (you can replace this with the actual file path)
file_path = 'newAreaCalculation_no_negative_filter_2.csv'  # Assuming this is the file name provided

# Load the file into a DataFrame
df = pd.read_csv(file_path)

# Check for negative Area Ratios
negative_area_ratios = df[df['Area Ratio'] < 0]
positive_area_ratios = df[df['Area Ratio'] > 1]

# If there are negative values, print them
if not negative_area_ratios.empty:
    print("Negative Area Ratios found:")
    print(negative_area_ratios)
else:
    print("No negative Area Ratios found.")


if not positive_area_ratios.empty:
    print("positive_area_ratios found:")
    print(positive_area_ratios)
else:
    print("No positive_area_ratios found.")