import pandas as pd
import re

# File paths
references_file = "FICE_prove_2015_paper/references_with_citations.csv"
gpt_extract_file = "FICE_Area_Preparation/csv file/all_gpt_extract.csv"
output_file = "FICE_prove_2015_paper/2015_scientific_entity.csv"

# Function to format titles
def format_title(title):
    if not isinstance(title, str):  # Check if the title is not a string
        return ""
    cleaned_title = re.sub(r'{|}', '', title)  # Remove curly braces
    return cleaned_title.title()  # Convert to title case

# Read the CSV files
references_df = pd.read_csv(references_file)
gpt_extract_df = pd.read_csv(gpt_extract_file)

# Ensure the required column names exist
if "Title" not in references_df.columns or "Title" not in gpt_extract_df.columns:
    raise ValueError("The column 'Title' is missing in one of the datasets.")

# Convert all titles to strings and apply the formatting function
references_df["Formatted_Title"] = references_df["Title"].astype(str).apply(format_title)
gpt_extract_df["Formatted_Title"] = gpt_extract_df["Title"].astype(str).apply(format_title)

# Create a dictionary for exact Formatted_Title lookups
title_to_entity = dict(zip(gpt_extract_df["Formatted_Title"], gpt_extract_df["Scientific Entity Disambigious"]))

# Add the Scientific Entity Disambigious column by looking up each formatted title in the dictionary
references_df["Scientific Entity Disambigious"] = references_df["Formatted_Title"].map(title_to_entity)

# Extract required columns
result_df = references_df[["Title", "Year", "citations", "Scientific Entity Disambigious"]]

# Save the results to a CSV file
result_df.to_csv(output_file, index=False)

print(f"Matching complete. Results saved to {output_file}.")
