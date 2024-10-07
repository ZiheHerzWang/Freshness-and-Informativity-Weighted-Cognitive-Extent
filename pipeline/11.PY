import csv
import re
from keyphrase_vectorizers import KeyphraseCountVectorizer

# Initialize the vectorizer with English stop words
vectorizer = KeyphraseCountVectorizer(stop_words='english')

# Function to check if the string contains any alphanumeric characters
def contains_valid_content(s):
    return any(c.isalnum() for c in s)

# Function to clean and format title
def clean_title(title):
    # Remove curly braces and replace escaped single quotes
    cleaned_title = re.sub(r'{|}', '', title)
    return cleaned_title.title()

data_list = []

# Open and read the CSV file, processing each row
with open('output_results.csv', 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.DictReader(f)
    for row in reader:
        title = row['title']
        year = row['year']
        citations = row['citations']
        print(title)
        # Extract important words if the title has valid content
        if title and contains_valid_content(title):
            title = clean_title(title)
            print(title)
            try:
                title_vectorized = vectorizer.fit_transform([title])
                important_words = ', '.join(vectorizer.get_feature_names_out())
            except Exception as e:
                print(f"Error occurred: {e}")
                important_words = ''
        else:
            important_words = ''

        # Append the cleaned title, important words, year, and citations to the list
        data_list.append([title, year, important_words, citations])

# Write the processed data to a new CSV file
output_file_path = 'processed_output_with_important_words.csv'
with open(output_file_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Title', 'Year', 'Important Words', 'Citations'])  # Define column titles
    writer.writerows(data_list)

print(f"Processing complete. Data written to {output_file_path}")
