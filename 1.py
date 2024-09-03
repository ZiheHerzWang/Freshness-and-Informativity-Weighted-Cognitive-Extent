import csv
import re
import nltk
from keyphrase_vectorizers import KeyphraseCountVectorizer

# Download necessary NLTK resources
nltk.download('stopwords')

# Regex pattern to extract title and year from the given text
title_pattern = r'title\s*=\s*\"(.*?)\"'
year_pattern = r'year\s*=\s*\"(\d{4})\"'

# Initialize the vectorizer
vectorizer = KeyphraseCountVectorizer(stop_words='english')

# Function to clean and format title
def format_title(title):
    # Remove curly braces
    cleaned_title = re.sub(r'{|}', '', title)
    # Title case conversion
    return cleaned_title.title()

# Function to check for valid content
def contains_valid_content(s):
    return any(c.isalnum() for c in s)

# Process the CSV file
def process_csv(input_file, output_file):
    data_list = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            category, text = row[0], row[1]
            title_matches = re.findall(title_pattern, text)
            year_matches = re.findall(year_pattern, text)
            title = title_matches[0] if title_matches else None
            year = year_matches[0] if year_matches else None

            if title and contains_valid_content(title):
                title = format_title(title)  # Clean and format the title
                print(title)
                try:
                    # Vectorize the title and extract keywords
                    transformed_title = vectorizer.fit_transform([title])
                    keywords = ', '.join(vectorizer.get_feature_names_out())
                except Exception as e:
                    print(f"Error occurred: {e}")
                    keywords = ''
                data_list.append([category, year, keywords])
            else:
                data_list.append([category, year, ''])

    # Write the processed data to a new CSV file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Year', 'Important Words'])
        writer.writerows(data_list)

# Example usage
process_csv('output.csv', 'output_cleaned.csv')
