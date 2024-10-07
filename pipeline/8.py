import csv
import re
import openai
import time

openai.api_key = "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"

def format_title(title):
    cleaned_title = re.sub(r'{|}', '', title)
    return cleaned_title.title()

def extract_scientific_entities(title, model="gpt-4", temperature=0):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Extract scientific entities directly from titles without additional text."},
                    {"role": "user", "content": f"Extract scientific entities: {title}"}
                ],
                temperature=temperature
            )
            text = response['choices'][0]['message']['content']

            # Cleaning up the text output
            cleaned_text = re.sub(r'\r?\n|\r', ', ', text)
            cleaned_text = re.sub(r',\s*-\s*', ', ', cleaned_text)
            cleaned_text = re.sub(r'^-\s*', '', cleaned_text)
            cleaned_text = re.sub(r'\s*,\s*', ', ', cleaned_text).strip(', ')
            cleaned_text = re.sub(r'\d+\.\s*', '', cleaned_text).lower()
            
            return cleaned_text
        except Exception as e:
            if 'rate limit' in str(e).lower():
                print("Rate limit hit. Waiting for 10 seconds before retrying...")
                time.sleep(1)  # Wait for 10 seconds to retry
            else:
                return str(e)

data_list = []

# Read the file and process each line
with open('output_results.csv', 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        original_title = row['title']
        year = row['year']
        citations = row['citations']

        # Clean and format the title
        cleaned_title = format_title(original_title)

        # Extract important words from the title
        if cleaned_title and any(c.isalnum() for c in cleaned_title):
            print(cleaned_title)
            try:
                important_words = extract_scientific_entities(cleaned_title)
            except Exception as e:
                print(f"Error occurred: {e}")
                important_words = ''
        else:
            important_words = ''
        print("important word: "+important_words)

        # Add the original title, important words, year, and citation counts to the list
        data_list.append([original_title, year, important_words, citations])

# Write the processed data to a new CSV file
output_file_path = 'processed_output_with_important_words.csv'
with open(output_file_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Title', 'Year', 'Important Words', 'Citations'])  # Define column titles
    writer.writerows(data_list)

print(f"Processing complete. Data written to {output_file_path}")
