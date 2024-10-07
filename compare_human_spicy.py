import pandas as pd
import spacy

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

# Read the CSV file
file_path = 'human_extracts_words.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Initialize counters
ner_missed = 0
human_missed_ratio = []

# Loop over each row in the dataframe
for index, row in df.iterrows():
    title = row['formatted_title']
    human_words = set(row['human_extract_words'].split())  # Assume space-separated human words

    # Use SpaCy to extract entities
    doc = nlp(title)
    extracted_entities = {ent.text for ent in doc.ents}

    # Calculate NER entities not found in human words
    ner_missed += len(extracted_entities - human_words)

    # Calculate the percentage of human words that were not extracted by NER
    missed_human_words = human_words - extracted_entities
    human_missed_ratio.append(len(missed_human_words) / len(human_words) if len(human_words) > 0 else 0)

# Final statistics
total_human_missed_ratio = sum(human_missed_ratio) / len(human_missed_ratio) if human_missed_ratio else 0

print(f"NER extracted entities not in human-extracted words: {ner_missed}")
print(f"Average proportion of human-extracted words missed by NER: {total_human_missed_ratio:.2%}")
