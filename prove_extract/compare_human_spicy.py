import os
import pandas as pd
import spacy
import re

def clean_word(word):
    return re.sub(r'[^\w\s]', '', word).strip().lower()

nlp = spacy.load('en_core_web_sm')

file_path = 'human_extracts_words.csv'  
df = pd.read_csv(file_path)

output_folder = 'output_folder_spacy_ner' 
os.makedirs(output_folder, exist_ok=True)

ner_missed = 0
human_missed_ratio = []
spacy_missed_ratio = []
precision_list = []
recall_list = []
f1_list = []

spacy_extracted_words = []

for index, row in df.iterrows():
    title = row['formatted_title']

    human_words_str = str(row['Human extract words']) if pd.notnull(row['Human extract words']) else ''
    human_words = set(clean_word(phrase) for phrase in human_words_str.split(','))  

    doc = nlp(title)
    extracted_entities = {clean_word(ent.text) for ent in doc.ents}  

    spacy_extracted_words.append(", ".join(extracted_entities))

    spacy_only = extracted_entities - human_words
    spacy_missed_ratio.append(len(spacy_only) / len(extracted_entities) if len(extracted_entities) > 0 else 0)

    missed_human_words = human_words - extracted_entities
    human_missed_ratio.append(len(missed_human_words) / len(human_words) if len(human_words) > 0 else 0)

    true_positives = len(extracted_entities & human_words)
    total_spacy_words = len(extracted_entities)
    total_human_words = len(human_words)
    
    precision = true_positives / total_spacy_words if total_spacy_words > 0 else 0
    recall = true_positives / total_human_words if total_human_words > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1_score)

    ner_missed += len(spacy_only)

total_spacy_missed_ratio = sum(spacy_missed_ratio) / len(spacy_missed_ratio) if spacy_missed_ratio else 0
total_human_missed_ratio = sum(human_missed_ratio) / len(human_missed_ratio) if human_missed_ratio else 0
average_precision = sum(precision_list) / len(precision_list) if precision_list else 0
average_recall = sum(recall_list) / len(recall_list) if recall_list else 0
average_f1_score = sum(f1_list) / len(f1_list) if f1_list else 0

output_file_path = os.path.join(output_folder, 'spacy_extracted_words_cleaned.csv')
df['SpaCy Extracted Words'] = spacy_extracted_words
df.to_csv(output_file_path, index=False)

print(f"Number of phrases extracted by SciBERT but not by humans: {ner_missed}")
print(f"Average proportion of phrases extracted by SciBERT but not by humans: {total_scibert_missed_ratio:.2%}")
print(f"Average proportion of phrases extracted by humans but not by SciBERT: {total_human_missed_ratio:.2%}")
print(f"Average Precision: {average_precision:.2%}")
print(f"Average Recall: {average_recall:.2%}")
print(f"Average F1-score: {average_f1_score:.2%}")
print(f"Phrases extracted by SciBERT have been saved to: {output_file_path}")
