import os
import pandas as pd
import re

def clean_phrase(phrase):
    return re.sub(r'[^\w\s]', '', phrase).strip()

important_words_file = 'random_200_english_rows.csv'  
human_words_file = 'human_extracts_words.csv'  

df_important = pd.read_csv(important_words_file)
df_human = pd.read_csv(human_words_file)

if 'Important Words' not in df_important.columns or 'Human extract words' not in df_human.columns:
    raise ValueError("CSV 文件中缺少 'Important Words' 或 'Human extract words' 列")

ner_missed = 0
human_missed_ratio = []
gpt_missed_ratio = []
precision_list = []
recall_list = []
f1_list = []

results = []

for index, row in df_important.iterrows():
    important_words_str = str(row['Important Words']) if pd.notnull(row['Important Words']) else ''
    important_words = set(clean_phrase(phrase) for phrase in important_words_str.split(','))  

    human_row = df_human.iloc[index]
    human_words_str = str(human_row['Human extract words']) if pd.notnull(human_row['Human extract words']) else ''
    human_words = set(clean_phrase(phrase) for phrase in human_words_str.split(','))  

    gpt_only = important_words - human_words
    gpt_missed_ratio.append(len(gpt_only) / len(important_words) if len(important_words) > 0 else 0)

    human_only = human_words - important_words
    human_missed_ratio.append(len(human_only) / len(human_words) if len(human_words) > 0 else 0)

    true_positives = len(important_words & human_words)
    total_gpt_words = len(important_words)
    total_human_words = len(human_words)
    
    precision = true_positives / total_gpt_words if total_gpt_words > 0 else 0
    recall = true_positives / total_human_words if total_human_words > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1_score)

    results.append({
        'Index': index,
        'GPT Extracted Words': important_words,
        'Human Extracted Words': human_words,
        'GPT Missed Ratio': len(gpt_only) / len(important_words) if len(important_words) > 0 else 0,
        'Human Missed Ratio': len(human_only) / len(human_words) if len(human_words) > 0 else 0,
        'GPT Only': gpt_only,
        'Human Only': human_only,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    })

average_gpt_missed_ratio = sum(gpt_missed_ratio) / len(gpt_missed_ratio) if gpt_missed_ratio else 0
average_human_missed_ratio = sum(human_missed_ratio) / len(human_missed_ratio) if human_missed_ratio else 0
average_precision = sum(precision_list) / len(precision_list) if precision_list else 0
average_recall = sum(recall_list) / len(recall_list) if recall_list else 0
average_f1_score = sum(f1_list) / len(f1_list) if f1_list else 0

output_folder = 'output_folder'
os.makedirs(output_folder, exist_ok=True)  
comparison_output_file = os.path.join(output_folder, 'comparison_gpt_human_phrases.csv')
comparison_df = pd.DataFrame(results)
comparison_df.to_csv(comparison_output_file, index=False)
print(f"Average proportion of phrases extracted by GPT but not by humans: {average_gpt_missed_ratio:.2%}")
print(f"Average proportion of phrases extracted by humans but not by GPT: {average_human_missed_ratio:.2%}")
print(f"Average Precision: {average_precision:.2%}")
print(f"Average Recall: {average_recall:.2%}")
print(f"Average F1-score: {average_f1_score:.2%}")
print(f"Comparison results have been saved to: {comparison_output_file}")
