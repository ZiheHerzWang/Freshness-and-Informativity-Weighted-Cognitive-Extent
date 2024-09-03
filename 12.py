import pandas as pd
import numpy as np

df = pd.read_csv('processed_output_with_important_words_new.csv')

df['Citations'] = pd.to_numeric(df['Citations'], errors='coerce')

df['Log_Citations'] = np.log1p(df['Citations'])

threshold_25 = df['Log_Citations'].quantile(0.25)
threshold_50 = df['Log_Citations'].quantile(0.50)
threshold_75 = df['Log_Citations'].quantile(0.75)

print(threshold_25)
print(threshold_50)
print(threshold_75)

def classify_by_log_threshold(value):
    if value <= threshold_25:
        return "Q1"
    elif value <= threshold_50:
        return "Q2"
    elif value <= threshold_75:
        return "Q3"
    else:
        return "Q4"

df['Quartile'] = df['Log_Citations'].apply(classify_by_log_threshold)

df.to_csv('classified_by_quartile.csv', index=False)
