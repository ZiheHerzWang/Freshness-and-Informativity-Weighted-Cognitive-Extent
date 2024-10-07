import pandas as pd

df = pd.read_csv('word_counts_11111.csv')

chunk_size = -(-len(df) // 20)  

for i in range(20):
    start = i * chunk_size
    end = start + chunk_size
    
    chunk_df = df.iloc[start:end]
    
    chunk_filename = f'word_counts_11111.csv_2_{i}.csv'
    chunk_df.to_csv(chunk_filename, index=False)

