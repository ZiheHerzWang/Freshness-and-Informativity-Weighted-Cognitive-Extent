import pandas as pd
#Unique Word,Year,Ground Truth,Fitted Value,Area Ratio
df = pd.read_csv(rf'newAreaCalculation_no_negative_filter_2.csv')
grouped_df = df.groupby('Unique Word').agg({'Ground Truth': 'sum'}).reset_index()
filtered_unique_words = grouped_df[grouped_df['Ground Truth'] > 10]['Unique Word']
filtered_df = df[df['Unique Word'].isin(filtered_unique_words)]
filtered_df.to_csv(rf'newAreaCalculation_no_negative_filter_less10_delete.csv', index=False)
