import pandas as pd

df = pd.read_csv('newAreaCalculation_no_negative_filter_2.csv')
grouped_df = df.groupby('Unique Word').agg({'Count': 'sum'}).reset_index()
filtered_unique_words = grouped_df[grouped_df['Count'] > 5]['Unique Word']
filtered_df = df[df['Unique Word'].isin(filtered_unique_words)]
filtered_df.to_csv('filtered_newAreaCalculation.csv', index=False)
