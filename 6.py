import pandas as pd

df = pd.read_csv(rf'originMethod\newAreaCalculation_no_negative_filter.csv')
grouped_df = df.groupby('Unique Word').agg({'Count': 'sum'}).reset_index()
filtered_unique_words = grouped_df[grouped_df['Count'] > 10]['Unique Word']
filtered_df = df[df['Unique Word'].isin(filtered_unique_words)]
filtered_df.to_csv(rf'originMethod\newAreaCalculation_no_negative_filter_less10_delete.csv', index=False)
