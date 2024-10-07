import pandas as pd

def count_articles_per_year(filename):
    data = pd.read_csv(filename)
    
    data['year'] = pd.to_numeric(data['Year'], errors='coerce').fillna(0).astype(int)
    
    yearly_counts = data.groupby('year').size()
    
    min_year = yearly_counts.index.min()
    max_year = yearly_counts.index.max()
    
    return min_year, max_year, yearly_counts

min_year, max_year, article_counts = count_articles_per_year('FAISS_new_output_gpt4_all_new.csv')

df_counts = article_counts.reset_index()
df_counts.columns = ['year', 'ArticleCount']

output_filename = 'yearly_article_counts.csv'
df_counts.to_csv(output_filename, index=False)

