import pandas as pd

# 读取首次出现年份的文件
word_first_appearance_df = pd.read_csv('word_first_appearance.csv')

word_first_appearance_dict = dict(zip(word_first_appearance_df['Word'].str.lower(), word_first_appearance_df['First Appearance Year']))

df = pd.read_csv('different_try_of_area_calculation/newAreaCalculation_by_length_updated.csv')

df.columns = df.columns.str.strip()

def calculate_paper_novelty(important_words, year):
    words = [word.strip().lower() for word in important_words.split(',')]  
    novelty_score = 0
    
    for word in words:
        first_appearance_year = word_first_appearance_dict.get(word, None)
        if first_appearance_year is not None and first_appearance_year == year:
            novelty_score += 1
            
    return novelty_score

df['Novelty Score'] = df.apply(lambda row: calculate_paper_novelty(row['Important Words'], int(row['Year'])), axis=1)

df.to_csv('new_novelty_calculation_by_first_occurrence.csv', index=False)

print("Novelty scores calculated and saved to 'new_novelty_calculation_by_first_occurrence.csv'")
