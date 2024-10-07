import pandas as pd

# 读取CSV文件
df = pd.read_csv('new_output_gpt4_all.csv')  # 更新为你的CSV文件路径

# 清理数据列，确保年份是数字，且空值被填充
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
df['Important Words'] = df['Important Words'].fillna("")

# 初始化词汇首次出现记录字典
word_first_appearance = {}

# 遍历每一行数据并统计首次出现年份
for _, row in df.iterrows():
    year = int(row['Year'])
    important_words = row['Important Words'].lower().split(', ')  # 处理重要词汇列

    for word in important_words:
        if word not in word_first_appearance:
            # 如果词汇首次出现，记录其年份
            word_first_appearance[word] = year
        elif word_first_appearance[word] > year:
            # 如果发现该词汇在更早的年份出现，更新该年份
            word_first_appearance[word] = year

# 打印每个词汇的首次出现年份
for word, first_year in word_first_appearance.items():
    print(f"Word: {word}, First Appearance Year: {first_year}")

# 可选：将结果保存为CSV文件
output_file = 'word_first_appearance.csv'
df_output = pd.DataFrame(list(word_first_appearance.items()), columns=['Word', 'First Appearance Year'])
df_output.to_csv(output_file, index=False)

print(f"Word first appearance years saved to {output_file}")
