import pandas as pd
import re

# 读取CSV文件
file_path = 'new_output_gpt4_all.csv'
df = pd.read_csv(file_path)

# 定义一个函数，只选择包含纯英文字符的行
def is_english(text):
    return bool(re.match(r'^[\x00-\x7F]+$', text))

# 过滤出标题为纯英文的行
df_english = df[df['formatted_title'].apply(is_english)]

# 随机选择200行
random_rows = df_english.sample(n=200)

# 将结果保存到新的CSV文件中
output_file = 'random_200_english_rows.csv'
random_rows.to_csv(output_file, index=False)

print(f"已将随机选择的200行纯英文标题数据保存到文件: {output_file}")
