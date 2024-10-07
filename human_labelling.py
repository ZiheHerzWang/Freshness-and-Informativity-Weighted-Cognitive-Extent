import pandas as pd
import re
import os

# 读取 CSV 文件
file_path = 'random_200_english_rows.csv'  # 确认你的CSV文件路径正确
df = pd.read_csv(file_path)

# 检查文件内容是否正确读取
print(f"CSV file contains {len(df)} rows.")

# 函数移除标题中的标点符号
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text).lower()

# 函数检查输入是否匹配标题中的单词（忽略大小写和标点）
def check_input(input_words, title):
    cleaned_title = clean_text(title)  # 清理标题
    cleaned_input_words = clean_text(input_words)  # 清理输入
    return cleaned_input_words in cleaned_title  # 检查输入是否在标题中出现

# 函数查找标题的索引
def find_title_index(title):
    cleaned_title = clean_text(title)
    for idx, row in df.iterrows():
        if clean_text(row['formatted_title']) == cleaned_title:
            return idx
    return None

# 函数显示当前行的标题
def display_row(row_number):
    global current_row
    global current_input_words
    current_row = row_number
    try:
        title = df['formatted_title'].iloc[row_number]
        year = df['Year'].iloc[row_number]
        print(f"Year: {year}, Title: {title}")
        
        # 重置当前行的输入存储
        current_input_words = []
    except Exception as e:
        print(f"Error displaying row {row_number}: {e}")

# 函数保存所有输入到 CSV 文件（追加模式）
def save_results():
    output_file = 'human_extracts_words.csv'
    output_df = pd.DataFrame(user_inputs)
    
    # 检查文件是否存在
    file_exists = os.path.isfile(output_file)
    
    # 如果文件已经存在，则以追加模式写入，否则写入时包含列名
    output_df.to_csv(output_file, mode='a', header=not file_exists, index=False)
    print(f"Data saved to {output_file}")

# 全局变量存储当前的行号和用户输入的 "Human extracts Words"
current_row = 0
user_inputs = []
current_input_words = []  # 用于存储当前行的输入

# 选择从哪个标题开始
while True:
    search_title = input("Please enter a title to start from the next entry (or press Enter to start from the first title): ").strip()
    if search_title:
        title_index = find_title_index(search_title)
        if title_index is not None and title_index < len(df) - 1:
            current_row = title_index + 1
            print(f"Starting from the next title after: {search_title}")
            break
        else:
            print(f"Title '{search_title}' not found or it is the last title. Please try again.")
    else:
        current_row = 0  # 默认从第一行开始
        break

# 开始程序主循环
display_row(current_row)  # 显示从指定行开始的第一行

while True:
    command = input("Input 'next' to go to next title, 'previous' to go back, 'show' to show current extracts, or 'save' to save and exit. Input human extract words directly to add them: ").lower()
    
    if command == 'next':
        # 保存当前行的数据
        title = df['formatted_title'].iloc[current_row]
        year = df['Year'].iloc[current_row]
        # 将所有当前输入的单词保存
        user_inputs.append({'Year': year, 
                            'formatted_title': title, 
                            'Human extract words': ', '.join(current_input_words)})
        
        # 检查是否已经到最后一行
        if current_row < len(df) - 1:
            display_row(current_row + 1)
        else:
            # 如果已经是最后一行，自动保存并提示结束
            save_results()
            print("You have finished all entries. Data has been saved and the process is complete.")
            break
    
    elif command == 'previous':
        if current_row > 0:
            # 保存当前行的数据
            title = df['formatted_title'].iloc[current_row]
            year = df['Year'].iloc[current_row]
            user_inputs.append({'Year': year, 
                                'formatted_title': title, 
                                'Human extract words': ', '.join(current_input_words)})
            # 进入上一行
            display_row(current_row - 1)
        else:
            print("This is the first row.")
    
    elif command == 'show':
        # 显示当前行的Human extract words
        if current_input_words:
            print(f"Current extracts for this title: {', '.join(current_input_words)}")
        else:
            print("No extracts added for this title yet.")
    
    elif command == 'save':
        # 保存当前行的数据
        title = df['formatted_title'].iloc[current_row]
        year = df['Year'].iloc[current_row]
        user_inputs.append({'Year': year, 
                            'formatted_title': title, 
                            'Human extract words': ', '.join(current_input_words)})
        save_results()
        break
    
    else:
        # 处理用户输入的 Human extract words
        title = df['formatted_title'].iloc[current_row]
        if check_input(command, title):
            current_input_words.append(command)  # 将输入的单词添加到当前行的输入列表中
        else:
            print("Error: Your input contains words not in the title!")
