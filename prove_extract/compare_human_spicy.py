import os
import pandas as pd
import spacy
import re

# 去除标点符号并转换为小写的函数
def clean_word(word):
    return re.sub(r'[^\w\s]', '', word).strip().lower()

# 加载 SpaCy 模型
nlp = spacy.load('en_core_web_sm')

# 读取 CSV 文件
file_path = 'human_extracts_words.csv'  # 替换为实际文件路径
df = pd.read_csv(file_path)

# 定义输出文件夹并确保其存在
output_folder = 'output_folder_spacy_ner'  # 替换为所需的文件夹路径
os.makedirs(output_folder, exist_ok=True)

# 初始化统计变量
ner_missed = 0
human_missed_ratio = []
spacy_missed_ratio = []
precision_list = []
recall_list = []
f1_list = []

# 创建一个列表用于存储 SpaCy 提取的词汇，以便输出
spacy_extracted_words = []

# 遍历数据框中的每一行
for index, row in df.iterrows():
    title = row['formatted_title']

    # 将人工提取词转换为字符串，并检查是否为 null
    human_words_str = str(row['Human extract words']) if pd.notnull(row['Human extract words']) else ''
    human_words = set(clean_word(phrase) for phrase in human_words_str.split(','))  # 假定词组以逗号分隔

    # 使用 SpaCy 提取实体，按词组来进行比较
    doc = nlp(title)
    extracted_entities = {clean_word(ent.text) for ent in doc.ents}  # 提取实体并清理

    # 保存提取的实体以便之后输出
    spacy_extracted_words.append(", ".join(extracted_entities))

    # 计算 SpaCy 提取但未在人类提取词汇中出现的实体
    spacy_only = extracted_entities - human_words
    spacy_missed_ratio.append(len(spacy_only) / len(extracted_entities) if len(extracted_entities) > 0 else 0)

    # 计算未被 SpaCy 提取的人工词汇比例
    missed_human_words = human_words - extracted_entities
    human_missed_ratio.append(len(missed_human_words) / len(human_words) if len(human_words) > 0 else 0)

    # 计算 Precision, Recall 和 F1-score
    true_positives = len(extracted_entities & human_words)
    total_spacy_words = len(extracted_entities)
    total_human_words = len(human_words)
    
    precision = true_positives / total_spacy_words if total_spacy_words > 0 else 0
    recall = true_positives / total_human_words if total_human_words > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1_score)

    # 累积 NER 提取但不在人类提取中的实体数量
    ner_missed += len(spacy_only)

# 最终统计
total_spacy_missed_ratio = sum(spacy_missed_ratio) / len(spacy_missed_ratio) if spacy_missed_ratio else 0
total_human_missed_ratio = sum(human_missed_ratio) / len(human_missed_ratio) if human_missed_ratio else 0
average_precision = sum(precision_list) / len(precision_list) if precision_list else 0
average_recall = sum(recall_list) / len(recall_list) if recall_list else 0
average_f1_score = sum(f1_list) / len(f1_list) if f1_list else 0

# 将 SpaCy 提取的词汇保存到新的 CSV 文件中
output_file_path = os.path.join(output_folder, 'spacy_extracted_words_cleaned.csv')
df['SpaCy Extracted Words'] = spacy_extracted_words
df.to_csv(output_file_path, index=False)

# 输出结果
print(f"SpaCy 提取的词组未在人类提取词组中出现的数量: {ner_missed}")
print(f"SpaCy 提取的词组未在人类提取词组中出现的平均比例: {total_spacy_missed_ratio:.2%}")
print(f"人类提取的词组未被 SpaCy 提取的平均比例: {total_human_missed_ratio:.2%}")
print(f"平均 Precision: {average_precision:.2%}")
print(f"平均 Recall: {average_recall:.2%}")
print(f"平均 F1-score: {average_f1_score:.2%}")
print(f"SpaCy 提取的词组已保存到: {output_file_path}")
