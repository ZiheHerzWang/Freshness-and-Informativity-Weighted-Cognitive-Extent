import matplotlib.pyplot as plt
import numpy as np

# 定义三个模型的 Precision, Recall 和 F1-score
metrics = ['Precision', 'Recall', 'F1-score']
spacy_values = [7.92, 6.50, 6.73]
scibert_values = [3.54, 8.46, 4.78]
gpt4_values = [59.90, 78.57, 66.04]

# 设置图形
x = np.arange(len(metrics))  # 每个指标的位置
width = 0.2  # 条形的宽度

# 创建图表
fig, ax = plt.subplots()

# 绘制每个模型的条形
rects1 = ax.bar(x - width, spacy_values, width, label='SpaCy')
rects2 = ax.bar(x, scibert_values, width, label='SciBERT')
rects3 = ax.bar(x + width, gpt4_values, width, label='GPT-4')

# 添加一些文本标签
ax.set_ylabel('Scores (%)')
ax.set_title('Model Comparison: Precision, Recall, F1-score')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# 在条形上添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 调整布局
fig.tight_layout()

# 保存图表为 PNG 文件
output_path = 'model_comparison.png'
plt.savefig(output_path)

# 如果不需要显示图像，请注释掉 plt.show()
# plt.show()

print(f"图表已保存到 {output_path}")
