import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 20})
# Define metrics and model values
metrics = ['Precision', 'Recall', 'F1-score']
spacy_values = [7.92, 6.50, 6.73]
scibert_values = [3.54, 8.46, 4.78]
gpt4_values = [59.90, 78.57, 66.04]

# Set positions for each metric
x = np.arange(len(metrics))  
width = 0.2  # Width of each bar

# Create a wider figure to extend the x-axis
fig, ax = plt.subplots(figsize=(10, 9))  # Increased width to 12

# Plot each model's bars
rects1 = ax.bar(x - width, spacy_values, width, label='SpaCy')
rects2 = ax.bar(x, scibert_values, width, label='SciBERT')
rects3 = ax.bar(x + width, gpt4_values, width, label='GPT-4')

# Set labels
ax.set_ylabel('Scores (%)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)

# Move the legend to the upper left corner
ax.legend(loc='upper left', prop={'size': 15})

# Add value labels above the bars
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

# Adjust layout to prevent clipping of labels
fig.tight_layout()

# Save the figure
output_path = 'model_comparison_left_legend.png'
plt.savefig(output_path)

# Optional: Uncomment the following line to display the plot
# plt.show()

print(f"Chart saved at {output_path}")
