import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('different_try_of_area_calculation/newAreaCalculation_by_length_updated.csv')
df = pd.read_csv('new_novelty_calculation_by_first_occurrence.csv')

df['Quartile'] = df['Quartile'].astype(str)
df['Novelty Score'] = df['Novelty Score'].fillna(0)  

data_to_plot = [df[df['Quartile'] == 'Q1']['Novelty Score'],
                df[df['Quartile'] == 'Q2']['Novelty Score'],
                df[df['Quartile'] == 'Q3']['Novelty Score'],
                df[df['Quartile'] == 'Q4']['Novelty Score']]

plt.figure(figsize=(10, 6))
plt.boxplot(data_to_plot, labels=['Q1', 'Q2', 'Q3', 'Q4'], showmeans=True)

plt.xlabel('quantiles ')
plt.ylabel('Novelty Score')
plt.title('Box Plot of Novelty by quantiles ')
plt.grid(True)
plt.show()

# After the plot, filter for low novelty scores in Q4 and print the titles
low_novelty_q4 = df[(df['Quartile'] == 'Q4') & (df['Novelty Score'] < 0.05)]
low_novelty_titles_q4 = low_novelty_q4['Title']

# Print the titles of papers with low novelty scores in Q4
print("Titles of papers with low novelty scores in Q4:")
for title in low_novelty_titles_q4:
    print(title)