import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('newAreaCalculation_by_length.csv')

df['Quartile'] = df['Quartile'].astype(str)
df['Novelty Score'] = df['Novelty Score'].fillna(0)  

data_to_plot = [df[df['Quartile'] == 'Q1']['Novelty Score'],
                df[df['Quartile'] == 'Q2']['Novelty Score'],
                df[df['Quartile'] == 'Q3']['Novelty Score'],
                df[df['Quartile'] == 'Q4']['Novelty Score']]

plt.figure(figsize=(10, 6))
plt.boxplot(data_to_plot, labels=['Q1', 'Q2', 'Q3', 'Q4'], showmeans=True)

plt.xlabel('Quartile')
plt.ylabel('Novelty Score')
plt.title('Box Plot of Novelty by Quartile')
plt.grid(True)
plt.show()
