import pandas as pd
import glob

path_folder = "C:/Users/zihe0/Desktop/try/Novelty-Life-Time/new_pipeline/parallel/area_csv_file"
path_file = f"{path_folder}/word_counts_fits_process_*.csv"
merged_area_path = f"{path_folder}/merged_area_file.csv"
all_files = glob.glob(path_file)
list_data = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    list_data.append(df)

merged_data = pd.concat(list_data, axis=0, ignore_index=True)
merged_data = merged_data[merged_data['Area Ratio'] >= 0]
merged_data['Area Ratio'] = merged_data['Area Ratio'].apply(lambda x: 1 if x > 1 else x)
merged_data.to_csv(merged_area_path, index=False)

