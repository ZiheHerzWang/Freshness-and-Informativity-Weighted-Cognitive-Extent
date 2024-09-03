import pandas as pd

def filter_by_length(input_file, output_file):
    df = pd.read_csv(input_file)
    df_filtered = df[df['Unique Word'].str.len() > 3]
    df_filtered.to_csv(output_file, index=False)

if __name__ == "__main__":
    input = rf"originMethod\newAreaCalculation_no_negative_filter_less10_delete.csv"
    output = rf"originMethod\newAreaCalculation_no_negative_filter_less10_delete_length_check.csv"
    filter_by_length(input, output)
