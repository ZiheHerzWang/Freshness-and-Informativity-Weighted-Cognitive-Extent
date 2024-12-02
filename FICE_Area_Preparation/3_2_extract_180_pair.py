#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import random
import torch

# Load the CSV file with words
file_path = rf'C:\Users\zihe0\Desktop\try\Novelty-Life-Time\FICE_Area_Preparation\csv file\Unique_word.csv'
df = pd.read_csv(file_path)

# Extract the 'Word' column
words = df['Word'].tolist()

# Load tokenizer and model for similarity scoring
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
model.eval()

# Define similarity bins and parameters
similarity_bins = {interval: [] for interval in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
target_bins = [0.0, 0.1, 0.2, 0.4]
pairs_needed_for_target_bins = 10
max_total_pairs = 180
batch_size = 32  # Number of pairs to process in each batch

# Function to calculate similarity scores in batches
def get_batch_similarity_scores(pairs):
    inputs = tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs], 
                       return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    # Apply sigmoid to obtain similarity scores
    return torch.sigmoid(logits).squeeze().tolist()  # List of scores for each pair

# Function to find the correct bin for the similarity score
def find_bin(similarity):
    for interval in similarity_bins.keys():
        if interval <= similarity < interval + 0.1:
            return interval
    return None

# Continue sampling pairs in batches until bins and total pairs requirements are met
while sum(len(v) for v in similarity_bins.values()) < max_total_pairs and \
      any(len(similarity_bins[interval]) < (pairs_needed_for_target_bins if interval in target_bins else max_total_pairs)
          for interval in similarity_bins):

    # Sample a batch of random pairs
    pairs = [(random.choice(words), random.choice(words)) for _ in range(batch_size)]
    pairs = [(w1, w2) for w1, w2 in pairs if w1 != w2]  # Ensure pairs are distinct words
    
    # Calculate similarity scores for the batch
    similarities = get_batch_similarity_scores(pairs)
    
    # Assign pairs to bins based on similarity scores
    for (word1, word2), similarity in zip(pairs, similarities):
        if similarity >= 0.0:
            bin_interval = find_bin(similarity)
            if bin_interval is not None:
                if (bin_interval in target_bins and len(similarity_bins[bin_interval]) < pairs_needed_for_target_bins) or \
                   (bin_interval not in target_bins):
                    similarity_bins[bin_interval].append({'word1': word1, 'word2': word2, 'similarity': similarity})
                    print(f"Added pair: word1: {word1}, word2: {word2}, similarity: {similarity}, bin: {bin_interval}")

# Combine all bins into a single list of pairs
random_pairs = [pair for bin_pairs in similarity_bins.values() for pair in bin_pairs]

# Save results to a CSV file
similarity_df = pd.DataFrame(random_pairs)
output_file = r'FICE_Area_Preparation/csv file/word_similarity_scores_with_bins.csv'
similarity_df.to_csv(output_file, index=False)

print(f"Similarity scores saved to {output_file}")
