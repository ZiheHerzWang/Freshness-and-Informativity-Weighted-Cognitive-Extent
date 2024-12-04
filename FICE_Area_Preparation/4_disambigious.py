#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from tqdm import tqdm

# Load the CSV file
file_path = r"FICE_Area_Preparation/csv file/all_gpt_extract.csv"
df = pd.read_csv(file_path)
print(f"Total rows to process: {len(df)}")

# Load tokenizer and model onto GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2").to(device)
model.eval()

# Similarity threshold for clustering and bin size
SIMILARITY_THRESHOLD = 0.5
BIN_SIZE = 250

# Function to get similarity score between a word and representatives in batch
def get_similarity_scores(word, representatives):
    pairs = [(word, rep) for rep in representatives]
    inputs = tokenizer([p[0] for p in pairs], [p[1] for p in pairs],
                       return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.sigmoid(logits).squeeze()
    return scores.tolist() if scores.dim() > 0 else [scores.item()]

# Function to process a single bin of titles
def process_bin(bin_data):
    # Extract and process Scientific Entity
    all_words = []
    for Scientific_Entity in bin_data['Scientific Entity']:
        words = [w.strip() for w in Scientific_Entity.split(',')]
        all_words.extend(words)
    
    # Remove duplicates to get unique entities
    unique_entities = list(set(all_words))
    
    # Clustering logic
    clusters = []
    representative_words = []

    for word in tqdm(unique_entities, desc="processing entities similarity in bin"):
        if not clusters:
            clusters.append({'representative': word, 'words': [word]})
            representative_words.append(word)
        else:
            similarities = get_similarity_scores(word, representative_words)
            max_similarity = max(similarities)
            best_match_idx = similarities.index(max_similarity)

            if max_similarity >= SIMILARITY_THRESHOLD:
                clusters[best_match_idx]['words'].append(word)
                # Update the representative to the shortest word
                if len(word) < len(clusters[best_match_idx]['representative']):
                    clusters[best_match_idx]['representative'] = word
                    representative_words[best_match_idx] = word
            else:
                clusters.append({'representative': word, 'words': [word]})
                representative_words.append(word)
        
    print(f"length of clusters: {len(clusters)}")
    print(f"length of representative_words: {len(representative_words)}")

    # Create a mapping of each word to its representative
    
    word_to_representative = {}
    for cluster in clusters:
        rep_word = cluster['representative']
        for word in cluster['words']:
            word_to_representative[word] = rep_word

    # Update the Important_cluster_word in the bin data
    def replace_with_representative(Scientific_Entity):
        words = [w.strip() for w in Scientific_Entity.split(',')]
        replaced_words = [word_to_representative.get(w, w) for w in words]
        return ", ".join(set(replaced_words))

    bin_data.loc[:, 'Scientific Entity Disambigious'] = bin_data['Scientific Entity'].apply(replace_with_representative)

    return bin_data

# Process the data in bins of 250, merging the last smaller bin with the previous one if necessary
all_processed_data = []
for i in range(0, len(df), BIN_SIZE):
    # Check if the last bin has less than 250, if so merge with previous
    if i + BIN_SIZE > len(df) and len(df) - i < BIN_SIZE:
        current_bin = df[i - BIN_SIZE:i + BIN_SIZE]  # Merge last with previous
    else:
        current_bin = df[i:i + BIN_SIZE]

    print(f"Processing bin {i // BIN_SIZE + 1} with {len(current_bin)} titles")
    processed_bin = process_bin(current_bin)
    all_processed_data.append(processed_bin)

# Combine all processed bins back into a single DataFrame
final_df = pd.concat(all_processed_data, ignore_index=True)

# Save the updated DataFrame to the same CSV file
final_df.to_csv(file_path, index=False)
print(f"Clustering complete! Updated Important_cluster_word saved to {file_path}")
